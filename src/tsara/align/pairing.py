"""Pairing two species measured on different clocks.

A thin layer, deliberately
---------------------------
Pairing is not its own operation. It is
:func:`~tsara.align.binning.bin_streams_onto_cells` with three decisions
layered on top:

1. **which cells** -- those of the wider-supported of the two streams;
2. **which stretch** -- optionally restricted to an event or window;
3. **which rows survive** -- a pair needs a real measurement of *both*
   species, so a cell missing either is dropped.

Everything else, including the averaging, the uncertainty propagation and the
counts and coverage that qualify each value, belongs to the binner and is
shared with the campaign-wide grid. Writing pairing as its own implementation
was a design inversion: the two-species case is the general case with two
variables selected, and building the special one first would have left two
implementations of one idea to drift apart.

Why *wider*, not *slower*
--------------------------
Before Phase 3.5 the rule in ``docs/METHODS.md`` §1.3 said "the slower
instrument", and rate and support can disagree. Measured on the 2024 drives,
the iWAS canisters sample every 530 s but each sample integrates for only
14.9 s -- medians over all 261 fills of the ten 2024 drive days. Against a 60 s
stationary mean the canister is thirty-five times slower by rate and four times
*narrower* by support. Pairing on the canister's clock
would evaluate a 60 s mean over 15 s, which is exactly what the interval model
forbids; pairing on the mean's clock is admissible, and the coverage of 0.25
is what says how much to trust it.

So the direction is always the same: a value may be averaged onto a wider
support, never split onto a narrower one.

Why a pair-specific clock exists at all
----------------------------------------
A campaign-wide grid's period is set by the *worst* instrument in it; a pair's
clock by the worse of *that pair*. With 60 s stationary means in the archive,
every 1 Hz-against-1 Hz ratio computed off a single grid would come from sixty
times fewer points -- roughly eight times the confidence interval -- with every
sub-minute plume flattened before the fit saw it. The grid is what makes a
whole campaign comparable; the pair clock is what keeps a fast pair from being
punished for a slow instrument elsewhere. Both are wanted, and they are the
same code.

What N means afterwards
------------------------
Every returned pair contains at least one real measurement of each species, so
the regression sample size is the number of real pairs. That is the whole
reason gases are binned rather than interpolated: interpolated points pose as
independent samples and silently inflate the degrees of freedom of every fit
downstream (§1.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.align.binning import (
    TsaraAlignError,
    bin_streams_onto_cells,
    median_width_s,
    resolve_variable,
    stream_cells,
)
from tsara.core.naming import TIME_COORD, coverage_name
from tsara.core.support import CellBounds

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import xarray as xr

    from tsara.core.propagation import PropagationForm

logger = logging.getLogger(__name__)

__all__ = ["PairedSpecies", "pair_species"]

#: Attrs the paired product adds on top of the binner's, documented in §11.4.
PAIRING_CLOCK_ATTR = "tsara_pairing_clock"
PAIRING_REASON_ATTR = "tsara_pairing_clock_reason"
PAIRING_COVERAGE_ATTR = "tsara_pairing_min_coverage"
PAIRING_DROPPED_ATTR = "tsara_pairing_cells_dropped"
PAIRING_CANDIDATE_ATTR = "tsara_pairing_cells_considered"


@dataclass(frozen=True)
class PairedSpecies:
    """Two species on one clock, with everything needed to fit them.

    Attributes
    ----------
    dataset : xarray.Dataset
        The pairs. One entry per surviving cell, carrying both species'
        values, whatever uncertainty components were available, and per
        species the contributing sample count and cell coverage. Cells are
        described by CF ``time_bnds`` exactly as a stream's are, so the
        product says what interval each pair describes.
    y_name, x_name : str
        Names of the two species as they appear in the dataset.
    clock : str
        Instrument whose cells were used, i.e. the wider-supported one.
    n_pairs : int
        How many pairs survived. The N of any fit that follows.
    """

    dataset: xr.Dataset
    y_name: str
    x_name: str
    clock: str
    n_pairs: int

    def __len__(self) -> int:
        """Return the number of surviving pairs."""
        return self.n_pairs


def _restrict(cells: CellBounds, interval: tuple[pd.Timestamp, pd.Timestamp]) -> CellBounds:
    """Return the cells overlapping ``interval``.

    Overlap rather than containment: an event boundary rarely falls on a cell
    edge, and dropping the two partly-covered end cells would quietly shorten
    every event by up to one cell on each side.
    """
    start, stop = (int(pd.Timestamp(edge).value) for edge in interval)
    if stop <= start:
        raise TsaraAlignError(
            f"Pairing interval must have positive duration; got {interval[0]} to {interval[1]}."
        )
    keep = np.flatnonzero((cells.stop_ns > start) & (cells.start_ns < stop))
    return CellBounds(start_ns=cells.start_ns[keep], stop_ns=cells.stop_ns[keep])


def pair_species(
    streams: Mapping[str, xr.Dataset],
    y: str | tuple[str, str],
    x: str | tuple[str, str],
    *,
    interval: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    min_coverage: float = 0.0,
    propagation_form: PropagationForm = "ar1_neff",
) -> PairedSpecies:
    """Put two species on one clock, dropping any pair that is not real.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        A campaign's native-rate streams.
    y, x : str or tuple of (str, str)
        The two species. A bare canonical name is looked up across the
        streams and must be unambiguous; a tuple names the instrument.
    interval : tuple of pandas.Timestamp, optional
        Restrict pairing to cells overlapping this window -- an event, in
        Phase 6. ``None`` pairs the whole record.
    min_coverage : float, optional
        Drop pairs whose cell was covered by less than this fraction of
        contributing data. Default 0.0 drops nothing; coverage is recorded
        either way (:class:`~tsara.config.analysis.PairingConfig`).
    propagation_form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form reduces a correlated random component (§3.4).

    Returns
    -------
    PairedSpecies
        The pairs, and which stream's clock they are on.

    Raises
    ------
    TsaraAlignError
        If a species cannot be resolved, a stream has no cells, the interval
        selects nothing, or no cell holds a real measurement of both species.
    """
    if not streams:
        raise TsaraAlignError("No streams to pair.")
    y_instrument, y_variable = resolve_variable(streams, y)
    x_instrument, x_variable = resolve_variable(streams, x)
    if (y_instrument, y_variable) == (x_instrument, x_variable):
        raise TsaraAlignError(
            f"Cannot pair '{y_variable}' on '{y_instrument}' with itself; a ratio of "
            "a species to itself is 1 by construction."
        )

    y_cells = stream_cells(streams[y_instrument], y_instrument)
    x_cells = stream_cells(streams[x_instrument], x_instrument)
    y_width, x_width = median_width_s(y_cells), median_width_s(x_cells)
    if y_instrument == x_instrument:
        clock, target = y_instrument, y_cells
        reason = "both species share one instrument"
    elif y_width >= x_width:
        clock, target = y_instrument, y_cells
        reason = f"wider cells ({y_width:.6g} s vs {x_width:.6g} s)"
    else:
        clock, target = x_instrument, x_cells
        reason = f"wider cells ({x_width:.6g} s vs {y_width:.6g} s)"

    if interval is not None:
        target = _restrict(target, interval)
        if len(target) == 0:
            raise TsaraAlignError(
                f"No cells of '{clock}' overlap the interval {interval[0]} to {interval[1]}."
            )
    n_considered = len(target)

    joined = bin_streams_onto_cells(
        streams,
        target,
        [(y_instrument, y_variable), (x_instrument, x_variable)],
        propagation_form=propagation_form,
    )
    # The binner names a column after its variable, suffixing with the
    # instrument only when two selected streams claim the same name -- which
    # for a pair means one species compared across two analyzers.
    collides = y_variable == x_variable
    y_name = f"{y_variable}_{y_instrument}" if collides else y_variable
    x_name = f"{x_variable}_{x_instrument}" if collides else x_variable

    surviving = np.flatnonzero(
        np.isfinite(joined[y_name].values)
        & np.isfinite(joined[x_name].values)
        & (joined[coverage_name(y_name)].values >= min_coverage)
        & (joined[coverage_name(x_name)].values >= min_coverage)
    )
    if surviving.size == 0:
        raise TsaraAlignError(
            f"'{y_name}' and '{x_name}' produced no usable pairs over {n_considered} "
            f"candidate cell(s) of '{clock}'. Either the records do not overlap, or "
            f"every candidate cell was masked or below min_coverage={min_coverage}."
        )
    dataset = joined.isel({TIME_COORD: surviving})
    dataset.attrs.update(
        {
            "tsara_stage": "paired",
            PAIRING_CLOCK_ATTR: clock,
            PAIRING_REASON_ATTR: reason,
            PAIRING_COVERAGE_ATTR: float(min_coverage),
            PAIRING_CANDIDATE_ATTR: int(n_considered),
            PAIRING_DROPPED_ATTR: int(n_considered - surviving.size),
        }
    )
    logger.info(
        "Paired %s vs %s on '%s' (%s): %d of %d candidate cells kept.",
        y_name,
        x_name,
        clock,
        reason,
        surviving.size,
        n_considered,
    )
    return PairedSpecies(
        dataset=dataset,
        y_name=y_name,
        x_name=x_name,
        clock=clock,
        n_pairs=int(surviving.size),
    )
