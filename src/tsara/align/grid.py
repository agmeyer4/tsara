"""The uniform grid: one support for a whole campaign.

What it is for, and what it is not for
---------------------------------------
``docs/METHODS.md`` §1.4: a single uniform ``(time × variable)`` cube is built
**only** for the products that inherently need one — the continuous rolling
state, and the matrix a receptor model such as PMF consumes. Baselines,
detection and cross-species regression all run at native rate and never see
it (§1.1, "synchronize late").

It is not an alternative to pairing (§11.4) and the difference is resolution.
A grid's period is set by the **worst** instrument in it; a pair's clock by
the worse of *that pair*. With 60 s stationary means in a campaign, every
1 Hz-against-1 Hz ratio computed off a single grid would come from sixty times
fewer points, and every sub-minute plume would be flattened before the fit saw
it. The grid is what makes a whole campaign comparable; the pair clock is what
keeps a fast pair from being punished for a slow instrument elsewhere.

There is deliberately no "PMF matrix" object
---------------------------------------------
A receptor-model matrix is this function called with a chosen set of columns.
Which columns is a scientific decision — raw concentrations or the
enhancements §5 will compute, which species, whether met belongs in the same
cube — and hard-coding any of it would make the block need editing every time
that decision changed. Hence the ``variables`` argument, and hence a grid
period validated against *the variables being gridded* rather than against
every stream the campaign happens to contain.

The period rule
---------------
**The grid period must be at least the widest cell among the selected
variables.** A 60 s mean evaluated on 1 s cells is the same value repeated
sixty times: resolution the instrument never had, and sixty points where there
is one measurement. That is the prohibition the whole interval model exists to
enforce (§10), so it is an error naming the offending variable and the
smallest period that would work, not a warning.

Which variables go in is therefore a real lever: a run that excludes the
canisters can be gridded far finer than one that includes them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.align.binning import (
    TsaraAlignError,
    bin_streams_onto_cells,
    median_width_s,
    select_variables,
    stream_cells,
)
from tsara.core.support import CellBounds
from tsara.core.timebase import to_utc_naive_stamp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import xarray as xr

    from tsara.config.analysis import OutputGridConfig
    from tsara.core.propagation import PropagationForm

logger = logging.getLogger(__name__)

__all__ = ["build_output_grid", "grid_cells"]

#: Attrs the gridded product carries, documented in ``docs/METHODS.md`` §11.7.
GRID_FREQ_ATTR = "tsara_grid_freq"
GRID_WIDEST_CELL_ATTR = "tsara_grid_widest_source_cell_s"
GRID_VARIABLES_ATTR = "tsara_grid_variables"


def grid_cells(
    streams: Mapping[str, xr.Dataset],
    config: OutputGridConfig,
    variables: Sequence[str | tuple[str, str]] | None = None,
) -> CellBounds:
    """Build the uniform cells a campaign's output grid sits on.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    config : OutputGridConfig
        Grid period and optional explicit start and end.
    variables : sequence of str or (str, str), optional
        Which variables the grid is for. ``None`` means all of them, and the
        period rule is checked against all of them accordingly.

    Returns
    -------
    CellBounds
        Abutting cells of exactly ``config.freq``.

    Raises
    ------
    TsaraAlignError
        If the period is shorter than the widest selected cell, if the
        requested window is empty, or if no variable was selected.

    Notes
    -----
    With no explicit ``start``, the grid begins at the largest whole multiple
    of the period at or before the earliest selected cell. Anchoring to the
    epoch rather than to whenever the data happened to start means two runs
    over overlapping periods produce cells that line up, which is what lets
    their outputs be compared at all.
    """
    selection = select_variables(streams, variables)
    if not selection:
        raise TsaraAlignError("No variables selected, so there is nothing to build a grid for.")
    period_ns = int(pd.Timedelta(config.freq).value)
    instruments = sorted({instrument for instrument, _ in selection})
    cells = {
        instrument: stream_cells(streams[instrument], instrument) for instrument in instruments
    }

    widest_instrument = max(instruments, key=lambda name: median_width_s(cells[name]))
    widest_s = median_width_s(cells[widest_instrument])
    if period_ns < widest_s * 1e9:
        raise TsaraAlignError(
            f"Grid period {config.freq} is shorter than the widest selected cell: "
            f"'{widest_instrument}' has {widest_s:.6g} s cells. Evaluating those on "
            f"a finer grid would repeat one measurement across several cells, which "
            f"is resolution the instrument never had (METHODS §11.7). Use a period of "
            f"at least {widest_s:.6g} s, or leave '{widest_instrument}' out of the "
            "selection."
        )

    earliest = min(int(bounds.start_ns.min()) for bounds in cells.values())
    latest = max(int(bounds.stop_ns.max()) for bounds in cells.values())
    if config.start is not None:
        start_ns = int(to_utc_naive_stamp(pd.Timestamp(config.start)).value)
    else:
        # Floor division on a negative epoch still rounds toward minus
        # infinity in Python, which is what "the multiple at or before" means
        # on both sides of 1970.
        start_ns = (earliest // period_ns) * period_ns
    stop_ns = (
        int(to_utc_naive_stamp(pd.Timestamp(config.end)).value)
        if config.end is not None
        else latest
    )
    if stop_ns <= start_ns:
        raise TsaraAlignError(
            f"The requested grid window is empty: it starts at "
            f"{pd.Timestamp(start_ns)} and ends at {pd.Timestamp(stop_ns)}."
        )
    n_cells = int(np.ceil((stop_ns - start_ns) / period_ns))
    edges = start_ns + np.arange(n_cells + 1, dtype=np.int64) * period_ns
    return CellBounds(start_ns=edges[:-1], stop_ns=edges[1:])


def build_output_grid(
    streams: Mapping[str, xr.Dataset],
    config: OutputGridConfig,
    variables: Sequence[str | tuple[str, str]] | None = None,
    *,
    propagation_form: PropagationForm = "ar1_neff",
) -> xr.Dataset:
    """Put a campaign's variables on one uniform support.

    A thin layer over :func:`~tsara.align.binning.bin_streams_onto_cells`: the
    only thing this adds is *which cells*, and the rule that the period has to
    respect the data going into it.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    config : OutputGridConfig
        Grid period and optional window.
    variables : sequence of str or (str, str), optional
        Which variables the grid holds. ``None`` takes every non-sigma
        variable in every stream.
    propagation_form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form reduces a correlated random component (§3.4).

    Returns
    -------
    xarray.Dataset
        The grid, with each variable's uncertainty, contributing count and
        coverage alongside it, and CF cell boundaries. Cells with no data are
        ``nan`` with a count of zero — never interpolated (§1.2).

    Raises
    ------
    TsaraAlignError
        If the period is shorter than the widest selected cell, or the
        selection or window is empty.
    """
    selection = select_variables(streams, variables)
    target = grid_cells(streams, config, variables)
    gridded = bin_streams_onto_cells(streams, target, selection, propagation_form=propagation_form)
    widest = max(
        median_width_s(stream_cells(streams[instrument], instrument)) for instrument, _ in selection
    )
    gridded.attrs["tsara_stage"] = "gridded"
    gridded.attrs[GRID_FREQ_ATTR] = str(config.freq)
    gridded.attrs[GRID_WIDEST_CELL_ATTR] = float(widest)
    # Recorded because the grid period was validated against exactly this
    # selection, and a reader cannot tell from the columns alone whether a
    # variable is absent because it was excluded or because it had no data.
    gridded.attrs[GRID_VARIABLES_ATTR] = ", ".join(
        f"{instrument}.{name}" for instrument, name in selection
    )
    logger.info(
        "Built a %s grid of %d cells over %d variable(s); widest source cell %.6g s.",
        config.freq,
        len(target),
        len(selection),
        widest,
    )
    return gridded
