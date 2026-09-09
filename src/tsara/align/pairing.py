"""Pairing two species measured on different clocks.

The operation
-------------
To regress one species against another they must be compared over the same
air. When the two come from instruments running at different rates, that
means choosing a clock and moving one stream onto it. TSARA's rule
(``docs/METHODS.md`` §1.3):

1. The **pairing clock** is the cells of whichever stream has the **wider
   support**. The other is averaged onto them, weighted by overlap.
2. A cell with no contributing data from either species is dropped. A pair
   is never fabricated.
3. Every surviving pair records how much of its cell was actually covered.

Why *wider*, not *slower*
--------------------------
Before Phase 3.5 this rule said "the slower instrument", and rate and support
can disagree. Measured on the 2024 drives, the iWAS canisters sample every
441 s but each sample integrates for only 14.7 s. Against a 60 s stationary
mean, the canister is thirty times slower by rate and four times *narrower*
by support. Pairing on the canister's clock would evaluate a 60 s mean over
15 s, which is exactly what the interval model forbids; pairing on the mean's
clock is admissible, and the coverage of 0.25 is what says how much to trust
it.

So the direction is always the same: a value may be averaged onto a wider
support, never split onto a narrower one.

What N means afterwards
------------------------
Every returned pair contains at least one real measurement of each species,
so the regression sample size is the number of real pairs. That is the whole
reason gases are binned rather than interpolated: interpolated points pose as
independent samples and silently inflate the degrees of freedom of every fit
downstream (§1.2).

Uncertainty travels with the values
------------------------------------
Both members arrive carrying whatever uncertainty budget ingestion could
resolve, and pairing does two things with it, in order:

1. If a declared figure was quoted at a different interval from the cells it
   sits on (``uncertainty_at_width``, recorded but never acted on at
   ingestion — §10.8), it is moved onto those cells here, at the point of
   use. Without a decorrelation timescale that move is refused and recorded
   as ``unscaled``, because the naive root-N is not merely imprecise but
   confidently wrong.
2. The binned member's uncertainty is propagated through the same overlap
   weights that formed its value (§3), random and systematic separately.

Every one of those decisions is recorded in the output's attributes. A sigma
that was reduced by averaging and a sigma that was not look identical once
written to a file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from tsara import __version__
from tsara.core.bundle import pin_time_encoding
from tsara.core.exceptions import TsaraError
from tsara.core.naming import (
    CELL_METHODS_ATTR,
    TIME_BOUNDS_VAR,
    TIME_COORD,
    sigma_rand_name,
    sigma_sys_name,
)
from tsara.core.propagation import (
    PropagationForm,
    propagate_random_binned,
    propagate_systematic_binned,
    sigma_at_support,
)
from tsara.core.support import CellBounds, attach_time_bounds, overlap_pairs

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = ["PairedSpecies", "TsaraAlignError", "pair_species"]

#: Attrs the paired product carries, documented in ``docs/METHODS.md`` §11.3.
PAIRING_CLOCK_ATTR = "tsara_pairing_clock"
PAIRING_REASON_ATTR = "tsara_pairing_clock_reason"
PAIRING_COVERAGE_ATTR = "tsara_pairing_min_coverage"
PAIRING_DROPPED_ATTR = "tsara_pairing_cells_dropped"
PAIRING_CANDIDATE_ATTR = "tsara_pairing_cells_considered"
PAIRING_INSTRUMENT_ATTR = "tsara_source_instrument"
PAIRING_BINNED_ATTR = "tsara_pairing_binned"
PROPAGATION_FORM_ATTR = "tsara_propagation_form"
SIGMA_AT_SUPPORT_ATTR = "tsara_sigma_at_support"

#: Seconds per nanosecond, for turning cell widths into the units the
#: propagation module speaks.
_NS_PER_S = 1_000_000_000.0


class TsaraAlignError(TsaraError):
    """Raised when two streams cannot be paired as asked.

    Its own type because the failures are about *combining* streams rather
    than about reading them: a species that no stream declares, a name two
    streams both declare, a stream with no cells, or a pair whose records
    never overlap in time.
    """


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
        Canonical names of the two species, in the dataset.
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


def _resolve(
    streams: Mapping[str, xr.Dataset], reference: str | tuple[str, str]
) -> tuple[str, str]:
    """Return ``(instrument, variable)`` for a species reference.

    A bare name is searched for across the streams. Two instruments measuring
    the same canonical name is a real situation -- it is how a campaign
    compares two analyzers -- so the ambiguity is an error naming both rather
    than a silent choice of the first.
    """
    if isinstance(reference, tuple):
        instrument, variable = reference
        if instrument not in streams:
            raise TsaraAlignError(
                f"No stream named '{instrument}'; available: {sorted(map(str, streams))}."
            )
        if variable not in streams[instrument].data_vars:
            raise TsaraAlignError(
                f"Stream '{instrument}' has no variable '{variable}'; available: "
                f"{sorted(map(str, streams[instrument].data_vars))}."
            )
        return instrument, variable
    holders = [name for name, stream in streams.items() if reference in stream.data_vars]
    if not holders:
        raise TsaraAlignError(
            f"No stream measures '{reference}'. Streams and their variables: "
            + "; ".join(f"{n}: {sorted(map(str, s.data_vars))}" for n, s in streams.items())
        )
    if len(holders) > 1:
        raise TsaraAlignError(
            f"'{reference}' is measured by more than one instrument ({sorted(holders)}), "
            "so naming it alone is ambiguous. Pass a (instrument, variable) pair to "
            "say which one."
        )
    return holders[0], reference


def _cells(stream: xr.Dataset, instrument: str) -> CellBounds:
    """Return a stream's cells, or raise naming the instrument."""
    if TIME_BOUNDS_VAR not in stream.coords and TIME_BOUNDS_VAR not in stream.data_vars:
        raise TsaraAlignError(
            f"Stream '{instrument}' carries no '{TIME_BOUNDS_VAR}', so there is no "
            "interval to pair over. Streams gain cells at ingestion (METHODS §10); "
            "a bundle written before format 2 must be reloaded to acquire them."
        )
    bounds = np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]").astype(np.int64)
    if bounds.size == 0:
        raise TsaraAlignError(f"Stream '{instrument}' has no cells to pair over.")
    return CellBounds(start_ns=bounds[:, 0].copy(), stop_ns=bounds[:, 1].copy())


def _median_width_s(cells: CellBounds) -> float:
    """Return the median cell width in seconds."""
    return float(np.median(cells.width_ns)) / _NS_PER_S


def _sigma_on_cells(
    stream: xr.Dataset,
    variable: str,
    sigma_variable: str,
    cell_width_s: float,
    form: PropagationForm,
) -> tuple[np.ndarray, str] | None:
    """Return a sigma restated on the stream's own cells, and how.

    Ingestion stores a declared figure exactly as declared and records the
    interval it was quoted at, deliberately performing no arithmetic on it
    (§10.8). This is the point of use, so this is where the arithmetic
    happens -- or is refused and said so.
    """
    if sigma_variable not in stream.data_vars:
        return None
    values = np.asarray(stream[sigma_variable].values, dtype=np.float64)
    quoted = stream[variable].attrs.get("uncertainty_at_width")
    if quoted is None:
        return values, "unchanged"
    tau = stream[variable].attrs.get("decorrelation_timescale")
    tau_s = float(pd.Timedelta(tau).total_seconds()) if tau is not None else None
    quoted_s = float(pd.Timedelta(quoted).total_seconds())
    moved, provenance = sigma_at_support(
        values,
        quoted_width_s=quoted_s,
        target_width_s=cell_width_s,
        tau_s=tau_s,
        form=form,
    )
    if provenance == "unscaled":
        logger.warning(
            "%s declares its uncertainty at %s but its cells are %.3f s wide, and no "
            "decorrelation timescale was declared, so the figure is used unscaled. "
            "Declare decorrelation_timescale to move it onto the cells.",
            variable,
            quoted,
            cell_width_s,
        )
    return np.asarray(moved, dtype=np.float64), provenance


def _restrict(cells: CellBounds, interval: tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    """Return the indices of cells overlapping ``interval``.

    Overlap rather than containment: an event boundary rarely falls on a cell
    edge, and dropping the two partly-covered end cells would quietly shorten
    every event by up to one cell on each side.
    """
    start, stop = (int(pd.Timestamp(edge).value) for edge in interval)
    if stop <= start:
        raise TsaraAlignError(
            f"Pairing interval must have positive duration; got {interval[0]} to {interval[1]}."
        )
    return np.flatnonzero((cells.stop_ns > start) & (cells.start_ns < stop))


def _subset(cells: CellBounds, index: np.ndarray) -> CellBounds:
    """Return the cells at ``index``."""
    return CellBounds(start_ns=cells.start_ns[index], stop_ns=cells.stop_ns[index])


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
        A campaign's native-rate streams, e.g. a
        :class:`~tsara.ingest.campaign.StreamCollection` or the ``streams``
        of a :class:`~tsara.synthetic.generator.SyntheticDataset`.
    y, x : str or tuple of (str, str)
        The two species. A bare canonical name is looked up across the
        streams and must be unambiguous; a tuple names the instrument
        explicitly.
    interval : tuple of pandas.Timestamp, optional
        Restrict pairing to cells overlapping this window — an event, in
        Phase 6. ``None`` pairs the whole record.
    min_coverage : float, optional
        Drop pairs whose cell was covered by less than this fraction of
        contributing partner data. Default 0.0 drops nothing; coverage is
        recorded either way (:class:`~tsara.config.analysis.PairingConfig`).
    propagation_form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form reduces a correlated random component (§3.4).

    Returns
    -------
    PairedSpecies
        The pairs, and which stream's clock they are on.

    Raises
    ------
    TsaraAlignError
        If a species cannot be resolved, a stream has no cells, or the two
        records never overlap in time.

    Notes
    -----
    When both species come from the same instrument they already share a
    clock, and no binning happens at all — the values pass through
    untouched. That path is not merely an optimization: overlap-weighted
    averaging of a cell onto itself is the identity mathematically but not
    in floating point, and same-instrument species pairs (several gases
    retrieved from one spectrum) are the most common case there is.
    """
    if not streams:
        raise TsaraAlignError("No streams to pair.")
    y_instrument, y_name = _resolve(streams, y)
    x_instrument, x_name = _resolve(streams, x)
    if y_name == x_name and y_instrument == x_instrument:
        raise TsaraAlignError(
            f"Cannot pair '{y_name}' on '{y_instrument}' with itself; a ratio of a "
            "species to itself is 1 by construction."
        )

    y_cells = _cells(streams[y_instrument], y_instrument)
    x_cells = _cells(streams[x_instrument], x_instrument)
    same_clock = y_instrument == x_instrument

    if same_clock:
        clock, reason = y_instrument, "both species share one instrument"
        target_cells = y_cells
    elif _median_width_s(y_cells) >= _median_width_s(x_cells):
        clock, reason = (
            y_instrument,
            f"wider cells ({_median_width_s(y_cells):.6g} s vs {_median_width_s(x_cells):.6g} s)",
        )
        target_cells = y_cells
    else:
        clock, reason = (
            x_instrument,
            f"wider cells ({_median_width_s(x_cells):.6g} s vs {_median_width_s(y_cells):.6g} s)",
        )
        target_cells = x_cells

    keep = np.arange(len(target_cells), dtype=np.int64)
    if interval is not None:
        keep = _restrict(target_cells, interval)
        if keep.size == 0:
            raise TsaraAlignError(
                f"No cells of '{clock}' overlap the interval {interval[0]} to {interval[1]}."
            )
        target_cells = _subset(target_cells, keep)
    n_considered = len(target_cells)

    columns: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    for instrument, name in ((y_instrument, y_name), (x_instrument, x_name)):
        native = instrument == clock
        columns.update(
            _one_species(
                streams[instrument],
                instrument=instrument,
                name=name,
                target_cells=target_cells,
                keep=keep if native else None,
                native=native,
                propagation_form=propagation_form,
            )
        )

    finite = np.isfinite(columns[y_name][0]) & np.isfinite(columns[x_name][0])
    covered = (columns[f"coverage_{y_name}"][0] >= min_coverage) & (
        columns[f"coverage_{x_name}"][0] >= min_coverage
    )
    surviving = np.flatnonzero(finite & covered)
    if surviving.size == 0:
        raise TsaraAlignError(
            f"'{y_name}' and '{x_name}' produced no usable pairs over "
            f"{n_considered} candidate cell(s) of '{clock}'. Either the records do "
            "not overlap, or every candidate cell was masked or below "
            f"min_coverage={min_coverage}."
        )
    final_cells = _subset(target_cells, surviving)

    data_vars = {
        column: (TIME_COORD, values[surviving], attrs)
        for column, (values, attrs) in columns.items()
    }
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={TIME_COORD: np.asarray(final_cells.midpoint_ns, dtype="datetime64[ns]")},
        attrs={
            "tsara_version": __version__,
            "tsara_stage": "paired",
            PAIRING_CLOCK_ATTR: clock,
            PAIRING_REASON_ATTR: reason,
            PAIRING_COVERAGE_ATTR: float(min_coverage),
            PAIRING_CANDIDATE_ATTR: int(n_considered),
            PAIRING_DROPPED_ATTR: int(n_considered - surviving.size),
            PROPAGATION_FORM_ATTR: propagation_form,
        },
    )
    # One implementation of the CF layout, shared with ingestion and the
    # generator: bounds coordinate, the `bounds` attribute, the axis
    # declarations, and the exclusion of the sigma companions -- a sigma
    # describes the uncertainty OF a cell's value, and where that value is an
    # average the two differ by exactly sqrt(N_eff), so `time: mean` on a
    # sigma would be a false claim (§10.2).
    attach_time_bounds(dataset, final_cells, "mean")
    _correct_cell_methods(dataset, streams, y_instrument, y_name, x_instrument, x_name, clock)
    # Pinned here rather than by whoever eventually writes the file. A paired
    # series is a plain Dataset with no bundle of its own, so a user
    # inspecting one in a notebook is the one who calls `to_netcdf` -- and
    # without the pin xarray picks different reference epochs for `time` and
    # `time_bnds` and warns that the result is counter to CF (METHODS §10.2).
    pin_time_encoding(dataset)
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


def _correct_cell_methods(
    dataset: xr.Dataset,
    streams: Mapping[str, xr.Dataset],
    y_instrument: str,
    y_name: str,
    x_instrument: str,
    x_name: str,
    clock: str,
) -> None:
    """Fix the cell methods that a blanket ``time: mean`` gets wrong.

    :func:`~tsara.core.support.attach_time_bounds` stamps one method on every
    non-sigma variable, which is right for a species that was binned onto
    these cells and wrong for three other things this product carries.

    * The species already on the pairing clock was not averaged by this
      stage at all, so it keeps whatever its own stream declared -- stamping
      ``time: mean`` on a point sample would assert an averaging that never
      happened.
    * A contributing-sample count is a **sum** over the cell, which is CF's
      own word for it.
    * A coverage fraction is a property of the cell rather than a statistic
      of the data inside it, so it gets no cell method for the same reason
      the sigma companions get none. What it is is said by its description.
    """
    for instrument, name in ((y_instrument, y_name), (x_instrument, x_name)):
        if instrument == clock:
            declared = streams[instrument][name].attrs.get(CELL_METHODS_ATTR)
            if declared is None:
                dataset[name].attrs.pop(CELL_METHODS_ATTR, None)
            else:
                dataset[name].attrs[CELL_METHODS_ATTR] = declared
        dataset[f"n_source_{name}"].attrs[CELL_METHODS_ATTR] = f"{TIME_COORD}: sum"
        dataset[f"coverage_{name}"].attrs.pop(CELL_METHODS_ATTR, None)


def _one_species(
    stream: xr.Dataset,
    *,
    instrument: str,
    name: str,
    target_cells: CellBounds,
    keep: np.ndarray | None,
    native: bool,
    propagation_form: PropagationForm,
) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    """Return one species' columns on the pairing clock.

    ``native`` means this species already lives on the target cells, so its
    values pass through and are merely restricted to ``keep``. Otherwise it
    is averaged onto them with overlap weights, and its uncertainty is
    propagated through the same weights.
    """
    source_cells = _cells(stream, instrument)
    cell_width_s = _median_width_s(source_cells)
    values = np.asarray(stream[name].values, dtype=np.float64)
    base_attrs: dict[str, object] = dict(stream[name].attrs)
    base_attrs[PAIRING_INSTRUMENT_ATTR] = instrument
    base_attrs[PAIRING_BINNED_ATTR] = int(not native)

    sigmas = {
        component: _sigma_on_cells(stream, name, sigma_name(name), cell_width_s, propagation_form)
        for component, sigma_name in (("random", sigma_rand_name), ("systematic", sigma_sys_name))
    }

    if native:
        index = keep if keep is not None else np.arange(len(source_cells), dtype=np.int64)
        columns: dict[str, tuple[np.ndarray, dict[str, object]]] = {
            name: (values[index], base_attrs)
        }
        n_cells = index.size
        columns[f"n_source_{name}"] = (
            np.ones(n_cells, dtype=np.int64),
            _count_attrs(name, native=True),
        )
        columns[f"coverage_{name}"] = (
            np.ones(n_cells, dtype=np.float64),
            _coverage_attrs(name, native=True),
        )
        for component, resolved in sigmas.items():
            if resolved is None:
                continue
            sigma_values, provenance = resolved
            columns[_sigma_column(component, name)] = (
                sigma_values[index],
                _sigma_attrs(stream, name, component, provenance, form="native"),
            )
        return columns

    pairs = overlap_pairs(source_cells, target_cells)
    n_target = len(target_cells)
    weight = pairs.overlap_ns.astype(np.float64)
    paired = values[pairs.source_index]
    contributes = np.isfinite(paired) & (weight > 0)
    effective_weight = np.where(contributes, weight, 0.0)
    weight_sum = np.bincount(pairs.target_index, weights=effective_weight, minlength=n_target)
    value_sum = np.bincount(
        pairs.target_index,
        weights=effective_weight * np.where(contributes, paired, 0.0),
        minlength=n_target,
    )
    binned = np.full(n_target, np.nan, dtype=np.float64)
    filled = weight_sum > 0
    binned[filled] = value_sum[filled] / weight_sum[filled]
    counts = np.bincount(
        pairs.target_index, weights=contributes.astype(np.float64), minlength=n_target
    ).astype(np.int64)
    width = target_cells.width_ns.astype(np.float64)
    # Coverage can exceed 1 slightly, and is left alone when it does. Fixed
    # width cells centred on jittered timestamps overlap each other, so their
    # overlaps with one target cell can sum past its width (METHODS §10.2,
    # where the effect is documented as benign: the value is a weighted MEAN,
    # so the weights normalize). Clipping would hide a real property of the
    # source record behind a tidier number.
    coverage = np.zeros(n_target, dtype=np.float64)
    wide = width > 0
    coverage[wide] = weight_sum[wide] / width[wide]

    columns = {name: (binned, base_attrs)}
    columns[f"n_source_{name}"] = (counts, _count_attrs(name, native=False))
    columns[f"coverage_{name}"] = (coverage, _coverage_attrs(name, native=False))

    tau = stream[name].attrs.get("decorrelation_timescale")
    tau_s = float(pd.Timedelta(tau).total_seconds()) if tau is not None else None
    spacing_s = _spacing_s(source_cells)
    for component, resolved in sigmas.items():
        if resolved is None:
            continue
        sigma_values, provenance = resolved
        long_form = sigma_values[pairs.source_index]
        if component == "random":
            result = propagate_random_binned(
                long_form,
                effective_weight,
                pairs.target_index,
                n_target,
                spacing_s=spacing_s,
                tau_s=tau_s,
                form=propagation_form,
            )
        else:
            result = propagate_systematic_binned(
                long_form, effective_weight, pairs.target_index, n_target
            )
        columns[_sigma_column(component, name)] = (
            result.sigma,
            _sigma_attrs(stream, name, component, provenance, form=result.form),
        )
    return columns


def _spacing_s(cells: CellBounds) -> float:
    """Return the source stream's sampling interval, in seconds.

    The median gap between consecutive cell starts, which is the cadence in
    the sense the correlation correction needs -- how far apart the samples
    are, not how wide each one is. Falls back to the cell width for a
    single-cell record, where there is no gap to measure.
    """
    if len(cells) < 2:
        return max(_median_width_s(cells), 1.0 / _NS_PER_S)
    deltas = np.diff(cells.start_ns)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return max(_median_width_s(cells), 1.0 / _NS_PER_S)
    return float(np.median(positive)) / _NS_PER_S


def _sigma_column(component: str, name: str) -> str:
    """Return the output column for one uncertainty component."""
    return sigma_rand_name(name) if component == "random" else sigma_sys_name(name)


def _count_attrs(name: str, *, native: bool) -> dict[str, object]:
    """Return attrs for a contributing-sample count."""
    return {
        "description": (
            f"Native cells of {name} contributing to each pair."
            + (" One by construction: this species is on the pairing clock." if native else "")
        ),
        "units": "1",
    }


def _coverage_attrs(name: str, *, native: bool) -> dict[str, object]:
    """Return attrs for a coverage fraction."""
    return {
        "description": (
            f"Fraction of each pairing cell covered by contributing {name} data."
            + (" One by construction: this species defines the cell." if native else "")
        ),
        "units": "1",
    }


def _sigma_attrs(
    stream: xr.Dataset,
    name: str,
    component: str,
    at_support: str,
    *,
    form: str,
) -> dict[str, object]:
    """Return attrs for a propagated uncertainty component.

    Carries three separate facts a reader cannot re-derive from the number:
    where the budget came from (ingestion's provenance), whether the declared
    figure had to be moved onto the stream's own cells and how, and which
    form propagated it through the binning.
    """
    # Spelled in full rather than built from a prefix. The suite discovers the
    # attribute vocabulary by reading string constants out of the source, so a
    # prefix plus an f-string would enter that vocabulary as a fragment that
    # matches nothing in the methods document and can never be documented.
    source_key = (
        "uncertainty_source_random" if component == "random" else "uncertainty_source_systematic"
    )
    return {
        "units": stream[name].attrs.get("units", ""),
        "description": f"{component.capitalize()} 1-sigma for {name} on the pairing clock.",
        "uncertainty_component": component,
        "uncertainty_source": stream[name].attrs.get(source_key, "unknown"),
        SIGMA_AT_SUPPORT_ATTR: at_support,
        PROPAGATION_FORM_ATTR: form,
    }
