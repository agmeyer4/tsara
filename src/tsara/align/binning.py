"""Putting any set of variables onto any set of cells.

The one operation
-----------------
TSARA joins data in exactly one way: every value is averaged onto target
cells, weighted by how much of it falls inside each one. Everything the
package does with more than one clock is that operation with a different
target:

* two species compared for a regression — the target is the cells of the
  wider-supported member (:mod:`tsara.align.pairing`, ``docs/METHODS.md``
  §1.3);
* a campaign-wide matrix for receptor modelling or a continuous rolling
  state — the target is a uniform grid (§1.4).

They are not two designs. Writing them as two implementations was the
mistake this module exists to undo: the pairwise form is the general form
with two variables selected and incomplete rows dropped.

Deliberately variable-agnostic
-------------------------------
This function does not know what a species is. It takes whatever variables it
is handed — raw concentrations today, baselines and enhancements once Phase 5
computes them, met, anything a later stage invents — and puts them on the
support asked for. That is the point: TSARA is a loader and transformer for
sweeping analysis choices, so the joining block must not need editing every
time a new kind of variable appears upstream of it.

What travels with a variable
-----------------------------
Automatically, so a caller cannot forget:

* its **uncertainty components**, propagated through the *same* overlap
  weights that formed the value, random and systematic separately (§3);
* **how many** source cells contributed and **how much** of the target cell
  they covered — the two numbers that separate a well-determined value from
  a number that merely exists;
* everything the source declared about itself, plus where it came from.

Three behaviours are not negotiable and are handled here rather than left to
callers. A variable declaring ``circular: 1`` is vector-averaged, never
arithmetically (§11.5). A stream whose cells already *are* the target passes
through untouched, because averaging a cell onto itself is the identity
mathematically and not in floating point. And a target cell with no
contributing data stays ``nan``: gases are binned, never interpolated (§1.2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from tsara import __version__
from tsara.core.bundle import pin_time_encoding
from tsara.core.circular import bin_circular_onto_cells
from tsara.core.exceptions import TsaraError
from tsara.core.naming import (
    CELL_METHODS_ATTR,
    TIME_BOUNDS_VAR,
    TIME_COORD,
    is_sigma_name,
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
    from collections.abc import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "TsaraAlignError",
    "VariableRef",
    "bin_streams_onto_cells",
    "resolve_variable",
    "select_variables",
]

#: How a caller names a variable: by its canonical name, or by the instrument
#: that measured it when more than one did.
VariableRef = "str | tuple[str, str]"

#: Attrs the joined product carries, documented in ``docs/METHODS.md`` §11.2.
SOURCE_INSTRUMENT_ATTR = "tsara_source_instrument"
BINNED_ATTR = "tsara_binned"
PROPAGATION_FORM_ATTR = "tsara_propagation_form"
SIGMA_AT_SUPPORT_ATTR = "tsara_sigma_at_support"
RESULTANT_SUFFIX = "_resultant_length"
DISPERSION_SUFFIX = "_dispersion"

#: Nanoseconds per second, for turning cell widths into the units the
#: propagation module speaks.
NS_PER_S = 1_000_000_000.0


class TsaraAlignError(TsaraError):
    """Raised when variables cannot be put on a common support as asked.

    Its own type because the failures are about *combining* streams rather
    than about reading them: a variable no stream declares, a name two
    streams both declare where one was expected, or a stream carrying no
    cells to bin from.
    """


def resolve_variable(
    streams: Mapping[str, xr.Dataset], reference: str | tuple[str, str]
) -> tuple[str, str]:
    """Return ``(instrument, variable)`` for a caller's reference.

    A bare name is searched across the streams. Two instruments measuring the
    same canonical name is a real situation — it is how a campaign compares
    two analyzers — so ambiguity is an error naming both rather than a silent
    choice of the first.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    reference : str or tuple of (str, str)
        Canonical name, or an explicit instrument and variable.

    Returns
    -------
    tuple of (str, str)
        Instrument name and variable name.

    Raises
    ------
    TsaraAlignError
        If the reference names nothing, or names something ambiguously.
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
            "so naming it alone is ambiguous. Pass an (instrument, variable) pair to "
            "say which one."
        )
    return holders[0], reference


def stream_cells(stream: xr.Dataset, instrument: str) -> CellBounds:
    """Return a stream's cells, or raise naming the instrument.

    Parameters
    ----------
    stream : xarray.Dataset
        The stream.
    instrument : str
        Its name, used only so the error says which one is at fault.

    Returns
    -------
    CellBounds
        The stream's cell boundaries.

    Raises
    ------
    TsaraAlignError
        If the stream has no bounds or no rows.
    """
    if TIME_BOUNDS_VAR not in stream.coords and TIME_BOUNDS_VAR not in stream.data_vars:
        raise TsaraAlignError(
            f"Stream '{instrument}' carries no '{TIME_BOUNDS_VAR}', so there is no "
            "interval to bin over. Streams gain cells at ingestion (METHODS §10); "
            "a bundle written before format 2 must be reloaded to acquire them."
        )
    bounds = np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]").astype(np.int64)
    if bounds.size == 0:
        raise TsaraAlignError(f"Stream '{instrument}' has no cells to bin from.")
    return CellBounds(start_ns=bounds[:, 0].copy(), stop_ns=bounds[:, 1].copy())


def median_width_s(cells: CellBounds) -> float:
    """Return the median cell width in seconds."""
    return float(np.median(cells.width_ns)) / NS_PER_S


def cadence_s(cells: CellBounds) -> float:
    """Return the interval between consecutive cell starts, in seconds.

    The cadence in the sense a correlation correction needs — how far apart
    the samples are, not how wide each one is. Falls back to the cell width
    where there is no gap to measure, which happens for a single-cell record
    and for a file that nests one sample inside another.
    """
    if len(cells) >= 2:
        deltas = np.diff(cells.start_ns)
        positive = deltas[deltas > 0]
        if positive.size:
            return float(np.median(positive)) / NS_PER_S
    return max(median_width_s(cells), 1.0 / NS_PER_S)


def select_variables(
    streams: Mapping[str, xr.Dataset],
    variables: Sequence[str | tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Resolve which variables a call acts on, as ``(instrument, name)`` pairs.

    Public because more than one stage needs the *same* answer: the binner
    uses it to decide what to join, and the output grid uses it to decide
    which cell widths its period has to respect (``docs/METHODS.md`` §11.7).
    Two implementations of "which variables did the caller mean" would let
    the grid validate one set and bin another.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    variables : sequence of str or (str, str), optional
        Explicit selection. ``None`` takes every non-sigma variable in every
        stream, in stream order.

    Returns
    -------
    list of tuple of (str, str)
        Instrument and variable name for each selection.

    Notes
    -----
    Sigma companions are excluded from the default because they are not
    variables in their own right: each travels automatically with the value
    it describes, and selecting one directly would produce a column with no
    parent and no meaning.
    """
    if variables is not None:
        return [resolve_variable(streams, reference) for reference in variables]
    return [
        (instrument, str(name))
        for instrument, stream in streams.items()
        for name in stream.data_vars
        if not is_sigma_name(str(name))
    ]


def _output_names(selection: Sequence[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Return the column name for each selected variable.

    A canonical name is kept as it is when only one selected stream carries
    it, which is the ordinary case and keeps a joined product readable. When
    two streams carry the same name — a campaign comparing two analyzers —
    both are suffixed with their instrument rather than one silently winning.
    The spelling therefore depends on the *selection*, which is why every
    column also records its source instrument in an attribute.
    """
    claimed: dict[str, list[str]] = {}
    for instrument, variable in selection:
        claimed.setdefault(variable, []).append(instrument)
    names: dict[tuple[str, str], str] = {}
    for instrument, variable in selection:
        if len(claimed[variable]) > 1:
            names[instrument, variable] = f"{variable}_{instrument}"
        else:
            names[instrument, variable] = variable
    return names


def _sigma_on_cells(
    stream: xr.Dataset,
    variable: str,
    sigma_variable: str,
    cell_width_s: float,
    form: PropagationForm,
) -> tuple[np.ndarray, str] | None:
    """Return a declared sigma restated on the stream's own cells, and how.

    Ingestion stores a declared figure exactly as declared and records the
    interval it was quoted at, deliberately performing no arithmetic on it
    (§10.8). This is the point of use, so this is where the arithmetic
    happens — or is refused and says so.
    """
    if sigma_variable not in stream.data_vars:
        return None
    values = np.asarray(stream[sigma_variable].values, dtype=np.float64)
    quoted = stream[variable].attrs.get("uncertainty_at_width")
    if quoted is None:
        return values, "unchanged"
    tau = stream[variable].attrs.get("decorrelation_timescale")
    tau_s = float(pd.Timedelta(tau).total_seconds()) if tau is not None else None
    moved, provenance = sigma_at_support(
        values,
        quoted_width_s=float(pd.Timedelta(quoted).total_seconds()),
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


def _same_cells(source: CellBounds, target: CellBounds) -> bool:
    """Return whether two sets of cells are identical.

    Tested on the *cells*, not on the instrument name, because that is the
    property that matters: any stream already on the target support must pass
    through untouched. Averaging a cell onto itself is the identity
    mathematically and not in floating point, and several gases retrieved
    from one spectrum is the commonest case there is.
    """
    return len(source) == len(target) and bool(
        np.array_equal(source.start_ns, target.start_ns)
        and np.array_equal(source.stop_ns, target.stop_ns)
    )


def bin_streams_onto_cells(
    streams: Mapping[str, xr.Dataset],
    target: CellBounds,
    variables: Sequence[str | tuple[str, str]] | None = None,
    *,
    propagation_form: PropagationForm = "ar1_neff",
) -> xr.Dataset:
    """Put any set of variables onto one set of cells.

    The joining primitive. Each variable is averaged onto every target cell,
    weighted by overlap; its uncertainty is propagated through the same
    weights; and the count and coverage that qualify the result travel with
    it (``docs/METHODS.md`` §11.2).

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        A campaign's streams, e.g. a
        :class:`~tsara.ingest.campaign.StreamCollection`.
    target : CellBounds
        The cells to bin onto — another stream's, or a uniform grid's.
    variables : sequence of str or (str, str), optional
        Which variables to include. ``None`` takes every non-sigma variable
        in every stream, which is the exploratory default; a receptor-model
        matrix normally names its columns.
    propagation_form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form reduces a correlated random component (§3.4).

    Returns
    -------
    xarray.Dataset
        One column per selected variable, plus ``n_source_<name>`` and
        ``coverage_<name>`` for each, plus whatever uncertainty components
        were available. Angular variables gain ``<name>_resultant_length``
        and ``<name>_dispersion`` instead of a sigma. Cells are described by
        CF ``time_bnds``.

    Raises
    ------
    TsaraAlignError
        If no streams are given, a variable cannot be resolved, or a stream
        has no cells.
    """
    if not streams:
        raise TsaraAlignError("No streams to bin.")
    if len(target) == 0:
        raise TsaraAlignError("No target cells to bin onto.")
    selection = select_variables(streams, variables)
    if not selection:
        raise TsaraAlignError(
            "No variables selected. Every stream holds only uncertainty companions, "
            "which travel with the values they describe rather than being binned "
            "on their own."
        )
    names = _output_names(selection)
    data_vars: dict[str, tuple[str, np.ndarray, dict[str, object]]] = {}
    native: dict[str, str | None] = {}
    for instrument, variable in selection:
        column = names[instrument, variable]
        columns, declared_method = _one_variable(
            streams[instrument],
            instrument=instrument,
            variable=variable,
            column=column,
            target=target,
            propagation_form=propagation_form,
        )
        data_vars.update({name: (TIME_COORD, *rest) for name, rest in columns.items()})
        native[column] = declared_method

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={TIME_COORD: np.asarray(target.midpoint_ns, dtype="datetime64[ns]")},
        attrs={
            "tsara_version": __version__,
            "tsara_stage": "binned",
            PROPAGATION_FORM_ATTR: propagation_form,
        },
    )
    attach_time_bounds(dataset, target, "mean")
    _correct_cell_methods(dataset, names.values(), native)
    # Pinned here rather than by whoever eventually writes the file: a joined
    # product is an ordinary Dataset with no bundle of its own, so the user
    # inspecting one in a notebook is the one who calls `to_netcdf`, and
    # without the pin xarray picks different reference epochs for `time` and
    # `time_bnds` and warns that the result is counter to CF (§10.2).
    pin_time_encoding(dataset)
    return dataset


def _correct_cell_methods(
    dataset: xr.Dataset, columns: Iterable[str], native: Mapping[str, str | None]
) -> None:
    """Fix the cell methods a blanket ``time: mean`` gets wrong.

    :func:`~tsara.core.support.attach_time_bounds` stamps one method on every
    non-sigma variable, which is right for a value that was averaged onto
    these cells and wrong for three other things this product carries.

    * A variable already on the target cells was not averaged by this stage
      at all, so it keeps whatever its own stream declared. Stamping
      ``time: mean`` on a point sample would assert an averaging that never
      happened.
    * A contributing-sample count is a **sum** over the cell, CF's own word.
    * A coverage fraction, a resultant length and a dispersion are properties
      of the cell rather than statistics of the data inside it, so they get
      no cell method — the same reason the sigma companions get none (§10.2).
    """
    for column in columns:
        declared = native.get(column, "__binned__")
        if declared != "__binned__":
            if declared is None:
                dataset[column].attrs.pop(CELL_METHODS_ATTR, None)
            else:
                dataset[column].attrs[CELL_METHODS_ATTR] = declared
        dataset[f"n_source_{column}"].attrs[CELL_METHODS_ATTR] = f"{TIME_COORD}: sum"
        for suffix in ("coverage_{}", "{}" + RESULTANT_SUFFIX, "{}" + DISPERSION_SUFFIX):
            name = suffix.format(column)
            if name in dataset.data_vars:
                dataset[name].attrs.pop(CELL_METHODS_ATTR, None)


def _one_variable(
    stream: xr.Dataset,
    *,
    instrument: str,
    variable: str,
    column: str,
    target: CellBounds,
    propagation_form: PropagationForm,
) -> tuple[dict[str, tuple[np.ndarray, dict[str, object]]], str | None]:
    """Return one variable's columns on the target cells.

    The second return value says whether the variable was already on the
    target support, and if so what cell method its own stream declared — the
    only thing the caller cannot re-derive from the columns themselves.
    """
    source = stream_cells(stream, instrument)
    values = np.asarray(stream[variable].values, dtype=np.float64)
    attrs: dict[str, object] = dict(stream[variable].attrs)
    attrs[SOURCE_INSTRUMENT_ATTR] = instrument
    circular = str(attrs.get("circular", 0)) not in ("0", "False", "None", "")
    already_here = _same_cells(source, target)
    attrs[BINNED_ATTR] = int(not already_here)

    if already_here:
        columns: dict[str, tuple[np.ndarray, dict[str, object]]] = {column: (values, attrs)}
        n_cells = len(target)
        columns[f"n_source_{column}"] = (
            np.ones(n_cells, dtype=np.int64),
            _count_attrs(column, native=True),
        )
        columns[f"coverage_{column}"] = (
            np.ones(n_cells, dtype=np.float64),
            _coverage_attrs(column, native=True),
        )
        for component, sigma_name in (("random", sigma_rand_name), ("systematic", sigma_sys_name)):
            resolved = _sigma_on_cells(
                stream, variable, sigma_name(variable), median_width_s(source), propagation_form
            )
            if resolved is None:
                continue
            sigma_values, provenance = resolved
            columns[sigma_name(column)] = (
                sigma_values,
                _sigma_attrs(stream, variable, component, provenance, form="native"),
            )
        return columns, stream[variable].attrs.get(CELL_METHODS_ATTR)

    if circular:
        return _bin_circular(stream, variable, column, source, target, attrs), "__binned__"
    return _bin_scalar(
        stream, variable, column, source, target, values, attrs, propagation_form
    ), "__binned__"


def _bin_circular(
    stream: xr.Dataset,
    variable: str,
    column: str,
    source: CellBounds,
    target: CellBounds,
    attrs: dict[str, object],
) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    """Vector-average an angular variable and carry its quality numbers.

    An angle has no meaningful arithmetic mean, so it gets none. What it gets
    instead is the mean resultant length, which is the honest statement of
    how well determined the direction is, and the exact circular standard
    deviation derived from it (§11.5).
    """
    result = bin_circular_onto_cells(
        source, np.asarray(stream[variable].values, dtype=np.float64), target
    )
    units = str(attrs.get("units", "degrees"))
    return {
        column: (result.mean_deg, attrs),
        f"n_source_{column}": (result.n_source, _count_attrs(column, native=False)),
        f"coverage_{column}": (result.coverage, _coverage_attrs(column, native=False)),
        f"{column}{RESULTANT_SUFFIX}": (
            result.resultant_length,
            {
                "units": "1",
                "description": (
                    f"Mean resultant length of {column} per cell: 1 when every sample "
                    "agreed, 0 when they cancelled and there is no direction."
                ),
            },
        ),
        f"{column}{DISPERSION_SUFFIX}": (
            result.dispersion_deg,
            {
                "units": units,
                "description": (
                    f"Exact circular standard deviation of {column} per cell, "
                    "unbounded by construction (METHODS §11.5)."
                ),
            },
        ),
    }


def _bin_scalar(
    stream: xr.Dataset,
    variable: str,
    column: str,
    source: CellBounds,
    target: CellBounds,
    values: np.ndarray,
    attrs: dict[str, object],
    propagation_form: PropagationForm,
) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    """Overlap-weighted mean of one scalar variable, with its uncertainty."""
    pairs = overlap_pairs(source, target)
    n_target = len(target)
    paired = values[pairs.source_index]
    contributes = np.isfinite(paired) & (pairs.overlap_ns > 0)
    weight = np.where(contributes, pairs.overlap_ns, 0.0).astype(np.float64)
    weight_sum = np.bincount(pairs.target_index, weights=weight, minlength=n_target)
    value_sum = np.bincount(
        pairs.target_index,
        weights=weight * np.where(contributes, paired, 0.0),
        minlength=n_target,
    )
    binned = np.full(n_target, np.nan, dtype=np.float64)
    filled = weight_sum > 0
    binned[filled] = value_sum[filled] / weight_sum[filled]
    counts = np.bincount(
        pairs.target_index, weights=contributes.astype(np.float64), minlength=n_target
    ).astype(np.int64)
    # Coverage can exceed 1 slightly, and is left alone when it does.
    # Fixed-width cells centred on jittered timestamps overlap each other, so
    # their overlaps with one target cell can sum past its width (§10.2, where
    # the effect is documented as benign: the value is a weighted MEAN, so the
    # weights normalize). Clipping would hide a real property of the source
    # record behind a tidier number.
    width = target.width_ns.astype(np.float64)
    coverage = np.zeros(n_target, dtype=np.float64)
    wide = width > 0
    coverage[wide] = weight_sum[wide] / width[wide]

    columns: dict[str, tuple[np.ndarray, dict[str, object]]] = {
        column: (binned, attrs),
        f"n_source_{column}": (counts, _count_attrs(column, native=False)),
        f"coverage_{column}": (coverage, _coverage_attrs(column, native=False)),
    }
    tau = stream[variable].attrs.get("decorrelation_timescale")
    tau_s = float(pd.Timedelta(tau).total_seconds()) if tau is not None else None
    spacing = cadence_s(source)
    cell_width = median_width_s(source)
    for component, sigma_name in (("random", sigma_rand_name), ("systematic", sigma_sys_name)):
        resolved = _sigma_on_cells(
            stream, variable, sigma_name(variable), cell_width, propagation_form
        )
        if resolved is None:
            continue
        sigma_values, provenance = resolved
        long_form = sigma_values[pairs.source_index]
        if component == "random":
            result = propagate_random_binned(
                long_form,
                weight,
                pairs.target_index,
                n_target,
                spacing_s=spacing,
                tau_s=tau_s,
                form=propagation_form,
            )
        else:
            result = propagate_systematic_binned(long_form, weight, pairs.target_index, n_target)
        columns[sigma_name(column)] = (
            result.sigma,
            _sigma_attrs(stream, variable, component, provenance, form=result.form),
        )
    return columns


def _count_attrs(column: str, *, native: bool) -> dict[str, object]:
    """Return attrs for a contributing-sample count."""
    return {
        "description": (
            f"Source cells of {column} contributing to each target cell."
            + (" One by construction: already on this support." if native else "")
        ),
        "units": "1",
    }


def _coverage_attrs(column: str, *, native: bool) -> dict[str, object]:
    """Return attrs for a coverage fraction."""
    return {
        "description": (
            f"Fraction of each target cell covered by contributing {column} data."
            + (" One by construction: already on this support." if native else "")
        ),
        "units": "1",
    }


def _sigma_attrs(
    stream: xr.Dataset,
    variable: str,
    component: str,
    at_support: str,
    *,
    form: str,
) -> dict[str, object]:
    """Return attrs for a propagated uncertainty component.

    Carries three facts a reader cannot re-derive from the number: where the
    budget came from, whether a declared figure had to be moved onto the
    stream's own cells and how, and which form propagated it through the
    binning.
    """
    # Spelled in full rather than built from a prefix. The suite discovers the
    # attribute vocabulary by reading string constants out of the source, so a
    # prefix plus an f-string would enter that vocabulary as a fragment that
    # matches nothing in the methods document and can never be documented.
    source_key = (
        "uncertainty_source_random" if component == "random" else "uncertainty_source_systematic"
    )
    return {
        "units": stream[variable].attrs.get("units", ""),
        "description": f"{component.capitalize()} 1-sigma for {variable} on the target cells.",
        "uncertainty_component": component,
        "uncertainty_source": stream[variable].attrs.get(source_key, "unknown"),
        SIGMA_AT_SUPPORT_ATTR: at_support,
        PROPAGATION_FORM_ATTR: form,
    }
