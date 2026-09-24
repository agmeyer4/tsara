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
* **how many** readings contributed, **how much** of the target cell they
  covered, and **how much** of the value rests on air outside the cell — the
  three numbers that separate a well-determined value from a number that
  merely exists (§11.2.4);
* per column, what the join did to the readings behind it and by how much:
  ``tsara_support_transform``, ``tsara_width_ratio_max``,
  ``tsara_borrowed_share`` and ``tsara_n_readings``;
* everything the input stream declared about itself, plus where it came from.

Three behaviours are not negotiable and are handled here rather than left to
callers. A variable declaring ``circular: 1`` is vector-averaged, never
arithmetically (§11.5). A stream whose cells already *are* the target passes
through untouched, because averaging a cell onto itself is the identity
mathematically and not in floating point. And a target cell with no
contributing data stays ``nan``: gases are binned, never interpolated (§1.2).

One behaviour is a policy. A reading at least :data:`~tsara.align.cells.COPY_RATIO` times as wide
as a target cell it touches would be *copied* across rows, and is refused
unless ``finer_support="allow"`` asks for it by name; everything narrower is
allowed, recorded per column, and named in one warning per call (§11.2.4).

Where the words come from
-------------------------
Which variables a call means, and the refusal of a product that is itself a
join, are :mod:`tsara.align.variables`; a stream's cells and every measure of
how readings meet target cells are :mod:`tsara.align.cells`. This module is
the operation spoken in that vocabulary, laid out to be read from the top:
the entry point first, then its steps in the order it takes them, then what
each column of the product says about itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from tsara import __version__
from tsara.align.cells import (
    COPY_RATIO,
    cadence_s,
    median_width_s,
    pair_width_ratios,
    stream_cells,
    targets_overlap,
)
from tsara.align.variables import TsaraAlignError, VariableRef, select_variables
from tsara.core.bundle import pin_time_encoding
from tsara.core.circular import bin_circular_onto_cells
from tsara.core.naming import (
    CELL_METHODS_ATTR,
    DISPERSION_SUFFIX,
    RESULTANT_LENGTH_SUFFIX,
    TIME_COORD,
    borrowed_name,
    coverage_name,
    is_circular,
    n_readings_name,
    sigma_rand_name,
    sigma_sys_name,
)
from tsara.core.propagation import (
    PropagationForm,
    propagate_random_binned,
    propagate_systematic_binned,
    sigma_at_support,
)
from tsara.core.support import (
    CellBounds,
    attach_time_bounds,
    bin_onto_cells,
    contributing_weights,
    overlap_pairs,
)
from tsara.core.timebase import NS_PER_S

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    from tsara.core.support import OverlapPairs

logger = logging.getLogger(__name__)

__all__ = [
    "BINNED_ATTR",
    "BORROWED_ATTR",
    "FinerSupport",
    "INSTRUMENT_ATTR",
    "PROPAGATION_FORM_ATTR",
    "READINGS_ATTR",
    "SIGMA_AT_SUPPORT_ATTR",
    "SupportTransform",
    "TRANSFORM_ATTR",
    "WIDTH_RATIO_ATTR",
    "bin_streams_onto_cells",
]


#: Attrs the joined product carries, documented in ``docs/METHODS.md`` §11.2.
INSTRUMENT_ATTR = "tsara_instrument"
BINNED_ATTR = "tsara_binned"
PROPAGATION_FORM_ATTR = "tsara_propagation_form"
SIGMA_AT_SUPPORT_ATTR = "tsara_sigma_at_support"


#: Per-column attrs recording how a join changed the support of what it holds
#: (``docs/METHODS.md`` §11.2.4): the worst thing that happened to any reading
#: behind the column, in a word; the largest reading-to-cell width ratio among
#: the readings that formed a value; the share of the column's covered time
#: whose value rests on air outside its cell; and how many distinct readings
#: stand behind the column's rows.
TRANSFORM_ATTR = "tsara_support_transform"
WIDTH_RATIO_ATTR = "tsara_width_ratio_max"
BORROWED_ATTR = "tsara_borrowed_share"
READINGS_ATTR = "tsara_n_readings"


#: What a join may do with a reading at or beyond :data:`~tsara.align.cells.COPY_RATIO`.
#: ``refuse`` is the default and the interpolation rule's guarantee (§1.2);
#: ``allow`` copies the reading across rows, labels every affected column
#: ``copied``, records how much of each value was borrowed, and warns.
FinerSupport = Literal["refuse", "allow"]


#: The vocabulary of ``tsara_support_transform``, one word per column naming
#: the worst thing the join did to any reading behind it. Derived from two
#: numbers without a tolerance: ``passthrough`` when the stream's cells are
#: the target; ``averaged`` when every contributing reading sat wholly inside
#: its cell (borrowed share exactly 0); ``straddled`` when some reading lay
#: across a cell boundary but none was wider than a cell it filled;
#: ``narrowed`` when one was, by less than :data:`~tsara.align.cells.COPY_RATIO`; ``copied`` at
#: or beyond it. The word *shared* is kept for a different fact, a reading
#: that formed more than one row's value (:func:`~tsara.align.cells.shared_readings`, §11.4.1):
#: a straddled reading in a product whose rows sit cells apart is not shared.
SupportTransform = Literal["passthrough", "averaged", "straddled", "narrowed", "copied"]


#: Sentinel standing where a variable's own ``cell_methods`` would be, saying
#: that this call averaged it rather than passing it through.
#:
#: ``None`` cannot do the job: a stream that declares no cell method at all is
#: a real case, and must stay distinguishable from one this stage averaged.
#: Named rather than repeated as a literal in four places, where a typo in one
#: of them would silently restore a cell method to a variable that was binned.
_BINNED_HERE = "__binned__"


@dataclass(frozen=True)
class _SupportChange:
    """How a join changed one column's support: its four attrs, and the warning's input."""

    transform: SupportTransform
    width_ratio_max: float
    borrowed_share: float
    rows: int
    readings: int


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def bin_streams_onto_cells(
    streams: Mapping[str, xr.Dataset],
    target: CellBounds,
    variables: Sequence[VariableRef] | None = None,
    *,
    propagation_form: PropagationForm = "ar1_neff",
    finer_support: FinerSupport = "refuse",
) -> xr.Dataset:
    """Put any set of variables onto one set of cells.

    The joining primitive. Each variable is averaged onto every target cell,
    weighted by overlap; its uncertainty is propagated through the same
    weights; and the count, coverage and borrowed share that qualify the
    result travel with it (``docs/METHODS.md`` §11.2, §11.2.4).

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        A campaign's streams, e.g. a
        :class:`~tsara.ingest.campaign.StreamCollection`.
    target : CellBounds
        The cells to bin onto — another stream's, a uniform grid's, or any
        cells a caller builds. They need not tile, be sorted, or be disjoint:
        each row is the definition applied to that cell alone, which is what
        lets §5 roll overlapping windows and §6 evaluate nested event windows
        without a second implementation. Two consequences worth knowing:
        unsorted targets give a non-monotonic ``time`` coordinate, and
        overlapping or repeated targets give repeated ``time`` labels. Both
        save and reload and both index with ``.sel``; ``resample`` and
        ``interp`` expect an ordered, unique index and do not.
    variables : sequence of str or (str, str), optional
        Which variables to include. ``None`` takes every non-sigma variable
        in every stream, which is the exploratory default; a receptor-model
        matrix normally names its columns.
    propagation_form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form reduces a correlated random component (§3.4).
    finer_support : {'refuse', 'allow'}, optional
        What to do when a reading is at least :data:`~tsara.align.cells.COPY_RATIO` times as
        wide as a target cell it touches, so that it would be copied across
        rows. ``refuse`` (the default) raises; ``allow`` copies it, labels the
        column ``copied``, records how much of each value was borrowed and
        warns (§11.2.4). Config: ``AlignmentConfig.finer_support``.

    Returns
    -------
    xarray.Dataset
        One column per selected variable, plus ``n_readings_<name>``,
        ``coverage_<name>`` and ``borrowed_<name>`` for each, plus whatever
        uncertainty components were available. Angular variables gain
        ``<name>_resultant_length`` and ``<name>_dispersion`` instead of a
        sigma. Every column records ``tsara_support_transform``,
        ``tsara_width_ratio_max``, ``tsara_borrowed_share`` and
        ``tsara_n_readings``. Cells are described by CF ``time_bnds``.

    Raises
    ------
    TsaraAlignError
        If no streams are given, a variable cannot be resolved, a stream has
        no cells, or a reading would be copied across rows and
        ``finer_support`` is ``refuse``.
    """
    # 1. Validate what was asked for, and resolve it to (instrument, variable) pairs.
    if not streams:
        raise TsaraAlignError("No streams to bin.")
    if len(target) == 0:
        raise TsaraAlignError("No target cells to bin onto.")
    selection = select_variables(streams, variables)
    if not selection:
        raise TsaraAlignError(
            "No variables selected. Every stream holds only companion columns — "
            "uncertainties, counts, coverage fractions, borrowed shares, angular "
            "quality numbers — which travel with the values they describe rather "
            "than being binned on their own."
        )
    # 2. Decide each output column's name (suffixed only on a collision).
    names = _output_names(selection)
    # 3. Each instrument's cells, their overlaps with the target and the
    # reading-to-cell width ratio of every overlap, found ONCE per instrument
    # rather than once per variable: all three depend only on the cells, and a
    # spectral stream can carry a thousand columns on one clock. A stream
    # already on the target cells passes through and needs no search at all --
    # nor the copy check, since a reading that *is* its target fills exactly
    # one cell. Every other stream is refused here (or loudly allowed) if the
    # join would copy its readings across rows, before anything is averaged.
    joins: dict[str, tuple[CellBounds, OverlapPairs | None, np.ndarray | None]] = {}
    for instrument in dict.fromkeys(name for name, _ in selection):
        readings = stream_cells(streams[instrument], instrument)
        if _same_cells(readings, target):
            joins[instrument] = (readings, None, None)
            continue
        found = overlap_pairs(readings, target)
        found_ratios = pair_width_ratios(readings, target, found)
        _refuse_upsampling(readings, target, instrument, found, found_ratios, finer_support)
        joins[instrument] = (readings, found, found_ratios)
    # 4. Put each variable on the target cells: its value, its companions
    # (count, coverage, borrowed share, sigmas or angular quality), its
    # attributes, and the record of what the join did to its support.
    data_vars: dict[str, tuple[str, np.ndarray, dict[str, object]]] = {}
    # Per column: the cell method a passed-through variable declared, or a
    # sentinel saying this call averaged it. Needed to fix cell_methods below.
    native: dict[str, str | None] = {}
    changes: dict[str, _SupportChange] = {}
    for instrument, variable in selection:
        column = names[instrument, variable]
        cells, pairs, ratios = joins[instrument]
        columns, declared_method, change = _one_variable(
            streams[instrument],
            instrument=instrument,
            variable=variable,
            column=column,
            target=target,
            propagation_form=propagation_form,
            readings=cells,
            pairs=pairs,
            ratios=ratios,
        )
        data_vars.update({name: (TIME_COORD, *rest) for name, rest in columns.items()})
        native[column] = declared_method
        changes[column] = change

    # 5. Assemble the product: one row per target cell, `time` at each midpoint.
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={TIME_COORD: np.asarray(target.midpoint_ns, dtype="datetime64[ns]")},
        attrs={
            "tsara_version": __version__,
            "tsara_stage": "binned",
        },
    )
    # 6. Describe the cells (CF `time_bnds`), then correct the blanket
    # `time: mean` that attaching them stamps on every column.
    attach_time_bounds(dataset, target, "mean")
    _correct_cell_methods(dataset, names.values(), native)
    # 7. Say, once, which columns rest on borrowed air (§11.2.4).
    _warn_about_support_changes(changes, targets_overlap(target))
    # Pinned here rather than by whoever eventually writes the file: a joined
    # product is an ordinary Dataset with no bundle of its own, so the user
    # inspecting one in a notebook is the one who calls `to_netcdf`, and
    # without the pin xarray picks different reference epochs for `time` and
    # `time_bnds` and warns that the result is counter to CF (§10.2).
    pin_time_encoding(dataset)
    return dataset


# ---------------------------------------------------------------------------
# Its steps, in the order it takes them
# ---------------------------------------------------------------------------


def _output_names(selection: Sequence[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Return the column name for each selected variable.

    A canonical name is kept as it is when only one selected stream carries
    it, which is the ordinary case and keeps a joined product readable. When
    two streams carry the same name — a campaign comparing two analyzers —
    both are suffixed with their instrument rather than one silently winning.
    The spelling therefore depends on the *selection*, which is why every
    column also records the instrument it came from in an attribute.
    """
    # First pass: which instruments claim each canonical name.
    claimed: dict[str, list[str]] = {}
    for instrument, variable in selection:
        claimed.setdefault(variable, []).append(instrument)
    # Second pass: suffix a name with its instrument only where two claim it.
    names: dict[tuple[str, str], str] = {}
    for instrument, variable in selection:
        if len(claimed[variable]) > 1:
            names[instrument, variable] = f"{variable}_{instrument}"
        else:
            names[instrument, variable] = variable
    return names


def _same_cells(readings: CellBounds, target: CellBounds) -> bool:
    """Return whether two sets of cells are identical.

    Tested on the *cells*, not on the instrument name, because that is the
    property that matters: any stream already on the target support must pass
    through untouched. Averaging a cell onto itself is the identity
    mathematically and not in floating point, and several gases retrieved
    from one spectrum is the commonest case there is.
    """
    # Same length first, so the element-wise comparison is only made when it can succeed.
    return len(readings) == len(target) and bool(
        np.array_equal(readings.start_ns, target.start_ns)
        and np.array_equal(readings.stop_ns, target.stop_ns)
    )


def _refuse_upsampling(
    readings: CellBounds,
    target: CellBounds,
    instrument: str,
    pairs: OverlapPairs,
    ratios: np.ndarray,
    finer_support: FinerSupport,
) -> None:
    """Refuse, or loudly allow, a join that would copy a reading across rows.

    Both of this phase's products already prevent this by choosing their
    target: pairing takes the wider-supported member's cells, and the grid
    checks its period with this same ratio before building anything. The
    primitive they share has to refuse it too, because it is public and is
    the documented way to build a receptor-model matrix from a chosen set of
    columns -- a caller supplying their own target cells would otherwise get
    the one thing this package promises not to do by default, with
    ``coverage`` reporting 1.0 and nothing else to notice it by.

    Asked of the cells alone, before any value is looked at: what is refused
    is the operation requested for this instrument, not the readings that
    happen to be finite today.
    """
    touching = pairs.overlap_ns > 0
    if not touching.any():
        return
    worst = int(np.argmax(np.where(touching, ratios, -np.inf)))
    ratio = float(ratios[worst])
    if ratio < COPY_RATIO:
        return
    # Any cell wider than half the widest reading would not be a copy of it.
    widest_s = float(readings.width_ns[pairs.reading_index[touching]].max()) / NS_PER_S
    if finer_support == "allow":
        logger.warning(
            "finer_support='allow': readings of '%s' up to %.3g times as wide as the cells "
            "they fill will be copied across rows, resolution the instrument never had. "
            "Every affected column is labelled 'copied' and carries how much of each value "
            "was borrowed (METHODS §11.2.4); a fit or receptor model must not treat its rows "
            "as independent measurements.",
            instrument,
            ratio,
        )
        return
    raise TsaraAlignError(
        f"Stream '{instrument}' has {median_width_s(readings):.6g} s cells and the target "
        f"cells are {median_width_s(target):.6g} s, so one of its readings is {ratio:.3g} "
        "times as wide as a cell it fills. That reading would be copied across rows, which "
        "is resolution the instrument never had (METHODS §11.2.4), and the copies would "
        f"enter a fit as independent measurements. Bin onto cells wider than "
        f"{widest_s / COPY_RATIO:.6g} s, pair on the wider-supported stream, interpolate a "
        "smooth non-gas field with tsara.align.auxiliary (§11.6), or pass "
        "finer_support='allow' to copy it, labelled."
    )


def _one_variable(
    stream: xr.Dataset,
    *,
    instrument: str,
    variable: str,
    column: str,
    target: CellBounds,
    propagation_form: PropagationForm,
    readings: CellBounds,
    pairs: OverlapPairs | None,
    ratios: np.ndarray | None,
) -> tuple[dict[str, tuple[np.ndarray, dict[str, object]]], str | None, _SupportChange]:
    """Return one variable's columns on the target cells, and what the join did to it.

    ``readings`` are the variable's own cells, which its values describe;
    ``pairs`` their overlaps with the target and ``ratios`` the width ratio of
    each — or both ``None`` when the caller found the stream already on the
    target cells, so that there is nothing to search and the variable passes
    through.

    The second return value says whether the variable was already on the
    target support, and if so what cell method its own stream declared — the
    only thing the caller cannot re-derive from the columns themselves. The
    third is the column's support record, already written into its attrs and
    returned so the caller can warn about all columns at once.
    """
    if stream[variable].dims != (TIME_COORD,):
        # Named rather than broadcast against: without this the cell
        # boundaries, or any other array carrying a second dimension, reach
        # the weighting as a shape mismatch and surface as an untyped
        # `ValueError: operands could not be broadcast together`, which says
        # nothing about which stream or which variable was at fault.
        raise TsaraAlignError(
            f"'{variable}' on stream '{instrument}' has dimensions "
            f"{stream[variable].dims}, and only one value per cell can be binned. "
            f"A variable over ('{TIME_COORD}',) is what this operation averages; "
            "anything else describes the cells rather than varying over them."
        )
    values = np.asarray(stream[variable].values, dtype=np.float64)
    # Everything the input stream declared travels with the column, plus where it came from.
    attrs: dict[str, object] = dict(stream[variable].attrs)
    attrs[INSTRUMENT_ATTR] = instrument
    # A bool, an int, or a string after a netCDF round trip; the one predicate
    # that reads all three lives in `core.naming` and is shared with the
    # auxiliary interpolator, which has to make the same decision.
    circular = is_circular(attrs)
    if circular:
        _warn_of_a_dropped_direction_sigma(stream, variable, instrument)
    # Already on the target cells (the caller found nothing to search)? Then
    # pass through rather than average onto itself.
    attrs[BINNED_ATTR] = int(pairs is not None)

    if pairs is None or ratios is None:
        # The companions below must be exactly what the binned path would
        # produce for one reading covering its target, so that a product's
        # columns and their meaning do not depend on whether a stream happened
        # to share the target's cells. A masked value contributes nothing,
        # there as here: no count, no coverage, no borrowed share, no quality
        # number.
        present = np.isfinite(values)
        columns: dict[str, tuple[np.ndarray, dict[str, object]]] = {column: (values, attrs)}
        columns[n_readings_name(column)] = (
            present.astype(np.int64),
            _count_attrs(column, native=True),
        )
        columns[coverage_name(column)] = (
            present.astype(np.float64),
            _coverage_attrs(column, native=True),
        )
        # A reading on its own cell borrows nothing from beyond it.
        columns[borrowed_name(column)] = (
            np.where(present, 0.0, np.nan),
            _borrowed_attrs(column, native=True),
        )
        n_present = int(np.count_nonzero(present))
        change = _SupportChange("passthrough", 1.0, 0.0, n_present, n_present)
        _record(attrs, change)
        if circular:
            # One reading agrees with itself: R is 1 and the dispersion 0,
            # which is what vector-averaging that single reading returns.
            # No sigma, matching the binned path (§11.5).
            columns[f"{column}{RESULTANT_LENGTH_SUFFIX}"] = (
                np.where(present, 1.0, np.nan),
                _resultant_attrs(column),
            )
            columns[f"{column}{DISPERSION_SUFFIX}"] = (
                np.where(present, 0.0, np.nan),
                _dispersion_attrs(column, str(attrs.get("units", "degrees"))),
            )
            return columns, stream[variable].attrs.get(CELL_METHODS_ATTR), change
        # A reading on its own cell needs no propagation: each sigma is used as
        # it stands, moved only if it was quoted at another interval.
        for component, sigma_name in (("random", sigma_rand_name), ("systematic", sigma_sys_name)):
            resolved = _sigma_on_cells(
                stream, variable, sigma_name(variable), median_width_s(readings), propagation_form
            )
            if resolved is None:
                continue
            sigma_values, provenance = resolved
            columns[sigma_name(column)] = (
                sigma_values,
                _sigma_attrs(stream, variable, component, provenance, form="native"),
            )
        return columns, stream[variable].attrs.get(CELL_METHODS_ATTR), change

    # Not on the target cells: average it, as a direction or as a number.
    if circular:
        columns, borrowed, weight = _bin_circular(
            stream, variable, column, readings, target, attrs, pairs
        )
    else:
        columns, borrowed, weight = _bin_scalar(
            stream, variable, column, readings, target, values, attrs, propagation_form, pairs
        )
    # The third qualifier, per cell, and the four-number record, per column.
    columns[borrowed_name(column)] = (borrowed, _borrowed_attrs(column, native=False))
    change = _summarize(pairs, ratios, weight, borrowed, columns[column][0])
    _record(attrs, change)
    return columns, _BINNED_HERE, change


def _warn_of_a_dropped_direction_sigma(stream: xr.Dataset, variable: str, instrument: str) -> None:
    """Say so when a direction carries a declared sigma that no join propagates.

    Ingestion resolves and stores a declared uncertainty on a circular
    variable exactly as on any other; both join paths then drop it, because a
    direction's companions are the mean resultant length and the exact
    dispersion, which describe the spread of its *readings* and are not the
    instrument's precision (§11.5). Until the Phase-4.6 walkthrough that
    happened without a word. Propagating a direction sigma properly is a
    small-angle, von-Mises question that no real manifest has yet asked; what
    is owed meanwhile is the sentence.
    """
    declared = [
        name
        for name in (sigma_rand_name(variable), sigma_sys_name(variable))
        if name in stream.data_vars
    ]
    if not declared:
        return
    logger.warning(
        "'%s' on '%s' is circular and carries %s, which no join propagates: a direction's "
        "companions are its mean resultant length and exact dispersion, which describe the "
        "spread of its readings, not the instrument's precision (METHODS §11.5). The "
        "declared figure stays on the stream and is absent from this product.",
        variable,
        instrument,
        " and ".join(declared),
    )


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
        # This component was never declared or reported for the variable.
        return None
    values = np.asarray(stream[sigma_variable].values, dtype=np.float64)
    quoted = stream[variable].attrs.get("uncertainty_at_width")
    if quoted is None:
        # No quoted interval: the sigma already describes the stream's own cells.
        return values, "unchanged"
    # A figure quoted at another interval: moving it needs a decorrelation
    # timescale, and `sigma_at_support` refuses (returns "unscaled") without one.
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


def _bin_circular(
    stream: xr.Dataset,
    variable: str,
    column: str,
    readings: CellBounds,
    target: CellBounds,
    attrs: dict[str, object],
    pairs: OverlapPairs,
) -> tuple[dict[str, tuple[np.ndarray, dict[str, object]]], np.ndarray, np.ndarray]:
    """Vector-average an angular variable and carry its quality numbers.

    An angle has no meaningful arithmetic mean, so it gets none. What it gets
    instead is the mean resultant length, which is the honest statement of
    how well determined the direction is, and the exact circular standard
    deviation derived from it (§11.5). Returns the columns, the per-cell
    borrowed share, and the per-pair weights the caller's record needs.
    """
    angles = np.asarray(stream[variable].values, dtype=np.float64)
    # The same overlaps as a scalar, different arithmetic on the pairs.
    result = bin_circular_onto_cells(readings, angles, target, pairs=pairs)
    units = str(attrs.get("units", "degrees"))
    columns = {
        column: (result.mean_deg, attrs),
        n_readings_name(column): (result.n_readings, _count_attrs(column, native=False)),
        coverage_name(column): (result.coverage, _coverage_attrs(column, native=False)),
        f"{column}{RESULTANT_LENGTH_SUFFIX}": (
            result.resultant_length,
            _resultant_attrs(column),
        ),
        f"{column}{DISPERSION_SUFFIX}": (
            result.dispersion_deg,
            _dispersion_attrs(column, units),
        ),
    }
    return columns, result.borrowed, contributing_weights(pairs, angles)


def _bin_scalar(
    stream: xr.Dataset,
    variable: str,
    column: str,
    readings: CellBounds,
    target: CellBounds,
    values: np.ndarray,
    attrs: dict[str, object],
    propagation_form: PropagationForm,
    pairs: OverlapPairs,
) -> tuple[dict[str, tuple[np.ndarray, dict[str, object]]], np.ndarray, np.ndarray]:
    """Overlap-weighted mean of one scalar variable, with its uncertainty.

    The mean, the count, the coverage and the borrowed share are
    :func:`~tsara.core.support.bin_onto_cells`, called with the overlaps the
    caller found once for the whole instrument; this function used to carry
    a second spelling of that arithmetic, and two implementations of one
    idea is the shape the Phase-4 reframe rejected. What is added here is the
    uncertainty, propagated through exactly the weights that formed the value.
    Returns the columns, the per-cell borrowed share, and the per-pair weights
    the caller's record needs.
    """
    n_target = len(target)
    binned = bin_onto_cells(readings, values, target, pairs=pairs)
    columns: dict[str, tuple[np.ndarray, dict[str, object]]] = {
        column: (binned.values, attrs),
        n_readings_name(column): (binned.n_readings, _count_attrs(column, native=False)),
        coverage_name(column): (binned.coverage, _coverage_attrs(column, native=False)),
    }
    # Uncertainty, propagated through exactly the weights that formed the value
    # (docs/METHODS.md §3): each pair's overlap, or zero where the reading is
    # masked, from the one shared definition. The declared timescale, if any,
    # says how correlated the random errors of neighbouring readings are;
    # without one they are independent, which is what declaring a component
    # random means.
    weight = contributing_weights(pairs, values)
    tau = stream[variable].attrs.get("decorrelation_timescale")
    tau_s = float(pd.Timedelta(tau).total_seconds()) if tau is not None else None
    # How far apart readings are (for the correlated-error forms), and how wide
    # each is (for moving a sigma quoted at another interval onto the cells).
    spacing = cadence_s(readings)
    cell_width = median_width_s(readings)
    for component, sigma_name in (("random", sigma_rand_name), ("systematic", sigma_sys_name)):
        resolved = _sigma_on_cells(
            stream, variable, sigma_name(variable), cell_width, propagation_form
        )
        if resolved is None:
            continue
        sigma_values, provenance = resolved
        # Each pair's sigma, aligned with `weight` and `pairs.target_index`.
        long_form = sigma_values[pairs.reading_index]
        if component == "random":
            # sqrt(sum w^2 sigma^2) / sum w, inflated for correlation when tau is declared.
            result = propagate_random_binned(
                long_form,
                weight,
                pairs.target_index,
                n_target,
                spacing_s=spacing,
                # Long-form sample times, which only the pairwise form uses.
                # Built here rather than inside the propagation module because
                # this is where the readings are.
                times_s=readings.midpoint_ns[pairs.reading_index].astype(np.float64) / NS_PER_S,
                tau_s=tau_s,
                form=propagation_form,
            )
        else:
            # sum w sigma / sum w: shared errors do not average down.
            result = propagate_systematic_binned(long_form, weight, pairs.target_index, n_target)
        columns[sigma_name(column)] = (
            result.sigma,
            _sigma_attrs(stream, variable, component, provenance, form=result.form),
        )
    return columns, binned.borrowed, weight


def _summarize(
    pairs: OverlapPairs,
    ratios: np.ndarray,
    weight: np.ndarray,
    borrowed: np.ndarray,
    values: np.ndarray,
) -> _SupportChange:
    """Reduce a binned column's per-pair and per-cell numbers to its record (§11.2.4).

    Over the pairs that formed a value, not over the cells: a masked reading
    took part in nothing, so it neither widens the worst ratio nor counts as
    a reading behind the column.
    """
    contributing = weight > 0
    ratio_max = float(ratios[contributing].max()) if contributing.any() else float("nan")
    # The column's share: every contributing pair's borrowed time over every
    # contributing pair's time -- the per-cell shares weighted by how much of
    # each cell was measured, so a sliver cell cannot dominate it.
    weight_sum = np.bincount(pairs.target_index, weights=weight, minlength=pairs.n_target)
    covered = weight_sum > 0
    if covered.any():
        share = float(np.sum(borrowed[covered] * weight_sum[covered]) / weight_sum[covered].sum())
        borrowed_max = float(np.max(borrowed[covered]))
    else:
        share = borrowed_max = float("nan")
    rows = int(np.count_nonzero(np.isfinite(values)))
    readings = int(np.unique(pairs.reading_index[contributing]).size)
    return _SupportChange(_label(ratio_max, borrowed_max), ratio_max, share, rows, readings)


def _label(ratio_max: float, borrowed_max: float) -> SupportTransform:
    """Name the worst thing a join did to a column, from its two numbers.

    Tolerance-free by construction: a reading exactly as wide as its cell
    divides to exactly 1.0 in float64 (the widths are integers), and a
    reading wholly inside its cell borrows exactly 0.0
    (:func:`~tsara.core.support.borrowed_share`).
    """
    if not np.isfinite(ratio_max):
        return "averaged"  # nothing contributed, so nothing was borrowed either
    if ratio_max >= COPY_RATIO:
        return "copied"
    if ratio_max > 1.0:
        return "narrowed"
    if borrowed_max > 0.0:
        return "straddled"
    return "averaged"


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
    * A coverage fraction, a borrowed share, a resultant length and a
      dispersion are properties of the cell rather than statistics of the
      data inside it, so they get no cell method — the same reason the sigma
      companions get none (§10.2).
    """
    for column in columns:
        declared = native.get(column, _BINNED_HERE)
        if declared != _BINNED_HERE:
            # Passed through: restore what the stream itself declared, or nothing.
            if declared is None:
                dataset[column].attrs.pop(CELL_METHODS_ATTR, None)
            else:
                dataset[column].attrs[CELL_METHODS_ATTR] = declared
        # A contributing count is a sum over the cell.
        dataset[n_readings_name(column)].attrs[CELL_METHODS_ATTR] = f"{TIME_COORD}: sum"
        # Properties of the cell, not statistics of the data in it: no method.
        for name in (
            coverage_name(column),
            borrowed_name(column),
            f"{column}{RESULTANT_LENGTH_SUFFIX}",
            f"{column}{DISPERSION_SUFFIX}",
        ):
            if name in dataset.data_vars:
                dataset[name].attrs.pop(CELL_METHODS_ATTR, None)


def _warn_about_support_changes(changes: Mapping[str, _SupportChange], overlapping: bool) -> None:
    """One warning per join, naming the columns whose values rest on borrowed air.

    Two conditions, both tolerance-free (§11.2.4). A column is *narrowed*, or
    *copied* under ``finer_support='allow'``, when some reading that formed a
    value is wider than the cell it filled; the ratio says by how much, and
    cadence jitter is listed with its 1.02 rather than hidden behind a
    threshold chosen by taste. A column is *shared* when its rows outnumber
    the distinct readings behind them, so a fit or receptor model treating
    rows as independent counts some reading more than once -- asked only of
    disjoint targets, since overlapping ones share readings by construction.
    Blending itself (a reading straddling a boundary, neither wider than its
    cell nor repeated) is recorded in the borrowed share and not warned about:
    measured, no value of that share separates a half-phase blend from
    ordinary jitter.
    """

    def shared(change: _SupportChange) -> bool:
        return not overlapping and change.rows > change.readings

    flagged = [
        (column, change)
        for column, change in changes.items()
        if change.width_ratio_max > 1.0 or shared(change)
    ]
    if not flagged:
        return

    def describe(column: str, change: _SupportChange) -> str:
        parts = []
        if change.width_ratio_max > 1.0:
            parts.append(f"{change.transform}, readings up to {change.width_ratio_max:.3g}x a cell")
        if shared(change):
            parts.append(f"{change.rows} rows from {change.readings} readings")
        parts.append(f"borrowed share {change.borrowed_share:.2f}")
        return f"'{column}' ({'; '.join(parts)})"

    # At most eight named: a canister's fifty VOCs share one sampling pattern
    # and would otherwise repeat one sentence fifty times.
    listed = ", ".join(describe(column, change) for column, change in flagged[:8])
    logger.warning(
        "%d column(s) of this join rest on air the readings did not measure over the "
        "cells reported (METHODS §11.2.4): %s%s. A reading wider than the cell it fills "
        "has its value stand for a shorter interval than it measured; rows outnumbering "
        "readings means some reading sits in more than one row, which a fit or receptor "
        "model treating rows as independent counts more than once. Each column records "
        "tsara_support_transform, tsara_width_ratio_max, tsara_borrowed_share and "
        "tsara_n_readings, and carries the share borrowed per cell in borrowed_<name>.",
        len(flagged),
        listed,
        " ..." if len(flagged) > 8 else "",
    )


# ---------------------------------------------------------------------------
# What each column of the product says about itself
# ---------------------------------------------------------------------------


def _record(attrs: dict[str, object], change: _SupportChange) -> None:
    """Write a column's support record into its attrs."""
    attrs[TRANSFORM_ATTR] = change.transform
    attrs[WIDTH_RATIO_ATTR] = change.width_ratio_max
    attrs[BORROWED_ATTR] = change.borrowed_share
    attrs[READINGS_ATTR] = change.readings


def _count_attrs(column: str, *, native: bool) -> dict[str, object]:
    """Return attrs for a contributing-sample count."""
    return {
        "description": (
            f"Readings of {column} contributing to each target cell."
            + (
                " Already on this support: one where a value is present, zero where masked."
                if native
                else ""
            )
        ),
        "units": "1",
    }


def _coverage_attrs(column: str, *, native: bool) -> dict[str, object]:
    """Return attrs for a coverage fraction."""
    return {
        "description": (
            f"Fraction of each target cell covered by contributing {column} data."
            + (
                " Already on this support: one where a value is present, zero where masked."
                if native
                else ""
            )
        ),
        "units": "1",
    }


def _borrowed_attrs(column: str, *, native: bool) -> dict[str, object]:
    """Return attrs for a per-cell borrowed share."""
    return {
        "description": (
            f"Share of each cell's {column} value resting on air outside the cell: the "
            "coverage-weighted fraction of each contributing reading's own cell lying beyond "
            "the target (METHODS §11.2.4). Exactly zero for pure averaging."
            + (
                " Already on this support: zero where a value is present, nothing where masked."
                if native
                else ""
            )
        ),
        "units": "1",
    }


def _resultant_attrs(column: str) -> dict[str, object]:
    """Return attrs for a mean resultant length."""
    return {
        "units": "1",
        "description": (
            f"Mean resultant length of {column} per cell: 1 when every contributing "
            "reading agreed, 0 when they cancelled and there is no direction. Biased "
            "high when few readings contribute; read it with the count (METHODS §11.5)."
        ),
    }


def _dispersion_attrs(column: str, units: str) -> dict[str, object]:
    """Return attrs for an exact circular standard deviation."""
    return {
        "units": units,
        "description": (
            f"Exact circular standard deviation of {column} per cell, "
            "unbounded by construction (METHODS §11.5)."
        ),
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
    # attribute vocabulary by reading string constants out of the code, so a
    # prefix plus an f-string would enter that vocabulary as a fragment that
    # matches nothing in the methods document and can never be documented.
    provenance_key = (
        "uncertainty_provenance_random"
        if component == "random"
        else "uncertainty_provenance_systematic"
    )
    return {
        "units": stream[variable].attrs.get("units", ""),
        "description": f"{component.capitalize()} 1-sigma for {variable} on the target cells.",
        "uncertainty_component": component,
        "uncertainty_provenance": stream[variable].attrs.get(provenance_key, "unknown"),
        SIGMA_AT_SUPPORT_ATTR: at_support,
        PROPAGATION_FORM_ATTR: form,
    }
