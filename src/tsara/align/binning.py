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
* **how many** readings contributed and **how much** of the target cell
  they covered — the two numbers that separate a well-determined value from
  a number that merely exists;
* everything the input stream declared about itself, plus where it came from.

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
    DISPERSION_SUFFIX,
    RESULTANT_LENGTH_SUFFIX,
    TIME_BOUNDS_VAR,
    TIME_COORD,
    coverage_name,
    is_companion_name,
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
INSTRUMENT_ATTR = "tsara_instrument"
BINNED_ATTR = "tsara_binned"
PROPAGATION_FORM_ATTR = "tsara_propagation_form"
SIGMA_AT_SUPPORT_ATTR = "tsara_sigma_at_support"

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
        # Explicit (instrument, variable): nothing to search for, only to check.
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
    # A bare name: find every stream that carries it, then insist on exactly one.
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
    # Bounds are normally a coordinate; a stream may also carry them as a data variable.
    if TIME_BOUNDS_VAR not in stream.coords and TIME_BOUNDS_VAR not in stream.data_vars:
        raise TsaraAlignError(
            f"Stream '{instrument}' carries no '{TIME_BOUNDS_VAR}', so there is no "
            "interval to bin over. Streams gain cells at ingestion (METHODS §10); "
            "a bundle written before format 2 must be reloaded to acquire them."
        )
    # (time, 2) datetime64 edges -> int64 nanoseconds, the unit every overlap is measured in.
    bounds = np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]").astype(np.int64)
    if bounds.size == 0:
        raise TsaraAlignError(f"Stream '{instrument}' has no cells to bin from.")
    # Copies, so the CellBounds owns contiguous arrays rather than views into the dataset.
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
        # Start-to-start steps; zero steps (two cells starting together) say nothing about spacing.
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
    Companion columns are excluded from the default because they are not
    variables in their own right: each travels automatically with the value it
    describes, and selecting one directly would produce a column with no parent
    and no meaning (:func:`~tsara.core.naming.is_companion_name`). That covers
    the uncertainty components a stream arrives with *and* the counts, coverage
    fractions and angular quality numbers this module itself adds, so a joined
    product can be joined again — which is what Phase 5 does with a baseline —
    without growing a ``coverage_coverage_ch4`` on every pass.

    The cell boundaries are excluded too. They are metadata describing the
    rows rather than a variable over them, and they are *usually* a coordinate
    and therefore invisible here; a stream that carries them as a data variable
    is a shape :func:`stream_cells` deliberately accepts, so this must accept
    it as well rather than trying to average a set of boundaries.
    """
    if variables is not None:
        # An explicit selection: resolve each reference, refusing ambiguity.
        return [resolve_variable(streams, reference) for reference in variables]
    # The default: every value column in every stream, in stream order, skipping
    # the companions (sigmas, counts, coverage, angular quality) and the bounds.
    return [
        (instrument, str(name))
        for instrument, stream in streams.items()
        for name in stream.data_vars
        if not is_companion_name(str(name)) and str(name) != TIME_BOUNDS_VAR
    ]


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


def measure_replication(readings: CellBounds, target: CellBounds) -> tuple[bool, float, float]:
    """Say whether any one reading would be spread over two target cells' worth of time.

    The test for the one direction this operation must not run in. Averaging a
    fast stream onto slow cells discards resolution the slow instrument never
    had, which is honest; evaluating a slow value on fast cells hands back rows
    the instrument never reported, which is the interpolation rule (§1.2)
    restated for a step function.

    **The rule: a reading is replicated when the time it shares with the
    target cells adds up to at least twice the width of the widest target cell
    it touches.** Measured on the overlaps, as the rule it replaces was, and
    for the same reason — a comparison of median widths needs a tolerance,
    because real analyzers disagree about their own nominal rate (1.023 s
    against a nominal 1 s in the 2026 archive), and this needs none:

    * 60 s onto 1 s is sixty cells' worth, refused;
    * 2 s onto 1 s is two cells' worth at *any* phase, refused;
    * 1.023 s onto 1 s is a hair over one, allowed;
    * equal widths half a period apart are one cell's worth, allowed.

    The rule this replaced counted target cells lying *wholly* inside one
    reading, refusing at two. It agreed on every case above except one: a
    perfectly regular 2 s record offset from a 1 s grid by anything but zero
    wholly contains only one target cell, so it passed, and each of its
    readings fed two or three rows (§11.2.1). What remains below the line —
    a reading partly shared between neighbouring rows — is not refused but
    counted, by :func:`readings_behind`.

    Parameters
    ----------
    readings : CellBounds
        Cells being averaged.
    target : CellBounds
        Cells to average onto.

    Returns
    -------
    tuple of (bool, float, float)
        Whether any reading is replicated; the largest multiple of a
        touched target cell's width that any one reading covers; and that
        reading's total covered time in seconds, half of which is the
        widest target cell that would *not* replicate it.
    """
    # Long form: one entry per overlapping (reading, target cell) pair.
    pairs = overlap_pairs(readings, target)
    # The width of the target cell in each pair.
    target_width = target.width_ns[pairs.target_index]
    # A zero-width target cell overlaps nothing by a positive amount and would
    # otherwise make a degenerate grid look infinitely replicated.
    touching = (pairs.overlap_ns > 0) & (target_width > 0)
    if not touching.any():
        return False, 0.0, 0.0
    # Which reading each touching pair belongs to.
    owner = pairs.reading_index[touching]
    # Integer nanoseconds summed in float64 are exact below 2**53 ns, about
    # 104 days of overlap for a single reading, so the comparison with
    # twice a width is exact for any cell this operation will meet.
    # Total time each reading shares with the target cells.
    covered = np.bincount(owner, weights=pairs.overlap_ns[touching], minlength=len(readings))
    # The widest target cell each reading touches.
    widest = np.zeros(len(readings), dtype=np.int64)
    np.maximum.at(widest, owner, target_width[touching])
    # Readings that touch at least one target cell.
    used = widest > 0
    # "How many target cells' worth of time" each reading covers; the worst
    # one is reported, so the refusal can name a width that would work.
    multiples = covered[used] / widest[used]
    worst = int(np.argmax(multiples))
    # The rule itself: two cells' worth or more is replication.
    replicated = bool(np.any(covered[used] >= 2.0 * widest[used]))
    return replicated, float(multiples[worst]), float(covered[used][worst]) / NS_PER_S


def touched_readings(readings: CellBounds, target: CellBounds) -> np.ndarray:
    """Return which readings overlap at least one target cell by a positive amount.

    The membership half of :func:`readings_behind`, separated so that a caller
    counting readings for many variables of one instrument — a canister
    carrying fifty VOCs on a campaign grid — finds the overlaps once rather
    than once per variable.

    Parameters
    ----------
    readings : CellBounds
        The instrument's cells.
    target : CellBounds
        The cells whose readings are being counted.

    Returns
    -------
    numpy.ndarray
        One boolean per reading.
    """
    links = overlap_pairs(readings, target)
    touched = np.zeros(len(readings), dtype=bool)
    # A pair touching only at a boundary (overlap 0) does not count, as in binning.
    touched[links.reading_index[links.overlap_ns > 0]] = True
    return touched


def readings_behind(
    stream: xr.Dataset, variable: str, readings: CellBounds, target: CellBounds
) -> int:
    """Return how many distinct readings of a variable the target cells draw on.

    Every finite reading overlapping at least one target cell by a positive
    amount — the same membership rule the binner uses to form a value, so the
    count describes the numbers actually reported. Fewer readings than occupied
    target cells means some reading appears in more than one row, which a fit
    or receptor model treating rows as independent would count more than once
    (§11.4.1, §11.7).

    Parameters
    ----------
    stream : xarray.Dataset
        The stream holding the variable.
    variable : str
        The variable's name in that stream.
    readings : CellBounds
        The stream's cells.
    target : CellBounds
        The cells whose readings are being counted.

    Returns
    -------
    int
        Distinct finite readings behind the target cells.
    """
    # A reading counts when it both overlaps a target cell and holds a value.
    finite = np.isfinite(np.asarray(stream[variable].values, dtype=np.float64))
    return int(np.count_nonzero(touched_readings(readings, target) & finite))


def _refuse_upsampling(readings: CellBounds, target: CellBounds, instrument: str) -> None:
    """Raise if binning onto ``target`` would replicate ``readings``.

    Both of this phase's products already prevent this by choosing their
    target: :func:`~tsara.align.pairing.pair_species` pairs on the
    wider-supported member, and
    :func:`~tsara.align.grid.build_output_grid` checks its period with this
    same rule before building anything. The primitive they share has to refuse
    it too, because it is public and is the documented way to build a
    receptor-model matrix from a chosen set of columns — a caller supplying
    their own target cells would otherwise get the one thing this package
    promises not to do, with ``coverage`` reporting 1.0 and nothing else to
    notice it by.
    """
    replicated, multiple, covered_s = measure_replication(readings, target)
    if not replicated:
        return
    raise TsaraAlignError(
        f"Stream '{instrument}' has {median_width_s(readings):.6g} s cells and the target "
        f"cells are {median_width_s(target):.6g} s, so one of its measurements would cover "
        f"{multiple:.3g} target cells' worth of time on its own. Evaluating it on a shorter "
        "support is resolution the instrument never had (METHODS §11.2.1), and the "
        "replicated rows would enter a fit as independent measurements. Bin onto cells "
        f"wider than {covered_s / 2:.6g} s, pair on the wider-supported stream, or — for a "
        "smooth non-gas field — interpolate it with tsara.align.auxiliary (§11.6)."
    )


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
        One column per selected variable, plus ``n_readings_<name>`` and
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
    # 1. Validate what was asked for, and resolve it to (instrument, variable) pairs.
    if not streams:
        raise TsaraAlignError("No streams to bin.")
    if len(target) == 0:
        raise TsaraAlignError("No target cells to bin onto.")
    selection = select_variables(streams, variables)
    if not selection:
        raise TsaraAlignError(
            "No variables selected. Every stream holds only companion columns — "
            "uncertainties, counts, coverage fractions, angular quality numbers — "
            "which travel with the values they describe rather than being binned "
            "on their own."
        )
    # 2. Decide each output column's name (suffixed only on a collision).
    names = _output_names(selection)
    # 3. Refuse the direction that would replicate readings (METHODS §11.2.1).
    # Once per instrument rather than once per variable: the question is about
    # cells, and a spectral stream can carry a thousand columns on one clock.
    for instrument in dict.fromkeys(name for name, _ in selection):
        _refuse_upsampling(stream_cells(streams[instrument], instrument), target, instrument)
    # 4. Put each variable on the target cells: its value, its companions
    # (count, coverage, sigmas or angular quality), and its attributes.
    data_vars: dict[str, tuple[str, np.ndarray, dict[str, object]]] = {}
    # Per column: the cell method a passed-through variable declared, or a
    # sentinel saying this call averaged it. Needed to fix cell_methods below.
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

    # 5. Assemble the product: one row per target cell, `time` at each midpoint.
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={TIME_COORD: np.asarray(target.midpoint_ns, dtype="datetime64[ns]")},
        attrs={
            "tsara_version": __version__,
            "tsara_stage": "binned",
            PROPAGATION_FORM_ATTR: propagation_form,
        },
    )
    # 6. Describe the cells (CF `time_bnds`), then correct the blanket
    # `time: mean` that attaching them stamps on every column.
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
            f"{column}{RESULTANT_LENGTH_SUFFIX}",
            f"{column}{DISPERSION_SUFFIX}",
        ):
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
    # The variable's own cells, which its values describe.
    readings = stream_cells(stream, instrument)
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
    # `circular` may arrive as a bool, an int or a string (after a netCDF round
    # trip), so it is read by its spelling rather than by truthiness.
    circular = str(attrs.get("circular", 0)) not in ("0", "False", "None", "")
    # Already on the target cells? Then pass through rather than average onto itself.
    already_here = _same_cells(readings, target)
    attrs[BINNED_ATTR] = int(not already_here)

    if already_here:
        # The companions below must be exactly what the binned path would
        # produce for one reading covering its target, so that a product's
        # columns and their meaning do not depend on whether a stream happened
        # to share the target's cells. A masked value contributes nothing,
        # there as here: no count, no coverage, no quality number.
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
            return columns, stream[variable].attrs.get(CELL_METHODS_ATTR)
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
        return columns, stream[variable].attrs.get(CELL_METHODS_ATTR)

    # Not on the target cells: average it, as a direction or as a number.
    if circular:
        return _bin_circular(stream, variable, column, readings, target, attrs), "__binned__"
    return _bin_scalar(
        stream, variable, column, readings, target, values, attrs, propagation_form
    ), "__binned__"


def _bin_circular(
    stream: xr.Dataset,
    variable: str,
    column: str,
    readings: CellBounds,
    target: CellBounds,
    attrs: dict[str, object],
) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    """Vector-average an angular variable and carry its quality numbers.

    An angle has no meaningful arithmetic mean, so it gets none. What it gets
    instead is the mean resultant length, which is the honest statement of
    how well determined the direction is, and the exact circular standard
    deviation derived from it (§11.5).
    """
    # Same overlap search as a scalar, different arithmetic on the pairs.
    result = bin_circular_onto_cells(
        readings, np.asarray(stream[variable].values, dtype=np.float64), target
    )
    units = str(attrs.get("units", "degrees"))
    return {
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


def _bin_scalar(
    stream: xr.Dataset,
    variable: str,
    column: str,
    readings: CellBounds,
    target: CellBounds,
    values: np.ndarray,
    attrs: dict[str, object],
    propagation_form: PropagationForm,
) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    """Overlap-weighted mean of one scalar variable, with its uncertainty."""
    # Long form: one entry per overlapping (reading, target cell) pair,
    # with the overlap in nanoseconds. Everything below aggregates these pairs.
    pairs = overlap_pairs(readings, target)
    n_target = len(target)
    # The reading's value in each pair.
    paired = values[pairs.reading_index]
    # A pair contributes when its reading holds a value and the overlap is positive.
    contributes = np.isfinite(paired) & (pairs.overlap_ns > 0)
    # Its weight is the overlap; a pair that does not contribute weighs nothing.
    weight = np.where(contributes, pairs.overlap_ns, 0.0).astype(np.float64)
    # Per target cell: the total contributing overlap (the denominator) ...
    weight_sum = np.bincount(pairs.target_index, weights=weight, minlength=n_target)
    # ... and the overlap-weighted sum of values (the numerator). A masked value
    # is zeroed before multiplying, because 0 * nan is nan, not 0.
    value_sum = np.bincount(
        pairs.target_index,
        weights=weight * np.where(contributes, paired, 0.0),
        minlength=n_target,
    )
    # The weighted mean where anything contributed; nan everywhere else, never
    # a value borrowed from a neighbour.
    binned = np.full(n_target, np.nan, dtype=np.float64)
    filled = weight_sum > 0
    binned[filled] = value_sum[filled] / weight_sum[filled]
    # n_readings: how many readings contributed to each target cell.
    counts = np.bincount(
        pairs.target_index, weights=contributes.astype(np.float64), minlength=n_target
    ).astype(np.int64)
    # Coverage can exceed 1 slightly, and is left alone when it does.
    # Fixed-width cells centred on jittered timestamps overlap each other, so
    # their overlaps with one target cell can sum past its width (§10.2, where
    # the effect is documented as benign: the value is a weighted MEAN, so the
    # weights normalize). Clipping would hide a real property of the input
    # record behind a tidier number.
    # coverage: the contributing overlap as a fraction of the target cell's width.
    width = target.width_ns.astype(np.float64)
    coverage = np.zeros(n_target, dtype=np.float64)
    wide = width > 0
    coverage[wide] = weight_sum[wide] / width[wide]

    columns: dict[str, tuple[np.ndarray, dict[str, object]]] = {
        column: (binned, attrs),
        n_readings_name(column): (counts, _count_attrs(column, native=False)),
        coverage_name(column): (coverage, _coverage_attrs(column, native=False)),
    }
    # Uncertainty, propagated through exactly the weights that formed the value
    # (docs/METHODS.md §3). The declared timescale, if any, says how correlated
    # the random errors of neighbouring readings are; without one they are
    # independent, which is what declaring a component random means.
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
    return columns


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
