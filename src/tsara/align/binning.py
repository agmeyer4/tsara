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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

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
    borrowed_name,
    coverage_name,
    is_circular,
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
    "COPY_RATIO",
    "FinerSupport",
    "SupportTransform",
    "TsaraAlignError",
    "VariableRef",
    "bin_streams_onto_cells",
    "resolve_variable",
    "select_variables",
    "targets_overlap",
]

#: How a caller names a variable: by its canonical name, or by the instrument
#: that measured it when more than one did.
#:
#: A real type alias, not a string that looks like one. It was written as
#: ``"str | tuple[str, str]"`` — an ordinary ``str`` object — and exported, so
#: it described the convention in prose while checking nothing and appearing in
#: no signature. Every function here and in the modules layered on top now
#: annotates with it, which is what makes the name worth having.
VariableRef: TypeAlias = str | tuple[str, str]

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
READINGS_ATTR = "tsara_readings"

#: Reading width over target width at which a join is a *copy*: one reading
#: filling two cells' worth of rows (§11.2.4). A named constant rather than a
#: knob. It is the line at which a reading holds less than one row's worth of
#: information, and it needs no tolerance: cadence jitter on the archive tops
#: out at 1.024 against a refusal at 2.
COPY_RATIO = 2.0

#: What a join may do with a reading at or beyond :data:`COPY_RATIO`.
#: ``refuse`` is the default and the interpolation rule's guarantee (§1.2);
#: ``allow`` copies the reading across rows, labels every affected column
#: ``copied``, records how much of each value was borrowed, and warns.
FinerSupport = Literal["refuse", "allow"]

#: The vocabulary of ``tsara_support_transform``, one word per column naming
#: the worst thing the join did to any reading behind it. Derived from two
#: numbers without a tolerance: ``passthrough`` when the stream's cells are
#: the target; ``averaged`` when every contributing reading sat wholly inside
#: its cell (borrowed share exactly 0); ``shared`` when some reading straddled
#: a cell boundary but none was wider than a cell it filled; ``narrowed`` when
#: one was, by less than :data:`COPY_RATIO`; ``copied`` at or beyond it.
SupportTransform = Literal["passthrough", "averaged", "shared", "narrowed", "copied"]

#: What ``tsara_stage`` says on a dataset this package built by *joining*
#: measurements, as opposed to one it read from an archive (``ingest``) or
#: manufactured (``synthetic``).
#:
#: The list is deliberately of the three join stages rather than an allow-list
#: of the two source stages. A later phase may well produce a native-rate
#: product of its own — Phase 5's baselines are computed cell by cell over each
#: stream's own cells, so every baseline value still stands for one reading —
#: and binning that is honest. What is refused is joining something whose rows
#: are already the *output* of a join (§11.2.3).
JOINED_STAGES = frozenset({"binned", "paired", "gridded"})

#: Sentinel standing where a variable's own ``cell_methods`` would be, saying
#: that this call averaged it rather than passing it through.
#:
#: ``None`` cannot do the job: a stream that declares no cell method at all is
#: a real case, and must stay distinguishable from one this stage averaged.
#: Named rather than repeated as a literal in four places, where a typo in one
#: of them would silently restore a cell method to a variable that was binned.
_BINNED_HERE = "__binned__"


class TsaraAlignError(TsaraError):
    """Raised when variables cannot be put on a common support as asked.

    Its own type because the failures are about *combining* streams rather
    than about reading them: a variable no stream declares, a name two
    streams both declare where one was expected, or a stream carrying no
    cells to bin from.
    """


def resolve_variable(streams: Mapping[str, xr.Dataset], reference: VariableRef) -> tuple[str, str]:
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
    variables: Sequence[VariableRef] | None = None,
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

    Raises
    ------
    TsaraAlignError
        If a reference names nothing or names it ambiguously, or if a selected
        dataset is a product of an earlier join (:func:`_refuse_a_joined_product`).

    Notes
    -----
    Companion columns are excluded from the default because they are not
    variables in their own right: each travels automatically with the value it
    describes, and selecting one directly would produce a column with no parent
    and no meaning (:func:`~tsara.core.naming.is_companion_name`). The
    everyday case is the uncertainty components, which arrive on every stream
    that declares a budget and must not be averaged as though they were data;
    the counts, coverage fractions and angular quality numbers this module adds
    are covered by the same predicate, so nothing that is handed a dataset of
    TSARA's own making can bin a ``coverage_ch4`` into a
    ``coverage_coverage_ch4``.

    The cell boundaries are excluded too. They are metadata describing the
    rows rather than a variable over them, and they are *usually* a coordinate
    and therefore invisible here; a stream that carries them as a data variable
    is a shape :func:`stream_cells` deliberately accepts, so this must accept
    it as well rather than trying to average a set of boundaries.
    """
    if variables is not None:
        # An explicit selection: resolve each reference, refusing ambiguity.
        selection = [resolve_variable(streams, reference) for reference in variables]
    else:
        # The default: every value column in every stream, in stream order, skipping
        # the companions (sigmas, counts, coverage, angular quality) and the bounds.
        selection = [
            (instrument, str(name))
            for instrument, stream in streams.items()
            for name in stream.data_vars
            if not is_companion_name(str(name)) and str(name) != TIME_BOUNDS_VAR
        ]
    # Both routes end here, and so does every caller: the binner joins what this
    # returns, and the grid checks its period against it (§11.2.3).
    for instrument in dict.fromkeys(name for name, _ in selection):
        _refuse_a_joined_product(streams[instrument], instrument)
    return selection


def _refuse_a_joined_product(stream: xr.Dataset, instrument: str) -> None:
    """Raise if a dataset is itself the product of an earlier join.

    A join reads three things from each input row: the value, the interval the
    row describes, and how much of that interval holds data. On a *stream* all
    three are properties of a measurement. On a product they are properties of
    a **row**, and the second pass cannot tell the difference — it takes each
    row as a measurement covering its whole cell, so a minute row holding one
    15 s canister fill is weighted, counted and reported as a fully measured
    minute.

    Measured on a one-hour campaign of a 1 Hz analyzer with outages beside a
    canister (§11.2.3): five-minute rows built from the campaign's 60 s rows
    are up to 2.9 ppb from the same rows built from the streams, and report a
    coverage of 1.000 for a row where the streams say 0.933. A *nested*
    re-join can be made exact by weighting each input row by its own coverage
    (1.6e-12 ppb in the same test); a non-nested one cannot be repaired at all,
    since cutting a row assigns its mean to both sides and the air in the two
    halves differs — 5.8 ppb apart, weighted or not. That is narrowing, and the
    information is gone from the input rather than mishandled by the join.

    So this is a refusal rather than a warning, on the precedent of the
    antimeridian: refuse the case, do not model it. Nothing needs the
    capability — every product can be rebuilt from the native streams, exactly,
    and a sweep over grid period does precisely that.
    """
    stage = str(stream.attrs.get("tsara_stage", ""))
    if stage not in JOINED_STAGES:
        return
    raise TsaraAlignError(
        f"'{instrument}' is a product this package built (tsara_stage is '{stage}'), "
        "and a product may not be joined again. Its rows are not readings: their "
        "coverage, counts and overlap weights describe the rows of the first join "
        "rather than the air, so a second pass would report a fully covered row over "
        "an interval that was barely measured (METHODS §11.2.3). Build what you want "
        "directly from the native streams, which is always possible and exact."
    )


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


def pair_width_ratios(readings: CellBounds, target: CellBounds, pairs: OverlapPairs) -> np.ndarray:
    """Return reading width over target width for every overlapping pair.

    The one number that says which kind of join a pair is (§11.2.4). At or
    below 1 the reading fits inside its target and is averaged, or straddles
    a boundary and is shared between rows. Above 1 the reading is wider than
    the cell it fills, so its value -- a mean over the whole reading -- stands
    for a shorter interval than it measured: the join *narrows* it. At
    :data:`COPY_RATIO` and beyond one reading fills two cells' worth of rows,
    which is the interpolation rule (§1.2) restated for a step function, and
    is refused unless a caller asks for it by name.

    Measured per pair rather than summed over a reading's rows, and that is
    load-bearing. The rule this replaced added up the time a reading shared
    with *all* target cells and refused at twice the widest, which assumed the
    targets do not overlap: sliding windows sixty seconds wide every ten
    seconds share every 30 s reading with six of them and were refused as a
    copy, while a 60 s mean stood on a single 15 s cell -- four times as wide
    as the cell it fills -- passed, because one narrow cell never adds up to
    two. A ratio per pair has neither hole, needs no tolerance (real jitter
    tops out at 1.024 on the archive against a refusal at 2), and reproduces
    every verdict the summed rule gave where that rule was right.

    Parameters
    ----------
    readings : CellBounds
        Cells being averaged.
    target : CellBounds
        Cells to average onto.
    pairs : OverlapPairs
        Their overlaps, from :func:`~tsara.core.support.overlap_pairs`.

    Returns
    -------
    numpy.ndarray
        One ratio per pair; zero for a pair whose target cell has no width,
        which overlaps nothing by a positive amount and weighs nothing.
    """
    reading_width = readings.width_ns[pairs.reading_index].astype(np.float64)
    target_width = target.width_ns[pairs.target_index].astype(np.float64)
    return np.divide(
        reading_width, target_width, out=np.zeros_like(reading_width), where=target_width > 0
    )


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


def phase_offset_s(readings: CellBounds, target: CellBounds) -> float | None:
    """Return how far a same-width stream sits out of phase with the target, or ``None``.

    The one blend that is both exact and actionable (§11.2.4, §11.9.1): when
    every reading is as wide as every target cell and the two tilings are
    offset, each target value blends two readings and each reading lends part
    of itself to two rows -- the borrowed share says how much, ``2f(1 - f)``
    -- and the remedy belongs to the caller, who can move the grid onto the
    instrument's boundaries or pair on a coarser common clock. ``None`` when
    the widths differ (cadence jitter included: a 1.023 s reading on a 1 s
    cell is narrowed, and not a question of phase), when either side is
    empty, or when the tilings coincide.

    Parameters
    ----------
    readings : CellBounds
        The stream's cells.
    target : CellBounds
        The cells it is being put on.

    Returns
    -------
    float or None
        The first non-zero offset of a reading's start from the target's
        tiling, in seconds; ``None`` when the situation does not arise.
    """
    if len(readings) == 0 or len(target) == 0:
        return None
    period = int(target.width_ns[0])
    if (
        period <= 0
        or not np.all(target.width_ns == period)
        or not np.all(readings.width_ns == period)
    ):
        return None
    # How far each reading starts past a target boundary; zero everywhere means in phase.
    offsets = (readings.start_ns - int(target.start_ns.min())) % period
    shifted = offsets[offsets != 0]
    if shifted.size == 0:
        return None
    return float(shifted[0]) / NS_PER_S


def targets_overlap(target: CellBounds) -> bool:
    """Say whether any two target cells overlap by a positive amount.

    Overlapping targets -- sliding windows, nested event windows -- share
    their readings by construction, so rows outnumbering readings is the
    question asked rather than a defect to warn about (§11.2.4). Exact in
    integer nanoseconds, no tolerance: a stream whose fixed-width cells
    overlap by jitter counts as overlapping too, and a join onto its cells
    keeps its per-column record while forgoing the sharing warning, which
    pairing asks again of its surviving rows.

    Parameters
    ----------
    target : CellBounds
        The cells.

    Returns
    -------
    bool
        True if some cell starts before an earlier cell stops.
    """
    if len(target) < 2:
        return False
    order = np.argsort(target.start_ns, kind="stable")
    start, stop = target.start_ns[order], target.stop_ns[order]
    return bool(np.any(start[1:] < np.maximum.accumulate(stop)[:-1]))


@dataclass(frozen=True)
class _SupportChange:
    """How a join changed one column's support: its four attrs, and the warning's input."""

    transform: SupportTransform
    width_ratio_max: float
    borrowed_share: float
    rows: int
    readings: int


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
        return "shared"
    return "averaged"


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


def _record(attrs: dict[str, object], change: _SupportChange) -> None:
    """Write a column's support record into its attrs."""
    attrs[TRANSFORM_ATTR] = change.transform
    attrs[WIDTH_RATIO_ATTR] = change.width_ratio_max
    attrs[BORROWED_ATTR] = change.borrowed_share
    attrs[READINGS_ATTR] = change.readings


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
        "tsara_readings, and carries the share borrowed per cell in borrowed_<name>.",
        len(flagged),
        listed,
        " ..." if len(flagged) > 8 else "",
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
        What to do when a reading is at least :data:`COPY_RATIO` times as
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
        ``tsara_readings``. Cells are described by CF ``time_bnds``.

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
