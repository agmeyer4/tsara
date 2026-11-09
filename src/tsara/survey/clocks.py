"""The clock survey: a shift per segment from tracer pairs, chained to one clock, written as rows.

Where this sits
---------------
Before the pipeline, on streams ingested without a correction table, this
measures how far apart each instrument's files put the same air, segment by
segment, and turns the measurements into the rows of a correction table
(:mod:`tsara.core.clocks`) for a person to review and name in the manifest
(``docs/METHODS.md`` §12.4). It never changes a stream and never applies what
it writes.

The sign
--------
Every number here is a **shift**, the correction table's own: seconds added
to a recorded time, ``corrected = recorded + shift``. An instrument whose file
shows the air 7 s after its partner's is measured at -7 s onto that partner,
and the row the survey writes carries the same -7 s.

What it does, in order
----------------------
1. **Segments.** Time is cut into segments of the set length on one grid
   shared by every instrument (whole multiples of the segment from the epoch),
   so that segment k of an instrument and of its partner are the same stretch
   of time. An instrument is measured only on the segments where it has
   readings.
2. **Tiers.** On each segment, each of the instrument's tiers gives a shift
   onto its partner: its peak on whole seconds
   (:func:`tsara.survey.shifts.pooled_shift`), then the shift itself on the
   readings' own cells, to the survey's resolution
   (:func:`tsara.survey.shifts.refined_shift`), or the whole-second peak,
   named, where the joined readings' cells line up with the others' only at
   some steps of it. The first tier that decides sets the segment's shift; a
   later tier that decides too checks it, and a disagreement larger than the
   tolerance is named. A tier whose partner is undecided in the segment
   cannot set it, so the next tier that decides and reaches the reference
   clock does.
3. **The chain.** The shift onto the partner plus the shift the partner's own
   rows apply, where the segment's air lands on the partner's clock, is the
   shift onto the reference clock: through the rows, so that a partner's
   segment set aside as an outlier, or bridged, passes on the shift its own
   readings are corrected by. The reference clock's instrument is zero
   everywhere; a segment no row of its partner reaches is undecided. A partner
   measured as an after-check has no rows and passes on its residuals.
4. **Outliers.** A decided segment differing by more than the tolerance from
   both its decided neighbours, while they agree with each other, is set
   aside: one segment against two is more likely a coincidence of plumes than
   two clock steps in a row. It stays inside the row around it. Neighbours
   are the adjacent segments, past only segments too short to measure: across
   a stretch with no readings the clock may really have been elsewhere.
5. **Rows.** Consecutive decided segments whose shifts agree within the
   tolerance become one row; a larger jump starts a new row and is named a
   clock step. A segment with readings but no decision ends the row before it,
   so its readings stay uncorrected and ingestion says so; segments with no
   readings at all do not. Up to ``max_bridged`` of undecided segments are
   bridged instead, when the shifts either side agree and the readings run
   across without a logging stop. A row of three or more segments whose
   least-squares line changes by more than the tolerance across it is a clock
   drift, written with ``shift_end``.
6. **The recorded clock.** A table is keyed on the clock each file recorded,
   while a stream's times are cell midpoints after any ``time_shift``; row
   edges are moved back by both, using the stream's own label and nominal
   width (``docs/METHODS.md`` §12.4).

A stream that already carries clock corrections is measured as an after-check
and gets no rows: its shifts are residuals of a table already applied.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.clocks import ClockCorrection, shift_at, write_clock_corrections
from tsara.core.naming import (
    CLOCK_CORRECTIONS_ATTR,
    SUPPORT_LABEL_ATTR,
    SUPPORT_WIDTH_ATTR,
    TIME_COORD,
    TIME_SHIFT_ATTR,
)
from tsara.core.timebase import NS_PER_S, SECOND_NS, SECONDS_PER_HOUR, iso_utc
from tsara.survey.shifts import (
    CellSeries,
    SecondsSeries,
    SegmentShift,
    TsaraSurveyError,
    cell_series,
    pooled_shift,
    refined_shift,
    seconds_series,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt
    import xarray as xr

    from tsara.config.survey import SurveyConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ClockSurvey",
    "FAULT_COLUMNS",
    "FAULT_KINDS",
    "MEASUREMENT_COLUMNS",
    "SEGMENT_COLUMNS",
    "survey_clocks",
]

#: The faults the survey names (``docs/METHODS.md`` §12.5).
FAULT_KINDS = (
    "clock step",
    "clock drift",
    "fast drift",
    "undecided",
    "bridged",
    "tiers disagree",
    "outlier",
    "after-check",
    "whole seconds",
)

#: A gap between readings at least this long, across a clock step, is named in
#: the step's detail: steps at logging stops are the signature of times
#: counted by rows (§12.2).
_LOGGING_STOP_NS = 60 * SECOND_NS

#: Same-signed clock steps in a row that make a fast drift.
_FAST_DRIFT_STEPS = 3

#: The columns of the survey's three tables and their types, declared so that
#: an empty table has the same types as a full one and a saved table reloads
#: exactly (:mod:`tsara.survey.bundle`).
MEASUREMENT_COLUMNS: dict[str, str] = {
    "instrument": "str",
    "tier": "int64",
    "partner": "str",
    "start": "str",
    "end": "str",
    "shift_onto_partner_s": "float64",
    "r": "float64",
    "n_pairs": "int64",
    "at_edge": "bool",
    "pairs_disagree": "bool",
    "noise_removed": "bool",
    "refined": "bool",
    "decided": "bool",
}
SEGMENT_COLUMNS: dict[str, str] = {
    "instrument": "str",
    "start": "str",
    "end": "str",
    "tier": "Int64",
    "partner": "str",
    "shift_onto_partner_s": "float64",
    "shift_onto_reference_s": "float64",
    "noise_removed": "boolean",
    "refined": "boolean",
    "status": "str",
}
FAULT_COLUMNS: dict[str, str] = {
    "instrument": "str",
    "kind": "str",
    "start": "str",
    "end": "str",
    "detail": "str",
}

#: Give the prepared series of one variable of one instrument: on whole
#: seconds for the search, on the readings' own cells for the shift.
_SeriesOf = Callable[[str, str], SecondsSeries]
_CellsOf = Callable[[str, str], CellSeries]


@dataclass(frozen=True)
class ClockSurvey:
    """What the clock survey measured, and the table it derived.

    Attributes
    ----------
    measurements : pandas.DataFrame
        One row per instrument, tier and segment with readings: ``instrument``,
        ``tier`` (from 1), ``partner``, ``start``, ``end``,
        ``shift_onto_partner_s`` (seconds added to the instrument's recorded
        times to put it on its partner's clock, to the survey's resolution),
        ``r`` and ``n_pairs`` (of the whole-second search), ``at_edge``,
        ``pairs_disagree``, ``noise_removed`` (the joined readings' stated
        noise was removed, :func:`~tsara.survey.shifts.refined_shift`),
        ``refined`` (measured to the resolution, not left at the whole-second
        peak), ``decided``.
    segments : pandas.DataFrame
        One row per instrument and segment with readings: ``instrument``,
        ``start``, ``end``, ``tier`` (the tier that decided, NA for none),
        ``partner``, ``shift_onto_partner_s`` (onto that partner),
        ``shift_onto_reference_s`` (the chain's, the number a row carries),
        ``noise_removed`` and ``refined`` (the deciding tier's; NA for none),
        and ``status``:
        ``decided``, ``outlier``, ``too short`` (fewer readings than any tier
        needs), ``too few pairs`` (readings enough, but no tier paired enough
        seconds at its peak), ``weak correlation`` (a tier paired enough, and
        its peak fell below ``min_r``), ``at search edge``, ``pairs disagree``
        or ``partner undecided``.
    rows : tuple of ClockCorrection
        The derived correction rows, provenance ``measured``.
    faults : pandas.DataFrame
        Named faults: ``instrument``, ``kind`` (one of :data:`FAULT_KINDS`),
        ``start``, ``end``, ``detail``.
    config : SurveyConfig
        The settings it ran with.
    """

    measurements: pd.DataFrame
    segments: pd.DataFrame
    rows: tuple[ClockCorrection, ...]
    faults: pd.DataFrame
    config: SurveyConfig

    def write_table(self, path: str | Path) -> Path:
        """Write the derived rows as a correction table, for review before use.

        Raises
        ------
        TsaraClockError
            If the survey derived no rows.
        """
        return write_clock_corrections(self.rows, path)


@dataclass(frozen=True)
class _Settings:
    """The survey's durations as integers, read once."""

    segment_ns: int
    search_s: int
    resolution_ns: int
    running_median_s: int
    max_bridged_ns: int
    tolerance_s: float
    min_pairs: int
    min_r: float


@dataclass(frozen=True)
class _Measured:
    """One tier's shift on one segment."""

    tier: int
    partner: str
    k: int
    shift: SegmentShift
    decided: bool
    min_pairs: int


@dataclass(frozen=True)
class _Fault:
    kind: str
    first: int
    last: int
    detail: str


@dataclass(frozen=True)
class _Applied:
    """What an instrument surveyed already puts on the reference clock, for those tied to it.

    Its rows, read as ingestion would apply them (on the clock its files
    recorded, ``offset_ns`` before the stream's times); or, for a stream
    measured as an after-check, which gets no rows, its segments' residuals.
    """

    rows: tuple[ClockCorrection, ...]
    offset_ns: int
    residuals: Mapping[int, float] | None

    def onto_reference(self, k: int, time_ns: float) -> float | None:
        """Return the shift onto the reference clock at a stream time; None where none holds it."""
        if self.residuals is not None:
            return self.residuals.get(k)
        recorded = np.array([round(time_ns) - self.offset_ns], dtype=np.int64)
        shift, covered = shift_at(self.rows, recorded)
        return float(shift[0]) / NS_PER_S if bool(covered[0]) else None


def survey_clocks(streams: Mapping[str, xr.Dataset], config: SurveyConfig) -> ClockSurvey:
    """Measure each instrument's shift onto the reference clock and derive correction rows.

    Parameters
    ----------
    streams : Mapping of str to xarray.Dataset
        The campaign's streams, ingested without a correction table (a
        :class:`~tsara.ingest.campaign.StreamCollection` or a plain dict).
    config : SurveyConfig
        The reference clock, each instrument's tiers, and the settings.

    Returns
    -------
    ClockSurvey
        Measurements, segment shifts, rows and faults.

    Raises
    ------
    TsaraSurveyError
        If a surveyed instrument, a partner or a tracer variable is missing, a
        tracer variable is circular, or the streams hold no readings.
    """
    order = config.composition_order()
    needed = set(order) | {tier.partner for tiers in config.tracers.values() for tier in tiers}
    missing = sorted(name for name in needed if name not in streams)
    if missing:
        raise TsaraSurveyError(
            f"The survey needs stream(s) {missing}, which were not given; given: {sorted(streams)}."
        )
    settings = _settings(config)
    edges = _segment_edges([streams[name] for name in sorted(needed)], settings.segment_ns)
    seconds: dict[tuple[str, str], SecondsSeries] = {}
    cells: dict[tuple[str, str], CellSeries] = {}

    def series_of(instrument: str, variable: str) -> SecondsSeries:
        if (instrument, variable) not in seconds:
            seconds[(instrument, variable)] = seconds_series(
                streams[instrument], variable, running_median_s=settings.running_median_s
            )
        return seconds[(instrument, variable)]

    def cells_of(instrument: str, variable: str) -> CellSeries:
        if (instrument, variable) not in cells:
            cells[(instrument, variable)] = cell_series(
                streams[instrument], variable, running_median_s=settings.running_median_s
            )
        return cells[(instrument, variable)]

    measurements: list[dict[str, object]] = []
    segments: list[dict[str, object]] = []
    faults: list[dict[str, object]] = []
    rows: list[ClockCorrection] = []
    applied: dict[str, _Applied] = {}
    for name in order:
        stream = streams[name]
        present = _segments_with_readings(stream, edges)
        measured = _measure(name, config, settings, edges, present, series_of, cells_of)
        chosen, through = _chosen(measured, config, edges, applied)
        shifts = {k: m.shift.shift_s + through[k] for k, m in chosen.items()}
        fewest = _fewest_pairs(name, config, settings)
        short = {k for k, count in present.items() if count < fewest}
        outliers = _outliers(shifts, settings.tolerance_s, short)
        decided = {k: shift for k, shift in shifts.items() if k not in outliers}
        statuses = _statuses(present, shifts, outliers, measured, fewest)
        own_faults = (
            _tier_disagreements(measured, config, edges, applied, settings)
            + [
                _Fault("outlier", k, k, f"shift {shifts[k]:+g} s against neighbours")
                for k in sorted(outliers)
            ]
            + _whole_seconds(chosen)
        )
        if CLOCK_CORRECTIONS_ATTR in stream.attrs:
            own_faults.append(_after_check(edges, shifts, settings.tolerance_s))
            applied[name] = _Applied(rows=(), offset_ns=0, residuals=decided)
        else:
            new_rows, row_faults, covered = _rows(
                name, config, settings, edges, statuses, decided, stream, chosen
            )
            rows.extend(new_rows)
            own_faults += row_faults + _undecided(present, statuses, covered)
            applied[name] = _Applied(
                rows=tuple(new_rows), offset_ns=_recorded_offset_ns(stream), residuals=None
            )

        measurements += [_measurement_row(name, m, edges) for m in measured]
        segments += [_segment_row(name, k, edges, chosen, shifts, statuses) for k in present]
        faults += [_fault_row(name, f, edges) for f in own_faults]
        kinds = Counter(f.kind for f in own_faults)
        logger.info(
            "Clock survey, '%s': %d of %d segments decided, %d row(s)%s.",
            name,
            len(decided),
            len(present),
            sum(1 for row in rows if row.instrument == name),
            "".join(f", {n} {kind}" for kind, n in sorted(kinds.items())),
        )
    return ClockSurvey(
        measurements=_table(measurements, MEASUREMENT_COLUMNS),
        segments=_table(segments, SEGMENT_COLUMNS),
        rows=tuple(rows),
        faults=_table(faults, FAULT_COLUMNS),
        config=config,
    )


def _table(records: Sequence[Mapping[str, object]], columns: Mapping[str, str]) -> pd.DataFrame:
    """Build one of the survey's tables with its declared columns and types."""
    return pd.DataFrame(list(records), columns=list(columns)).astype(dict(columns))


def _settings(config: SurveyConfig) -> _Settings:
    """Read the survey's durations once, as integers."""
    return _Settings(
        segment_ns=int(pd.Timedelta(config.segment).value),
        search_s=int(pd.Timedelta(config.search).total_seconds()),
        resolution_ns=int(pd.Timedelta(config.resolution).value),
        running_median_s=int(pd.Timedelta(config.running_median).total_seconds()),
        max_bridged_ns=int(pd.Timedelta(config.max_bridged).value),
        tolerance_s=pd.Timedelta(config.tolerance).value / NS_PER_S,
        min_pairs=config.min_pairs,
        min_r=config.min_r,
    )


def _segment_edges(streams: Sequence[xr.Dataset], segment_ns: int) -> npt.NDArray[np.int64]:
    """Return the shared segment edges: whole multiples of the segment from the epoch."""
    times = [t for t in (_times_ns(stream) for stream in streams) if len(t)]
    if not times:
        raise TsaraSurveyError("The streams the survey needs hold no readings.")
    first = min(int(t[0]) for t in times)
    last = max(int(t[-1]) for t in times)
    start = (first // segment_ns) * segment_ns
    stop = (last // segment_ns + 1) * segment_ns
    return np.arange(start, stop + 1, segment_ns, dtype=np.int64)


def _times_ns(stream: xr.Dataset) -> npt.NDArray[np.int64]:
    """Return a stream's times as epoch nanoseconds."""
    return np.asarray(stream[TIME_COORD].values, dtype="datetime64[ns]").astype(np.int64)


def _segments_with_readings(stream: xr.Dataset, edges: npt.NDArray[np.int64]) -> dict[int, int]:
    """Return the segments in which the stream has readings, with how many, in order."""
    counts = np.diff(np.searchsorted(_times_ns(stream), edges))
    return {int(k): int(counts[k]) for k in np.flatnonzero(counts)}


def _fewest_pairs(name: str, config: SurveyConfig, settings: _Settings) -> int:
    """Return the fewest paired seconds any of the instrument's tiers accepts."""
    return min(tier.min_pairs or settings.min_pairs for tier in config.tracers[name])


def _measure(
    name: str,
    config: SurveyConfig,
    settings: _Settings,
    edges: npt.NDArray[np.int64],
    present: Iterable[int],
    series_of: _SeriesOf,
    cells_of: _CellsOf,
) -> list[_Measured]:
    """Measure every tier of an instrument on every segment it has readings in.

    The whole-second search finds each segment's peak; the shift is then
    measured on the readings' own cells to the survey's resolution, wherever
    the peak is inside the search (``docs/METHODS.md`` §12.4).
    """
    measured = []
    for tier_number, tier in enumerate(config.tracers[name], start=1):
        pairs = [
            (series_of(name, own), series_of(tier.partner, theirs)) for own, theirs in tier.pairs
        ]
        on_cells = [
            (cells_of(name, own), cells_of(tier.partner, theirs)) for own, theirs in tier.pairs
        ]
        min_pairs = tier.min_pairs or settings.min_pairs
        for k in present:
            first_s, stop_s = int(edges[k] // SECOND_NS), int(edges[k + 1] // SECOND_NS)
            shift = pooled_shift(
                pairs,
                first_s,
                stop_s,
                search_s=settings.search_s,
                min_pairs=min_pairs,
                min_r=settings.min_r,
                tolerance_s=settings.tolerance_s,
            )
            if np.isfinite(shift.shift_s) and not shift.at_edge:
                shift_s, at_edge, removed, refined = refined_shift(
                    on_cells,
                    int(edges[k]),
                    int(edges[k + 1]),
                    coarse_s=shift.shift_s,
                    resolution_ns=settings.resolution_ns,
                    min_pairs=min_pairs,
                )
                shift = replace(
                    shift,
                    shift_s=shift_s,
                    at_edge=at_edge,
                    noise_removed=removed,
                    refined=refined,
                )
            decided = bool(
                np.isfinite(shift.shift_s)
                and not shift.at_edge
                and not shift.pairs_disagree
                and shift.r >= settings.min_r
            )
            measured.append(_Measured(tier_number, tier.partner, k, shift, decided, min_pairs))
    return measured


def _chosen(
    measured: Sequence[_Measured],
    config: SurveyConfig,
    edges: npt.NDArray[np.int64],
    applied: Mapping[str, _Applied],
) -> tuple[dict[int, _Measured], dict[int, float]]:
    """Return, per segment, the first tier that decided it and reaches the reference clock.

    Also returns, per segment, the partner's own shift onto the reference clock
    that the chosen tier composes with (:func:`_through_partner`). A tier whose
    partner's correction does not reach where the air lands has nothing to
    compose with, so it gives way to the next that decides: that is what a
    backup tier is for. Found in the walkthrough: taking the first tier that
    decided, composable or not, left a segment undecided while its backup had
    measured the shift straight onto the reference clock.
    """
    chosen: dict[int, _Measured] = {}
    through: dict[int, float] = {}
    for m in sorted(measured, key=lambda m: (m.k, m.tier)):
        if not m.decided or m.k in chosen:
            continue
        partner = _through_partner(m, config, edges, applied)
        if partner is not None:
            chosen[m.k] = m
            through[m.k] = partner
    return chosen, through


def _through_partner(
    m: _Measured,
    config: SurveyConfig,
    edges: npt.NDArray[np.int64],
    applied: Mapping[str, _Applied],
) -> float | None:
    """Return the partner's own shift onto the reference clock, where this segment's air lands.

    Zero for the instrument keeping the reference clock. For any other partner,
    the shift its own rows apply at the segment's midpoint moved by the shift
    measured onto it, which is where the segment's air is on the partner's
    clock; None when no row holds that time. Found in the second walkthrough:
    composing through the partner's segment measurement instead left every
    instrument chained through a partner's outlier segment uncorrected there,
    although the partner's own row corrected its readings.
    """
    if m.partner == config.reference_clock:
        return 0.0
    landing = (int(edges[m.k]) + int(edges[m.k + 1])) / 2 + m.shift.shift_s * NS_PER_S
    return applied[m.partner].onto_reference(m.k, landing)


def _outliers(shifts: Mapping[int, float], tolerance_s: float, short: set[int]) -> set[int]:
    """Return the decided segments that disagree with both neighbours, which agree.

    A neighbour is the adjacent segment, reached past segments too short to
    measure only, and it must be decided. Found in the walkthrough: taking the
    nearest decided segment however far away set aside a stretch between two
    logging stops hours apart, where the clock may really have been elsewhere.
    """
    out = set()
    for k, shift in shifts.items():
        before, after = _neighbour(k, -1, short), _neighbour(k, 1, short)
        if before not in shifts or after not in shifts:
            continue
        lb, la = shifts[before], shifts[after]
        if abs(shift - lb) > tolerance_s and abs(shift - la) > tolerance_s:
            if abs(lb - la) <= tolerance_s:
                out.add(k)
    return out


def _neighbour(k: int, step: int, short: set[int]) -> int:
    """Return the segment next to k in one direction, past segments too short to measure."""
    k += step
    while k in short:
        k += step
    return k


def _statuses(
    present: Mapping[int, int],
    composed: Mapping[int, float],
    outliers: set[int],
    measured: Sequence[_Measured],
    fewest_pairs: int,
) -> dict[int, str]:
    """Say why each segment with readings is decided or not.

    A segment holding fewer readings than any tier needs paired seconds is
    ``too short``: undecidable by construction rather than by the evidence, as
    the slivers a record's first and last segments usually are. One that no
    tier decided says why: ``too few pairs`` when no tier paired enough seconds
    at its peak (the tracer missing, a partner's hole, zero air), ``weak
    correlation`` when one did and the air did not correlate (flat air, a
    parked van). Found in the second walkthrough: the two were one status, two
    different stories for a reviewer.
    """
    edge = {m.k for m in measured if m.shift.at_edge}
    disagree = {m.k for m in measured if m.shift.pairs_disagree}
    paired = {m.k for m in measured if m.shift.n_pairs >= m.min_pairs}
    # Decided by a tier, but by none that reaches the reference clock.
    stranded = {m.k for m in measured if m.decided}
    statuses = {}
    for k, count in present.items():
        if k in outliers:
            statuses[k] = "outlier"
        elif k in composed:
            statuses[k] = "decided"
        elif count < fewest_pairs:
            statuses[k] = "too short"
        elif k in stranded:
            statuses[k] = "partner undecided"
        elif k in edge:
            statuses[k] = "at search edge"
        elif k in disagree:
            statuses[k] = "pairs disagree"
        elif k in paired:
            statuses[k] = "weak correlation"
        else:
            statuses[k] = "too few pairs"
    return statuses


def _tier_disagreements(
    measured: Sequence[_Measured],
    config: SurveyConfig,
    edges: npt.NDArray[np.int64],
    applied: Mapping[str, _Applied],
    settings: _Settings,
) -> list[_Fault]:
    """Name each segment where two tiers put the instrument on the reference clock differently.

    Tiers may have different partners, so each tier's shift is composed to the
    reference clock through its own partner before they are compared; a tier
    whose partner's correction does not reach the segment cannot be compared.
    """
    by_segment: dict[int, list[tuple[int, float]]] = {}
    for m in measured:
        if not m.decided:
            continue
        partner = _through_partner(m, config, edges, applied)
        if partner is None:
            continue
        by_segment.setdefault(m.k, []).append((m.tier, m.shift.shift_s + partner))
    faults = []
    for k, tiers in sorted(by_segment.items()):
        shifts = [shift for _, shift in tiers]
        if max(shifts) - min(shifts) > settings.tolerance_s:
            detail = ", ".join(f"tier {tier} {shift:+g} s" for tier, shift in tiers)
            faults.append(_Fault("tiers disagree", k, k, detail + " onto the reference clock"))
    return faults


def _after_check(
    edges: npt.NDArray[np.int64], shifts: Mapping[int, float], tolerance_s: float
) -> _Fault:
    """Name a stream measured as an after-check, with the residuals that are its point.

    Every decided shift of a corrected stream is a residual of the table
    already applied, and should sit near zero. Found in the second
    walkthrough: the fault said only that no rows were derived, so a reviewer
    had to open the segments table to learn whether the table had worked.
    """
    detail = (
        "the stream carries clock corrections already: measured as an after-check, "
        "no rows derived; "
    )
    if shifts:
        residuals = np.array(list(shifts.values()))
        beyond = int(np.count_nonzero(np.abs(residuals) > tolerance_s))
        detail += (
            f"{len(residuals)} decided segment{'s' if len(residuals) != 1 else ''}, residuals "
            f"{residuals.min():+g} to {residuals.max():+g} s, {beyond} beyond ±{tolerance_s:g} s"
        )
    else:
        detail += "no segment decided"
    return _Fault("after-check", 0, len(edges) - 2, detail)


def _rows(
    name: str,
    config: SurveyConfig,
    settings: _Settings,
    edges: npt.NDArray[np.int64],
    statuses: Mapping[int, str],
    decided: Mapping[int, float],
    stream: xr.Dataset,
    chosen: Mapping[int, _Measured],
) -> tuple[list[ClockCorrection], list[_Fault], set[int]]:
    """Group decided segments into rows; name the clock steps, drifts and bridges.

    Returns the rows, the faults, and the segments whose readings the rows
    cover. Segments too short to measure neither break a row nor end one:
    inside a row they are covered, and at the record's two ends the first and
    last rows reach over them. Undecided segments end a row, unless up to
    ``max_bridged`` of them lie between shifts that agree and the readings run
    across them without a logging stop: then the row bridges them. Between two
    rows across a clock step, the boundary goes at the one gap in the readings
    the step explains, a hole whose excess over the spacing is the jump, else a
    single logging stop (:func:`_step_boundary`); failing that, at the segment
    edges.
    """
    # What ends a row: readings with no decision. An outlier stays inside the
    # row around it, and a segment too short to measure is no evidence either way.
    undecided = {
        k for k, status in statuses.items() if status not in ("decided", "outlier", "too short")
    }
    runs: list[list[int]] = []
    #: Between run i and run i + 1: the jump when it was a clock step, else 0.
    jumps: list[float] = []
    #: Per run, how many undecided segments it bridges.
    bridged: list[int] = []
    faults: list[_Fault] = []
    times = _times_ns(stream)
    for k in sorted(decided):
        if not runs:
            runs.append([k])
            bridged.append(0)
            continue
        last = runs[-1][-1]
        jump = decided[k] - decided[last]
        between = undecided & set(range(last + 1, k))
        if abs(jump) > settings.tolerance_s:
            faults.append(_step(edges, last, k, decided, stream))
            jumps.append(jump)
            runs.append([k])
            bridged.append(0)
        elif between and not _bridgeable(times, edges, last, k, settings.max_bridged_ns):
            jumps.append(0.0)
            runs.append([k])
            bridged.append(0)
        else:
            if between:
                faults.append(
                    _Fault(
                        "bridged",
                        last + 1,
                        k - 1,
                        f"{len(between)} segment(s) with no decision between shifts of "
                        f"{decided[last]:+g} s and {decided[k]:+g} s, readings unbroken",
                    )
                )
                bridged[-1] += len(between)
            runs[-1].append(k)
    faults += _fast_drifts(runs, jumps)

    spans = [[int(edges[run[0]]), int(edges[run[-1] + 1])] for run in runs]
    for i, jump in enumerate(jumps):
        if jump != 0.0:
            boundary = _step_boundary(
                times,
                int(edges[runs[i][-1]]),
                int(edges[runs[i + 1][0] + 1]),
                jump,
                settings.tolerance_s,
            )
            if boundary is not None:
                spans[i][1] = spans[i + 1][0] = boundary
    if runs:
        _reach_over_short_ends(runs, spans, statuses, edges)
    covered = _covered(times, edges, statuses, spans)

    offset = _recorded_offset_ns(stream)
    rows = []
    for run, (start_ns, end_ns), n_bridged in zip(runs, spans, bridged, strict=True):
        shifts = np.array([decided[k] for k in run])
        shift_ns, shift_end_ns = _row_shift(
            run, shifts, edges, start_ns, end_ns, settings.tolerance_s
        )
        if shift_end_ns is not None:
            hours = (end_ns - start_ns) / NS_PER_S / SECONDS_PER_HOUR
            faults.append(
                _Fault(
                    "clock drift",
                    run[0],
                    run[-1],
                    f"shift {shift_ns / NS_PER_S:+.2f} s to {shift_end_ns / NS_PER_S:+.2f} s, "
                    f"{(shift_end_ns - shift_ns) / NS_PER_S / hours:+.2f} s per hour",
                )
            )
        tiers = ", ".join(str(t) for t in sorted({chosen[k].tier for k in run}))
        rows.append(
            ClockCorrection(
                instrument=name,
                start_ns=start_ns - offset,
                end_ns=end_ns - offset,
                shift_ns=shift_ns,
                shift_end_ns=shift_end_ns,
                reference_clock=config.reference_clock,
                provenance="measured",
                reason=(
                    f"clock survey: {len(run)} segment{'s' if len(run) != 1 else ''}, "
                    f"tier {tiers}, shifts {shifts.min():+g} to {shifts.max():+g} s"
                    + (f", {n_bridged} undecided bridged" if n_bridged else "")
                ),
                source="survey",
                line=0,
            )
        )
    return rows, faults, covered


def _bridgeable(
    times: npt.NDArray[np.int64],
    edges: npt.NDArray[np.int64],
    before: int,
    after: int,
    max_bridged_ns: int,
) -> bool:
    """Say whether a row may bridge the segments between two decided ones whose shifts agree.

    Only up to ``max_bridged`` of them, and only where the readings run on
    without a logging stop from the segment before to the segment after: steps
    happen where a logger stopped (§12.2), so a stop is where the clock may
    have changed unseen.
    """
    if int(edges[after] - edges[before + 1]) > max_bridged_ns:
        return False
    across = times[(times >= edges[before]) & (times < edges[after + 1])]
    return not (np.diff(across) >= _LOGGING_STOP_NS).any()


def _covered(
    times: npt.NDArray[np.int64],
    edges: npt.NDArray[np.int64],
    statuses: Mapping[int, str],
    spans: Sequence[Sequence[int]],
) -> set[int]:
    """Return the segments with readings every one of which a row's span holds.

    Counted on the rows' final spans, not their segments: rows meeting at a
    step's gap inside an undecided segment correct all of its readings, and it
    is not undecided. Found in the walkthrough: counting segments named such a
    segment undecided while its readings were corrected.
    """
    if not spans:
        return set()
    starts = np.array([span[0] for span in spans], dtype=np.int64)
    ends = np.array([span[1] for span in spans], dtype=np.int64)
    # The span starting last at or before each reading, which holds it if the
    # reading is also before that span's end (spans neither overlap nor touch out of order).
    which = np.searchsorted(starts, times, side="right") - 1
    held = (which >= 0) & (times < ends[np.clip(which, 0, None)])
    segment = np.searchsorted(edges, times, side="right") - 1
    loose = set(segment[~held].tolist())
    return {k for k in statuses if k not in loose}


def _step_boundary(
    times: npt.NDArray[np.int64], first_ns: int, stop_ns: int, jump_s: float, tolerance_s: float
) -> int | None:
    """Return where a step's rows should meet: the reading after the one gap the step left.

    Searched over ``[first_ns, stop_ns)``, from the last decided segment
    before the step to the first after it, so across any undecided segments
    between. A gap can place the step only if the step explains it. First the
    hole a clock leaves when it jumps forward, its shift falling by
    ``-jump_s``: a gap longer than twice the readings' median spacing whose
    excess over that spacing is the jump, within the tolerance. A clock set
    back leaves no hole, and no gap can match it, since an excess is never
    negative. Failing a hole, a logging stop of a minute or more, where clocks
    step (§12.2). Only the one such gap places the step: when there is none,
    or more than one of the kind looked for, the rows meet at the segment
    edges, which None says, and an undecided segment between them stays
    undecided and named. Found in the second walkthrough: the longest gap of
    any kind was taken, so a 5 s dropout inside an undecided stretch placed a
    clock set back 10 s, and the readings between them took a shift 10 s
    wrong without being named.
    """
    # Both runs' segments are decided, and a correlation needs two paired
    # readings, so there are always at least four readings here to take gaps
    # between.
    across = times[(times >= first_ns) & (times < stop_ns)]
    gaps = np.diff(across)
    spacing = float(np.median(gaps))
    excess_s = (gaps - spacing) / NS_PER_S
    holes = (gaps > 2 * spacing) & (np.abs(excess_s + jump_s) <= tolerance_s)
    stops = gaps >= _LOGGING_STOP_NS
    for candidates in (holes, stops):
        found = np.flatnonzero(candidates)
        if len(found) == 1:
            return int(across[found[0] + 1])
        if len(found) > 1:
            return None
    return None


def _reach_over_short_ends(
    runs: Sequence[Sequence[int]],
    spans: list[list[int]],
    statuses: Mapping[int, str],
    edges: npt.NDArray[np.int64],
) -> None:
    """Extend the first row back and the last row on over the record's short end segments."""
    k = runs[0][0] - 1
    while statuses.get(k) == "too short":
        spans[0][0] = int(edges[k])
        k -= 1
    k = runs[-1][-1] + 1
    while statuses.get(k) == "too short":
        spans[-1][1] = int(edges[k + 1])
        k += 1


def _step(
    edges: npt.NDArray[np.int64],
    before: int,
    after: int,
    shifts: Mapping[int, float],
    stream: xr.Dataset,
) -> _Fault:
    """Name a clock step, and the longest gap in the readings across it if it is a logging stop."""
    times = _times_ns(stream)
    across = times[(times >= edges[before]) & (times < edges[after + 1])]
    detail = f"shift {shifts[before]:+g} s to {shifts[after]:+g} s"
    gap = int(np.max(np.diff(across))) if len(across) > 1 else 0
    if gap >= _LOGGING_STOP_NS:
        detail += f", across a logging stop of {gap / NS_PER_S:.0f} s"
    return _Fault("clock step", before, after, detail)


def _fast_drifts(runs: Sequence[Sequence[int]], jumps: Sequence[float]) -> list[_Fault]:
    """Name a series of same-signed clock steps: a drift faster than the tolerance per segment."""
    faults = []
    i = 0
    while i < len(jumps):
        j = i
        while j < len(jumps) and jumps[j] != 0.0 and np.sign(jumps[j]) == np.sign(jumps[i]):
            j += 1
        if jumps[i] != 0.0 and j - i >= _FAST_DRIFT_STEPS:
            faults.append(
                _Fault(
                    "fast drift",
                    runs[i][0],
                    runs[j][-1],
                    f"{j - i} clock steps of one sign in a row: a drift faster than the "
                    "tolerance per segment, which a shorter segment resolves",
                )
            )
        i = max(j, i + 1)
    return faults


def _row_shift(
    run: Sequence[int],
    shifts: npt.NDArray[np.float64],
    edges: npt.NDArray[np.int64],
    start_ns: int,
    end_ns: int,
    tolerance_s: float,
) -> tuple[int, int | None]:
    """Return a row's shift, and its shift at the end when the row drifts.

    The line through the run's segment shifts is evaluated at the row's own start
    and end, which may lie beyond the run's segments: a row reaching over a
    record's short end segments, or meeting its neighbour at a step's gap. Found
    building notebook 08: evaluated at the last segment's edge instead, a row
    reaching 10 minutes further spread the same change over a longer stretch, so
    a clock drifting 6 s an hour was corrected at 5.4.
    """
    if len(run) >= 3:
        mids = np.array([(edges[k] + edges[k + 1]) / 2 - start_ns for k in run], dtype=np.float64)
        slope, intercept = np.polyfit(mids, shifts, 1)
        at_end = intercept + slope * float(end_ns - start_ns)
        if abs(at_end - intercept) > tolerance_s:
            return _ns(intercept), _ns(at_end)
    return _ns(float(np.median(shifts))), None


def _ns(seconds: float) -> int:
    """Return seconds as whole nanoseconds."""
    return int(round(float(seconds) * NS_PER_S))


def _whole_seconds(chosen: Mapping[int, _Measured]) -> list[_Fault]:
    """Name each run of decided segments whose shift is the whole-second peak, not refined.

    A segment a tier decided is left unrefined only when the joined readings'
    cells line up with the others' at only some steps of the resolution, or
    at none, or cover too few whole at the best step
    (:func:`tsara.survey.shifts.refined_shift`): its shift is then
    good to a second, not to the resolution asked for, and the reviewer is
    told so. A segment later set aside as an outlier is named too, since it
    was measured the same way.
    """
    runs: list[list[int]] = []
    for k in sorted(chosen):
        if chosen[k].shift.refined:
            continue
        if runs and k == runs[-1][-1] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    return [
        _Fault(
            "whole seconds",
            run[0],
            run[-1],
            f"{len(run)} segment{'s' if len(run) != 1 else ''} measured on whole seconds: "
            "on the readings' own cells too few are covered whole at some step of the "
            "resolution",
        )
        for run in runs
    ]


def _undecided(
    present: Mapping[int, int], statuses: Mapping[int, str], covered: set[int]
) -> list[_Fault]:
    """Name each run of consecutive segments with readings that no row covers."""
    runs: list[list[int]] = []
    for k in present:
        if k in covered:
            continue
        if runs and k == runs[-1][-1] + 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    faults = []
    for run in runs:
        reasons = Counter(statuses[k] for k in run)
        detail = ", ".join(f"{n} {reason}" for reason, n in sorted(reasons.items()))
        faults.append(_Fault("undecided", run[0], run[-1], detail))
    return faults


def _recorded_offset_ns(stream: xr.Dataset) -> int:
    """Return how far a stream's times sit after the clock its files recorded.

    A stream's time is its cell midpoint after any ``time_shift``; a table is
    keyed on the label each file wrote. A start label sits half a cell before
    the midpoint, an end label half a cell after; ``mid`` and ``unknown``
    (centred when assumed) sit on it.
    """
    offset = 0
    shift = stream.attrs.get(TIME_SHIFT_ATTR)
    if shift:
        offset += int(pd.Timedelta(str(shift)).value)
    width_s = stream.attrs.get(SUPPORT_WIDTH_ATTR)
    if width_s is not None and np.isfinite(float(width_s)):
        half = round(float(width_s) * NS_PER_S / 2)
        offset += {"start": half, "end": -half}.get(str(stream.attrs.get(SUPPORT_LABEL_ATTR)), 0)
    return offset


def _measurement_row(name: str, m: _Measured, edges: npt.NDArray[np.int64]) -> dict[str, object]:
    return {
        "instrument": name,
        "tier": m.tier,
        "partner": m.partner,
        "start": iso_utc(int(edges[m.k])),
        "end": iso_utc(int(edges[m.k + 1])),
        "shift_onto_partner_s": m.shift.shift_s,
        "r": m.shift.r,
        "n_pairs": m.shift.n_pairs,
        "at_edge": m.shift.at_edge,
        "pairs_disagree": m.shift.pairs_disagree,
        "noise_removed": m.shift.noise_removed,
        "refined": m.shift.refined,
        "decided": m.decided,
    }


def _segment_row(
    name: str,
    k: int,
    edges: npt.NDArray[np.int64],
    chosen: Mapping[int, _Measured],
    composed: Mapping[int, float],
    statuses: Mapping[int, str],
) -> dict[str, object]:
    m = chosen.get(k)
    return {
        "instrument": name,
        "start": iso_utc(int(edges[k])),
        "end": iso_utc(int(edges[k + 1])),
        "tier": pd.NA if m is None else m.tier,
        "partner": "" if m is None else m.partner,
        "shift_onto_partner_s": float("nan") if m is None else m.shift.shift_s,
        "shift_onto_reference_s": composed.get(k, float("nan")),
        "noise_removed": pd.NA if m is None else m.shift.noise_removed,
        "refined": pd.NA if m is None else m.shift.refined,
        "status": statuses[k],
    }


def _fault_row(name: str, fault: _Fault, edges: npt.NDArray[np.int64]) -> dict[str, object]:
    return {
        "instrument": name,
        "kind": fault.kind,
        "start": iso_utc(int(edges[fault.first])),
        "end": iso_utc(int(edges[fault.last + 1])),
        "detail": fault.detail,
    }
