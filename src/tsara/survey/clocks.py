"""The clock survey: a lag per segment from tracer pairs, chained to one clock, written as rows.

Where this sits
---------------
Before the pipeline, on streams ingested without a correction table, this
measures how far apart each instrument's files put the same air, segment by
segment, and turns the measurements into the rows of a correction table
(:mod:`tsara.core.clocks`) for a person to review and name in the manifest
(``docs/METHODS.md`` §12.4). It never changes a stream and never applies what
it writes.

What it does, in order
----------------------
1. **Segments.** Time is cut into segments of the set length on one grid
   shared by every instrument (whole multiples of the segment from the epoch),
   so that segment k of an instrument and of its partner are the same stretch
   of time. An instrument is measured only on the segments where it has
   readings.
2. **Tiers.** On each segment, each of the instrument's tiers gives a lag
   against its partner: its peak on whole seconds
   (:func:`tsara.survey.lags.pooled_lag`), then the lag itself on the
   readings' own cells, to the survey's resolution
   (:func:`tsara.survey.lags.refined_lag`). The first tier
   that decides sets the segment's lag; a later tier that decides too checks
   it, and a disagreement larger than the tolerance is named.
3. **The chain.** The lag against the partner plus the partner's own lag to
   the reference clock in the same segment is the lag to the reference clock.
   The reference clock's instrument is zero everywhere; a segment whose
   partner is undecided there is undecided.
4. **Outliers.** A decided segment differing by more than the tolerance from
   both its decided neighbours, while they agree with each other, is set
   aside: one segment against two is more likely a coincidence of plumes than
   two clock steps in a row. It stays inside the row around it.
5. **Rows.** Consecutive decided segments whose lags agree within the
   tolerance become one row; a larger jump starts a new row and is named a
   clock step. A segment with readings but no decision ends the row before it,
   so its readings stay uncorrected and ingestion says so; segments with no
   readings at all do not. A row of three or more segments whose least-squares
   line changes by more than the tolerance across it is a clock drift, written
   with ``shift_end``. The shift is minus the lag.
6. **The recorded clock.** A table is keyed on the clock each file recorded,
   while a stream's times are cell midpoints after any ``time_shift``; row
   edges are moved back by both, using the stream's own label and nominal
   width (``docs/METHODS.md`` §12.4).

A stream that already carries clock corrections is measured as an after-check
and gets no rows: its lags are residuals of a table already applied.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.clocks import ClockCorrection, write_clock_corrections
from tsara.core.naming import (
    CLOCK_CORRECTIONS_ATTR,
    SUPPORT_LABEL_ATTR,
    SUPPORT_WIDTH_ATTR,
    TIME_COORD,
    TIME_SHIFT_ATTR,
)
from tsara.core.timebase import NS_PER_S, SECOND_NS, SECONDS_PER_HOUR, iso_utc
from tsara.survey.lags import (
    CellSeries,
    SecondsSeries,
    SegmentLag,
    TsaraSurveyError,
    cell_series,
    pooled_lag,
    refined_lag,
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
    "tiers disagree",
    "outlier",
    "after-check",
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
    "lag_s": "float64",
    "r": "float64",
    "n_pairs": "int64",
    "at_edge": "bool",
    "pairs_disagree": "bool",
    "noise_removed": "bool",
    "decided": "bool",
}
SEGMENT_COLUMNS: dict[str, str] = {
    "instrument": "str",
    "start": "str",
    "end": "str",
    "tier": "int64",
    "partner": "str",
    "lag_s": "float64",
    "lag_to_reference_s": "float64",
    "noise_removed": "boolean",
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
#: seconds for the search, on the readings' own cells for the lag.
_SeriesOf = Callable[[str, str], SecondsSeries]
_CellsOf = Callable[[str, str], CellSeries]


@dataclass(frozen=True)
class ClockSurvey:
    """What the clock survey measured, and the table it derived.

    Attributes
    ----------
    measurements : pandas.DataFrame
        One row per instrument, tier and segment with readings: ``instrument``,
        ``tier`` (from 1), ``partner``, ``start``, ``end``, ``lag_s`` (seconds
        the instrument shows the air later than its partner, to the survey's
        resolution), ``r`` and ``n_pairs`` (of the whole-second search),
        ``at_edge``, ``pairs_disagree``, ``noise_removed`` (the joined
        readings' stated noise was removed, :func:`~tsara.survey.lags.refined_lag`),
        ``decided``.
    segments : pandas.DataFrame
        One row per instrument and segment with readings: ``instrument``,
        ``start``, ``end``, ``tier`` (the tier that decided, 0 for none),
        ``partner``, ``lag_s`` (against that partner),
        ``lag_to_reference_s``, ``noise_removed`` (the deciding tier's; NA
        for none), and ``status``: ``decided``, ``outlier``,
        ``too short`` (fewer readings than any tier needs), ``no tier decided``,
        ``at search edge``, ``pairs disagree`` or ``partner undecided``.
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
    tolerance_s: float
    min_pairs: int
    min_r: float


@dataclass(frozen=True)
class _Measured:
    """One tier's lag on one segment."""

    tier: int
    partner: str
    k: int
    lag: SegmentLag
    decided: bool


@dataclass(frozen=True)
class _Fault:
    kind: str
    first: int
    last: int
    detail: str


def survey_clocks(streams: Mapping[str, xr.Dataset], config: SurveyConfig) -> ClockSurvey:
    """Measure each instrument's lag to the reference clock and derive correction rows.

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
        Measurements, segment lags, rows and faults.

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
    to_reference: dict[str, dict[int, float]] = {}
    for name in order:
        stream = streams[name]
        present = _segments_with_readings(stream, edges)
        measured = _measure(name, config, settings, edges, present, series_of, cells_of)
        chosen = _chosen(measured)
        lags = _compose(config, chosen, to_reference)
        outliers = _outliers(lags, settings.tolerance_s)
        decided = {k: lag for k, lag in lags.items() if k not in outliers}
        statuses = _statuses(
            present, chosen, lags, outliers, measured, _fewest_pairs(name, config, settings)
        )
        own_faults = _tier_disagreements(measured, config, to_reference, settings) + [
            _Fault("outlier", k, k, f"lag {lags[k]:+g} s against neighbours")
            for k in sorted(outliers)
        ]
        if CLOCK_CORRECTIONS_ATTR in stream.attrs:
            own_faults.append(
                _Fault(
                    "after-check",
                    0,
                    len(edges) - 2,
                    "the stream carries clock corrections already: measured as an "
                    "after-check, no rows derived",
                )
            )
        else:
            new_rows, row_faults, covered = _rows(
                name, config, settings, edges, statuses, decided, stream, chosen
            )
            rows.extend(new_rows)
            own_faults += row_faults + _undecided(present, statuses, covered)
        to_reference[name] = decided

        measurements += [_measurement_row(name, m, edges) for m in measured]
        segments += [_segment_row(name, k, edges, chosen, lags, statuses) for k in present]
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

    The whole-second search finds each segment's peak; the lag is then
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
            lag = pooled_lag(
                pairs,
                first_s,
                stop_s,
                search_s=settings.search_s,
                min_pairs=min_pairs,
                min_r=settings.min_r,
                tolerance_s=settings.tolerance_s,
            )
            if np.isfinite(lag.lag_s) and not lag.at_edge:
                lag_s, at_edge, removed = refined_lag(
                    on_cells,
                    int(edges[k]),
                    int(edges[k + 1]),
                    coarse_s=lag.lag_s,
                    resolution_ns=settings.resolution_ns,
                    min_pairs=min_pairs,
                )
                lag = replace(lag, lag_s=lag_s, at_edge=at_edge, noise_removed=removed)
            decided = bool(
                np.isfinite(lag.lag_s)
                and not lag.at_edge
                and not lag.pairs_disagree
                and lag.r >= settings.min_r
            )
            measured.append(_Measured(tier_number, tier.partner, k, lag, decided))
    return measured


def _chosen(measured: Sequence[_Measured]) -> dict[int, _Measured]:
    """Return, per segment, the first tier that decided it."""
    chosen: dict[int, _Measured] = {}
    for m in sorted(measured, key=lambda m: (m.k, m.tier)):
        if m.decided and m.k not in chosen:
            chosen[m.k] = m
    return chosen


def _compose(
    config: SurveyConfig,
    chosen: Mapping[int, _Measured],
    to_reference: Mapping[str, Mapping[int, float]],
) -> dict[int, float]:
    """Return the lag to the reference clock of each segment the chain decides."""
    composed: dict[int, float] = {}
    for k, m in chosen.items():
        if m.partner == config.reference_clock:
            composed[k] = m.lag.lag_s
        elif k in to_reference[m.partner]:
            composed[k] = m.lag.lag_s + to_reference[m.partner][k]
    return composed


def _outliers(lags: Mapping[int, float], tolerance_s: float) -> set[int]:
    """Return the decided segments that disagree with both decided neighbours, which agree."""
    keys = sorted(lags)
    out = set()
    for before, k, after in zip(keys, keys[1:], keys[2:], strict=False):
        lb, lk, la = lags[before], lags[k], lags[after]
        if (
            abs(lk - lb) > tolerance_s
            and abs(lk - la) > tolerance_s
            and abs(lb - la) <= tolerance_s
        ):
            out.add(k)
    return out


def _statuses(
    present: Mapping[int, int],
    chosen: Mapping[int, _Measured],
    composed: Mapping[int, float],
    outliers: set[int],
    measured: Sequence[_Measured],
    fewest_pairs: int,
) -> dict[int, str]:
    """Say why each segment with readings is decided or not.

    A segment holding fewer readings than any tier needs paired seconds is
    ``too short``: undecidable by construction rather than by the evidence, as
    the slivers a record's first and last segments usually are.
    """
    edge = {m.k for m in measured if m.lag.at_edge}
    disagree = {m.k for m in measured if m.lag.pairs_disagree}
    statuses = {}
    for k, count in present.items():
        if k in outliers:
            statuses[k] = "outlier"
        elif k in composed:
            statuses[k] = "decided"
        elif count < fewest_pairs:
            statuses[k] = "too short"
        elif k in chosen:
            statuses[k] = "partner undecided"
        elif k in edge:
            statuses[k] = "at search edge"
        elif k in disagree:
            statuses[k] = "pairs disagree"
        else:
            statuses[k] = "no tier decided"
    return statuses


def _tier_disagreements(
    measured: Sequence[_Measured],
    config: SurveyConfig,
    to_reference: Mapping[str, Mapping[int, float]],
    settings: _Settings,
) -> list[_Fault]:
    """Name each segment where two tiers put the instrument on the reference clock differently.

    Tiers may have different partners, so each tier's lag is composed to the
    reference clock through its own partner before they are compared; a tier
    whose partner is undecided in the segment cannot be compared.
    """
    by_segment: dict[int, list[tuple[int, float]]] = {}
    for m in measured:
        if not m.decided:
            continue
        if m.partner == config.reference_clock:
            composed = m.lag.lag_s
        elif m.k in to_reference[m.partner]:
            composed = m.lag.lag_s + to_reference[m.partner][m.k]
        else:
            continue
        by_segment.setdefault(m.k, []).append((m.tier, composed))
    faults = []
    for k, tiers in sorted(by_segment.items()):
        lags = [lag for _, lag in tiers]
        if max(lags) - min(lags) > settings.tolerance_s:
            detail = ", ".join(f"tier {tier} {lag:+g} s" for tier, lag in tiers)
            faults.append(_Fault("tiers disagree", k, k, detail + " to the reference clock"))
    return faults


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
    """Group decided segments into rows; name the clock steps and drifts.

    Returns the rows, the faults, and the segments the rows cover. Segments too
    short to measure neither break a row nor end one: inside a row they are
    covered, and at the record's two ends the first and last rows reach over
    them. Between two rows across a clock step, the boundary goes where the
    readings have their longest gap, when it is a gap and not jitter: a step
    happens where a logger stopped or its clock jumped, not at a segment edge.
    """
    # What ends a row: readings with no decision. An outlier stays inside the
    # row around it, and a segment too short to measure is no evidence either way.
    undecided = {
        k for k, status in statuses.items() if status not in ("decided", "outlier", "too short")
    }
    runs: list[list[int]] = []
    #: Between run i and run i + 1: the jump when it was a clock step, else 0.
    jumps: list[float] = []
    faults: list[_Fault] = []
    for k in sorted(decided):
        if not runs:
            runs.append([k])
            continue
        last = runs[-1][-1]
        jump = decided[k] - decided[last]
        if abs(jump) > settings.tolerance_s:
            faults.append(_step(edges, last, k, decided, stream))
            jumps.append(jump)
            runs.append([k])
        elif undecided & set(range(last + 1, k)):
            jumps.append(0.0)
            runs.append([k])
        else:
            runs[-1].append(k)
    faults += _fast_drifts(runs, jumps)

    spans = [[int(edges[run[0]]), int(edges[run[-1] + 1])] for run in runs]
    times = _times_ns(stream)
    for i, jump in enumerate(jumps):
        if jump != 0.0:
            boundary = _step_boundary(
                times, int(edges[runs[i][-1]]), int(edges[runs[i + 1][0] + 1])
            )
            if boundary is not None:
                spans[i][1] = spans[i + 1][0] = boundary
    covered = {k for run in runs for k in range(run[0], run[-1] + 1)}
    if runs:
        _reach_over_short_ends(runs, spans, statuses, edges, covered)

    offset = _recorded_offset_ns(stream)
    rows = []
    for run, (start_ns, end_ns) in zip(runs, spans, strict=True):
        lags = np.array([decided[k] for k in run])
        shift_ns, shift_end_ns = _shifts(run, lags, edges, start_ns, end_ns, settings.tolerance_s)
        if shift_end_ns is not None:
            hours = (end_ns - start_ns) / NS_PER_S / SECONDS_PER_HOUR
            faults.append(
                _Fault(
                    "clock drift",
                    run[0],
                    run[-1],
                    f"lag {-shift_ns / NS_PER_S:+.2f} s to {-shift_end_ns / NS_PER_S:+.2f} s, "
                    f"{(shift_ns - shift_end_ns) / NS_PER_S / hours:+.2f} s per hour",
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
                    f"tier {tiers}, lags {lags.min():+g} to {lags.max():+g} s"
                ),
                source="survey",
                line=0,
            )
        )
    return rows, faults, covered


def _step_boundary(times: npt.NDArray[np.int64], first_ns: int, stop_ns: int) -> int | None:
    """Return where a step's rows should meet: the reading after the longest gap, if one.

    Searched over ``[first_ns, stop_ns)``, the last segment before the step to
    the first after it. A gap counts when it exceeds twice the readings'
    median spacing there; otherwise the readings run on unbroken and the step
    stays at the segment edge, which None says.
    """
    # Both segments are decided, and a correlation needs two paired readings,
    # so there are always at least four readings here to take gaps between.
    across = times[(times >= first_ns) & (times < stop_ns)]
    gaps = np.diff(across)
    longest = int(np.argmax(gaps))
    if gaps[longest] <= 2 * np.median(gaps):
        return None
    return int(across[longest + 1])


def _reach_over_short_ends(
    runs: Sequence[Sequence[int]],
    spans: list[list[int]],
    statuses: Mapping[int, str],
    edges: npt.NDArray[np.int64],
    covered: set[int],
) -> None:
    """Extend the first row back and the last row on over the record's short end segments."""
    k = runs[0][0] - 1
    while statuses.get(k) == "too short":
        spans[0][0] = int(edges[k])
        covered.add(k)
        k -= 1
    k = runs[-1][-1] + 1
    while statuses.get(k) == "too short":
        spans[-1][1] = int(edges[k + 1])
        covered.add(k)
        k += 1


def _step(
    edges: npt.NDArray[np.int64],
    before: int,
    after: int,
    lags: Mapping[int, float],
    stream: xr.Dataset,
) -> _Fault:
    """Name a clock step, and the longest gap in the readings across it if it is a logging stop."""
    times = _times_ns(stream)
    across = times[(times >= edges[before]) & (times < edges[after + 1])]
    detail = f"lag {lags[before]:+g} s to {lags[after]:+g} s"
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


def _shifts(
    run: Sequence[int],
    lags: npt.NDArray[np.float64],
    edges: npt.NDArray[np.int64],
    start_ns: int,
    end_ns: int,
    tolerance_s: float,
) -> tuple[int, int | None]:
    """Return a row's shift, and its shift at the end when the row drifts.

    The line through the run's segment lags is evaluated at the row's own start
    and end, which may lie beyond the run's segments: a row reaching over a
    record's short end segments, or meeting its neighbour at a step's gap. Found
    building notebook 08: evaluated at the last segment's edge instead, a row
    reaching 10 minutes further spread the same change over a longer stretch, so
    a clock drifting 6 s an hour was corrected at 5.4.
    """
    if len(run) >= 3:
        mids = np.array([(edges[k] + edges[k + 1]) / 2 - start_ns for k in run], dtype=np.float64)
        slope, intercept = np.polyfit(mids, lags, 1)
        at_end = intercept + slope * float(end_ns - start_ns)
        if abs(at_end - intercept) > tolerance_s:
            return -_ns(intercept), -_ns(at_end)
    return -_ns(float(np.median(lags))), None


def _ns(seconds: float) -> int:
    """Return seconds as whole nanoseconds."""
    return int(round(float(seconds) * NS_PER_S))


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
        "lag_s": m.lag.lag_s,
        "r": m.lag.r,
        "n_pairs": m.lag.n_pairs,
        "at_edge": m.lag.at_edge,
        "pairs_disagree": m.lag.pairs_disagree,
        "noise_removed": m.lag.noise_removed,
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
        "tier": 0 if m is None else m.tier,
        "partner": "" if m is None else m.partner,
        "lag_s": float("nan") if m is None else m.lag.lag_s,
        "lag_to_reference_s": composed.get(k, float("nan")),
        "noise_removed": pd.NA if m is None else m.lag.noise_removed,
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
