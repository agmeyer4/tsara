"""Tests for the detector (tsara.plumes.hysteresis)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.plumes import (
    Events,
    Records,
    TsaraPlumeError,
    expected_chance_rate,
    find_events,
    find_records,
)

HOUR = 3600 * SECOND
NAN = math.nan


def cells_at(midpoints_s: npt.ArrayLike, width_s: float | npt.ArrayLike = 1.0) -> CellBounds:
    """Cells of ``width_s`` centred on the given midpoints, in seconds."""
    mid = np.round(np.asarray(midpoints_s, dtype=np.float64) * SECOND).astype(np.int64)
    half = np.round(np.asarray(width_s, dtype=np.float64) * SECOND / 2).astype(np.int64)
    return CellBounds(start_ns=mid - half, stop_ns=mid + half)


def detect(
    z: list[float],
    *,
    mids: npt.ArrayLike | None = None,
    width: float | npt.ArrayLike = 1.0,
    finite: npt.ArrayLike | None = None,
    gap_s: float = 0.0,
    enter: float = 3.0,
    exit_: float = 1.0,
) -> tuple[Events, CellBounds]:
    """Run the detector on a hand-written z, one record unless the gaps say otherwise."""
    values = np.asarray(z, dtype=np.float64)
    cells = cells_at(np.arange(values.size) if mids is None else mids, width)
    mask = np.ones(values.size, dtype=bool) if finite is None else np.asarray(finite, dtype=bool)
    records = find_records(cells, mask, gap_ns=2 * HOUR, max_length_ns=6 * HOUR)
    events = find_events(
        np.where(mask, values, NAN),
        cells,
        records,
        enter=enter,
        exit_=exit_,
        max_internal_gap_ns=int(gap_s * SECOND),
    )
    return events, cells


# ---------------------------------------------------------------------------
# One event, by hand
# ---------------------------------------------------------------------------


def test_a_run_above_exit_that_reaches_entry_is_one_event_and_records_itself() -> None:
    events, cells = detect([0.0, 2.0, 3.5, 2.0, 0.5, 0.0])
    assert events.n == 1
    assert events.first.tolist() == [1] and events.last.tolist() == [3]
    assert events.peak.tolist() == [2]
    assert events.z_max.tolist() == [3.5]
    assert events.n_readings.tolist() == [3]
    assert events.start_ns.tolist() == [cells.start_ns[1]]
    assert events.stop_ns.tolist() == [cells.stop_ns[3]]
    assert events.covered.tolist() == [1.0]
    assert events.membership.tolist() == [-1, 0, 0, 0, -1, -1]


def test_a_run_that_never_reaches_entry_is_no_event() -> None:
    events, _ = detect([0.0, 2.0, 2.9, 2.0, 0.0])
    assert events.n == 0
    assert events.membership.tolist() == [-1] * 5


def test_a_reading_exactly_at_a_threshold_is_at_it() -> None:
    """z >= exit continues a run and z >= entry qualifies one."""
    events, _ = detect([0.0, 1.0, 3.0, 1.0, 0.99])
    assert events.first.tolist() == [1] and events.last.tolist() == [3]


def test_the_earliest_of_two_equal_peaks_is_the_peak() -> None:
    events, _ = detect([0.0, 3.5, 3.5, 0.0])
    assert events.peak.tolist() == [1]


def test_there_is_no_minimum_duration() -> None:
    """One reading above entry is an event: width cannot tell a chance
    crossing from a narrow plume (METHODS §9.5, §6.8)."""
    events, _ = detect([0.0, 0.0, 4.0, 0.0, 0.0])
    assert events.n_readings.tolist() == [1]


# ---------------------------------------------------------------------------
# Bridging, and what is never bridged
# ---------------------------------------------------------------------------


def test_a_dip_shorter_than_the_gap_is_bridged_and_belongs_to_the_event() -> None:
    """Runs at readings 1-2 and 4, one reading below exit between them: the dip
    runs from reading 2's cell stop (2.5 s) to reading 4's cell start (3.5 s)."""
    z = [0.0, 3.5, 2.0, 0.5, 2.0, 0.0]
    bridged, _ = detect(z, gap_s=1.5)
    assert bridged.first.tolist() == [1] and bridged.last.tolist() == [4]
    assert bridged.n_readings.tolist() == [4]
    assert bridged.membership.tolist() == [-1, 0, 0, 0, 0, -1]
    # A dip exactly as long as the gap is not shorter than it, so the second
    # run stands alone and, never reaching entry, is no event.
    alone, _ = detect(z, gap_s=1.0)
    assert alone.first.tolist() == [1] and alone.last.tolist() == [2]


def test_bridging_can_join_two_events_into_one() -> None:
    z = [0.0, 3.5, 0.5, 3.5, 0.0]
    assert detect(z, gap_s=0.0)[0].n == 2
    assert detect(z, gap_s=5.0)[0].n == 1


def test_a_run_never_crosses_a_dropout_even_where_a_bridge_would_reach() -> None:
    """Readings every second but one missing: the 2 s step is a dropout."""
    mids = [0.0, 1.0, 2.0, 3.0, 5.0, 6.0]
    z = [0.0, 3.5, 2.0, 2.0, 2.0, 3.5]
    events, _ = detect(z, mids=mids, gap_s=10.0)
    assert events.first.tolist() == [1, 4] and events.last.tolist() == [3, 5]


def test_a_blank_reading_ends_a_run_and_is_never_bridged() -> None:
    """A blank enhancement (NaN z at a finite reading) says nothing about the air."""
    inside = detect([0.0, 3.5, NAN, 3.5, 0.0], gap_s=10.0)[0]
    assert inside.first.tolist() == [1, 3]
    in_dip = detect([0.0, 3.5, 0.5, NAN, 0.5, 3.5, 0.0], gap_s=10.0)[0]
    assert in_dip.first.tolist() == [1, 5]
    assert in_dip.membership[3] == -1


def test_a_record_boundary_ends_a_run() -> None:
    events, _ = detect([3.5, 3.5], mids=[0.0, 3 * 3600.0], gap_s=10 * 3600.0)
    assert events.n == 2


# ---------------------------------------------------------------------------
# What an event's cells cover
# ---------------------------------------------------------------------------


def test_readings_arriving_every_two_or_three_seconds_cover_under_half_and_say_so() -> None:
    """The 07-18 shape: 1 s rows, a finite reading at rows 0, 2, 5, 7 and 9.
    The event holds rows 2, 5 and 7, whose three 1 s cells cover half of the
    6 s from row 2's cell start to row 7's cell stop. The empty rows between
    are not dropouts (median spacing 2.5 s) and belong to nothing."""
    finite = np.zeros(10, dtype=bool)
    finite[[0, 2, 5, 7, 9]] = True
    z = np.zeros(10)
    z[[2, 5, 7]] = [3.5, 2.0, 3.2]
    events, _ = detect(list(z), finite=finite)
    assert events.first.tolist() == [2] and events.last.tolist() == [7]
    assert events.n_readings.tolist() == [3]
    assert events.covered.tolist() == [0.5]
    assert events.membership.tolist() == [-1, -1, 0, -1, -1, 0, -1, 0, -1, -1]


def test_overlapping_jittered_cells_are_counted_once() -> None:
    """1.024 s cells a second apart: three of them cover 3.024 s exactly, the
    event's whole interval, where adding their widths would claim 3.072."""
    events, _ = detect([0.0, 3.5, 3.5, 3.5, 0.0], width=1.024)
    assert events.covered.tolist() == [1.0]


def test_each_event_counts_its_own_cells_from_its_own_first() -> None:
    """2.5 s cells a second apart, events at readings 0 and 2: the first event's
    cell reaches 0.5 s into the second's, and must not be taken off it. Their
    dip is zero where the cells overlap, so with nothing bridged they stay two."""
    events, _ = detect([3.5, 0.0, 3.5], width=2.5)
    assert events.n == 2
    assert events.covered.tolist() == [1.0, 1.0]
    assert detect([3.5, 0.0, 3.5], width=2.5, gap_s=0.001)[0].n == 1


def test_a_cell_wider_than_its_neighbours_is_clipped_to_the_event() -> None:
    """A 4 s cell between two 1 s cells reaches past both ends of the event."""
    events, _ = detect([0.0, 3.5, 3.5, 3.5, 0.0], width=[1.0, 1.0, 4.0, 1.0, 1.0])
    assert events.covered.tolist() == [1.0]


# ---------------------------------------------------------------------------
# Against a slow reference written from the definition
# ---------------------------------------------------------------------------


def test_the_detector_matches_a_slow_reference_one_reading_at_a_time() -> None:
    """Random records with jittered spacing, dropouts, empty rows, blank
    readings, a record boundary, overlapping cells and three bridging gaps."""
    rng = np.random.default_rng(8)
    for trial in range(60):
        n = 400
        steps = rng.choice([1.0, 2.0, 3.0, 7.0], size=n, p=[0.6, 0.2, 0.17, 0.03])
        steps[n // 2] = 3 * 3600.0  # a record boundary
        mids = np.cumsum(steps)
        widths = rng.choice([1.0, 1.024, 2.5], size=n, p=[0.7, 0.2, 0.1])
        cells = cells_at(mids, widths)
        finite = rng.random(n) > 0.05
        z = np.convolve(rng.normal(size=n), np.ones(4) / 2, mode="same")
        z[rng.random(n) < 0.03] = NAN  # blank enhancement at a finite reading
        z[~finite] = NAN
        records = find_records(cells, finite, gap_ns=2 * HOUR, max_length_ns=6 * HOUR)
        gap = [0, 2 * SECOND, 5 * SECOND][trial % 3]
        found = find_events(z, cells, records, enter=2.5, exit_=0.5, max_internal_gap_ns=gap)
        expected = _slow_events(z, cells, records, 2.5, 0.5, gap)
        assert list(zip(found.first, found.last, found.peak, strict=True)) == expected["spans"]
        assert found.membership.tolist() == expected["membership"]


def _slow_events(
    z: npt.NDArray[np.float64],
    cells: CellBounds,
    records: Records,
    enter: float,
    exit_: float,
    gap: int,
) -> dict[str, list]:  # type: ignore[type-arg]
    spans: list[tuple[int, int, int]] = []
    membership = [-1] * z.size
    current: list[int] = []  # the event's readings so far, ending above exit
    dip: list[int] = []  # readings below exit since its last reading above

    def close() -> None:
        if current and max(z[i] for i in current) >= enter:
            peak = max(current, key=lambda i: (z[i], -i))
            for i in current:
                membership[i] = len(spans)
            spans.append((current[0], current[-1], peak))

    previous = None
    for i in range(z.size):
        if records.index[i] < 0:
            continue
        broken = (
            previous is None
            or records.index[i] != records.index[previous]
            or bool(records.dropout_before[i])
            or math.isnan(z[i])
            or math.isnan(z[previous])
        )
        previous = i
        if broken:
            close()
            current, dip = [], []
        if math.isnan(z[i]):
            continue
        if z[i] >= exit_:
            if current and dip and max(cells.start_ns[i] - cells.stop_ns[current[-1]], 0) >= gap:
                close()  # the dip was not bridged: it belongs to neither run
                current, dip = [], []
            current = current + dip + [i]
            dip = []
        elif current:
            dip.append(i)
    close()
    return {"spans": spans, "membership": membership}


# ---------------------------------------------------------------------------
# The chance rate: closed form and Monte Carlo
# ---------------------------------------------------------------------------


def test_the_closed_form_gives_the_rates_methods_quotes() -> None:
    """79.7, 4.85 and 0.11 an hour at 1 s for entries 2, 3 and 4 (exit 1), and
    exit 1, 1.5 or 2 changes each by under 0.4 % (METHODS §6.8)."""
    per_hour = {e: 3600 * expected_chance_rate(e, 1.0) for e in (2.0, 3.0, 4.0)}
    assert round(per_hour[2.0], 1) == 79.7
    assert round(per_hour[3.0], 2) == 4.85
    assert round(per_hour[4.0], 2) == 0.11
    for enter in (2.0, 3.0, 4.0):
        for exit_ in (1.5, 1.9):
            ratio = expected_chance_rate(enter, exit_) / expected_chance_rate(enter, 1.0)
            assert abs(ratio - 1) < 0.004


def test_white_noise_crosses_as_often_as_the_closed_form_says() -> None:
    """Monte Carlo. Rule: z drawn N(0, 1), independent, on contiguous 1 s cells,
    200 h in one record, seed 12; nothing bridged. The count of events is held
    to five Poisson standard errors of the closed form (79.7 and 4.85 an hour
    expect about 15,950 and 970). Bridging can only merge events, so with
    dips bridged the count is never higher."""
    n = 200 * 3600
    z = np.random.default_rng(12).normal(size=n)
    cells = CellBounds(
        start_ns=np.arange(n, dtype=np.int64) * SECOND,
        stop_ns=np.arange(1, n + 1, dtype=np.int64) * SECOND,
    )
    records = find_records(cells, np.ones(n, dtype=bool), gap_ns=HOUR, max_length_ns=300 * HOUR)
    for enter in (2.0, 3.0):
        expected = n * expected_chance_rate(enter, 1.0)
        plain = find_events(z, cells, records, enter=enter, exit_=1.0, max_internal_gap_ns=0)
        assert abs(plain.n - expected) < 5 * math.sqrt(expected), enter
        bridged = find_events(
            z, cells, records, enter=enter, exit_=1.0, max_internal_gap_ns=5 * SECOND
        )
        assert bridged.n <= plain.n


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_what_cannot_be_detected_is_refused() -> None:
    cells = cells_at([0.0, 1.0, 2.0])
    records = find_records(cells, np.ones(3, dtype=bool), gap_ns=HOUR, max_length_ns=HOUR)
    with pytest.raises(TsaraPlumeError, match="does not match"):
        find_events(np.zeros(2), cells, records, enter=3.0, exit_=1.0, max_internal_gap_ns=0)
    with pytest.raises(TsaraPlumeError, match="must exceed"):
        find_events(np.zeros(3), cells, records, enter=1.0, exit_=1.0, max_internal_gap_ns=0)
    with pytest.raises(TsaraPlumeError, match="negative"):
        find_events(np.zeros(3), cells, records, enter=3.0, exit_=1.0, max_internal_gap_ns=-1)


def test_a_variable_with_no_finite_reading_has_no_event() -> None:
    events, _ = detect([NAN, NAN], finite=[False, False])
    assert events.n == 0
    assert events.covered.size == 0 and events.membership.tolist() == [-1, -1]
