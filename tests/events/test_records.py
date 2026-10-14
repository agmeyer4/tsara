"""Tests for records and dropouts (tsara.events.records)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pytest

from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import DROPOUT_SPACING_FACTOR, TsaraEventError, find_records

HOUR = 3600 * SECOND


def cells_at(midpoints_s: npt.ArrayLike, width_s: float = 1.0) -> CellBounds:
    """Cells of ``width_s`` centred on the given midpoints, in seconds."""
    mid = np.round(np.asarray(midpoints_s, dtype=np.float64) * SECOND).astype(np.int64)
    half = int(width_s * SECOND) // 2
    return CellBounds(start_ns=mid - half, stop_ns=mid + half)


def all_finite(n: int) -> npt.NDArray[np.bool_]:
    return np.ones(n, dtype=bool)


# ---------------------------------------------------------------------------
# Pieces, split at gaps
# ---------------------------------------------------------------------------


def test_a_gap_longer_than_the_rule_splits_and_the_extent_is_the_readings_cells() -> None:
    mids = [0.0, 1.0, 2.0, 3 * 3600.0, 3 * 3600.0 + 1, 3 * 3600.0 + 2]
    cells = cells_at(mids)
    records = find_records(cells, all_finite(6), gap_ns=2 * HOUR, max_length_ns=6 * HOUR)
    assert records.index.tolist() == [0, 0, 0, 1, 1, 1]
    assert records.n == 2
    assert records.start_ns.tolist() == [cells.start_ns[0], cells.start_ns[3]]
    assert records.stop_ns.tolist() == [cells.stop_ns[2], cells.stop_ns[5]]


def test_a_gap_exactly_as_long_as_the_rule_does_not_split() -> None:
    """More than the gap splits; the gap itself does not."""
    records = find_records(
        cells_at([0.0, 7200.0]), all_finite(2), gap_ns=2 * HOUR, max_length_ns=6 * HOUR
    )
    assert records.index.tolist() == [0, 0]


def test_a_reading_that_is_not_finite_belongs_to_no_record_and_splits_nothing() -> None:
    finite = np.array([True, True, False, True])
    records = find_records(
        cells_at([0.0, 1.0, 2.0, 3.0]), finite, gap_ns=2 * HOUR, max_length_ns=6 * HOUR
    )
    assert records.index.tolist() == [0, 0, -1, 0]


def test_no_finite_reading_means_no_record() -> None:
    records = find_records(
        cells_at([0.0, 1.0]), np.zeros(2, dtype=bool), gap_ns=HOUR, max_length_ns=HOUR
    )
    assert records.n == 0
    assert records.index.tolist() == [-1, -1]
    assert not records.dropout_before.any()


# ---------------------------------------------------------------------------
# Parts, cut to the maximum length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_hours", "expected"),
    [
        # 13 hourly readings span 12 h: two parts of exactly 6 h. The reading at
        # 6 h opens the second part, and the one at 12 h, on its closing edge,
        # stays in it.
        (6, [0] * 6 + [1] * 7),
        # Three parts of exactly 4 h.
        (5, [0] * 4 + [1] * 4 + [2] * 5),
        # A maximum longer than the piece leaves it whole.
        (13, [0] * 13),
    ],
)
def test_a_long_piece_is_cut_into_the_fewest_equal_parts(
    max_hours: int, expected: list[int]
) -> None:
    records = find_records(
        cells_at(np.arange(13) * 3600.0),
        all_finite(13),
        gap_ns=2 * HOUR,
        max_length_ns=max_hours * HOUR,
    )
    assert records.index.tolist() == expected


def test_an_empty_part_is_not_a_record_and_the_numbering_stays_consecutive() -> None:
    """Readings at 0, 0.1, 1.9 and 2.0 h with no gap over 2 h, cut to 0.5 h:
    four parts of which the middle two are empty."""
    records = find_records(
        cells_at([0.0, 360.0, 6840.0, 7200.0]),
        all_finite(4),
        gap_ns=2 * HOUR,
        max_length_ns=HOUR // 2,
    )
    assert records.index.tolist() == [0, 0, 1, 1]
    assert records.n == 2


def test_parts_are_exactly_equal_where_whole_nanoseconds_would_not_be() -> None:
    """A 10 ns piece cut into three parts has edges at 3.33 and 6.67 ns.

    Whole-nanosecond parts of ceil(10 / 3) = 4 ns would put the reading at
    7 ns in the second part; exactly equal parts put it in the third.
    """
    mids = np.array([0, 3, 4, 6, 7, 10], dtype=np.int64)
    cells = CellBounds(start_ns=mids, stop_ns=mids + 1)
    records = find_records(cells, all_finite(6), gap_ns=100, max_length_ns=4)
    assert records.index.tolist() == [0, 0, 1, 1, 2, 2]


def test_a_year_long_piece_is_cut_without_overflow() -> None:
    """3.2e16 ns cut into ~1 460 six-hour parts: o * n overflows int64 at the end."""
    year = 365 * 24 * HOUR
    mids = np.linspace(0, year, 2001).astype(np.int64)
    cells = CellBounds(start_ns=mids, stop_ns=mids + SECOND)
    records = find_records(cells, all_finite(mids.size), gap_ns=year, max_length_ns=6 * HOUR)
    assert records.n == 1460
    assert np.all(np.diff(records.index) >= 0)
    assert records.index[-1] == 1459


def test_records_match_a_slow_reference_written_from_the_definition() -> None:
    """The definition, one reading at a time, with part edges as exact fractions,
    half the trials with the platform parked and moving at random."""
    rng = np.random.default_rng(3)
    for trial in range(60):
        steps = rng.choice([1, 2, 3, 900, 9000], size=300, p=[0.5, 0.3, 0.17, 0.02, 0.01])
        mids = np.cumsum(steps).astype(np.int64) * SECOND
        finite = rng.random(mids.size) > 0.1
        cells = CellBounds(start_ns=mids - SECOND // 2, stop_ns=mids + SECOND // 2)
        gap, cap = 3000 * SECOND, int(rng.integers(600, 20000)) * SECOND
        parked = None
        if trial % 2:
            # stretches of random length, so that some last the gap and some do not
            state = np.repeat(np.arange(40) % 2, rng.integers(1, 30, size=40))[: mids.size]
            parked = np.pad(state, (0, mids.size - state.size), mode="edge") == 0
        found = find_records(cells, finite, gap_ns=gap, max_length_ns=cap, parked=parked)
        assert found.index.tolist() == _slow_records(mids, finite, gap, cap, parked)


def _slow_records(
    mids: npt.NDArray[np.int64],
    finite: npt.NDArray[np.bool_],
    gap: int,
    cap: int,
    parked: npt.NDArray[np.bool_] | None = None,
) -> list[int]:
    kept = [i for i in range(mids.size) if finite[i]]
    # Where the platform's stops split: both ends of every parked stretch
    # lasting at least the gap, a stretch lasting from its first reading to the
    # next stretch's first, or to its own last reading at the end. A moving
    # stretch never splits.
    cuts: set[int] = set()
    if parked is not None:
        stretches: list[list[int]] = []
        for i in kept:
            if stretches and parked[i] == parked[stretches[-1][0]]:
                stretches[-1].append(i)
            else:
                stretches.append([i])
        for k, stretch in enumerate(stretches):
            end = stretches[k + 1][0] if k + 1 < len(stretches) else stretch[-1]
            if parked[stretch[0]] and int(mids[end]) - int(mids[stretch[0]]) >= gap:
                cuts.add(stretch[0])
                if k + 1 < len(stretches):
                    cuts.add(stretches[k + 1][0])
    pieces: list[list[int]] = []
    for i in kept:
        if pieces and i not in cuts and int(mids[i]) - int(mids[pieces[-1][-1]]) <= gap:
            pieces[-1].append(i)
        else:
            pieces.append([i])
    out = [-1] * mids.size
    record = 0
    for piece in pieces:
        first, span = int(mids[piece[0]]), int(mids[piece[-1]]) - int(mids[piece[0]])
        n = 1
        while Fraction(span, n) > cap:
            n += 1
        used: dict[int, int] = {}
        for i in piece:
            part = n - 1
            for k in range(n):
                if int(mids[i]) - first < Fraction(span * (k + 1), n):
                    part = k
                    break
            used.setdefault(part, len(used))
            out[i] = record + used[part]
        record += len(used)
    return out


# ---------------------------------------------------------------------------
# Pieces, split where the platform stays parked
# ---------------------------------------------------------------------------


def minutes(*stretches: tuple[int, int]) -> tuple[CellBounds, npt.NDArray[np.bool_]]:
    """One reading a minute through stretches of (moving, minutes), in order.

    Returns the cells and, per reading, whether the platform is parked.
    """
    moving = np.concatenate([np.full(n, s) for s, n in stretches])
    return cells_at(np.arange(moving.size) * 60.0), moving == 0


def test_a_lasting_stop_splits_like_an_outage_and_short_stops_stay_with_the_drive() -> None:
    """An ARC day in miniature: three hours parked at base, then a drive of
    legs and short stops ending in a 100 min stop. Only the base stretch lasts
    the 2 h gap, so it is one piece and the drive, stops and all, another."""
    cells, parked = minutes((0, 180), (1, 70), (0, 25), (1, 46), (0, 7), (1, 72), (0, 100))
    records = find_records(
        cells, all_finite(parked.size), gap_ns=2 * HOUR, max_length_ns=24 * HOUR, parked=parked
    )
    assert records.index.tolist() == [0] * 180 + [1] * (parked.size - 180)


def test_a_lasting_stop_inside_short_legs_splits_at_both_its_ends() -> None:
    cells, parked = minutes((1, 30), (0, 180), (1, 30))
    records = find_records(
        cells, all_finite(parked.size), gap_ns=2 * HOUR, max_length_ns=24 * HOUR, parked=parked
    )
    assert records.index.tolist() == [0] * 30 + [1] * 180 + [2] * 30


def test_a_long_leg_never_splits_so_a_short_stop_stays_with_the_drive() -> None:
    """Three hours at base, then two legs of 150 min with a 20 min stop between
    them. The legs last longer than the gap but are moving, so nothing splits
    at their ends: the stop stays with the drive (the rule built first in the
    walkthrough cut it out as a record of its own)."""
    cells, parked = minutes((0, 180), (1, 150), (0, 20), (1, 150))
    records = find_records(
        cells, all_finite(parked.size), gap_ns=2 * HOUR, max_length_ns=24 * HOUR, parked=parked
    )
    assert records.index.tolist() == [0] * 180 + [1] * 320


def test_a_stretch_lasts_until_the_next_one_starts() -> None:
    """120 readings a minute apart span 119 minutes, but the stretch lasts until
    the next begins at 120: exactly the gap, which is enough. One second less
    is not."""
    cells, parked = minutes((0, 120), (1, 30))
    lasting = find_records(
        cells, all_finite(parked.size), gap_ns=2 * HOUR, max_length_ns=24 * HOUR, parked=parked
    )
    assert lasting.n == 2
    short = find_records(
        cells,
        all_finite(parked.size),
        gap_ns=2 * HOUR + SECOND,
        max_length_ns=24 * HOUR,
        parked=parked,
    )
    assert short.n == 1


def test_the_last_stretch_lasts_to_its_own_last_reading() -> None:
    cells, parked = minutes((1, 60), (0, 121))
    split = find_records(
        cells, all_finite(parked.size), gap_ns=2 * HOUR, max_length_ns=24 * HOUR, parked=parked
    )
    assert split.index.tolist() == [0] * 60 + [1] * 121
    cells, parked = minutes((1, 60), (0, 120))
    kept = find_records(
        cells, all_finite(parked.size), gap_ns=2 * HOUR, max_length_ns=24 * HOUR, parked=parked
    )
    assert kept.n == 1


def test_only_the_finite_readings_states_are_read() -> None:
    """A reading that is not finite neither starts nor ends a stretch."""
    cells, parked = minutes((0, 130), (1, 30))
    finite = all_finite(parked.size)
    finite[125:130] = False
    parked[125:130] = False  # never read
    records = find_records(cells, finite, gap_ns=2 * HOUR, max_length_ns=24 * HOUR, parked=parked)
    assert records.n == 2
    assert records.index[124] == 0 and records.index[130] == 1


# ---------------------------------------------------------------------------
# Dropouts
# ---------------------------------------------------------------------------


def test_a_jittered_record_keeps_its_steps_and_loses_a_missing_row() -> None:
    """The 07-18 shape: readings every 2 or 3 s, median 2 s. A 3 s step is
    1.5 x the median and not more, so not a dropout; a 4 s step is."""
    steps = [2, 3, 2, 2, 3, 2, 4, 2]
    mids = np.concatenate([[0], np.cumsum(steps)]).astype(float)
    records = find_records(cells_at(mids), all_finite(mids.size), gap_ns=HOUR, max_length_ns=HOUR)
    assert records.median_spacing_ns.tolist() == [2 * SECOND]
    assert records.dropout_before.tolist() == [False] * 7 + [True, False]
    assert DROPOUT_SPACING_FACTOR == 1.5


def test_a_missing_row_of_a_regular_record_is_a_dropout_and_a_masked_one_too() -> None:
    """1 s readings: a row absent from the file and a row masked to NaN leave the
    same 2 s hole, and both are dropouts."""
    absent = find_records(
        cells_at([0.0, 1.0, 2.0, 4.0, 5.0]), all_finite(5), gap_ns=HOUR, max_length_ns=HOUR
    )
    masked = find_records(
        cells_at([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([True, True, True, False, True, True]),
        gap_ns=HOUR,
        max_length_ns=HOUR,
    )
    assert absent.dropout_before.tolist() == [False, False, False, True, False]
    assert masked.dropout_before.tolist() == [False, False, False, False, True, False]


def test_each_record_judges_dropouts_by_its_own_spacing() -> None:
    """A 1 s record and a 10 s record: 10 s steps are not dropouts in the second."""
    first = [0.0, 1.0, 2.0, 3.0]
    second = [5 * 3600.0 + 10.0 * k for k in range(4)]
    records = find_records(
        cells_at(first + second), all_finite(8), gap_ns=2 * HOUR, max_length_ns=6 * HOUR
    )
    assert records.median_spacing_ns.tolist() == [SECOND, 10 * SECOND]
    assert not records.dropout_before.any()


def test_a_record_of_one_reading_has_no_spacing() -> None:
    records = find_records(
        cells_at([0.0, 5 * 3600.0, 5 * 3600.0 + 1]),
        all_finite(3),
        gap_ns=HOUR,
        max_length_ns=HOUR,
    )
    assert np.isnan(records.median_spacing_ns[0])
    assert records.median_spacing_ns[1] == SECOND


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_what_cannot_be_split_is_refused() -> None:
    cells = cells_at([0.0, 1.0, 2.0])
    with pytest.raises(TsaraEventError, match="one per reading"):
        find_records(cells, all_finite(2), gap_ns=HOUR, max_length_ns=HOUR)
    with pytest.raises(TsaraEventError, match="parked flags for 3 cells"):
        find_records(
            cells, all_finite(3), gap_ns=HOUR, max_length_ns=HOUR, parked=np.zeros(2, dtype=bool)
        )
    with pytest.raises(TsaraEventError, match="positive gap"):
        find_records(cells, all_finite(3), gap_ns=0, max_length_ns=HOUR)
    with pytest.raises(TsaraEventError, match="positive gap"):
        find_records(cells, all_finite(3), gap_ns=HOUR, max_length_ns=-1)
    with pytest.raises(TsaraEventError, match="not in time order"):
        find_records(cells_at([0.0, 2.0, 1.0]), all_finite(3), gap_ns=HOUR, max_length_ns=HOUR)
