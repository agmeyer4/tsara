"""Tests for the rolling opening (tsara.baseline.opening).

Evidence of four kinds (METHODS §11.1): the textbook opening as a
pencil-checkable limit (q = 0, whole cells), a reference written from the
definition one reading at a time and compared bitwise with the block form,
the closed forms on a straight ramp without noise, and mutation (recorded in
the commit message). The measurements on generated and real air are in
METHODS §6.5, under their rules.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsara.baseline import (
    RollingOpening,
    TsaraBaselineError,
    rolling_opening,
    rolling_quantile,
    window_cells,
)
from tsara.core.support import CellBounds, overlap_pairs
from tsara.core.timebase import SECOND_NS as SECOND


def cells(width_s: float, n: int, step_s: float | None = None) -> CellBounds:
    """Return ``n`` cells of ``width_s``, starting at 0, ``step_s`` (default: the width) apart."""
    step = width_s if step_s is None else step_s
    start = np.arange(n, dtype=np.int64) * int(step * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def drive_record(n: int = 2500, *, seed: int = 0) -> tuple[CellBounds, np.ndarray]:
    """A drive-shaped record: 1 s cells ~2.3 s apart, jitter, dropouts, masks, plumes, ramps."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(2.3, 0.05, n)
    steps[rng.integers(0, n, 4)] += 300.0
    mid = np.cumsum(steps)
    start = (mid * SECOND).astype(np.int64) - SECOND // 2
    readings = CellBounds(start_ns=start, stop_ns=start + SECOND)
    values = 1900 + 25 * np.sin(mid / 700) + rng.normal(0, 0.7, n)
    for _ in range(25):
        centre, width, amplitude = rng.uniform(0, mid[-1]), rng.uniform(3, 400), rng.uniform(5, 300)
        values += amplitude * np.exp(-0.5 * ((mid - centre) / width) ** 2)
    values[rng.random(n) < 0.03] = np.nan
    return readings, values


def reference(
    readings: CellBounds,
    values: np.ndarray,
    window_ns: int,
    quantiles: list[float],
    *,
    dilation: str,
    min_readings: list[int],
    carry: np.ndarray,
) -> RollingOpening:
    """The definition, one reading at a time.

    The first pass is the rolling quantile, whose own reference is in
    ``test_quantile.py``. The second pass is written out here: for each
    reading, the first pass's values at the readings whose cells overlap its
    window, weighted by that overlap, sorted, the Hazen position of each, and
    the value (and each carried number) interpolated at 1 - q or at the top.
    """
    windows = window_cells(readings, window_ns)
    first = rolling_quantile(readings, values, windows, quantiles, carry=carry)
    assert first.carried is not None
    pairs = overlap_pairs(readings, windows)
    n, k = len(readings), len(quantiles)
    out, sigma, carried = (np.full((n, k), np.nan) for _ in range(3))
    for column, q in enumerate(quantiles):
        blank = (first.n_readings < min_readings[column]) | first.too_wide
        low = np.where(blank, np.nan, first.values[:, column])
        wanted = 1.0 if dilation == "max" else 1.0 - q
        for t in range(n):
            if not np.isfinite(low[t]):
                continue
            here = pairs.target_index == t
            j = pairs.reading_index[here]
            w = pairs.overlap_ns[here].astype(float)
            keep = np.isfinite(low[j]) & (w > 0)
            j, w = j[keep], w[keep]
            order = np.argsort(low[j], kind="stable")
            v = low[j][order]
            riders = (first.sigma[j, column][order], first.carried[j, column][order])
            cumulative = np.cumsum(w[order])
            position = (cumulative - w[order] / 2) / cumulative[-1]
            # The engine's bracket: the first position at or above the wanted
            # one, and the one before it; past either end both are the end.
            below = int(np.searchsorted(position, wanted, side="left"))
            upper, lower = min(below, v.size - 1), max(below - 1, 0)
            span = position[upper] - position[lower]
            f = (wanted - position[lower]) / span if span > 0 else 0.0
            out[t, column] = v[lower] + f * (v[upper] - v[lower])
            sigma[t, column] = riders[0][lower] + f * (riders[0][upper] - riders[0][lower])
            carried[t, column] = riders[1][lower] + f * (riders[1][upper] - riders[1][lower])
    return RollingOpening(out, sigma, carried, first.n_readings, first.coverage, first.too_wide)


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------


def test_at_q_zero_both_forms_are_the_textbook_opening() -> None:
    """With whole cells and q = 0, the first pass is the minimum over each window and both
    second passes take the highest, which is the greyscale opening by a flat window.

    A 3 s window centred on a 1 s cell holds that cell and its two neighbours
    whole, so the opening is the maximum over t-1..t+1 of the minimum over
    j-1..j+1, truncated at the ends: checkable by pencil.
    """
    readings = cells(1.0, 9)
    values = np.array([5.0, 1.0, 4.0, 6.0, 8.0, 3.0, 7.0, 9.0, 2.0])
    pad = np.concatenate([[np.inf], values, [np.inf]])
    erosion = np.minimum(np.minimum(pad[:-2], pad[1:-1]), pad[2:])
    padded = np.concatenate([[-np.inf], erosion, [-np.inf]])
    textbook = np.maximum(np.maximum(padded[:-2], padded[1:-1]), padded[2:])
    # By pencil: erosion [1, 1, 1, 4, 3, 3, 3, 2, 2], then the highest of each three.
    assert textbook.tolist() == [1.0, 1.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 2.0]
    for dilation in ("max", "rank"):
        got = rolling_opening(
            readings, values, 3 * SECOND, [0.0], dilation=dilation, min_readings=[1]
        )
        assert got.values[:, 0].tolist() == textbook.tolist(), dilation


@pytest.mark.parametrize("dilation", ["rank", "max"])
@pytest.mark.parametrize("window_s", [60.0, 600.0])
def test_the_block_form_agrees_with_the_definition(dilation: str, window_s: float) -> None:
    """Bitwise, values and both carried sigmas, on a drive-shaped record with masks and dropouts."""
    readings, values = drive_record()
    carry = np.random.default_rng(1).uniform(0.5, 2.0, values.size)
    quantiles, minimum = [0.05, 0.25], [20, 4]
    got = rolling_opening(
        readings,
        values,
        int(window_s * SECOND),
        quantiles,
        dilation=dilation,  # type: ignore[arg-type]
        min_readings=minimum,
        carry=carry,
    )
    want = reference(
        readings,
        values,
        int(window_s * SECOND),
        quantiles,
        dilation=dilation,
        min_readings=minimum,
        carry=carry,
    )
    assert got.carried is not None and want.carried is not None
    assert np.isfinite(got.values).sum() > 0.5 * got.values.size
    assert np.array_equal(got.values, want.values, equal_nan=True)
    assert np.array_equal(got.sigma, want.sigma, equal_nan=True)
    assert np.array_equal(got.carried, want.carried, equal_nan=True)
    assert np.array_equal(got.n_readings, want.n_readings)


def test_the_maximum_rests_on_one_window_and_carries_its_sigmas() -> None:
    """With ``max`` the opening at a reading is one first-pass value, and its sigmas are that
    window's: the reading the maximum came from is found and its numbers compared."""
    readings, values = drive_record(n=800, seed=3)
    carry = np.arange(values.size, dtype=float)  # a distinct number per reading names it
    window = 120 * SECOND
    first = rolling_quantile(readings, values, window_cells(readings, window), [0.05], carry=carry)
    got = rolling_opening(
        readings, values, window, [0.05], dilation="max", min_readings=[1], carry=carry
    )
    assert got.carried is not None and first.carried is not None
    finite = np.isfinite(got.values[:, 0])
    # Each opening is some first-pass value, and its sigmas are that window's.
    for t in np.flatnonzero(finite)[::37]:
        hits = np.flatnonzero(first.values[:, 0] == got.values[t, 0])
        assert hits.size >= 1
        assert got.sigma[t, 0] in first.sigma[hits, 0]
        assert got.carried[t, 0] in first.carried[hits, 0]
    assert (got.values[finite, 0] >= first.values[finite, 0]).all()


# ---------------------------------------------------------------------------
# What it is for: a ramp, and the rule before the spread
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.01, 0.05, 0.25])
def test_on_a_straight_ramp_rank_lies_on_the_line_and_max_sits_above_it(q: float) -> None:
    """Closed forms, one window from the record's ends, on whole 1 s cells and an odd window.

    The first pass sits (1/2 - q) W s below the line; ``rank`` reads the
    mirror position and lands back on it; ``max`` reads the far end of the
    window and lands (q W - 1/2 cell) s above it, which is the continuous
    q W s to within half a cell. The rolling quantile's lag is checked too,
    since it is what the opening removes.
    """
    slope = 0.05  # per second, 3 per minute
    window_s, n = 301, 1500
    readings = cells(1.0, n)
    line = 1900 + slope * np.arange(n)
    inside = slice(window_s, n - window_s)
    lag = rolling_quantile(readings, line, window_cells(readings, window_s * SECOND), [q])
    assert np.allclose(lag.values[inside, 0], line[inside] - (0.5 - q) * window_s * slope)
    rank = rolling_opening(
        readings, line, window_s * SECOND, [q], dilation="rank", min_readings=[1]
    )
    top = rolling_opening(readings, line, window_s * SECOND, [q], dilation="max", min_readings=[1])
    assert np.allclose(rank.values[inside, 0], line[inside], rtol=0, atol=1e-9)
    overshoot = top.values[inside, 0] - line[inside]
    assert np.allclose(overshoot, (q * window_s - 0.5) * slope, rtol=0, atol=1e-9)
    assert np.all(np.abs(overshoot - q * window_s * slope) <= 0.5 * slope + 1e-9)


@pytest.mark.parametrize("dilation", ["rank", "max"])
def test_a_plume_shorter_than_one_minus_q_windows_is_left_whole(dilation: str) -> None:
    """Every window containing a reading of such a plume holds more than a share q of air
    outside it, so no first-pass value rises and the baseline stays on the background;
    a plume wider than the window lifts it, which is what the window is for (§6.1)."""
    q, window_s, n = 0.1, 101, 1200
    readings = cells(1.0, n)
    for width, lifted in ((int((1 - q) * window_s) - 2, False), (window_s + 30, True)):
        values = np.zeros(n)
        values[500 : 500 + width] = 40.0
        got = rolling_opening(
            readings,
            values,
            window_s * SECOND,
            [q],
            dilation=dilation,  # type: ignore[arg-type]
            min_readings=[1],
        )
        assert (got.values[:, 0] > 0).any() == lifted, width


def test_a_window_failing_the_count_lends_its_value_to_no_neighbour() -> None:
    """The count rule is applied between the passes, not after them.

    A dense record ends in a three-reading plume; a lone reading of the same
    plume sits just past the end. The windows near the end hold the plume's
    readings and little else, so their low quantile is the plume itself. They
    hold fewer readings than the rule asks for, and so take no part in the
    second pass: the dense readings keep a baseline of clean air instead of
    taking the plume's value from the thin windows beside them.
    """
    starts = np.concatenate([np.arange(40), [43]]).astype(np.int64) * SECOND
    readings = CellBounds(start_ns=starts, stop_ns=starts + SECOND)
    values = np.zeros(41)
    values[37:] = 100.0
    window = 9 * SECOND
    first = rolling_quantile(readings, values, window_cells(readings, window), [0.05])
    thin = first.n_readings < 6
    assert thin[-1] and (first.values[thin, 0] == 100.0).any()
    got = rolling_opening(readings, values, window, [0.05], dilation="max", min_readings=[6])
    assert np.isnan(got.values[thin, 0]).all()
    assert (got.values[~thin, 0] < 1.0).all()
    leaky = rolling_opening(readings, values, window, [0.05], dilation="max", min_readings=[1])
    assert (leaky.values[~thin, 0] == 100.0).any()


def test_a_reading_too_wide_for_the_window_takes_no_part_in_the_second_pass() -> None:
    """The width rule, like the count rule, blanks a first-pass value before it can spread."""
    starts = (np.arange(30) * SECOND).astype(np.int64)
    stops = starts + SECOND
    stops[15] = starts[15] + 25 * SECOND  # one reading ten times the window
    readings = CellBounds(start_ns=starts, stop_ns=stops)
    values = np.zeros(30)
    values[15] = 50.0
    got = rolling_opening(
        readings, values, int(2.5 * SECOND), [0.25], dilation="max", min_readings=[1]
    )
    first = rolling_quantile(readings, values, window_cells(readings, int(2.5 * SECOND)), [0.25])
    assert first.too_wide.any()
    assert np.isnan(got.values[first.too_wide, 0]).all()
    assert np.isfinite(got.values[~first.too_wide, 0]).all()


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_many_small_blocks_give_the_same_answer_as_one() -> None:
    readings, values = drive_record(n=600, seed=5)
    args = (readings, values, 300 * SECOND, [0.05, 0.1])
    one = rolling_opening(*args, dilation="rank", min_readings=[20, 10])
    many = rolling_opening(*args, dilation="rank", min_readings=[20, 10], block_elements=97)
    assert np.array_equal(one.values, many.values, equal_nan=True)
    assert np.array_equal(one.sigma, many.sigma, equal_nan=True)
    assert one.carried is None and many.carried is None


def test_the_first_pass_qualifiers_are_the_rolling_quantile_s() -> None:
    readings, values = drive_record(n=600, seed=6)
    window = 300 * SECOND
    first = rolling_quantile(readings, values, window_cells(readings, window), [0.05])
    got = rolling_opening(readings, values, window, [0.05], dilation="rank", min_readings=[20])
    assert np.array_equal(got.n_readings, first.n_readings)
    assert np.array_equal(got.coverage, first.coverage)
    assert np.array_equal(got.too_wide, first.too_wide)


def test_bad_arguments_are_refused_by_name() -> None:
    readings = cells(1.0, 10)
    with pytest.raises(TsaraBaselineError, match="'rank' or 'max', not 'min'"):
        rolling_opening(
            readings,
            np.zeros(10),
            5 * SECOND,
            [0.1],
            dilation="min",  # type: ignore[arg-type]
            min_readings=[1],
        )
    with pytest.raises(TsaraBaselineError, match="2 count.*1 quantile"):
        rolling_opening(
            readings, np.zeros(10), 5 * SECOND, [0.1], dilation="max", min_readings=[1, 2]
        )
