"""Tests for the weighted rolling quantile (tsara.baseline.quantile).

Five kinds of evidence (METHODS §11.1): pencil-checkable fixtures for the
definition, the per-window reference against the block form (bitwise), the
closed form at equal weights (numpy's Hazen rule), a Monte Carlo of the
Woodruff uncertainty against the scatter of the estimator, and mutation
(recorded in the commit message).
"""

from __future__ import annotations

import numpy as np
import pytest

from tsara.baseline import (
    MAX_BLOCK_ELEMENTS,
    RollingQuantile,
    TsaraBaselineError,
    rolling_quantile,
    weighted_quantile,
    window_cells,
)
from tsara.core.support import COPY_RATIO, CellBounds, overlap_pairs
from tsara.core.timebase import SECOND_NS as SECOND


def cells(start_s: float, width_s: float, n: int, step_s: float | None = None) -> CellBounds:
    """Return ``n`` cells of ``width_s`` whose starts are ``step_s`` (default: the width) apart."""
    step = width_s if step_s is None else step_s
    start = (np.arange(n, dtype=np.int64) * int(step * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def jittered_record(
    n: int = 3000, *, seed: int = 0, masked: float = 0.03, dropouts: int = 4
) -> tuple[CellBounds, np.ndarray]:
    """A record shaped like a drive: 1 s cells ~2.3 s apart with jitter, dropouts, masks."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(2.3, 0.05, n)
    steps[rng.integers(0, n, dropouts)] += 300.0
    mid = np.cumsum(steps)
    start = (mid * SECOND).astype(np.int64) - SECOND // 2
    readings = CellBounds(start_ns=start, stop_ns=start + SECOND)
    values = 1900 + 20 * np.sin(mid / 900) + rng.normal(0, 0.7, n)
    for _ in range(20):
        centre, width, amplitude = rng.uniform(0, mid[-1]), rng.uniform(3, 60), rng.uniform(10, 500)
        values += amplitude * np.exp(-0.5 * ((mid - centre) / width) ** 2)
    values[rng.random(n) < masked] = np.nan
    return readings, values


def reference(
    readings: CellBounds, values: np.ndarray, windows: CellBounds, quantiles: np.ndarray
) -> RollingQuantile:
    """The definition applied window by window, through the overlap search every join uses."""
    pairs = overlap_pairs(readings, windows)
    n = len(windows)
    out = np.full((n, quantiles.size), np.nan)
    sigma = np.full((n, quantiles.size), np.nan)
    count = np.zeros(n, dtype=np.int64)
    coverage = np.zeros(n)
    n_eff = np.full(n, np.nan)
    wide = np.zeros(n, dtype=bool)
    for i in range(n):
        here = pairs.target_index == i
        idx = pairs.reading_index[here]
        weight = np.where(np.isfinite(values[idx]), pairs.overlap_ns[here], 0).astype(float)
        keep = weight > 0
        count[i] = keep.sum()
        coverage[i] = weight.sum() / windows.width_ns[i]
        if not keep.any():
            continue
        wide[i] = bool(np.any(readings.width_ns[idx][keep] / windows.width_ns[i] >= COPY_RATIO))
        n_eff[i] = weight.sum() ** 2 / (weight**2).sum()
        out[i] = weighted_quantile(values[idx], weight, quantiles)
        error = np.sqrt(quantiles * (1 - quantiles) / n_eff[i])
        low = weighted_quantile(values[idx], weight, np.clip(quantiles - error, 0, 1))
        high = weighted_quantile(values[idx], weight, np.clip(quantiles + error, 0, 1))
        sigma[i] = (high - low) / 2
    return RollingQuantile(out, sigma, count, coverage, n_eff, wide)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def close(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal to rounding: for sums the two forms take in different orders."""
    return bool(np.allclose(a, b, rtol=1e-12, atol=0, equal_nan=True))


def agree(got: RollingQuantile, want: RollingQuantile) -> None:
    """Values bitwise; counts and flags exact; the differently summed qualifiers to rounding."""
    assert same(got.values, want.values), "values"
    assert same(got.n_readings, want.n_readings), "n_readings"
    assert same(got.too_wide, want.too_wide), "too_wide"
    assert close(got.coverage, want.coverage), "coverage"
    assert close(got.n_effective, want.n_effective), "n_effective"
    assert close(got.sigma, want.sigma), "sigma"


# ---------------------------------------------------------------------------
# The definition, on one window
# ---------------------------------------------------------------------------


def test_the_weighted_quantile_by_hand() -> None:
    """Values 1, 2, 3 with weights 1, 1, 2: positions 1/8, 3/8, 3/4.

    The median sits a third of the way from 2 (at 3/8) to 3 (at 3/4);
    anything below 1/8 is the lowest reading and anything above 3/4 the
    highest, clamped rather than extrapolated.
    """
    values, weights = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0])
    got = weighted_quantile(values, weights, [0.05, 0.125, 0.5, 0.75, 0.9])
    assert got == pytest.approx([1.0, 1.0, 2.0 + 1 / 3, 3.0, 3.0])


def test_the_order_of_the_readings_does_not_matter() -> None:
    values, weights = np.array([3.0, 1.0, 2.0]), np.array([2.0, 1.0, 1.0])
    assert weighted_quantile(values, weights, 0.5) == pytest.approx(2.0 + 1 / 3)


def test_equal_weights_reduce_to_hazen_positions() -> None:
    """Numpy's `method="hazen"` is (k - 0.5) / N, the equal-weight case of §6.3."""
    rng = np.random.default_rng(1)
    values = rng.normal(size=257)
    quantiles = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 1.0])
    got = weighted_quantile(values, np.ones(values.size), quantiles)
    assert got == pytest.approx(np.quantile(values, quantiles, method="hazen"), rel=1e-12)


def test_a_masked_or_weightless_reading_takes_no_part() -> None:
    values = np.array([np.nan, 1.0, 2.0, 100.0])
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    assert weighted_quantile(values, weights, 1.0) == pytest.approx(2.0)
    assert np.isnan(weighted_quantile(values, np.zeros(4), 0.5)).all()


def test_one_reading_is_every_quantile() -> None:
    assert weighted_quantile([7.0], [3.0], [0.0, 0.5, 1.0]) == pytest.approx([7.0, 7.0, 7.0])


@pytest.mark.parametrize("bad", [[-0.1], [1.5], [np.nan], []])
def test_a_quantile_outside_the_unit_interval_is_refused(bad: list[float]) -> None:
    with pytest.raises(TsaraBaselineError, match="quantile"):
        weighted_quantile([1.0, 2.0], [1.0, 1.0], bad)


def test_mismatched_or_negative_weights_are_refused() -> None:
    with pytest.raises(TsaraBaselineError, match="same length"):
        weighted_quantile([1.0, 2.0], [1.0], 0.5)
    with pytest.raises(TsaraBaselineError, match="negative"):
        weighted_quantile([1.0, 2.0], [1.0, -1.0], 0.5)


# ---------------------------------------------------------------------------
# The block form against the reference
# ---------------------------------------------------------------------------


QUANTILES = np.array([0.01, 0.05, 0.10, 0.5])


@pytest.mark.parametrize("window_s", [30.0, 120.0, 600.0])
def test_the_block_form_agrees_with_the_reference(window_s: float) -> None:
    """Values bitwise (same cumulative sums, same expression); the rest to rounding.

    The sigma depends on the effective count, whose squared-weight sum the
    two forms take in different orders, so it too is held to rounding.
    """
    readings, values = jittered_record()
    windows = window_cells(readings, int(window_s * SECOND))
    got = rolling_quantile(readings, values, windows, QUANTILES)
    agree(got, reference(readings, values, windows, QUANTILES))
    assert got.values.shape == (len(readings), QUANTILES.size)


def test_many_small_blocks_give_the_same_answer_as_one() -> None:
    readings, values = jittered_record(n=800)
    windows = window_cells(readings, 120 * SECOND)
    whole = rolling_quantile(readings, values, windows, QUANTILES)
    pieces = rolling_quantile(readings, values, windows, QUANTILES, block_elements=1)
    assert pieces.values.shape == whole.values.shape
    for field in ("values", "sigma", "n_readings", "coverage", "n_effective", "too_wide"):
        assert same(getattr(whole, field), getattr(pieces, field)), field
    assert MAX_BLOCK_ELEMENTS > 1


def test_windows_need_not_be_centred_on_the_readings() -> None:
    """Any cells will do: samples on their own cells, rolled over windows centred elsewhere."""
    readings, values = jittered_record(n=500)
    windows = cells(100.0, 45.0, 20, step_s=37.0)
    got = rolling_quantile(readings, values, windows, np.array([0.5]))
    agree(got, reference(readings, values, windows, np.array([0.5])))


def test_a_window_touching_nothing_is_blank_with_a_count_of_zero() -> None:
    readings = cells(0.0, 1.0, 10)
    windows = cells(1000.0, 10.0, 3)
    got = rolling_quantile(readings, np.arange(10.0), windows, [0.05])
    assert np.isnan(got.values).all() and np.isnan(got.sigma).all()
    assert (got.n_readings == 0).all() and (got.coverage == 0).all()
    assert np.isnan(got.n_effective).all() and not got.too_wide.any()


def test_empty_inputs_return_empty_results() -> None:
    empty = CellBounds(start_ns=np.array([], dtype=np.int64), stop_ns=np.array([], dtype=np.int64))
    readings = cells(0.0, 1.0, 5)
    assert rolling_quantile(readings, np.arange(5.0), empty, [0.5]).values.shape == (0, 1)
    assert rolling_quantile(empty, np.array([]), cells(0.0, 10.0, 2), [0.5]).values.shape == (2, 1)


# ---------------------------------------------------------------------------
# What every window records (§6.3, §6.4)
# ---------------------------------------------------------------------------


def test_a_reading_counts_in_proportion_to_the_share_of_its_cell_inside_the_window() -> None:
    """Membership by overlap, not by midpoint: half a cell inside is half a weight.

    Two 10 s readings, the window [5, 25) covering half of the first and all
    of the second. Weights 5 and 10: positions 1/6 and 2/3, so the median is
    half-way from the first value to the second, where midpoint membership
    would take the whole first reading and put the median at a quarter.
    """
    readings = cells(0.0, 10.0, 2)
    window = CellBounds(start_ns=np.array([5 * SECOND]), stop_ns=np.array([25 * SECOND]))
    got = rolling_quantile(readings, np.array([0.0, 6.0]), window, [0.5])
    assert got.values[0, 0] == pytest.approx(4.0)  # 0 + (0.5 - 1/6) / (2/3 - 1/6) * 6
    assert got.coverage[0] == pytest.approx(0.75)
    assert got.n_readings[0] == 2
    assert got.n_effective[0] == pytest.approx(15**2 / (25 + 100))


def test_the_edge_of_a_record_is_no_special_case() -> None:
    """The first window holds only its right half: coverage 0.5, count half, no padding.

    The window on the first reading (cell [0, 1), midpoint 0.5 s) spans
    [-59.5, 60.5) s: sixty whole cells and half of the sixty-first. In the
    middle of the record a window spans [240.5, 360.5): half a cell at each
    end and 119 whole ones, coverage exactly 1.
    """
    readings = cells(0.0, 1.0, 600)
    windows = window_cells(readings, 120 * SECOND)
    got = rolling_quantile(readings, np.arange(600.0), windows, [0.05])
    assert got.coverage[0] == pytest.approx(60.5 / 120)
    assert got.n_readings[0] == 61
    assert got.coverage[300] == pytest.approx(1.0)
    assert got.n_readings[300] == 121
    # And the value at the edge is the quantile of what is there, nothing invented:
    # sixty readings at full weight and the sixty-first at half.
    weights = np.r_[np.ones(60), 0.5]
    assert got.values[0, 0] == pytest.approx(weighted_quantile(np.arange(61.0), weights, 0.05))


def test_a_masked_reading_reduces_coverage_and_count_but_not_the_window() -> None:
    readings = cells(0.0, 1.0, 100)
    values = np.arange(100.0)
    values[40:65] = np.nan
    windows = window_cells(readings, 20 * SECOND)
    got = rolling_quantile(readings, values, windows, [0.5])
    # Reading 52's window spans [42.5, 62.5): every cell in it is masked.
    assert got.n_readings[52] == 0 and np.isnan(got.values[52, 0])
    # Reading 35's window spans [25.5, 45.5): half of cell 25, cells 26-39 whole,
    # and the masked cells 40-45 count for nothing.
    assert got.n_readings[35] == 15 and got.coverage[35] == pytest.approx(14.5 / 20)


def test_a_reading_at_least_twice_as_wide_as_the_window_is_flagged() -> None:
    """A 60 s mean under a 20 s window would be copied; the flag says so, the caller blanks."""
    readings = cells(0.0, 60.0, 10)
    narrow = window_cells(readings, 20 * SECOND)
    got = rolling_quantile(readings, np.arange(10.0), narrow, [0.5])
    assert got.too_wide.all()
    wide = window_cells(readings, 600 * SECOND)
    assert not rolling_quantile(readings, np.arange(10.0), wide, [0.5]).too_wide.any()
    # Exactly at the line counts, as it does for a join (§11.2.4).
    at_the_line = window_cells(readings, 30 * SECOND)
    assert rolling_quantile(readings, np.arange(10.0), at_the_line, [0.5]).too_wide.all()


def test_bad_arguments_are_refused_by_name() -> None:
    readings = cells(0.0, 1.0, 5)
    windows = window_cells(readings, 10 * SECOND)
    with pytest.raises(TsaraBaselineError, match="one to one"):
        rolling_quantile(readings, np.arange(4.0), windows, [0.5])
    with pytest.raises(TsaraBaselineError, match="block_elements"):
        rolling_quantile(readings, np.arange(5.0), windows, [0.5], block_elements=0)
    with pytest.raises(TsaraBaselineError, match="quantile"):
        rolling_quantile(readings, np.arange(5.0), windows, [2.0])
    flat = CellBounds(start_ns=np.array([0, SECOND]), stop_ns=np.array([0, SECOND]))
    with pytest.raises(TsaraBaselineError, match="positive duration"):
        rolling_quantile(readings, np.arange(5.0), flat, [0.5])


# ---------------------------------------------------------------------------
# Woodruff's uncertainty against the scatter of the estimator (§6.7)
# ---------------------------------------------------------------------------


def test_the_reported_sigma_tracks_the_scatter_of_the_quantile_over_independent_windows() -> None:
    """Monte Carlo, not algebra: 2000 windows of 200 independent normal draws.

    The reported one-sigma figure (Woodruff) is compared with the standard
    deviation of the estimated 5th percentile across the windows, which is
    the sampling error it claims to describe, and its interval is checked to
    cover the true quantile about 68 % of the time. Loose bounds, because
    Woodruff's interval is asymmetric in the tail and a half-width is a
    summary of it; what is being caught is a factor, not a percent.
    """
    rng = np.random.default_rng(7)
    per_window, n_windows = 200, 2000
    readings = cells(0.0, 1.0, per_window * n_windows)
    values = rng.standard_normal(per_window * n_windows)
    windows = cells(0.0, float(per_window), n_windows)
    got = rolling_quantile(readings, values, windows, [0.05])
    truth = float(np.quantile(rng.standard_normal(2_000_000), 0.05))
    estimates, sigmas = got.values[:, 0], got.sigma[:, 0]
    assert (got.n_readings == per_window).all()
    ratio = float(np.mean(sigmas) / np.std(estimates))
    assert 0.8 < ratio < 1.25, ratio
    covered = float(np.mean(np.abs(estimates - truth) <= sigmas))
    assert 0.60 < covered < 0.76, covered


def test_a_position_above_the_highest_reading_clamps_in_a_partly_filled_row() -> None:
    """A window with fewer candidates than the block is wide has unfilled slots.

    A quantile above the window's highest position must return that reading,
    not read the slot beyond it. The record's dropouts give windows with far
    fewer readings than the widest window, so most rows are partly filled.
    """
    readings, values = jittered_record(n=1500, dropouts=6)
    windows = window_cells(readings, 120 * SECOND)
    top = np.array([0.99, 1.0])
    got = rolling_quantile(readings, values, windows, top)
    agree(got, reference(readings, values, windows, top))
    assert np.isfinite(got.values[got.n_readings > 0]).all()


def test_a_window_holding_one_reading_has_that_value_and_no_spread() -> None:
    """One reading: every quantile is it, and the Woodruff interval collapses to it.

    The upper Woodruff position of a one-reading window is 1.0, above the
    reading's own position of 0.5, and the block holds wider windows beside
    it, so this is exactly the unfilled-slot case.
    """
    readings = cells(0.0, 1.0, 40)
    values = np.arange(40.0)
    values[10:20] = np.nan
    values[21:] = np.nan
    windows = window_cells(readings, 10 * SECOND)
    got = rolling_quantile(readings, values, windows, [0.05, 0.5])
    lone = got.n_readings == 1
    assert lone.any() and got.n_readings.max() > 1
    # A lone window holds reading 20, or reading 9 seen from across the masked gap.
    assert np.isin(got.values[lone], [9.0, 20.0]).all()
    assert (got.values[lone][:, 0] == got.values[lone][:, 1]).all()
    assert (got.sigma[lone] == 0.0).all()


def test_a_carried_number_of_the_wrong_shape_is_refused() -> None:
    readings = cells(0.0, 1.0, 5)
    with pytest.raises(TsaraBaselineError, match="carry has shape"):
        rolling_quantile(
            readings, np.arange(5.0), window_cells(readings, 10 * SECOND), [0.5], carry=np.ones(4)
        )
