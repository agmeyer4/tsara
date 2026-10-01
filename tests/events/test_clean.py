"""Tests for the clean level and spread (tsara.events.clean)."""

from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import norm

from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import (
    MAD_TO_SIGMA,
    TsaraEventError,
    available_clean_level_estimators,
    clean_air,
    describe_clean_air,
    find_records,
    get_clean_level_estimator,
    half_sample_mode,
    quantization_step,
    register_clean_level_estimator,
)

HOUR = 3600 * SECOND

# ---------------------------------------------------------------------------
# The half-sample mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "mode"),
    [
        ([5.0], 5.0),
        ([1.0, 3.0], 2.0),
        ([1.0, 2.0, 4.0], 1.5),  # the closer pair
        ([1.0, 3.0, 4.0], 3.5),
        ([1.0, 2.0, 3.0], 2.0),  # equal gaps: the middle value
        # Six values: the shortest three are 4, 5, 6 (width 2), then equal gaps.
        ([36.0, 0.0, 5.0, 20.0, 4.0, 6.0], 5.0),
        # Two equally short halves, [0, 1] and [10, 11]: the earliest is kept.
        ([0.0, 1.0, 10.0, 11.0], 0.5),
        # A plateau of identical readings more than half the sample wide.
        ([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0], 0.0),
    ],
)
def test_the_half_sample_mode_by_hand(values: list[float], mode: float) -> None:
    assert half_sample_mode(np.array(values)) == mode


def test_no_values_have_no_mode() -> None:
    assert math.isnan(half_sample_mode(np.empty(0)))


def test_the_half_sample_mode_matches_a_slow_reference_from_the_paper() -> None:
    """Bickel and Frühwirth (2006), one interval at a time, on samples with ties,
    steps, heavy tails and every size from 1 to 60."""
    rng = np.random.default_rng(5)
    for _ in range(600):
        n = int(rng.integers(1, 61))
        kind = rng.integers(3)
        if kind == 0:
            x = rng.normal(size=n)
        elif kind == 1:
            x = np.round(rng.normal(size=n) * 2) / 2  # written in steps: many ties
        else:
            x = rng.normal(size=n) + (rng.random(n) < 0.4) * rng.exponential(5.0, n)
        assert half_sample_mode(x) == _slow_half_sample_mode(list(x))


def _slow_half_sample_mode(values: list[float]) -> float:
    x = sorted(values)
    while len(x) > 3:
        half = -(-len(x) // 2)
        best, best_width = 0, math.inf
        for j in range(len(x) - half + 1):
            width = x[j + half - 1] - x[j]
            if width < best_width:
                best, best_width = j, width
        x = x[best : best + half]
    if len(x) == 1:
        return x[0]
    if len(x) == 2:
        return float(np.mean(x))
    if x[1] - x[0] < x[2] - x[1]:
        return float(np.mean(x[:2]))
    if x[1] - x[0] > x[2] - x[1]:
        return float(np.mean(x[1:]))
    return x[1]


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------


def test_level_spread_and_count_by_hand() -> None:
    """Sorted: -2, -1, 0, 0, 0, 5, 9. The half-sample mode narrows to [-1, 0, 0, 0]
    and then to a pair of zeros, so the level is 0; below it lie -2 and -1, at
    distances 2 and 1, whose median is 1.5."""
    found = describe_clean_air(np.array([0.0, 9.0, -1.0, 0.0, 5.0, -2.0, 0.0]))
    assert found.level == 0.0
    assert found.n_below == 2
    assert found.spread == pytest.approx(MAD_TO_SIGMA * 1.5, abs=1e-15)
    assert not found.floored


def test_the_floor_holds_a_spread_that_would_fall_below_a_rounding_step() -> None:
    """Three readings a hundredth below a plateau: measured, the spread is 0.0148,
    far below what readings written in steps of 1 can resolve (1/sqrt 12)."""
    values = np.array([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    free = describe_clean_air(values)
    held = describe_clean_air(values, floor=1 / math.sqrt(12))
    assert free.spread == pytest.approx(MAD_TO_SIGMA * 0.01, abs=1e-15)
    assert held.spread == 1 / math.sqrt(12)
    assert held.floored and not free.floored
    assert held.level == free.level == 0.0


def test_nothing_below_the_level_leaves_no_spread_and_non_finite_values_are_ignored() -> None:
    found = describe_clean_air(np.array([1.0, np.nan, 1.0, np.inf, 1.0]))
    assert found.level == 1.0
    assert found.n_below == 0
    assert math.isnan(found.spread)
    empty = describe_clean_air(np.array([np.nan]))
    assert math.isnan(empty.level) and empty.n_below == 0


def test_on_plume_free_gaussian_air_the_spread_is_the_noise() -> None:
    """Closed form: 1.4826 times the median of a half-normal is sigma.

    Rule: N(0, 0.7), no plumes, 21 600 readings (6 h at 1 s), seeds 0-99.
    Measured over 300 seeds the spread is 0.999 sigma on average with a
    record-to-record scatter of 8 %, and the level -0.008 sigma with 0.13
    (METHODS §6.8, the half-sample mode's jitter); a 100-seed mean is held to
    about four standard errors of either.
    """
    level, spread = [], []
    for seed in range(100):
        x = np.random.default_rng(seed).normal(0.0, 0.7, 21600)
        found = describe_clean_air(x)
        level.append(found.level / 0.7)
        spread.append(found.spread / 0.7)
    assert abs(np.mean(spread) - 1.0) < 0.03
    assert abs(np.mean(level)) < 0.055


def test_the_minimum_count_table_of_methods_reproduces() -> None:
    """METHODS §6.8's table of spread and chance crossing against the count below.

    Its rule: one generator seeded 11 runs through the counts 10, 15, 20, 30,
    50, 100, 300 and 1000 in that order, drawing round(n / 0.35) readings
    4000 times per count as N(0, 1) plus, with probability 0.3, an exponential
    excess of mean 5 (normal, then the uniform, then the exponential). The
    spread's truth is 1; a chance crossing is the probability that one
    plume-free reading exceeds m + 3 s, over the nominal 0.00135.
    """
    table = {
        30: ("0.62-1.49", "0.00-58"),
        100: ("0.74-1.36", "0.00-21"),
        300: ("0.81-1.27", "0.01-11"),
        1000: ("0.86-1.20", "0.03-6"),
    }
    nominal = norm.sf(3.0)
    rng = np.random.default_rng(11)
    for n in (10, 15, 20, 30, 50, 100, 300, 1000):
        total = round(n / 0.35)
        spread, chance = [], []
        for _ in range(4000):
            x = rng.normal(size=total) + (rng.random(total) < 0.3) * rng.exponential(5.0, total)
            if n in table:
                found = describe_clean_air(x)
                spread.append(found.spread)
                chance.append(norm.sf(found.level + 3 * found.spread) / nominal)
        if n in table:
            s10, s90 = np.nanpercentile(spread, [10, 90])
            c10, c90 = np.nanpercentile(chance, [10, 90])
            assert (f"{s10:.2f}-{s90:.2f}", f"{c10:.2f}-{c90:.0f}") == table[n], n


# ---------------------------------------------------------------------------
# The quantization step
# ---------------------------------------------------------------------------


def test_the_declared_step_wins_and_otherwise_the_smallest_gap_is_the_step() -> None:
    stepped = np.array([1900.0, 1900.5, np.nan, 1901.0, 1900.5])
    assert quantization_step(stepped) == 0.5
    assert quantization_step(stepped, declared=0.25) == 0.25
    assert quantization_step(np.array([3.0, 3.0, np.nan])) == 0.0
    # Steps of 0.5 with values missing between them: the gaps are 0.5, 1.5 and
    # 1.0, and the step is the smallest, not a typical one.
    assert quantization_step(np.array([0.0, 0.5, 2.0, 3.0])) == 0.5


# ---------------------------------------------------------------------------
# A whole variable
# ---------------------------------------------------------------------------


def two_records(n_each: int) -> tuple[CellBounds, npt.NDArray[np.int64]]:
    """Two records of ``n_each`` 1 s readings, three hours apart."""
    first = np.arange(n_each, dtype=np.int64) * SECOND
    mids = np.concatenate([first, first + 3 * HOUR])
    return CellBounds(start_ns=mids - SECOND // 2, stop_ns=mids + SECOND // 2), mids


def test_every_record_and_sweep_point_is_described_on_its_own() -> None:
    """Two records, a sweep of two windows by one quantile. Record 0 is quiet air
    around 0; record 1 is quiet air around 10 at the first window and around 20
    at the second, so a level that leaked between records or points would show."""
    cells, _ = two_records(400)
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, 800)
    enhancement = np.empty((800, 2, 1))
    enhancement[:, 0, 0] = noise + np.where(np.arange(800) < 400, 0.0, 10.0)
    enhancement[:, 1, 0] = noise + np.where(np.arange(800) < 400, 0.0, 20.0)
    records = find_records(cells, np.ones(800, dtype=bool), gap_ns=2 * HOUR, max_length_ns=HOUR)
    air = clean_air(
        enhancement,
        1900.0 + noise,
        records,
        estimator="half_sample_mode",
        min_clean_readings=100,
    )
    assert air.level.shape == air.spread.shape == air.blank.shape == (2, 2, 1)
    for r, point, offset in [(0, 0, 0.0), (0, 1, 0.0), (1, 0, 10.0), (1, 1, 20.0)]:
        alone = describe_clean_air(enhancement[records.index == r, point, 0])
        assert air.level[r, point, 0] == alone.level
        assert air.spread[r, point, 0] == alone.spread
        assert abs(air.level[r, point, 0] - offset) < 0.5
    assert not air.blank.any()
    assert air.min_clean_readings == 100


def test_a_record_with_too_few_readings_below_its_level_is_blank_and_keeps_its_count() -> None:
    """Record 0 has 400 readings, about 200 below its level; record 1 has 30."""
    cells = two_records(400)[0]
    keep = np.r_[np.ones(400, dtype=bool), np.arange(400) < 30]
    values = np.where(keep, np.random.default_rng(2).normal(size=800), np.nan)
    records = find_records(cells, keep, gap_ns=2 * HOUR, max_length_ns=HOUR)
    air = clean_air(
        values[:, None], values, records, estimator="half_sample_mode", min_clean_readings=100
    )
    assert air.blank[:, 0].tolist() == [False, True]
    assert np.isnan(air.level[1, 0]) and np.isnan(air.spread[1, 0])
    assert 0 < air.n_below[1, 0] < 100
    assert air.n_below[0, 0] >= 100


def test_each_record_takes_its_own_floor_from_its_own_readings() -> None:
    """Record 0 is written in steps of 0.5 and record 1 in steps of 2, unless a
    step is declared, which then applies to both."""
    cells = two_records(300)[0]
    rng = np.random.default_rng(4)
    readings = np.r_[np.round(rng.normal(size=300) * 2) / 2, np.round(rng.normal(size=300) / 2) * 2]
    records = find_records(cells, np.ones(600, dtype=bool), gap_ns=2 * HOUR, max_length_ns=HOUR)
    measured = clean_air(
        readings[:, None], readings, records, estimator="half_sample_mode", min_clean_readings=1
    )
    declared = clean_air(
        readings[:, None],
        readings,
        records,
        estimator="half_sample_mode",
        min_clean_readings=1,
        quantization=0.1,
    )
    assert measured.floor.tolist() == [0.5 / math.sqrt(12), 2 / math.sqrt(12)]
    assert declared.floor.tolist() == [0.1 / math.sqrt(12)] * 2


def test_a_blank_record_reports_no_floor_action() -> None:
    """The floor acted on what was measured, but a blank record reports nothing."""
    cells = CellBounds(
        start_ns=np.arange(9, dtype=np.int64) * SECOND,
        stop_ns=np.arange(1, 10, dtype=np.int64) * SECOND,
    )
    values = np.array([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    records = find_records(cells, np.ones(9, dtype=bool), gap_ns=HOUR, max_length_ns=HOUR)
    kept = clean_air(
        values[:, None],
        values,
        records,
        estimator="half_sample_mode",
        min_clean_readings=3,
        quantization=1.0,
    )
    blank = clean_air(
        values[:, None],
        values,
        records,
        estimator="half_sample_mode",
        min_clean_readings=4,
        quantization=1.0,
    )
    assert kept.floored[0, 0] and not kept.blank[0, 0]
    assert blank.blank[0, 0] and not blank.floored[0, 0]
    assert blank.n_below[0, 0] == 3


def test_what_does_not_fit_or_names_no_estimator_is_refused() -> None:
    cells = two_records(10)[0]
    records = find_records(cells, np.ones(20, dtype=bool), gap_ns=2 * HOUR, max_length_ns=HOUR)
    with pytest.raises(TsaraEventError, match="do not match"):
        clean_air(
            np.zeros((19, 1)),
            np.zeros(20),
            records,
            estimator="half_sample_mode",
            min_clean_readings=1,
        )
    with pytest.raises(TsaraEventError, match="do not match"):
        clean_air(
            np.zeros((20, 1)),
            np.zeros(19),
            records,
            estimator="half_sample_mode",
            min_clean_readings=1,
        )
    with pytest.raises(TsaraEventError, match="No clean-level estimator registered as 'shorth'"):
        clean_air(
            np.zeros((20, 1)), np.zeros(20), records, estimator="shorth", min_clean_readings=1
        )
    # Refused before any work, so a variable with nothing to describe cannot
    # carry a misspelt estimator through unnoticed.
    none = find_records(cells, np.zeros(20, dtype=bool), gap_ns=2 * HOUR, max_length_ns=HOUR)
    with pytest.raises(TsaraEventError, match="'shorth'"):
        clean_air(np.zeros((20, 1)), np.zeros(20), none, estimator="shorth", min_clean_readings=1)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_half_sample_mode_is_registered_under_its_configuration_name() -> None:
    assert available_clean_level_estimators() == ("half_sample_mode",)
    assert get_clean_level_estimator("half_sample_mode") is half_sample_mode


def test_a_name_is_registered_once_unless_replaced_on_purpose(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_clean_level_estimator("half_sample_mode")(half_sample_mode)
    with pytest.raises(ValueError, match="non-empty"):
        register_clean_level_estimator("  ")
    with caplog.at_level(logging.WARNING, logger="tsara.events.clean"):
        register_clean_level_estimator("half_sample_mode", replace=True)(half_sample_mode)
    assert "Replacing clean-level estimator 'half_sample_mode'" in caplog.text
    assert get_clean_level_estimator("half_sample_mode") is half_sample_mode
