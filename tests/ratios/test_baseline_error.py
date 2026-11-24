"""Tests for the baseline's error as a slow covariance (tsara.ratios.baseline_error, METHODS §5.4).

The evidence, in order of strength:

* closed forms: a random walk's semivariogram is half its diffusion times the
  lag, white noise's is the noise (and is removed), a gas that is another's
  walk times b has a cross-semivariogram b times the walk's;
* the cells inside an event of either gas are left out, so a plume changes
  nothing;
* the wobble and the slow covariance worked out on paper;
* a semivariogram that dips, as one read from a record's fragments between
  its events can, is stored as its non-decreasing fit, and the
  cross-semivariogram never exceeds what the two gases' allow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.core.timebase import SECOND_NS as SECOND
from tsara.ratios.baseline_error import (
    LAGS_S,
    MIN_LAG_PAIRS,
    Wander,
    _filled,
    _non_decreasing,
    _semivariograms,
    slow_covariance,
    wander,
    wobble,
)

VARIABLES = ("van.c2h6", "van.ch4")


def _pairs(
    n: int, windows: tuple[float, ...] = (600.0,), quantiles: tuple[float, ...] = (0.05,)
) -> xr.Dataset:
    """An empty paired product of n abutting 1 s cells: only its sweep coordinates are read."""
    start = np.arange(n, dtype=np.int64) * SECOND
    return xr.Dataset(
        coords={
            "time": (start + SECOND // 2).astype("datetime64[ns]"),
            "baseline_window": np.asarray(windows),
            "baseline_quantile": np.asarray(quantiles),
        }
    )


def _swept(values: np.ndarray) -> np.ndarray:
    return values[:, None, None]


def _no_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "instrument",
            "species",
            "baseline_window",
            "baseline_quantile",
            "enter_multiple",
            "exit_multiple",
            "start_time",
            "end_time",
        ]
    )


def _measure(
    x: np.ndarray, y: np.ndarray, sx: float, sy: float, catalog: pd.DataFrame | None = None
) -> Wander:
    n = x.size
    pairs = _pairs(n)
    midpoints = pairs.time.values.astype("datetime64[ns]").astype(np.int64)
    return wander(
        pairs,
        _no_events() if catalog is None else catalog,
        (_swept(y), _swept(x)),
        (np.full(n, sy), np.full(n, sx)),
        midpoints,
        VARIABLES,
    )


def _lag(value: float) -> int:
    return int(np.argmin(np.abs(LAGS_S - value)))


def test_a_random_walks_semivariogram_is_half_its_diffusion_times_the_lag() -> None:
    rng = np.random.default_rng(3)
    n, step, noise = 40_000, 0.3, 0.7
    walk = np.cumsum(rng.normal(0, step, n))
    x = walk + rng.normal(0, noise, n)
    y = 0.05 * walk + rng.normal(0, 0.05, n)
    measured = _measure(x, y, noise, 0.05)
    assert measured.described[0, 0]
    for lag in (10.0, 30.0, 100.0):
        k = _lag(lag)
        expected = 0.5 * step**2 * LAGS_S[k]
        assert measured.reference[0, 0, k] == pytest.approx(expected, rel=0.1)
        # The numerator is the walk times 0.05: its own wander 0.05^2 times, and
        # the cross-semivariogram 0.05 times, the walk's.
        assert measured.numerator[0, 0, k] == pytest.approx(0.05**2 * expected, rel=0.15)
        assert measured.cross[0, 0, k] == pytest.approx(0.05 * expected, rel=0.15)


def test_white_noise_wanders_nowhere_once_its_variance_is_removed() -> None:
    rng = np.random.default_rng(4)
    n = 20_000
    measured = _measure(rng.normal(0, 0.7, n), rng.normal(0, 0.05, n), 0.7, 0.05)
    assert measured.described[0, 0]
    # The noise's semivariogram, 0.49 at every lag, is gone to within sampling.
    assert np.all(measured.reference[0, 0] < 0.03)
    assert np.all(np.abs(measured.cross[0, 0]) < 0.003)


def test_cells_inside_an_event_of_either_gas_are_left_out() -> None:
    rng = np.random.default_rng(5)
    n = 6_000
    x, y = rng.normal(0, 0.7, n), rng.normal(0, 0.05, n)
    quiet = _measure(x, y, 0.7, 0.05)
    # A tall plume, on the numerator only, inside one of the numerator's events.
    x2, y2 = x.copy(), y.copy()
    y2[2000:2600] += 40.0
    x2[4000:4300] += 300.0
    catalog = pd.DataFrame(
        [
            {
                "instrument": "van",
                "species": species,
                "baseline_window": 600.0,
                "baseline_quantile": 0.05,
                "enter_multiple": entry,
                "exit_multiple": 1.0,
                "start_time": pd.Timestamp(0) + pd.Timedelta(seconds=start),
                "end_time": pd.Timestamp(0) + pd.Timedelta(seconds=end),
            }
            for species, start, end, entry in (
                ("c2h6", 2000, 2600, 3.0),
                ("c2h6", 2100, 2200, 5.0),
                ("ch4", 4000, 4300, 3.0),
            )
        ]
    )
    plumes = _measure(x2, y2, 0.7, 0.05, catalog)
    # Only the cells between the events are read in both, so the plumes change
    # nothing; the quiet record reads every cell, so its numbers differ a little.
    inside_free = _measure(
        np.delete(x2, np.r_[2000:2600, 4000:4300]),
        np.delete(y2, np.r_[2000:2600, 4000:4300]),
        0.7,
        0.05,
    )
    assert np.all(plumes.reference[0, 0] < 0.05)
    assert np.all(plumes.numerator[0, 0] < 0.003)
    assert quiet.described[0, 0] and plumes.described[0, 0] and inside_free.described[0, 0]


def test_without_the_readings_sigma_no_lag_is_described() -> None:
    n = 2_000
    pairs = _pairs(n)
    midpoints = pairs.time.values.astype("datetime64[ns]").astype(np.int64)
    values = _swept(np.zeros(n))
    measured = wander(
        pairs, _no_events(), (values, values), (None, np.ones(n)), midpoints, VARIABLES
    )
    assert not measured.described.any()
    assert not measured.reference.any()


def test_a_record_too_short_for_any_lag_describes_none() -> None:
    measured = _measure(np.zeros(MIN_LAG_PAIRS), np.zeros(MIN_LAG_PAIRS), 0.7, 0.05)
    assert not measured.described.any()
    single = _measure(np.zeros(1), np.zeros(1), 0.7, 0.05)
    assert not single.described.any()


def test_the_wobble_is_the_clean_spread_beyond_the_readings_noise() -> None:
    pairs = _pairs(3).assign(
        clean_spread_ch4=(
            ("time", "baseline_window", "baseline_quantile"),
            np.array([1.0, 0.5, np.nan])[:, None, None],
        )
    )
    sigma = np.array([0.6, 0.7, 0.6])
    got = wobble(pairs, "clean_spread_ch4", sigma)
    assert got is not None
    np.testing.assert_array_equal(got[:, 0, 0], [1.0 - 0.36, 0.0, np.nan])
    assert wobble(pairs, None, sigma) is None
    assert wobble(pairs, "clean_spread_ch4", None) is None


def test_the_slow_covariance_on_paper() -> None:
    lag = np.zeros((1, 1, LAGS_S.size))
    table = lag + 0.5 * LAGS_S  # g(L) = L / 2 for each gas
    flat = Wander(
        reference=table,
        numerator=table,
        cross=np.zeros_like(table),
        described=np.ones((1, 1), bool),
    )
    # On the lags themselves, where reading the table in log lag is exact.
    dt = np.array([0.0, LAGS_S[20], LAGS_S[29], LAGS_S[33]])
    one = np.ones(4)
    got = slow_covariance(
        dt,
        amplitude=2.0 * one,
        window_s=600.0 * one,
        slope=0.1 * one,
        sweep=np.zeros(4, dtype=np.intp),
        wandering=flat,
    )
    # wobble 2 (1 - |dt| / 600)+ ; wander (1 + 0.01) dt / 2 ; cross 0
    expected = 2.0 * np.fmax(0, 1 - dt / 600) - 1.01 * 0.5 * dt
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_a_wander_the_cross_term_cancels_is_clamped_at_zero_not_turned_negative() -> None:
    table = np.zeros((1, 1, LAGS_S.size)) + 1.0
    flat = Wander(
        reference=table, numerator=table, cross=table * 10.0, described=np.ones((1, 1), bool)
    )
    got = slow_covariance(
        np.array([30.0]),
        amplitude=np.zeros(1),
        window_s=np.array([600.0]),
        slope=np.ones(1),
        sweep=np.zeros(1, dtype=np.intp),
        wandering=flat,
    )
    # 1 + 1 - 2 x 10 < 0: no wander, not a negative one.
    assert got[0] == 0.0


def test_a_lag_shorter_than_half_the_pair_clocks_spacing_pairs_no_cell_with_itself() -> None:
    """On a 10 s clock, lags below 5 s have no pairs: never a cell with itself, whose
    difference is zero, so they take the first described lag's value (held at the end)."""
    rng = np.random.default_rng(6)
    n = 6_000
    walk = np.cumsum(rng.normal(0, 1.0, n))
    start = np.arange(n, dtype=np.int64) * 10 * SECOND
    pairs = xr.Dataset(
        coords={
            "time": (start + 5 * SECOND).astype("datetime64[ns]"),
            "baseline_window": np.asarray([600.0]),
            "baseline_quantile": np.asarray([0.05]),
        }
    )
    midpoints = pairs.time.values.astype("datetime64[ns]").astype(np.int64)
    x = walk + rng.normal(0, 0.7, n)
    measured = wander(
        pairs,
        _no_events(),
        (_swept(0.05 * walk), _swept(x)),
        (np.full(n, 0.05), np.full(n, 0.7)),
        midpoints,
        VARIABLES,
    )
    short = LAGS_S < 5.0
    first = int(np.argmax(~short))
    assert measured.described[0, 0]
    np.testing.assert_array_equal(measured.reference[0, 0, short], measured.reference[0, 0, first])
    # One cell apart, the walk's step of 1 ppb: half its variance.
    assert measured.reference[0, 0, first] == pytest.approx(0.5, rel=0.15)


def test_the_non_decreasing_fit_pools_a_dip_into_its_neighbours() -> None:
    # Worked by hand: 3, 2, 2 pool to 7/3; 5, 4 pool to 4.5; the rest already rise.
    got = _non_decreasing(np.array([0.0, 1.0, 3.0, 2.0, 2.0, 5.0, 4.0, 6.0]))
    np.testing.assert_allclose(got, [0.0, 1.0, 7 / 3, 7 / 3, 7 / 3, 4.5, 4.5, 6.0])
    rising = np.array([0.0, 0.5, 0.5, 2.0])
    np.testing.assert_array_equal(_non_decreasing(rising), rising)
    # Falling throughout pools into one block, the mean.
    np.testing.assert_allclose(_non_decreasing(np.array([4.0, 2.0, 0.0])), [2.0, 2.0, 2.0])


def test_a_semivariogram_that_dips_is_stored_as_its_non_decreasing_fit() -> None:
    """A background with a 120 s cycle: its semivariogram peaks at 60 s and dips at 120 s,
    as a measured one can; the stored wander is the non-decreasing fit to the measured."""
    rng = np.random.default_rng(7)
    n = 20_000
    seconds = np.arange(n, dtype=np.float64)
    cycle = 3.0 * np.sin(2 * np.pi * seconds / 120.0)
    x = cycle + rng.normal(0, 0.7, n)
    y = 0.05 * cycle + rng.normal(0, 0.05, n)
    measured = _measure(x, y, 0.7, 0.05)
    raw = _filled(_semivariograms(seconds + 0.5, (x, y), (np.full(n, 0.7), np.full(n, 0.05)))[0])
    peak, dip = _lag(60.0), _lag(120.0)
    assert raw[dip] < 0.5 * raw[peak]  # the raw curve dips
    np.testing.assert_allclose(measured.reference[0, 0], _non_decreasing(raw))
    assert np.all(np.diff(measured.reference[0, 0]) >= 0)
    assert np.all(np.diff(measured.numerator[0, 0]) >= 0)


def test_the_cross_semivariogram_never_exceeds_what_the_two_gases_allow() -> None:
    """The reference declared noisier than it is: its wander, the noise removed, is zero at
    every lag, so whatever the two gases seem to share is clipped to zero; and on a gas
    that is the other's walk times b the bound is met, not cut."""
    rng = np.random.default_rng(8)
    n = 20_000
    walk = np.cumsum(rng.normal(0, 0.3, n))
    measured = _measure(rng.normal(0, 0.5, n) + 0.0 * walk, 0.05 * walk, 0.9, 0.05)
    assert np.all(measured.reference[0, 0] == 0.0)
    assert np.all(measured.cross[0, 0] == 0.0)
    both = _measure(walk + rng.normal(0, 0.7, n), 0.05 * walk + rng.normal(0, 0.05, n), 0.7, 0.05)
    bound = np.sqrt(both.reference[0, 0] * both.numerator[0, 0])
    assert np.all(np.abs(both.cross[0, 0]) <= bound + 1e-12)
    k = _lag(100.0)
    assert both.cross[0, 0, k] == pytest.approx(0.05 * 0.5 * 0.3**2 * LAGS_S[k], rel=0.15)
