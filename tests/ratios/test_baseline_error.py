"""Tests for the baseline's error as a slow covariance (tsara.ratios.baseline_error, METHODS §5.4).

The evidence, in order of strength:

* closed forms: a random walk's semivariogram is half its diffusion times the
  lag, white noise's is the noise (and is removed), a gas that is another's
  walk times b has a cross-semivariogram b times the walk's;
* the cells inside an event of either gas are left out, so a plume changes
  nothing;
* the wobble, the model's residual and the slow covariance worked out on paper;
* the model is a valid error by construction: for any parameters within its
  bounds and any slope, no member's points can be weighed into a negative
  variance, which a measured curve that dips could do;
* the fit recovers a model it was handed, weighs each gas by its own size, and
  reads only the lags a member at the window can span.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.core.timebase import SECOND_NS as SECOND
from tsara.ratios.baseline_error import (
    _TIMESCALE_MAX_WINDOWS,
    _TIMESCALE_MIN_S,
    _TIMESCALE_STEPS,
    LAGS_S,
    MIN_LAG_PAIRS,
    Wander,
    _fit_model,
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
        reference, numerator, cross = measured.measured[0, 0, :, k]
        assert reference == pytest.approx(expected, rel=0.1)
        # The numerator is the walk times 0.05: its own wander 0.05^2 times, and
        # the cross-semivariogram 0.05 times, the walk's.
        assert numerator == pytest.approx(0.05**2 * expected, rel=0.15)
        assert cross == pytest.approx(0.05 * expected, rel=0.15)
    # The model follows the walk over the lags a 10 min member can span: a long
    # decorrelation time, the straight line's start.
    model = measured.curves(LAGS_S)[0, 0]
    within = LAGS_S <= 600
    np.testing.assert_allclose(model[0, within], 0.5 * step**2 * LAGS_S[within], rtol=0.2, atol=0.2)
    assert measured.timescale[0, 0] > 600


def test_white_noise_wanders_nowhere_once_its_variance_is_removed() -> None:
    rng = np.random.default_rng(4)
    n = 20_000
    measured = _measure(rng.normal(0, 0.7, n), rng.normal(0, 0.05, n), 0.7, 0.05)
    assert measured.described[0, 0]
    # The noise's semivariogram, 0.49 at every lag, is gone to within sampling,
    # and so is the model's.
    assert np.nanmax(measured.measured[0, 0, 0]) < 0.03
    assert np.nanmax(np.abs(measured.measured[0, 0, 2])) < 0.003
    assert np.all(measured.curves(LAGS_S)[0, 0, 0] < 0.03)


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
    assert np.nanmax(plumes.measured[0, 0, 0]) < 0.05
    assert np.nanmax(plumes.measured[0, 0, 1]) < 0.003
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
    assert not measured.nugget.any() and not measured.sill.any()
    assert np.isnan(measured.timescale).all() and np.isnan(measured.measured).all()
    # And the model then says no wander at any lag.
    assert not measured.curves(LAGS_S).any()


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


def _model(
    nugget: tuple[float, float, float], sill: tuple[float, float, float], timescale: float
) -> Wander:
    """A one-sweep-point model, the three curves in the order reference, numerator, cross."""
    return Wander(
        nugget=np.asarray(nugget, dtype=float).reshape(1, 1, 3),
        sill=np.asarray(sill, dtype=float).reshape(1, 1, 3),
        timescale=np.full((1, 1), timescale),
        measured=np.full((1, 1, 3, LAGS_S.size), np.nan),
        described=np.ones((1, 1), bool),
    )


def test_the_models_curves_on_paper() -> None:
    """Zero at a lag of zero, a semivariogram's value by definition; the nugget at once
    beyond it, rising to nugget plus sill with the timescale."""
    model = _model((0.2, 0.01, 0.03), (4.0, 0.02, 0.1), 50.0)
    got = model.curves([0.0, 1e-9, 50.0, -50.0])[0, 0]
    e = 1 - np.exp(-1.0)
    np.testing.assert_array_equal(got[:, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(got[0, 1:], [0.2, 0.2 + 4.0 * e, 0.2 + 4.0 * e], rtol=1e-9)
    np.testing.assert_allclose(got[2, 1:], [0.03, 0.03 + 0.1 * e, 0.03 + 0.1 * e], rtol=1e-9)


def test_the_residuals_curve_on_paper() -> None:
    model = _model((0.2, 0.01, 0.03), (4.0, 0.02, 0.1), 50.0)
    nugget, sill, timescale = model.residual(np.zeros(2, dtype=np.intp), np.array([0.05, -0.5]))
    # n_y + b^2 n_x - 2 b n_xy, and the same for the sills.
    np.testing.assert_allclose(nugget, [0.01 + 0.0025 * 0.2 - 0.1 * 0.03, 0.01 + 0.05 + 0.03])
    np.testing.assert_allclose(sill, [0.02 + 0.0025 * 4.0 - 0.1 * 0.1, 0.02 + 1.0 + 0.1])
    np.testing.assert_array_equal(timescale, [50.0, 50.0])
    # A failed fit's slope is NaN, and so is its curve.
    assert np.isnan(model.residual(np.zeros(1, dtype=np.intp), np.array([np.nan]))[0]).all()


def test_the_slow_covariance_on_paper() -> None:
    model = _model((0.2, 0.01, 0.0), (4.0, 0.02, 0.0), 50.0)
    dt = np.array([0.0, 10.0, -50.0, 900.0])
    one = np.ones(4)
    got = slow_covariance(
        dt,
        amplitude=2.0 * one,
        window_s=600.0 * one,
        slope=0.1 * one,
        sweep=np.zeros(4, dtype=np.intp),
        wandering=model,
    )
    # wobble 2 (1 - |dt| / 600)+ ; the residual's curve, zero at dt = 0, else
    # (0.01 + 0.01 x 0.2) + (0.02 + 0.01 x 4)(1 - exp(-|dt| / 50)).
    curve = np.where(dt == 0, 0.0, 0.012 + 0.06 * (1 - np.exp(-np.abs(dt) / 50.0)))
    expected = 2.0 * np.fmax(0, 1 - np.abs(dt) / 600) - curve
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_a_sweep_point_with_no_lag_read_wanders_nowhere() -> None:
    empty = Wander(
        nugget=np.zeros((1, 1, 3)),
        sill=np.zeros((1, 1, 3)),
        timescale=np.full((1, 1), np.nan),
        measured=np.full((1, 1, 3, LAGS_S.size), np.nan),
        described=np.zeros((1, 1), bool),
    )
    got = slow_covariance(
        np.array([0.0, 30.0]),
        amplitude=np.zeros(2),
        window_s=np.full(2, 600.0),
        slope=np.full(2, 0.05),
        sweep=np.zeros(2, dtype=np.intp),
        wandering=empty,
    )
    np.testing.assert_array_equal(got, [0.0, 0.0])


@pytest.mark.parametrize("seed", range(6))
def test_the_model_is_a_valid_error_for_any_parameters_and_slope(seed: int) -> None:
    """The property the measured curve lacked: weighed by any vector that sums to zero
    (a slope's weights do), the slow covariance of a member's points is never negative.
    Parameters drawn anywhere within the model's bounds, cross at its bound or inside,
    slopes of either sign; irregular times, as a pair clock with holes gives."""
    rng = np.random.default_rng(seed)
    for _ in range(40):
        nx, ny, sx, sy = rng.exponential(1.0, 4) * (rng.random(4) < 0.8)
        nxy = rng.uniform(-1, 1) * np.sqrt(nx * ny)
        sxy = rng.choice([-1.0, 1.0, rng.uniform(-1, 1)]) * np.sqrt(sx * sy)
        model = _model((nx, ny, nxy), (sx, sy, sxy), float(rng.uniform(0.5, 3000)))
        times = np.sort(rng.uniform(0, 400, 60))
        dt = (times[:, None] - times[None, :]).ravel()
        b = float(rng.normal(0, 2))
        S = slow_covariance(
            dt,
            amplitude=np.full(dt.size, float(rng.exponential(1.0))),
            window_s=np.full(dt.size, float(rng.choice([120.0, 600.0, 3600.0]))),
            slope=np.full(dt.size, b),
            sweep=np.zeros(dt.size, dtype=np.intp),
            wandering=model,
        ).reshape(times.size, times.size)
        # Every direction that sums to zero: S projected off the constant.
        centre = np.eye(times.size) - 1.0 / times.size
        lowest = np.linalg.eigvalsh(centre @ S @ centre)[0]
        assert lowest >= -1e-9 * max(1.0, np.abs(S).max())


def test_a_curve_no_background_could_have_gives_a_negative_variance_and_its_model_cannot() -> None:
    """The shape notebook 10 §2's measured curve had: flat from 20 s to 80 s, then six times
    higher by 160 s, so that the background would change more over 152 s than over two
    76 s steps (sqrt g(2L) > 2 sqrt g(L), the triangle inequality broken). Weighed by
    (1, -2, 1) at 0, 76 and 152 s, as a member's slope can be, that curve gives a negative
    variance; the model fitted to it does not. (A dip alone is not the fault: a cycling
    background has a curve that dips and is a valid error.)"""
    window = 600.0
    measured = np.full((3, LAGS_S.size), np.nan)
    measured[0] = np.interp(LAGS_S, [1.0, 20.0, 80.0, 160.0, 7200.0], [0.05, 0.6, 0.6, 3.6, 3.6])
    measured[1] = 0.0025 * measured[0]
    measured[2] = 0.0
    times = np.array([0.0, 76.0, 152.0])
    u = np.array([1.0, -2.0, 1.0])  # sums to zero
    dt = np.abs(times[:, None] - times[None, :])
    as_measured = -np.where(dt == 0, 0.0, np.interp(dt, LAGS_S, measured[0]))
    assert u @ as_measured @ u < 0
    nugget, sill, timescale = _fit_model(measured, window)
    fitted = -_model(tuple(nugget), tuple(sill), timescale).curves(dt.ravel())[0, 0, 0]
    assert u @ fitted.reshape(dt.shape) @ u >= 0


def _grid(window: float) -> np.ndarray:
    grid = np.geomspace(_TIMESCALE_MIN_S, _TIMESCALE_MAX_WINDOWS * window, _TIMESCALE_STEPS)
    return np.asarray(grid, dtype=np.float64)


def _curves(
    nugget: tuple[float, float, float], sill: tuple[float, float, float], timescale: float
) -> np.ndarray:
    curves: np.ndarray = _model(nugget, sill, timescale).curves(LAGS_S)[0, 0]
    return curves


def test_the_fit_recovers_a_model_it_is_handed() -> None:
    window = 600.0
    truth_timescale = float(_grid(window)[40])
    curves = _curves((0.3, 0.004, 0.02), (2.0, 0.01, 0.1), truth_timescale)
    nugget, sill, timescale = _fit_model(curves, window)
    assert timescale == pytest.approx(truth_timescale, rel=1e-12)
    np.testing.assert_allclose(nugget, [0.3, 0.004, 0.02], atol=1e-10)
    np.testing.assert_allclose(sill, [2.0, 0.01, 0.1], rtol=1e-8)


def test_the_cross_is_held_within_what_the_two_gases_allow() -> None:
    """A cross-semivariogram measured larger than the two gases' own allow (as noise in a
    short record can make it) is fitted at the bound, never past it."""
    window = 600.0
    curves = _curves((0.0, 0.0, 0.0), (1.0, 0.01, 0.0), float(_grid(window)[40]))
    curves[2] = 3.0 * np.sqrt(curves[0] * curves[1])
    nugget, sill, _ = _fit_model(curves, window)
    assert abs(sill[2]) <= np.sqrt(sill[0] * sill[1]) * (1 + 1e-12)
    assert abs(nugget[2]) <= np.sqrt(nugget[0] * nugget[1]) * (1 + 1e-12) + 1e-15
    assert sill[2] == pytest.approx(np.sqrt(sill[0] * sill[1]))


def test_each_gas_counts_by_its_own_size_not_its_units() -> None:
    """Two gases whose curves decorrelate at different times share one timescale, a
    compromise; written in units a million times larger, one gas's curve may not pull the
    compromise its way, and its nugget and sill scale with it."""
    window = 600.0
    curves = _curves((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 20.0)
    curves[1] = _curves((0.0, 0.0, 0.0), (0.0, 0.01, 0.0), 300.0)[1]
    nugget, sill, timescale = _fit_model(curves, window)
    curves[0] *= 1e6
    nugget_big, sill_big, timescale_big = _fit_model(curves, window)
    assert timescale_big == timescale
    assert 20.0 < timescale < 300.0
    np.testing.assert_allclose(sill_big[:2], sill[:2] * [1e6, 1.0], rtol=1e-9)
    np.testing.assert_allclose(nugget_big[:2], nugget[:2] * [1e6, 1.0], rtol=1e-9, atol=1e-12)


def test_only_the_lags_a_member_at_the_window_can_span_are_fitted() -> None:
    """A curve that follows a model up to the window and leaps beyond it: the fit at a
    2 min window recovers the model; with fewer than three lags read within the window, it
    reads every lag it has."""
    window = 120.0
    truth_timescale = float(_grid(window)[30])
    curves = _curves((0.1, 0.001, 0.0), (1.0, 0.01, 0.0), truth_timescale)
    curves[:, LAGS_S > window] *= 25.0
    nugget, sill, timescale = _fit_model(curves, window)
    assert timescale == pytest.approx(truth_timescale, rel=1e-12)
    np.testing.assert_allclose(sill[:2], [1.0, 0.01], rtol=1e-8)
    # Read only at two lags inside the window (and every lag beyond): fitted to all.
    sparse = curves.copy()
    inside = np.flatnonzero(LAGS_S <= window)
    sparse[:, inside[2:]] = np.nan
    _, sill_sparse, _ = _fit_model(sparse, window)
    assert sill_sparse[0] > 5.0


def test_a_lag_shorter_than_half_the_pair_clocks_spacing_pairs_no_cell_with_itself() -> None:
    """On a 10 s clock, lags below 5 s have no pairs: never a cell with itself, whose
    difference is zero, so they are not read."""
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
    assert np.isnan(measured.measured[0, 0, :, short]).all()
    # One cell apart, the walk's step of 1 ppb: half its variance.
    assert measured.measured[0, 0, 0, first] == pytest.approx(0.5, rel=0.15)
