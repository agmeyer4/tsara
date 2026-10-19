"""Tests for the line fits (tsara.ratios.estimators).

The evidence is held to the five kinds of METHODS §11.1. A published
benchmark (Pearson's data with York's weights, whose answer York et al. 2004
and the independent R package bfsl both print); fixtures checkable with a
pencil; closed forms York must reduce to (Deming's line for equal sigmas,
weighted least squares when x is exact, the reciprocal slope when the axes
swap); independent reimplementations (a loop written from the paper, a
brute-force minimization of York's objective, statsmodels for least
squares); and Monte Carlo on draws written from the textbook rather than
from TSARA's generator, which is what shows the formulas are the right
ones rather than merely typed correctly.
"""

from __future__ import annotations

import logging
import typing

import numpy as np
import numpy.typing as npt
import pytest
import statsmodels.api as sm  # type: ignore[import-untyped]
from scipy.optimize import minimize_scalar

from tsara.config.analysis import RegressionConfig
from tsara.ratios import (
    MIN_FIT_POINTS,
    FitStatus,
    LineFit,
    TsaraRatioError,
    available_line_fits,
    get_line_fit,
    ols,
    register_line_fit,
    york,
)

Floats = npt.NDArray[np.float64]

#: Pearson (1901) with the weights York (1966) gave them: the standard test
#: of a York implementation.
PEARSON_X = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
PEARSON_Y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
PEARSON_WX = np.array([1000.0, 1000.0, 500.0, 800.0, 200.0, 80.0, 60.0, 20.0, 1.8, 1.0])
PEARSON_WY = np.array([1.0, 1.8, 4.0, 8.0, 20.0, 20.0, 70.0, 70.0, 100.0, 500.0])


def pearson() -> LineFit:
    """York's line through the Pearson-York data."""
    return york(PEARSON_X, PEARSON_Y, 1 / np.sqrt(PEARSON_WX), 1 / np.sqrt(PEARSON_WY))


def noisy_line(
    seed: int, n: int = 40, slope: float = 0.05, sx: float = 1.0, sy: float = 0.2
) -> tuple[Floats, Floats]:
    """Points about a known line with Gaussian noise on both axes."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.0, 50.0, n)
    return truth + rng.normal(0.0, sx, n), 3.0 + slope * truth + rng.normal(0.0, sy, n)


# ---------------------------------------------------------------------------
# The published benchmark
# ---------------------------------------------------------------------------


def test_york_reproduces_the_pearson_york_benchmark() -> None:
    """York et al. (2004); bfsl prints 5.47991 ± 0.29497, -0.48053 ± 0.05799, χ² 11.87."""
    fit = pearson()
    assert fit.status[0] == FitStatus.OK
    assert fit.intercept[0] == pytest.approx(5.47991, abs=5e-6)
    assert fit.slope[0] == pytest.approx(-0.48053, abs=5e-6)
    assert fit.intercept_sigma[0] == pytest.approx(0.29497, abs=5e-6)
    assert fit.slope_sigma[0] == pytest.approx(0.05799, abs=5e-6)
    # The goodness of fit: χ² = MSWD × (n − 2) on 8 degrees of freedom.
    assert fit.mswd[0] * 8 == pytest.approx(11.87, abs=5e-3)
    assert fit.n[0] == 10
    assert fit.method == "york"


# ---------------------------------------------------------------------------
# Pencil fixtures
# ---------------------------------------------------------------------------


def test_points_on_a_line_give_that_line_and_no_scatter() -> None:
    """(0, 1), (1, 3), (2, 5) lie on y = 1 + 2x: both fits find it exactly."""
    x, y = np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])
    for fit in (york(x, y, 0.1, 0.1), ols(x, y)):
        assert fit.slope[0] == pytest.approx(2.0, rel=1e-13)
        assert fit.intercept[0] == pytest.approx(1.0, rel=1e-13)
        assert fit.r_squared[0] == pytest.approx(1.0, rel=1e-13)
    assert york(x, y, 0.1, 0.1).mswd[0] == pytest.approx(0.0, abs=1e-20)


def test_least_squares_on_three_points_by_hand() -> None:
    """(0, 0), (1, 1), (2, 4): x̄ 1, Sxx 2, Sxy 4, residuals ⅓, −⅔, ⅓.

    So slope 2, intercept −⅓, s² = (⅔)/1, σ_slope = √(s²/Sxx) = √⅓,
    σ_intercept = √(s²(1/3 + 1/2)) = √(5/9), r² = 16 / (2 · 26/3) = 12/13.
    """
    fit = ols([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    assert fit.slope[0] == pytest.approx(2.0)
    assert fit.intercept[0] == pytest.approx(-1 / 3)
    assert fit.slope_sigma[0] == pytest.approx(np.sqrt(1 / 3))
    assert fit.intercept_sigma[0] == pytest.approx(np.sqrt(5 / 9))
    assert fit.r_squared[0] == pytest.approx(12 / 13)
    assert np.isnan(fit.mswd[0])
    assert fit.iterations[0] == 0
    assert fit.method == "ols"


# ---------------------------------------------------------------------------
# Closed forms York must reduce to
# ---------------------------------------------------------------------------


def test_equal_sigmas_give_demings_line() -> None:
    """With one σx and one σy for every point, York is Deming's line, λ = σy²/σx²."""
    x, y = noisy_line(1)
    lam = 0.2**2 / 1.0**2
    sxx, syy = np.var(x), np.var(y)
    sxy = np.cov(x, y, bias=True)[0, 1]
    deming = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy**2)) / (2 * sxy)
    assert york(x, y, 1.0, 0.2).slope[0] == pytest.approx(deming, rel=1e-11)


def test_an_exact_x_gives_weighted_least_squares() -> None:
    """σx = 0 everywhere: the weight is 1/σy², and York is weighted least squares of y on x."""
    x, y = noisy_line(2)
    sigma_y = np.linspace(0.1, 0.5, x.size)
    fit = york(x, y, 0.0, sigma_y)
    w = 1 / sigma_y**2
    x_bar = np.sum(w * x) / np.sum(w)
    slope = np.sum(w * (x - x_bar) * y) / np.sum(w * (x - x_bar) ** 2)
    assert fit.slope[0] == pytest.approx(slope, rel=1e-12)
    assert fit.intercept[0] == pytest.approx(np.sum(w * y) / np.sum(w) - slope * x_bar, rel=1e-12)
    assert fit.slope_sigma[0] == pytest.approx(1 / np.sqrt(np.sum(w * (x - x_bar) ** 2)), rel=1e-12)


def test_york_starts_from_least_squares() -> None:
    """An exact x and one σy everywhere: least squares is York's answer, so one step settles it.

    Started anywhere else, the first step would move the slope and a second
    would be needed to see it stop.
    """
    x, y = noisy_line(12)
    fit = york(x, y, 0.0, 0.2)
    assert fit.iterations[0] == 1
    assert fit.slope[0] == pytest.approx(ols(x, y).slope[0], rel=1e-12)


def test_swapping_the_axes_gives_the_reciprocal_slope() -> None:
    """York treats the axes alike, so x on y is 1/b with σ = σ_b / b²; least squares does not."""
    x, y = noisy_line(3, slope=0.5)
    forward, backward = york(x, y, 1.0, 0.2), york(y, x, 0.2, 1.0)
    assert backward.slope[0] == pytest.approx(1 / forward.slope[0], rel=1e-10)
    assert backward.slope_sigma[0] == pytest.approx(
        forward.slope_sigma[0] / forward.slope[0] ** 2, rel=1e-8
    )
    assert ols(y, x).slope[0] != pytest.approx(1 / ols(x, y).slope[0], rel=1e-3)


def test_a_common_scale_on_every_sigma_cancels_in_the_scaled_slope_sigma() -> None:
    """σ on both axes times k: the slope is unchanged, the MSWD divides by k², σ_b·√MSWD stays.

    Why the stage that combines uncertainties (§5) needs only the ratio of
    the two axes' precisions if it widens σ_b by √MSWD.
    """
    x, y = noisy_line(4)
    base = york(x, y, 1.0, 0.2)
    for k in (0.1, 3.0):
        scaled = york(x, y, k * 1.0, k * 0.2)
        assert scaled.slope[0] == pytest.approx(base.slope[0], rel=1e-11)
        assert scaled.mswd[0] == pytest.approx(base.mswd[0] / k**2, rel=1e-10)
        assert scaled.slope_sigma[0] * np.sqrt(scaled.mswd[0]) == pytest.approx(
            base.slope_sigma[0] * np.sqrt(base.mswd[0]), rel=1e-10
        )


def test_an_offset_on_either_axis_moves_only_the_intercept() -> None:
    """Methane near 2000 ppb fits as well as its enhancement: the sums are centred."""
    x, y = noisy_line(5)
    near, far = york(x, y, 1.0, 0.2), york(x + 2000.0, y + 400.0, 1.0, 0.2)
    assert far.slope[0] == pytest.approx(near.slope[0], rel=1e-9)
    assert far.slope_sigma[0] == pytest.approx(near.slope_sigma[0], rel=1e-9)
    ols_near, ols_far = ols(x, y), ols(x + 2000.0, y + 400.0)
    assert ols_far.slope[0] == pytest.approx(ols_near.slope[0], rel=1e-10)


# ---------------------------------------------------------------------------
# Independent reimplementations
# ---------------------------------------------------------------------------


def york_by_hand(
    x: Floats, y: Floats, sx: Floats, sy: Floats, r: float = 0.0
) -> tuple[float, float, float, float]:
    """York et al. (2004), written out from the paper in York's own ω and α.

    One group, a plain loop, no shared code with the package: the slope,
    the intercept, and their sigmas.
    """
    wx, wy = 1 / sx**2, 1 / sy**2
    alpha = np.sqrt(wx * wy)
    b = float(np.polyfit(x, y, 1)[0])
    for _ in range(10_000):
        w = wx * wy / (wx + b * b * wy - 2 * b * r * alpha)
        x_bar, y_bar = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
        u, v = x - x_bar, y - y_bar
        beta = w * (u / wy + b * v / wx - (b * u + v) * r / alpha)
        new = float(np.sum(w * beta * v) / np.sum(w * beta * u))
        if abs(new - b) <= 1e-15 * abs(new):
            b = new
            break
        b = new
    w = wx * wy / (wx + b * b * wy - 2 * b * r * alpha)
    x_bar, y_bar = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
    u, v = x - x_bar, y - y_bar
    beta = w * (u / wy + b * v / wx - (b * u + v) * r / alpha)
    a = float(y_bar - b * x_bar)
    adjusted = x_bar + beta
    mean_adjusted = np.sum(w * adjusted) / np.sum(w)
    sigma_b = float(1 / np.sqrt(np.sum(w * (adjusted - mean_adjusted) ** 2)))
    sigma_a = float(np.sqrt(1 / np.sum(w) + mean_adjusted**2 * sigma_b**2))
    return b, a, sigma_b, sigma_a


@pytest.mark.parametrize("r", [0.0, 0.6, -0.4])
def test_york_agrees_with_the_paper_written_out_by_hand(r: float) -> None:
    """Per-point sigmas that vary, with and without a correlation between the axes."""
    rng = np.random.default_rng(6)
    x, y = noisy_line(6, n=25)
    sx, sy = rng.uniform(0.5, 1.5, x.size), rng.uniform(0.1, 0.3, x.size)
    fit = york(x, y, sx, sy, r=r)
    b, a, sigma_b, sigma_a = york_by_hand(x, y, sx, sy, r)
    assert fit.slope[0] == pytest.approx(b, rel=1e-10)
    assert fit.intercept[0] == pytest.approx(a, rel=1e-10)
    assert fit.slope_sigma[0] == pytest.approx(sigma_b, rel=1e-10)
    assert fit.intercept_sigma[0] == pytest.approx(sigma_a, rel=1e-10)


@pytest.mark.parametrize("r", [0.0, 0.7])
def test_york_minimizes_its_own_objective(r: float) -> None:
    """York's slope minimizes S(b) = Σ (Y − a − bX)² / (σy² + b²σx² − 2brσxσy).

    A brute-force check that shares nothing with the iteration: for each b
    the best a is the weighted mean residual, so S is a function of b alone,
    minimized numerically.
    """
    rng = np.random.default_rng(7)
    x, y = noisy_line(7, n=30)
    sx, sy = rng.uniform(0.5, 1.5, x.size), rng.uniform(0.1, 0.3, x.size)

    def objective(b: float) -> float:
        w = 1 / (sy**2 + b * b * sx**2 - 2 * b * r * sx * sy)
        a = np.sum(w * (y - b * x)) / np.sum(w)
        return float(np.sum(w * (y - a - b * x) ** 2))

    best = minimize_scalar(objective, bracket=(0.0, 0.05, 0.2), tol=1e-14)
    assert york(x, y, sx, sy, r=r).slope[0] == pytest.approx(best.x, rel=1e-6)


def test_least_squares_agrees_with_statsmodels() -> None:
    """Slope, intercept, their standard errors and r² against statsmodels' OLS."""
    for seed in range(5):
        x, y = noisy_line(seed, n=15)
        fit = ols(x, y)
        reference = sm.OLS(y, sm.add_constant(x)).fit()
        assert fit.intercept[0] == pytest.approx(reference.params[0], rel=1e-10)
        assert fit.slope[0] == pytest.approx(reference.params[1], rel=1e-10)
        assert fit.intercept_sigma[0] == pytest.approx(reference.bse[0], rel=1e-10)
        assert fit.slope_sigma[0] == pytest.approx(reference.bse[1], rel=1e-10)
        assert fit.r_squared[0] == pytest.approx(reference.rsquared, rel=1e-10)


# ---------------------------------------------------------------------------
# Many groups at once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["york", "ols"])
def test_a_group_fitted_among_others_gets_exactly_what_it_gets_alone(method: str) -> None:
    """Shuffled rows, a reading shared by two fits, a hole, rows in no fit: bit for bit.

    Exactness rather than closeness, because it is a property of the design:
    every group's sums run over its own rows in their own order, and a group
    stops iterating the moment it converges.
    """
    fit_line = get_line_fit(method)
    rng = np.random.default_rng(8)
    x, y = noisy_line(8, n=60)
    sx, sy = rng.uniform(0.5, 1.5, x.size), rng.uniform(0.1, 0.3, x.size)
    # Three fits over one record: rows 0-29, rows 20-59 without 40-44
    # (overlapping the first, with a hole), and rows 10-49 every other row.
    members = [np.arange(30), np.r_[20:40, 45:60], np.arange(10, 50, 2)]
    rows = np.concatenate(members)
    labels = np.concatenate([np.full(m.size, k) for k, m in enumerate(members)])
    # Rows in no fit, interleaved, and every row shuffled.
    rows = np.r_[rows, [0, 1, 2]]
    labels = np.r_[labels, [-1, -1, -1]]
    order = rng.permutation(rows.size)
    rows, labels = rows[order], labels[order]
    together = fit_line(x[rows], y[rows], sx[rows], sy[rows], labels)
    for k, member in enumerate(members):
        mine = labels == k
        alone = fit_line(x[rows][mine], y[rows][mine], sx[rows][mine], sy[rows][mine])
        assert alone.n[0] == member.size
        for field in ("slope", "intercept", "slope_sigma", "intercept_sigma", "r_squared"):
            assert np.array_equal(getattr(together, field)[k], getattr(alone, field)[0])
        assert together.iterations[k] == alone.iterations[0]


def test_the_event_index_can_be_passed_as_the_labels() -> None:
    """-1 outside any event, so readings between events take part in no fit."""
    x = np.arange(12, dtype=float)
    y = np.r_[2 * x[:6], 3 * x[6:]] + 1.0
    labels = np.array([-1, 0, 0, 0, 0, -1, -1, 1, 1, 1, 1, -1])
    fit = ols(x, y, group=labels)
    np.testing.assert_allclose(fit.slope, [2.0, 3.0], rtol=1e-12)
    assert fit.n.tolist() == [4, 4]


def test_a_group_with_no_rows_is_reported_not_skipped() -> None:
    """n_groups beyond the largest label: the empty fits hold no line, and say why."""
    fit = york([0.0, 1.0, 2.0], [1.0, 2.0, 4.0], 0.1, 0.1, [1, 1, 1], n_groups=3)
    assert fit.status.tolist() == [FitStatus.TOO_FEW_POINTS, FitStatus.OK, FitStatus.TOO_FEW_POINTS]
    assert fit.n.tolist() == [0, 3, 0]
    assert np.isnan(fit.slope[[0, 2]]).all()
    # Stating exactly the count the labels need is as good as leaving it out.
    exact = york([0.0, 1.0, 2.0], [1.0, 2.0, 4.0], 0.1, 0.1, [1, 1, 1], n_groups=2)
    assert exact.status.tolist() == [FitStatus.TOO_FEW_POINTS, FitStatus.OK]


def test_no_rows_at_all_gives_no_fits() -> None:
    """Nothing in, nothing out, and no error: a sweep point can hold no events."""
    for fit in (york([], [], 0.1, 0.1, np.array([], dtype=int)), ols([], [], group=[-1, -1][:0])):
        assert fit.slope.shape == (0,)
        assert fit.status.shape == (0,)


# ---------------------------------------------------------------------------
# What a fit cannot do, it says
# ---------------------------------------------------------------------------


def test_fewer_than_three_points_hold_no_line() -> None:
    """Two points fix a line exactly but leave nothing to judge it by."""
    assert MIN_FIT_POINTS == 3
    for fit in (york([0.0, 1.0], [0.0, 1.0], 0.1, 0.1), ols([0.0, 1.0], [0.0, 1.0])):
        assert fit.status[0] == FitStatus.TOO_FEW_POINTS
        assert np.isnan(fit.slope[0])
        assert fit.iterations[0] == 0


def test_no_spread_in_x_holds_no_line() -> None:
    """Every x the same: the line would be vertical."""
    for fit in (
        york([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], 0.1, 0.1),
        ols([1.0, 1.0, 1.0], [0.0, 1.0, 2.0]),
    ):
        assert fit.status[0] == FitStatus.NO_SPREAD
        assert np.isnan(fit.slope[0]) and np.isnan(fit.slope_sigma[0])


def test_a_slope_of_zero_converges() -> None:
    """A source with none of the numerator gas: y does not vary, and the slope settles at 0."""
    fit = york([0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 5.0, 5.0], 0.1, 0.1)
    assert fit.status[0] == FitStatus.OK
    assert fit.slope[0] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(fit.r_squared[0])  # no variance in y to explain


def test_rows_without_finite_x_or_y_are_dropped_and_counted() -> None:
    """A gap is not a point; its sigma, declared or not, does not matter."""
    x = np.array([0.0, 1.0, np.nan, 2.0, 3.0])
    y = np.array([1.0, 3.0, 4.0, np.inf, 7.0])
    sx = np.array([0.1, 0.1, np.nan, 0.1, 0.1])
    fit = york(x, y, sx, 0.1)
    assert fit.status[0] == FitStatus.OK
    assert fit.n[0] == 3
    assert fit.n_dropped[0] == 2
    assert fit.slope[0] == pytest.approx(2.0)


def test_york_fits_no_line_without_a_declared_sigma(caplog: pytest.LogCaptureFixture) -> None:
    """No sigma at all, or one point without one: no line, the reason, one warning."""
    x, y = noisy_line(9, n=10)
    sy = np.full(x.size, 0.2)
    sy[4] = np.nan
    labels = np.r_[np.zeros(5, int), np.ones(5, int)]
    with caplog.at_level(logging.WARNING, logger="tsara.ratios.estimators"):
        fit = york(x, y, 1.0, sy, labels)
    assert fit.status.tolist() == [FitStatus.MISSING_SIGMA, FitStatus.OK]
    assert np.isnan(fit.slope[0]) and np.isfinite(fit.slope[1])
    assert "York fitted no line to 1 of 2 groups" in caplog.text
    assert york(x, y).status[0] == FitStatus.MISSING_SIGMA
    # Least squares needs no sigma and is not stopped by a missing one.
    assert ols(x, y, 1.0, sy, labels).ok.all()


def test_a_slope_still_moving_at_the_limit_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """One step from least squares does not reach York's slope: no line, and a warning."""
    with caplog.at_level(logging.WARNING, logger="tsara.ratios.estimators"):
        fit = york(
            PEARSON_X, PEARSON_Y, 1 / np.sqrt(PEARSON_WX), 1 / np.sqrt(PEARSON_WY), max_iter=1
        )
    assert fit.status[0] == FitStatus.NOT_CONVERGED
    assert np.isnan(fit.slope[0])
    assert fit.iterations[0] == 0
    assert "did not settle in 1 of 1 groups" in caplog.text
    assert pearson().iterations[0] == 8


def test_ok_is_the_status_read_as_a_mask() -> None:
    fit = york([0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.5, 0.0], 0.1, 0.1, [0, 0, 0, 1])
    assert fit.ok.tolist() == [True, False]


# ---------------------------------------------------------------------------
# A call that is wrong in itself is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": [[0.0, 1.0, 2.0]]}, "one-dimensional"),
        ({"y": [0.0, 1.0]}, "same length"),
        ({"group": [0, 0]}, "label every row"),
        ({"group": [0.0, 0.0, 0.0]}, "must be integers"),
        ({"group": [0, 1, 2], "n_groups": 2}, "labelled for group 2"),
        ({"sigma_x": [0.1, 0.1]}, "does not broadcast"),
        ({"sigma_y": -0.1}, "negative"),
        ({"sigma_x": 0.0, "sigma_y": 0.0}, "zero on both axes"),
        ({"r": 1.0}, "strictly between -1 and 1"),
        ({"r": np.nan}, "strictly between -1 and 1"),
        ({"rtol": 0.0}, "rtol must be positive"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
    ],
)
def test_york_refuses_a_call_that_does_not_fit_together(
    kwargs: dict[str, typing.Any], message: str
) -> None:
    call: dict[str, typing.Any] = {
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 1.0, 2.5],
        "sigma_x": 0.1,
        "sigma_y": 0.1,
    }
    call.update(kwargs)
    with pytest.raises(TsaraRatioError, match=message):
        york(**call)


def test_a_point_outside_every_fit_may_declare_anything_but_a_negative_sigma() -> None:
    """Both sigmas zero only matters on a point that is fitted; a negative one never makes sense."""
    fit = york(
        [0.0, 1.0, 2.0, 9.0],
        [0.0, 1.0, 2.5, 9.0],
        [0.1, 0.1, 0.1, 0.0],
        [0.1, 0.1, 0.1, 0.0],
        [0, 0, 0, -1],
    )
    assert fit.ok.all()
    with pytest.raises(TsaraRatioError, match="negative"):
        ols([0.0, 1.0, 2.0], [0.0, 1.0, 2.5], sigma_x=[0.1, 0.1, -1.0])


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_registered_fits_are_the_ones_the_configuration_names() -> None:
    """``RegressionConfig.methods`` and the registry must not drift apart."""
    annotation = RegressionConfig.model_fields["methods"].annotation
    (literal,) = typing.get_args(annotation)[:1]
    assert available_line_fits() == tuple(sorted(typing.get_args(literal)))
    assert get_line_fit("york") is york
    assert get_line_fit("ols") is ols


def test_an_unknown_fit_is_refused_naming_the_known_ones() -> None:
    with pytest.raises(
        TsaraRatioError, match=r"No line estimator registered as 'odr'.*'ols', 'york'"
    ):
        get_line_fit("odr")


def test_registering_a_name_twice_is_refused_unless_asked(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_line_fit("york")(ols)
    with pytest.raises(ValueError, match="non-empty"):
        register_line_fit("  ")
    try:
        with caplog.at_level(logging.WARNING, logger="tsara.ratios.estimators"):
            register_line_fit("ols", replace=True)(ols)
        assert "Replacing line estimator 'ols'" in caplog.text
    finally:
        register_line_fit("ols", replace=True)(ols)


# ---------------------------------------------------------------------------
# Monte Carlo: textbook draws, not the generator
# ---------------------------------------------------------------------------

#: Fits per Monte Carlo case. At 2000, a coverage's binomial standard error
#: is about 0.010 at 68 % and 0.005 at 95 %.
K = 2000


def draws(
    seed: int, n: int, sx: float, sy: float, r: float = 0.0
) -> tuple[Floats, Floats, npt.NDArray[np.intp]]:
    """K fits of n points about y = 2 + 0.05 x, true x uniform on [0, 20] (sd 5.77).

    Errors Gaussian with standard deviations sx and sy and correlation r,
    written from the definition: e_y = sy (r z₁ + √(1 − r²) z₂), e_x = sx z₁.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.0, 20.0, (K, n))
    z1, z2 = rng.standard_normal((K, n)), rng.standard_normal((K, n))
    x = truth + sx * z1
    y = 2.0 + 0.05 * truth + sy * (r * z1 + np.sqrt(1 - r * r) * z2)
    return x.ravel(), y.ravel(), np.repeat(np.arange(K), n)


def coverage(fit: LineFit, k: float) -> float:
    """Share of fits whose slope lies within k of its own sigma of the truth."""
    return float(np.mean(np.abs(fit.slope - 0.05) <= k * fit.slope_sigma))


def test_york_recovers_the_slope_that_least_squares_dilutes() -> None:
    """Noise on x half the spread of x: least squares reads var(x)/(var(x) + σx²) = 0.80 of it.

    York reads the truth, its sigma covers it about as often as a standard
    error should, and its MSWD averages 1 when the sigmas are right.
    """
    x, y, labels = draws(10, n=30, sx=2.887, sy=0.1)
    dilution = 5.7735**2 / (5.7735**2 + 2.887**2)
    least_squares = ols(x, y, group=labels)
    assert np.mean(least_squares.slope) / 0.05 == pytest.approx(dilution, abs=0.02)
    fit = york(x, y, 2.887, 0.1, labels)
    assert fit.ok.all()
    assert np.median(fit.slope) / 0.05 == pytest.approx(1.0, abs=0.02)
    assert 0.62 <= coverage(fit, 1) <= 0.72
    assert 0.92 <= coverage(fit, 2) <= 0.98
    assert np.mean(fit.mswd) == pytest.approx(1.0, abs=0.05)
    assert coverage(least_squares, 2) < 0.6


def test_a_correlation_declared_restores_the_slope_and_one_ignored_tilts_it() -> None:
    """Correlated errors have a slope of their own, r·σy/σx; ignoring r leans the line toward it."""
    x, y, labels = draws(11, n=30, sx=2.887, sy=0.5, r=0.8)
    declared = york(x, y, 2.887, 0.5, labels, r=0.8)
    ignored = york(x, y, 2.887, 0.5, labels)
    assert np.median(declared.slope) / 0.05 == pytest.approx(1.0, abs=0.03)
    assert 0.92 <= coverage(declared, 2) <= 0.98
    assert np.median(ignored.slope) / 0.05 > 1.3
