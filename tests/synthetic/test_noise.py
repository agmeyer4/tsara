"""Tests for two-component error injection and quantization.

These tests assert the *defining properties* of the two components rather
than merely that noise was added: random error must average down, systematic
error must not, and an AR(1) random component must exhibit the requested
correlation. Those three properties are what make generated data a valid test
bed for METHODS.md §3.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tsara.synthetic.config import TrueComponent, TrueUncertainty
from tsara.synthetic.noise import (
    _ar1_standardized,
    apply_uncertainty,
    draw_random_error,
    draw_systematic_error,
    quantization_floor,
    quantize,
)


def _times(n: int, freq: str = "1s") -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq=freq)


# ---------------------------------------------------------------------------
# Random component
# ---------------------------------------------------------------------------


def test_random_error_matches_the_declared_sigma() -> None:
    n = 200_000
    values = np.full(n, 100.0)
    component = TrueComponent(absolute=2.0)
    realization = draw_random_error(values, component, _times(n), np.random.default_rng(0))
    assert realization.error.std() == pytest.approx(2.0, rel=0.02)
    assert np.all(realization.sigma == 2.0)
    assert realization.reported_sigma is None


def test_relative_sigma_scales_with_the_signal() -> None:
    values = np.array([100.0, 200.0])
    component = TrueComponent(relative=0.1)
    realization = draw_random_error(values, component, _times(2), np.random.default_rng(0))
    assert realization.sigma == pytest.approx([10.0, 20.0])


def test_random_error_averages_down() -> None:
    """The defining property of a random component (METHODS.md §3.2)."""
    n = 40_000
    values = np.zeros(n)
    component = TrueComponent(absolute=5.0)
    realization = draw_random_error(values, component, _times(n), np.random.default_rng(1))
    expected_se = 5.0 / math.sqrt(n)
    assert abs(realization.error.mean()) < 5.0 * expected_se


def test_reported_sigma_honours_the_configured_bias() -> None:
    """An instrument that understates its own error must do so measurably."""
    values = np.full(10, 50.0)
    component = TrueComponent(absolute=4.0, report_as="err", report_bias=0.8)
    realization = draw_random_error(values, component, _times(10), np.random.default_rng(0))
    assert realization.reported_sigma is not None
    assert np.allclose(realization.reported_sigma, 0.8 * realization.sigma)


def test_single_sample_skips_the_ar1_path() -> None:
    realization = draw_random_error(
        np.array([1.0]),
        TrueComponent(absolute=1.0),
        _times(1),
        np.random.default_rng(0),
        decorrelation_timescale_s=10.0,
    )
    assert realization.error.shape == (1,)


# ---------------------------------------------------------------------------
# AR(1) correlation
# ---------------------------------------------------------------------------


def test_ar1_has_unit_variance_and_the_requested_lag1_correlation() -> None:
    n = 400_000
    tau_s = 10.0
    series = _ar1_standardized(_times(n), tau_s, np.random.default_rng(0))
    expected_rho = math.exp(-1.0 / tau_s)
    assert series.std() == pytest.approx(1.0, rel=0.02)
    assert np.corrcoef(series[:-1], series[1:])[0, 1] == pytest.approx(expected_rho, rel=0.02)


def test_ar1_does_not_start_from_a_burn_in_transient() -> None:
    """A filter started at zero would leave the first samples under-dispersed."""
    firsts = np.array(
        [_ar1_standardized(_times(50), 20.0, np.random.default_rng(s))[0] for s in range(4000)]
    )
    assert firsts.std() == pytest.approx(1.0, rel=0.05)


def test_ar1_on_irregular_times_uses_the_exact_recursion() -> None:
    """Jittered timestamps take the per-point path, not a median-dt shortcut."""
    rng = np.random.default_rng(0)
    base = _times(60_000).to_numpy().astype("datetime64[ns]").astype(np.int64)
    jitter = rng.integers(-3 * 10**8, 3 * 10**8, size=base.size)
    times = pd.DatetimeIndex(np.sort(base + jitter))

    series = _ar1_standardized(times, 10.0, np.random.default_rng(1))
    assert series.std() == pytest.approx(1.0, rel=0.05)
    # Still correlated at roughly the right magnitude.
    assert np.corrcoef(series[:-1], series[1:])[0, 1] == pytest.approx(
        math.exp(-1.0 / 10.0), rel=0.1
    )


def test_ar1_with_short_tau_approaches_white_noise() -> None:
    series = _ar1_standardized(_times(50_000), 1e-3, np.random.default_rng(0))
    assert abs(np.corrcoef(series[:-1], series[1:])[0, 1]) < 0.05


# ---------------------------------------------------------------------------
# Systematic component
# ---------------------------------------------------------------------------


def test_systematic_error_is_constant_for_a_constant_signal() -> None:
    values = np.full(1000, 100.0)
    realization, draws = draw_systematic_error(
        values, TrueComponent(absolute=3.0), np.random.default_rng(0)
    )
    assert realization.error.std() == pytest.approx(0.0, abs=1e-12)
    assert len(draws) == 2


def test_systematic_error_does_not_average_down() -> None:
    """The defining property of a systematic component (METHODS.md §3.3).

    Averaging a huge number of samples must leave the error essentially
    untouched — a pipeline that shrinks it by sqrt(N) is wrong, and this is
    the data that proves it.
    """
    n = 100_000
    values = np.full(n, 100.0)
    means = []
    for seed in range(300):
        realization, _ = draw_systematic_error(
            values, TrueComponent(absolute=3.0), np.random.default_rng(seed)
        )
        means.append(realization.error.mean())
    # Scatter of the *record means* is still the full sigma, not sigma/sqrt(N).
    assert np.std(means) == pytest.approx(3.0, rel=0.15)


def test_systematic_relative_error_scales_the_signal() -> None:
    values = np.array([100.0, 200.0, 400.0])
    realization, (_, g_rel) = draw_systematic_error(
        values, TrueComponent(relative=0.05), np.random.default_rng(0)
    )
    assert realization.error == pytest.approx(0.05 * values * g_rel)


def test_systematic_reported_sigma_is_emitted_when_configured() -> None:
    values = np.full(5, 100.0)
    realization, _ = draw_systematic_error(
        values, TrueComponent(relative=0.01, report_as="sys_err"), np.random.default_rng(0)
    )
    assert realization.reported_sigma is not None


# ---------------------------------------------------------------------------
# apply_uncertainty
# ---------------------------------------------------------------------------


def test_apply_uncertainty_with_no_budget_is_a_copy() -> None:
    values = np.arange(10, dtype=float)
    applied = apply_uncertainty(values, None, _times(10), np.random.default_rng(0))
    assert np.array_equal(applied.values, values)
    assert applied.values is not values
    assert applied.sigma_rand is None
    assert applied.sigma_sys is None
    assert applied.reported == {}
    assert applied.scalars == {}


def test_apply_uncertainty_random_only() -> None:
    applied = apply_uncertainty(
        np.full(100, 50.0),
        TrueUncertainty(random=TrueComponent(absolute=1.0)),
        _times(100),
        np.random.default_rng(0),
    )
    assert applied.sigma_rand is not None
    assert applied.sigma_sys is None
    assert applied.scalars == {}


def test_apply_uncertainty_systematic_only_records_its_draws() -> None:
    applied = apply_uncertainty(
        np.full(100, 50.0),
        TrueUncertainty(systematic=TrueComponent(absolute=1.0)),
        _times(100),
        np.random.default_rng(0),
    )
    assert applied.sigma_rand is None
    assert applied.sigma_sys is not None
    assert set(applied.scalars) == {"true_sys_abs_draw", "true_sys_rel_draw"}


def test_apply_uncertainty_both_components_with_reported_columns() -> None:
    applied = apply_uncertainty(
        np.full(100, 50.0),
        TrueUncertainty(
            random=TrueComponent(absolute=1.0, report_as="rand_err"),
            systematic=TrueComponent(relative=0.01, report_as="sys_err"),
            decorrelation_timescale="5s",
        ),
        _times(100),
        np.random.default_rng(0),
    )
    assert set(applied.reported) == {"rand_err", "sys_err"}
    assert applied.sigma_rand is not None
    assert applied.sigma_sys is not None


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def test_quantize_snaps_to_the_grid() -> None:
    values = np.array([1.004, 1.006, -1.004])
    assert quantize(values, 0.01) == pytest.approx([1.00, 1.01, -1.00])


def test_quantize_output_is_an_exact_multiple_of_the_step() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(1900.0, 5.0, 1000)
    quantized = quantize(values, 0.05)
    assert np.allclose(np.remainder(quantized / 0.05 + 0.5, 1.0) - 0.5, 0.0, atol=1e-9)


def test_quantization_defeats_a_median_noise_estimator() -> None:
    """The adversarial case the METHODS.md §2.5 floor exists to survive.

    With noise well under the reporting step, over half the first differences
    are exactly zero, so the median collapses and a naive estimator reports
    zero noise — which would make every sample a detection.
    """
    rng = np.random.default_rng(0)
    values = quantize(rng.normal(1900.0, 0.001, 5000), 0.01)
    naive_sigma = 1.4826 * np.median(np.abs(np.diff(values))) / math.sqrt(2.0)
    assert naive_sigma == 0.0
    assert quantization_floor(0.01) > 0.0


def test_quantization_floor_is_the_uniform_rounding_sd() -> None:
    assert quantization_floor(0.12) == pytest.approx(0.12 / math.sqrt(12.0))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_quantize_rejects_nonpositive_resolution(bad: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        quantize(np.array([1.0]), bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_quantization_floor_rejects_nonpositive_resolution(bad: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        quantization_floor(bad)
