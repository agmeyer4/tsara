"""Tests for measuring the statistical shape of real timeseries."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from tsara.synthetic.profiling import (
    RealDataProfile,
    TsaraProfilingError,
    diff_mad_sigma,
    profile_series,
)

# ---------------------------------------------------------------------------
# diff_mad
# ---------------------------------------------------------------------------


def test_diff_mad_recovers_gaussian_sigma() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 2.5, 200_000)
    assert diff_mad_sigma(values) == pytest.approx(2.5, rel=0.02)


def test_diff_mad_is_immune_to_a_large_plume() -> None:
    """The whole reason `diff_mad` is preferred over signal MAD (METHODS.md §2.5)."""
    rng = np.random.default_rng(0)
    n = 20_000
    noise = rng.normal(0.0, 1.0, n)
    # A broad enhancement occupying most of the record.
    plume = 200.0 * np.exp(-0.5 * ((np.arange(n) - n / 2) / (n / 5)) ** 2)

    clean = diff_mad_sigma(noise)
    contaminated = diff_mad_sigma(noise + plume)
    naive_mad = 1.4826 * np.median(np.abs((noise + plume) - np.median(noise + plume)))

    assert contaminated == pytest.approx(clean, rel=0.05)
    # ...whereas the signal-MAD estimate is inflated by orders of magnitude.
    assert naive_mad > 10.0 * contaminated


def test_diff_mad_of_a_short_array_is_nan() -> None:
    assert math.isnan(diff_mad_sigma(np.array([1.0])))


# ---------------------------------------------------------------------------
# profile_series: happy paths
# ---------------------------------------------------------------------------


def test_profile_reports_basic_shape(autocorrelated_series: pd.Series) -> None:
    profile = profile_series(autocorrelated_series, name="test", block_length=128)
    assert profile.name == "test"
    assert profile.n_blocks > 0
    assert profile.block_length == 128
    assert profile.sample_period_s == pytest.approx(1.0)
    assert profile.n_source_points == len(autocorrelated_series)
    assert profile.decorrelation_timescale_s is not None


def test_profile_recovers_ar1_autocorrelation_on_a_clean_record() -> None:
    """On a plume-free record, rho1 really is the noise autocorrelation."""
    rng = np.random.default_rng(5)
    n = 20_000
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0.0, 1.0)
    index = pd.date_range("2026-01-01", periods=n, freq="1s")
    series = pd.Series(1900.0 + noise, index=index)

    profile = profile_series(series, name="clean", block_length=256)
    assert profile.lag1_autocorr == pytest.approx(0.8, abs=0.1)
    assert profile.decorrelation_timescale_s == pytest.approx(-1.0 / math.log(0.8), rel=0.5)


def test_plume_leakage_inflates_the_residual_autocorrelation(
    autocorrelated_series: pd.Series,
) -> None:
    """A documented limitation, asserted so it cannot silently change.

    The fixture's noise is AR(1) with rho = 0.8, but broad plumes leaking
    through the profiling baseline are smooth and therefore push the
    *residual's* rho1 far higher. The profile is honest about describing the
    substrate, not the instrument noise.
    """
    profile = profile_series(autocorrelated_series, name="dense", block_length=128)
    assert profile.lag1_autocorr > 0.9


def test_profile_blocks_are_mean_centred(autocorrelated_series: pd.Series) -> None:
    """Centring is what lets blocks be stitched without level jumps."""
    profile = profile_series(autocorrelated_series, name="test", block_length=128)
    assert np.allclose(profile.residual_blocks.mean(axis=1), 0.0, atol=1e-9)


def test_profile_background_statistics_track_the_real_level(
    autocorrelated_series: pd.Series,
) -> None:
    profile = profile_series(autocorrelated_series, name="test", block_length=128)
    assert profile.background_median == pytest.approx(1900.0, abs=15.0)
    assert profile.background_iqr > 0.0


def test_plume_dense_record_shows_residual_exceeding_noise(
    autocorrelated_series: pd.Series, caplog: pytest.LogCaptureFixture
) -> None:
    """The expected signature of the owner's plume-dense real data.

    Leakage is a documented feature, not a failure: it is what makes a
    bootstrapped substrate an adversarial noise-estimation test case.
    """
    tall = autocorrelated_series.copy()
    centre = 3000
    index = np.arange(len(tall))
    tall += 400.0 * np.exp(-0.5 * ((index - centre) / 500.0) ** 2)

    with caplog.at_level(logging.INFO, logger="tsara.synthetic.profiling"):
        profile = profile_series(tall, name="dense", block_length=128)
    assert profile.residual_sigma > profile.noise_sigma
    assert "plume energy leaked" in caplog.text


def test_profile_summary_is_informative(autocorrelated_series: pd.Series) -> None:
    profile = profile_series(autocorrelated_series, name="test", block_length=128)
    summary = profile.summary()
    assert "test" in summary
    assert "noise_sigma" in summary


def test_profile_summary_reports_absent_tau() -> None:
    profile = RealDataProfile(
        name="flat",
        residual_blocks=np.zeros((2, 4)),
        residual_sigma=0.0,
        noise_sigma=0.0,
        lag1_autocorr=-0.5,
        decorrelation_timescale_s=None,
        background_median=0.0,
        background_iqr=0.0,
        sample_period_s=1.0,
        n_source_points=8,
    )
    assert "tau=none" in profile.summary()


def test_profile_caps_the_block_count(autocorrelated_series: pd.Series) -> None:
    profile = profile_series(autocorrelated_series, name="test", block_length=64, max_blocks=5)
    assert profile.n_blocks == 5


def test_profile_without_a_cap_keeps_every_block(
    autocorrelated_series: pd.Series,
) -> None:
    profile = profile_series(autocorrelated_series, name="test", block_length=64, max_blocks=None)
    assert profile.n_blocks > 5


def test_blocks_never_straddle_a_data_gap() -> None:
    """A block spanning a dropout would fabricate a jump the instrument never made."""
    rng = np.random.default_rng(0)
    first = pd.date_range("2026-01-01 00:00", periods=300, freq="1s")
    second = pd.date_range("2026-01-01 02:00", periods=300, freq="1s")
    index = first.append(second)
    values = pd.Series(rng.normal(100.0, 1.0, 600), index=index)

    profile = profile_series(values, name="gapped", block_length=128, baseline_window="60s")
    # 300 samples per contiguous run -> exactly two 128-sample blocks each.
    assert profile.n_blocks == 4


def test_anticorrelated_residual_has_no_ar1_fit() -> None:
    """Alternating residuals give rho1 < 0, for which no AR(1) tau exists."""
    index = pd.date_range("2026-01-01", periods=2000, freq="1s")
    values = pd.Series(100.0 + np.where(np.arange(2000) % 2 == 0, 1.0, -1.0), index=index)
    profile = profile_series(values, name="alt", block_length=64, baseline_window="30s")
    assert profile.lag1_autocorr <= 0.0
    assert profile.decorrelation_timescale_s is None


def test_constant_residual_is_treated_as_uncorrelated() -> None:
    """corrcoef on a constant array is nan; that must not leak into the profile."""
    index = pd.date_range("2026-01-01", periods=1000, freq="1s")
    values = pd.Series(np.full(1000, 42.0), index=index)
    profile = profile_series(values, name="const", block_length=64, baseline_window="30s")
    assert profile.lag1_autocorr == 0.0
    assert profile.decorrelation_timescale_s is None


def test_nan_samples_are_excluded(autocorrelated_series: pd.Series) -> None:
    holed = autocorrelated_series.copy()
    holed.iloc[100:200] = np.nan
    profile = profile_series(holed, name="holed", block_length=64)
    assert profile.n_source_points == len(holed) - 100
    assert np.all(np.isfinite(profile.residual_blocks))


def test_non_numeric_values_are_coerced() -> None:
    index = pd.date_range("2026-01-01", periods=600, freq="1s")
    rng = np.random.default_rng(0)
    values = pd.Series([str(round(v, 3)) for v in rng.normal(100.0, 1.0, 600)], index=index)
    profile = profile_series(values, name="strings", block_length=128, baseline_window="60s")
    assert profile.noise_sigma == pytest.approx(1.0, rel=0.2)


# ---------------------------------------------------------------------------
# profile_series: failure modes
# ---------------------------------------------------------------------------


def test_requires_a_datetime_index() -> None:
    with pytest.raises(TsaraProfilingError, match="DatetimeIndex"):
        profile_series(pd.Series([1.0, 2.0, 3.0]), name="x")


def test_requires_a_sorted_index(autocorrelated_series: pd.Series) -> None:
    shuffled = autocorrelated_series.iloc[::-1]
    with pytest.raises(TsaraProfilingError, match="monotonically increasing"):
        profile_series(shuffled, name="x")


@pytest.mark.parametrize("quantile", [0.0, 0.6, 1.0, -0.1])
def test_rejects_out_of_range_quantile(autocorrelated_series: pd.Series, quantile: float) -> None:
    with pytest.raises(TsaraProfilingError, match="baseline_quantile"):
        profile_series(autocorrelated_series, name="x", baseline_quantile=quantile)


def test_rejects_a_degenerate_block_length(autocorrelated_series: pd.Series) -> None:
    with pytest.raises(TsaraProfilingError, match="block_length must be at least 2"):
        profile_series(autocorrelated_series, name="x", block_length=1)


def test_rejects_a_record_shorter_than_one_block(
    autocorrelated_series: pd.Series,
) -> None:
    with pytest.raises(TsaraProfilingError, match="fewer than block_length"):
        profile_series(autocorrelated_series.iloc[:50], name="x", block_length=512)


def test_rejects_a_record_with_too_few_residual_samples() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="1s")
    values = pd.Series([1.0, np.nan, np.nan, np.nan], index=index)
    with pytest.raises(TsaraProfilingError, match="fewer than block_length"):
        profile_series(values, name="x", block_length=2)


def test_rejects_a_record_too_fragmented_for_any_block() -> None:
    """Every contiguous run is shorter than one block."""
    stamps: list[pd.Timestamp] = []
    for minute in range(50):
        stamps.extend(
            pd.date_range(f"2026-01-01 00:{minute:02d}:00", periods=10, freq="1s").tolist()
        )
    values = pd.Series(np.arange(len(stamps), dtype=float), index=pd.DatetimeIndex(stamps))
    with pytest.raises(TsaraProfilingError, match="no gap-free run reached"):
        profile_series(values, name="x", block_length=64, baseline_window="10s")


def test_two_residual_samples_is_enough() -> None:
    """Boundary: exactly block_length finite samples must succeed."""
    index = pd.date_range("2026-01-01", periods=2, freq="1s")
    values = pd.Series([1.0, 3.0], index=index)
    profile = profile_series(values, name="tiny", block_length=2, baseline_window="10s")
    assert profile.n_blocks == 1


def test_rejects_a_record_whose_baseline_is_entirely_undefined() -> None:
    """Samples too far apart for the baseline window leave no residual.

    With min_periods=2, a centred window containing a single sample yields
    NaN, so every residual is NaN and there is nothing to characterize.
    """
    index = pd.DatetimeIndex(["2026-01-01 00:00:00", "2026-01-01 01:00:00"])
    values = pd.Series([100.0, 101.0], index=index)
    with pytest.raises(TsaraProfilingError, match="fewer than two finite residual"):
        profile_series(values, name="sparse", block_length=2, baseline_window="10s")
