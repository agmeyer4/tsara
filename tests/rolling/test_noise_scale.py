"""Tests for the noise scale (tsara.rolling.noise).

Ground truth is the point: the generator manufactures a stream with a known
random sigma and writes it beside the readings as `truth_sigma_rand_<x>`,
plume-dense and quantized on request, which is what the Phase-2 test cases
were built for.
"""

from __future__ import annotations

import logging
from typing import get_args

import numpy as np
import pytest
import xarray as xr

from tsara import load_synthetic
from tsara.config.analysis import DetectionConfig, NoiseEstimator
from tsara.core.support import CellBounds, stream_cells
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.rolling import (
    NoiseRequest,
    TsaraRollingError,
    available_noise_estimators,
    get_noise_estimator,
    noise_scale,
    register_noise_estimator,
)
from tsara.rolling.noise import (
    _ESTIMATORS,
    DROPOUT_SPACING_FACTOR,
    MIN_NOISE_SAMPLES,
    noise_by_diff_mad,
    noise_by_mad,
)
from tsara.synthetic import generate

TEN_MINUTES = 600 * SECOND


def cells(start_s: float, width_s: float, n: int, step_s: float | None = None) -> CellBounds:
    step = width_s if step_s is None else step_s
    start = (np.arange(n, dtype=np.int64) * int(step * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def make_stream(bounds: CellBounds, values: np.ndarray, **attrs: object) -> xr.Dataset:
    dataset = xr.Dataset(
        data_vars={
            "ch4": ("time", values, {"units": "ppb", "role": "gas", "field": "ch4", **attrs})
        },
        coords={
            "time": bounds.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
    )
    dataset["time"].attrs["bounds"] = "time_bnds"
    return dataset


def request(bounds: CellBounds, values: np.ndarray, window_ns: int = TEN_MINUTES) -> NoiseRequest:
    return NoiseRequest(
        readings=bounds, values=values, window_ns=window_ns, block_elements=2_000_000
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_configured_estimators_are_the_registered_ones() -> None:
    assert available_noise_estimators() == tuple(sorted(get_args(NoiseEstimator)))
    assert get_noise_estimator("diff_mad") is noise_by_diff_mad
    assert get_noise_estimator("mad") is noise_by_mad


def test_an_unregistered_estimator_is_refused_listing_what_exists() -> None:
    with pytest.raises(TsaraRollingError, match="qn.*Available.*diff_mad"):
        get_noise_estimator("qn")


def test_a_name_is_registered_once_unless_replacement_is_asked_for(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def estimator(request: NoiseRequest, /) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    try:
        register_noise_estimator("test_estimator")(estimator)
        with pytest.raises(ValueError, match="already registered as 'test_estimator'"):
            register_noise_estimator("test_estimator")(estimator)
        with caplog.at_level(logging.WARNING, logger="tsara.rolling.noise"):
            register_noise_estimator("test_estimator", replace=True)(estimator)
        assert "Replacing noise estimator 'test_estimator'" in caplog.text
    finally:
        _ESTIMATORS.pop("test_estimator", None)
    with pytest.raises(ValueError, match="non-empty"):
        register_noise_estimator(" ")


# ---------------------------------------------------------------------------
# diff_mad: what it recovers, what it ignores, what it drops
# ---------------------------------------------------------------------------


def test_diff_mad_recovers_the_sigma_of_white_noise_on_a_flat_background() -> None:
    rng = np.random.default_rng(2)
    bounds = cells(0.0, 1.0, 3600)
    sigma = noise_by_diff_mad(request(bounds, 1900 + rng.normal(0, 0.7, 3600)))
    inside = sigma[300:3300]
    assert np.isfinite(inside).all()
    assert np.median(inside) == pytest.approx(0.7, rel=0.05)


def test_diff_mad_barely_moves_under_a_plume_where_mad_inflates() -> None:
    """A 100 ppb plume 100 s wide in a 10 min window (METHODS §2.5, measured).

    The difference estimator reads the noise within 20 %: a plume's slope is
    a fraction of a ppb per second against a difference noise of a ppb. The
    signal's MAD about its rolling median reads three to four times the
    noise, because the residuals inside the plume are the plume's own shape.
    """
    rng = np.random.default_rng(4)
    bounds = cells(0.0, 1.0, 3600)
    t = np.arange(3600.0)
    values = 1900 + rng.normal(0, 0.7, 3600) + 100 * np.exp(-0.5 * ((t - 1800) / 100) ** 2)
    by_difference = noise_by_diff_mad(request(bounds, values))
    by_signal = noise_by_mad(request(bounds, values))
    assert by_difference[1800] == pytest.approx(0.7, rel=0.2)
    assert by_signal[1800] > 2.5 * 0.7


def test_a_difference_across_a_dropout_is_dropped() -> None:
    """A gap with a 50 ppb step across it would put one huge difference in every window
    touching it; dropped, the estimate stays the noise. The rule is on the median
    spacing between consecutive finite readings, so a masked stretch is a gap too."""
    rng = np.random.default_rng(6)
    n = 2400
    bounds = cells(0.0, 1.0, n)
    values = 1900 + rng.normal(0, 0.7, n)
    values[1200:] += 50.0
    values[1180:1220] = np.nan  # a masked stretch straddling the step
    sigma = noise_by_diff_mad(request(bounds, values, window_ns=120 * SECOND))
    assert np.nanmax(sigma) < 1.0
    # The same step with a shorter gap that the rule keeps (1 masked reading, spacing 2 < 1.5 x 1?
    # no: 2 > 1.5, dropped too) -- so a step across a single missing reading is also excluded.
    values2 = 1900 + rng.normal(0, 0.7, n)
    values2[1200:] += 50.0
    values2[1200] = np.nan
    sigma2 = noise_by_diff_mad(request(bounds, values2, window_ns=120 * SECOND))
    assert np.nanmax(sigma2) < 1.0
    assert DROPOUT_SPACING_FACTOR < 2.0


def test_jittered_spacing_is_kept_and_the_rule_is_on_spacing_not_width() -> None:
    """1 s cells every 2.3 s with jitter: every difference is kept; a rule on cell
    width would drop them all (the 07-18 drive's shape)."""
    rng = np.random.default_rng(8)
    n = 1500
    steps = rng.normal(2.3, 0.05, n)
    mid = np.cumsum(steps)
    start = (mid * SECOND).astype(np.int64) - SECOND // 2
    bounds = CellBounds(start_ns=start, stop_ns=start + SECOND)
    sigma = noise_by_diff_mad(request(bounds, 1900 + rng.normal(0, 0.7, n)))
    assert np.isfinite(sigma[200:-200]).all()
    assert np.median(sigma[200:-200]) == pytest.approx(0.7, rel=0.06)


def test_a_window_with_too_few_differences_is_blank() -> None:
    bounds = cells(0.0, 1.0, 40)
    sigma = noise_by_diff_mad(request(bounds, 1900 + np.arange(40.0), window_ns=8 * SECOND))
    assert np.isnan(sigma).all()  # at most 8 differences per window, fewer than MIN_NOISE_SAMPLES
    assert MIN_NOISE_SAMPLES == 10
    wider = noise_by_diff_mad(request(bounds, 1900 + np.arange(40.0), window_ns=30 * SECOND))
    assert np.isfinite(wider[15:25]).all()


def test_fewer_than_two_finite_readings_give_no_scale() -> None:
    bounds = cells(0.0, 1.0, 20)
    values = np.full(20, np.nan)
    values[3] = 1900.0
    assert np.isnan(noise_by_diff_mad(request(bounds, values))).all()
    assert np.isnan(noise_by_mad(request(bounds, values))).all()


def test_mad_recovers_white_noise_on_a_flat_background() -> None:
    rng = np.random.default_rng(9)
    bounds = cells(0.0, 1.0, 3600)
    sigma = noise_by_mad(request(bounds, 1900 + rng.normal(0, 0.7, 3600)))
    assert np.median(sigma[300:3300]) == pytest.approx(0.7, rel=0.05)


# ---------------------------------------------------------------------------
# The ladder and the floor (§2.3, §2.5)
# ---------------------------------------------------------------------------


def test_a_declared_or_reported_sigma_tops_the_ladder_unfloored() -> None:
    bounds = cells(0.0, 1.0, 100)
    stream = make_stream(bounds, np.round(1900 + np.arange(100.0) * 0.001, 2))
    stream["sigma_rand_ch4"] = ("time", np.full(100, 0.5), {"uncertainty_provenance": "reported"})
    result = noise_scale(stream, "ch4", bounds, estimator="diff_mad", window_ns=TEN_MINUTES)
    assert (result.sigma == 0.5).all()
    assert result.provenance == "reported"
    assert result.estimator is None
    assert np.isnan(result.resolution) and result.floor_fraction == 0.0


def test_without_a_declared_sigma_the_named_estimator_runs_and_is_labelled_empirical() -> None:
    rng = np.random.default_rng(12)
    bounds = cells(0.0, 1.0, 1200)
    stream = make_stream(bounds, 1900 + rng.normal(0, 0.7, 1200))
    result = noise_scale(stream, "ch4", bounds, estimator="diff_mad", window_ns=TEN_MINUTES)
    assert result.provenance == "empirical" and result.estimator == "diff_mad"
    assert np.median(result.sigma[300:900]) == pytest.approx(0.7, rel=0.06)
    assert result.floor_fraction == 0.0  # continuous data: the detected step is negligible
    assert result.resolution < 1e-4  # the closest pair among 1200 normal draws
    with pytest.raises(TsaraRollingError, match="No noise estimator registered as 'qn'"):
        noise_scale(stream, "ch4", bounds, estimator="qn", window_ns=TEN_MINUTES)


def test_a_quantized_record_is_floored_at_the_rounding_scale() -> None:
    """Noise of 0.004 written in 0.01 steps: more than half the differences are zero, the
    median difference is zero, and the floor delta / sqrt(12) is what stops every reading
    being a plume."""
    rng = np.random.default_rng(13)
    bounds = cells(0.0, 1.0, 1200)
    stream = make_stream(bounds, np.round(1900 + rng.normal(0, 0.004, 1200), 2))
    result = noise_scale(stream, "ch4", bounds, estimator="diff_mad", window_ns=TEN_MINUTES)
    assert result.resolution == pytest.approx(0.01)
    assert np.nanmin(result.sigma) == pytest.approx(0.01 / np.sqrt(12))
    assert result.floor_fraction > 0.5
    # A declared quantization wins over the detected one.
    stream["ch4"].attrs["quantization"] = 0.05
    declared = noise_scale(stream, "ch4", bounds, estimator="diff_mad", window_ns=TEN_MINUTES)
    assert declared.resolution == 0.05
    assert np.nanmin(declared.sigma) == pytest.approx(0.05 / np.sqrt(12))


def test_a_record_of_one_distinct_value_has_no_resolution_to_floor_at() -> None:
    bounds = cells(0.0, 1.0, 100)
    stream = make_stream(bounds, np.full(100, 1900.0))
    result = noise_scale(stream, "ch4", bounds, estimator="diff_mad", window_ns=TEN_MINUTES)
    assert result.resolution == 0.0
    assert (result.sigma[np.isfinite(result.sigma)] == 0.0).all()


# ---------------------------------------------------------------------------
# Ground truth from the generator
# ---------------------------------------------------------------------------


def test_on_a_plume_dense_record_the_estimate_is_biased_high_by_the_measured_factor() -> None:
    """The example campaign's Picarro, its reported sigma removed so the ladder falls to
    the estimator, against the true random sigma the generator wrote beside it.

    Not 1.00. Thirty percent of the readings sit inside plumes whose slopes
    exceed the noise per 2 s step, so a window's median absolute difference
    lands near the 70th percentile of the noise differences: 1.30 times the
    truth over all readings, 1.21 on readings outside plumes (whose windows
    still hold plumes). METHODS §2.5 records the number and its cause, and
    names the Phase-6 remedy, re-estimating outside detected plumes. This
    test pins the measurement so the document cannot drift from the code.
    """
    data = generate(load_synthetic("examples/configs/synthetic_example.yaml"))
    stream = data.streams["picarro"].drop_vars(["sigma_rand_ch4", "sigma_sys_ch4"], errors="ignore")
    assert "sigma_rand_ch4" not in stream.data_vars
    truth = stream["truth_sigma_rand_ch4"].values
    enhancement = stream["truth_enhancement_ch4"].values
    result = noise_scale(
        stream, "ch4", stream_cells(stream, "picarro"), estimator="diff_mad", window_ns=TEN_MINUTES
    )
    inside = np.isfinite(result.sigma)
    assert inside.mean() > 0.95
    ratio = result.sigma / truth
    quiet = inside & (enhancement < 1.0)
    assert 0.25 < np.mean(enhancement > 3 * truth) < 0.35  # the record is plume-dense
    assert np.median(ratio[inside]) == pytest.approx(1.30, abs=0.06)
    assert np.median(ratio[quiet]) == pytest.approx(1.21, abs=0.06)
    # The noise really is the truth: the reading's error against the true value.
    observed = stream["ch4"].values - stream["truth_background_ch4"].values - enhancement
    assert np.std(observed) == pytest.approx(np.median(truth), rel=0.03)


def test_the_state_carries_the_noise_scale_with_its_record() -> None:
    from tsara.config.analysis import BaselineConfig
    from tsara.rolling import rolling_state

    rng = np.random.default_rng(14)
    bounds = cells(0.0, 1.0, 1800)
    stream = make_stream(bounds, 1900 + rng.normal(0, 0.7, 1800))
    settings = BaselineConfig.model_validate({"windows": ["2min"], "quantiles": [0.05]})
    state = rolling_state(stream, instrument="a", baseline=settings)
    noise = state["noise_ch4"]
    assert noise.dims == ("time",)
    assert noise.attrs["uncertainty_provenance"] == "empirical"
    assert noise.attrs["tsara_noise_estimator"] == "diff_mad"
    assert noise.attrs["tsara_noise_window"] == "10min"
    assert noise.attrs["tsara_noise_min_samples"] == MIN_NOISE_SAMPLES
    assert noise.attrs["tsara_noise_floor_fraction"] == 0.0
    assert 0 <= noise.attrs["tsara_noise_blank_fraction"] < 0.2
    assert np.median(noise.values[300:1500]) == pytest.approx(0.7, rel=0.06)
    # The knobs come from the detection config, where the sigma is used.
    other = rolling_state(
        stream,
        instrument="a",
        baseline=settings,
        noise=DetectionConfig(noise_estimator="mad", noise_window="5min"),
    )
    assert other["noise_ch4"].attrs["tsara_noise_estimator"] == "mad"
    assert other["noise_ch4"].attrs["tsara_noise_window"] == "5min"
    five = rolling_state(
        stream, instrument="a", baseline=settings, noise=DetectionConfig(noise_window="5min")
    )["noise_ch4"].values
    # A 5 min window and a 10 min window give different estimates, so the
    # window really came from the config and not from a default.
    assert not np.array_equal(five, noise.values, equal_nan=True)
    assert np.median(five[300:1500]) == pytest.approx(0.7, rel=0.08)
    # A declared sigma is copied, labelled, and carries no estimator record.
    stream["sigma_rand_ch4"] = ("time", np.full(1800, 0.6), {"uncertainty_provenance": "declared"})
    declared = rolling_state(stream, instrument="a", baseline=settings)["noise_ch4"]
    assert (declared.values == 0.6).all()
    assert declared.attrs["uncertainty_provenance"] == "declared"
    assert "tsara_noise_estimator" not in declared.attrs
