"""Tests for parametric and bootstrap background rendering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tsara.synthetic.background import (
    TsaraSyntheticError,
    _median_period_s,
    render_background,
)
from tsara.synthetic.config import BootstrapBackground, ParametricBackground
from tsara.synthetic.profiling import RealDataProfile


def _times(n: int, freq: str = "1s", start: str = "2026-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq=freq)


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------


def test_constant_background_is_exactly_the_offset() -> None:
    values = render_background(
        ParametricBackground(kind="parametric", offset=1900.0),
        _times(100),
        np.random.default_rng(0),
    )
    assert np.all(values == 1900.0)


def test_diurnal_minimum_falls_at_midnight_for_zero_phase() -> None:
    """Phase 0 is documented to put the *minimum* at midnight UTC."""
    times = _times(24 * 60, freq="1min", start="2026-06-01")
    values = render_background(
        ParametricBackground(
            kind="parametric", offset=100.0, diurnal_amplitude=10.0, diurnal_period="24h"
        ),
        times,
        np.random.default_rng(0),
    )
    assert values.min() == pytest.approx(90.0, abs=1e-6)
    assert values.max() == pytest.approx(110.0, abs=0.05)
    assert times[int(np.argmin(values))].hour == 0


def test_diurnal_phase_shifts_the_minimum() -> None:
    times = _times(24 * 60, freq="1min", start="2026-06-01")
    values = render_background(
        ParametricBackground(
            kind="parametric",
            offset=100.0,
            diurnal_amplitude=10.0,
            diurnal_phase_hours=6.0,
        ),
        times,
        np.random.default_rng(0),
    )
    assert times[int(np.argmin(values))].hour == 6


def test_two_streams_at_different_rates_breathe_in_phase() -> None:
    """The diurnal term is phased off the epoch, not each stream's start."""
    config = ParametricBackground(kind="parametric", offset=0.0, diurnal_amplitude=10.0)
    rng = np.random.default_rng(0)
    fast = render_background(config, _times(3600, freq="1s"), rng)
    slow = render_background(config, _times(360, freq="10s"), rng)
    # Same wall-clock instants must give the same background value.
    assert fast[::10] == pytest.approx(slow)


def test_drift_is_linear_in_days() -> None:
    times = _times(49, freq="1h")
    values = render_background(
        ParametricBackground(kind="parametric", offset=0.0, drift_per_day=24.0),
        times,
        np.random.default_rng(0),
    )
    assert values[0] == pytest.approx(0.0)
    assert values[24] == pytest.approx(24.0)
    assert values[48] == pytest.approx(48.0)


def test_negative_drift_is_allowed() -> None:
    values = render_background(
        ParametricBackground(kind="parametric", offset=100.0, drift_per_day=-10.0),
        _times(25, freq="1h"),
        np.random.default_rng(0),
    )
    assert values[-1] == pytest.approx(90.0)


def test_random_walk_magnitude_is_sampling_rate_independent() -> None:
    """A 10 Hz stream must not wander ten times further than a 1 Hz one."""
    config = ParametricBackground(kind="parametric", offset=0.0, random_walk_std=10.0)

    def endpoint_spread(freq: str, periods: int) -> float:
        ends = [
            render_background(config, _times(periods, freq=freq), np.random.default_rng(s))[-1]
            for s in range(400)
        ]
        return float(np.std(ends))

    slow = endpoint_spread("10s", 8641)  # one day
    fast = endpoint_spread("1s", 86401)  # one day
    assert slow == pytest.approx(10.0, rel=0.15)
    assert fast == pytest.approx(10.0, rel=0.15)


def test_random_walk_starts_at_the_offset() -> None:
    values = render_background(
        ParametricBackground(kind="parametric", offset=50.0, random_walk_std=5.0),
        _times(100),
        np.random.default_rng(0),
    )
    assert values[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_requires_the_named_profile() -> None:
    with pytest.raises(TsaraSyntheticError, match="was not supplied"):
        render_background(
            BootstrapBackground(kind="bootstrap", profile="missing"),
            _times(10),
            np.random.default_rng(0),
            profiles={},
        )


def test_bootstrap_error_lists_available_profiles(
    white_noise_profile: RealDataProfile,
) -> None:
    with pytest.raises(TsaraSyntheticError, match=r"\['white'\]"):
        render_background(
            BootstrapBackground(kind="bootstrap", profile="other"),
            _times(10),
            np.random.default_rng(0),
            profiles={"white": white_noise_profile},
        )


def test_bootstrap_without_profiles_argument_raises() -> None:
    with pytest.raises(TsaraSyntheticError):
        render_background(
            BootstrapBackground(kind="bootstrap", profile="p"),
            _times(10),
            np.random.default_rng(0),
        )


def test_bootstrap_centres_on_the_profile_median(
    white_noise_profile: RealDataProfile,
) -> None:
    values = render_background(
        BootstrapBackground(kind="bootstrap", profile="white"),
        _times(4000),
        np.random.default_rng(0),
        profiles={"white": white_noise_profile},
    )
    assert values.mean() == pytest.approx(1900.0, abs=0.5)
    assert values.std() == pytest.approx(3.0, rel=0.1)


def test_bootstrap_scale_multiplies_the_fluctuations(
    white_noise_profile: RealDataProfile,
) -> None:
    values = render_background(
        BootstrapBackground(kind="bootstrap", profile="white", scale=3.0),
        _times(4000),
        np.random.default_rng(0),
        profiles={"white": white_noise_profile},
    )
    assert values.std() == pytest.approx(9.0, rel=0.1)


def test_bootstrap_layers_onto_a_parametric_base(
    white_noise_profile: RealDataProfile,
) -> None:
    values = render_background(
        BootstrapBackground(
            kind="bootstrap",
            profile="white",
            base=ParametricBackground(kind="parametric", offset=50.0, diurnal_amplitude=20.0),
        ),
        _times(2000),
        np.random.default_rng(0),
        profiles={"white": white_noise_profile},
    )
    # The base sets the level, not the profile's own median.
    assert 20.0 < values.mean() < 60.0


def test_bootstrap_produces_exactly_the_requested_length(
    white_noise_profile: RealDataProfile,
) -> None:
    """Block length need not divide the record length."""
    values = render_background(
        BootstrapBackground(kind="bootstrap", profile="white"),
        _times(1001),
        np.random.default_rng(0),
        profiles={"white": white_noise_profile},
    )
    assert values.shape == (1001,)


def test_bootstrap_preserves_block_autocorrelation() -> None:
    """Block resampling must not flatten correlated noise into white noise."""
    rng = np.random.default_rng(0)
    correlated = np.zeros((60, 256))
    for row in range(60):
        series = np.zeros(256)
        for i in range(1, 256):
            series[i] = 0.9 * series[i - 1] + rng.normal()
        correlated[row] = series - series.mean()

    profile = RealDataProfile(
        name="red",
        residual_blocks=correlated,
        residual_sigma=float(correlated.std()),
        noise_sigma=1.0,
        lag1_autocorr=0.9,
        decorrelation_timescale_s=9.5,
        background_median=0.0,
        background_iqr=1.0,
        sample_period_s=1.0,
        n_source_points=15360,
    )
    values = render_background(
        BootstrapBackground(kind="bootstrap", profile="red"),
        _times(8000),
        np.random.default_rng(1),
        profiles={"red": profile},
    )
    lag1 = np.corrcoef(values[:-1], values[1:])[0, 1]
    # Seams between blocks dilute it slightly, but it stays clearly red.
    assert lag1 > 0.8


def test_bootstrap_warns_on_a_sampling_rate_mismatch(
    white_noise_profile: RealDataProfile, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="tsara.synthetic.background"):
        render_background(
            BootstrapBackground(kind="bootstrap", profile="white"),
            _times(500, freq="10s"),
            np.random.default_rng(0),
            profiles={"white": white_noise_profile},
        )
    assert "replayed onto" in caplog.text


def test_bootstrap_is_quiet_at_a_matching_rate(
    white_noise_profile: RealDataProfile, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="tsara.synthetic.background"):
        render_background(
            BootstrapBackground(kind="bootstrap", profile="white"),
            _times(500, freq="1s"),
            np.random.default_rng(0),
            profiles={"white": white_noise_profile},
        )
    assert caplog.text == ""


def test_bootstrap_skips_the_rate_check_for_a_single_sample(
    white_noise_profile: RealDataProfile,
) -> None:
    values = render_background(
        BootstrapBackground(kind="bootstrap", profile="white"),
        _times(1),
        np.random.default_rng(0),
        profiles={"white": white_noise_profile},
    )
    assert values.shape == (1,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_median_period_of_a_single_sample_is_zero() -> None:
    assert _median_period_s(_times(1)) == 0.0


def test_median_period_of_a_regular_index() -> None:
    assert _median_period_s(_times(10, freq="5s")) == pytest.approx(5.0)
