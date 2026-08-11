"""Shared fixtures for the synthetic-data test suite.

The builders below return *known-good* objects that individual tests copy and
corrupt one field at a time, matching the convention established in the
top-level conftest: everything is valid except the thing under test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.synthetic.config import (
    EMGShape,
    GaussianShape,
    InstrumentSpec,
    LognormalAmplitude,
    ParametricBackground,
    RatioSpec,
    SourceSpec,
    SpeciesSpec,
    StationarySite,
    SyntheticConfig,
    TrueComponent,
    TrueUncertainty,
)
from tsara.synthetic.profiling import RealDataProfile

START = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture()
def flat_background() -> ParametricBackground:
    """A perfectly flat background: isolates plume/noise effects in a test."""
    return ParametricBackground(kind="parametric", offset=1900.0)


@pytest.fixture()
def noise_free_config(flat_background: ParametricBackground) -> SyntheticConfig:
    """One instrument, two gases, no noise, one source with a fixed ratio.

    The controlled case: with no noise and zero ratio spread, injected
    enhancements must reproduce the configured ratio *exactly*, so any
    deviation is a bug in the generator rather than a statistical fluctuation.
    """
    return SyntheticConfig(
        name="noise_free",
        start=START,
        duration="1h",
        seed=1234,
        platform=StationarySite(kind="stationary", latitude=40.0, longitude=-111.0),
        instruments={
            "analyzer": InstrumentSpec(
                native_rate="1s",
                species={
                    "ch4": SpeciesSpec(background=flat_background, units="ppb"),
                    "c2h6": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=2.0),
                        units="ppb",
                    ),
                },
            )
        },
        sources={
            "pad": SourceSpec(
                rate_per_hour=10.0,
                shape=GaussianShape(kind="gaussian", sigma="20s"),
                reference_species="ch4",
                amplitude=LognormalAmplitude(kind="lognormal", median=100.0, sigma_log=0.4),
                ratios={"c2h6": RatioSpec(mean=0.05)},
            )
        },
    )


@pytest.fixture()
def noisy_config(flat_background: ParametricBackground) -> SyntheticConfig:
    """One instrument with a full two-component error budget and a reported column."""
    return SyntheticConfig(
        name="noisy",
        start=START,
        duration="1h",
        seed=99,
        platform=StationarySite(
            kind="stationary", latitude=40.0, longitude=-111.0, altitude_m=1500.0
        ),
        instruments={
            "analyzer": InstrumentSpec(
                native_rate="1s",
                species={
                    "ch4": SpeciesSpec(
                        background=flat_background,
                        units="ppb",
                        uncertainty=TrueUncertainty(
                            random=TrueComponent(absolute=2.0, report_as="ch4_err"),
                            systematic=TrueComponent(relative=0.01),
                        ),
                    )
                },
            )
        },
        sources={
            "pad": SourceSpec(
                rate_per_hour=4.0,
                shape=EMGShape(kind="emg", sigma="15s", tau="30s"),
                reference_species="ch4",
                amplitude=LognormalAmplitude(kind="lognormal", median=80.0, sigma_log=0.5),
            )
        },
    )


@pytest.fixture()
def source_dict() -> dict[str, Any]:
    """Raw dict for a valid source, for validation-failure tests."""
    return {
        "rate_per_hour": 2.0,
        "shape": {"kind": "gaussian", "sigma": "20s"},
        "reference_species": "ch4",
        "amplitude": {"kind": "lognormal", "median": 100.0, "sigma_log": 0.5},
        "ratios": {"c2h6": {"mean": 0.05}},
    }


@pytest.fixture()
def synthetic_dict() -> dict[str, Any]:
    """Raw dict for a valid minimal SyntheticConfig, for validation tests."""
    return {
        "name": "cfg",
        "start": "2026-01-01T00:00:00Z",
        "duration": "1h",
        "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
        "instruments": {
            "analyzer": {
                "native_rate": "1s",
                "species": {
                    "ch4": {
                        "background": {"kind": "parametric", "offset": 1900.0},
                        "units": "ppb",
                    }
                },
            }
        },
    }


@pytest.fixture()
def white_noise_profile() -> RealDataProfile:
    """A profile built from synthetic white noise, for bootstrap tests."""
    rng = np.random.default_rng(0)
    blocks = rng.normal(0.0, 3.0, size=(40, 128))
    blocks -= blocks.mean(axis=1, keepdims=True)
    return RealDataProfile(
        name="white",
        residual_blocks=blocks,
        residual_sigma=3.0,
        noise_sigma=3.0,
        lag1_autocorr=0.0,
        decorrelation_timescale_s=None,
        background_median=1900.0,
        background_iqr=4.0,
        sample_period_s=1.0,
        n_source_points=5120,
    )


@pytest.fixture()
def autocorrelated_series() -> pd.Series:
    """A realistic-ish 1 Hz record: drifting background, AR(1) noise, plumes.

    Deliberately plume-dense, matching the owner's real data, so profiling
    tests exercise the leakage path rather than an idealized clean record.
    """
    rng = np.random.default_rng(11)
    n = 6000
    times = pd.date_range("2026-03-01", periods=n, freq="1s")

    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0.0, 1.0)

    background = 1900.0 + 6.0 * np.sin(np.arange(n) / 900.0)
    plumes = np.zeros(n)
    for centre in (800, 2100, 2400, 4500):
        plumes += 60.0 * np.exp(-0.5 * ((np.arange(n) - centre) / 40.0) ** 2)
    return pd.Series(background + noise + plumes, index=times)
