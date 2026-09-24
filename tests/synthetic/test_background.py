"""Tests for realizing a field's background, parametric or bootstrapped."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.config.synthetic import BootstrapBackground, ParametricBackground
from tsara.core.timebase import SECOND_NS, SECONDS_PER_DAY
from tsara.synthetic.background import (
    RealizedBackground,
    TsaraSyntheticError,
    realize_background,
)
from tsara.synthetic.profiling import RealDataProfile

#: The campaign every test realizes over: 2026-01-01, two days.
START_NS = int(pd.Timestamp("2026-01-01").value)
DAY_NS = 86_400 * 1_000_000_000


def _realize(
    config: ParametricBackground | BootstrapBackground,
    *,
    days: float = 2.0,
    resolution_s: float = 1.0,
    seed: int = 0,
    profiles: dict[str, RealDataProfile] | None = None,
) -> RealizedBackground:
    return realize_background(
        config,
        start_ns=START_NS,
        end_ns=START_NS + int(days * DAY_NS),
        truth_resolution_ns=int(resolution_s * SECOND_NS),
        rng=np.random.default_rng(seed),
        profiles=profiles,
        field="ch4",
    )


def _seconds(offsets_s: Any) -> np.ndarray:
    """Epoch seconds at ``offsets_s`` past the campaign start."""
    return START_NS / 1e9 + np.asarray(offsets_s, dtype=np.float64)


# ---------------------------------------------------------------------------
# Parametric
# ---------------------------------------------------------------------------


def test_constant_background_is_exactly_the_offset() -> None:
    background = _realize(ParametricBackground(kind="parametric", offset=1900.0))
    assert np.all(background.at(_seconds(np.linspace(-50.0, 90_000.0, 101))) == 1900.0)


def test_diurnal_minimum_falls_at_midnight_for_zero_phase() -> None:
    """Phase 0 is documented to put the *minimum* at midnight UTC."""
    background = _realize(
        ParametricBackground(
            kind="parametric", offset=100.0, diurnal_amplitude=10.0, diurnal_period="24h"
        )
    )
    minutes = np.arange(24 * 60) * 60.0
    values = background.at(_seconds(minutes))
    assert values.min() == pytest.approx(90.0, abs=1e-6)
    assert values.max() == pytest.approx(110.0, abs=0.05)
    assert int(np.argmin(values)) == 0  # the campaign starts at midnight


def test_diurnal_phase_shifts_the_minimum() -> None:
    background = _realize(
        ParametricBackground(
            kind="parametric", offset=100.0, diurnal_amplitude=10.0, diurnal_phase_hours=6.0
        )
    )
    values = background.at(_seconds(np.arange(24 * 60) * 60.0))
    assert int(np.argmin(values)) == 6 * 60


def test_one_realization_answers_every_rate_identically() -> None:
    """Replaces "two streams breathe in phase": there is now one background.

    Asked at 1 s and at 10 s, the same instants return the same numbers,
    stochastic term included -- the property that makes two instruments on one
    field measure one atmosphere.
    """
    background = _realize(
        ParametricBackground(
            kind="parametric", offset=0.0, diurnal_amplitude=10.0, random_walk_std=20.0
        )
    )
    fast = background.at(_seconds(np.arange(3600.0)))
    slow = background.at(_seconds(np.arange(0.0, 3600.0, 10.0)))
    assert np.array_equal(fast[::10], slow)


def test_drift_is_linear_in_days_from_the_campaign_start() -> None:
    background = _realize(ParametricBackground(kind="parametric", offset=0.0, drift_per_day=24.0))
    values = background.at(_seconds([0.0, SECONDS_PER_DAY, 2 * SECONDS_PER_DAY]))
    assert values == pytest.approx([0.0, 24.0, 48.0])


def test_drift_does_not_restart_where_a_query_begins() -> None:
    """The defect the atmosphere removed: drift from the first rendered sample.

    A query starting an hour in reads an hour of drift, not zero -- which is
    what made a mean instrument's sub-samples and a point instrument's
    samples disagree by a constant before Phase 4.5.
    """
    background = _realize(ParametricBackground(kind="parametric", offset=0.0, drift_per_day=24.0))
    assert background.at(_seconds([3600.0, 3601.0])) == pytest.approx([1.0, 1.0 + 1.0 / 3600.0])


def test_negative_drift_is_allowed() -> None:
    background = _realize(
        ParametricBackground(kind="parametric", offset=100.0, drift_per_day=-10.0)
    )
    assert background.at(_seconds([SECONDS_PER_DAY]))[0] == pytest.approx(90.0)


def test_a_background_without_a_stochastic_term_draws_nothing() -> None:
    """What byte-identity through the redesign rests on.

    Every configuration without a random walk must leave the run's generator
    exactly where it found it, or every later noise draw would move.
    """
    rng = np.random.default_rng(5)
    before = copy.deepcopy(rng.bit_generator.state)
    background = realize_background(
        ParametricBackground(
            kind="parametric", offset=1.0, diurnal_amplitude=3.0, drift_per_day=2.0
        ),
        start_ns=START_NS,
        end_ns=START_NS + DAY_NS,
        truth_resolution_ns=SECOND_NS,
        rng=rng,
    )
    assert rng.bit_generator.state == before
    assert background.stochastic == ()


def test_random_walk_magnitude_does_not_depend_on_the_node_spacing() -> None:
    """Increments scale with sqrt(dt): a finer truth clock resolves, not wanders."""
    config = ParametricBackground(kind="parametric", offset=0.0, random_walk_std=10.0)

    def endpoint_spread(resolution_s: float) -> float:
        ends = [
            _realize(config, days=1.0, resolution_s=resolution_s, seed=s).at(
                _seconds([SECONDS_PER_DAY])
            )[0]
            for s in range(400)
        ]
        return float(np.std(ends))

    assert endpoint_spread(600.0) == pytest.approx(10.0, rel=0.15)
    assert endpoint_spread(10.0) == pytest.approx(10.0, rel=0.15)


def test_random_walk_starts_at_the_offset() -> None:
    background = _realize(ParametricBackground(kind="parametric", offset=50.0, random_walk_std=5.0))
    assert background.at(_seconds([0.0]))[0] == pytest.approx(50.0)


def test_random_walk_is_linear_between_its_nodes() -> None:
    """D2: drawn on nodes, deterministic between them."""
    background = _realize(
        ParametricBackground(kind="parametric", offset=0.0, random_walk_std=50.0),
        resolution_s=10.0,
    )
    nodes = background.at(_seconds([120.0, 130.0]))
    assert background.at(_seconds([125.0]))[0] == pytest.approx(nodes.mean(), rel=0, abs=1e-12)
    assert background.at(_seconds([122.5]))[0] == pytest.approx(
        0.75 * nodes[0] + 0.25 * nodes[1], rel=0, abs=1e-12
    )


def test_a_random_walk_holds_its_edge_values_outside_the_campaign() -> None:
    """Cells may reach past the ends; the atmosphere stays independent of them."""
    background = _realize(
        ParametricBackground(kind="parametric", offset=0.0, random_walk_std=50.0), days=0.01
    )
    end_s = 0.01 * SECONDS_PER_DAY
    assert background.at(_seconds([-30.0]))[0] == background.at(_seconds([0.0]))[0]
    last_node = np.ceil(end_s)
    assert background.at(_seconds([last_node + 30.0]))[0] == background.at(_seconds([last_node]))[0]


def test_a_campaign_shorter_than_one_node_spacing_still_has_two_nodes() -> None:
    background = _realize(
        ParametricBackground(kind="parametric", offset=0.0, random_walk_std=5.0),
        days=1.0 / 86_400.0,
        resolution_s=3600.0,
    )
    (walk,) = background.stochastic
    assert walk.node_s.size == 2


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_requires_the_named_profile() -> None:
    with pytest.raises(TsaraSyntheticError, match="was not supplied"):
        _realize(BootstrapBackground(kind="bootstrap", profile="missing"), profiles={})


def test_bootstrap_error_names_the_field_and_lists_available_profiles(
    white_noise_profile: RealDataProfile,
) -> None:
    with pytest.raises(TsaraSyntheticError, match=r"Field 'ch4'.*\['white'\]"):
        _realize(
            BootstrapBackground(kind="bootstrap", profile="other"),
            profiles={"white": white_noise_profile},
        )


def test_bootstrap_without_profiles_argument_raises() -> None:
    with pytest.raises(TsaraSyntheticError):
        _realize(BootstrapBackground(kind="bootstrap", profile="p"))


def test_a_profile_without_a_sampling_period_is_refused(
    white_noise_profile: RealDataProfile,
) -> None:
    import dataclasses

    flat = dataclasses.replace(white_noise_profile, sample_period_s=0.0)
    with pytest.raises(TsaraSyntheticError, match="no cadence"):
        _realize(BootstrapBackground(kind="bootstrap", profile="white"), profiles={"white": flat})


def test_bootstrap_centres_on_the_profile_median(
    white_noise_profile: RealDataProfile,
) -> None:
    background = _realize(
        BootstrapBackground(kind="bootstrap", profile="white"),
        profiles={"white": white_noise_profile},
    )
    values = background.at(_seconds(np.arange(4000.0)))
    assert values.mean() == pytest.approx(1900.0, abs=0.5)
    assert values.std() == pytest.approx(3.0, rel=0.1)


def test_bootstrap_scale_multiplies_the_fluctuations(
    white_noise_profile: RealDataProfile,
) -> None:
    background = _realize(
        BootstrapBackground(kind="bootstrap", profile="white", scale=3.0),
        profiles={"white": white_noise_profile},
    )
    assert background.at(_seconds(np.arange(4000.0))).std() == pytest.approx(9.0, rel=0.1)


def test_bootstrap_layers_onto_a_parametric_base(
    white_noise_profile: RealDataProfile,
) -> None:
    background = _realize(
        BootstrapBackground(
            kind="bootstrap",
            profile="white",
            base=ParametricBackground(kind="parametric", offset=50.0, diurnal_amplitude=20.0),
        ),
        profiles={"white": white_noise_profile},
    )
    # The base sets the level, not the profile's own median.
    assert 20.0 < background.at(_seconds(np.arange(2000.0))).mean() < 60.0


def test_a_base_random_walk_is_added_after_the_fluctuations(
    white_noise_profile: RealDataProfile,
) -> None:
    background = _realize(
        BootstrapBackground(
            kind="bootstrap",
            profile="white",
            base=ParametricBackground(kind="parametric", offset=0.0, random_walk_std=10.0),
        ),
        profiles={"white": white_noise_profile},
    )
    fluctuations, walk = background.stochastic
    assert np.diff(fluctuations.node_s)[0] == pytest.approx(white_noise_profile.sample_period_s)
    assert walk.values[0] == 0.0


def test_bootstrap_replays_at_the_profile_period_whatever_the_query_spacing() -> None:
    """Replaces the rate-mismatch warning: there is no mismatch to warn about.

    Blocks used to be replayed sample-for-sample on each instrument's clock,
    so a 2 s profile on a 1 s instrument halved every real timescale. Nodes
    are now the profile's own samples, so the correlation a profile carries
    is the correlation the field has, at any query spacing.
    """
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
        decorrelation_timescale_s=19.0,
        background_median=0.0,
        background_iqr=1.0,
        sample_period_s=2.0,
        n_record_points=15360,
    )
    background = _realize(
        BootstrapBackground(kind="bootstrap", profile="red"), profiles={"red": profile}
    )
    (fluctuations,) = background.stochastic
    assert np.allclose(np.diff(fluctuations.node_s), 2.0)
    # Sampled at the profile's own 2 s, the lag-1 correlation is the profile's,
    # diluted only slightly by seams between blocks.
    at_profile_rate = background.at(_seconds(np.arange(0.0, 16_000.0, 2.0)))
    lag1 = np.corrcoef(at_profile_rate[:-1], at_profile_rate[1:])[0, 1]
    assert lag1 > 0.8
