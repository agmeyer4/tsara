"""Tests for plume shape kernels, event scheduling, and the ground-truth catalog."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsara.synthetic.config import (
    EMGShape,
    GaussianShape,
    LognormalAmplitude,
    NestedSpec,
    RatioSpec,
    SyntheticConfig,
    UniformAmplitude,
)
from tsara.synthetic.plumes import (
    GROUND_TRUTH_COLUMNS,
    GroundTruth,
    GroundTruthEvent,
    _draw_amplitude,
    _draw_ratio,
    _emg_log_shape,
    _is_missing,
    build_kernel,
    schedule_events,
)

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def test_gaussian_kernel_is_unit_peak_and_symmetric() -> None:
    kernel = build_kernel(GaussianShape(kind="gaussian", sigma="10s"))
    assert kernel.peak_offset_s == 0.0
    assert kernel.support_before_s == kernel.support_after_s
    assert kernel.evaluate(np.array([0.0]))[0] == pytest.approx(1.0)
    # Symmetry about the centre.
    left = kernel.evaluate(np.array([-7.0]))[0]
    right = kernel.evaluate(np.array([7.0]))[0]
    assert left == pytest.approx(right)


def test_gaussian_kernel_matches_the_analytic_form() -> None:
    kernel = build_kernel(GaussianShape(kind="gaussian", sigma="10s"))
    dt = np.array([-15.0, -5.0, 0.0, 5.0, 15.0])
    assert kernel.evaluate(dt) == pytest.approx(np.exp(-0.5 * (dt / 10.0) ** 2))


def test_kernel_is_zero_outside_its_support() -> None:
    kernel = build_kernel(GaussianShape(kind="gaussian", sigma="10s"))
    outside = np.array([-1000.0, 1000.0])
    assert np.all(kernel.evaluate(outside) == 0.0)


def test_kernel_handles_an_entirely_out_of_support_window() -> None:
    """The early-return branch when no sample falls inside the support."""
    kernel = build_kernel(GaussianShape(kind="gaussian", sigma="10s"))
    result = kernel.evaluate(np.array([5000.0, 6000.0]))
    assert result.shape == (2,)
    assert np.all(result == 0.0)


def test_emg_kernel_is_unit_peak_and_right_skewed() -> None:
    kernel = build_kernel(EMGShape(kind="emg", sigma="10s", tau="40s"))
    # The exponential tail drags the mode later than the Gaussian centre.
    assert kernel.peak_offset_s > 0.0
    # And makes the support asymmetric.
    assert kernel.support_after_s > kernel.support_before_s
    peak = kernel.evaluate(np.array([kernel.peak_offset_s]))[0]
    assert peak == pytest.approx(1.0, rel=1e-9)


def test_emg_never_exceeds_unit_peak() -> None:
    """`sampled_peak_amplitude` must stay bounded by `true_amplitude`."""
    kernel = build_kernel(EMGShape(kind="emg", sigma="8s", tau="25s"))
    dense = np.linspace(-kernel.support_before_s, kernel.support_after_s, 200_000)
    assert kernel.evaluate(dense).max() <= 1.0 + 1e-12


def test_emg_tail_decays_exponentially_with_tau() -> None:
    """Far down the tail the shape must fall as exp(-t/tau)."""
    tau_s = 30.0
    kernel = build_kernel(EMGShape(kind="emg", sigma="5s", tau="30s"))
    t1, t2 = 100.0, 130.0
    ratio = kernel.evaluate(np.array([t2]))[0] / kernel.evaluate(np.array([t1]))[0]
    assert ratio == pytest.approx(np.exp(-(t2 - t1) / tau_s), rel=1e-3)


def test_emg_asymptotic_branch_is_continuous_with_the_stable_branch() -> None:
    """A tau >> sigma shape pushes the far tail through the erfcx cutoff.

    Both branches must agree where they meet, otherwise the shape would have
    a step discontinuity in its tail.
    """
    sigma_over_tau = 0.05  # tau = 20 * sigma, so the tail reaches z < -26
    u = np.linspace(30.0, 45.0, 2001)
    values = _emg_log_shape(u, sigma_over_tau)
    # Smooth and monotonically decreasing across the branch switch.
    diffs = np.diff(values)
    assert np.all(diffs < 0.0)
    assert np.all(np.isfinite(values))
    assert np.abs(np.diff(diffs)).max() < 1e-6


def test_emg_with_large_tau_is_finite_everywhere() -> None:
    kernel = build_kernel(EMGShape(kind="emg", sigma="1s", tau="60s"))
    dense = np.linspace(-kernel.support_before_s, kernel.support_after_s, 50_000)
    values = kernel.evaluate(dense)
    assert np.all(np.isfinite(values))
    assert values.max() == pytest.approx(1.0, rel=1e-6)


def test_kernel_duration_property() -> None:
    kernel = build_kernel(GaussianShape(kind="gaussian", sigma="10s"))
    assert kernel.duration_s == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------


def test_uniform_amplitude_draws_stay_in_range() -> None:
    rng = np.random.default_rng(0)
    spec = UniformAmplitude(kind="uniform", low=10.0, high=20.0)
    draws = [_draw_amplitude(spec, rng) for _ in range(500)]
    assert all(10.0 <= value <= 20.0 for value in draws)


def test_lognormal_amplitude_draws_recover_the_median() -> None:
    rng = np.random.default_rng(0)
    spec = LognormalAmplitude(kind="lognormal", median=100.0, sigma_log=0.5)
    draws = np.array([_draw_amplitude(spec, rng) for _ in range(20_000)])
    assert np.median(draws) == pytest.approx(100.0, rel=0.03)
    assert np.all(draws > 0.0)


def test_zero_spread_ratio_draws_are_identical() -> None:
    rng = np.random.default_rng(0)
    spec = RatioSpec(mean=0.05)
    assert {_draw_ratio(spec, rng) for _ in range(20)} == {0.05}


def test_spread_ratio_draws_recover_the_mean() -> None:
    rng = np.random.default_rng(3)
    spec = RatioSpec(mean=0.05, relative_spread=0.2)
    draws = np.array([_draw_ratio(spec, rng) for _ in range(50_000)])
    assert draws.mean() == pytest.approx(0.05, rel=0.01)
    assert draws.std() / draws.mean() == pytest.approx(0.2, rel=0.05)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def test_schedule_events_produces_roughly_the_configured_rate(
    noise_free_config: SyntheticConfig,
) -> None:
    """Poisson counts over many runs must centre on rate x duration."""
    counts = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        counts.append(len(schedule_events(noise_free_config, rng)))
    # 10 per hour over 1 hour.
    assert np.mean(counts) == pytest.approx(10.0, abs=1.5)


def test_schedule_events_is_sorted_by_centre_time(
    noise_free_config: SyntheticConfig,
) -> None:
    events = schedule_events(noise_free_config, np.random.default_rng(0))
    centres = [event.center_time for event in events]
    assert centres == sorted(centres)


def test_schedule_events_realizes_the_configured_ratio(
    noise_free_config: SyntheticConfig,
) -> None:
    events = schedule_events(noise_free_config, np.random.default_rng(0))
    assert events
    for event in events:
        assert event.ratios["ch4"] == 1.0
        assert event.ratios["c2h6"] == pytest.approx(0.05)
        # The defining relation between amplitude and ratio.
        assert event.amplitudes["c2h6"] == pytest.approx(
            event.amplitudes["ch4"] * event.ratios["c2h6"]
        )


def test_zero_drawn_events_is_handled(noise_free_config: SyntheticConfig) -> None:
    """A source can legitimately draw zero events in a short record."""
    quiet = noise_free_config.model_copy(
        update={
            "sources": {
                "pad": noise_free_config.sources["pad"].model_copy(update={"rate_per_hour": 1e-9})
            }
        }
    )
    assert schedule_events(quiet, np.random.default_rng(0)) == []


def test_config_without_sources_schedules_nothing(
    noise_free_config: SyntheticConfig,
) -> None:
    empty = noise_free_config.model_copy(update={"sources": {}})
    assert schedule_events(empty, np.random.default_rng(0)) == []


def test_inter_species_lag_shifts_only_the_lagged_species(
    noise_free_config: SyntheticConfig,
) -> None:
    lagged = noise_free_config.model_copy(
        update={
            "sources": {
                "pad": noise_free_config.sources["pad"].model_copy(
                    update={"inter_species_lag": {"c2h6": "45s"}}
                )
            }
        }
    )
    events = schedule_events(lagged, np.random.default_rng(0))
    assert events
    event = events[0]
    assert event.species_center("ch4") == event.center_time
    assert event.species_center("c2h6") - event.center_time == pd.Timedelta("45s")
    assert event.species_peak_time("c2h6") - event.species_peak_time("ch4") == pd.Timedelta("45s")


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def _nested_config(base: SyntheticConfig, **nested_kwargs: object) -> SyntheticConfig:
    nested = NestedSpec(
        probability=1.0,
        shape=GaussianShape(kind="gaussian", sigma="3s"),
        amplitude_factor=0.5,
        **nested_kwargs,  # type: ignore[arg-type]
    )
    return base.model_copy(
        update={"sources": {"pad": base.sources["pad"].model_copy(update={"nested": nested})}}
    )


def test_nested_children_are_linked_and_inside_their_parent(
    noise_free_config: SyntheticConfig,
) -> None:
    events = schedule_events(_nested_config(noise_free_config), np.random.default_rng(0))
    by_id = {event.event_id: event for event in events}
    children = [event for event in events if event.parent_event_id is not None]
    assert children

    for child in children:
        parent = by_id[child.parent_event_id or ""]
        # Placed within one parent sigma of the parent centre.
        offset_s = abs((child.center_time - parent.center_time).total_seconds())
        assert offset_s <= parent.kernel.sigma_s
        # Narrower than its parent, and scaled by amplitude_factor.
        assert child.kernel.sigma_s < parent.kernel.sigma_s
        assert child.amplitudes["ch4"] == pytest.approx(0.5 * parent.amplitudes["ch4"])


def test_nested_child_inherits_parent_ratios_by_default(
    noise_free_config: SyntheticConfig,
) -> None:
    events = schedule_events(_nested_config(noise_free_config), np.random.default_rng(0))
    by_id = {event.event_id: event for event in events}
    for child in (e for e in events if e.parent_event_id is not None):
        parent = by_id[child.parent_event_id or ""]
        assert child.ratios == parent.ratios


def test_nested_child_can_carry_its_own_ratio(
    noise_free_config: SyntheticConfig,
) -> None:
    """A chemically distinct child superimposed on its parent."""
    config = _nested_config(noise_free_config, ratios={"c2h6": RatioSpec(mean=0.4)})
    events = schedule_events(config, np.random.default_rng(0))
    children = [event for event in events if event.parent_event_id is not None]
    assert children
    for child in children:
        assert child.ratios["c2h6"] == pytest.approx(0.4)
        # The reference species is still 1 by definition.
        assert child.ratios["ch4"] == 1.0


def test_partial_nested_ratio_override_keeps_other_species(
    noise_free_config: SyntheticConfig,
) -> None:
    """Overriding one species must not silently delete the parent's others."""
    source = noise_free_config.sources["pad"].model_copy(
        update={
            "ratios": {
                "c2h6": RatioSpec(mean=0.05),
                "co2": RatioSpec(mean=10.0),
            }
        }
    )
    config = noise_free_config.model_copy(update={"sources": {"pad": source}})
    config = _nested_config(config, ratios={"c2h6": RatioSpec(mean=0.4)})
    events = schedule_events(config, np.random.default_rng(0))
    children = [event for event in events if event.parent_event_id is not None]
    assert children
    for child in children:
        assert child.ratios["c2h6"] == pytest.approx(0.4)
        # co2 was not overridden, so it inherits the parent's realized value.
        assert child.ratios["co2"] == pytest.approx(10.0)


def test_zero_nesting_probability_produces_no_children(
    noise_free_config: SyntheticConfig,
) -> None:
    config = noise_free_config.model_copy(
        update={
            "sources": {
                "pad": noise_free_config.sources["pad"].model_copy(
                    update={
                        "nested": NestedSpec(
                            probability=0.0,
                            shape=GaussianShape(kind="gaussian", sigma="3s"),
                            amplitude_factor=0.5,
                        )
                    }
                )
            }
        }
    )
    events = schedule_events(config, np.random.default_rng(0))
    assert all(event.parent_event_id is None for event in events)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def _event(**overrides: object) -> GroundTruthEvent:
    fields: dict[str, object] = {
        "event_id": "pad_00001",
        "parent_event_id": None,
        "source_name": "pad",
        "species": "ch4",
        "instrument": "analyzer",
        "reference_species": "ch4",
        "start_time": pd.Timestamp("2026-01-01T00:00:00"),
        "peak_time": pd.Timestamp("2026-01-01T00:01:00"),
        "end_time": pd.Timestamp("2026-01-01T00:02:00"),
        "true_amplitude": 100.0,
        "sampled_peak_amplitude": 99.5,
        "true_baseline_at_peak": 1900.0,
        "true_ratio_to_reference": 1.0,
        "latitude": 40.0,
        "longitude": -111.0,
    }
    fields.update(overrides)
    return GroundTruthEvent(**fields)  # type: ignore[arg-type]


def test_ground_truth_frame_has_the_fixed_column_order() -> None:
    frame = GroundTruth(events=(_event(),)).to_frame()
    assert tuple(frame.columns) == GROUND_TRUTH_COLUMNS


def test_ground_truth_round_trips_through_a_frame() -> None:
    truth = GroundTruth(events=(_event(), _event(event_id="pad_00002", species="c2h6")))
    restored = GroundTruth.from_frame(truth.to_frame())
    assert len(restored) == 2
    assert restored.events[0].event_id == "pad_00001"
    assert restored.events[1].species == "c2h6"


def test_ground_truth_preserves_missing_parent_and_coordinates() -> None:
    truth = GroundTruth(events=(_event(parent_event_id=None, latitude=None, longitude=None),))
    restored = GroundTruth.from_frame(truth.to_frame())
    assert restored.events[0].parent_event_id is None
    assert restored.events[0].latitude is None
    assert restored.events[0].longitude is None


def test_ground_truth_preserves_a_parent_link() -> None:
    truth = GroundTruth(events=(_event(parent_event_id="pad_00000"),))
    restored = GroundTruth.from_frame(truth.to_frame())
    assert restored.events[0].parent_event_id == "pad_00000"


def test_empty_ground_truth_frame_has_typed_columns() -> None:
    """A plume-free control dataset must still save and load like any other."""
    frame = GroundTruth(events=()).to_frame()
    assert tuple(frame.columns) == GROUND_TRUTH_COLUMNS
    assert frame.empty
    assert frame["true_amplitude"].dtype == float
    assert pd.api.types.is_datetime64_any_dtype(frame["start_time"])
    assert len(GroundTruth.from_frame(frame)) == 0


def test_empty_and_populated_frames_share_one_time_representation() -> None:
    """The control case must be concatenable with every other case.

    The empty path has to impose dtypes by hand, so it is the one place the
    representation can drift. tz-naive nanoseconds on both sides keeps a
    plume-free run comparable with a plume-dense one.
    """
    empty = GroundTruth(events=()).to_frame()
    populated = GroundTruth(events=(_event(),)).to_frame()
    for column in ("start_time", "peak_time", "end_time"):
        assert empty[column].dt.tz is None
        assert populated[column].dt.tz is None
        assert empty[column].dtype == populated[column].dtype
    combined = pd.concat([empty, populated], ignore_index=True)
    assert len(GroundTruth.from_frame(combined)) == 1


def test_from_frame_rejects_missing_columns() -> None:
    frame = GroundTruth(events=(_event(),)).to_frame().drop(columns=["true_amplitude"])
    with pytest.raises(ValueError, match="missing columns"):
        GroundTruth.from_frame(frame)


def test_for_species_filters_rows() -> None:
    truth = GroundTruth(events=(_event(), _event(species="c2h6")))
    assert len(truth.for_species("c2h6")) == 1
    assert truth.for_species("c2h6")[0].species == "c2h6"


def test_event_ids_are_deduplicated_in_order() -> None:
    truth = GroundTruth(events=(_event(), _event(species="c2h6"), _event(event_id="pad_00002")))
    assert truth.event_ids == ("pad_00001", "pad_00002")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (float("nan"), True), (pd.NaT, True), (1.0, False), ("x", False)],
)
def test_is_missing(value: object, expected: bool) -> None:
    assert _is_missing(value) is expected


def test_emg_log_shape_all_asymptotic() -> None:
    """Every point past the erfcx cutoff must use the closed-form tail.

    Exercises the branch in isolation: with sigma/tau small and u large,
    z = (sigma/tau - u)/sqrt(2) is below the cutoff everywhere, so the
    stable erfcx path is never taken.
    """
    values = _emg_log_shape(np.linspace(60.0, 90.0, 200), 0.02)
    assert np.all(np.isfinite(values))
    # Pure exponential decay: the log-shape is linear in u.
    second_difference = np.diff(np.diff(values))
    assert np.abs(second_difference).max() < 1e-9
