"""End-to-end tests for the synthetic dataset generator.

The assertions here are the ones that matter most for the whole project: if
injected ground truth is not internally consistent, every later phase is
being scored against a broken answer key.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tsara.core.circular import wrap_degrees
from tsara.core.naming import (
    BOUNDS_ATTR,
    BOUNDS_DIM,
    CELL_METHODS_ATTR,
    SUPPORT_COVERAGE_ATTR,
    SUPPORT_LABEL_ATTR,
    SUPPORT_LABEL_PROVENANCE_ATTR,
    SUPPORT_METHOD_PROVENANCE_ATTR,
    SUPPORT_WIDTH_PROVENANCE_ATTR,
    TIME_BOUNDS_VAR,
    TIME_COORD,
)
from tsara.synthetic.atmosphere import realize_atmosphere
from tsara.synthetic.background import TsaraSyntheticError
from tsara.synthetic.config import (
    AtmosphereSpec,
    FieldSpec,
    GaussianShape,
    InstrumentSpec,
    MeasurementSpec,
    NestedSpec,
    ParametricBackground,
    RatioSpec,
    SyntheticConfig,
    TrueComponent,
    TrueUncertainty,
)
from tsara.synthetic.generator import (
    TRUTH_PREFIX,
    _build_cells,
    _build_times,
    generate,
)
from tsara.synthetic.profiling import RealDataProfile

WithSources = Callable[[SyntheticConfig, dict[str, Any]], SyntheticConfig]

SITE: dict[str, Any] = {"kind": "stationary", "latitude": 40.0, "longitude": -111.0}


def _one_instrument(
    *, name: str, duration: str, seed: int = 0, sources: dict[str, Any] | None = None, **inst: Any
) -> SyntheticConfig:
    """One instrument measuring a flat methane field; ``inst`` sets its clock."""
    return SyntheticConfig.model_validate(
        {
            "name": name,
            "start": "2026-01-01T00:00:00Z",
            "duration": duration,
            "seed": seed,
            "platform": SITE,
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1900.0}}
                },
                "sources": sources or {},
            },
            "instruments": {"inst": {"native_rate": "1s", "measures": {"ch4": {}}, **inst}},
        }
    )


def _add_field(
    config: SyntheticConfig, instrument: str, field: str, spec: dict[str, Any], **inst: Any
) -> SyntheticConfig:
    """Return ``config`` with one more field, measured by ``instrument``.

    The instrument is created when absent, with ``inst`` as its clock.
    """
    payload = config.model_dump()
    payload["atmosphere"]["fields"][field] = spec
    entry = payload["instruments"].setdefault(instrument, {"measures": {}, **inst})
    entry["measures"][field] = {}
    return SyntheticConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_generates_one_stream_per_instrument(noise_free_config: SyntheticConfig) -> None:
    dataset = generate(noise_free_config)
    assert set(dataset.streams) == {"analyzer"}
    assert dataset.config is noise_free_config


def test_stream_carries_observable_and_truth_variables(
    noise_free_config: SyntheticConfig,
) -> None:
    stream = generate(noise_free_config).streams["analyzer"]
    assert "ch4" in stream
    assert f"{TRUTH_PREFIX}background_ch4" in stream
    assert f"{TRUTH_PREFIX}enhancement_ch4" in stream


def test_truth_decomposition_sums_to_the_observable_without_noise(
    noise_free_config: SyntheticConfig,
) -> None:
    """With no error budget, observable == background + enhancement exactly."""
    stream = generate(noise_free_config).streams["analyzer"]
    reconstructed = (
        stream[f"{TRUTH_PREFIX}background_ch4"] + stream[f"{TRUTH_PREFIX}enhancement_ch4"]
    )
    assert np.allclose(stream["ch4"].values, reconstructed.values)


def test_observable_view_hides_the_answer_key(noise_free_config: SyntheticConfig) -> None:
    dataset = generate(noise_free_config)
    observable = dataset.observable("analyzer")
    assert set(observable.data_vars) == {"ch4", "c2h6"}
    assert not any(str(name).startswith(TRUTH_PREFIX) for name in observable.data_vars)


def test_observable_rejects_an_unknown_stream(noise_free_config: SyntheticConfig) -> None:
    dataset = generate(noise_free_config)
    with pytest.raises(KeyError, match="No stream named"):
        dataset.observable("nope")


def test_streams_self_describe_as_synthetic(noise_free_config: SyntheticConfig) -> None:
    """A synthetic file mistaken for a measurement is a scientific hazard."""
    attrs = generate(noise_free_config).streams["analyzer"].attrs
    assert attrs["tsara_stage"] == "synthetic"
    assert "SYNTHETIC DATA" in attrs["description"]
    assert attrs["synthetic_seed"] == noise_free_config.seed
    assert attrs["native_rate"] == "1s"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_the_dataset(noisy_config: SyntheticConfig) -> None:
    first = generate(noisy_config).streams["analyzer"]["ch4"].values
    second = generate(noisy_config).streams["analyzer"]["ch4"].values
    assert np.array_equal(first, second)


def test_different_seeds_produce_different_data(noisy_config: SyntheticConfig) -> None:
    other = noisy_config.model_copy(update={"seed": noisy_config.seed + 1})
    assert not np.array_equal(
        generate(noisy_config).streams["analyzer"]["ch4"].values,
        generate(other).streams["analyzer"]["ch4"].values,
    )


# ---------------------------------------------------------------------------
# Ground truth consistency — the load-bearing assertions
# ---------------------------------------------------------------------------


def test_ground_truth_ratio_matches_the_injected_enhancements(
    noise_free_config: SyntheticConfig,
) -> None:
    """The single most important invariant in the package.

    With no noise, the injected enhancement of one species divided by that of
    the reference must equal the configured ratio exactly at every peak.
    """
    dataset = generate(noise_free_config)
    stream = dataset.streams["analyzer"]
    times = pd.DatetimeIndex(stream["time"].values)
    ch4 = stream[f"{TRUTH_PREFIX}enhancement_ch4"].values
    c2h6 = stream[f"{TRUTH_PREFIX}enhancement_c2h6"].values

    # Restrict to isolated, well-resolved samples so overlapping events do
    # not confound the pointwise comparison.
    strong = ch4 > 0.5 * ch4.max()
    assert strong.sum() > 10
    assert times.size == ch4.size
    assert np.allclose(c2h6[strong] / ch4[strong], 0.05, rtol=1e-9)


def test_sampled_peak_never_exceeds_the_true_amplitude(
    noise_free_config: SyntheticConfig,
) -> None:
    truth = generate(noise_free_config).ground_truth
    for event in truth.events:
        if math.isnan(event.sampled_peak_amplitude):
            continue
        assert event.sampled_peak_amplitude <= event.true_amplitude * (1.0 + 1e-9)


def test_ground_truth_windows_bracket_their_peak(
    noise_free_config: SyntheticConfig,
) -> None:
    for event in generate(noise_free_config).ground_truth.events:
        assert event.start_time <= event.peak_time <= event.end_time


def test_ground_truth_records_the_true_baseline(
    noise_free_config: SyntheticConfig,
) -> None:
    """The flat fixture background makes the expected value unambiguous."""
    for event in generate(noise_free_config).ground_truth.events:
        expected = 1900.0 if event.species == "ch4" else 2.0
        assert event.true_baseline_at_peak == pytest.approx(expected)


def test_reference_species_ratio_is_exactly_one(
    noise_free_config: SyntheticConfig,
) -> None:
    for event in generate(noise_free_config).ground_truth.events:
        if event.species == event.reference_species:
            assert event.true_ratio_to_reference == 1.0


def test_every_event_appears_once_per_participating_species(
    noise_free_config: SyntheticConfig,
) -> None:
    truth = generate(noise_free_config).ground_truth
    frame = truth.to_frame()
    counts = frame.groupby("event_id")["species"].nunique()
    assert set(counts.unique()) == {2}


def test_stationary_events_carry_the_site_coordinates(
    noise_free_config: SyntheticConfig,
) -> None:
    for event in generate(noise_free_config).ground_truth.events:
        assert event.latitude == pytest.approx(40.0)
        assert event.longitude == pytest.approx(-111.0)


# ---------------------------------------------------------------------------
# Noise pathways
# ---------------------------------------------------------------------------


def test_noisy_stream_emits_truth_and_reported_sigmas(
    noisy_config: SyntheticConfig,
) -> None:
    stream = generate(noisy_config).streams["analyzer"]
    assert f"{TRUTH_PREFIX}sigma_rand_ch4" in stream
    assert f"{TRUTH_PREFIX}sigma_sys_ch4" in stream
    # The reported column keeps its exact configured name, unprefixed.
    assert "ch4_err" in stream
    assert "ch4_err" in generate(noisy_config).observable("analyzer")


def test_systematic_draws_are_recorded_in_attrs(noisy_config: SyntheticConfig) -> None:
    attrs = generate(noisy_config).streams["analyzer"]["ch4"].attrs
    assert "true_sys_abs_draw" in attrs
    assert "true_sys_rel_draw" in attrs


def test_noise_free_species_emit_no_sigma_variables(
    noise_free_config: SyntheticConfig,
) -> None:
    stream = generate(noise_free_config).streams["analyzer"]
    assert f"{TRUTH_PREFIX}sigma_rand_ch4" not in stream
    assert f"{TRUTH_PREFIX}sigma_sys_ch4" not in stream


def test_injected_noise_matches_the_declared_budget(
    noisy_config: SyntheticConfig,
) -> None:
    """Observable minus truth must have the declared random spread."""
    stream = generate(noisy_config).streams["analyzer"]
    residual = (
        stream["ch4"].values
        - stream[f"{TRUTH_PREFIX}background_ch4"].values
        - stream[f"{TRUTH_PREFIX}enhancement_ch4"].values
    )
    # The systematic component shifts the mean but not the scatter.
    assert np.std(residual) == pytest.approx(2.0, rel=0.1)


# ---------------------------------------------------------------------------
# Quantization and circular variables
# ---------------------------------------------------------------------------


def _single_field_config(
    background: dict[str, Any] | None = None, **measurement: Any
) -> SyntheticConfig:
    """One methane field measured by one 1 s instrument; ``measurement`` sets its error."""
    return SyntheticConfig(
        name="single",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="30min",
        seed=5,
        platform=SITE,  # type: ignore[arg-type]
        atmosphere=AtmosphereSpec.model_validate(
            {
                "fields": {
                    "ch4": {
                        "units": "ppb",
                        "background": background or {"kind": "parametric", "offset": 1900.0},
                    }
                }
            }
        ),
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s", measures={"ch4": MeasurementSpec(**measurement)}
            )
        },
    )


def test_quantized_values_land_on_the_reporting_grid() -> None:
    config = _single_field_config(
        quantization=0.01, uncertainty=TrueUncertainty(random=TrueComponent(absolute=1.0))
    )
    stream = generate(config).streams["inst"]
    scaled = stream["ch4"].values / 0.01
    assert np.allclose(scaled, np.round(scaled), atol=1e-6)
    assert stream["ch4"].attrs["quantization"] == pytest.approx(0.01)


def test_circular_variable_wraps_into_zero_to_360() -> None:
    """Noise near the 0/360 discontinuity must be able to cross it."""
    config = SyntheticConfig(
        name="wind",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="6h",
        seed=3,
        platform=SITE,  # type: ignore[arg-type]
        atmosphere=AtmosphereSpec(
            fields={
                "wind_dir": FieldSpec(
                    background=ParametricBackground(
                        kind="parametric", offset=10.0, random_walk_std=2000.0
                    ),
                    role="met",
                    circular=True,
                    units="degrees",
                )
            }
        ),
        instruments={
            "met": InstrumentSpec(native_rate="10s", measures={"wind_dir": MeasurementSpec()})
        },
    )
    values = generate(config).streams["met"]["wind_dir"].values
    assert values.min() >= 0.0
    assert values.max() < 360.0
    # The walk is large enough that wrapping actually occurred.
    assert values.max() - values.min() > 180.0


def test_the_wrap_closes_the_interval_it_documents() -> None:
    """A direction a hair west of north must land on 0, not on a full turn.

    The generator wraps through `core.circular.wrap_degrees` rather than
    `np.mod`, which does not close the half-open interval it appears to:
    a tiny negative value modulo 360 rounds *up* to exactly 360.0 in float64.
    A reading that left [0, 360) would be a direction no consumer expects,
    reported with complete confidence.

    Pinned at the helper rather than through a generated campaign, and
    deliberately so: the window where the two spellings disagree is about
    3e-14 degrees wide, so reaching it from a background plus noise is a
    coincidence no seed can be relied on to produce. What the campaign above
    checks is the invariant; what this checks is that the invariant needs the
    helper to hold.
    """
    assert np.mod(-1e-17, 360.0) == 360.0
    assert float(wrap_degrees(-1e-17)) == 0.0


def test_met_species_receive_no_plumes(noise_free_config: SyntheticConfig) -> None:
    config = _add_field(
        noise_free_config,
        "analyzer",
        "temperature",
        {"role": "aux", "units": "K", "background": {"kind": "parametric", "offset": 290.0}},
    )
    dataset = generate(config)
    enhancement = dataset.streams["analyzer"][f"{TRUTH_PREFIX}enhancement_temperature"]
    assert np.all(enhancement.values == 0.0)
    assert not dataset.ground_truth.for_species("temperature")


# ---------------------------------------------------------------------------
# Clocks: jitter and dropouts
# ---------------------------------------------------------------------------


def test_jitter_produces_an_irregular_but_increasing_clock() -> None:
    config = _one_instrument(name="jittered", duration="20min", seed=1, timestamp_jitter="0.3s")
    times = pd.DatetimeIndex(generate(config).streams["inst"]["time"].values)
    assert times.is_monotonic_increasing
    deltas = np.diff(times.to_numpy().astype("datetime64[ns]").astype(np.int64))
    assert len(np.unique(deltas)) > 100


def test_dropouts_delete_samples_and_open_gaps() -> None:
    config = _one_instrument(
        name="gappy",
        duration="6h",
        seed=2,
        dropouts={"rate_per_day": 200.0, "duration": "120s"},
    )
    times = pd.DatetimeIndex(generate(config).streams["inst"]["time"].values)
    assert len(times) < 6 * 3600
    deltas = np.diff(times.to_numpy().astype("datetime64[ns]").astype(np.int64))
    assert deltas.max() > 10 * 10**9


def test_zero_drawn_dropouts_leaves_the_clock_intact() -> None:
    config = _one_instrument(
        name="lucky",
        duration="10min",
        dropouts={"rate_per_day": 1e-9, "duration": "60s"},
    )
    assert len(generate(config).streams["inst"]["time"]) == 600


def test_a_rate_coarser_than_the_record_still_yields_one_sample() -> None:
    config = _one_instrument(name="too_slow", duration="1s", native_rate="1h")
    assert len(generate(config).streams["inst"]["time"]) == 1


def test_dropouts_removing_every_sample_is_reported() -> None:
    """Absurd outage settings must fail loudly, not return an empty stream."""
    config = _one_instrument(
        name="wiped",
        duration="1min",
        dropouts={"rate_per_day": 40_000.0, "duration": "1h"},
    )
    with pytest.raises(TsaraSyntheticError, match="removed every sample"):
        generate(config)


def test_an_outage_can_predate_the_record_start() -> None:
    """An instrument may already be down when logging begins.

    Restricting onsets to the record itself would leave the first samples
    artificially immune to dropouts, which is an artifact rather than a
    property of real loggers.
    """
    config = _one_instrument(
        name="early_gap",
        duration="30min",
        seed=1,
        dropouts={"rate_per_day": 400.0, "duration": "300s"},
    )
    times = pd.DatetimeIndex(generate(config).streams["inst"]["time"].values)
    # The stream begins later than the configured start, i.e. an outage was
    # already in progress. (xarray stores datetimes tz-naive; TSARA is UTC
    # internally, so a naive comparison is the correct one here.)
    assert times[0] > pd.Timestamp("2026-01-01")


# ---------------------------------------------------------------------------
# Multi-rate and mobile
# ---------------------------------------------------------------------------


def test_instruments_keep_their_own_native_clocks(
    noise_free_config: SyntheticConfig,
) -> None:
    """The multi-rate case the whole 'synchronize late' design exists for."""
    config = _add_field(
        noise_free_config,
        "fast",
        "co2",
        {"units": "ppm", "background": {"kind": "parametric", "offset": 420.0}},
        native_rate="0.1s",
    )
    dataset = generate(config)
    assert len(dataset.streams["fast"]["time"]) == 10 * len(dataset.streams["analyzer"]["time"])


def test_mobile_platform_emits_a_separate_gps_stream() -> None:
    config = SyntheticConfig.model_validate(
        {
            "name": "mobile",
            "start": "2026-01-01T00:00:00Z",
            "duration": "1h",
            "seed": 8,
            "platform": {
                "kind": "mobile",
                "start_latitude": 40.0,
                "start_longitude": -111.0,
                "gps_rate": "1s",
                "pattern": "circuit",
                "radius_m": 400.0,
            },
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1900.0}}
                },
                "sources": {
                    "pad": {
                        "rate_per_hour": 8.0,
                        "shape": {"kind": "gaussian", "sigma": "20s"},
                        "reference_species": "ch4",
                        "amplitude": {"kind": "uniform", "low": 50.0, "high": 150.0},
                    }
                },
            },
            "instruments": {"analyzer": {"native_rate": "2s", "measures": {"ch4": {}}}},
        }
    )
    dataset = generate(config)
    assert set(dataset.streams) == {"analyzer", "gps"}
    # GPS runs at its own rate, twice the analyzer's.
    assert len(dataset.streams["gps"]["time"]) == 2 * len(dataset.streams["analyzer"]["time"])
    # And the gas stream carries time-varying coordinates.
    latitude = dataset.streams["analyzer"]["latitude"]
    assert latitude.dims == ("time",)
    assert float(latitude.std()) > 0.0
    # Events are geolocated at their peaks.
    assert all(event.latitude is not None for event in dataset.ground_truth.events)


def test_stationary_platform_uses_scalar_coordinates(
    noisy_config: SyntheticConfig,
) -> None:
    stream = generate(noisy_config).streams["analyzer"]
    assert stream["latitude"].dims == ()
    assert float(stream["latitude"]) == pytest.approx(40.0)
    assert float(stream["altitude"]) == pytest.approx(1500.0)


def test_stationary_without_altitude_omits_the_coordinate(
    noise_free_config: SyntheticConfig,
) -> None:
    assert "altitude" not in generate(noise_free_config).streams["analyzer"].coords


# ---------------------------------------------------------------------------
# Nesting and plume density
# ---------------------------------------------------------------------------


def test_nested_children_appear_in_the_catalog_with_parent_links(
    noise_free_config: SyntheticConfig, with_sources: WithSources
) -> None:
    pad = noise_free_config.atmosphere.sources["pad"]
    config = with_sources(
        noise_free_config,
        {
            "pad": pad.model_copy(
                update={
                    "nested": NestedSpec(
                        probability=1.0,
                        shape=GaussianShape(kind="gaussian", sigma="3s"),
                        amplitude_factor=0.6,
                    )
                }
            )
        },
    )
    truth = generate(config).ground_truth
    children = [event for event in truth.events if event.parent_event_id is not None]
    parents = {event.event_id for event in truth.events if event.parent_event_id is None}
    assert children
    assert all(child.parent_event_id in parents for child in children)


def test_nested_child_can_carry_a_species_its_parent_never_emits(
    noise_free_config: SyntheticConfig, with_sources: WithSources
) -> None:
    """The landfill-plus-blip case, end to end.

    Parent emits methane only; the nested child is thermogenic and carries
    ethane. Ethane enhancement must therefore appear *only* under children,
    and every ethane truth row must be a child row. This is the generator-side
    proof that relaxing the schema restriction actually renders — the config
    layer permitting it would be worthless if the injection path did not.
    """
    config = with_sources(
        noise_free_config,
        {
            "landfill": noise_free_config.atmosphere.sources["pad"].model_copy(
                update={
                    "ratios": {},  # parent: methane only, no ethane
                    "nested": NestedSpec(
                        probability=1.0,
                        shape=GaussianShape(kind="gaussian", sigma="3s"),
                        amplitude_factor=0.5,
                        ratios={"c2h6": RatioSpec(mean=0.06)},
                    ),
                }
            )
        },
    )
    dataset = generate(config)

    ethane_rows = [event for event in dataset.ground_truth.events if event.species == "c2h6"]
    assert ethane_rows, "the nested child should have produced ethane truth rows"
    assert all(row.parent_event_id is not None for row in ethane_rows)
    assert all(row.true_ratio_to_reference == pytest.approx(0.06) for row in ethane_rows)

    # Parent methane rows still exist, and outnumber the children's ethane.
    methane_parents = [
        event
        for event in dataset.ground_truth.events
        if event.species == "ch4" and event.parent_event_id is None
    ]
    assert len(methane_parents) == len(ethane_rows)

    # And the ethane stream really does carry the injected enhancement.
    enhancement = dataset.streams["analyzer"][f"{TRUTH_PREFIX}enhancement_c2h6"].values
    assert enhancement.max() > 0.0


def test_plume_dense_configuration_produces_overlapping_events(
    noise_free_config: SyntheticConfig, with_sources: WithSources
) -> None:
    """The required adversarial case: enhancements occupy most of the record."""
    dense = with_sources(
        noise_free_config,
        {
            "pad": noise_free_config.atmosphere.sources["pad"].model_copy(
                update={"rate_per_hour": 400.0}
            )
        },
    )
    dataset = generate(dense)
    enhancement = dataset.streams["analyzer"][f"{TRUTH_PREFIX}enhancement_ch4"].values
    assert (enhancement > 0.0).mean() > 0.95

    # Overlap: at least one pair of ch4 events shares time.
    frame = dataset.ground_truth.to_frame()
    ch4 = frame[frame["species"] == "ch4"].sort_values("start_time")
    starts = ch4["start_time"].to_numpy()
    ends = ch4["end_time"].to_numpy()
    overlaps = int((starts[1:] < ends[:-1]).sum())
    assert overlaps > 0


def test_bootstrap_background_flows_through_the_generator(
    white_noise_profile: RealDataProfile,
) -> None:
    config = _single_field_config(background={"kind": "bootstrap", "profile": "white"})
    stream = generate(config, profiles={"white": white_noise_profile}).streams["inst"]
    assert float(stream["ch4"].std()) == pytest.approx(3.0, rel=0.2)


def test_missing_profile_is_reported_at_generate_time() -> None:
    config = _single_field_config(background={"kind": "bootstrap", "profile": "absent"})
    with pytest.raises(TsaraSyntheticError, match="not supplied"):
        generate(config)


def test_source_free_config_yields_an_empty_catalog(
    noise_free_config: SyntheticConfig, with_sources: WithSources
) -> None:
    """The control case for measuring an algorithm's false-positive rate."""
    control = with_sources(noise_free_config, {})
    dataset = generate(control)
    assert len(dataset.ground_truth) == 0
    enhancement = dataset.streams["analyzer"][f"{TRUTH_PREFIX}enhancement_ch4"].values
    assert np.all(enhancement == 0.0)


# ---------------------------------------------------------------------------
# Timezone normalization
# ---------------------------------------------------------------------------


def test_aware_and_naive_starts_produce_identical_streams() -> None:
    """TSARA is UTC internally, so both spellings must mean the same instant.

    A tz-aware axis would also fail to encode to netCDF at save time, so this
    normalization is load-bearing for persistence as well as correctness.
    """
    aware = _single_field_config(
        background={"kind": "parametric", "offset": 1900.0, "diurnal_amplitude": 15.0},
        uncertainty=TrueUncertainty(random=TrueComponent(absolute=1.0)),
    ).model_copy(update={"name": "tz", "duration": "10min", "seed": 11})
    naive = aware.model_copy(update={"start": datetime(2026, 1, 1)})

    aware_stream = generate(aware).streams["inst"]
    naive_stream = generate(naive).streams["inst"]
    assert np.array_equal(aware_stream["time"].values, naive_stream["time"].values)
    assert np.array_equal(aware_stream["ch4"].values, naive_stream["ch4"].values)


def test_stream_time_axis_is_timezone_naive(noisy_config: SyntheticConfig) -> None:
    times = pd.DatetimeIndex(generate(noisy_config).streams["analyzer"]["time"].values)
    assert times.tz is None


def test_ground_truth_windows_can_slice_their_own_stream(noisy_config: SyntheticConfig) -> None:
    """The harness's central operation, and the one the tz rule exists for.

    Scoring any later phase means taking a ground-truth event window and
    pulling the stream samples inside it. If the catalog's timestamps kept the
    config's timezone while the clocks were normalized to naive UTC, pandas
    would raise `TypeError: Cannot compare tz-naive and tz-aware ...` here —
    and `noisy_config`, like the shipped example, declares a tz-aware start.
    """
    dataset = generate(noisy_config)
    stream = dataset.streams["analyzer"]
    event = dataset.ground_truth.events[0]

    assert event.peak_time.tz is None
    window = stream.sel(time=slice(event.start_time, event.end_time))
    assert window.sizes["time"] > 0
    # And the window really does bracket the peak on the stream's own clock.
    assert window["time"].values[0] <= np.datetime64(event.peak_time)
    assert np.datetime64(event.peak_time) <= window["time"].values[-1]


def test_ground_truth_is_identical_across_timezone_spellings() -> None:
    """The catalog, not only the streams, must be spelling-independent."""
    aware = _one_instrument(
        name="tz_truth",
        duration="30min",
        seed=5,
        sources={
            "pad": {
                "rate_per_hour": 30.0,
                "shape": {"kind": "gaussian", "sigma": "10s"},
                "reference_species": "ch4",
                "amplitude": {"kind": "uniform", "low": 50.0, "high": 150.0},
            }
        },
    )
    naive = aware.model_copy(update={"start": datetime(2026, 1, 1)})

    aware_events = generate(aware).ground_truth.events
    naive_events = generate(naive).ground_truth.events
    assert len(aware_events) > 0
    for from_aware, from_naive in zip(aware_events, naive_events):
        assert from_aware.peak_time == from_naive.peak_time
        assert from_aware.peak_time.tz is None
        assert from_naive.peak_time.tz is None


def test_every_stream_uses_nanosecond_time_resolution() -> None:
    """One dataset, one time representation — regardless of jitter.

    The jitter branch casts to `datetime64[ns]`, while an unjittered clock
    would otherwise inherit its unit from the config's start (microseconds,
    for a `datetime.datetime`). Mixed resolutions in one dataset would also
    make save/load change dtypes, since netCDF stores nanoseconds.
    """
    config = SyntheticConfig.model_validate(
        {
            "name": "units",
            "start": "2026-01-01T00:00:00Z",
            "duration": "5min",
            "seed": 3,
            "platform": SITE,
            "atmosphere": {
                "fields": {
                    "ch4": {"background": {"kind": "parametric", "offset": 1900.0}},
                    "co2": {"background": {"kind": "parametric", "offset": 410.0}},
                }
            },
            "instruments": {
                "plain": {"native_rate": "1s", "measures": {"ch4": {}}},
                "jittered": {
                    "native_rate": "1s",
                    "timestamp_jitter": "100ms",
                    "measures": {"co2": {}},
                },
            },
        }
    )
    streams = generate(config).streams
    assert {str(stream["time"].dtype) for stream in streams.values()} == {"datetime64[ns]"}


def test_an_event_inside_a_data_gap_has_no_sampled_peak() -> None:
    """A plume the instrument was down for must record NaN, not a fake peak.

    The ground truth still lists the event (it physically happened); only the
    *sampled* amplitude is unknown, which is exactly the distinction between
    `true_amplitude` and `sampled_peak_amplitude`.
    """
    config = _one_instrument(
        name="gap_event",
        duration="4h",
        seed=1,
        dropouts={"rate_per_day": 600.0, "duration": "600s"},
        sources={
            "pad": {
                "rate_per_hour": 60.0,
                "shape": {"kind": "gaussian", "sigma": "5s"},
                "reference_species": "ch4",
                "amplitude": {"kind": "uniform", "low": 50.0, "high": 150.0},
            }
        },
    )
    truth = generate(config).ground_truth
    missed = [e for e in truth.events if math.isnan(e.sampled_peak_amplitude)]
    assert missed
    # The event is still catalogued with its true (physical) amplitude.
    assert all(event.true_amplitude > 0.0 for event in missed)


# ---------------------------------------------------------------------------
# Temporal support (Phase 3.5)
# ---------------------------------------------------------------------------


def _flat_config(**support: Any) -> SyntheticConfig:
    """A single 60 s instrument on a flat background, with no noise."""
    return SyntheticConfig.model_validate(
        {
            "name": "cells",
            "start": "2026-01-01T00:00:00Z",
            "duration": "1h",
            "seed": 7,
            "platform": SITE,
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1900.0}}
                }
            },
            "instruments": {
                "slow": {"native_rate": "60s", "support": support, "measures": {"ch4": {}}}
            },
        }
    )


def test_every_stream_carries_cells_including_gps(synthetic_dict: dict[str, Any]) -> None:
    dataset = generate(SyntheticConfig.model_validate(synthetic_dict))
    for name, stream in dataset.streams.items():
        assert TIME_BOUNDS_VAR in stream.coords, name
        assert stream[TIME_COORD].attrs[BOUNDS_ATTR] == TIME_BOUNDS_VAR, name
        assert stream.sizes[BOUNDS_DIM] == 2, name


def test_the_default_cell_is_centred_so_no_timestamp_moves() -> None:
    """The compatibility guarantee: existing configs emit the same clock."""
    dataset = generate(_flat_config())
    stream = dataset.streams["slow"]
    bounds = stream[TIME_BOUNDS_VAR].values.astype("datetime64[ns]").astype(np.int64)
    stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    assert np.array_equal(bounds[:, 0] + 30_000_000_000, stamps)
    assert np.array_equal(bounds[:, 1] - 30_000_000_000, stamps)


def test_a_start_label_moves_time_to_the_cell_midpoint() -> None:
    centred = generate(_flat_config()).streams["slow"]
    labelled = generate(_flat_config(label="start")).streams["slow"]
    shift = labelled[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64) - centred[
        TIME_COORD
    ].values.astype("datetime64[ns]").astype(np.int64)
    assert np.all(shift == 30_000_000_000)
    assert labelled.attrs[SUPPORT_LABEL_ATTR] == "start"


def test_point_and_mean_are_recorded_as_cf_cell_methods() -> None:
    point = generate(_flat_config()).streams["slow"]
    averaged = generate(_flat_config(method="mean")).streams["slow"]
    assert point["ch4"].attrs[CELL_METHODS_ATTR] == "time: point"
    assert averaged["ch4"].attrs[CELL_METHODS_ATTR] == "time: mean"


def test_a_duty_cycled_instrument_has_narrow_cells_and_low_coverage() -> None:
    """A sampler that integrates for 15 s a minute, the canister shape."""
    stream = generate(_flat_config(method="mean", width="15s")).streams["slow"]
    bounds = stream[TIME_BOUNDS_VAR].values.astype("datetime64[ns]").astype(np.int64)
    assert np.all(bounds[:, 1] - bounds[:, 0] == 15_000_000_000)
    assert stream.attrs[SUPPORT_COVERAGE_ATTR] == pytest.approx(0.25, abs=0.01)


def test_a_cell_wider_than_the_clock_is_refused() -> None:
    with pytest.raises(ValidationError, match="would make cells overlap"):
        _flat_config(method="mean", width="120s")


def test_the_generator_declares_all_three_support_facts() -> None:
    stream = generate(_flat_config(method="mean")).streams["slow"]
    assert stream.attrs[SUPPORT_LABEL_PROVENANCE_ATTR] == "declared"
    assert stream.attrs[SUPPORT_WIDTH_PROVENANCE_ATTR] == "declared"
    assert stream.attrs[SUPPORT_METHOD_PROVENANCE_ATTR] == "declared"


# --- does `mean` actually average? -----------------------------------------


def test_averaging_a_flat_background_is_exact() -> None:
    """No approximation to make: every subsample is the same number."""
    stream = generate(_flat_config(method="mean")).streams["slow"]
    assert np.allclose(stream["ch4"].values, 1900.0, rtol=0, atol=1e-12)


def test_averaging_a_linear_drift_reproduces_the_closed_form() -> None:
    """The midpoint rule is exact for a linear function, so this pins the cell
    geometry and the drift's origin rather than the quadrature.

    Compared as absolute values, which the previous generator could not be:
    it measured drift from the FIRST RENDERED SAMPLE of each instrument, and a
    `mean` instrument's first sample sits half a sub-step before its first cell
    midpoint, so every value carried a small constant offset (about 1.3e-4 ppb
    here) and only increments could be checked. Drift is now measured from the
    campaign start, once, for the whole atmosphere, so the 60 s cell starting
    at minute k reads exactly 1000 + (k + 0.5)/60 ppb.
    """
    config = SyntheticConfig.model_validate(
        {
            "name": "drift",
            "start": "2026-01-01T00:00:00Z",
            "duration": "1h",
            "seed": 3,
            "platform": SITE,
            "atmosphere": {
                "fields": {
                    "ch4": {
                        "units": "ppb",
                        "background": {
                            "kind": "parametric",
                            "offset": 1000.0,
                            "drift_per_day": 24.0,
                        },
                    }
                }
            },
            "instruments": {
                "slow": {
                    "native_rate": "60s",
                    "support": {"method": "mean", "label": "start"},
                    "measures": {"ch4": {}},
                }
            },
        }
    )
    values = generate(config).streams["slow"]["ch4"].values
    # 24 ppb/day is 1 ppb/hour; the k-th cell's midpoint is (k + 0.5) minutes in.
    expected = 1000.0 + (np.arange(60) + 0.5) / 60.0
    assert np.allclose(values, expected, rtol=0, atol=1e-9)


def test_the_cell_average_converges_as_the_fine_grid_refines() -> None:
    """Validates the quadrature itself, without needing a closed form.

    The midpoint rule's error falls as the inverse square of the subsample
    count, so quadrupling the count should cut the error by roughly sixteen.
    Asserted loosely (a factor of eight) because the events are drawn at
    random phases and a cell whose plume sits near its edge converges more
    slowly than one whose plume sits in the middle.
    """

    def values(subsamples: int) -> np.ndarray:
        config = SyntheticConfig.model_validate(
            {
                "name": "converge",
                "start": "2026-01-01T00:00:00Z",
                "duration": "2h",
                "seed": 11,
                "platform": SITE,
                "atmosphere": {
                    "fields": {
                        "ch4": {
                            "units": "ppb",
                            "background": {"kind": "parametric", "offset": 1900.0},
                        }
                    },
                    "sources": {
                        "leak": {
                            "rate_per_hour": 20.0,
                            "reference_species": "ch4",
                            "shape": {"kind": "gaussian", "sigma": "20s"},
                            "amplitude": {"kind": "uniform", "low": 100.0, "high": 100.001},
                            "ratios": {},
                        }
                    },
                },
                "instruments": {
                    "slow": {
                        "native_rate": "60s",
                        "support": {"method": "mean", "subsamples": subsamples},
                        "measures": {"ch4": {}},
                    }
                },
            }
        )
        return np.asarray(generate(config).streams["slow"]["ch4"].values)

    reference = values(2048)
    coarse = np.max(np.abs(values(16) - reference))
    finer = np.max(np.abs(values(64) - reference))
    assert coarse > 0, "a coarse grid must actually differ, or this proves nothing"
    assert finer < coarse / 8.0


def test_a_narrow_plume_inside_a_wide_cell_comes_out_diluted() -> None:
    """The scientific point of `mean`, checked against the closed form.

    A Gaussian of amplitude A and width sigma sitting wholly inside a cell of
    width W averages to A * sigma * sqrt(2*pi) / W. With sigma 3 s in a 60 s
    cell that is 12.53% of the true amplitude, and no event can exceed it: one
    straddling a cell boundary is split and reads lower still. The answer key
    records that diluted peak, which is what a later stage needs in order to
    say an event was never resolvable on this instrument.
    """
    config = SyntheticConfig.model_validate(
        {
            "name": "dilute",
            "start": "2026-01-01T00:00:00Z",
            "duration": "6h",
            "seed": 5,
            "platform": SITE,
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1900.0}}
                },
                "sources": {
                    "blip": {
                        "rate_per_hour": 10.0,
                        "reference_species": "ch4",
                        "shape": {"kind": "gaussian", "sigma": "3s"},
                        "amplitude": {"kind": "uniform", "low": 100.0, "high": 100.001},
                        "ratios": {},
                    }
                },
            },
            "instruments": {
                "slow": {
                    "native_rate": "60s",
                    "support": {"method": "mean"},
                    "measures": {"ch4": {}},
                }
            },
        }
    )
    rows = generate(config).ground_truth.for_species("ch4")
    ratios = np.array([r.sampled_peak_amplitude / r.true_amplitude for r in rows])
    # An event whose whole support window falls past the end of the record has
    # no sampled peak at all, which the generator records as nan rather than
    # as zero. Pre-existing edge behaviour, unrelated to cells.
    ratios = ratios[np.isfinite(ratios)]
    ceiling = 3.0 * math.sqrt(2.0 * math.pi) / 60.0
    assert ratios.size > 10
    assert np.all(ratios > 0.0)
    assert np.all(ratios <= ceiling + 1e-12)
    # Some event lands near enough to a cell centre to reach the closed form.
    assert ratios.max() == pytest.approx(ceiling, rel=1e-3)


def test_the_observable_view_still_carries_its_cells(
    synthetic_dict: dict[str, Any],
) -> None:
    """It is meant to be what ingestion would have produced, and a real stream
    has cells. Selecting variables by name would have dropped them."""
    dataset = generate(SyntheticConfig.model_validate(synthetic_dict))
    for name in dataset.streams:
        observable = dataset.observable(name)
        assert TIME_BOUNDS_VAR in observable.coords, name
        assert not [v for v in observable.data_vars if str(v).startswith(TRUTH_PREFIX)], name


def test_a_jittered_mean_instrument_gets_an_exact_answer_key() -> None:
    """Regression: plume injection used to locate events by binary search over
    the flattened fine grid, and that grid is not sorted.

    Jitter is permitted up to just under half the sampling interval, so
    full-width cells centred on jittered stamps overlap and the flat grid
    descends. Measured at the time, 151 of 300 adjacent cells overlapped and
    the search mis-selected cells by up to 0.8% of a plume's peak. Events are
    now located against the cell boundaries, which are sorted whatever the
    jitter. Checked against a reference that evaluates every event on every
    cell with no search at all.
    """
    config = SyntheticConfig.model_validate(
        {
            "name": "jittered",
            "start": "2026-01-01T00:00:00Z",
            "duration": "1h",
            "seed": 7,
            "platform": SITE,
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1900.0}}
                },
                "sources": {
                    "leak": {
                        "rate_per_hour": 60.0,
                        "reference_species": "ch4",
                        "shape": {"kind": "gaussian", "sigma": "3s"},
                        "amplitude": {"kind": "uniform", "low": 100.0, "high": 100.001},
                        "ratios": {},
                    }
                },
            },
            "instruments": {
                "a": {
                    "native_rate": "10s",
                    # Close to the schema's ceiling of half the sampling rate,
                    # which is where the cells overlap most.
                    "timestamp_jitter": "4s",
                    "support": {"method": "mean"},
                    "measures": {"ch4": {}},
                }
            },
        }
    )
    produced = np.asarray(generate(config).streams["a"][f"{TRUTH_PREFIX}enhancement_ch4"].values)

    # Rebuild the same atmosphere and clock, in the generator's draw order,
    # then brute-force the enhancement.
    rng = np.random.default_rng(config.seed)
    events = realize_atmosphere(config, rng).events
    start = pd.Timestamp(config.start).tz_localize(None)
    times = _build_times(start, start + pd.Timedelta(config.duration), "10s", "4s", None, rng, "a")
    grid = _build_cells(times, config.instruments["a"])
    bounds, fine = grid.cells, grid.fine_ns
    assert np.any(bounds.start_ns[1:] < bounds.stop_ns[:-1]), (
        "this configuration must actually produce overlapping cells, or the test proves nothing"
    )
    n_sub = fine.shape[1]
    fine_seconds = fine / 1e9
    expected = np.zeros(len(bounds))
    for event in events:
        amplitude = event.amplitudes.get("ch4")
        if amplitude is None:
            continue
        centre = event.species_center("ch4").value / 1e9
        expected += amplitude * event.kernel.evaluate((fine_seconds - centre).reshape(-1)).reshape(
            -1, n_sub
        ).mean(axis=1)

    assert np.allclose(produced, expected, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# One atmosphere, several instruments (Phase 4.5)
# ---------------------------------------------------------------------------


def _two_analyzers(**fast_measurement: Any) -> SyntheticConfig:
    """A methane plume field measured by a 1 s analyzer and a 60 s mean analyzer."""
    return SyntheticConfig.model_validate(
        {
            "name": "two_analyzers",
            "start": "2026-01-01T00:00:00Z",
            "duration": "2h",
            "seed": 21,
            "platform": SITE,
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1900.0}}
                },
                "sources": {
                    "pad": {
                        "rate_per_hour": 12.0,
                        "reference_species": "ch4",
                        "shape": {"kind": "gaussian", "sigma": "30s"},
                        "amplitude": {"kind": "uniform", "low": 50.0, "high": 150.0},
                    }
                },
            },
            "instruments": {
                "fast": {"native_rate": "1s", "measures": {"ch4": fast_measurement}},
                "minute": {
                    "native_rate": "60s",
                    "support": {"method": "mean"},
                    "measures": {"ch4": {"name": "ch4_minute"}},
                },
            },
        }
    )


def test_a_renamed_measurement_is_written_under_its_name_and_keeps_its_field() -> None:
    stream = generate(_two_analyzers()).streams["minute"]
    assert "ch4_minute" in stream.data_vars
    assert "ch4" not in stream.data_vars
    assert stream["ch4_minute"].attrs["field"] == "ch4"
    # The answer key follows the variable's name, as every stream column does.
    assert f"{TRUTH_PREFIX}enhancement_ch4_minute" in stream.data_vars


def test_each_instrument_measuring_a_species_gets_its_own_truth_rows() -> None:
    """One event, seen twice: once per variable, each through its own cells."""
    frame = generate(_two_analyzers()).ground_truth.to_frame()
    per_instrument = frame.groupby("instrument")["event_id"].apply(list).to_dict()
    assert per_instrument["fast"] == per_instrument["minute"]
    assert set(frame["field"]) == {"ch4"}
    assert set(frame.loc[frame["instrument"] == "minute", "species"]) == {"ch4_minute"}
    assert set(frame.loc[frame["instrument"] == "fast", "species"]) == {"ch4"}
    # Same event, same true amplitude; the wide cells record it diluted.
    fast = frame[frame["instrument"] == "fast"].set_index("event_id")
    minute = frame[frame["instrument"] == "minute"].set_index("event_id")
    assert np.array_equal(fast["true_amplitude"], minute["true_amplitude"])
    seen = np.isfinite(minute["sampled_peak_amplitude"]) & np.isfinite(
        fast["sampled_peak_amplitude"]
    )
    assert np.all(minute["sampled_peak_amplitude"][seen] < fast["sampled_peak_amplitude"][seen])


def test_a_measurement_error_belongs_to_its_instrument() -> None:
    """Noise on one analyzer must not reach the other's record of the same air."""
    noisy = generate(_two_analyzers(uncertainty={"random": {"absolute": 5.0}}))
    quiet = generate(_two_analyzers())
    assert not np.array_equal(
        noisy.streams["fast"]["ch4"].values, quiet.streams["fast"]["ch4"].values
    )
    assert np.array_equal(
        noisy.streams["minute"]["ch4_minute"].values, quiet.streams["minute"]["ch4_minute"].values
    )


def test_the_dataset_carries_the_atmosphere_it_sampled() -> None:
    dataset = generate(_two_analyzers())
    assert dataset.atmosphere is not None
    assert set(dataset.atmosphere.fields) == {"ch4"}
    assert len(dataset.atmosphere.events) == len(dataset.ground_truth.event_ids)
