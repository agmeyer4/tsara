"""End-to-end tests for the synthetic dataset generator.

The assertions here are the ones that matter most for the whole project: if
injected ground truth is not internally consistent, every later phase is
being scored against a broken answer key.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from tsara.synthetic.background import TsaraSyntheticError
from tsara.synthetic.config import (
    BootstrapBackground,
    DropoutSpec,
    GaussianShape,
    InstrumentSpec,
    MobileTrack,
    NestedSpec,
    ParametricBackground,
    RatioSpec,
    SpeciesSpec,
    SyntheticConfig,
    TrueComponent,
    TrueUncertainty,
    UniformAmplitude,
)
from tsara.synthetic.generator import TRUTH_PREFIX, generate
from tsara.synthetic.profiling import RealDataProfile

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


def _single_species_config(**species_kwargs: object) -> SyntheticConfig:
    return SyntheticConfig(
        name="single",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="30min",
        seed=5,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0),
                        units="ppb",
                        **species_kwargs,  # type: ignore[arg-type]
                    )
                },
            )
        },
    )


def test_quantized_values_land_on_the_reporting_grid() -> None:
    config = _single_species_config(
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
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "met": InstrumentSpec(
                native_rate="10s",
                species={
                    "wind_dir": SpeciesSpec(
                        background=ParametricBackground(
                            kind="parametric", offset=10.0, random_walk_std=2000.0
                        ),
                        role="met",
                        circular=True,
                        units="degrees",
                    )
                },
            )
        },
    )
    values = generate(config).streams["met"]["wind_dir"].values
    assert values.min() >= 0.0
    assert values.max() < 360.0
    # The walk is large enough that wrapping actually occurred.
    assert values.max() - values.min() > 180.0


def test_met_species_receive_no_plumes(noise_free_config: SyntheticConfig) -> None:
    config = noise_free_config.model_copy(
        update={
            "instruments": {
                "analyzer": noise_free_config.instruments["analyzer"].model_copy(
                    update={
                        "species": {
                            **noise_free_config.instruments["analyzer"].species,
                            "temperature": SpeciesSpec(
                                background=ParametricBackground(kind="parametric", offset=290.0),
                                role="aux",
                                units="K",
                            ),
                        }
                    }
                )
            }
        }
    )
    dataset = generate(config)
    enhancement = dataset.streams["analyzer"][f"{TRUTH_PREFIX}enhancement_temperature"]
    assert np.all(enhancement.values == 0.0)
    assert not dataset.ground_truth.for_species("temperature")


# ---------------------------------------------------------------------------
# Clocks: jitter and dropouts
# ---------------------------------------------------------------------------


def test_jitter_produces_an_irregular_but_increasing_clock() -> None:
    config = SyntheticConfig(
        name="jittered",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="20min",
        seed=1,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                timestamp_jitter="0.3s",
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0)
                    )
                },
            )
        },
    )
    times = pd.DatetimeIndex(generate(config).streams["inst"]["time"].values)
    assert times.is_monotonic_increasing
    deltas = np.diff(times.to_numpy().astype("datetime64[ns]").astype(np.int64))
    assert len(np.unique(deltas)) > 100


def test_dropouts_delete_samples_and_open_gaps() -> None:
    config = SyntheticConfig(
        name="gappy",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="6h",
        seed=2,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                dropouts=DropoutSpec(rate_per_day=200.0, duration="120s"),
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0)
                    )
                },
            )
        },
    )
    times = pd.DatetimeIndex(generate(config).streams["inst"]["time"].values)
    assert len(times) < 6 * 3600
    deltas = np.diff(times.to_numpy().astype("datetime64[ns]").astype(np.int64))
    assert deltas.max() > 10 * 10**9


def test_zero_drawn_dropouts_leaves_the_clock_intact() -> None:
    config = SyntheticConfig(
        name="lucky",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="10min",
        seed=0,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                dropouts=DropoutSpec(rate_per_day=1e-9, duration="60s"),
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0)
                    )
                },
            )
        },
    )
    assert len(generate(config).streams["inst"]["time"]) == 600


def test_a_rate_coarser_than_the_record_still_yields_one_sample() -> None:
    config = SyntheticConfig(
        name="too_slow",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="1s",
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1h",
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0)
                    )
                },
            )
        },
    )
    assert len(generate(config).streams["inst"]["time"]) == 1


def test_dropouts_removing_every_sample_is_reported() -> None:
    """Absurd outage settings must fail loudly, not return an empty stream."""
    config = SyntheticConfig(
        name="wiped",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="1min",
        seed=0,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                dropouts=DropoutSpec(rate_per_day=40_000.0, duration="1h"),
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0)
                    )
                },
            )
        },
    )
    with pytest.raises(TsaraSyntheticError, match="removed every sample"):
        generate(config)


def test_an_outage_can_predate_the_record_start() -> None:
    """An instrument may already be down when logging begins.

    Restricting onsets to the record itself would leave the first samples
    artificially immune to dropouts, which is an artifact rather than a
    property of real loggers.
    """
    config = SyntheticConfig(
        name="early_gap",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="30min",
        seed=1,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                dropouts=DropoutSpec(rate_per_day=400.0, duration="300s"),
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0)
                    )
                },
            )
        },
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
    config = noise_free_config.model_copy(
        update={
            "instruments": {
                **noise_free_config.instruments,
                "fast": InstrumentSpec(
                    native_rate="0.1s",
                    species={
                        "co2": SpeciesSpec(
                            background=ParametricBackground(kind="parametric", offset=420.0),
                            units="ppm",
                        )
                    },
                ),
            }
        }
    )
    dataset = generate(config)
    assert len(dataset.streams["fast"]["time"]) == 10 * len(dataset.streams["analyzer"]["time"])


def test_mobile_platform_emits_a_separate_gps_stream() -> None:
    config = SyntheticConfig(
        name="mobile",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="1h",
        seed=8,
        platform=MobileTrack(
            kind="mobile",
            start_latitude=40.0,
            start_longitude=-111.0,
            gps_rate="1s",
            pattern="circuit",
            radius_m=400.0,
        ),
        instruments={
            "analyzer": InstrumentSpec(
                native_rate="2s",
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0),
                        units="ppb",
                    )
                },
            )
        },
        sources={
            "pad": {  # type: ignore[dict-item]
                "rate_per_hour": 8.0,
                "shape": GaussianShape(kind="gaussian", sigma="20s"),
                "reference_species": "ch4",
                "amplitude": UniformAmplitude(kind="uniform", low=50.0, high=150.0),
            }
        },
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
    noise_free_config: SyntheticConfig,
) -> None:
    config = noise_free_config.model_copy(
        update={
            "sources": {
                "pad": noise_free_config.sources["pad"].model_copy(
                    update={
                        "nested": NestedSpec(
                            probability=1.0,
                            shape=GaussianShape(kind="gaussian", sigma="3s"),
                            amplitude_factor=0.6,
                        )
                    }
                )
            }
        }
    )
    truth = generate(config).ground_truth
    children = [event for event in truth.events if event.parent_event_id is not None]
    parents = {event.event_id for event in truth.events if event.parent_event_id is None}
    assert children
    assert all(child.parent_event_id in parents for child in children)


def test_nested_child_can_carry_a_species_its_parent_never_emits(
    noise_free_config: SyntheticConfig,
) -> None:
    """The landfill-plus-blip case, end to end.

    Parent emits methane only; the nested child is thermogenic and carries
    ethane. Ethane enhancement must therefore appear *only* under children,
    and every ethane truth row must be a child row. This is the generator-side
    proof that relaxing the schema restriction actually renders — the config
    layer permitting it would be worthless if the injection path did not.
    """
    config = noise_free_config.model_copy(
        update={
            "sources": {
                "landfill": noise_free_config.sources["pad"].model_copy(
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
            }
        }
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
    noise_free_config: SyntheticConfig,
) -> None:
    """The required adversarial case: enhancements occupy most of the record."""
    dense = noise_free_config.model_copy(
        update={
            "sources": {
                "pad": noise_free_config.sources["pad"].model_copy(update={"rate_per_hour": 400.0})
            }
        }
    )
    dataset = generate(dense)
    enhancement = dataset.streams["analyzer"][f"{TRUTH_PREFIX}enhancement_ch4"].values
    assert (enhancement > 0.0).mean() > 0.95

    # Overlap: at least one pair of ch4 events shares time.
    frame = dataset.ground_truth.to_frame()
    ch4 = frame[frame["species"] == "ch4"].sort_values("start_time")
    overlaps = (ch4["start_time"].values[1:] < ch4["end_time"].values[:-1]).sum()
    assert overlaps > 0


def test_bootstrap_background_flows_through_the_generator(
    white_noise_profile: RealDataProfile,
) -> None:
    config = _single_species_config()
    config = config.model_copy(
        update={
            "instruments": {
                "inst": InstrumentSpec(
                    native_rate="1s",
                    species={
                        "ch4": SpeciesSpec(
                            background=BootstrapBackground(kind="bootstrap", profile="white"),
                            units="ppb",
                        )
                    },
                )
            }
        }
    )
    stream = generate(config, profiles={"white": white_noise_profile}).streams["inst"]
    assert float(stream["ch4"].std()) == pytest.approx(3.0, rel=0.2)


def test_missing_profile_is_reported_at_generate_time() -> None:
    config = _single_species_config()
    config = config.model_copy(
        update={
            "instruments": {
                "inst": InstrumentSpec(
                    native_rate="1s",
                    species={
                        "ch4": SpeciesSpec(
                            background=BootstrapBackground(kind="bootstrap", profile="absent")
                        )
                    },
                )
            }
        }
    )
    with pytest.raises(TsaraSyntheticError, match="not supplied"):
        generate(config)


def test_source_free_config_yields_an_empty_catalog(
    noise_free_config: SyntheticConfig,
) -> None:
    """The control case for measuring an algorithm's false-positive rate."""
    control = noise_free_config.model_copy(update={"sources": {}})
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
    aware = SyntheticConfig(
        name="tz",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="10min",
        seed=11,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(
                            kind="parametric", offset=1900.0, diurnal_amplitude=15.0
                        ),
                        uncertainty=TrueUncertainty(random=TrueComponent(absolute=1.0)),
                    )
                },
            )
        },
    )
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
    aware = SyntheticConfig(
        name="tz_truth",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="30min",
        seed=5,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0),
                        units="ppb",
                    )
                },
            )
        },
        sources={
            "pad": {  # type: ignore[dict-item]
                "rate_per_hour": 30.0,
                "shape": GaussianShape(kind="gaussian", sigma="10s"),
                "reference_species": "ch4",
                "amplitude": UniformAmplitude(kind="uniform", low=50.0, high=150.0),
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
    config = SyntheticConfig(
        name="units",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="5min",
        seed=3,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "plain": InstrumentSpec(
                native_rate="1s",
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0)
                    )
                },
            ),
            "jittered": InstrumentSpec(
                native_rate="1s",
                timestamp_jitter="100ms",
                species={
                    "co2": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=410.0)
                    )
                },
            ),
        },
    )
    streams = generate(config).streams
    assert {str(stream["time"].dtype) for stream in streams.values()} == {"datetime64[ns]"}


def test_an_event_inside_a_data_gap_has_no_sampled_peak() -> None:
    """A plume the instrument was down for must record NaN, not a fake peak.

    The ground truth still lists the event (it physically happened); only the
    *sampled* amplitude is unknown, which is exactly the distinction between
    `true_amplitude` and `sampled_peak_amplitude`.
    """
    config = SyntheticConfig(
        name="gap_event",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        duration="4h",
        seed=1,
        platform={"kind": "stationary", "latitude": 40.0, "longitude": -111.0},  # type: ignore[arg-type]
        instruments={
            "inst": InstrumentSpec(
                native_rate="1s",
                dropouts=DropoutSpec(rate_per_day=600.0, duration="600s"),
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0),
                        units="ppb",
                    )
                },
            )
        },
        sources={
            "pad": {  # type: ignore[dict-item]
                "rate_per_hour": 60.0,
                "shape": GaussianShape(kind="gaussian", sigma="5s"),
                "reference_species": "ch4",
                "amplitude": UniformAmplitude(kind="uniform", low=50.0, high=150.0),
            }
        },
    )
    truth = generate(config).ground_truth
    missed = [e for e in truth.events if math.isnan(e.sampled_peak_amplitude)]
    assert missed
    # The event is still catalogued with its true (physical) amplitude.
    assert all(event.true_amplitude > 0.0 for event in missed)
