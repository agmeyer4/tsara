"""Tests for reading and writing the on-disk TSARA bundle."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tsara.core.bundle import BUNDLE_STAGE_KEY
from tsara.core.naming import (
    BOUNDS_ATTR,
    SUPPORT_LABEL_SOURCE_ATTR,
    SUPPORT_METHOD_SOURCE_ATTR,
    SUPPORT_WIDTH_SOURCE_ATTR,
    TIME_BOUNDS_VAR,
    TIME_COORD,
)
from tsara.core.support import TsaraSupportError
from tsara.synthetic.bundle import (
    BUNDLE_CONFIG,
    BUNDLE_GROUND_TRUTH,
    BUNDLE_MANIFEST,
    BUNDLE_STREAMS_DIR,
    TsaraBundleError,
    load_bundle,
    save_bundle,
)
from tsara.synthetic.config import SyntheticConfig
from tsara.synthetic.generator import SyntheticDataset, generate
from tsara.synthetic.profiling import RealDataProfile

WithSources = Callable[[SyntheticConfig, dict[str, Any]], SyntheticConfig]

# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_bundle_layout_matches_the_convention(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    bundle = generate(noisy_config).save(tmp_path / "run")
    assert (bundle / BUNDLE_MANIFEST).is_file()
    assert (bundle / BUNDLE_CONFIG).is_file()
    assert (bundle / BUNDLE_GROUND_TRUTH).is_file()
    assert (bundle / BUNDLE_STREAMS_DIR / "analyzer.nc").is_file()


def test_round_trip_preserves_the_observable_data(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    original = generate(noisy_config)
    restored = SyntheticDataset.load(original.save(tmp_path / "run"))
    assert np.array_equal(
        original.streams["analyzer"]["ch4"].values,
        restored.streams["analyzer"]["ch4"].values,
    )


def test_round_trip_preserves_the_answer_key(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    original = generate(noisy_config)
    restored = SyntheticDataset.load(original.save(tmp_path / "run"))
    assert len(restored.ground_truth) == len(original.ground_truth)
    for before, after in zip(original.ground_truth.events, restored.ground_truth.events):
        assert after.event_id == before.event_id
        assert after.true_amplitude == pytest.approx(before.true_amplitude)
        assert after.true_ratio_to_reference == pytest.approx(before.true_ratio_to_reference)
        assert after.parent_event_id == before.parent_event_id


def test_round_trip_preserves_the_config(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    """A bundle must be reproducible from itself alone."""
    original = generate(noisy_config)
    restored = SyntheticDataset.load(original.save(tmp_path / "run"))
    assert restored.config == original.config
    # And regenerating from the restored config reproduces the same data.
    assert np.array_equal(
        generate(restored.config).streams["analyzer"]["ch4"].values,
        original.streams["analyzer"]["ch4"].values,
    )


def test_round_trip_preserves_attrs_and_coordinates(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    original = generate(noisy_config)
    restored = SyntheticDataset.load(original.save(tmp_path / "run"))
    stream = restored.streams["analyzer"]
    assert stream.attrs["tsara_stage"] == "synthetic"
    assert float(stream["latitude"]) == pytest.approx(40.0)
    assert "true_sys_abs_draw" in stream["ch4"].attrs


def test_round_trip_of_a_mobile_multi_stream_bundle(tmp_path: Path) -> None:
    config = SyntheticConfig.model_validate(
        {
            "name": "mobile",
            "start": "2026-01-01T00:00:00Z",
            "duration": "20min",
            "seed": 3,
            "platform": {
                "kind": "mobile",
                "start_latitude": 40.0,
                "start_longitude": -111.0,
            },
            "atmosphere": {
                "fields": {
                    "ch4": {"background": {"kind": "parametric", "offset": 1900.0}, "units": "ppb"}
                }
            },
            "instruments": {"analyzer": {"native_rate": "1s", "measures": {"ch4": {}}}},
        }
    )
    original = generate(config)
    restored = SyntheticDataset.load(original.save(tmp_path / "run"))
    assert set(restored.streams) == {"analyzer", "gps"}
    assert np.allclose(
        restored.streams["analyzer"]["latitude"].values,
        original.streams["analyzer"]["latitude"].values,
    )


def test_empty_catalog_round_trips(
    noise_free_config: SyntheticConfig, with_sources: WithSources, tmp_path: Path
) -> None:
    """The plume-free control case must persist like any other."""
    control = with_sources(noise_free_config, {})
    restored = SyntheticDataset.load(generate(control).save(tmp_path / "run"))
    assert len(restored.ground_truth) == 0


def test_round_trip_preserves_the_time_dtype(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    """Save/load must not silently change the time representation.

    netCDF stores nanoseconds, so any stream built at a coarser resolution
    would come back with a different dtype than it went in with — making a
    loaded bundle subtly unequal to the dataset that produced it.
    """
    original = generate(noisy_config)
    restored = SyntheticDataset.load(original.save(tmp_path / "run"))
    before = original.streams["analyzer"]["time"]
    after = restored.streams["analyzer"]["time"]
    assert before.dtype == after.dtype == np.dtype("datetime64[ns]")
    assert np.array_equal(before.values, after.values)


def test_round_trip_keeps_ground_truth_windows_usable(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    """Slicing a stream by a truth window must survive persistence.

    Parquet preserves whatever timezone the catalog carried, so a bundle
    written from an aware catalog would reload still unable to index its own
    streams.
    """
    restored = SyntheticDataset.load(generate(noisy_config).save(tmp_path / "run"))
    event = restored.ground_truth.events[0]
    assert event.peak_time.tz is None
    window = restored.streams["analyzer"].sel(time=slice(event.start_time, event.end_time))
    assert window.sizes["time"] > 0


def test_saving_twice_overwrites_in_place(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    dataset = generate(noisy_config)
    first = dataset.save(tmp_path / "run")
    second = dataset.save(tmp_path / "run")
    assert first == second
    assert len(load_bundle(second).streams) == 1


def test_manifest_records_bundle_contents(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    dataset = generate(noisy_config)
    bundle = dataset.save(tmp_path / "run")
    manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    assert manifest["stage"] == "synthetic"
    assert manifest["streams"] == ["analyzer"]
    assert manifest["n_ground_truth_rows"] == len(dataset.ground_truth)


def test_save_accepts_a_string_path(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    bundle = save_bundle(generate(noisy_config), str(tmp_path / "run"))
    assert bundle.is_dir()


# ---------------------------------------------------------------------------
# The atmosphere, rebuilt from the saved config (Phase 4.5)
# ---------------------------------------------------------------------------


def _wandering(profile: str | None = None) -> SyntheticConfig:
    """A campaign whose truth has a random walk, so a rebuild is not trivially flat."""
    background: dict[str, Any] = {
        "kind": "parametric",
        "offset": 1900.0,
        "diurnal_amplitude": 20.0,
        "random_walk_std": 30.0,
    }
    if profile is not None:
        background = {"kind": "bootstrap", "profile": profile, "base": background}
    return SyntheticConfig.model_validate(
        {
            "name": "wandering",
            "start": "2026-01-01T00:00:00Z",
            "duration": "30min",
            "seed": 17,
            "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
            "atmosphere": {
                "fields": {"ch4": {"units": "ppb", "background": background}},
                "sources": {
                    "pad": {
                        "rate_per_hour": 20.0,
                        "reference_species": "ch4",
                        "shape": {"kind": "gaussian", "sigma": "20s"},
                        "amplitude": {"kind": "uniform", "low": 50.0, "high": 150.0},
                    }
                },
            },
            "instruments": {
                "analyzer": {
                    "native_rate": "1s",
                    "timestamp_jitter": "0.2s",
                    "measures": {"ch4": {"uncertainty": {"random": {"absolute": 2.0}}}},
                }
            },
        }
    )


def test_a_loaded_bundle_rebuilds_the_atmosphere_exactly(tmp_path: Path) -> None:
    """The generator draws the air first, so its seed alone reproduces it."""
    original = generate(_wandering())
    restored = SyntheticDataset.load(original.save(tmp_path / "run"))
    assert original.atmosphere is not None
    assert restored.atmosphere is not None
    # At instants no instrument sampled, stochastic term and plumes included.
    instants = np.arange("2026-01-01T00:00:00", "2026-01-01T00:30:00", 7, dtype="datetime64[s]")
    assert np.array_equal(
        restored.atmosphere.value("ch4", instants), original.atmosphere.value("ch4", instants)
    )
    assert [e.event_id for e in restored.atmosphere.events] == [
        e.event_id for e in original.atmosphere.events
    ]


def test_a_bootstrap_bundle_loads_without_its_profile_but_without_an_atmosphere(
    white_noise_profile: RealDataProfile, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The streams and answer key are complete; only the rebuild needs real data."""
    config = _wandering(profile="white")
    bundle = generate(config, profiles={"white": white_noise_profile}).save(tmp_path / "run")
    with caplog.at_level(logging.INFO, logger="tsara.synthetic.bundle"):
        restored = load_bundle(bundle)
    assert restored.atmosphere is None
    assert "profile(s) ['white'] were not supplied" in caplog.text
    assert len(restored.ground_truth) > 0


def test_a_bootstrap_bundle_rebuilds_its_atmosphere_given_the_profile(
    white_noise_profile: RealDataProfile, tmp_path: Path
) -> None:
    config = _wandering(profile="white")
    original = generate(config, profiles={"white": white_noise_profile})
    restored = SyntheticDataset.load(
        original.save(tmp_path / "run"), profiles={"white": white_noise_profile}
    )
    assert original.atmosphere is not None
    assert restored.atmosphere is not None
    instants = original.streams["analyzer"]["time"].values
    assert np.array_equal(
        restored.atmosphere.background("ch4", instants),
        original.atmosphere.background("ch4", instants),
    )


def test_a_bundle_from_before_the_atmosphere_is_refused_by_name(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    """No converter, deliberately; the message says what happened and what to do."""
    bundle = generate(noisy_config).save(tmp_path / "run")
    payload = yaml.safe_load((bundle / BUNDLE_CONFIG).read_text())
    fields = payload.pop("atmosphere")["fields"]
    payload["sources"] = {}
    for instrument in payload["instruments"].values():
        instrument["species"] = {name: fields[name] for name in instrument.pop("measures")}
    (bundle / BUNDLE_CONFIG).write_text(yaml.safe_dump(payload))
    with pytest.raises(TsaraBundleError, match="written before the synthetic atmosphere existed"):
        load_bundle(bundle)


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "not-a-mapping-of-instruments", "instruments": ["analyzer"]},
        {"name": "instruments-without-species", "instruments": {"analyzer": {"native_rate": "1s"}}},
        ["not", "a", "mapping"],
    ],
)
def test_an_invalid_config_is_left_to_validation(
    noisy_config: SyntheticConfig, tmp_path: Path, payload: object
) -> None:
    """Only the old schema's unmistakable shape gets the old-schema message."""
    from pydantic import ValidationError

    bundle = generate(noisy_config).save(tmp_path / "run")
    (bundle / BUNDLE_CONFIG).write_text(yaml.safe_dump(payload))
    with pytest.raises(ValidationError):
        load_bundle(bundle)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_saving_over_a_file_is_rejected(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    target = tmp_path / "not_a_dir"
    target.write_text("occupied")
    with pytest.raises(TsaraBundleError, match="not a directory"):
        generate(noisy_config).save(target)


def test_loading_a_missing_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TsaraBundleError, match="does not exist"):
        load_bundle(tmp_path / "absent")


def test_loading_a_directory_without_a_manifest_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(TsaraBundleError, match="not a TSARA bundle"):
        load_bundle(tmp_path / "empty")


def test_incompatible_bundle_version_is_rejected(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    """A future layout must fail loudly rather than be misread."""
    bundle = generate(noisy_config).save(tmp_path / "run")
    manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    manifest["bundle_format_version"] = 99
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(TsaraBundleError, match="format version 99"):
        load_bundle(bundle)


def test_a_bundle_from_another_stage_is_rejected(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    """Say "wrong kind of bundle", not "corrupt bundle".

    An ingest bundle has the same skeleton as this one -- ``bundle.json`` at
    the same format version, beside a ``streams/`` directory -- and differs
    only in the stage-specific files. Without this check the loader reached
    the missing ``config.yaml`` first and reported a *damaged synthetic*
    bundle, which sends the reader looking for the wrong problem.
    """
    bundle = generate(noisy_config).save(tmp_path / "run")
    manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    manifest[BUNDLE_STAGE_KEY] = "ingest"
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(TsaraBundleError, match="'ingest' stage"):
        load_bundle(bundle)


def test_missing_config_is_reported(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    bundle = generate(noisy_config).save(tmp_path / "run")
    (bundle / BUNDLE_CONFIG).unlink()
    with pytest.raises(TsaraBundleError, match=BUNDLE_CONFIG):
        load_bundle(bundle)


def test_missing_ground_truth_is_reported(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    bundle = generate(noisy_config).save(tmp_path / "run")
    (bundle / BUNDLE_GROUND_TRUTH).unlink()
    with pytest.raises(TsaraBundleError, match=BUNDLE_GROUND_TRUTH):
        load_bundle(bundle)


def test_missing_stream_file_is_reported(noisy_config: SyntheticConfig, tmp_path: Path) -> None:
    bundle = generate(noisy_config).save(tmp_path / "run")
    (bundle / BUNDLE_STREAMS_DIR / "analyzer.nc").unlink()
    with pytest.raises(TsaraBundleError, match="analyzer.nc' is missing"):
        load_bundle(bundle)


# ---------------------------------------------------------------------------
# Cell boundaries (Phase 3.5)
# ---------------------------------------------------------------------------


def _strip_cells(bundle: Path, *, version: int = 1) -> None:
    """Remove every stream's cells, and stamp the bundle at ``version``.

    At version 1 that reproduces the older layout, which had no way to record
    cells. At version 2 it reproduces the *other* absence — a stream that
    could not have cells, or lost them — which must not be migrated.
    """
    import xarray as xr

    for target in sorted((bundle / BUNDLE_STREAMS_DIR).glob("*.nc")):
        with xr.open_dataset(target, engine="netcdf4", decode_coords="all") as opened:
            stream = opened.load()
        stream = stream.drop_vars(TIME_BOUNDS_VAR)
        # Rebuild the attrs rather than popping in place: version 1 never had
        # the bounds attribute at all, and leaving it behind would write a file
        # naming a variable that is not there, which is a different defect from
        # the one this helper is meant to reproduce.
        stream[TIME_COORD].attrs = {
            key: value for key, value in stream[TIME_COORD].attrs.items() if key != BOUNDS_ATTR
        }
        # And the encoding, which is where `decode_coords="all"` put the
        # bounds name on the way in. Clearing only attrs leaves xarray to
        # write the declaration straight back out, producing a file that
        # names a variable it does not contain.
        stream[TIME_COORD].encoding = {}
        stream.to_netcdf(target, engine="netcdf4")
    manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    manifest["bundle_format_version"] = version
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(manifest))


def test_cells_survive_a_bundle_round_trip_exactly(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    dataset = generate(noisy_config)
    reloaded = load_bundle(dataset.save(tmp_path / "run"))
    for name, stream in dataset.streams.items():
        assert TIME_BOUNDS_VAR in reloaded.streams[name].coords, name
        assert np.array_equal(
            reloaded.streams[name][TIME_BOUNDS_VAR].values.astype("datetime64[ns]"),
            stream[TIME_BOUNDS_VAR].values.astype("datetime64[ns]"),
        ), name


def test_a_version_1_bundle_is_migrated_rather_than_refused(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    """The older layout has an exact honest reading, so admit and label it."""
    bundle = generate(noisy_config).save(tmp_path / "run")
    _strip_cells(bundle)

    reloaded = load_bundle(bundle)
    for name, stream in reloaded.streams.items():
        assert TIME_BOUNDS_VAR in stream.coords, name
        # Centred on the original timestamps, so nothing moved.
        bounds = stream[TIME_BOUNDS_VAR].values.astype("datetime64[ns]").astype(np.int64)
        stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
        midpoints = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) // 2
        assert np.array_equal(midpoints, stamps), name
        # And every field of the support says where it came from.
        assert stream.attrs[SUPPORT_LABEL_SOURCE_ATTR] == "assumed", name
        assert stream.attrs[SUPPORT_WIDTH_SOURCE_ATTR] == "inferred", name
        assert stream.attrs[SUPPORT_METHOD_SOURCE_ATTR] == "assumed", name


def test_a_version_2_bundle_without_cells_is_not_completed(
    noisy_config: SyntheticConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The same absence means opposite things at the two versions.

    Version 1 had no way to record cells, so completing a stream adds
    information. From version 2 on, a stream without them either could not
    have them or has lost them, and inventing a cadence would paper over the
    second while promoting the first up the provenance ladder. The loader ran
    the migration unconditionally, so version was never consulted.
    """
    bundle = generate(noisy_config).save(tmp_path / "run")
    _strip_cells(bundle, version=2)
    with caplog.at_level(logging.INFO, logger="tsara.synthetic.bundle"):
        reloaded = load_bundle(bundle)
    for name, stream in reloaded.streams.items():
        assert TIME_BOUNDS_VAR not in stream.coords, name
    assert "assumed cells" not in caplog.text


def test_the_migration_says_what_it_did(
    noisy_config: SyntheticConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A weak reading applied silently would be worse than no reading."""
    bundle = generate(noisy_config).save(tmp_path / "run")
    _strip_cells(bundle)
    with caplog.at_level(logging.INFO, logger="tsara.synthetic.bundle"):
        load_bundle(bundle)
    assert "assumed cells" in caplog.text


def test_saving_refuses_a_stream_whose_cells_were_destroyed(
    noisy_config: SyntheticConfig, tmp_path: Path
) -> None:
    """The guard at the persistence boundary: an xarray resample upstream
    leaves the bounds attribute naming a variable that no longer exists."""
    dataset = generate(noisy_config)
    name = next(iter(dataset.streams))
    dataset.streams[name] = dataset.streams[name].drop_vars(TIME_BOUNDS_VAR)
    with pytest.raises(TsaraSupportError, match="names 'time_bnds'"):
        save_bundle(dataset, tmp_path / "broken")
