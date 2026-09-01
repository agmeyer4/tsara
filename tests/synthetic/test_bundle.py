"""Tests for reading and writing the on-disk TSARA bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tsara.core.bundle import BUNDLE_STAGE_KEY
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
    )
    original = generate(config)
    restored = SyntheticDataset.load(original.save(tmp_path / "run"))
    assert set(restored.streams) == {"analyzer", "gps"}
    assert np.allclose(
        restored.streams["analyzer"]["latitude"].values,
        original.streams["analyzer"]["latitude"].values,
    )


def test_empty_catalog_round_trips(noise_free_config: SyntheticConfig, tmp_path: Path) -> None:
    """The plume-free control case must persist like any other."""
    control = noise_free_config.model_copy(update={"sources": {}})
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
