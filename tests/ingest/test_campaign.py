"""Tests for campaign orchestration and bundle persistence.

These build a small archive on disk and ingest it end to end, because the
behaviour worth testing here is the *order* the pieces run in — concatenate
before masking, sort before assembling — which no unit test of an individual
piece can observe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tsara.config.manifest import Manifest
from tsara.core.bundle import BUNDLE_MANIFEST, BUNDLE_STREAMS_DIR, TsaraBundleError
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.bundle import BUNDLE_MANIFEST_CONFIG, load_streams, save_streams
from tsara.ingest.campaign import StreamCollection, ingest_campaign


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{t},{v}" for t, v in rows)
    path.write_text(f"t,CH4\n{body}\n", encoding="utf-8")


def _manifest(base: Path, **overrides: Any) -> Manifest:
    spec: dict[str, Any] = {
        "name": "test_campaign",
        "base_path": str(base),
        "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
        "instruments": {
            "picarro": {
                "loader": {
                    "format": "csv",
                    "path_template": "picarro/*.csv",
                    "time": {"column": "t", "format": "%Y-%m-%d %H:%M:%S"},
                },
                "variables": {"ch4": {"column": "CH4", "role": "gas", "units": "ppb"}},
            }
        },
    }
    spec.update(overrides)
    return Manifest.model_validate(spec)


def _archive(tmp_path: Path) -> Path:
    """Two files, deliberately crawled in an order that is not time order."""
    base = tmp_path / "data"
    _write_csv(
        base / "picarro" / "b_later.csv",
        [("2026-01-01 00:00:10", 1902.0), ("2026-01-01 00:00:12", 1903.0)],
    )
    _write_csv(
        base / "picarro" / "a_earlier.csv",
        [("2026-01-01 00:00:00", 1900.0), ("2026-01-01 00:00:02", 1901.0)],
    )
    return base


# ---------------------------------------------------------------------------
# Ingesting
# ---------------------------------------------------------------------------


def test_ingests_every_instrument(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))

    assert set(collection.streams) == {"picarro"}
    assert collection["picarro"].sizes["time"] == 4


def test_files_are_concatenated_across_the_archive(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    assert collection["picarro"].attrs["n_source_files"] == 2


def test_records_are_sorted_into_time_order(tmp_path: Path) -> None:
    """Crawl order is path order, which is not time order."""
    stream = ingest_campaign(_manifest(_archive(tmp_path)))["picarro"]
    times = stream["time"].values
    assert bool((np.diff(times) > np.timedelta64(0)).all())
    assert stream["ch4"].values.tolist() == [1900.0, 1901.0, 1902.0, 1903.0]


def test_duplicate_timestamps_keep_the_first_and_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Overlapping files are real; silently averaging them would invent a value."""
    base = tmp_path / "data"
    _write_csv(base / "picarro" / "a.csv", [("2026-01-01 00:00:00", 1900.0)])
    _write_csv(base / "picarro" / "b.csv", [("2026-01-01 00:00:00", 9999.0)])

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.campaign"):
        stream = ingest_campaign(_manifest(base))["picarro"]

    assert stream.sizes["time"] == 1
    assert stream["ch4"].values.tolist() == [1900.0]
    assert "sharing a timestamp" in caplog.text


def test_qaqc_sees_the_whole_record_not_one_file(tmp_path: Path) -> None:
    """A per-file range rule would be identical here; a per-file *count* is not."""
    base = _archive(tmp_path)
    manifest = _manifest(base)
    spec = manifest.model_dump(mode="json")
    spec["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [{"kind": "range", "min": 1901.5}]
    stream = ingest_campaign(Manifest.model_validate(spec))["picarro"]
    # 2 of the 4 concatenated samples, counted once against the whole record.
    assert stream["ch4"].attrs["qaqc_masked"] == "range:2"


def test_selecting_a_subset_of_instruments(tmp_path: Path) -> None:
    base = _archive(tmp_path)
    _write_csv(base / "other" / "x.csv", [("2026-01-01 00:00:00", 5.0)])
    manifest = _manifest(base)
    spec = manifest.model_dump(mode="json")
    spec["instruments"]["other"] = {
        "loader": {
            "format": "csv",
            "path_template": "other/*.csv",
            "time": {"column": "t", "format": "%Y-%m-%d %H:%M:%S"},
        },
        "variables": {"co2": {"column": "CH4", "role": "gas", "units": "ppm"}},
    }
    full = Manifest.model_validate(spec)

    assert set(ingest_campaign(full).streams) == {"picarro", "other"}
    assert set(ingest_campaign(full, instruments=["picarro"]).streams) == {"picarro"}


def test_unknown_instrument_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TsaraIngestError, match="no instrument"):
        ingest_campaign(_manifest(_archive(tmp_path)), instruments=["nope"])


def test_instrument_matching_no_files_is_an_error(tmp_path: Path) -> None:
    base = tmp_path / "data"
    base.mkdir()
    with pytest.raises(TsaraIngestError, match="No files found"):
        ingest_campaign(_manifest(base))


def test_an_unreadable_file_is_skipped_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One truncated file must not cost a whole campaign's run."""
    base = _archive(tmp_path)
    (base / "picarro" / "c_broken.csv").write_text("t,CH4\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="tsara.ingest.campaign"):
        stream = ingest_campaign(_manifest(base))["picarro"]

    assert stream.sizes["time"] == 4
    assert "Skipping" in caplog.text


def test_losing_every_file_is_an_error(tmp_path: Path) -> None:
    base = tmp_path / "data"
    (base / "picarro").mkdir(parents=True)
    (base / "picarro" / "broken.csv").write_text("t,CH4\n", encoding="utf-8")
    with pytest.raises(TsaraIngestError, match="none could be read"):
        ingest_campaign(_manifest(base))


def test_a_single_file_needs_no_concatenation(tmp_path: Path) -> None:
    base = tmp_path / "data"
    _write_csv(base / "picarro" / "only.csv", [("2026-01-01 00:00:00", 1900.0)])
    assert ingest_campaign(_manifest(base))["picarro"].sizes["time"] == 1


# ---------------------------------------------------------------------------
# The collection
# ---------------------------------------------------------------------------


def test_collection_behaves_like_a_mapping(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    assert len(collection) == 1
    assert "picarro" in collection
    assert "absent" not in collection
    assert collection["picarro"] is collection.streams["picarro"]


def test_collection_keeps_its_manifest(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    assert collection.manifest.name == "test_campaign"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_round_trip(tmp_path: Path) -> None:
    original = ingest_campaign(_manifest(_archive(tmp_path)))
    save_streams(original, tmp_path / "bundle")
    reloaded = load_streams(tmp_path / "bundle")

    assert set(reloaded.streams) == set(original.streams)
    assert reloaded.manifest.name == original.manifest.name
    np.testing.assert_array_equal(
        reloaded["picarro"]["ch4"].values, original["picarro"]["ch4"].values
    )


def test_round_trip_preserves_the_time_axis_exactly(tmp_path: Path) -> None:
    """netCDF stores ns; an unpinned axis would change dtype here."""
    original = ingest_campaign(_manifest(_archive(tmp_path)))
    save_streams(original, tmp_path / "bundle")
    reloaded = load_streams(tmp_path / "bundle")

    assert reloaded["picarro"]["time"].dtype == "datetime64[ns]"
    np.testing.assert_array_equal(
        reloaded["picarro"]["time"].values, original["picarro"]["time"].values
    )


def test_round_trip_preserves_provenance_attrs(tmp_path: Path) -> None:
    original = ingest_campaign(_manifest(_archive(tmp_path)))
    save_streams(original, tmp_path / "bundle")
    reloaded = load_streams(tmp_path / "bundle")

    assert reloaded["picarro"].attrs["tsara_stage"] == "ingest"
    assert reloaded["picarro"]["ch4"].attrs["uncertainty_source"] == "empirical"


def test_bundle_layout(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    bundle = save_streams(collection, tmp_path / "bundle")

    assert (bundle / BUNDLE_MANIFEST).is_file()
    assert (bundle / BUNDLE_MANIFEST_CONFIG).is_file()
    assert (bundle / BUNDLE_STREAMS_DIR / "picarro.nc").is_file()


def test_save_accepts_a_string_path(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    assert save_streams(collection, str(tmp_path / "bundle")).is_dir()


def test_saving_over_a_file_is_an_error(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    target = tmp_path / "afile"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(TsaraBundleError, match="not a directory"):
        save_streams(collection, target)


def test_loading_a_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TsaraBundleError, match="not an existing directory"):
        load_streams(tmp_path / "nope")


def test_loading_a_directory_without_a_descriptor_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(TsaraBundleError, match="not a TSARA bundle"):
        load_streams(tmp_path / "empty")


def test_invalid_descriptor_json_is_an_error(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / BUNDLE_MANIFEST).write_text("{not json", encoding="utf-8")
    with pytest.raises(TsaraBundleError, match="not valid JSON"):
        load_streams(bundle)


def test_incompatible_format_version_is_refused(tmp_path: Path) -> None:
    """Refusing beats misinterpreting; that is what a version is for."""
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    bundle = save_streams(collection, tmp_path / "bundle")
    (bundle / BUNDLE_MANIFEST).write_text(
        '{"bundle_format_version": 99, "stage": "ingest", "streams": []}',
        encoding="utf-8",
    )
    with pytest.raises(TsaraBundleError, match="format version 99"):
        load_streams(bundle)


def test_a_bundle_from_another_stage_is_refused(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    bundle = save_streams(collection, tmp_path / "bundle")
    (bundle / BUNDLE_MANIFEST).write_text(
        '{"bundle_format_version": 1, "stage": "synthetic", "streams": []}',
        encoding="utf-8",
    )
    with pytest.raises(TsaraBundleError, match="'synthetic' stage"):
        load_streams(bundle)


def test_a_missing_stream_file_is_an_error(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    bundle = save_streams(collection, tmp_path / "bundle")
    (bundle / BUNDLE_STREAMS_DIR / "picarro.nc").unlink()
    with pytest.raises(TsaraBundleError, match="is missing"):
        load_streams(bundle)


def test_a_missing_manifest_is_an_error(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    bundle = save_streams(collection, tmp_path / "bundle")
    (bundle / BUNDLE_MANIFEST_CONFIG).unlink()
    with pytest.raises(TsaraBundleError, match="bundle is incomplete"):
        load_streams(bundle)


def test_an_unreadable_manifest_is_an_error(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    bundle = save_streams(collection, tmp_path / "bundle")
    (bundle / BUNDLE_MANIFEST_CONFIG).write_text("name: []\n", encoding="utf-8")
    with pytest.raises(TsaraBundleError, match="Could not read the manifest"):
        load_streams(bundle)


def test_the_saved_manifest_records_the_resolved_base_path(tmp_path: Path) -> None:
    """A bundle records what ran, not what a relative path meant somewhere."""
    base = _archive(tmp_path)
    collection = ingest_campaign(_manifest(base))
    save_streams(collection, tmp_path / "bundle")
    assert load_streams(tmp_path / "bundle").manifest.base_path == base


def test_collection_is_reconstructed_as_the_same_type(tmp_path: Path) -> None:
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    save_streams(collection, tmp_path / "bundle")
    assert isinstance(load_streams(tmp_path / "bundle"), StreamCollection)
