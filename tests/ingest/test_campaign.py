"""Tests for campaign orchestration and bundle persistence.

These build a small archive on disk and ingest it end to end, because the
behaviour worth testing here is the *order* the pieces run in — concatenate
before masking, sort before assembling — which no unit test of an individual
piece can observe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.config.manifest import InstrumentConfig, Manifest, StationaryPlatform
from tsara.core.bundle import BUNDLE_MANIFEST, BUNDLE_STREAMS_DIR, TsaraBundleError
from tsara.core.naming import (
    LOD_COUNT_KEY,
    RAW_TIME_START_COLUMN,
    RAW_TIME_STOP_COLUMN,
    SUPPORT_WIDTH_ATTR,
    SUPPORT_WIDTH_PROVENANCE_ATTR,
    TIME_BOUNDS_VAR,
    TIME_SHIFT_ATTR,
)
from tsara.core.support import (
    CellBounds,
    attach_time_bounds,
    declared_bounds_name,
    nominal_cadence_ns,
)
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.bundle import BUNDLE_MANIFEST_CONFIG, load_streams, save_streams
from tsara.ingest.campaign import (
    StreamCollection,
    _ingest_instrument,
    _merge_file_attrs,
    _n_within_file,
    ingest_campaign,
)
from tsara.ingest.streams import build_stream

# Spelled out rather than imported from conftest, as in tests/config/test_loader.py:
# `from tests.conftest import ...` resolves only when pytest runs from the repo root.
RespellBundle = Callable[[Path], int]


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


def _frame() -> pd.DataFrame:
    """A minimal one-column table for exercising build_stream directly."""
    index = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01"])
    return pd.DataFrame({"CH4": [1900.0, 1901.0]}, index=index)


def _instrument() -> InstrumentConfig:
    return InstrumentConfig.model_validate(
        {
            "loader": {
                "format": "csv",
                "path_template": "*.csv",
                "time": {"column": "t", "format": "%Y-%m-%d %H:%M:%S"},
            },
            "variables": {"ch4": {"column": "CH4", "role": "gas", "units": "ppb"}},
        }
    )


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
    assert collection["picarro"].attrs["n_files"] == 2


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


def test_collection_iterates_over_instrument_names(tmp_path: Path) -> None:
    """A half-mapping fails here with ``KeyError: 0``, not a useful error.

    This is the regression test for the missing ``__iter__``: with only
    ``__getitem__``/``__len__`` defined, Python's legacy iteration protocol
    indexes the object with integers, so every one of the four idioms below
    raised ``KeyError: 0`` from inside ``__getitem__`` -- an error naming
    neither iteration nor this class. The docstring promised a mapping; the
    test asserting that promise only checked lookup, which is why the gap
    survived a whole phase.
    """
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    assert list(collection) == ["picarro"]
    assert sorted(collection) == ["picarro"]
    assert dict(collection) == collection.streams
    seen: list[str] = []
    for name in collection:  # the idiom that used to raise
        seen.append(name)
    assert seen == ["picarro"]


def test_collection_supplies_the_rest_of_the_mapping_api(tmp_path: Path) -> None:
    """``keys``/``items``/``values``/``get`` come free from the ABC."""
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    assert list(collection.keys()) == ["picarro"]
    assert [name for name, _ in collection.items()] == ["picarro"]
    assert [stream.sizes["time"] for stream in collection.values()] == [
        collection["picarro"].sizes["time"]
    ]
    assert collection.get("absent") is None


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
    assert reloaded["picarro"]["ch4"].attrs["uncertainty_provenance"] == "empirical"


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


# ---------------------------------------------------------------------------
# Reconciling what the files said about themselves
# ---------------------------------------------------------------------------


def test_agreeing_file_attrs_are_carried_through() -> None:
    merged = _merge_file_attrs([{"icartt_pi": "Lin, John"}, {"icartt_pi": "Lin, John"}])
    assert merged["icartt_pi"] == "Lin, John"


def test_disagreeing_file_attrs_say_so_rather_than_picking() -> None:
    """Silently choosing one would put a false statement in a self-describing file."""
    merged = _merge_file_attrs([{"icartt_revision": "R0"}, {"icartt_revision": "R1"}])
    assert merged["icartt_revision"] == "R0; R1"


def test_many_distinct_values_are_summarised_not_listed() -> None:
    """An ICARTT data date differs per file; a 1000-file instrument must not
    write a 1000-item attr."""
    per_file: list[Mapping[str, object]] = [
        {"icartt_data_date": f"2024-08-{d:02d}"} for d in range(1, 21)
    ]
    merged = _merge_file_attrs(per_file)
    value = merged["icartt_data_date"]
    assert isinstance(value, str)
    assert value.startswith("2024-08-01 ... 2024-08-20")
    assert "20 distinct values" in value


def test_lod_counts_are_summed_across_files_not_reconciled() -> None:
    """A tally over files is the tally over the concatenated record."""
    merged = _merge_file_attrs(
        [
            {LOD_COUNT_KEY: {"Benzene_PPBV": 10, "Toluene_PPBV": 2}},
            {LOD_COUNT_KEY: {"Benzene_PPBV": 5}},
        ]
    )
    assert merged[LOD_COUNT_KEY] == {"Benzene_PPBV": 15, "Toluene_PPBV": 2}


def test_no_file_attrs_merges_to_nothing() -> None:
    assert _merge_file_attrs([]) == {}


def test_file_attrs_reach_the_saved_stream(tmp_path: Path) -> None:
    """The whole point: a stream found on disk explains itself (CLAUDE.md 5)."""
    base = _archive(tmp_path)
    stream = ingest_campaign(_manifest(base))["picarro"]
    # A CSV declares nothing about itself, so nothing is invented for it.
    assert "icartt_pi" not in stream.attrs
    assert stream.attrs["tsara_stage"] == "ingest"


def test_tsara_attrs_win_a_name_collision(tmp_path: Path) -> None:
    """What the package did is not negotiable; a header field is whatever
    the producer wrote."""
    dataset = build_stream(
        _frame(),
        _instrument(),
        name="x",
        platform=StationaryPlatform(kind="stationary", latitude=0.0, longitude=0.0),
        file_attrs={"tsara_stage": "not-really", "icartt_pi": "Someone"},
    )
    assert dataset.attrs["tsara_stage"] == "ingest"
    assert dataset.attrs["icartt_pi"] == "Someone"


# ---------------------------------------------------------------------------
# Duplicate timestamps: naming the cause rather than guessing it
# ---------------------------------------------------------------------------


def test_duplicates_within_one_file_are_reported_as_such(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Measured on the real PTR-MS set: all 7,242 were within-file and none
    from overlap, while the message asked 'Overlapping files?'."""
    base = tmp_path / "data"
    _write_csv(
        base / "picarro" / "a.csv",
        [("2026-01-01 00:00:00", 1900.0), ("2026-01-01 00:00:00", 1901.0)],
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.campaign"):
        ingest_campaign(_manifest(base))

    assert "1 duplicated within a single file" in caplog.text
    assert "0 from overlap between files" in caplog.text


def test_duplicates_across_files_are_reported_as_such(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    base = tmp_path / "data"
    _write_csv(base / "picarro" / "a.csv", [("2026-01-01 00:00:00", 1900.0)])
    _write_csv(base / "picarro" / "b.csv", [("2026-01-01 00:00:00", 9999.0)])

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.campaign"):
        ingest_campaign(_manifest(base))

    assert "0 duplicated within a single file" in caplog.text
    assert "1 from overlap between files" in caplog.text
    assert "path templates" in caplog.text


def test_within_file_duplicates_are_counted_exactly_when_unsorted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Counted per file before concatenation, so a file that is internally
    out of order does not get its duplicates blamed on overlap."""
    base = tmp_path / "data"
    _write_csv(
        base / "picarro" / "a.csv",
        [
            ("2026-01-01 00:00:00", 1900.0),
            ("2026-01-01 00:00:05", 1901.0),
            ("2026-01-01 00:00:00", 1902.0),
        ],
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.campaign"):
        ingest_campaign(_manifest(base))

    assert "1 duplicated within a single file" in caplog.text
    assert "0 from overlap between files" in caplog.text


# ---------------------------------------------------------------------------
# Bundles do not accumulate streams that are no longer theirs
# ---------------------------------------------------------------------------


def test_resaving_a_subset_removes_the_stale_stream_file(tmp_path: Path) -> None:
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
    full = ingest_campaign(Manifest.model_validate(spec))
    bundle = save_streams(full, tmp_path / "bundle")
    assert (bundle / BUNDLE_STREAMS_DIR / "other.nc").is_file()

    subset = StreamCollection(streams={"picarro": full["picarro"]}, manifest=full.manifest)
    save_streams(subset, bundle)

    assert not (bundle / BUNDLE_STREAMS_DIR / "other.nc").exists()
    assert (bundle / BUNDLE_STREAMS_DIR / "picarro.nc").is_file()
    assert set(load_streams(bundle).streams) == {"picarro"}


def test_unrelated_files_in_the_streams_directory_are_left_alone(tmp_path: Path) -> None:
    base = _archive(tmp_path)
    collection = ingest_campaign(_manifest(base))
    bundle = save_streams(collection, tmp_path / "bundle")
    note = bundle / BUNDLE_STREAMS_DIR / "README.txt"
    note.write_text("mine", encoding="utf-8")

    save_streams(collection, bundle)

    assert note.is_file()


def test_ingested_streams_carry_cells_of_their_own(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Assembly resolves support now, so nothing is left for the loader.

    Until stream assembly read the manifest's declaration, a saved ingest
    bundle had no cells and the loader completed it with assumed ones. It
    does not any more, and the absence of that notice is the check.
    """
    save_streams(ingest_campaign(_manifest(_archive(tmp_path))), tmp_path / "bundle")
    with caplog.at_level(logging.INFO, logger="tsara.ingest.bundle"):
        reloaded = load_streams(tmp_path / "bundle")
    assert "assumed cells" not in caplog.text
    for name in reloaded.streams:
        assert declared_bounds_name(reloaded[name]) is not None, name
        assert TIME_BOUNDS_VAR in reloaded[name].coords, name


def test_streams_that_already_carry_cells_are_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The shape the reader stage will produce: declared cells must survive.

    Start-labelled, because that is the only geometry an assumed migration
    could not accidentally reproduce -- centred cadence cells would be
    identical to the real ones and the substitution would be invisible.
    """
    collection = ingest_campaign(_manifest(_archive(tmp_path)))
    expected = {}
    for name, stream in collection.streams.items():
        stamps = stream["time"].values.astype("datetime64[ns]").astype(np.int64)
        cadence = nominal_cadence_ns(stamps)
        assert cadence is not None
        attach_time_bounds(stream, CellBounds.from_label(stamps, cadence, "start"), "mean")
        expected[name] = stream[TIME_BOUNDS_VAR].values.copy()

    save_streams(collection, tmp_path / "bundle")
    with caplog.at_level(logging.INFO, logger="tsara.ingest.bundle"):
        reloaded = load_streams(tmp_path / "bundle")

    assert "assumed cells" not in caplog.text
    for name, bounds in expected.items():
        assert np.array_equal(reloaded[name][TIME_BOUNDS_VAR].values, bounds), name


# ---------------------------------------------------------------------------
# Support resolution across an instrument's files (Phase 3.5)
# ---------------------------------------------------------------------------


def test_each_file_keeps_its_own_cadence(tmp_path: Path) -> None:
    """Cadence is measured per FILE, and that is load-bearing.

    One instrument's files can legitimately disagree: in the target archive
    some met records run at 1 s in one file and 5 s in another. A single
    instrument-wide cadence would give one of them cells of the wrong width,
    and it would be a whole file's worth rather than a stray row.
    """
    base = tmp_path / "data"
    _write_csv(
        base / "picarro" / "a_fast.csv",
        [("2026-01-01 00:00:00", 1900.0), ("2026-01-01 00:00:01", 1901.0)],
    )
    _write_csv(
        base / "picarro" / "b_slow.csv",
        [("2026-01-01 01:00:00", 1902.0), ("2026-01-01 01:00:05", 1903.0)],
    )
    ingested = _ingest_instrument(
        _manifest(base), "picarro", _manifest(base).instruments["picarro"]
    )
    widths = (ingested.frame[RAW_TIME_STOP_COLUMN] - ingested.frame[RAW_TIME_START_COLUMN]).tolist()
    assert widths == [pd.Timedelta("1s")] * 2 + [pd.Timedelta("5s")] * 2
    # And so no single nominal width is reported for the instrument.
    assert ingested.support.width_ns is None
    assert ingested.support.width_provenance == "inferred"


def test_a_gap_does_not_widen_the_cell_before_it(tmp_path: Path) -> None:
    """The rule the whole width design rests on: a dropped row leaves a hole
    in the tiling, never a wider cell. Under the rejected alternative the
    widest cell in the real archive would have been 23 days."""
    base = tmp_path / "data"
    _write_csv(
        base / "picarro" / "a.csv",
        [
            ("2026-01-01 00:00:00", 1900.0),
            ("2026-01-01 00:00:01", 1901.0),
            ("2026-01-01 03:00:00", 1902.0),
            ("2026-01-01 03:00:01", 1903.0),
        ],
    )
    ingested = _ingest_instrument(
        _manifest(base), "picarro", _manifest(base).instruments["picarro"]
    )
    widths = (ingested.frame[RAW_TIME_STOP_COLUMN] - ingested.frame[RAW_TIME_START_COLUMN]).unique()
    assert list(widths) == [pd.Timedelta("1s")]


def _write_csv_cells(path: Path, rows: list[tuple[str, str, float]]) -> None:
    """A file that states each row's own cell.

    The shape 169 of the 2024 archive's 1122 ICARTT files have: a start time
    on the axis and a companion stop column, so widths can differ row to row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{start},{stop},{value}" for start, stop, value in rows)
    path.write_text(f"t,stop,CH4\n{body}\n", encoding="utf-8")


def _cells_manifest(base: Path) -> Manifest:
    """The same manifest, with the stop column declared."""
    spec = _manifest(base).model_dump(mode="json")
    spec["instruments"]["picarro"]["loader"]["support"] = {"stop_column": "stop"}
    return Manifest.model_validate(spec)


def test_cells_are_centred_before_the_record_is_sorted(tmp_path: Path) -> None:
    """Centring can genuinely reorder a stream, so the sort has to come after.

    With one width per row, centring is not a translation: a wide cell moves
    its timestamp further than a narrow one. Here a 100 s cell starts first
    but is centred *after* the 10 s cell that starts ten seconds later, so
    sorting the raw axis and centring afterwards leaves the stream
    non-monotonic -- which `build_stream` refuses, failing an archive that is
    perfectly legitimate.

    The ordering was already correct and already explained in a comment; what
    was missing was anything that would notice if it changed. Swapping the two
    calls passed the entire suite.
    """
    base = tmp_path / "data"
    _write_csv_cells(
        base / "picarro" / "a.csv",
        [
            ("2026-01-01 00:00:00", "2026-01-01 00:01:40", 1900.0),  # 100 s, mid :50
            ("2026-01-01 00:00:10", "2026-01-01 00:00:20", 1901.0),  # 10 s, mid :15
        ],
    )
    manifest = _cells_manifest(base)
    ingested = _ingest_instrument(manifest, "picarro", manifest.instruments["picarro"])

    assert ingested.frame.index.is_monotonic_increasing
    # The narrow cell comes first, because its midpoint does.
    assert ingested.frame["CH4"].tolist() == [1901.0, 1900.0]
    assert [str(stamp) for stamp in ingested.frame.index] == [
        "2026-01-01 00:00:15",
        "2026-01-01 00:00:50",
    ]


def test_the_duplicate_split_is_counted_on_the_axis_rows_are_dropped_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The two halves of the dropped-row message must describe one axis.

    Three rows share a start; two declare the same stop and one a longer one.
    On the raw axis all three are duplicates, but after centring only the two
    identical cells still collide -- so a within-file count taken before
    centring exceeded the total taken after it, and the "overlap between
    files" remainder came out **negative**, telling the user to check path
    templates for a single-file instrument.
    """
    base = tmp_path / "data"
    _write_csv_cells(
        base / "picarro" / "a.csv",
        [
            ("2026-01-01 00:00:10", "2026-01-01 00:00:20", 1900.0),
            ("2026-01-01 00:00:10", "2026-01-01 00:00:20", 1901.0),
            ("2026-01-01 00:00:10", "2026-01-01 00:00:30", 1902.0),
            ("2026-01-01 00:00:40", "2026-01-01 00:00:50", 1903.0),
        ],
    )
    manifest = _cells_manifest(base)
    with caplog.at_level(logging.WARNING):
        ingested = _ingest_instrument(manifest, "picarro", manifest.instruments["picarro"])

    # The duplicated cell is dropped; the wider cell sharing its start is not,
    # because it describes a different interval of air.
    assert len(ingested.frame) == 3
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "dropped 1 row(s)" in message
    assert "1 duplicated within a single file" in message
    assert "0 from overlap between files" in message


def test_overlap_between_files_is_still_counted(tmp_path: Path) -> None:
    """The other half of the split still works: two files, one shared cell."""
    base = tmp_path / "data"
    for name in ("a.csv", "b.csv"):
        _write_csv_cells(
            base / "picarro" / name,
            [("2026-01-01 00:00:00", "2026-01-01 00:00:10", 1900.0)],
        )
    manifest = _cells_manifest(base)
    frames = [pd.DataFrame(index=pd.DatetimeIndex(["2026-01-01 00:00:05"]))] * 2
    assert _n_within_file(pd.concat(frames).index, [1, 1]) == 0
    ingested = _ingest_instrument(manifest, "picarro", manifest.instruments["picarro"])
    assert len(ingested.frame) == 1


def test_boundaries_survive_sorting_and_de_duplication(tmp_path: Path) -> None:
    """Cells ride along as columns precisely so this needs no special care."""
    ingested = _ingest_instrument(
        _manifest(_archive(tmp_path)),
        "picarro",
        _manifest(_archive(tmp_path)).instruments["picarro"],
    )
    assert ingested.frame.index.is_monotonic_increasing
    starts = ingested.frame[RAW_TIME_START_COLUMN]
    assert starts.is_monotonic_increasing
    assert (ingested.frame.index - starts == pd.Timedelta("1s")).all()


def test_a_version_1_ingest_bundle_is_still_migrated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Bundles written before cells existed stay readable.

    Assembly gives every new stream its cells, so the loader's migration path
    is now reached only by an older bundle -- which is exactly the population
    it was built for.
    """
    import json

    import xarray as xr

    bundle = tmp_path / "bundle"
    save_streams(ingest_campaign(_manifest(_archive(tmp_path))), bundle)
    for target in sorted((bundle / BUNDLE_STREAMS_DIR).glob("*.nc")):
        with xr.open_dataset(target, engine="netcdf4", decode_coords="all") as opened:
            stream = opened.load()
        stream = stream.drop_vars(TIME_BOUNDS_VAR)
        stream["time"].attrs = {
            key: value for key, value in stream["time"].attrs.items() if key != "bounds"
        }
        stream["time"].encoding = {}
        stream.to_netcdf(target, engine="netcdf4")
    descriptor = json.loads((bundle / BUNDLE_MANIFEST).read_text())
    descriptor["bundle_format_version"] = 1
    (bundle / BUNDLE_MANIFEST).write_text(json.dumps(descriptor))

    with caplog.at_level(logging.INFO, logger="tsara.ingest.bundle"):
        reloaded = load_streams(bundle)
    assert "assumed cells" in caplog.text
    for name in reloaded.streams:
        assert declared_bounds_name(reloaded[name]) is not None, name


def test_a_format_2_ingest_bundle_is_respelled_on_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, respell_as_format_2: RespellBundle
) -> None:
    """Format 3 only renamed attributes, so an older bundle reads back identical.

    Compared with `identical` against a reload of the same bundle before it
    was respelled, which checks every attribute of the dataset and of each
    variable, not only the renamed ones.
    """
    bundle = tmp_path / "bundle"
    save_streams(ingest_campaign(_manifest(_archive(tmp_path))), bundle)
    expected = load_streams(bundle)
    assert respell_as_format_2(bundle) > 0

    with caplog.at_level(logging.INFO, logger="tsara.ingest.bundle"):
        reloaded = load_streams(bundle)
    assert "current vocabulary" in caplog.text
    for name in expected.streams:
        assert reloaded[name].identical(expected[name]), name


def test_a_version_2_stream_without_cells_is_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Two absences that look identical on disk and mean opposite things.

    In a version-1 stream, no `time_bnds` means the format could not record
    one, so completing it adds information. In a version-2 stream it means
    ingestion *could not know*: no declared width, and no file long enough to
    measure a cadence — and it said so instead of inventing a number.

    The loader ran the migration on every stream regardless of version, so
    saving and reloading this instrument attached 60 s cells and moved
    `width_provenance` from `assumed` to `inferred`. A stream was promoted up the
    provenance ladder by nothing but a trip through disk, and the log line
    said it had been "written before cell boundaries existed" about a bundle
    written a moment earlier at version 2.
    """
    base = tmp_path / "data"
    # One row per file, so no file has a measurable cadence — but the
    # concatenated record has three timestamps a minute apart, which is
    # exactly what the migration would have measured.
    for i, stamp in enumerate(("00:00:00", "00:01:00", "00:02:00")):
        _write_csv(base / "picarro" / f"c{i}.csv", [(f"2026-01-01 {stamp}", 1900.0 + i)])

    streams = ingest_campaign(_manifest(base))
    assert declared_bounds_name(streams["picarro"]) is None
    assert streams["picarro"].attrs[SUPPORT_WIDTH_PROVENANCE_ATTR] == "assumed"

    bundle = tmp_path / "bundle"
    save_streams(streams, bundle)
    with caplog.at_level(logging.INFO, logger="tsara.ingest.bundle"):
        reloaded = load_streams(bundle)

    assert declared_bounds_name(reloaded["picarro"]) is None
    assert reloaded["picarro"].attrs[SUPPORT_WIDTH_PROVENANCE_ATTR] == "assumed"
    assert SUPPORT_WIDTH_ATTR not in reloaded["picarro"].attrs
    assert "assumed cells" not in caplog.text


def test_a_zero_clock_correction_is_not_announced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Declaring no offset is a legitimate way of saying "already corrected",
    and it should not read like something happened."""
    manifest = _manifest(_archive(tmp_path))
    manifest.instruments["picarro"].__dict__["time_shift"] = "0s"
    with caplog.at_level(logging.INFO, logger="tsara.ingest.campaign"):
        streams = ingest_campaign(manifest)
    assert "time_shift" not in caplog.text
    assert streams["picarro"].attrs[TIME_SHIFT_ATTR] == "0s"
