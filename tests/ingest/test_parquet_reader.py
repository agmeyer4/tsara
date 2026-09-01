"""Tests for the parquet reader.

Parquet's distinguishing feature is that it *stores the index*, so the
normal case has no timestamp to parse at all. What still has to be got
right is normalization: real files in the target archive carry tz-aware
indexes at both microsecond and nanosecond resolution, sometimes for the
same instrument, and neither satisfies the ``RawTable`` contract untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tsara.config.manifest import CSVLoader, ParquetLoader
from tsara.ingest.base import TIME_INDEX_NAME, TsaraIngestError
from tsara.ingest.parquet_reader import read_parquet

LOADER = ParquetLoader(path_template="*.parquet")


def _loader(**kwargs: Any) -> ParquetLoader:
    kwargs.setdefault("path_template", "*.parquet")
    return ParquetLoader(**kwargs)


def _write(tmp_path: Path, frame: pd.DataFrame, name: str = "f.parquet") -> Path:
    path = tmp_path / name
    frame.to_parquet(path)
    return path


def _indexed(tz: str | None = "UTC", unit: str = "ns") -> pd.DataFrame:
    """A frame shaped like the archive's files: species on a time index."""
    index = pd.DatetimeIndex(
        pd.to_datetime(["2026-02-03 16:59:08", "2026-02-03 16:59:10"]).astype(
            f"datetime64[{unit}]"
        ),
        name="TIMESTAMP",
    )
    if tz is not None:
        index = index.tz_localize(tz)
    return pd.DataFrame({"CH4_sync": [1900.0, 1901.0], "CO2_ppm": [415.0, 415.5]}, index=index)


# ---------------------------------------------------------------------------
# The default path: the file already knows its time axis
# ---------------------------------------------------------------------------


def test_reads_the_stored_index(tmp_path: Path) -> None:
    table = read_parquet(_write(tmp_path, _indexed()), LOADER)

    assert list(table.frame.columns) == ["CH4_sync", "CO2_ppm"]
    assert table.frame["CH4_sync"].tolist() == [1900.0, 1901.0]
    assert table.frame.index[0] == pd.Timestamp("2026-02-03 16:59:08")


def test_index_is_normalized_to_naive_utc_nanoseconds(tmp_path: Path) -> None:
    index = read_parquet(_write(tmp_path, _indexed()), LOADER).frame.index
    assert index.dtype == "datetime64[ns]"
    assert pd.DatetimeIndex(index).tz is None
    assert index.name == TIME_INDEX_NAME


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_both_stored_resolutions_become_nanoseconds(tmp_path: Path, unit: str) -> None:
    """The archive carries both, sometimes for the same instrument."""
    table = read_parquet(_write(tmp_path, _indexed(unit=unit)), LOADER)
    assert table.frame.index.dtype == "datetime64[ns]"


def test_timezone_aware_index_is_converted(tmp_path: Path) -> None:
    """Unlike the text readers, parquet routinely hands back aware indexes."""
    frame = _indexed(tz="Etc/GMT-2")
    table = read_parquet(_write(tmp_path, frame), LOADER)
    assert table.frame.index[0] == pd.Timestamp("2026-02-03 14:59:08")


def test_naive_stored_index_is_treated_as_utc(tmp_path: Path) -> None:
    table = read_parquet(_write(tmp_path, _indexed(tz=None)), LOADER)
    assert table.frame.index[0] == pd.Timestamp("2026-02-03 16:59:08")


def test_column_names_are_preserved(tmp_path: Path) -> None:
    """Canonical renaming stays a manifest concern, as for every reader."""
    frame = _indexed().rename(columns={"CH4_sync": "CH4 (ppm)"})
    assert "CH4 (ppm)" in read_parquet(_write(tmp_path, frame), LOADER).frame.columns


# ---------------------------------------------------------------------------
# The declared-time path: time lives in a column
# ---------------------------------------------------------------------------


def test_time_block_reads_a_column(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"t": ["2026-02-03 16:59:08", "2026-02-03 16:59:10"], "CH4_sync": [1900.0, 1901.0]}
    )
    table = read_parquet(
        _write(tmp_path, frame),
        _loader(time={"column": "t", "format": "%Y-%m-%d %H:%M:%S"}),
    )
    assert table.frame.index[0] == pd.Timestamp("2026-02-03 16:59:08")


def test_time_block_reads_epoch_seconds(tmp_path: Path) -> None:
    frame = pd.DataFrame({"EPOCH_TIME": [1770137948.0], "CH4_sync": [1900.0]})
    table = read_parquet(
        _write(tmp_path, frame), _loader(time={"column": "EPOCH_TIME", "format": "unix"})
    )
    assert table.frame.index.dtype == "datetime64[ns]"


def test_default_range_index_is_not_promoted_to_a_column(tmp_path: Path) -> None:
    """A RangeIndex carries nothing; promoting it would add junk."""
    frame = pd.DataFrame({"EPOCH_TIME": [1770137948.0], "CH4_sync": [1900.0]})
    table = read_parquet(
        _write(tmp_path, frame), _loader(time={"column": "EPOCH_TIME", "format": "unix"})
    )
    assert list(table.frame.columns) == ["EPOCH_TIME", "CH4_sync"]


def test_a_meaningful_stored_index_is_promoted_not_discarded(tmp_path: Path) -> None:
    """The named column may BE the stored index; losing it would hide it."""
    frame = pd.DataFrame({"CH4_sync": [1900.0, 1901.0]})
    frame.index = pd.Index(["2026-02-03 16:59:08", "2026-02-03 16:59:10"], name="stamp")
    table = read_parquet(
        _write(tmp_path, frame),
        _loader(time={"column": "stamp", "format": "%Y-%m-%d %H:%M:%S"}),
    )
    assert table.frame.index[0] == pd.Timestamp("2026-02-03 16:59:08")
    assert "stamp" in table.frame.columns


def test_time_block_wins_over_a_stored_datetime_index(tmp_path: Path) -> None:
    """An explicit manifest instruction is never overridden by the file."""
    frame = _indexed()
    frame["other"] = ["2020-01-01 00:00:00", "2020-01-01 00:00:01"]
    table = read_parquet(
        _write(tmp_path, frame),
        _loader(time={"column": "other", "format": "%Y-%m-%d %H:%M:%S"}),
    )
    assert table.frame.index[0] == pd.Timestamp("2020-01-01 00:00:00")


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_non_datetime_index_without_a_time_block_is_an_error(tmp_path: Path) -> None:
    """The message has to name the fix, since the manifest can supply one."""
    frame = pd.DataFrame({"CH4_sync": [1900.0, 1901.0]})
    with pytest.raises(TsaraIngestError, match="not a DatetimeIndex.*'time:' block"):
        read_parquet(_write(tmp_path, frame), LOADER)


def test_empty_file_is_an_error(tmp_path: Path) -> None:
    frame = _indexed().iloc[:0]
    with pytest.raises(TsaraIngestError, match="no data rows"):
        read_parquet(_write(tmp_path, frame), LOADER)


def test_wrong_loader_type_is_refused(tmp_path: Path) -> None:
    bad: Any = CSVLoader.model_validate(
        {"path_template": "*.csv", "time": {"column": "t", "format": "unix"}}
    )
    with pytest.raises(TsaraIngestError, match="registered under the wrong format"):
        read_parquet(tmp_path / "f.parquet", bad)


def test_unparseable_timestamps_are_dropped_and_counted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frame = pd.DataFrame({"t": ["2026-02-03 16:59:08", "not-a-time"], "CH4_sync": [1900.0, 1901.0]})
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.parquet_reader"):
        table = read_parquet(
            _write(tmp_path, frame),
            _loader(time={"column": "t", "format": "%Y-%m-%d %H:%M:%S"}),
        )
    assert len(table.frame) == 1
    assert "Dropped 1 of 2 rows" in caplog.text


def test_no_parseable_timestamp_at_all_is_an_error(tmp_path: Path) -> None:
    frame = pd.DataFrame({"t": ["nope", "also-nope"], "CH4_sync": [1900.0, 1901.0]})
    with pytest.raises(TsaraIngestError, match="No row .* produced a valid timestamp"):
        read_parquet(
            _write(tmp_path, frame),
            _loader(time={"column": "t", "format": "%Y-%m-%d %H:%M:%S"}),
        )
