"""Tests for the delimited-text reader.

The cases here are drawn from the shapes real logger files actually take —
whitespace padding, split date/time fields, US-format dates, ``NA`` for
missing, headerless output — rather than from an idealized CSV. Every
"exotic" case below corresponds to a file format a gas analyzer really
writes.

Two invariants are asserted repeatedly and deliberately: the index is
tz-naive UTC at nanosecond resolution, and column names are left exactly as
the file wrote them. Everything downstream of this module depends on both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tsara.config.manifest import CSVLoader, ICARTTLoader
from tsara.ingest.base import TIME_INDEX_NAME, TsaraIngestError
from tsara.ingest.csv_reader import read_csv


def _loader(**kwargs: Any) -> CSVLoader:
    """Build a CSVLoader, defaulting the fields most tests do not vary."""
    kwargs.setdefault("path_template", "*.csv")
    kwargs.setdefault("time", {"column": "t", "format": "%Y-%m-%d %H:%M:%S"})
    return CSVLoader(**kwargs)


def _write(tmp_path: Path, text: str, name: str = "f.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The common cases
# ---------------------------------------------------------------------------


def test_simple_comma_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "t,ch4\n2026-01-01 00:00:00,1900.0\n2026-01-01 00:00:02,1905.5\n",
    )
    table = read_csv(path, _loader())

    assert list(table.frame.columns) == ["t", "ch4"]
    assert table.frame["ch4"].tolist() == [1900.0, 1905.5]
    assert table.path == path


def test_index_is_naive_utc_nanoseconds(tmp_path: Path) -> None:
    """The contract every downstream stage relies on."""
    path = _write(tmp_path, "t,ch4\n2026-01-01 00:00:00,1900.0\n")
    index = read_csv(path, _loader()).frame.index

    assert isinstance(index, pd.DatetimeIndex)
    assert index.tz is None
    assert index.dtype == "datetime64[ns]"
    assert index.name == TIME_INDEX_NAME


def test_raw_column_names_are_preserved(tmp_path: Path) -> None:
    """Canonical renaming is a manifest concern, not a reader concern.

    Real headers carry spaces, parentheses and slashes (``CH4 (ppm)``,
    ``C2/C1``); the reader must not tidy them, or the manifest's ``column:``
    references would stop matching.
    """
    path = _write(tmp_path, "Time Stamp,CH4 (ppm),C2/C1\n2026-01-01 00:00:00,2.2,-1\n")
    table = read_csv(path, _loader(time={"column": "Time Stamp"}))
    assert list(table.frame.columns) == ["Time Stamp", "CH4 (ppm)", "C2/C1"]


def test_unix_epoch_seconds(tmp_path: Path) -> None:
    path = _write(tmp_path, "EPOCH_TIME,ch4\n1767225600.0,1900.0\n1767225602.0,1901.0\n")
    table = read_csv(path, _loader(time={"column": "EPOCH_TIME", "format": "unix"}))

    assert table.frame.index[0] == pd.Timestamp("2026-01-01 00:00:00")
    assert table.frame.index.dtype == "datetime64[ns]"


def test_iso8601_sentinel(tmp_path: Path) -> None:
    path = _write(tmp_path, "t,ch4\n2026-01-01T00:00:00,1900.0\n")
    table = read_csv(path, _loader(time={"column": "t", "format": "iso8601"}))
    assert table.frame.index[0] == pd.Timestamp("2026-01-01 00:00:00")


def test_format_none_infers(tmp_path: Path) -> None:
    """Supported, but the schema documents it as the riskier choice."""
    path = _write(tmp_path, "t,ch4\n2026-01-01 00:00:00,1900.0\n")
    table = read_csv(path, _loader(time={"column": "t"}))
    assert table.frame.index[0] == pd.Timestamp("2026-01-01 00:00:00")


# ---------------------------------------------------------------------------
# Delimiters
# ---------------------------------------------------------------------------


def test_whitespace_run_delimiter(tmp_path: Path) -> None:
    """The Picarro DataLog shape: columns padded to fixed-ish widths."""
    path = _write(
        tmp_path,
        "DATE       TIME          CH4\n"
        "2026-01-01 00:00:00      1900.0\n"
        "2026-01-01 00:00:02      1901.0\n",
        name="f.dat",
    )
    table = read_csv(
        path,
        _loader(
            delimiter=r"\s+",
            time={"columns": ["DATE", "TIME"], "format": "%Y-%m-%d %H:%M:%S"},
        ),
    )
    assert list(table.frame.columns) == ["DATE", "TIME", "CH4"]
    assert table.frame.index[1] == pd.Timestamp("2026-01-01 00:00:02")


def test_tab_delimiter(tmp_path: Path) -> None:
    path = _write(tmp_path, "t\tch4\n2026-01-01 00:00:00\t1900.0\n", name="f.tsv")
    table = read_csv(path, _loader(delimiter="\t"))
    assert table.frame["ch4"].tolist() == [1900.0]


def test_multi_character_delimiter_uses_the_python_engine(tmp_path: Path) -> None:
    """Any separator that is not ',' or '\\s+' is a regex the C parser lacks."""
    path = _write(tmp_path, "t;;ch4\n2026-01-01 00:00:00;;1900.0\n")
    table = read_csv(path, _loader(delimiter=";;"))
    assert table.frame["ch4"].tolist() == [1900.0]


# ---------------------------------------------------------------------------
# Time built from several columns
# ---------------------------------------------------------------------------


def test_split_date_and_time_columns(tmp_path: Path) -> None:
    path = _write(tmp_path, "DATE,TIME,ch4\n2026-01-01,00:00:00,1900.0\n")
    table = read_csv(
        path, _loader(time={"columns": ["DATE", "TIME"], "format": "%Y-%m-%d %H:%M:%S"})
    )
    assert table.frame.index[0] == pd.Timestamp("2026-01-01 00:00:00")


def test_custom_join_separator(tmp_path: Path) -> None:
    path = _write(tmp_path, "DATE,TIME,ch4\n2026-01-01,00:00:00,1900.0\n")
    table = read_csv(
        path,
        _loader(time={"columns": ["DATE", "TIME"], "join": "T", "format": "%Y-%m-%dT%H:%M:%S"}),
    )
    assert table.frame.index[0] == pd.Timestamp("2026-01-01 00:00:00")


def test_missing_half_of_a_split_timestamp_becomes_nat(tmp_path: Path) -> None:
    """`.str.cat` propagates NA, so an incomplete timestamp is simply dropped."""
    path = _write(
        tmp_path,
        "DATE,TIME,ch4\n2026-01-01,00:00:00,1900.0\n2026-01-01,,1901.0\n",
    )
    table = read_csv(
        path, _loader(time={"columns": ["DATE", "TIME"], "format": "%Y-%m-%d %H:%M:%S"})
    )
    assert len(table.frame) == 1


def test_time_columns_remain_as_columns(tmp_path: Path) -> None:
    """The reader does not decide which columns are needed; the manifest does."""
    path = _write(tmp_path, "t,ch4\n2026-01-01 00:00:00,1900.0\n")
    assert "t" in read_csv(path, _loader()).frame.columns


# ---------------------------------------------------------------------------
# Headerless files
# ---------------------------------------------------------------------------


def test_headerless_with_full_names(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-01-01 00:00:00,1900.0\n2026-01-01 00:00:02,1901.0\n")
    table = read_csv(path, _loader(header_row=None, column_names=["t", "ch4"]))

    assert list(table.frame.columns) == ["t", "ch4"]
    assert table.frame["ch4"].tolist() == [1900.0, 1901.0]


def test_headerless_names_only_a_prefix(tmp_path: Path) -> None:
    """The 522-column Spectralite case: name what you need, ignore the rest."""
    path = _write(tmp_path, "2026-01-01 00:00:00,1900.0,7,8,9\n")
    table = read_csv(path, _loader(header_row=None, column_names=["t", "ch4"]))

    assert list(table.frame.columns) == ["t", "ch4", "column_2", "column_3", "column_4"]
    assert table.frame["column_4"].tolist() == [9]


def test_headerless_names_wider_than_the_file_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-01-01 00:00:00,1900.0\n")
    with pytest.raises(TsaraIngestError, match="describes a wider file"):
        read_csv(path, _loader(header_row=None, column_names=["t", "ch4", "co2"]))


def test_headerless_first_row_is_data_not_names(tmp_path: Path) -> None:
    """The bug a headerless misconfiguration would cause: losing row one."""
    path = _write(tmp_path, "2026-01-01 00:00:00,1900.0\n2026-01-01 00:00:02,1901.0\n")
    assert len(read_csv(path, _loader(header_row=None, column_names=["t", "ch4"])).frame) == 2


def test_headerless_with_whitespace_delimiter(tmp_path: Path) -> None:
    """Probe and read must agree on the engine, or the widths disagree."""
    path = _write(tmp_path, "2026-01-01T00:00:00   1900.0   5\n", name="f.dat")
    table = read_csv(
        path,
        _loader(delimiter=r"\s+", header_row=None, column_names=["t"], time={"column": "t"}),
    )
    assert list(table.frame.columns) == ["t", "column_1", "column_2"]


def test_headerless_with_multichar_delimiter(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-01-01 00:00:00;;1900.0;;5\n")
    table = read_csv(path, _loader(delimiter=";;", header_row=None, column_names=["t", "ch4"]))
    assert list(table.frame.columns) == ["t", "ch4", "column_2"]


# ---------------------------------------------------------------------------
# Header placement, comments, missing values
# ---------------------------------------------------------------------------


def test_header_row_below_a_preamble(tmp_path: Path) -> None:
    path = _write(tmp_path, "junk line\nanother\nt,ch4\n2026-01-01 00:00:00,1900.0\n")
    table = read_csv(path, _loader(header_row=2))
    assert list(table.frame.columns) == ["t", "ch4"]


def test_comment_lines_are_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path, "# instrument notes\nt,ch4\n2026-01-01 00:00:00,1900.0\n")
    table = read_csv(path, _loader(comment="#"))
    assert list(table.frame.columns) == ["t", "ch4"]


def test_declared_na_values(tmp_path: Path) -> None:
    """The LGR shape: 'NA' where a calibrated value does not exist yet."""
    path = _write(
        tmp_path,
        "t,ch4\n2026-01-01 00:00:00,NA\n2026-01-01 00:00:02,1901.0\n",
    )
    table = read_csv(path, _loader(na_values=["NA"]))
    assert table.frame["ch4"].isna().tolist() == [True, False]


def test_declared_na_values_extend_rather_than_replace_defaults(tmp_path: Path) -> None:
    """Declaring '-9999' must not stop an empty field being missing."""
    path = _write(
        tmp_path,
        "t,ch4\n2026-01-01 00:00:00,-9999\n2026-01-01 00:00:02,\n",
    )
    table = read_csv(path, _loader(na_values=["-9999"]))
    assert table.frame["ch4"].isna().all()


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path, "t,ch4\n2026-01-01 00:00:00,1900.0\n\n\n")
    assert len(read_csv(path, _loader()).frame) == 1


# ---------------------------------------------------------------------------
# Timezones
# ---------------------------------------------------------------------------


def test_naive_local_time_is_converted_to_utc(tmp_path: Path) -> None:
    path = _write(tmp_path, "t,ch4\n2026-01-01 00:00:00,1900.0\n")
    table = read_csv(path, _loader(time={"column": "t", "timezone": "Etc/GMT-2"}))
    # Etc/GMT-2 is UTC+2, so local midnight is 22:00 the previous day.
    assert table.frame.index[0] == pd.Timestamp("2025-12-31 22:00:00")


def test_utc_declaration_is_a_no_op(tmp_path: Path) -> None:
    path = _write(tmp_path, "t,ch4\n2026-01-01 00:00:00,1900.0\n")
    table = read_csv(path, _loader(time={"column": "t", "timezone": "utc"}))
    assert table.frame.index[0] == pd.Timestamp("2026-01-01 00:00:00")


def test_explicit_offsets_in_the_file_win(tmp_path: Path) -> None:
    """A file that states its offset must not be shifted a second time."""
    path = _write(tmp_path, "t,ch4\n2026-01-01T00:00:00+02:00,1900.0\n")
    table = read_csv(
        path,
        _loader(time={"column": "t", "format": "iso8601", "timezone": "America/Denver"}),
    )
    assert table.frame.index[0] == pd.Timestamp("2025-12-31 22:00:00")
    assert pd.DatetimeIndex(table.frame.index).tz is None


def test_nonexistent_local_time_is_an_error(tmp_path: Path) -> None:
    """02:30 on a spring-forward date never happened in Denver."""
    path = _write(tmp_path, "t,ch4\n2026-03-08 02:30:00,1900.0\n")
    with pytest.raises(TsaraIngestError, match="ambiguous or nonexistent"):
        read_csv(path, _loader(time={"column": "t", "timezone": "America/Denver"}))


def test_timezone_is_ignored_for_epoch_seconds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Epoch seconds are UTC by definition; saying otherwise is a mistake."""
    path = _write(tmp_path, "t,ch4\n1767225600.0,1900.0\n")
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.csv_reader"):
        table = read_csv(
            path, _loader(time={"column": "t", "format": "unix", "timezone": "America/Denver"})
        )
    assert table.frame.index[0] == pd.Timestamp("2026-01-01 00:00:00")
    assert "epoch seconds are UTC" in caplog.text


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_wrong_loader_type_is_refused() -> None:
    """Would mean a reader was registered under the wrong format name."""
    with pytest.raises(TsaraIngestError, match="registered under the wrong format"):
        read_csv(Path("f.ict"), ICARTTLoader(path_template="*.ict"))


def test_missing_time_column_lists_what_was_found(tmp_path: Path) -> None:
    path = _write(tmp_path, "when,ch4\n2026-01-01 00:00:00,1900.0\n")
    with pytest.raises(TsaraIngestError, match=r"no column\(s\) \['t'\].*'when'"):
        read_csv(path, _loader())


def test_file_with_only_a_header_is_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "t,ch4\n")
    with pytest.raises(TsaraIngestError, match="no data rows"):
        read_csv(path, _loader())


def test_unparseable_rows_are_dropped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write(
        tmp_path,
        "t,ch4\n2026-01-01 00:00:00,1900.0\nnot-a-time,1901.0\n2026-01-01 00:00:04,1902.0\n",
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.csv_reader"):
        table = read_csv(path, _loader())

    assert len(table.frame) == 2
    assert table.frame["ch4"].tolist() == [1900.0, 1902.0]
    assert "Dropped 1 of 3 rows" in caplog.text


def test_no_parseable_timestamp_at_all_is_an_error(tmp_path: Path) -> None:
    """Most rows failing means a wrong `format:`, not a few bad records."""
    path = _write(tmp_path, "t,ch4\n01/02/2026 00:00:00,1900.0\n")
    with pytest.raises(TsaraIngestError, match="No row .* produced a valid timestamp"):
        read_csv(path, _loader())


def test_unparseable_epoch_values_are_dropped(tmp_path: Path) -> None:
    path = _write(tmp_path, "t,ch4\n1767225600.0,1900.0\nbroken,1901.0\n")
    table = read_csv(path, _loader(time={"column": "t", "format": "unix"}))
    assert len(table.frame) == 1
