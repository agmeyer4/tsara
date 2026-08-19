"""Tests for the file-format reader registry.

The registry's job is dispatch, but its *value* is the guarantees it enforces
on the way through: no silent reader replacement, a helpful error for an
unknown format, and — the important one — the :class:`RawTable` contract
checked for every reader rather than trusted to each. These tests target the
guarantees, not the dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from tsara.config.manifest import CSVLoader
from tsara.ingest import registry
from tsara.ingest.base import TIME_INDEX_NAME, RawTable, TsaraIngestError, check_raw_table
from tsara.ingest.registry import available_readers, get_reader, read_file, register_reader

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator

    from tsara.config.manifest import LoaderConfig


@pytest.fixture
def isolated_registry() -> Iterator[None]:
    """Run a test against a private copy of the registry.

    Registration is global and permanent by design (see
    :func:`register_reader`), so a test that registers anything would leak
    into every later test in the session. Snapshot-and-restore keeps these
    tests order-independent without weakening the production guarantee.
    """
    saved = dict(registry._READERS)
    try:
        yield
    finally:
        registry._READERS.clear()
        registry._READERS.update(saved)


def _good_table(path: Path) -> RawTable:
    """Build a contract-compliant RawTable for a fake file."""
    index = pd.DatetimeIndex(
        pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01"]).astype("datetime64[ns]"),
        name=TIME_INDEX_NAME,
    )
    return RawTable(frame=pd.DataFrame({"x": [1.0, 2.0]}, index=index), path=path)


def _csv_loader(**kwargs: Any) -> CSVLoader:
    """Minimal valid CSVLoader for dispatch tests."""
    kwargs.setdefault("time", {"column": "t", "format": "unix"})
    kwargs.setdefault("path_template", "*.csv")
    return CSVLoader(**kwargs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_builtin_readers_are_registered() -> None:
    """Importing tsara.ingest must be enough to make shipped formats work."""
    assert "csv" in available_readers()


def test_available_readers_is_sorted() -> None:
    names = available_readers()
    assert list(names) == sorted(names)


def test_register_and_retrieve(isolated_registry: None) -> None:
    @register_reader("fake")
    def _reader(path: Path, loader: LoaderConfig, /) -> RawTable:
        return _good_table(path)

    assert get_reader("fake") is _reader
    assert "fake" in available_readers()


def test_decorator_returns_the_function_unchanged(isolated_registry: None) -> None:
    """A registered reader stays directly callable, so it can be unit-tested."""

    @register_reader("fake")
    def _reader(path: Path, loader: LoaderConfig, /) -> RawTable:
        return _good_table(path)

    assert _reader(Path("x"), _csv_loader()).frame.shape == (2, 1)


def test_duplicate_registration_is_refused(isolated_registry: None) -> None:
    """Silent override would make ingestion depend on import order."""

    @register_reader("fake")
    def _first(path: Path, loader: LoaderConfig, /) -> RawTable:
        return _good_table(path)

    with pytest.raises(ValueError, match="already registered"):

        @register_reader("fake")
        def _second(path: Path, loader: LoaderConfig, /) -> RawTable:
            return _good_table(path)


@pytest.mark.parametrize("name", ["", "   "])
def test_blank_reader_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_reader(name)


def test_unknown_format_lists_what_is_available() -> None:
    """The usual cause is an unimported plugin, so show the real list."""
    with pytest.raises(TsaraIngestError, match=r"No reader registered.*csv"):
        get_reader("nope")


# ---------------------------------------------------------------------------
# read_file dispatch
# ---------------------------------------------------------------------------


def test_read_file_rejects_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(TsaraIngestError, match="not an existing file"):
        read_file(tmp_path / "absent.csv", _csv_loader())


def test_read_file_rejects_a_directory(tmp_path: Path) -> None:
    """A directory is a plausible mistake when path templates are involved."""
    with pytest.raises(TsaraIngestError, match="not an existing file"):
        read_file(tmp_path, _csv_loader())


def test_read_file_accepts_a_string_path(tmp_path: Path, isolated_registry: None) -> None:
    path = tmp_path / "f.csv"
    path.write_text("t\n1\n", encoding="utf-8")

    registry._READERS["csv"] = lambda p, loader: _good_table(p)
    assert read_file(str(path), _csv_loader()).path == path


def test_reader_exceptions_are_wrapped_with_the_filename(
    tmp_path: Path, isolated_registry: None
) -> None:
    """On a 300-file archive, 'which file?' is the first question."""
    path = tmp_path / "f.csv"
    path.write_text("t\n1\n", encoding="utf-8")

    def _boom(p: Path, loader: LoaderConfig, /) -> RawTable:
        raise ValueError("kaboom")

    registry._READERS["csv"] = _boom
    with pytest.raises(TsaraIngestError, match=r"failed on .*f\.csv.*kaboom"):
        read_file(path, _csv_loader())


def test_ingest_errors_from_readers_are_not_rewrapped(
    tmp_path: Path, isolated_registry: None
) -> None:
    """A reader's specific message must not be buried under a generic one."""
    path = tmp_path / "f.csv"
    path.write_text("t\n1\n", encoding="utf-8")

    def _boom(p: Path, loader: LoaderConfig, /) -> RawTable:
        raise TsaraIngestError("very specific diagnosis")

    registry._READERS["csv"] = _boom
    with pytest.raises(TsaraIngestError, match="^very specific diagnosis$"):
        read_file(path, _csv_loader())


def test_contract_is_enforced_on_every_reader(tmp_path: Path, isolated_registry: None) -> None:
    """Third-party readers get policed exactly like TSARA's own."""
    path = tmp_path / "f.csv"
    path.write_text("t\n1\n", encoding="utf-8")

    def _sloppy(p: Path, loader: LoaderConfig, /) -> RawTable:
        return RawTable(frame=pd.DataFrame({"x": [1.0]}), path=p)  # RangeIndex

    registry._READERS["csv"] = _sloppy
    with pytest.raises(TsaraIngestError, match="requires a DatetimeIndex"):
        read_file(path, _csv_loader())


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------


def test_check_passes_a_valid_table() -> None:
    table = _good_table(Path("f.csv"))
    assert check_raw_table(table, reader_name="csv") is table


def test_check_rejects_a_non_datetime_index() -> None:
    table = RawTable(frame=pd.DataFrame({"x": [1.0]}), path=Path("f.csv"))
    with pytest.raises(TsaraIngestError, match="RangeIndex index"):
        check_raw_table(table, reader_name="csv")


def test_check_rejects_a_timezone_aware_index() -> None:
    """The bug class this check exists for: survives every stage, then bites."""
    index = pd.DatetimeIndex(pd.date_range("2026-01-01", periods=2, freq="1s", tz="UTC"))
    index.name = TIME_INDEX_NAME
    table = RawTable(frame=pd.DataFrame({"x": [1.0, 2.0]}, index=index), path=Path("f.csv"))
    with pytest.raises(TsaraIngestError, match="timezone-aware"):
        check_raw_table(table, reader_name="csv")


def test_check_rejects_non_nanosecond_resolution() -> None:
    """A datetime64[s] index changes dtype across a netCDF round trip."""
    index = pd.DatetimeIndex(
        pd.to_datetime(["2026-01-01", "2026-01-02"]).astype("datetime64[s]"),
        name=TIME_INDEX_NAME,
    )
    table = RawTable(frame=pd.DataFrame({"x": [1.0, 2.0]}, index=index), path=Path("f.csv"))
    with pytest.raises(TsaraIngestError, match=r"datetime64\[s\]"):
        check_raw_table(table, reader_name="csv")


def test_check_rejects_a_misnamed_index() -> None:
    index = pd.DatetimeIndex(
        pd.to_datetime(["2026-01-01"]).astype("datetime64[ns]"), name="timestamp"
    )
    table = RawTable(frame=pd.DataFrame({"x": [1.0]}, index=index), path=Path("f.csv"))
    with pytest.raises(TsaraIngestError, match="index name"):
        check_raw_table(table, reader_name="csv")


def test_check_rejects_duplicate_columns() -> None:
    """Duplicate names make manifest column selection ambiguous."""
    index = pd.DatetimeIndex(
        pd.to_datetime(["2026-01-01"]).astype("datetime64[ns]"), name=TIME_INDEX_NAME
    )
    frame = pd.DataFrame([[1.0, 2.0]], index=index, columns=["x", "x"])
    with pytest.raises(TsaraIngestError, match="duplicate column"):
        check_raw_table(RawTable(frame=frame, path=Path("f.csv")), reader_name="csv")


def test_raw_table_defaults_to_empty_attrs() -> None:
    """A CSV declares no self-provenance; ICARTT will."""
    assert _good_table(Path("f.csv")).attrs == {}
