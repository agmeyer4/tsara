"""Tests for declared excluded periods: the file, its checks, and what a period covers.

The file is the whole interface, so most of these are refusals: every shape
that would mask the wrong readings (a local time, a misspelt scope column, a
variable without its instrument) has to fail at load, loudly, because a
period applied wrongly looks exactly like a period applied rightly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.config.manifest import Manifest
from tsara.core.naming import RAW_TIME_START_COLUMN, RAW_TIME_STOP_COLUMN
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.periods import (
    ExcludedPeriod,
    check_excluded_periods,
    describe_periods,
    period_mask,
    periods_for,
    read_excluded_periods,
)

HEADER = "start,end,instrument,variable,reason\n"


def _write(tmp_path: Path, body: str, name: str = "periods.csv", header: str = HEADER) -> Path:
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return path


def _ns(text: str) -> int:
    return int(pd.Timestamp(text).value)


def _period(
    start: str,
    end: str,
    instrument: str | None = None,
    variable: str | None = None,
    reason: str = "",
) -> ExcludedPeriod:
    return ExcludedPeriod(_ns(start), _ns(end), instrument, variable, reason, "p.csv")


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def test_reads_every_row_in_file_order(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "2026-02-03T18:32:00Z,2026-02-03T19:20:00Z,,,calibration\n"
        "2026-02-06T18:14:00Z,2026-02-06T18:32:00Z,van,,calibration\n"
        "2026-02-12T22:52:00Z,2026-02-12T23:15:00Z,van,ch4,bad channel\n",
    )
    periods = read_excluded_periods([path])
    assert [p.start_ns for p in periods] == [
        _ns("2026-02-03T18:32:00Z"),
        _ns("2026-02-06T18:14:00Z"),
        _ns("2026-02-12T22:52:00Z"),
    ]
    assert periods[0].end_ns == _ns("2026-02-03T19:20:00Z")
    assert [p.instrument for p in periods] == [None, "van", "van"]
    assert [p.variable for p in periods] == [None, None, "ch4"]
    assert [p.reason for p in periods] == ["calibration", "calibration", "bad channel"]
    assert all(p.source == "periods.csv" for p in periods)


def test_the_two_required_columns_are_enough(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-01-01T00:00:00Z,2026-01-01T01:00:00Z\n", header="start,end\n")
    (period,) = read_excluded_periods([path])
    assert (period.instrument, period.variable, period.reason) == (None, None, "")


def test_whitespace_around_cells_is_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path, " 2026-01-01T00:00:00Z , 2026-01-01T01:00:00Z , van , ch4 , why \n")
    (period,) = read_excluded_periods([path])
    assert (period.instrument, period.variable, period.reason) == ("van", "ch4", "why")
    assert period.start_ns == _ns("2026-01-01T00:00:00Z")


def test_an_offset_other_than_utc_is_converted(tmp_path: Path) -> None:
    """Mountain Standard Time, written with its offset, lands on the UTC timeline."""
    path = _write(
        tmp_path, "2026-02-03T11:32:00-07:00,2026-02-03T12:20:00-07:00\n", header="start,end\n"
    )
    (period,) = read_excluded_periods([path])
    assert period.start_ns == _ns("2026-02-03T18:32:00Z")
    assert period.end_ns == _ns("2026-02-03T19:20:00Z")


def test_a_timestamp_without_an_offset_is_refused(tmp_path: Path) -> None:
    """The one silent failure: a file written in local time would mask the wrong hours."""
    path = _write(tmp_path, "2026-02-03T18:32:00,2026-02-03T19:20:00Z\n", header="start,end\n")
    with pytest.raises(TsaraIngestError, match="line 2.*start.*no UTC offset"):
        read_excluded_periods([path])


def test_an_unreadable_timestamp_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-02-03T18:32:00Z,yesterday\n", header="start,end\n")
    with pytest.raises(TsaraIngestError, match="line 2.*end 'yesterday'"):
        read_excluded_periods([path])


def test_an_empty_timestamp_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, ",2026-02-03T19:20:00Z\n", header="start,end\n")
    with pytest.raises(TsaraIngestError, match="line 2: start is empty"):
        read_excluded_periods([path])


def test_end_must_follow_start(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-02-03T19:20:00Z,2026-02-03T19:20:00Z\n", header="start,end\n")
    with pytest.raises(TsaraIngestError, match="line 2.*not after start"):
        read_excluded_periods([path])


def test_a_variable_needs_its_instrument(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-02-03T18:32:00Z,2026-02-03T19:20:00Z,,ch4,\n")
    with pytest.raises(TsaraIngestError, match="line 2.*'ch4' needs its instrument"):
        read_excluded_periods([path])


def test_a_missing_required_column_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-02-03T18:32:00Z,van\n", header="start,instrument\n")
    with pytest.raises(TsaraIngestError, match=r"lacks the column\(s\) \['end'\]"):
        read_excluded_periods([path])


def test_an_unknown_column_is_refused(tmp_path: Path) -> None:
    """A misspelt scope column would otherwise widen a period to every instrument."""
    path = _write(
        tmp_path,
        "2026-02-03T18:32:00Z,2026-02-03T19:20:00Z,van\n",
        header="start,end,instrment\n",
    )
    with pytest.raises(TsaraIngestError, match=r"column\(s\) \['instrment'\] that mean nothing"):
        read_excluded_periods([path])


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    with pytest.raises(TsaraIngestError, match="lists no periods"):
        read_excluded_periods([path])


def test_a_file_with_no_header_at_all_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "blank.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(TsaraIngestError, match="Could not read"):
        read_excluded_periods([path])


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TsaraIngestError, match="not found"):
        read_excluded_periods([tmp_path / "nowhere.csv"])


def test_several_files_are_read_in_order(tmp_path: Path) -> None:
    first = _write(tmp_path, "2026-02-03T18:32:00Z,2026-02-03T19:20:00Z,,,a\n", name="a.csv")
    second = _write(tmp_path, "2026-01-21T17:11:00Z,2026-01-21T17:34:00Z,,,b\n", name="b.csv")
    periods = read_excluded_periods([first, second])
    assert [p.reason for p in periods] == ["a", "b"]
    assert [p.source for p in periods] == ["a.csv", "b.csv"]


def test_reading_says_how_many(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write(
        tmp_path,
        "2026-02-03T18:32:00Z,2026-02-03T19:20:00Z\n2026-02-06T18:14:00Z,2026-02-06T18:32:00Z\n",
        header="start,end\n",
    )
    with caplog.at_level(logging.INFO, logger="tsara.ingest.periods"):
        read_excluded_periods([path])
    assert "Read 2 excluded period(s)" in caplog.text


def test_no_files_means_no_periods() -> None:
    assert read_excluded_periods([]) == ()


# ---------------------------------------------------------------------------
# Checking against the manifest
# ---------------------------------------------------------------------------


def test_an_undeclared_instrument_is_refused(stationary_manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.model_validate(stationary_manifest_dict)
    period = _period("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", instrument="aeris")
    with pytest.raises(TsaraIngestError, match="instrument 'aeris', which the manifest"):
        check_excluded_periods([period], manifest)


def test_an_undeclared_variable_is_refused(stationary_manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.model_validate(stationary_manifest_dict)
    period = _period("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "picarro", "c2h6")
    with pytest.raises(TsaraIngestError, match="variable 'c2h6', which instrument 'picarro'"):
        check_excluded_periods([period], manifest)


def test_declared_scopes_pass(stationary_manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.model_validate(stationary_manifest_dict)
    check_excluded_periods(
        [
            _period("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"),
            _period("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "picarro"),
            _period("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "picarro", "ch4"),
        ],
        manifest,
    )


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_periods_for_keeps_campaign_wide_and_this_instruments_periods() -> None:
    everyone = _period("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z")
    mine = _period("2026-01-01T02:00:00Z", "2026-01-01T03:00:00Z", "van")
    theirs = _period("2026-01-01T04:00:00Z", "2026-01-01T05:00:00Z", "site")
    assert periods_for([everyone, mine, theirs], "van") == (everyone, mine)


def test_applies_to_a_variable() -> None:
    period = _period("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "van", "ch4")
    assert period.applies_to("van", "ch4")
    assert not period.applies_to("van", "c2h6")
    assert period.applies_to("van")  # some variable of the van
    assert not period.applies_to("site", "ch4")


# ---------------------------------------------------------------------------
# What a period covers
# ---------------------------------------------------------------------------


def _frame(*, bounds: bool) -> pd.DataFrame:
    """Four 2 s cells starting on the even seconds from 17:00:00."""
    index = pd.date_range("2026-02-03 17:00:00", periods=4, freq="2s", name="time")
    frame = pd.DataFrame({"x": np.arange(4.0)}, index=index)
    if bounds:
        frame[RAW_TIME_START_COLUMN] = index
        frame[RAW_TIME_STOP_COLUMN] = index + pd.Timedelta("2s")
    return frame


def test_a_cell_overlapping_a_period_is_masked_whole() -> None:
    """[17:00:03, 17:00:05) touches the cells [02, 04) and [04, 06), and no other."""
    covered = period_mask(
        _frame(bounds=True), [_period("2026-02-03T17:00:03Z", "2026-02-03T17:00:05Z")]
    )
    assert covered.tolist() == [False, True, True, False]


def test_a_cell_ending_where_a_period_starts_is_kept() -> None:
    covered = period_mask(
        _frame(bounds=True), [_period("2026-02-03T17:00:02Z", "2026-02-03T17:00:03Z")]
    )
    assert covered.tolist() == [False, True, False, False]


def test_a_cell_starting_where_a_period_ends_is_kept() -> None:
    covered = period_mask(
        _frame(bounds=True), [_period("2026-02-03T17:00:01Z", "2026-02-03T17:00:02Z")]
    )
    assert covered.tolist() == [True, False, False, False]


def test_without_cells_a_timestamp_inside_is_masked_and_one_at_the_end_is_not() -> None:
    """Instants: 17:00:02 lies in [02, 04), 17:00:04 does not."""
    covered = period_mask(
        _frame(bounds=False), [_period("2026-02-03T17:00:02Z", "2026-02-03T17:00:04Z")]
    )
    assert covered.tolist() == [False, True, False, False]


def test_periods_cover_by_union() -> None:
    covered = period_mask(
        _frame(bounds=True),
        [
            _period("2026-02-03T16:59:00Z", "2026-02-03T17:00:01Z"),
            _period("2026-02-03T17:00:07Z", "2026-02-03T17:00:09Z"),
        ],
    )
    assert covered.tolist() == [True, False, False, True]


def test_no_periods_cover_nothing() -> None:
    assert not period_mask(_frame(bounds=True), []).any()


# ---------------------------------------------------------------------------
# Describing
# ---------------------------------------------------------------------------


def test_describe_spells_span_scope_and_reason() -> None:
    assert (
        _period(
            "2026-02-03T18:32:00Z", "2026-02-03T19:20:00Z", "van", "ch4", "calibration"
        ).describe()
        == "2026-02-03T18:32:00Z/2026-02-03T19:20:00Z van.ch4 (calibration)"
    )
    assert (
        _period("2026-02-03T18:32:00Z", "2026-02-03T19:20:00Z").describe()
        == "2026-02-03T18:32:00Z/2026-02-03T19:20:00Z"
    )
    assert (
        _period("2026-02-03T18:32:00Z", "2026-02-03T19:20:00Z", "van").describe()
        == "2026-02-03T18:32:00Z/2026-02-03T19:20:00Z van"
    )


def test_describe_periods_joins_them() -> None:
    a = _period("2026-02-03T18:32:00Z", "2026-02-03T19:20:00Z", reason="a")
    b = _period("2026-02-06T18:14:00Z", "2026-02-06T18:32:00Z", reason="b")
    assert describe_periods([a, b]) == f"{a.describe()}; {b.describe()}"
