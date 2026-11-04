"""Tests for applying clock corrections at ingestion (``tsara.ingest.clocks``).

Named ``test_ingest_clocks`` because ``tests/core/test_clocks.py`` tests the
table's format, and test files need distinct basenames.

What is pinned here: a reading is shifted by the row holding its *recorded*
time; its cell moves with it and is never resized; a reading no row holds is
left alone, counted and named; a correction that puts readings before earlier
ones says so; and the correction runs before the cadence is measured, so a
drifting clock does not hand its drift to the cells. ``docs/METHODS.md`` §12.3.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tsara.config.loader import load_manifest
from tsara.config.manifest import Manifest
from tsara.core.clocks import ClockCorrection
from tsara.core.naming import RAW_TIME_START_COLUMN, RAW_TIME_STOP_COLUMN, TIME_BOUNDS_VAR
from tsara.ingest import ingest_campaign
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.clocks import ClockRecord, check_clock_corrections, corrections_for

SECOND = 1_000_000_000
T0 = "2024-08-05T18:00:00"


def _ns(text: str) -> int:
    return int(pd.Timestamp(text).value)


def _row(
    start_s: float,
    end_s: float,
    shift_s: float,
    shift_end_s: float | None = None,
    *,
    instrument: str = "picarro",
    reference: str = "gps",
) -> ClockCorrection:
    """A row on seconds after T0."""
    base = _ns(T0)
    return ClockCorrection(
        instrument=instrument,
        start_ns=base + round(start_s * SECOND),
        end_ns=base + round(end_s * SECOND),
        shift_ns=round(shift_s * SECOND),
        shift_end_ns=None if shift_end_s is None else round(shift_end_s * SECOND),
        reference_clock=reference,
        provenance="declared",
        reason="",
        source="clocks.csv",
        line=2,
    )


def _frame(seconds: list[float], *, bounds: bool = False) -> pd.DataFrame:
    """One file as a reader returns it: recorded times after T0, a value column."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(T0) + pd.Timedelta(seconds=s) for s in seconds], name="time"
    ).as_unit("ns")
    frame = pd.DataFrame({"CH4": np.arange(len(seconds), dtype=float)}, index=index)
    if bounds:
        frame[RAW_TIME_START_COLUMN] = np.asarray(index, dtype="datetime64[ns]")
        frame[RAW_TIME_STOP_COLUMN] = np.asarray(
            index + pd.Timedelta(seconds=2), dtype="datetime64[ns]"
        )
    return frame


def _offsets_s(index: pd.Index) -> list[float]:
    return [(t - pd.Timestamp(T0)) / pd.Timedelta(seconds=1) for t in index]


# ---------------------------------------------------------------------------
# Checked against the manifest
# ---------------------------------------------------------------------------


def test_an_undeclared_instrument_is_refused(mobile_manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.model_validate(mobile_manifest_dict)
    with pytest.raises(TsaraIngestError, match="names instrument 'aeris', which the manifest"):
        check_clock_corrections([_row(0, 10, 1, instrument="aeris")], manifest)


def test_an_undeclared_reference_clock_is_refused(mobile_manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.model_validate(mobile_manifest_dict)
    with pytest.raises(
        TsaraIngestError, match="names reference clock 'airmar', which the manifest"
    ):
        check_clock_corrections([_row(0, 10, 1, reference="airmar")], manifest)


def test_declared_names_pass(mobile_manifest_dict: dict[str, Any]) -> None:
    check_clock_corrections([_row(0, 10, 1)], Manifest.model_validate(mobile_manifest_dict))


def test_corrections_for_keeps_one_instruments_rows_in_order() -> None:
    rows = [_row(0, 10, 1), _row(0, 10, 2, instrument="miro"), _row(20, 30, 3)]
    assert corrections_for(rows, "picarro") == (rows[0], rows[2])
    assert corrections_for(rows, "aeris") == ()


# ---------------------------------------------------------------------------
# What one file's correction moves
# ---------------------------------------------------------------------------


def test_each_reading_moves_by_the_row_holding_its_recorded_time() -> None:
    record = ClockRecord((_row(0, 3, -10), _row(5, 8, 4)))
    out = record.correct(_frame([0, 1, 2, 3, 4, 5, 6, 7, 8]))
    # 0-2 s in the first row (-10 s), 3-4 s in none, 5-7 s in the second (+4 s), 8 s in none.
    assert _offsets_s(out.index) == [-10, -9, -8, 3, 4, 9, 10, 11, 8]
    assert out["CH4"].tolist() == list(range(9))


def test_a_cell_moves_with_its_reading_and_is_never_resized() -> None:
    """The boundary columns take the shift at the reading's recorded time, the
    same as the timestamp, even across a drift."""
    record = ClockRecord((_row(0, 100, -10, -8),))
    out = record.correct(_frame([0, 50], bounds=True))
    start = pd.DatetimeIndex(out[RAW_TIME_START_COLUMN])
    stop = pd.DatetimeIndex(out[RAW_TIME_STOP_COLUMN])
    assert _offsets_s(out.index) == [-10, 41]
    assert _offsets_s(start) == [-10, 41]
    assert list((stop - start) / pd.Timedelta(seconds=1)) == [2, 2]


def test_the_readers_frame_is_not_changed() -> None:
    frame = _frame([0, 1, 2], bounds=True)
    before = frame.copy()
    ClockRecord((_row(0, 10, 5),)).correct(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_an_instrument_without_rows_is_left_exactly_alone(caplog: pytest.LogCaptureFixture) -> None:
    frame = _frame([0, 1, 2])
    record = ClockRecord(())
    assert record.correct(frame) is frame
    with caplog.at_level(logging.INFO, logger="tsara.ingest.clocks"):
        record.report("picarro")
    assert caplog.records == []
    assert record.n_readings == 0
    assert np.isnan(record.uncorrected_share)


def test_counts_accumulate_over_files() -> None:
    record = ClockRecord((_row(0, 3, 1),))
    record.correct(_frame([0, 1, 2, 3]))
    record.correct(_frame([10, 11]))
    assert (record.n_readings, record.n_uncorrected) == (6, 3)
    assert record.uncorrected_share == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# What it says
# ---------------------------------------------------------------------------


def test_uncorrected_stretches_are_named(caplog: pytest.LogCaptureFixture) -> None:
    record = ClockRecord((_row(0, 3, 1), _row(5, 7, 1)))
    record.correct(_frame([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.clocks"):
        record.report("picarro")
    (message,) = [r.getMessage() for r in caplog.records]
    assert "5 of 10 readings (50.0%) lie in no clock-correction row" in message
    assert "2024-08-05T18:00:03Z/2024-08-05T18:00:04Z (2 readings)" in message
    assert "2024-08-05T18:00:07Z/2024-08-05T18:00:09Z (3 readings)" in message


def test_uncorrected_stretches_are_found_in_recorded_order_across_files(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file arriving out of time order must not split or merge stretches."""
    record = ClockRecord((_row(2, 4, 1),))
    record.correct(_frame([4, 5]))
    record.correct(_frame([0, 1, 2, 3]))
    assert record._uncorrected_stretches() == [
        (_ns(T0), _ns(T0) + SECOND, 2),
        (_ns(T0) + 4 * SECOND, _ns(T0) + 5 * SECOND, 2),
    ]


def test_a_long_list_of_stretches_is_summarized(caplog: pytest.LogCaptureFixture) -> None:
    # Ten one-reading holes between eleven rows.
    rows = tuple(_row(2 * k, 2 * k + 1, 1) for k in range(11))
    record = ClockRecord(rows)
    record.correct(_frame([float(s) for s in range(21)]))
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.clocks"):
        record.report("picarro")
    (message,) = [r.getMessage() for r in caplog.records]
    assert message.count("(1 reading)") == 8
    assert message.endswith("; and 2 more stretches.")


def test_readings_put_before_earlier_ones_are_counted(caplog: pytest.LogCaptureFixture) -> None:
    """A shift falling by 3 s between rows puts the readings recorded at 5 and
    6 s at 2 and 3 s, before the one recorded at 4 s; the one at 7 s lands on
    4 s exactly, which is not before it."""
    record = ClockRecord((_row(0, 5, 0), _row(5, 10, -3)))
    out = record.correct(_frame([float(s) for s in range(10)]))
    assert _offsets_s(out.index) == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.clocks"):
        record.report("picarro")
    (message,) = [r.getMessage() for r in caplog.records]
    assert "puts 2 reading(s) before an earlier reading of their own file" in message


def test_a_row_holding_no_reading_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    record = ClockRecord((_row(0, 10, 1), _row(100, 200, 2)))
    record.correct(_frame([0, 1, 2]))
    with caplog.at_level(logging.INFO, logger="tsara.ingest.clocks"):
        record.report("picarro")
    (message,) = [r.getMessage() for r in caplog.records]
    assert "2024-08-05T18:01:40Z/2024-08-05T18:03:20Z shift +2 s (moved later) onto gps" in message
    assert "holds none of its readings" in message


# ---------------------------------------------------------------------------
# Through ingestion
# ---------------------------------------------------------------------------


def _archive(tmp_path: Path, mobile_manifest_dict: dict[str, Any], corrections: str) -> Path:
    """A one-file Picarro whose clock reads 1.000 s per reading, plus a GPS, and a table."""
    raw = tmp_path / "raw"
    raw.mkdir()
    seconds = np.arange(0, 1000, 1.0)
    pd.DataFrame(
        {
            "EPOCH_TIME": pd.Timestamp(T0, tz="UTC").timestamp() + seconds,
            "CH4_dry": 2.0 + 0.001 * np.sin(seconds),
            "CO2_dry": 420.0,
        }
    ).to_csv(raw / "picarro.csv", index=False)
    gps_times = pd.Timestamp(T0) + pd.to_timedelta(seconds[::10], unit="s")
    pd.DataFrame(
        {
            "timestamp": gps_times.strftime("%Y-%m-%d %H:%M:%S"),
            "lat": 40.0,
            "lon": -111.0,
            "wdir": 180.0,
        }
    ).to_csv(raw / "gps.csv", index=False)
    spec = copy.deepcopy(mobile_manifest_dict)
    spec["base_path"] = "raw"
    spec["instruments"]["picarro"]["loader"]["path_template"] = "picarro.csv"
    spec["instruments"]["picarro"]["loader"]["support"] = {"label": "start"}
    spec["instruments"]["gps"]["loader"]["path_template"] = "gps.csv"
    spec["clock_corrections"] = "clocks.csv"
    (tmp_path / "clocks.csv").write_text(
        "instrument,start,end,shift,shift_end,reference_clock,provenance,reason\n" + corrections,
        encoding="utf-8",
    )
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def test_the_clock_is_corrected_before_the_cadence_is_measured(
    tmp_path: Path, mobile_manifest_dict: dict[str, Any]
) -> None:
    """A clock losing 7 s in 1000 s writes 1.000 s steps for readings 1.007 s
    apart. Corrected first, the cadence measured is the true one and every
    cell is 1.007 s wide; corrected after, every cell would carry the drift."""
    manifest = load_manifest(
        _archive(
            tmp_path,
            mobile_manifest_dict,
            "picarro,2024-08-05T18:00:00Z,2024-08-05T18:16:40Z,0s,7s,gps,declared,drift\n",
        )
    )
    stream = ingest_campaign(manifest)["picarro"]
    widths = np.diff(np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]"), axis=1)
    assert np.all(widths == np.timedelta64(1_007_000, "us"))
    times = np.asarray(stream["time"].values, dtype="datetime64[ns]")
    # Start-labelled, so the midpoint is half a cell after the corrected start:
    # the last reading, recorded at 999 s, is corrected to 999 + 6.993 s.
    expected_last = np.datetime64(f"{T0}") + np.timedelta64(
        999_000_000_000 + 6_993_000_000 + 503_500_000, "ns"
    )
    assert times[-1] == expected_last


def test_a_corrected_stream_records_its_rows_and_the_rest_records_nothing(
    tmp_path: Path, mobile_manifest_dict: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    manifest = load_manifest(
        _archive(
            tmp_path,
            mobile_manifest_dict,
            "picarro,2024-08-05T18:00:00Z,2024-08-05T18:15:00Z,-3s,,gps,measured,CH4 survey\n",
        )
    )
    with caplog.at_level(logging.INFO, logger="tsara.ingest"):
        streams = ingest_campaign(manifest)
    attrs = streams["picarro"].attrs
    assert attrs["tsara_clock_corrections"] == (
        "2024-08-05T18:00:00Z/2024-08-05T18:15:00Z shift -3 s (moved earlier) onto gps "
        "(measured; CH4 survey)"
    )
    assert attrs["tsara_clock_corrections_files"] == "clocks.csv"
    assert attrs["tsara_reference_clock"] == "gps"
    # 100 of the 1000 readings (900-999 s) are past the row's end.
    assert attrs["tsara_clock_uncorrected_share"] == pytest.approx(0.1)
    assert not any("clock" in key for key in streams["gps"].attrs)
    messages = [r.getMessage() for r in caplog.records]
    assert any("'gps': no clock correction declared" in m for m in messages)
    assert any("100 of 1000 readings (10.0%) lie in no clock-correction row" in m for m in messages)


def test_a_table_naming_an_undeclared_instrument_fails_before_any_file_is_read(
    tmp_path: Path, mobile_manifest_dict: dict[str, Any]
) -> None:
    manifest = load_manifest(
        _archive(
            tmp_path,
            mobile_manifest_dict,
            "aeris,2024-08-05T18:00:00Z,2024-08-05T18:15:00Z,-3s,,gps,measured,\n",
        )
    )
    (tmp_path / "raw" / "picarro.csv").unlink()
    with pytest.raises(TsaraIngestError, match="names instrument 'aeris'"):
        ingest_campaign(manifest)
