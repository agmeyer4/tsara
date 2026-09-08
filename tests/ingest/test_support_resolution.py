"""Tests for deciding what interval of air each ingested row describes.

The provenance ladder is the subject: every answer must be labelled with
where it came from, and no rung may pass itself off as a stronger one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.config.manifest import CSVLoader, ICARTTLoader, ParquetLoader, SupportSpec, TimeParsing
from tsara.core.naming import RAW_TIME_START_COLUMN, RAW_TIME_STOP_COLUMN
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.csv_reader import read_csv
from tsara.ingest.icartt import read_icartt
from tsara.ingest.parquet_reader import read_parquet
from tsara.ingest.support import (
    CANDIDATE_COLUMNS_KEY,
    LABEL_HINT_KEY,
    attach_declared_boundaries,
    resolve_support,
    shift_and_centre,
)

SECOND = 1_000_000_000


def _frame(n: int = 5, step: str = "60s") -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=n, freq=step).as_unit("ns")
    return pd.DataFrame({"ch4": np.arange(float(n))}, index=pd.DatetimeIndex(index, name="time"))


def _epoch(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype(np.int64)


# ---------------------------------------------------------------------------
# attach_declared_boundaries
# ---------------------------------------------------------------------------


def test_a_spec_naming_no_columns_changes_nothing() -> None:
    frame = _frame()
    out = attach_declared_boundaries(
        frame,
        SupportSpec(),
        parse=lambda column: pytest.fail("should not parse anything"),
        path=Path("f.csv"),
        max_dropped_fraction=0.5,
        reader_logger=logging.getLogger("t"),
    )
    assert out is frame


def test_a_missing_boundary_column_is_refused() -> None:
    """The manifest and the archive disagree about what this file is."""
    with pytest.raises(TsaraIngestError, match="no column 'Nope'"):
        attach_declared_boundaries(
            _frame(),
            SupportSpec(stop_column="Nope"),
            parse=lambda column: pd.DatetimeIndex([]),
            path=Path("f.csv"),
            max_dropped_fraction=0.5,
            reader_logger=logging.getLogger("t"),
        )


def test_without_a_start_column_the_time_axis_is_the_start() -> None:
    frame = _frame()
    frame["Stop"] = frame.index + pd.Timedelta("60s")
    out = attach_declared_boundaries(
        frame,
        SupportSpec(stop_column="Stop"),
        parse=lambda column: pd.DatetimeIndex(frame[column]),
        path=Path("f.csv"),
        max_dropped_fraction=0.5,
        reader_logger=logging.getLogger("t"),
    )
    assert np.array_equal(_epoch(out[RAW_TIME_START_COLUMN]), _epoch(out.index))


def test_rows_whose_boundary_will_not_parse_are_dropped() -> None:
    frame = _frame(5)
    frame["Stop"] = frame.index + pd.Timedelta("60s")
    stops: list[Any] = list(pd.DatetimeIndex(frame["Stop"]))
    stops[2] = pd.NaT
    out = attach_declared_boundaries(
        frame,
        SupportSpec(stop_column="Stop"),
        parse=lambda column: pd.DatetimeIndex(stops),
        path=Path("f.csv"),
        max_dropped_fraction=0.5,
        reader_logger=logging.getLogger("t"),
    )
    assert len(out) == 4


def test_wholesale_boundary_loss_is_a_misparse_not_a_bad_day() -> None:
    frame = _frame(5)
    frame["Stop"] = frame.index
    with pytest.raises(TsaraIngestError, match="exceeds max_dropped_fraction"):
        attach_declared_boundaries(
            frame,
            SupportSpec(stop_column="Stop"),
            parse=lambda column: pd.DatetimeIndex([pd.NaT] * 5),
            path=Path("f.csv"),
            max_dropped_fraction=0.5,
            reader_logger=logging.getLogger("t"),
        )


# ---------------------------------------------------------------------------
# resolve_support: the ladder
# ---------------------------------------------------------------------------


def _with_bounds(frame: pd.DataFrame, start_offset: str, width: str) -> pd.DataFrame:
    out = frame.copy()
    out[RAW_TIME_START_COLUMN] = frame.index + pd.Timedelta(start_offset)
    out[RAW_TIME_STOP_COLUMN] = out[RAW_TIME_START_COLUMN] + pd.Timedelta(width)
    return out


@pytest.mark.parametrize(
    ("offset", "expected"),
    [("0s", "start"), ("-60s", "end"), ("-30s", "mid")],
)
def test_a_file_that_states_its_cells_answers_the_label_itself(offset: str, expected: str) -> None:
    """No special handling needed for an independent variable named Time_Mid:
    if the file gives every boundary, where the index falls IS the label."""
    frame = _with_bounds(_frame(), offset, "60s")
    _, resolved = resolve_support(
        frame, SupportSpec(), widths_ns=None, label_hint=None, path=Path("f")
    )
    assert resolved.label == expected
    assert resolved.label_source == "reported"
    assert resolved.width_source == "reported"
    assert resolved.width_ns == 60 * SECOND


def test_an_index_inside_its_cell_but_nowhere_named_is_unknown() -> None:
    frame = _with_bounds(_frame(), "-17s", "60s")
    _, resolved = resolve_support(
        frame, SupportSpec(), widths_ns=None, label_hint=None, path=Path("f")
    )
    assert resolved.label == "unknown"


def test_varying_widths_have_no_nominal_value_to_report() -> None:
    """A sampler whose fills vary has no single number, and saying so beats
    reporting a mean nobody can use."""
    frame = _frame(4)
    frame[RAW_TIME_START_COLUMN] = frame.index
    frame[RAW_TIME_STOP_COLUMN] = frame.index + pd.to_timedelta([14, 15, 16, 15], unit="s")
    _, resolved = resolve_support(
        frame, SupportSpec(), widths_ns=None, label_hint=None, path=Path("f")
    )
    assert resolved.width_ns is None
    assert resolved.width_source == "reported"


def test_a_declared_width_and_label_beat_a_measured_cadence() -> None:
    frame = _frame(5, step="60s")
    widths = np.full(5, 60 * SECOND, dtype=np.int64)
    out, resolved = resolve_support(
        frame,
        SupportSpec(label="start", width="15s", method="mean"),
        widths_ns=widths,
        label_hint="mid",
        path=Path("f"),
    )
    assert (resolved.label, resolved.label_source) == ("start", "declared")
    assert (resolved.width_ns, resolved.width_source) == (15 * SECOND, "declared")
    assert (resolved.method, resolved.method_source) == ("mean", "declared")
    span = _epoch(out[RAW_TIME_STOP_COLUMN]) - _epoch(out[RAW_TIME_START_COLUMN])
    assert np.all(span == 15 * SECOND)


def test_a_readers_hint_is_used_but_marked_inferred() -> None:
    frame = _frame(5)
    out, resolved = resolve_support(
        frame,
        SupportSpec(),
        widths_ns=np.full(5, 60 * SECOND, dtype=np.int64),
        label_hint="start",
        path=Path("f"),
    )
    assert (resolved.label, resolved.label_source) == ("start", "inferred")
    assert (resolved.width_ns, resolved.width_source) == (60 * SECOND, "inferred")
    assert np.array_equal(_epoch(out[RAW_TIME_START_COLUMN]), _epoch(out.index))


def test_saying_nothing_yields_a_centred_cell_that_admits_it() -> None:
    """The weakest rung, and the one hundreds of real files sit on."""
    frame = _frame(5)
    out, resolved = resolve_support(
        frame,
        SupportSpec(),
        widths_ns=np.full(5, 60 * SECOND, dtype=np.int64),
        label_hint=None,
        path=Path("f"),
    )
    assert (resolved.label, resolved.label_source) == ("unknown", "assumed")
    assert (resolved.method, resolved.method_source) == ("point", "assumed")
    midpoints = _epoch(out[RAW_TIME_START_COLUMN]) + 30 * SECOND
    assert np.array_equal(midpoints, _epoch(out.index))


def test_mixed_cadences_leave_no_single_nominal_width() -> None:
    frame = _frame(4)
    widths = np.array([SECOND, SECOND, 5 * SECOND, 5 * SECOND], dtype=np.int64)
    out, resolved = resolve_support(
        frame, SupportSpec(), widths_ns=widths, label_hint=None, path=Path("f")
    )
    assert resolved.width_ns is None
    span = _epoch(out[RAW_TIME_STOP_COLUMN]) - _epoch(out[RAW_TIME_START_COLUMN])
    assert list(span) == [SECOND, SECOND, 5 * SECOND, 5 * SECOND]


def test_with_nothing_to_measure_no_cells_are_invented(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One sample implies no interval; guessing one would be a fabrication."""
    frame = _frame(1)
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.support"):
        out, resolved = resolve_support(
            frame, SupportSpec(), widths_ns=None, label_hint=None, path=Path("f")
        )
    assert RAW_TIME_START_COLUMN not in out.columns
    assert resolved.width_ns is None and resolved.width_source == "assumed"
    assert "too few samples" in caplog.text


# ---------------------------------------------------------------------------
# The readers
# ---------------------------------------------------------------------------


def test_a_csv_stop_column_is_parsed_in_the_declared_timezone(tmp_path: Path) -> None:
    """A boundary is a timestamp in the same file, written the same way.

    Re-deriving the timezone separately is how a stop column ends up an hour
    from the start it belongs to on a campaign that logs in local time.
    """
    path = tmp_path / "a.csv"
    path.write_text(
        "t,stop,ch4\n"
        "2026-01-01 02:00:00,2026-01-01 02:00:15,1900\n"
        "2026-01-01 02:01:00,2026-01-01 02:01:15,1901\n",
        encoding="utf-8",
    )
    loader = CSVLoader(
        path_template="*.csv",
        time=TimeParsing(column="t", format="%Y-%m-%d %H:%M:%S", timezone="Etc/GMT-2"),
        support=SupportSpec(stop_column="stop", method="mean"),
    )
    table = read_csv(path, loader)
    assert table.frame.index[0] == pd.Timestamp("2026-01-01 00:00:00")
    assert table.frame[RAW_TIME_STOP_COLUMN].iloc[0] == pd.Timestamp("2026-01-01 00:00:15")


def test_a_parquet_stop_column_needs_no_declared_format(tmp_path: Path) -> None:
    """Parquet stores real types, which is why `time:` is optional here."""
    path = tmp_path / "a.parquet"
    index = pd.date_range("2026-01-01", periods=3, freq="10s").as_unit("ns")
    pd.DataFrame(
        {"ch4": [1.0, 2.0, 3.0], "stop": index + pd.Timedelta("4s")},
        index=pd.DatetimeIndex(index, name="time"),
    ).to_parquet(path)
    loader = ParquetLoader(
        path_template="*.parquet", support=SupportSpec(stop_column="stop", method="mean")
    )
    table = read_parquet(path, loader)
    span = table.frame[RAW_TIME_STOP_COLUMN] - table.frame[RAW_TIME_START_COLUMN]
    assert (span == pd.Timedelta("4s")).all()


def _icartt(
    tmp_path: Path,
    *,
    independent: str = "Time_Start",
    columns: tuple[str, ...] = ("CH4_ppb",),
    rows: tuple[str, ...] = ("0.0, 1900.0", "60.0, 1901.0"),
    name: str = "f.ict",
) -> Path:
    """A minimal but valid FFI-1001 file, reproducing an archive shape."""
    variables = [f"{c}, ppb, {c}" for c in columns]
    normal = ["PLATFORM: Test", ", ".join([independent, *columns])]
    body = [
        ", ".join(["1.0"] * len(columns)),
        ", ".join(["-9999.0"] * len(columns)),
        *variables,
        "1",
        "SPECIAL COMMENTS:",
        str(len(normal)),
        *normal,
    ]
    head = [
        "PLACEHOLDER, 1001",
        "Doe, Jane",
        "Test Organization",
        "Test Instrument",
        "TESTMISSION",
        "1, 1",
        "2026, 01, 01, 2026, 01, 02",
        "0",
        f"{independent}, seconds_past_midnight, Seconds from midnight UTC",
        str(len(columns)),
    ]
    lines = head + body
    # NLHEAD counts the header lines including this one; one too many
    # silently eats the first data row.
    lines[0] = f"{len(lines)}, 1001"
    path = tmp_path / name
    path.write_text("\n".join([*lines, *rows]) + "\n", encoding="utf-8")
    return path


def test_an_icartt_canister_reads_its_own_fill_times(tmp_path: Path) -> None:
    """The motivating case: widths that genuinely vary from row to row."""
    path = _icartt(
        tmp_path,
        columns=("Time_Stop", "Benzene_pptv"),
        rows=("0.0, 14.1, 32.0", "600.0, 616.2, 41.0"),
    )
    loader = ICARTTLoader(
        path_template="*.ict", support=SupportSpec(stop_column="Time_Stop", method="mean")
    )
    table = read_icartt(path, loader)
    span = table.frame[RAW_TIME_STOP_COLUMN] - table.frame[RAW_TIME_START_COLUMN]
    assert list(span) == [pd.Timedelta("14.1s"), pd.Timedelta("16.2s")]

    _, resolved = resolve_support(
        table.frame, loader.support, widths_ns=None, label_hint=None, path=path
    )
    assert (resolved.label, resolved.label_source) == ("start", "reported")
    assert resolved.width_ns is None, "fills vary, so there is no nominal width"


@pytest.mark.parametrize(
    ("independent", "expected"),
    [("Time_Start", "start"), ("Time_Mid", "mid"), ("TIMESTAMP_UTC", None)],
)
def test_the_independent_variables_name_is_read_not_assumed(
    tmp_path: Path, independent: str, expected: str | None
) -> None:
    """The specification says the independent variable is a start time. Two
    instruments in the archive publish Time_Mid, and one vendor writes
    TIMESTAMP_UTC, which is a resampling grid rather than a start at all.
    Trusting the specification would put 30 s on every cell of the minute
    suite, in the direction nobody would check.
    """
    path = _icartt(tmp_path, independent=independent)
    table = read_icartt(path, ICARTTLoader(path_template="*.ict"))
    assert table.attrs.get(LABEL_HINT_KEY) == expected


def test_boundary_columns_are_reported_as_evidence_never_guessed(
    tmp_path: Path,
) -> None:
    """A file naming Time_Stop almost certainly means it, and "almost
    certainly" applied to the wrong column produces wrong cells for a whole
    campaign. So it is surfaced for a human and not acted on."""
    path = _icartt(tmp_path, columns=("Time_Stop", "CH4_ppb"), rows=("0.0, 15.0, 1900.0",))
    table = read_icartt(path, ICARTTLoader(path_template="*.ict"))
    assert "Time_Stop" in str(table.attrs[CANDIDATE_COLUMNS_KEY])
    # Present as evidence, absent as a decision.
    assert RAW_TIME_STOP_COLUMN not in table.frame.columns


def test_a_declared_stop_narrower_than_the_cadence_is_honoured(
    tmp_path: Path,
) -> None:
    """One real instrument declares a 59 s cell on a 60 s cadence. The gap is
    the instrument's, not a rounding error, and must survive."""
    path = _icartt(
        tmp_path,
        columns=("Time_Stop", "N_ppc"),
        rows=("0.0, 59.0, 12.0", "60.0, 119.0, 13.0"),
    )
    loader = ICARTTLoader(
        path_template="*.ict", support=SupportSpec(stop_column="Time_Stop", method="mean")
    )
    table = read_icartt(path, loader)
    span = table.frame[RAW_TIME_STOP_COLUMN] - table.frame[RAW_TIME_START_COLUMN]
    assert (span == pd.Timedelta("59s")).all()


def test_both_boundary_columns_may_be_named(tmp_path: Path) -> None:
    """Needed whenever the file's own time axis is not the cell start."""
    path = tmp_path / "a.csv"
    path.write_text(
        "t,begin,finish,ch4\n"
        "2026-01-01 00:00:30,2026-01-01 00:00:00,2026-01-01 00:01:00,1900\n"
        "2026-01-01 00:01:30,2026-01-01 00:01:00,2026-01-01 00:02:00,1901\n",
        encoding="utf-8",
    )
    loader = CSVLoader(
        path_template="*.csv",
        time=TimeParsing(column="t", format="%Y-%m-%d %H:%M:%S"),
        support=SupportSpec(start_column="begin", stop_column="finish", method="mean"),
    )
    table = read_csv(path, loader)
    assert table.frame[RAW_TIME_START_COLUMN].iloc[0] == pd.Timestamp("2026-01-01 00:00:00")
    _, resolved = resolve_support(
        table.frame, loader.support, widths_ns=None, label_hint=None, path=path
    )
    assert resolved.label == "mid", "derived from where the index actually falls"


def test_a_parquet_boundary_can_use_a_declared_format(tmp_path: Path) -> None:
    """A parquet file storing its times as text still parses by the loader's
    own rules, so the axis and its boundaries cannot diverge."""
    path = tmp_path / "a.parquet"
    pd.DataFrame(
        {
            "t": ["2026-01-01 00:00:00", "2026-01-01 00:00:10"],
            "stop": ["2026-01-01 00:00:04", "2026-01-01 00:00:14"],
            "ch4": [1.0, 2.0],
        }
    ).to_parquet(path)
    loader = ParquetLoader(
        path_template="*.parquet",
        time=TimeParsing(column="t", format="%Y-%m-%d %H:%M:%S"),
        support=SupportSpec(stop_column="stop"),
    )
    table = read_parquet(path, loader)
    span = table.frame[RAW_TIME_STOP_COLUMN] - table.frame[RAW_TIME_START_COLUMN]
    assert (span == pd.Timedelta("4s")).all()


def test_a_valid_pair_of_boundaries_passes_the_contract(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("t,stop,ch4\n2026-01-01 00:00:00,2026-01-01 00:00:15,1900\n", encoding="utf-8")
    loader = CSVLoader(
        path_template="*.csv",
        time=TimeParsing(column="t", format="%Y-%m-%d %H:%M:%S"),
        support=SupportSpec(stop_column="stop"),
    )
    # read_file applies the contract check; reaching a RawTable means it passed.
    from tsara.ingest.registry import read_file

    assert RAW_TIME_STOP_COLUMN in read_file(path, loader).frame.columns


def test_files_agreeing_on_a_label_have_it_inferred(tmp_path: Path) -> None:
    """The hint is used only when every file of an instrument agrees."""
    from tsara.config.manifest import Manifest
    from tsara.ingest.campaign import _ingest_instrument

    base = tmp_path / "data"
    base.mkdir()
    _icartt(base, name="a.ict", rows=("0.0, 1900.0", "60.0, 1901.0"))
    _icartt(base, name="b.ict", rows=("120.0, 1902.0", "180.0, 1903.0"))
    manifest = Manifest.model_validate(
        {
            "name": "c",
            "base_path": str(base),
            "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
            "instruments": {
                "voc": {
                    "loader": {"format": "icartt", "path_template": "*.ict"},
                    "variables": {"ch4": {"column": "CH4_ppb", "role": "gas", "units": "ppb"}},
                }
            },
        }
    )
    ingested = _ingest_instrument(manifest, "voc", manifest.instruments["voc"])
    assert (ingested.support.label, ingested.support.label_source) == ("start", "inferred")


def test_files_disagreeing_on_a_label_fall_back_to_centred(tmp_path: Path) -> None:
    """Picking a winner would put half the files half a cell out."""
    from tsara.config.manifest import Manifest
    from tsara.ingest.campaign import _ingest_instrument

    base = tmp_path / "data"
    base.mkdir()
    _icartt(base, name="a.ict", independent="Time_Start", rows=("0.0, 1900.0", "60.0, 1901.0"))
    _icartt(base, name="b.ict", independent="Time_Mid", rows=("120.0, 1902.0", "180.0, 1903.0"))
    manifest = Manifest.model_validate(
        {
            "name": "c",
            "base_path": str(base),
            "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
            "instruments": {
                "voc": {
                    "loader": {"format": "icartt", "path_template": "*.ict"},
                    "variables": {"ch4": {"column": "CH4_ppb", "role": "gas", "units": "ppb"}},
                }
            },
        }
    )
    ingested = _ingest_instrument(manifest, "voc", manifest.instruments["voc"])
    assert (ingested.support.label, ingested.support.label_source) == ("unknown", "assumed")


def test_a_clock_correction_applies_even_without_cells() -> None:
    """A record too short to have a cadence still has a clock."""
    from tsara.ingest.support import shift_and_centre

    frame = _frame(3)
    out = shift_and_centre(frame, shift_ns=-4 * SECOND)
    assert RAW_TIME_START_COLUMN not in out.columns
    assert (out.index - frame.index == pd.Timedelta("-4s")).all()


def test_shifting_moves_a_cell_without_resizing_it() -> None:
    frame = _with_bounds(_frame(3), "0s", "60s")
    out = shift_and_centre(frame, shift_ns=-4 * SECOND)
    span = out[RAW_TIME_STOP_COLUMN] - out[RAW_TIME_START_COLUMN]
    assert (span == pd.Timedelta("60s")).all()
    # And the index landed on the midpoint of the shifted cell.
    assert (out.index - out[RAW_TIME_START_COLUMN] == pd.Timedelta("30s")).all()


def test_a_declared_zero_width_cell_is_widened_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Real files do this: one airborne spectrometer declares stop equal to
    start on 0.71% of its rows. A cell of zero duration carries no weight in
    any overlap, so those rows would sit in the stream looking like data and
    never reach a single paired regression point."""
    frame = _frame(5, step="10s")
    frame[RAW_TIME_START_COLUMN] = frame.index
    stops = list(frame.index)
    stops[2] = frame.index[2]  # zero duration
    frame[RAW_TIME_STOP_COLUMN] = [
        s + pd.Timedelta("4s") if i != 2 else s for i, s in enumerate(frame.index)
    ]
    widths = np.full(5, 10 * SECOND, dtype=np.int64)
    del stops

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.support"):
        out, resolved = resolve_support(
            frame, SupportSpec(), widths_ns=widths, label_hint=None, path=Path("f")
        )
    assert resolved.n_widened == 1
    assert "zero duration" in caplog.text
    span = _epoch(out[RAW_TIME_STOP_COLUMN]) - _epoch(out[RAW_TIME_START_COLUMN])
    # Repaired to the file's own cell length (4 s), not to the 10 s spacing.
    assert span[2] == 4 * SECOND
    assert np.all(span > 0)


def test_the_label_is_read_before_any_cell_is_repaired() -> None:
    """The ordering trap: flooring first would let a handful of degenerate
    cells turn a whole stream's label into 'unknown'. On the real record that
    is 644 rows out of 90,673 deciding the answer for all of them."""
    frame = _frame(100, step="10s")
    frame[RAW_TIME_START_COLUMN] = frame.index
    frame[RAW_TIME_STOP_COLUMN] = [
        s + pd.Timedelta("4s") if i % 20 else s for i, s in enumerate(frame.index)
    ]
    _, resolved = resolve_support(
        frame,
        SupportSpec(),
        widths_ns=np.full(100, 10 * SECOND, dtype=np.int64),
        label_hint=None,
        path=Path("f"),
    )
    assert resolved.label == "start"
    assert resolved.label_source == "reported"
    assert resolved.n_widened == 5


def test_without_a_cadence_a_zero_width_cell_is_left_alone() -> None:
    """Nothing to widen it to, and inventing a width would be a fabrication."""
    frame = _frame(3)
    frame[RAW_TIME_START_COLUMN] = frame.index
    frame[RAW_TIME_STOP_COLUMN] = frame.index
    _, resolved = resolve_support(
        frame, SupportSpec(), widths_ns=None, label_hint=None, path=Path("f")
    )
    assert resolved.n_widened == 0


def test_a_duty_cycled_sampler_is_repaired_to_its_own_fill_length() -> None:
    """The trap the first version of this repair fell into.

    A canister filling for 15 s every 10 minutes is *supposed* to have cells
    far narrower than its spacing. Repairing a degenerate row to the spacing
    would inflate it to 10 minutes and quietly claim the sampler had been
    collecting the whole time. The file's own other cells are the answer.
    """
    index = pd.DatetimeIndex(
        pd.date_range("2026-01-01", periods=6, freq="10min").as_unit("ns"), name="time"
    )
    frame = pd.DataFrame({"benzene": np.arange(6.0)}, index=index)
    frame[RAW_TIME_START_COLUMN] = index
    frame[RAW_TIME_STOP_COLUMN] = [
        t + pd.Timedelta("15s") if i != 3 else t for i, t in enumerate(index)
    ]
    spacing = np.full(6, int(pd.Timedelta("10min").value), dtype=np.int64)

    out, resolved = resolve_support(
        frame, SupportSpec(method="mean"), widths_ns=spacing, label_hint=None, path=Path("f")
    )
    widths = out[RAW_TIME_STOP_COLUMN] - out[RAW_TIME_START_COLUMN]
    assert set(widths) == {pd.Timedelta("15s")}
    assert resolved.n_widened == 1


def test_a_file_of_nothing_but_zero_widths_falls_back_to_the_cadence() -> None:
    """No other cell to learn a duration from, so the spacing is all there is."""
    frame = _frame(4, step="10s")
    frame[RAW_TIME_START_COLUMN] = frame.index
    frame[RAW_TIME_STOP_COLUMN] = frame.index
    out, resolved = resolve_support(
        frame,
        SupportSpec(),
        widths_ns=np.full(4, 10 * SECOND, dtype=np.int64),
        label_hint=None,
        path=Path("f"),
    )
    assert resolved.n_widened == 4
    span = _epoch(out[RAW_TIME_STOP_COLUMN]) - _epoch(out[RAW_TIME_START_COLUMN])
    assert np.all(span == 10 * SECOND)


def test_a_declared_width_wider_than_the_cadence_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The schema cannot catch this: a manifest is validated before any file
    is read. By resolution time both numbers are known, and the consequence
    is not subtle — every adjacent cell overlaps and every sample is counted
    into more than one of them."""
    frame = _frame(10, step="60s")
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.support"):
        out, _ = resolve_support(
            frame,
            SupportSpec(width="120s", label="start", method="mean"),
            widths_ns=np.full(10, 60 * SECOND, dtype=np.int64),
            label_hint=None,
            path=Path("f"),
        )
    assert "cells will overlap" in caplog.text
    start = _epoch(out[RAW_TIME_START_COLUMN])
    stop = _epoch(out[RAW_TIME_STOP_COLUMN])
    assert np.all(start[1:] < stop[:-1]), "the warning describes something real"


def test_a_duty_cycled_width_narrower_than_the_cadence_is_not_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Narrower than the spacing is the definition of a duty cycle, not a
    mistake, and warning about it would train the user to ignore warnings."""
    frame = _frame(10, step="60s")
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.support"):
        resolve_support(
            frame,
            SupportSpec(width="15s", method="mean"),
            widths_ns=np.full(10, 60 * SECOND, dtype=np.int64),
            label_hint=None,
            path=Path("f"),
        )
    assert "overlap" not in caplog.text


def test_no_measured_cadence_means_nothing_to_compare_against() -> None:
    frame = _frame(3)
    _, resolved = resolve_support(
        frame,
        SupportSpec(width="60s", method="mean"),
        widths_ns=None,
        label_hint=None,
        path=Path("f"),
    )
    assert resolved.width_ns == 60 * SECOND
