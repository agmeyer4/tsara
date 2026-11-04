"""Tests for the clock-correction table: the file format, its checks, and the shift a row gives.

The file is the whole interface between the survey that writes it and the
ingestion that applies it, so most of these are refusals: every shape that
would shift the wrong readings, or shift them by the wrong amount, has to fail
at load, loudly, because a correction applied wrongly looks exactly like one
applied rightly. ``docs/METHODS.md`` §12.3.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsara.core.clocks import (
    ClockCorrection,
    TsaraClockError,
    read_clock_corrections,
    shift_at,
    write_clock_corrections,
)

HEADER = "instrument,start,end,shift,shift_end,reference_clock,provenance,reason\n"


def _write(tmp_path: Path, body: str, name: str = "clocks.csv", header: str = HEADER) -> Path:
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8")
    return path


def _ns(text: str) -> int:
    return int(pd.Timestamp(text).value)


def _row(
    start: str,
    end: str,
    shift_s: float,
    shift_end_s: float | None = None,
    instrument: str = "aeris",
) -> ClockCorrection:
    return ClockCorrection(
        instrument=instrument,
        start_ns=_ns(start),
        end_ns=_ns(end),
        shift_ns=round(shift_s * 1_000_000_000),
        shift_end_ns=None if shift_end_s is None else round(shift_end_s * 1_000_000_000),
        reference_clock="airmar",
        provenance="measured",
        reason="",
        source="clocks.csv",
        line=2,
    )


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def test_reads_every_column_of_every_row_in_file_order(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "aeris,2024-08-05T18:00:00Z,2024-08-05T22:00:00Z,-1s,,airmar,measured,CH4 against picarro\n"
        "picarro,2024-08-20T12:00:00-06:00,2024-08-20T20:00:00Z,-10s,-9.5s,airmar,declared,\n",
    )
    first, second = read_clock_corrections([path])
    assert first.instrument == "aeris"
    assert first.start_ns == _ns("2024-08-05T18:00:00Z")
    assert first.end_ns == _ns("2024-08-05T22:00:00Z")
    assert first.shift_ns == -1_000_000_000
    assert first.shift_end_ns is None
    assert first.reference_clock == "airmar"
    assert first.provenance == "measured"
    assert first.reason == "CH4 against picarro"
    assert (first.source, first.line) == ("clocks.csv", 2)
    # An offset other than Z is converted, not dropped: 12:00 at -06:00 is 18:00 UTC.
    assert second.start_ns == _ns("2024-08-20T18:00:00Z")
    assert second.shift_end_ns == -9_500_000_000
    assert (second.provenance, second.reason, second.line) == ("declared", "", 3)


def test_the_optional_columns_may_be_left_out(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "aeris,2024-08-05T18:00:00Z,2024-08-05T22:00:00Z,2s,airmar,measured\n",
        header="instrument,start,end,shift,reference_clock,provenance\n",
    )
    (row,) = read_clock_corrections([path])
    assert row.shift_end_ns is None
    assert row.reason == ""


def test_rows_from_several_files_are_read_in_order(tmp_path: Path) -> None:
    one = _write(
        tmp_path, "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,measured,\n", "a.csv"
    )
    two = _write(
        tmp_path, "aeris,2024-08-05T19:00:00Z,2024-08-05T20:00:00Z,2s,,airmar,measured,\n", "b.csv"
    )
    rows = read_clock_corrections([one, two])
    assert [(r.source, r.shift_ns) for r in rows] == [
        ("a.csv", 1_000_000_000),
        ("b.csv", 2_000_000_000),
    ]


@pytest.mark.parametrize(
    ("text", "expected_ns"),
    [
        ("0.5s", 500_000_000),
        ("-1min", -60_000_000_000),
        ("+2s", 2_000_000_000),
        ("250ms", 250_000_000),
    ],
)
def test_a_shift_is_a_signed_duration_with_its_unit(
    tmp_path: Path, text: str, expected_ns: int
) -> None:
    path = _write(
        tmp_path, f"aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,{text},,airmar,measured,\n"
    )
    assert read_clock_corrections([path])[0].shift_ns == expected_ns


def test_describe_names_the_stretch_shift_clock_and_provenance() -> None:
    row = _row("2024-08-20T18:00:00Z", "2024-08-20T20:00:00Z", -10, -9.5)
    assert row.describe() == (
        "2024-08-20T18:00:00Z/2024-08-20T20:00:00Z shift -10 s to -9.5 s (moved earlier) "
        "onto airmar (measured)"
    )
    with_reason = dataclasses.replace(row, shift_end_ns=None, reason="a step")
    assert with_reason.describe().endswith(
        "shift -10 s (moved earlier) onto airmar (measured; a step)"
    )
    assert row.where() == "clocks.csv:2"


@pytest.mark.parametrize(
    ("shift_s", "shift_end_s", "words"),
    [
        (-7.0, None, "(moved earlier)"),
        (4.0, None, "(moved later)"),
        (0.0, None, "(unchanged)"),
        (0.0, -2.0, "(moved earlier)"),
        (2.0, 0.0, "(moved later)"),
        (-1.0, 2.0, "(moved earlier, then later)"),
        (1.0, -2.0, "(moved later, then earlier)"),
    ],
)
def test_describe_says_which_way_the_readings_move(
    shift_s: float, shift_end_s: float | None, words: str
) -> None:
    """The sign is spelled in words beside the number, so that a reader of a
    warning does not have to remember which way a shift runs."""
    row = _row("2024-08-20T18:00:00Z", "2024-08-20T20:00:00Z", shift_s, shift_end_s)
    assert f"{words} onto airmar" in row.describe()


# ---------------------------------------------------------------------------
# Refusals, one row at a time
# ---------------------------------------------------------------------------


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TsaraClockError, match="not found"):
        read_clock_corrections([tmp_path / "absent.csv"])


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(TsaraClockError, match="Could not read"):
        read_clock_corrections([path])


def test_a_header_without_rows_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TsaraClockError, match="lists no corrections"):
        read_clock_corrections([_write(tmp_path, "")])


def test_a_missing_required_column_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,measured\n",
        header="instrument,start,end,shift,provenance\n",
    )
    with pytest.raises(TsaraClockError, match=r"lacks the column\(s\) \['reference_clock'\]"):
        read_clock_corrections([path])


def test_an_unknown_column_is_refused(tmp_path: Path) -> None:
    """A misspelt shift_end would read as absent, and absent means constant:
    a drift would be corrected as a step."""
    path = _write(
        tmp_path,
        "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,2s,airmar,measured\n",
        header="instrument,start,end,shift,shift_ned,reference_clock,provenance\n",
    )
    with pytest.raises(TsaraClockError, match=r"\['shift_ned'\] that mean nothing"):
        read_clock_corrections([path])


def test_a_timestamp_without_an_offset_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "aeris,2024-08-05T18:00:00,2024-08-05T19:00:00Z,1s,,airmar,measured,\n")
    with pytest.raises(
        TsaraClockError, match="line 2: start '2024-08-05T18:00:00' carries no UTC offset"
    ):
        read_clock_corrections([path])


def test_an_unreadable_timestamp_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "aeris,2024-08-05T18:00:00Z,yesterday,1s,,airmar,measured,\n")
    with pytest.raises(TsaraClockError, match="cannot read end 'yesterday' as a timestamp"):
        read_clock_corrections([path])


def test_an_empty_timestamp_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "aeris,,2024-08-05T19:00:00Z,1s,,airmar,measured,\n")
    with pytest.raises(TsaraClockError, match="start is empty"):
        read_clock_corrections([path])


def test_an_end_not_after_its_start_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "aeris,2024-08-05T19:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,measured,\n"
    )
    with pytest.raises(TsaraClockError, match="is not after start"):
        read_clock_corrections([path])


def test_a_shift_without_a_unit_is_refused(tmp_path: Path) -> None:
    """A bare 10 could be seconds or nanoseconds; pandas would read nanoseconds."""
    path = _write(
        tmp_path, "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,10,,airmar,measured,\n"
    )
    with pytest.raises(TsaraClockError, match="shift '10' has no unit"):
        read_clock_corrections([path])


def test_a_shift_end_without_a_unit_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,-3,airmar,measured,\n"
    )
    with pytest.raises(TsaraClockError, match="shift_end '-3' has no unit"):
        read_clock_corrections([path])


def test_an_unreadable_shift_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,fast,,airmar,measured,\n"
    )
    with pytest.raises(TsaraClockError, match="cannot read shift 'fast' as a duration"):
        read_clock_corrections([path])


def test_an_empty_shift_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,,,airmar,measured,\n")
    with pytest.raises(TsaraClockError, match="shift is empty"):
        read_clock_corrections([path])


def test_an_unknown_provenance_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,guessed,\n")
    with pytest.raises(TsaraClockError, match="provenance 'guessed' is not one of"):
        read_clock_corrections([path])


@pytest.mark.parametrize(
    ("body", "column"),
    [
        (",2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,measured,\n", "instrument"),
        ("aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,,measured,\n", "reference_clock"),
    ],
)
def test_an_empty_name_is_refused(tmp_path: Path, body: str, column: str) -> None:
    with pytest.raises(TsaraClockError, match=f"{column} is empty"):
        read_clock_corrections([_write(tmp_path, body)])


# ---------------------------------------------------------------------------
# Refusals of what rows say together
# ---------------------------------------------------------------------------


def test_two_overlapping_rows_of_one_instrument_are_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,measured,\n"
        "aeris,2024-08-05T18:59:59Z,2024-08-05T20:00:00Z,2s,,airmar,measured,\n",
    )
    with pytest.raises(
        TsaraClockError, match=r"clocks\.csv:2 and clocks\.csv:3 for instrument 'aeris' overlap"
    ):
        read_clock_corrections([path])


def test_overlapping_rows_in_two_files_are_refused(tmp_path: Path) -> None:
    one = _write(
        tmp_path, "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,measured,\n", "a.csv"
    )
    two = _write(
        tmp_path, "aeris,2024-08-05T18:30:00Z,2024-08-05T20:00:00Z,2s,,airmar,measured,\n", "b.csv"
    )
    with pytest.raises(TsaraClockError, match=r"a\.csv:2 and b\.csv:2"):
        read_clock_corrections([one, two])


def test_rows_that_only_touch_are_allowed(tmp_path: Path) -> None:
    """[start, end): a row ending where the next begins shares no instant."""
    path = _write(
        tmp_path,
        "aeris,2024-08-05T19:00:00Z,2024-08-05T20:00:00Z,2s,,airmar,measured,\n"
        "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,measured,\n",
    )
    assert len(read_clock_corrections([path])) == 2


def test_rows_of_different_instruments_may_overlap(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "aeris,2024-08-05T18:00:00Z,2024-08-05T20:00:00Z,1s,,airmar,measured,\n"
        "picarro,2024-08-05T18:00:00Z,2024-08-05T20:00:00Z,-2s,,airmar,measured,\n",
    )
    assert len(read_clock_corrections([path])) == 2


def test_a_row_onto_its_own_clock_is_refused(tmp_path: Path) -> None:
    path = _write(
        tmp_path, "airmar,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,declared,\n"
    )
    with pytest.raises(TsaraClockError, match="puts instrument 'airmar' on its own clock"):
        read_clock_corrections([path])


def test_two_reference_clocks_are_refused(tmp_path: Path) -> None:
    """Streams corrected onto two clocks would each line up with their own
    reference and sit apart from each other, with nothing to say so."""
    one = _write(
        tmp_path, "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,airmar,measured,\n", "a.csv"
    )
    two = _write(
        tmp_path, "miro,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,1s,,picarro,measured,\n", "b.csv"
    )
    with pytest.raises(TsaraClockError, match=r"2 reference clocks \['airmar', 'picarro'\]"):
        read_clock_corrections([one, two])


# ---------------------------------------------------------------------------
# The shift a row gives
# ---------------------------------------------------------------------------


def test_a_constant_row_shifts_every_time_it_holds_and_nothing_else() -> None:
    row = _row("2024-08-05T18:00:00Z", "2024-08-05T19:00:00Z", -3)
    times = np.array(
        [
            _ns(t)
            for t in (
                "2024-08-05T17:59:59Z",
                "2024-08-05T18:00:00Z",
                "2024-08-05T18:59:59Z",
                "2024-08-05T19:00:00Z",
            )
        ],
        dtype=np.int64,
    )
    shift, covered = shift_at([row], times)
    # [start, end): the start is held, the end is not.
    assert covered.tolist() == [False, True, True, False]
    assert shift.tolist() == [0, -3_000_000_000, -3_000_000_000, 0]


def test_a_drift_runs_linearly_in_recorded_time() -> None:
    """Pencil numbers: -10 s at the start, -8 s at the end of a 100 s stretch,
    so -9.5 s a quarter of the way and -9 s halfway."""
    row = _row("2024-08-05T18:00:00Z", "2024-08-05T18:01:40Z", -10, -8)
    start = _ns("2024-08-05T18:00:00Z")
    second = 1_000_000_000
    times = np.array(
        [start, start + 25 * second, start + 50 * second, start + 100 * second - 1], dtype=np.int64
    )
    shift, covered = shift_at([row], times)
    assert covered.all()
    assert shift[:3].tolist() == [-10 * second, -9_500_000_000, -9 * second]
    # One nanosecond before the end, the shift is within a nanosecond of shift_end.
    assert abs(int(shift[3]) - (-8 * second)) <= 1


def test_a_drift_matches_exact_rational_arithmetic() -> None:
    """An independent reference: the same definition in exact fractions,
    rounded half to even as numpy does, over awkward times and a drift that
    does not divide evenly."""
    row = _row("2024-08-05T18:00:00Z", "2024-08-05T19:13:17.123456789Z", 3.3, -7.123456789)
    rng = np.random.default_rng(5)
    times = rng.integers(row.start_ns, row.end_ns, size=2000, dtype=np.int64)
    shift, covered = shift_at([row], times)
    assert covered.all()
    assert row.shift_end_ns is not None
    for t, got in zip(times.tolist(), shift.tolist(), strict=True):
        fraction = Fraction(t - row.start_ns, row.end_ns - row.start_ns)
        want = row.shift_ns + round((row.shift_end_ns - row.shift_ns) * fraction)
        assert abs(got - want) <= 1


def test_each_time_takes_the_row_that_holds_it_in_any_order() -> None:
    early = _row("2024-08-05T18:00:00Z", "2024-08-05T19:00:00Z", 1)
    late = _row("2024-08-05T19:30:00Z", "2024-08-05T20:00:00Z", 4)
    times = np.array(
        [_ns(t) for t in ("2024-08-05T19:45:00Z", "2024-08-05T18:30:00Z", "2024-08-05T19:15:00Z")],
        dtype=np.int64,
    )
    shift, covered = shift_at([early, late], times)
    assert shift.tolist() == [4_000_000_000, 1_000_000_000, 0]
    assert covered.tolist() == [True, True, False]


def test_no_rows_shift_nothing() -> None:
    times = np.array([_ns("2024-08-05T18:00:00Z")], dtype=np.int64)
    shift, covered = shift_at([], times)
    assert shift.tolist() == [0]
    assert covered.tolist() == [False]


# ---------------------------------------------------------------------------
# Writing a table
# ---------------------------------------------------------------------------


def test_a_written_table_reads_back_exactly(tmp_path: Path) -> None:
    rows = [
        _row("2024-08-05T18:00:00Z", "2024-08-05T19:00:00Z", -10.5),
        _row("2024-08-05T19:00:00Z", "2024-08-05T20:00:00.5Z", -0.000000001, 3.25),
        dataclasses.replace(
            _row("2024-08-05T18:00:00Z", "2024-08-05T20:00:00Z", 2, instrument="miro"),
            reason='survey: 4 segments, "tier 1", lags +2 to +2 s',
        ),
    ]
    path = write_clock_corrections(rows, tmp_path / "written.csv")
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "instrument,start,end,shift,shift_end,reference_clock,provenance,reason"
    assert text[1] == "aeris,2024-08-05T18:00:00Z,2024-08-05T19:00:00Z,-10.5s,,airmar,measured,"
    assert ",-0.000000001s,3.25s," in text[2]
    back = read_clock_corrections([path])
    fields = (
        "instrument",
        "start_ns",
        "end_ns",
        "shift_ns",
        "shift_end_ns",
        "reference_clock",
        "provenance",
        "reason",
    )
    assert [[getattr(r, f) for f in fields] for r in back] == [
        [getattr(r, f) for f in fields] for r in rows
    ]
    assert [(r.source, r.line) for r in back] == [
        ("written.csv", 2),
        ("written.csv", 3),
        ("written.csv", 4),
    ]


def test_an_empty_table_is_not_written(tmp_path: Path) -> None:
    with pytest.raises(TsaraClockError, match="No clock corrections to write"):
        write_clock_corrections([], tmp_path / "empty.csv")
    assert not (tmp_path / "empty.csv").exists()


def test_a_table_that_would_be_refused_is_not_written(tmp_path: Path) -> None:
    rows = [
        _row("2024-08-05T18:00:00Z", "2024-08-05T19:00:00Z", 1),
        _row("2024-08-05T18:30:00Z", "2024-08-05T20:00:00Z", 2),
    ]
    with pytest.raises(TsaraClockError, match="overlap"):
        write_clock_corrections(rows, tmp_path / "bad.csv")
    assert not (tmp_path / "bad.csv").exists()
