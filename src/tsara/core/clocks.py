"""The clock-correction table: one file format, read by ingestion and written by the survey.

Where this sits
---------------
A clock error is not support (``docs/METHODS.md`` §10.7). Some of it is a
constant per instrument, which ``InstrumentConfig.time_shift`` declares; the
rest changes inside a campaign, and inside a drive: a logger that steps, one
that drifts, an inlet that changes. ``Manifest.clock_corrections`` names one or
more CSV files of such corrections, each row a stretch of one instrument and
the shift that puts it on a reference clock (§12.3). This module is the file
format: what a row is, what a file may and may not say, and the shift a row
gives at a recorded time.

It lives in ``core`` because two stages meet at it and stages never import
each other: ingestion reads a table and applies it (:mod:`tsara.ingest.clocks`),
and the clock survey measures one and writes it (§12.4). One format, written
down once, is what keeps the writer and the reader from drifting apart.

The file
--------
A CSV with a header. Required columns: ``instrument``, ``start``, ``end``,
``shift``, ``reference_clock``, ``provenance``; optional: ``shift_end`` and
``reason``. Any other column is refused, so that a misspelt ``shift_end``
cannot silently turn a drift into a constant.

* ``start`` and ``end`` bound the stretch, ``[start, end)``, on the
  instrument's own **recorded** clock, which is the only clock the readings
  carry when the table is applied. Each is ISO 8601 with an explicit UTC
  offset or ``Z``; a bare local time is refused, as in an excluded-periods
  file, because a table written in local time would correct the wrong hours of
  every day and nothing downstream could tell.
* ``shift`` is a signed duration with its unit (``-10s``, ``0.5s``), added to
  every timestamp in the stretch: the sign rule of ``time_shift``, so a logger
  running 5 s fast takes ``-5s``. A bare number is refused, since ``10`` could
  mean seconds or nanoseconds.
* ``shift_end``, when given, is the shift at ``end``: the shift then runs
  linearly in recorded time from ``shift`` at ``start``, which is how a
  steadily drifting clock is written. Empty means a constant shift.
* ``reference_clock`` names the instrument whose clock the shift puts this one
  on. ``provenance`` is ``measured`` (the survey wrote the row) or
  ``declared`` (a person did). ``reason`` is free text.

What a table may not say
------------------------
Two rows of one instrument that overlap: a reading in both would have two
shifts. A row whose reference clock is its own instrument. And more than one
reference clock across the whole table: a campaign is corrected onto one
clock, and streams corrected onto two would look aligned while being apart by
whatever separates the two clocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.exceptions import TsaraError
from tsara.core.timebase import NS_PER_S, SECOND_NS, iso_utc, timestamp_epoch_ns

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = [
    "ClockCorrection",
    "TsaraClockError",
    "read_clock_corrections",
    "shift_at",
    "write_clock_corrections",
]

#: The columns a correction file must carry, and the ones it may.
_REQUIRED_COLUMNS = ("instrument", "start", "end", "shift", "reference_clock", "provenance")
_OPTIONAL_COLUMNS = ("shift_end", "reason")

#: The order a written file's columns are in.
_WRITTEN_COLUMNS = (
    "instrument",
    "start",
    "end",
    "shift",
    "shift_end",
    "reference_clock",
    "provenance",
    "reason",
)

#: Who wrote a row: the clock survey, or a person.
_PROVENANCES = ("measured", "declared")


class TsaraClockError(TsaraError):
    """Raised when a clock-correction file cannot be read or says something impossible."""


@dataclass(frozen=True)
class ClockCorrection:
    """One row of a correction table: a stretch of one instrument and its shift.

    Attributes
    ----------
    instrument : str
        The instrument the row corrects.
    start_ns, end_ns : int
        The stretch, as nanoseconds since the epoch on the instrument's
        recorded clock: start inclusive, end exclusive.
    shift_ns : int
        The shift at ``start``, added to the recorded time.
    shift_end_ns : int or None
        The shift at ``end`` for a drift, the shift running linearly between;
        None for a constant shift.
    reference_clock : str
        The instrument whose clock the shift puts this one on.
    provenance : str
        ``measured`` or ``declared``.
    reason : str
        Free text from the file.
    source : str
        The file the row came from, for messages and attributes.
    line : int
        The row's line in that file (the header is line 1).
    """

    instrument: str
    start_ns: int
    end_ns: int
    shift_ns: int
    shift_end_ns: int | None
    reference_clock: str
    provenance: str
    reason: str
    source: str
    line: int

    def describe(self) -> str:
        """One line for messages and attributes: stretch, shift, clock, provenance, reason."""
        shift = _seconds(self.shift_ns)
        if self.shift_end_ns is not None:
            shift += f" to {_seconds(self.shift_end_ns)}"
        stretch = f"{iso_utc(self.start_ns)}/{iso_utc(self.end_ns)}"
        text = f"{stretch} shift {shift} onto {self.reference_clock}"
        detail = self.provenance + (f"; {self.reason}" if self.reason else "")
        return f"{text} ({detail})"

    def where(self) -> str:
        """Return the row's place, ``file:line``, for messages."""
        return f"{self.source}:{self.line}"


@dataclass(frozen=True)
class _Cell:
    """Parses one row's cells, with the row's place in every message."""

    path: Path
    line: int

    def error(self, message: str) -> TsaraClockError:
        """Return an error naming the file and line."""
        return TsaraClockError(f"Clock-correction file '{self.path}', line {self.line}: {message}")

    def name(self, text: object, column: str) -> str:
        """Read a required name: an instrument, or the one keeping the reference clock."""
        value = str(text).strip()
        if not value:
            raise self.error(f"{column} is empty.")
        return value

    def stamp(self, text: object, column: str) -> int:
        """Read a timestamp as epoch nanoseconds, refusing one without an explicit offset."""
        value = str(text).strip()
        if not value:
            raise self.error(f"{column} is empty.")
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError) as exc:
            raise self.error(f"cannot read {column} '{value}' as a timestamp.") from exc
        if stamp.tz is None:
            raise self.error(
                f"{column} '{value}' carries no UTC offset. Write it as ISO 8601 with an "
                "offset or Z (2024-08-20T18:00:00Z), so that a table written in local time "
                "cannot correct the wrong hours."
            )
        return timestamp_epoch_ns(stamp)

    def duration(self, text: object, column: str) -> int:
        """Read a signed duration with its unit, as nanoseconds."""
        value = str(text).strip()
        if not value:
            raise self.error(f"{column} is empty.")
        if not any(character.isalpha() for character in value):
            raise self.error(
                f"{column} '{value}' has no unit. Write a duration such as '-10s' or "
                "'0.5s': a bare number could be seconds or nanoseconds."
            )
        try:
            duration = pd.Timedelta(value)
        except (ValueError, TypeError) as exc:
            raise self.error(f"cannot read {column} '{value}' as a duration.") from exc
        return int(duration.value)


def read_clock_corrections(paths: Iterable[Path | str]) -> tuple[ClockCorrection, ...]:
    """Read every correction file a manifest names, and check the table they make.

    Parameters
    ----------
    paths : iterable of pathlib.Path
        ``Manifest.clock_corrections``, resolved to absolute paths by the
        loader.

    Returns
    -------
    tuple of ClockCorrection
        Every row of every file, in file order.

    Raises
    ------
    TsaraClockError
        If a file is missing, cannot be parsed, lists no rows, lacks a
        required column or carries an unknown one, or holds a cell that
        cannot be read: a timestamp without an offset, a shift without a
        unit, an unknown provenance, an empty instrument or reference clock,
        or an end not after its start. And if the rows together say
        something impossible: two rows of one instrument overlapping, a row
        referring an instrument to itself, or more than one reference clock.
    """
    rows: list[ClockCorrection] = []
    for path in paths:
        rows.extend(_read_one(Path(path)))
    _check_table(rows)
    return tuple(rows)


def _read_one(path: Path) -> list[ClockCorrection]:
    """Read one file, refusing every shape that would correct the wrong readings."""
    if not path.is_file():
        raise TsaraClockError(f"Clock-correction file not found: '{path}'.")
    try:
        table = pd.read_csv(
            path, dtype=str, keep_default_na=False, skipinitialspace=True, encoding="utf-8"
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise TsaraClockError(f"Could not read clock-correction file '{path}': {exc}") from exc
    table.columns = pd.Index([str(column).strip() for column in table.columns])

    missing = [column for column in _REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise TsaraClockError(
            f"Clock-correction file '{path}' lacks the column(s) {missing}; it needs "
            f"{list(_REQUIRED_COLUMNS)} and may carry {list(_OPTIONAL_COLUMNS)}. "
            f"Columns present: {list(table.columns)}."
        )
    unknown = [c for c in table.columns if c not in _REQUIRED_COLUMNS + _OPTIONAL_COLUMNS]
    if unknown:
        # A misspelt `shift_end` would otherwise read as absent, and absent
        # means a constant shift: a drift would be corrected as a step.
        raise TsaraClockError(
            f"Clock-correction file '{path}' has column(s) {unknown} that mean nothing "
            f"here. The columns are {list(_REQUIRED_COLUMNS + _OPTIONAL_COLUMNS)}."
        )
    if table.empty:
        raise TsaraClockError(
            f"Clock-correction file '{path}' lists no corrections. A pointer to an empty "
            "file looks active and does nothing; remove it from clock_corrections until "
            "there is something to list."
        )
    for column in _OPTIONAL_COLUMNS:
        if column not in table.columns:
            table[column] = ""

    rows: list[ClockCorrection] = []
    # Line numbers for messages: the header is line 1, so data starts at 2.
    for line, record in enumerate(table.to_dict("records"), start=2):
        cell = _Cell(path, line)
        instrument = cell.name(record["instrument"], "instrument")
        start = cell.stamp(record["start"], "start")
        end = cell.stamp(record["end"], "end")
        if end <= start:
            raise cell.error(f"end '{record['end']}' is not after start '{record['start']}'.")
        shift_end_text = str(record["shift_end"]).strip()
        provenance = str(record["provenance"]).strip()
        if provenance not in _PROVENANCES:
            raise cell.error(
                f"provenance '{provenance}' is not one of {list(_PROVENANCES)}: "
                "'measured' for a row the survey wrote, 'declared' for one a person did."
            )
        rows.append(
            ClockCorrection(
                instrument=instrument,
                start_ns=start,
                end_ns=end,
                shift_ns=cell.duration(record["shift"], "shift"),
                shift_end_ns=cell.duration(shift_end_text, "shift_end") if shift_end_text else None,
                reference_clock=cell.name(record["reference_clock"], "reference_clock"),
                provenance=provenance,
                reason=str(record["reason"]).strip(),
                source=path.name,
                line=line,
            )
        )
    logger.info("Read %d clock correction(s) from '%s'.", len(rows), path)
    return rows


def _check_table(rows: Sequence[ClockCorrection]) -> None:
    """Refuse what the rows say together: overlaps, self-reference, two reference clocks."""
    for row in rows:
        if row.reference_clock == row.instrument:
            raise TsaraClockError(
                f"Clock correction {row.where()} puts instrument '{row.instrument}' on its "
                "own clock. A row moves an instrument onto another instrument's clock."
            )
    clocks = sorted({row.reference_clock for row in rows})
    if len(clocks) > 1:
        # A campaign is corrected onto one clock. Streams put onto two would
        # line up with their own reference and be apart from each other by
        # whatever separates the two clocks, with nothing to say so.
        examples = {
            clock: next(r.where() for r in rows if r.reference_clock == clock) for clock in clocks
        }
        raise TsaraClockError(
            f"The clock-correction table names {len(clocks)} reference clocks {clocks} "
            f"(first at {examples}). A campaign is corrected onto one clock: chain each "
            "instrument to it, as the survey does (METHODS.md §12.4)."
        )
    by_instrument: dict[str, list[ClockCorrection]] = {}
    for row in rows:
        by_instrument.setdefault(row.instrument, []).append(row)
    for instrument, own in by_instrument.items():
        ordered = sorted(own, key=lambda r: (r.start_ns, r.end_ns))
        for before, after in zip(ordered, ordered[1:], strict=False):
            if after.start_ns < before.end_ns:
                raise TsaraClockError(
                    f"Clock corrections {before.where()} and {after.where()} for instrument "
                    f"'{instrument}' overlap ({before.describe()}; {after.describe()}). A "
                    "reading in both would have two shifts."
                )


def shift_at(
    rows: Sequence[ClockCorrection], recorded_ns: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Return the shift each recorded time takes, and whether a row holds it.

    Parameters
    ----------
    rows : Sequence of ClockCorrection
        One instrument's rows, which :func:`read_clock_corrections` has
        checked do not overlap.
    recorded_ns : numpy.ndarray of int64
        Recorded times as nanoseconds since the epoch, in any order.

    Returns
    -------
    shift_ns : numpy.ndarray of int64
        The shift of the row whose ``[start, end)`` holds each time: constant,
        or linear in recorded time from ``shift`` to ``shift_end`` for a
        drift, rounded to the nanosecond. Zero where no row holds the time.
    covered : numpy.ndarray of bool
        True where a row holds the time.

    Notes
    -----
    The drift is evaluated as ``shift + (shift_end - shift) * f`` with ``f`` the
    fraction of the stretch elapsed, in floating point. Its error is far below
    a nanosecond: ``f`` is exact to about 1e-16, and a drift's change across a
    stretch is seconds, not years.
    """
    times = np.asarray(recorded_ns, dtype=np.int64)
    shift = np.zeros(times.shape, dtype=np.int64)
    covered = np.zeros(times.shape, dtype=bool)
    for row in rows:
        inside = (times >= row.start_ns) & (times < row.end_ns)
        if not inside.any():
            continue
        if row.shift_end_ns is None:
            shift[inside] = row.shift_ns
        else:
            fraction = (times[inside] - row.start_ns) / (row.end_ns - row.start_ns)
            change = row.shift_end_ns - row.shift_ns
            shift[inside] = row.shift_ns + np.rint(change * fraction).astype(np.int64)
        covered |= inside
    return shift, covered


def write_clock_corrections(rows: Sequence[ClockCorrection], path: str | Path) -> Path:
    """Write rows as a correction file that :func:`read_clock_corrections` reads back exactly.

    The rows are checked as a table first, so that a file this writes is one
    ingestion accepts. Shifts are written as decimal seconds, exact to the
    nanosecond (``-10.5s``), since a person reviews the table before it is
    used.

    Parameters
    ----------
    rows : Sequence of ClockCorrection
        The rows, written in the order given.
    path : str or pathlib.Path
        The CSV to write; its directory must exist.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    TsaraClockError
        If there are no rows (an empty file is refused when read), or the rows
        say something impossible together (:func:`read_clock_corrections`).
    """
    if not rows:
        raise TsaraClockError(
            "No clock corrections to write: a table with no rows is refused when read."
        )
    _check_table(rows)
    target = Path(path)
    pd.DataFrame(
        [
            {
                "instrument": row.instrument,
                "start": iso_utc(row.start_ns),
                "end": iso_utc(row.end_ns),
                "shift": _duration_text(row.shift_ns),
                "shift_end": "" if row.shift_end_ns is None else _duration_text(row.shift_end_ns),
                "reference_clock": row.reference_clock,
                "provenance": row.provenance,
                "reason": row.reason,
            }
            for row in rows
        ],
        columns=list(_WRITTEN_COLUMNS),
    ).to_csv(target, index=False)
    logger.info("Wrote %d clock correction(s) to '%s'.", len(rows), target)
    return target


def _duration_text(ns: int) -> str:
    """Spell nanoseconds as exact decimal seconds with their unit: ``-10.5s``."""
    whole, fraction = divmod(abs(ns), SECOND_NS)
    text = f"{'-' if ns < 0 else ''}{whole}"
    if fraction:
        text += "." + f"{fraction:09d}".rstrip("0")
    return text + "s"


def _seconds(ns: int) -> str:
    """Spell a signed duration in seconds, for messages and attributes."""
    return f"{ns / NS_PER_S:+g} s"
