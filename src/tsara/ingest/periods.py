"""Periods a manifest declares to leave out: calibrations and known-bad stretches.

Where this sits
---------------
A campaign knows things about its data that no file states: when an
analyzer breathed a standard, when an inlet was off, when a logger wrote
nonsense for an hour. ``Manifest.exclude_periods`` points at one or more
files listing such periods; this module reads them, checks them against the
manifest, and says which readings they cover. The masking itself runs with
the other QA/QC rules (:mod:`tsara.ingest.qaqc`), so a period is counted and
reported exactly as a ``range`` or ``flag`` rule is (``docs/METHODS.md``
§9.4).

Why a file rather than rules in the manifest
--------------------------------------------
Conventions are data, not code (CLAUDE.md §5), and a list of intervals is
the shape a processing pipeline already produces: an archive's own ETL can
write it, a person can edit it, and one file serves every instrument of a
campaign. The 2026 van calibrations reached three analyzers at once, and one
row covers them all.

The file
--------
A CSV with a header. ``start`` and ``end`` are required: ISO 8601
timestamps carrying an explicit UTC offset or ``Z``. A timestamp with no
offset is refused rather than read as UTC, because a file written in local
time is the one silent failure this feature can have: it would mask the
wrong hours of every day and nothing downstream could tell. ``instrument``,
``variable`` and ``reason`` are optional columns and may be empty: an empty
instrument means every instrument, an empty variable every variable of the
instrument named, and a variable without its instrument is refused, since a
variable's name is unique only within its instrument (METHODS §1.6). Any
other column is refused, so that a misspelt ``instrment`` cannot silently
widen a period to every instrument.

What is masked
--------------
A reading is masked when its cell overlaps the period, ``[start, end)``
against ``[cell start, cell stop)``: a cell ending exactly where a period
starts is kept, and a cell straddling a period's edge is masked whole, since
part of the air it describes was bad. A stream without cells (a record too
short to have a measurable cadence) is masked by its timestamps instead.
Masked, never deleted, for the reason every QA/QC rule gives: the rows are
the instrument's clock, and a hole and a removed row are different facts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.naming import RAW_TIME_START_COLUMN, RAW_TIME_STOP_COLUMN
from tsara.core.timebase import epoch_ns, timestamp_epoch_ns
from tsara.ingest.base import TsaraIngestError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from tsara.config.manifest import Manifest

logger = logging.getLogger(__name__)

__all__ = [
    "ExcludedPeriod",
    "OPTIONAL_COLUMNS",
    "PERIOD_RULE_KIND",
    "REQUIRED_COLUMNS",
    "check_excluded_periods",
    "describe_periods",
    "period_mask",
    "periods_for",
    "read_excluded_periods",
]

#: The ``kind`` a period reports under in a variable's QA/QC counts, beside
#: ``range`` and ``flag``.
PERIOD_RULE_KIND = "period"

#: The columns a periods file must carry, and the ones it may.
REQUIRED_COLUMNS = ("start", "end")
OPTIONAL_COLUMNS = ("instrument", "variable", "reason")


@dataclass(frozen=True)
class ExcludedPeriod:
    """One declared period, with the scope it applies to.

    Attributes
    ----------
    start_ns, end_ns : int
        The period as nanoseconds since the epoch: start inclusive, end
        exclusive.
    instrument : str or None
        The instrument it is limited to; None for every instrument.
    variable : str or None
        The variable it is limited to; None for every variable of the
        instrument.
    reason : str
        Free text from the file, carried into the stream's attributes.
    source : str
        The file it came from, for messages and attributes.
    """

    start_ns: int
    end_ns: int
    instrument: str | None
    variable: str | None
    reason: str
    source: str

    def applies_to(self, instrument: str, variable: str | None = None) -> bool:
        """Whether this period covers ``variable`` of ``instrument``.

        With ``variable`` None the question is whether it covers any
        variable of the instrument.
        """
        if self.instrument is not None and self.instrument != instrument:
            return False
        return variable is None or self.variable is None or self.variable == variable

    def describe(self) -> str:
        """One line for messages and attributes: span, scope and reason."""
        text = f"{_iso(self.start_ns)}/{_iso(self.end_ns)}"
        if self.instrument is not None:
            text += f" {self.instrument}"
            if self.variable is not None:
                text += f".{self.variable}"
        if self.reason:
            text += f" ({self.reason})"
        return text


def read_excluded_periods(paths: Iterable[Path | str]) -> tuple[ExcludedPeriod, ...]:
    """Read every periods file a manifest names.

    Parameters
    ----------
    paths : iterable of pathlib.Path
        ``Manifest.exclude_periods``, resolved to absolute paths by the
        loader.

    Returns
    -------
    tuple of ExcludedPeriod
        Every row of every file, in file order.

    Raises
    ------
    TsaraIngestError
        If a file is missing, cannot be parsed, lists no periods, lacks a
        required column or carries an unknown one, holds a timestamp that
        cannot be read or carries no offset, names a variable without its
        instrument, or holds a period whose end is not after its start.
    """
    periods: list[ExcludedPeriod] = []
    for path in paths:
        periods.extend(_read_one(Path(path)))
    return tuple(periods)


def _read_one(path: Path) -> list[ExcludedPeriod]:
    """Read one file, refusing every shape that would mask the wrong thing."""
    if not path.is_file():
        raise TsaraIngestError(f"Excluded-periods file not found: '{path}'.")
    try:
        table = pd.read_csv(
            path, dtype=str, keep_default_na=False, skipinitialspace=True, encoding="utf-8"
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise TsaraIngestError(f"Could not read excluded-periods file '{path}': {exc}") from exc
    table.columns = pd.Index([str(column).strip() for column in table.columns])

    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise TsaraIngestError(
            f"Excluded-periods file '{path}' lacks the column(s) {missing}; it needs "
            f"{list(REQUIRED_COLUMNS)} and may carry {list(OPTIONAL_COLUMNS)}. "
            f"Columns present: {list(table.columns)}."
        )
    unknown = [c for c in table.columns if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if unknown:
        # A misspelt scope column would otherwise read as absent, and an
        # absent scope means "every instrument": the one way a typo could
        # widen a period rather than narrow it.
        raise TsaraIngestError(
            f"Excluded-periods file '{path}' has column(s) {unknown} that mean nothing "
            f"here. The columns are {list(REQUIRED_COLUMNS + OPTIONAL_COLUMNS)}."
        )
    if table.empty:
        raise TsaraIngestError(
            f"Excluded-periods file '{path}' lists no periods. A pointer to an empty "
            "file looks active and does nothing; remove it from exclude_periods until "
            "there is something to list."
        )
    for column in OPTIONAL_COLUMNS:
        if column not in table.columns:
            table[column] = ""

    periods: list[ExcludedPeriod] = []
    # Line numbers for messages: the header is line 1, so data starts at 2.
    for line, row in enumerate(table.to_dict("records"), start=2):
        start = _parse_stamp(str(row["start"]), path=path, line=line, column="start")
        end = _parse_stamp(str(row["end"]), path=path, line=line, column="end")
        if end <= start:
            raise TsaraIngestError(
                f"Excluded-periods file '{path}', line {line}: end '{row['end']}' is "
                f"not after start '{row['start']}'."
            )
        instrument = str(row["instrument"]).strip() or None
        variable = str(row["variable"]).strip() or None
        if variable is not None and instrument is None:
            raise TsaraIngestError(
                f"Excluded-periods file '{path}', line {line}: variable '{variable}' "
                "needs its instrument, since a variable's name is unique only within "
                "its instrument."
            )
        periods.append(
            ExcludedPeriod(
                start_ns=start,
                end_ns=end,
                instrument=instrument,
                variable=variable,
                reason=str(row["reason"]).strip(),
                source=path.name,
            )
        )
    logger.info("Read %d excluded period(s) from '%s'.", len(periods), path)
    return periods


def _parse_stamp(text: str, *, path: Path, line: int, column: str) -> int:
    """Read one timestamp cell as epoch nanoseconds, refusing one without an offset."""
    try:
        stamp = pd.Timestamp(text.strip())
    except (ValueError, TypeError) as exc:
        raise TsaraIngestError(
            f"Excluded-periods file '{path}', line {line}: cannot read {column} "
            f"'{text}' as a timestamp."
        ) from exc
    if pd.isna(stamp):
        raise TsaraIngestError(f"Excluded-periods file '{path}', line {line}: {column} is empty.")
    if stamp.tz is None:
        raise TsaraIngestError(
            f"Excluded-periods file '{path}', line {line}: {column} '{text}' carries no "
            "UTC offset. Write it as ISO 8601 with an offset or Z (2026-02-03T18:32:00Z), "
            "so that a file written in local time cannot mask the wrong hours."
        )
    return timestamp_epoch_ns(stamp)


def check_excluded_periods(periods: Sequence[ExcludedPeriod], manifest: Manifest) -> None:
    """Refuse a period naming an instrument or variable the manifest does not declare.

    Run before any file is read, so that a typo fails fast rather than
    after a thousand-file crawl.

    Raises
    ------
    TsaraIngestError
        If a period names an instrument the manifest lacks, or a variable
        its instrument lacks.
    """
    for period in periods:
        if period.instrument is None:
            continue
        instrument = manifest.instruments.get(period.instrument)
        if instrument is None:
            raise TsaraIngestError(
                f"Excluded period {period.describe()} in '{period.source}' names "
                f"instrument '{period.instrument}', which the manifest does not declare; "
                f"available: {sorted(manifest.instruments)}."
            )
        if period.variable is not None and period.variable not in instrument.variables:
            raise TsaraIngestError(
                f"Excluded period {period.describe()} in '{period.source}' names "
                f"variable '{period.variable}', which instrument '{period.instrument}' "
                f"does not declare; available: {sorted(instrument.variables)}."
            )


def periods_for(periods: Iterable[ExcludedPeriod], instrument: str) -> tuple[ExcludedPeriod, ...]:
    """Return the periods that apply to any variable of ``instrument``, in order."""
    return tuple(period for period in periods if period.applies_to(instrument))


def period_mask(frame: pd.DataFrame, periods: Sequence[ExcludedPeriod]) -> npt.NDArray[np.bool_]:
    """Which rows of ``frame`` any of ``periods`` covers.

    A row is covered when its cell overlaps a period, or, when the frame
    carries no cell boundaries, when its timestamp lies inside one
    (``start <= t < end``).

    Parameters
    ----------
    frame : pandas.DataFrame
        The instrument's table on a tz-naive UTC index, with the reserved
        boundary columns when orchestration resolved cells.
    periods : Sequence of ExcludedPeriod
        The periods applying to the variable in question.

    Returns
    -------
    numpy.ndarray of bool
        True where the row is covered.
    """
    covered = np.zeros(len(frame), dtype=bool)
    if not periods:
        return covered
    if RAW_TIME_START_COLUMN in frame.columns:
        start = epoch_ns(pd.DatetimeIndex(frame[RAW_TIME_START_COLUMN]))
        stop = epoch_ns(pd.DatetimeIndex(frame[RAW_TIME_STOP_COLUMN]))
        for period in periods:
            covered |= (start < period.end_ns) & (stop > period.start_ns)
        return covered
    instants = epoch_ns(pd.DatetimeIndex(frame.index))
    for period in periods:
        covered |= (instants >= period.start_ns) & (instants < period.end_ns)
    return covered


def describe_periods(periods: Iterable[ExcludedPeriod]) -> str:
    """One string naming every period, for a stream attribute."""
    return "; ".join(period.describe() for period in periods)


def _iso(ns: int) -> str:
    """Spell epoch nanoseconds as an ISO 8601 UTC timestamp."""
    return pd.Timestamp(ns, unit="ns").isoformat() + "Z"
