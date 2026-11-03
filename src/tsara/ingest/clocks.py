"""Correcting an instrument's clock at ingestion, from the table a manifest names.

Where this sits
---------------
``Manifest.clock_corrections`` names one or more correction files; their
format, and the shift a row gives at a recorded time, live in
:mod:`tsara.core.clocks`, because the clock survey writes the same format
(``docs/METHODS.md`` §12.4). This module checks the table against the
manifest and applies it.

When it is applied
------------------
To each file, **right after the file is read and before its cadence is
measured**. Two consequences make that the right moment rather than a later
one. The cadence, and the cells built from it, are then measured on corrected
times, so a drifting clock does not hand its drift to every cell width
(§10.5). And a reading is looked up by the time its own file recorded, which
is the clock the table's stretches are written on. ``time_shift`` is added
afterwards, unchanged, as a constant on top (§10.7).

A shift moves a reading's cell; it never resizes it. The timestamp and both
of a file's own cell boundaries, when it states them, move by the same
amount: the shift at the reading's recorded time.

What is recorded, and said
--------------------------
A reading no row holds keeps its recorded time. That is the decision of
2026-11-03 (warn and label, not refuse): a stretch the survey could not
measure is better kept as recorded and named than dropped. The stream
records every row for its instrument, the files, the reference clock and the
share of readings left uncorrected (:mod:`tsara.core.naming`), and a warning
names the uncorrected stretches. A second warning counts readings the
correction puts before an earlier reading of their own file, which a shift
that falls from one row to the next does; they are kept, sorted with the
rest, and any landing on an earlier reading's exact time are then dropped as
duplicates by the ordering that follows (§12.3). A row that holds none of its
instrument's readings is logged, since the data it names may not be loaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.clocks import shift_at
from tsara.core.naming import RAW_TIME_START_COLUMN, RAW_TIME_STOP_COLUMN
from tsara.core.timebase import epoch_ns, iso_utc
from tsara.ingest.base import TsaraIngestError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from tsara.config.manifest import Manifest
    from tsara.core.clocks import ClockCorrection

logger = logging.getLogger(__name__)

__all__ = ["ClockRecord", "check_clock_corrections", "corrections_for"]

#: How many uncorrected stretches a warning lists before summarizing.
_MAX_LISTED = 8


@dataclass
class ClockRecord:
    """One instrument's corrections, applied file by file, and what they did.

    Attributes
    ----------
    rows : tuple of ClockCorrection
        The instrument's rows (:func:`corrections_for`). Empty when the
        manifest declares none for it, and then :meth:`correct` changes
        nothing and nothing is recorded.
    """

    rows: tuple[ClockCorrection, ...]
    _recorded: list[npt.NDArray[np.int64]] = field(default_factory=list, init=False, repr=False)
    _covered: list[npt.NDArray[np.bool_]] = field(default_factory=list, init=False, repr=False)
    _row_hits: list[int] = field(default_factory=list, init=False, repr=False)
    _moved_back: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._row_hits = [0] * len(self.rows)

    def correct(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Shift one file's readings by the rows that hold them, and remember what happened.

        Parameters
        ----------
        frame : pandas.DataFrame
            One file as its reader returned it: a tz-naive UTC index of
            recorded times, and the reserved boundary columns when the file
            states its cells.

        Returns
        -------
        pandas.DataFrame
            A copy with the index and boundary columns shifted, rows in the
            file's own order. The frame itself when the instrument has no rows.
        """
        if not self.rows:
            return frame
        recorded = epoch_ns(pd.DatetimeIndex(frame.index))
        shift, covered = shift_at(self.rows, recorded)
        out = frame.copy()
        out.index = pd.DatetimeIndex(
            (recorded + shift).astype("datetime64[ns]"), name=frame.index.name
        )
        for column in (RAW_TIME_START_COLUMN, RAW_TIME_STOP_COLUMN):
            if column in out.columns:
                bounds = epoch_ns(pd.DatetimeIndex(out[column]))
                out[column] = (bounds + shift).astype("datetime64[ns]")

        self._recorded.append(recorded)
        self._covered.append(covered)
        for i, row in enumerate(self.rows):
            self._row_hits[i] += int(
                np.count_nonzero((recorded >= row.start_ns) & (recorded < row.end_ns))
            )
        self._moved_back += _count_moved_back(recorded, recorded + shift)
        return out

    @property
    def n_readings(self) -> int:
        """Readings seen so far, over every corrected file."""
        return sum(len(chunk) for chunk in self._recorded)

    @property
    def n_uncorrected(self) -> int:
        """Readings no row held, which keep their recorded times."""
        return sum(int(np.count_nonzero(~chunk)) for chunk in self._covered)

    @property
    def uncorrected_share(self) -> float:
        """The share of readings left uncorrected; NaN before any file is seen."""
        n = self.n_readings
        return self.n_uncorrected / n if n else float("nan")

    def report(self, instrument: str) -> None:
        """Log what the correction did: rows holding nothing, readings left or moved back."""
        if not self.rows:
            return
        for row, hits in zip(self.rows, self._row_hits, strict=True):
            if hits == 0:
                logger.info(
                    "Instrument '%s': clock correction %s (%s) holds none of its readings; "
                    "the data it names may not be loaded.",
                    instrument,
                    row.describe(),
                    row.where(),
                )
        if self.n_uncorrected:
            stretches = self._uncorrected_stretches()
            listed = "; ".join(
                f"{iso_utc(first)}/{iso_utc(last)} ({count} reading{'s' if count != 1 else ''})"
                for first, last, count in stretches[:_MAX_LISTED]
            )
            more = len(stretches) - _MAX_LISTED
            if more > 0:
                listed += f"; and {more} more stretch{'es' if more != 1 else ''}"
            logger.warning(
                "Instrument '%s': %d of %d readings (%.1f%%) lie in no clock-correction row "
                "and keep their recorded times, recorded in the stream as uncorrected: %s.",
                instrument,
                self.n_uncorrected,
                self.n_readings,
                100 * self.uncorrected_share,
                listed,
            )
        if self._moved_back:
            logger.warning(
                "Instrument '%s': the clock correction puts %d reading(s) before an earlier "
                "reading of their own file, as a shift that falls from one row to the next "
                "does. They are kept and sorted with the rest; any landing on an earlier "
                "reading's exact time is then dropped as a duplicate (METHODS.md §12.3).",
                instrument,
                self._moved_back,
            )

    def _uncorrected_stretches(self) -> list[tuple[int, int, int]]:
        """Return the runs of uncorrected readings in recorded time, as (first, last, count)."""
        recorded = np.concatenate(self._recorded)
        covered = np.concatenate(self._covered)
        order = np.argsort(recorded, kind="stable")
        recorded, covered = recorded[order], covered[order]
        edges = np.flatnonzero(np.diff(np.r_[0, (~covered).astype(np.int8), 0]))
        return [
            (int(recorded[start]), int(recorded[stop - 1]), int(stop - start))
            for start, stop in zip(edges[::2], edges[1::2], strict=True)
        ]


def check_clock_corrections(rows: Sequence[ClockCorrection], manifest: Manifest) -> None:
    """Refuse a row naming an instrument or reference clock the manifest does not declare.

    Run before any data file is read, so that a typo fails fast rather than
    after a thousand-file crawl.

    Raises
    ------
    TsaraIngestError
        If a row's instrument, or the instrument keeping its reference
        clock, is not in the manifest.
    """
    known = sorted(manifest.instruments)
    for row in rows:
        for role, name in (
            ("instrument", row.instrument),
            ("reference clock", row.reference_clock),
        ):
            if name not in manifest.instruments:
                raise TsaraIngestError(
                    f"Clock correction {row.where()} names {role} '{name}', which the "
                    f"manifest does not declare; available: {known}."
                )


def corrections_for(
    rows: Iterable[ClockCorrection], instrument: str
) -> tuple[ClockCorrection, ...]:
    """Return the rows that correct ``instrument``, in file order."""
    return tuple(row for row in rows if row.instrument == instrument)


def _count_moved_back(recorded: npt.NDArray[np.int64], corrected: npt.NDArray[np.int64]) -> int:
    """Count readings the correction puts before a reading recorded earlier in the same file."""
    order = np.argsort(recorded, kind="stable")
    in_order = corrected[order]
    # Empty slices for a file of fewer than two readings, which count zero.
    latest_before = np.maximum.accumulate(in_order)[:-1]
    return int(np.count_nonzero(in_order[1:] < latest_before))
