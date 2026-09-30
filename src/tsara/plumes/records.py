"""Records: the stretches of a stream the plumes stage describes one at a time.

The clean level and spread (``docs/METHODS.md`` §6.8) describe plume-free
air over a stretch of time, and the stretch matters: a ground site that runs
for eleven days without a two-hour gap, or a van that logs parked and
driving alike for a week, would otherwise be described by one number mixing
air that was never the same. So a variable's finite readings are split where
consecutive ones are more than a gap apart, and each piece is cut into equal
parts no longer than a maximum length. Each part is a **record**. Records
belong to a variable, are found from which of its readings are finite, and
are the same at every sweep point.

A record also fixes what the detector calls a **dropout**: two consecutive
finite readings farther apart than :data:`DROPOUT_SPACING_FACTOR` times the
record's median spacing between consecutive finite readings. An event never
runs across one (§6.8), because missing data is not turbulent air. The rule
is on spacing, not on cell width, for the reason §2.5 measured: the 07-18
drive's analyzer reports every 2 or 3 s on 1 s cells, so a rule on width
would call every step a dropout.

This module is the vocabulary of the subpackage and holds its error class;
:mod:`tsara.plumes.clean` describes the air in each record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.exceptions import TsaraError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from tsara.core.support import CellBounds

__all__ = ["DROPOUT_SPACING_FACTOR", "Records", "TsaraPlumeError", "find_records"]

#: A gap between consecutive finite readings longer than this many times the
#: record's median spacing is a dropout (§2.5, §6.8). 1.5 keeps every jittered
#: step of a record that reports every 2 or 3 s (median 2 s: a 3 s step is
#: not a dropout) and catches a single missing row of a regular one (a 2 s
#: step on 1 s spacing is).
DROPOUT_SPACING_FACTOR = 1.5


class TsaraPlumeError(TsaraError):
    """Raised when plume events cannot be found as asked.

    Its own type because the failures are about records, thresholds and
    what the baseline state handed over, rather than about reading, joining
    or rolling data: readings out of time order, a record rule of no
    length, a clean-level estimator nobody registered.
    """


@dataclass(frozen=True, eq=False)
class Records:
    """A variable's readings, sorted into records.

    Attributes
    ----------
    index : numpy.ndarray of int64
        Per reading, the record it belongs to, numbered from 0 in time
        order; -1 for a reading that is not finite, which belongs to none.
    start_ns, stop_ns : numpy.ndarray of int64
        Per record, the cell start of its first reading and the cell stop of
        its last.
    median_spacing_ns : numpy.ndarray of float64
        Per record, the median time between the midpoints of consecutive
        finite readings; NaN for a record of one reading.
    dropout_before : numpy.ndarray of bool
        Per reading, whether the gap back to the previous finite reading of
        its record is a dropout. False for a record's first reading (a
        record boundary breaks an event anyway) and for a reading that is
        not finite.
    """

    index: npt.NDArray[np.int64]
    start_ns: npt.NDArray[np.int64]
    stop_ns: npt.NDArray[np.int64]
    median_spacing_ns: npt.NDArray[np.float64]
    dropout_before: npt.NDArray[np.bool_]

    @property
    def n(self) -> int:
        """The number of records."""
        return int(self.start_ns.size)


def find_records(
    cells: CellBounds,
    finite: npt.NDArray[np.bool_],
    *,
    gap_ns: int,
    max_length_ns: int,
) -> Records:
    """Split a variable's finite readings into records, and mark its dropouts.

    Consecutive finite readings more than ``gap_ns`` apart, midpoint to
    midpoint, end one piece and start the next. A piece whose first and last
    midpoints are more than ``max_length_ns`` apart is cut into the fewest
    equal parts no longer than that, each reading going to the part its
    midpoint falls in; a part that holds no reading is not a record, so
    records are numbered consecutively.

    Parameters
    ----------
    cells : CellBounds
        The stream's cells, one per reading, in time order.
    finite : numpy.ndarray of bool
        Which readings of the variable are finite.
    gap_ns : int
        ``PlumesConfig.record_gap``, in nanoseconds.
    max_length_ns : int
        ``PlumesConfig.max_record_length``, in nanoseconds.

    Returns
    -------
    Records
        The record of every reading, each record's extent and median
        spacing, and where the dropouts are.

    Raises
    ------
    TsaraPlumeError
        If ``finite`` does not match the cells, a length is not positive, or
        the finite readings are not in time order.
    """
    mask = np.asarray(finite, dtype=bool)
    if mask.shape != cells.start_ns.shape:
        raise TsaraPlumeError(
            f"{mask.size} finite flags for {cells.start_ns.size} cells; there must be one per "
            "reading."
        )
    if gap_ns <= 0 or max_length_ns <= 0:
        raise TsaraPlumeError(
            f"A record needs a positive gap and maximum length; got {gap_ns} ns and "
            f"{max_length_ns} ns."
        )
    kept = np.flatnonzero(mask)
    index = np.full(mask.size, -1, dtype=np.int64)
    dropout_before = np.zeros(mask.size, dtype=bool)
    if kept.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return Records(index, empty, empty, np.empty(0), dropout_before)
    midpoints = cells.midpoint_ns[kept]
    spacing = np.diff(midpoints)
    if np.any(spacing < 0):
        raise TsaraPlumeError(
            "The finite readings are not in time order; a stream's cells must be sorted "
            "by their midpoints."
        )

    # 1. Pieces: split wherever two consecutive finite readings are more
    #    than the gap apart.
    piece_starts = np.concatenate([[0], np.flatnonzero(spacing > gap_ns) + 1])
    piece_stops = np.concatenate([piece_starts[1:], [kept.size]])

    # 2. Parts: cut each piece into the fewest equal parts no longer than the
    #    maximum.
    record_of = np.empty(kept.size, dtype=np.int64)
    next_record = 0
    for first, stop in zip(piece_starts, piece_stops, strict=True):
        record_of[first:stop] = next_record + _parts(midpoints[first:stop], max_length_ns)
        next_record = int(record_of[stop - 1]) + 1
    index[kept] = record_of

    # 3. Each record's extent, from its first and last readings.
    first_of = np.flatnonzero(np.concatenate([[True], np.diff(record_of) != 0]))
    last_of = np.concatenate([first_of[1:] - 1, [kept.size - 1]])
    start_ns = cells.start_ns[kept[first_of]]
    stop_ns = cells.stop_ns[kept[last_of]]

    # 4. Dropouts, against each record's own median spacing.
    median_spacing = np.full(first_of.size, np.nan)
    for r, (a, b) in enumerate(zip(first_of, last_of, strict=True)):
        if b == a:
            continue
        steps = spacing[a:b]
        median_spacing[r] = float(np.median(steps))
        dropout_before[kept[a + 1 : b + 1]] = steps > DROPOUT_SPACING_FACTOR * median_spacing[r]
    return Records(index, start_ns, stop_ns, median_spacing, dropout_before)


def _parts(midpoints: npt.NDArray[np.int64], max_length_ns: int) -> npt.NDArray[np.int64]:
    """Return each reading's part of one piece, numbered 0, 1, ... with no empty part.

    The piece spans ``span`` from its first midpoint to its last and is cut
    into ``n = ceil(span / max)`` parts of exactly ``span / n`` each, which is
    no longer than the maximum; a reading at offset ``o`` falls in part
    ``floor(o * n / span)``, and the one at the very end, on the last part's
    closing edge, stays in the last part. The product ``o * n`` is taken in
    Python integers: in int64 it overflows once a piece spans many weeks
    (a year is 3.2e16 ns, cut into about 1 460 six-hour parts).
    """
    offset = midpoints - midpoints[0]
    span = int(offset[-1])
    n_parts = max(1, -(-span // max_length_ns))
    if n_parts == 1:
        return np.zeros(offset.size, dtype=np.int64)
    exact = (offset.astype(object) * n_parts) // span
    part = np.minimum(exact.astype(np.int64), n_parts - 1)
    # A part with no reading in it (a long quiet stretch shorter than the
    # gap) is dropped, so that record numbers stay consecutive.
    _, dense = np.unique(part, return_inverse=True)
    return dense.astype(np.int64)
