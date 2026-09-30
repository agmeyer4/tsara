"""The detector: two-threshold runs of the statistic z, and what each event records.

With the clean level *m* and spread *s* of a record (:mod:`tsara.plumes.clean`),
each reading's statistic is z = (Δ − m) / s, and an **event** is a run of
readings with z ≥ exit that holds at least one with z ≥ entry
(``docs/METHODS.md`` §6.8): offline hysteresis, so that a plume hovering near
one threshold is not chopped into fragments. Four rules shape the runs.

- **A run never crosses a dropout** (:mod:`tsara.plumes.records`), a record
  boundary, or a reading whose enhancement is blank: missing data is not
  turbulent air, and a blank baseline says nothing about the air either.
- **A dip below exit shorter than** ``max_internal_gap`` **is bridged**, so
  that noise does not split one plume in two. The dip is the time between
  the cell stop of the last reading above exit and the cell start of the
  next one (zero where those cells overlap), and it is bridged only when
  strictly shorter, so a gap of zero bridges nothing; the readings in it
  belong to the event.
- **There is no minimum duration.** Width cannot tell a chance crossing from
  a narrow plume (§9.5's argument), so a one-reading event is an event.
- **An event is its readings' cells**: it runs from its first reading's cell
  start to its last reading's cell stop, and records the share of that
  interval its cells cover, counted once where jittered cells overlap. On a
  record whose 1 s cells arrive every 2 or 3 s the share is under half, and
  says so.

The rate at which chance alone makes events has a closed form for
independent Gaussian readings (:func:`expected_chance_rate`), which the
plume state records beside the events so that a sweep point's count can be
read against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm

from tsara.plumes.records import TsaraPlumeError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from tsara.core.support import CellBounds
    from tsara.plumes.records import Records

__all__ = ["Events", "describe_events", "expected_chance_rate", "find_events"]


@dataclass(frozen=True, eq=False)
class Events:
    """The events of one variable at one sweep point, in time order.

    Every per-event array has one entry per event; reading indices point into
    the stream's own rows.

    Attributes
    ----------
    number : numpy.ndarray of int64
        The event's number: 0, 1, ... in time order for events found here;
        the trigger's numbers for events a variable took from its trigger.
    first, last : numpy.ndarray of int64
        The event's first and last readings.
    peak : numpy.ndarray of int64
        Its reading with the largest score (the earliest, on a tie): z for
        events found here, which within one record is also the largest
        enhancement, since z is the enhancement shifted and scaled by the
        record's own level and spread; the enhancement itself for events
        taken from a trigger. The first reading when no score is finite.
    start_ns, stop_ns : numpy.ndarray of int64
        The first reading's cell start and the last reading's cell stop, or
        the trigger's interval for an event taken from one.
    n_readings : numpy.ndarray of int64
        Finite readings from first to last, a bridged dip's included.
    covered : numpy.ndarray of float64
        The share of [start, stop) its readings' cells cover, counted once
        where cells overlap.
    peak_score : numpy.ndarray of float64
        The score at the peak; NaN when none is finite.
    membership : numpy.ndarray of int64
        Per reading of the stream, the event it belongs to, numbered from 0;
        -1 for a reading in none, and for a reading that is not finite even
        when it lies inside an event's interval, since it has nothing to
        belong with.
    """

    number: npt.NDArray[np.int64]
    first: npt.NDArray[np.int64]
    last: npt.NDArray[np.int64]
    peak: npt.NDArray[np.int64]
    start_ns: npt.NDArray[np.int64]
    stop_ns: npt.NDArray[np.int64]
    n_readings: npt.NDArray[np.int64]
    covered: npt.NDArray[np.float64]
    peak_score: npt.NDArray[np.float64]
    membership: npt.NDArray[np.int64]

    @property
    def n(self) -> int:
        """The number of events."""
        return int(self.first.size)


def find_events(
    z: npt.NDArray[np.float64],
    cells: CellBounds,
    records: Records,
    *,
    enter: float,
    exit_: float,
    max_internal_gap_ns: int,
) -> Events:
    """Find the events of one variable at one sweep point.

    Parameters
    ----------
    z : numpy.ndarray of float64
        Per reading, (enhancement − clean level) / clean spread of its
        record; NaN wherever the reading, its baseline or its record's
        description is blank.
    cells : CellBounds
        The stream's cells, one per reading, in time order.
    records : Records
        The variable's records and dropouts
        (:func:`~tsara.plumes.records.find_records`).
    enter, exit_ : float
        The entry and exit multiples; entry must exceed exit.
    max_internal_gap_ns : int
        A dip below exit strictly shorter than this is bridged; 0 bridges
        nothing.

    Returns
    -------
    Events
        The events, in time order, and every reading's membership.

    Raises
    ------
    TsaraPlumeError
        If the arrays do not match, entry does not exceed exit, or the gap
        is negative.
    """
    values = np.asarray(z, dtype=np.float64)
    if values.shape != records.index.shape or values.shape != cells.start_ns.shape:
        raise TsaraPlumeError(
            f"z of shape {values.shape} does not match {records.index.size} readings and "
            f"{cells.start_ns.size} cells."
        )
    if not enter > exit_:
        raise TsaraPlumeError(f"The entry multiple ({enter}) must exceed the exit ({exit_}).")
    if max_internal_gap_ns < 0:
        raise TsaraPlumeError(f"max_internal_gap cannot be negative; got {max_internal_gap_ns} ns.")

    # Work along the finite readings only: a reading that is not finite is a
    # hole in time, and whether the hole breaks a run is the dropout rule's
    # business, already written into `records.dropout_before`.
    rows = np.flatnonzero(records.index >= 0)
    zk = values[rows]
    valid = np.isfinite(zk)

    # 1. Segments: stretches no run may cross. A new one starts at a record
    #    boundary, at a dropout and at a blank reading; the blank reading is
    #    never above exit, so no run holds it, and whatever follows it lies in
    #    a later segment than whatever came before.
    record_of = records.index[rows]
    breaks = np.ones(rows.size, dtype=bool)
    breaks[1:] = (record_of[1:] != record_of[:-1]) | ~valid[1:]
    breaks |= records.dropout_before[rows]
    segment = np.cumsum(breaks)

    # 2. Runs: consecutive readings at or above exit within one segment.
    above = valid & (zk >= exit_)
    opens = above & (breaks | ~np.r_[False, above[:-1]])
    closes = above & (np.r_[breaks[1:], True] | ~np.r_[above[1:], False])
    run_first, run_last = np.flatnonzero(opens), np.flatnonzero(closes)

    # 3. Bridges: a run joins the one before it when both lie in one segment
    #    and the dip between them, cell stop to cell start, is shorter than
    #    the gap allowed. Where wide cells overlap the dip is zero, not
    #    negative, so that a gap of zero bridges nothing. Whatever lies between
    #    them in one segment is finite and below exit, so it belongs to the
    #    event.
    starts, stops = cells.start_ns[rows], cells.stop_ns[rows]
    dip = np.maximum(starts[run_first[1:]] - stops[run_last[:-1]], 0)
    joins = (segment[run_first[1:]] == segment[run_last[:-1]]) & (dip < max_internal_gap_ns)
    group_opens = np.r_[True, ~joins] if run_first.size else np.empty(0, dtype=bool)
    group_first = run_first[group_opens]
    group_last = run_last[np.r_[group_opens[1:], True]] if run_first.size else run_last

    # 4. Keep the groups that reach entry: each group's largest z, taken over
    #    its own span (never NaN inside, a blank reading having broken it).
    member, within = _spans(group_first, group_last)
    highest = np.full(group_first.size, -np.inf)
    np.maximum.at(highest, member, zk[within])
    kept = np.flatnonzero(highest >= enter)

    # 5. Number the kept groups in time order, and describe them.
    renumber = np.full(group_first.size, -1, dtype=np.int64)
    renumber[kept] = np.arange(kept.size)
    membership = np.full(values.size, -1, dtype=np.int64)
    membership[rows[within]] = renumber[member]
    return describe_events(membership, values, cells)


def describe_events(
    membership: npt.NDArray[np.integer],
    score: npt.NDArray[np.float64],
    cells: CellBounds,
    *,
    intervals: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]
    | None = None,
) -> Events:
    """Describe the events a membership array holds.

    The last step of :func:`find_events`, and on its own the way to describe
    events found elsewhere: from a saved plume state's ``event_x``, or a
    variable's share of its trigger's events.

    Parameters
    ----------
    membership : numpy.ndarray of int
        Per reading, the number of the event it belongs to, -1 for none.
    score : numpy.ndarray of float64
        Per reading, the value whose largest is an event's peak: z, or for a
        triggered variable its enhancement. A reading with no finite score
        is passed over for the peak.
    cells : CellBounds
        The stream's cells.
    intervals : tuple of three numpy arrays, optional
        ``(numbers, start_ns, stop_ns)``: the interval of each event by
        number, for events taken from a trigger, whose interval is the
        trigger's rather than the span of this variable's own readings.

    Returns
    -------
    Events
        One entry per event number present, in increasing number.

    Raises
    ------
    TsaraPlumeError
        If the arrays do not match the cells, or an event has no interval.
    """
    member_of = np.asarray(membership, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    if member_of.shape != cells.start_ns.shape or values.shape != member_of.shape:
        raise TsaraPlumeError(
            f"A membership of shape {member_of.shape} and a score of shape {values.shape} "
            f"do not match {cells.start_ns.size} cells."
        )
    rows = np.flatnonzero(member_of >= 0)
    number, group = np.unique(member_of[rows], return_inverse=True)
    # Rows grouped by event, in time order within each: the first of a group
    # is its first reading, the last its last.
    order = np.lexsort((rows, group))
    rows, group = rows[order], group[order]
    counts = np.bincount(group, minlength=number.size).astype(np.int64)
    heads = np.cumsum(counts) - counts
    first, last = rows[heads], rows[heads + counts - 1]
    # The peak: sorted by group, then by score descending, then by row, the
    # head of each group. numpy sorts NaN last, so a reading with no score is
    # never the peak while one of its event has a score.
    peak = rows[np.lexsort((rows, -values[rows], group))][heads]
    if intervals is None:
        start_ns, stop_ns = cells.start_ns[first], cells.stop_ns[last]
    else:
        known, starts, stops = intervals
        where = np.searchsorted(known, number)
        if np.any(where >= known.size) or np.any(
            known[np.minimum(where, known.size - 1)] != number
        ):
            raise TsaraPlumeError("An event taken from a trigger has no interval of its own.")
        start_ns, stop_ns = starts[where], stops[where]
    covered = _covered(cells.start_ns[rows], cells.stop_ns[rows], group, start_ns, stop_ns)
    return Events(
        number=number.astype(np.int64),
        first=first,
        last=last,
        peak=peak,
        start_ns=start_ns,
        stop_ns=stop_ns,
        n_readings=counts,
        covered=covered,
        peak_score=values[peak],
        membership=member_of,
    )


def _spans(
    first: npt.NDArray[np.int64], last: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return every position from each ``first`` to its ``last``, with its span's number.

    Vectorized rather than one ``arange`` per span, since a 10 Hz record can
    hold tens of thousands of events at a sweep point.
    """
    lengths = (last - first + 1).astype(np.int64)
    number = np.repeat(np.arange(first.size, dtype=np.int64), lengths)
    step = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    return number, np.repeat(first.astype(np.int64), lengths) + step


def _covered(
    starts: npt.NDArray[np.int64],
    stops: npt.NDArray[np.int64],
    member: npt.NDArray[np.int64],
    event_start: npt.NDArray[np.int64],
    event_stop: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Return the share of each event's interval its readings' cells cover.

    The union of the cells, each clipped to the event's interval, so that
    jittered cells overlapping a little are not counted twice: each cell adds
    only what lies beyond the latest stop of the cells before it in the same
    event.
    """
    if member.size == 0:
        return np.empty(0, dtype=np.float64)
    starts = np.maximum(starts, event_start[member])
    stops = np.minimum(stops, event_stop[member])
    frame = pd.DataFrame({"member": member, "stop": stops})
    reach = frame.groupby("member")["stop"].cummax().to_numpy()
    before = np.r_[np.iinfo(np.int64).min, reach[:-1]]
    before[np.r_[True, member[1:] != member[:-1]]] = np.iinfo(np.int64).min
    added = np.clip(stops - np.maximum(starts, before), 0, None)
    union = np.bincount(member, weights=added.astype(np.float64), minlength=event_start.size)
    length = (event_stop - event_start).astype(np.float64)
    return np.asarray(union / length, dtype=np.float64)


def expected_chance_rate(enter: float, exit_: float) -> float:
    """Return the events chance alone makes per reading, on independent Gaussian readings.

    With ``p_x`` and ``p_e`` the upper-tail probabilities of the exit and
    entry multiples, a reading opens a run above exit with probability
    ``(1 - p_x) p_x``, the run's length is geometric, and each of its
    readings reaches entry with probability ``p_e / p_x``, so the rate is

        r = (1 - p_x) [p_x - (1 - p_x)(p_x - p_e) / (1 - p_x + p_e)]

    (``docs/METHODS.md`` §6.8). Oversampled, autocorrelated readings cross
    less often; bridged dips can only merge events, so with bridging this is
    an upper bound.

    Parameters
    ----------
    enter, exit_ : float
        The entry and exit multiples.

    Returns
    -------
    float
        Events per reading; multiply by readings per hour for a rate an hour.
    """
    p_exit, p_enter = float(norm.sf(exit_)), float(norm.sf(enter))
    return (1 - p_exit) * (p_exit - (1 - p_exit) * (p_exit - p_enter) / (1 - p_exit + p_enter))
