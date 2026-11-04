"""One shift: how many seconds to add to one series' times to put its plumes on another's.

Where this sits
---------------
The clock survey (:mod:`tsara.survey.clocks`) cuts records into segments and
asks, segment by segment, how far apart two instruments' files put the same
air (``docs/METHODS.md`` §12.4). This module answers that question once: for
one segment and one tier of tracer pairs, the shift at which the two series
correlate best, how well, and on how many paired seconds.

The sign
--------
Every clock number in TSARA is a **shift**: seconds added to a recorded time,
``corrected = recorded + shift``, the rule of ``time_shift`` and of the
correction table (§12.3). Here the shift puts the instrument on its partner's
clock: an instrument whose file shows the air 7 s after its partner's needs
-7 s, one whose file shows it 4 s before needs +4 s.

How a series is prepared
------------------------
Each variable is put on **whole seconds**: every timestamp floored to its
second, the first reading of each second kept. It is a step inside the
measurement, never written to a product, and the only one that lets a 1 s
analyzer and a 2 s one be compared second by second without interpolating a
gas. Then its **wiggles**: the variable minus its centred running median, so
that plumes, not the background's slow swings, decide the correlation. The
median needs a quarter of the readings its window could hold (at least three)
before it is trusted, which keeps a 1 s record at the 30 readings of the
alignment work and lets a 5 s weather record take part at all.

The shift of a segment
----------------------
For every shift s in the search range, Pearson's r between the instrument's
wiggles at t and the partner's at t + s, over the seconds of the segment where
both have a value, if there are at least the minimum number of them. A tier
with several pairs averages their correlation curves before the best shift is
chosen, so that each pair counts once whatever its cadence, and weak pairs
that share a shift strengthen each other. Averaging assumes the pairs share
their shift; when two pairs disagree it peaks between them, at a shift neither
supports (measured: pairs at -4 and -10 s pool to -7 s at r 0.76). So a pair
that is decided on its own and puts its best shift further than the tolerance
from the pooled one makes the tier's pairs disagree, and the tier undecided; a
pair too weak to decide alone cannot. A best shift at the edge of the search
is reported but never decided, since the true one may lie beyond.

The shift below a second
------------------------
Whole seconds throw away each reading's position within its second, so the
shift above can only be a whole number of seconds, and a one-sided one, lying
up to a second above the true shift: a file 7.7 s late (true shift -7.7 s)
peaks at -7. That search finds the peak; the
shift itself is then measured on the readings' own cells
(:func:`refined_shift`). For every s on a lattice of the survey's
``resolution`` within two seconds of the peak, the narrower-celled series'
wiggles are averaged, overlap-weighted, onto the wider-celled series' cells
put on the other clock by s, through the same join as every other in TSARA
(:func:`tsara.core.support.overlap_pairs`), and r is taken over the cells the
join covers fully. On air whose readings are exact cell means this is exact
where the two sets of cells line up, and on 2 and 4 s plumes at every phase;
on 12 s plumes the share of a joined reading lying outside the cell moves it
by up to 0.06 s (measured, ``docs/METHODS.md`` §12.4). With noise one bias is
left: averaging lowers the noise of the joined series, by an amount that
changes with s, and r rises where it falls. When the joined readings carry a
declared or reported random sigma, that noise is removed from the joined
series' variance (each cell's share is sum(w^2 sigma^2) / sum(w)^2 of its
overlap weights w), and the measurement says so; it is never removed with a
sigma the survey estimates itself, since a wrong one moves the shift (§12.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.exceptions import TsaraError
from tsara.core.naming import TIME_BOUNDS_VAR, TIME_COORD, sigma_rand_name
from tsara.core.support import CellBounds, nominal_cadence_ns, overlap_pairs, stream_cells
from tsara.core.timebase import NS_PER_S, SECOND_NS

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt
    import xarray as xr

__all__ = [
    "CellSeries",
    "SecondsSeries",
    "SegmentShift",
    "TsaraSurveyError",
    "cell_series",
    "pooled_shift",
    "refined_shift",
    "seconds_series",
]

#: The share of a running-median window that must hold readings before the
#: median is used, and the floor under it.
_MEDIAN_FILL = 0.25
_MEDIAN_MIN_READINGS = 3

#: How far either side of the whole-second peak the shift is looked for on the
#: readings' own cells. The whole-second peak can sit up to about 1.4 s from
#: the true shift (measured, METHODS §12.4), so two seconds hold it; a best
#: shift at either end means the peak was not where the whole seconds put it.
_REFINE_NS = 2 * SECOND_NS

#: The random-sigma provenances a stream can carry that are statements about
#: the readings rather than labels for their absence (ingest/uncertainty.py).
_STATED_SIGMA = ("declared", "reported")


class TsaraSurveyError(TsaraError):
    """Raised when the survey cannot measure what it was asked to."""


@dataclass(frozen=True)
class SecondsSeries:
    """One variable on whole seconds, its running median removed.

    Attributes
    ----------
    first_s : int
        The epoch second of ``values[0]``.
    values : numpy.ndarray
        One value per second from ``first_s``: the wiggle at a second holding a
        reading, NaN elsewhere.
    """

    first_s: int
    values: npt.NDArray[np.float64]

    def window(self, first_s: int, stop_s: int) -> npt.NDArray[np.float64]:
        """Return the values for seconds ``[first_s, stop_s)``, NaN outside the record."""
        out = np.full(stop_s - first_s, np.nan)
        lo, hi = max(first_s, self.first_s), min(stop_s, self.first_s + len(self.values))
        if lo < hi:
            out[lo - first_s : hi - first_s] = self.values[lo - self.first_s : hi - self.first_s]
        return out


@dataclass(frozen=True)
class SegmentShift:
    """The best shift of one segment for one tier.

    Attributes
    ----------
    shift_s : float
        Seconds added to the instrument's recorded times to put it on its
        partner's clock (negative when its file shows the air later); NaN when
        no shift had enough paired seconds.
    r : float
        The correlation at that shift, pooled over the tier's pairs.
    n_pairs : int
        Paired seconds behind it, summed over the tier's pairs.
    at_edge : bool
        True when the best shift is at the edge of the search.
    pairs_disagree : bool
        True when a pair decided on its own puts its best shift further than
        the tolerance from the pooled shift.
    noise_removed : bool
        True when the shift was measured on the readings' own cells with the
        joined readings' stated noise removed, for every pair of the tier
        (:func:`refined_shift`).
    """

    shift_s: float
    r: float
    n_pairs: int
    at_edge: bool
    pairs_disagree: bool = False
    noise_removed: bool = False


@dataclass(frozen=True)
class CellSeries:
    """One variable on its readings' own cells, its running median removed.

    Attributes
    ----------
    cells : CellBounds
        The cells of the readings that have a wiggle (a finite value, and
        enough neighbours for the running median), ordered by start.
    values : numpy.ndarray
        The wiggle of each of those readings.
    sigma : numpy.ndarray or None
        Each of those readings' random sigma, when the stream declares or
        reports one for every finite reading; None otherwise.
    """

    cells: CellBounds
    values: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64] | None


def seconds_series(stream: xr.Dataset, variable: str, *, running_median_s: int) -> SecondsSeries:
    """Put one variable of a stream on whole seconds and remove its running median.

    Parameters
    ----------
    stream : xarray.Dataset
        An instrument's stream, sorted in time.
    variable : str
        The variable to take.
    running_median_s : int
        Window of the centred running median, in seconds.

    Returns
    -------
    SecondsSeries
        The wiggles on whole seconds; all NaN when the variable has no reading.

    Raises
    ------
    TsaraSurveyError
        If the stream has no such variable, or it is circular: a direction has
        no running median to take, and its plumes are no straight line.
    """
    _check_tracer(stream, variable)
    times = np.asarray(stream[TIME_COORD].values, dtype="datetime64[ns]").astype(np.int64)
    values = np.asarray(stream[variable].values, dtype=np.float64)
    finite = np.isfinite(values)
    seconds = np.floor_divide(times[finite], SECOND_NS)
    seconds, first = np.unique(seconds, return_index=True)
    if not len(seconds):
        return SecondsSeries(first_s=0, values=np.full(0, np.nan))
    kept = values[finite][first]
    grid = np.full(int(seconds[-1] - seconds[0]) + 1, np.nan)
    grid[seconds - seconds[0]] = _wiggles(seconds * SECOND_NS, kept, running_median_s)
    return SecondsSeries(first_s=int(seconds[0]), values=grid)


def cell_series(stream: xr.Dataset, variable: str, *, running_median_s: int) -> CellSeries:
    """Keep one variable on its readings' own cells and remove its running median.

    Parameters
    ----------
    stream : xarray.Dataset
        An instrument's stream, sorted in time. Its cells are its
        ``time_bnds``; a stream without them is given cells of its nominal
        cadence centred on each time, the weakest reading of its support
        (:func:`tsara.core.support.ensure_time_bounds`), without being changed.
    variable : str
        The variable to take.
    running_median_s : int
        Window of the centred running median, in seconds, the same as
        :func:`seconds_series`'s.

    Returns
    -------
    CellSeries
        The cells and wiggles of the readings that have a wiggle, and their
        random sigma when the stream states one (provenance ``declared`` or
        ``reported``) for every finite reading. A reading without a wiggle is
        left out, so the cells around it are not covered whole and do not
        count (:func:`refined_shift`).

    Raises
    ------
    TsaraSurveyError
        As :func:`seconds_series`.
    """
    _check_tracer(stream, variable)
    times = np.asarray(stream[TIME_COORD].values, dtype="datetime64[ns]").astype(np.int64)
    values = np.asarray(stream[variable].values, dtype=np.float64)
    if TIME_BOUNDS_VAR in stream.variables:
        cells = stream_cells(stream, str(stream.attrs.get("instrument", "?")))
    else:
        cadence = nominal_cadence_ns(times)
        cells = CellBounds.from_label(times, cadence if cadence else SECOND_NS, "unknown")
    finite = np.isfinite(values)
    sigma = _stated_sigma(stream, variable, finite)
    # The median runs in time order; the join needs the cells ordered by start,
    # which differs where the cadence changes (a 5 s cell after a 1 s one can
    # start before it), so the readings are reordered only afterwards.
    wiggles = _wiggles(times[finite], values[finite], running_median_s)
    has = np.isfinite(wiggles)
    start, stop = cells.start_ns[finite][has], cells.stop_ns[finite][has]
    order = np.argsort(start, kind="stable")
    return CellSeries(
        cells=CellBounds(start[order], stop[order]),
        values=wiggles[has][order],
        sigma=None if sigma is None else sigma[has][order],
    )


def _check_tracer(stream: xr.Dataset, variable: str) -> None:
    """Refuse a variable the stream lacks, or a circular one."""
    if variable not in stream.data_vars:
        raise TsaraSurveyError(
            f"Stream '{stream.attrs.get('instrument', '?')}' has no variable '{variable}' "
            f"to use as a tracer; it has {sorted(map(str, stream.data_vars))}."
        )
    if stream[variable].attrs.get("circular"):
        raise TsaraSurveyError(
            f"'{variable}' is circular and cannot be a tracer: a direction has no running "
            "median to remove."
        )


def _wiggles(
    times_ns: npt.NDArray[np.int64], values: npt.NDArray[np.float64], running_median_s: int
) -> npt.NDArray[np.float64]:
    """Return values less their centred running median, where the window holds enough readings.

    The window must hold a quarter of the readings it could at the record's
    median spacing (at least three) before its median is used.
    """
    if not len(values):
        return values
    spacing = float(np.median(np.diff(times_ns))) / NS_PER_S if len(values) > 1 else 1.0
    min_periods = max(_MEDIAN_MIN_READINGS, int(_MEDIAN_FILL * running_median_s / spacing))
    median = (
        pd.Series(values, index=pd.DatetimeIndex(times_ns))
        .rolling(f"{running_median_s}s", center=True, min_periods=min_periods)
        .median()
        .to_numpy()
    )
    return np.asarray(values - median, dtype=np.float64)


def _stated_sigma(
    stream: xr.Dataset, variable: str, finite: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64] | None:
    """Return the finite readings' random sigma if the stream states one for each, else None."""
    companion = sigma_rand_name(variable)
    provenance = stream[variable].attrs.get("uncertainty_provenance_random")
    if provenance not in _STATED_SIGMA or companion not in stream.data_vars:
        return None
    sigma = np.asarray(stream[companion].values, dtype=np.float64)[finite]
    return sigma if np.all(np.isfinite(sigma)) else None


def pooled_shift(
    pairs: Sequence[tuple[SecondsSeries, SecondsSeries]],
    first_s: int,
    stop_s: int,
    *,
    search_s: int,
    min_pairs: int,
    min_r: float,
    tolerance_s: float,
) -> SegmentShift:
    """Return the best shift of one segment for one tier of tracer pairs.

    Parameters
    ----------
    pairs : Sequence of (SecondsSeries, SecondsSeries)
        Each tracer pair: the instrument's series, then the partner's.
    first_s, stop_s : int
        The segment, ``[first_s, stop_s)`` in epoch seconds of the instrument's
        series.
    search_s : int
        The shift is searched in ``[-search_s, search_s]``.
    min_pairs : int
        Paired seconds a shift needs before its correlation counts.
    min_r : float
        The correlation at which a pair is decided on its own.
    tolerance_s : float
        How far a pair decided on its own may put its best shift from the
        pooled one before the tier's pairs disagree.

    Returns
    -------
    SegmentShift
        The best shift; NaN shift and r when no shift of any pair had enough
        pairs.
    """
    shifts = np.arange(-search_s, search_s + 1)
    curves, counts = [], []
    for own, partner in pairs:
        curve, count = _curve(
            own.window(first_s, stop_s),
            partner.window(first_s - search_s, stop_s + search_s),
            shifts,
            search_s,
            min_pairs,
        )
        curves.append(curve)
        counts.append(count)
    stacked = np.vstack(curves)
    usable = np.isfinite(stacked).any(axis=0)
    if not usable.any():
        return SegmentShift(shift_s=float("nan"), r=float("nan"), n_pairs=0, at_edge=False)
    pooled = np.full(len(shifts), -np.inf)
    pooled[usable] = np.nanmean(stacked[:, usable], axis=0)
    best = int(np.argmax(pooled))
    n = int(sum(count[best] for count in counts))
    disagree = False
    for curve in curves:
        if np.isfinite(curve).any():
            peak = int(np.nanargmax(curve))
            disagree |= bool(
                curve[peak] >= min_r and abs(int(shifts[peak] - shifts[best])) > tolerance_s
            )
    return SegmentShift(
        shift_s=float(shifts[best]),
        r=float(pooled[best]),
        n_pairs=n,
        at_edge=abs(int(shifts[best])) == search_s,
        pairs_disagree=disagree,
    )


def _curve(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    shifts: npt.NDArray[np.int64],
    search_s: int,
    min_pairs: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return r and the paired seconds at every shift: ``a(t)`` against ``b(t + s)``.

    ``b`` starts ``search_s`` seconds before ``a``, so ``b(t + s)`` for the
    segment's seconds is the slice beginning ``search_s + s`` in.
    """
    r = np.full(len(shifts), np.nan)
    count = np.zeros(len(shifts), dtype=np.int64)
    a_ok = np.isfinite(a)
    for i, shift in enumerate(shifts):
        start = search_s + int(shift)
        y = b[start : start + len(a)]
        both = a_ok & np.isfinite(y)
        n = int(np.count_nonzero(both))
        count[i] = n
        if n < min_pairs:
            continue
        x, y = a[both], y[both]
        x, y = x - x.mean(), y - y.mean()
        spread = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
        if spread > 0:
            r[i] = float(np.dot(x, y)) / spread
    return r, count


def refined_shift(
    pairs: Sequence[tuple[CellSeries, CellSeries]],
    first_ns: int,
    stop_ns: int,
    *,
    coarse_s: float,
    resolution_ns: int,
    min_pairs: int,
) -> tuple[float, bool, bool]:
    """Return one segment's shift measured on the readings' own cells, to a resolution.

    Parameters
    ----------
    pairs : Sequence of (CellSeries, CellSeries)
        Each tracer pair of the tier: the instrument's series, then the
        partner's.
    first_ns, stop_ns : int
        The segment, ``[first_ns, stop_ns)`` in epoch nanoseconds on the
        instrument's clock.
    coarse_s : float
        The whole-second peak (:func:`pooled_shift`), around which the shift
        is looked for.
    resolution_ns : int
        The lattice step: shifts are multiples of it, within two seconds of
        the peak.
    min_pairs : int
        Fully covered cells a shift needs before its correlation counts.

    Returns
    -------
    shift_s : float
        The shift at which the tier's pooled correlation is highest: seconds
        added to the instrument's recorded times to put it on its partner's
        clock; NaN when no shift had enough fully covered cells.
    at_edge : bool
        True when it lies at either end of the lattice, so the true shift may
        lie beyond.
    noise_removed : bool
        True when every pair's joined readings stated a random sigma, and their
        noise was removed from the joined series' variance.

    Notes
    -----
    In each pair the series with the narrower cells (the instrument's, on a
    tie) is joined onto the other's cells put on its clock by s, as every
    join in TSARA puts the narrower support onto the wider
    (``docs/METHODS.md`` §1.3). The wider series' cells are chosen once, by
    the whole-second peak, so every s is judged on the same cells. Pairs are
    pooled by averaging their curves, as in :func:`pooled_shift`.
    """
    coarse_ns = round(coarse_s * NS_PER_S)
    lowest = -((_REFINE_NS - coarse_ns) // resolution_ns)  # ceil((coarse - 2 s) / step)
    highest = (coarse_ns + _REFINE_NS) // resolution_ns
    shifts_ns = np.arange(lowest, highest + 1, dtype=np.int64) * resolution_ns
    curves = []
    removed = True
    for own, partner in pairs:
        curve, noise = _cell_curve(own, partner, first_ns, stop_ns, coarse_ns, shifts_ns, min_pairs)
        curves.append(curve)
        removed &= noise
    stacked = np.vstack(curves)
    usable = np.isfinite(stacked).any(axis=0)
    if not usable.any():
        return float("nan"), False, removed
    pooled = np.full(len(shifts_ns), -np.inf)
    pooled[usable] = np.nanmean(stacked[:, usable], axis=0)
    best = int(np.argmax(pooled))
    return int(shifts_ns[best]) / NS_PER_S, best in (0, len(shifts_ns) - 1), removed


def _cell_curve(
    own: CellSeries,
    partner: CellSeries,
    first_ns: int,
    stop_ns: int,
    coarse_ns: int,
    shifts_ns: npt.NDArray[np.int64],
    min_pairs: int,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Return r at every shift on the readings' own cells, and whether the noise was removed.

    ``own`` at t against ``partner`` at t + s: the partner joined onto the
    instrument's cells moved on by s, or the instrument joined onto the
    partner's cells moved back by s, whichever has the narrower cells.
    """
    if np.median(own.cells.width_ns) >= np.median(partner.cells.width_ns):
        target, joined, sign, offset = own, partner, 1, 0
    else:
        target, joined, sign, offset = partner, own, -1, coarse_ns
    middle = target.cells.midpoint_ns
    inside = (middle >= first_ns + offset) & (middle < stop_ns + offset)
    start, stop = target.cells.start_ns[inside], target.cells.stop_ns[inside]
    x = target.values[inside]
    width = (stop - start).astype(np.float64)
    noise = joined.sigma is not None
    curve = np.full(len(shifts_ns), np.nan)
    for i, shift in enumerate(shifts_ns):
        found = overlap_pairs(joined.cells, CellBounds(start + sign * shift, stop + sign * shift))
        w = found.overlap_ns.astype(np.float64)
        total = np.bincount(found.target_index, weights=w, minlength=len(start))
        weighted = np.bincount(
            found.target_index, weights=w * joined.values[found.reading_index], minlength=len(start)
        )
        # A cell counts only when the joined readings cover all of it, so that
        # each joined value is the mean over exactly the cell it is compared with.
        full = total >= width
        if np.count_nonzero(full) < min_pairs:
            continue
        a = x[full] - x[full].mean()
        b = weighted[full] / total[full]
        b = b - b.mean()
        var_b = float(np.mean(b * b))
        if joined.sigma is not None:
            shares = np.bincount(
                found.target_index,
                weights=w * w * joined.sigma[found.reading_index] ** 2,
                minlength=len(start),
            )
            var_b -= float(np.mean(shares[full] / total[full] ** 2))
        var_a = float(np.mean(a * a))
        if var_a > 0 and var_b > 0:
            curve[i] = float(np.mean(a * b)) / np.sqrt(var_a * var_b)
    return curve, noise
