"""One lag: how many seconds later one series shows the same plumes than another.

Where this sits
---------------
The clock survey (:mod:`tsara.survey.clocks`) cuts records into segments and
asks, segment by segment, how far apart two instruments' files put the same
air (``docs/METHODS.md`` §12.4). This module answers that question once: for
one segment and one tier of tracer pairs, the lag at which the two series
correlate best, how well, and on how many paired seconds.

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

The lag of a segment
--------------------
For every lag L in the search range, Pearson's r between the instrument's
wiggles at t and the partner's at t - L, over the seconds of the segment where
both have a value, if there are at least the minimum number of them. L is
positive when the instrument shows the air later than its partner. A tier with
several pairs averages their correlation curves before the best lag is chosen,
so that each pair counts once whatever its cadence, and weak pairs that share
a lag strengthen each other. Averaging assumes the pairs share their lag; when
two pairs disagree it peaks between them, at a lag neither supports (measured:
pairs at 4 and 10 s pool to 7 s at r 0.76). So a pair that is decided on its
own and puts its best lag further than the tolerance from the pooled one makes
the tier's pairs disagree, and the tier undecided; a pair too weak to decide
alone cannot. A best lag at the edge of the search is reported but never
decided, since the true one may lie beyond.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.exceptions import TsaraError
from tsara.core.naming import TIME_COORD
from tsara.core.timebase import SECOND_NS

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt
    import xarray as xr

__all__ = ["SecondsSeries", "SegmentLag", "TsaraSurveyError", "pooled_lag", "seconds_series"]

#: The share of a running-median window that must hold readings before the
#: median is used, and the floor under it.
_MEDIAN_FILL = 0.25
_MEDIAN_MIN_READINGS = 3


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
class SegmentLag:
    """The best lag of one segment for one tier.

    Attributes
    ----------
    lag_s : float
        Seconds the instrument shows the air later than its partner; NaN when no
        lag had enough paired seconds.
    r : float
        The correlation at that lag, pooled over the tier's pairs.
    n_pairs : int
        Paired seconds behind it, summed over the tier's pairs.
    at_edge : bool
        True when the best lag is at the edge of the search.
    pairs_disagree : bool
        True when a pair decided on its own puts its best lag further than the
        tolerance from the pooled lag.
    """

    lag_s: float
    r: float
    n_pairs: int
    at_edge: bool
    pairs_disagree: bool = False


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
    times = np.asarray(stream[TIME_COORD].values, dtype="datetime64[ns]").astype(np.int64)
    values = np.asarray(stream[variable].values, dtype=np.float64)
    finite = np.isfinite(values)
    seconds = np.floor_divide(times[finite], SECOND_NS)
    seconds, first = np.unique(seconds, return_index=True)
    if not len(seconds):
        return SecondsSeries(first_s=0, values=np.full(0, np.nan))
    kept = values[finite][first]
    index = pd.DatetimeIndex(seconds * SECOND_NS)
    spacing = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 1.0
    min_periods = max(_MEDIAN_MIN_READINGS, int(_MEDIAN_FILL * running_median_s / spacing))
    median = (
        pd.Series(kept, index=index)
        .rolling(f"{running_median_s}s", center=True, min_periods=min_periods)
        .median()
        .to_numpy()
    )
    grid = np.full(int(seconds[-1] - seconds[0]) + 1, np.nan)
    grid[seconds - seconds[0]] = kept - median
    return SecondsSeries(first_s=int(seconds[0]), values=grid)


def pooled_lag(
    pairs: Sequence[tuple[SecondsSeries, SecondsSeries]],
    first_s: int,
    stop_s: int,
    *,
    search_s: int,
    min_pairs: int,
    min_r: float,
    tolerance_s: float,
) -> SegmentLag:
    """Return the best lag of one segment for one tier of tracer pairs.

    Parameters
    ----------
    pairs : Sequence of (SecondsSeries, SecondsSeries)
        Each tracer pair: the instrument's series, then the partner's.
    first_s, stop_s : int
        The segment, ``[first_s, stop_s)`` in epoch seconds of the instrument's
        series.
    search_s : int
        The lag is searched in ``[-search_s, search_s]``.
    min_pairs : int
        Paired seconds a lag needs before its correlation counts.
    min_r : float
        The correlation at which a pair is decided on its own.
    tolerance_s : float
        How far a pair decided on its own may put its best lag from the pooled
        one before the tier's pairs disagree.

    Returns
    -------
    SegmentLag
        The best lag; NaN lag and r when no lag of any pair had enough pairs.
    """
    lags = np.arange(-search_s, search_s + 1)
    curves, counts = [], []
    for own, partner in pairs:
        curve, count = _curve(
            own.window(first_s, stop_s),
            partner.window(first_s - search_s, stop_s + search_s),
            lags,
            search_s,
            min_pairs,
        )
        curves.append(curve)
        counts.append(count)
    stacked = np.vstack(curves)
    usable = np.isfinite(stacked).any(axis=0)
    if not usable.any():
        return SegmentLag(lag_s=float("nan"), r=float("nan"), n_pairs=0, at_edge=False)
    pooled = np.full(len(lags), -np.inf)
    pooled[usable] = np.nanmean(stacked[:, usable], axis=0)
    best = int(np.argmax(pooled))
    n = int(sum(count[best] for count in counts))
    disagree = False
    for curve in curves:
        if np.isfinite(curve).any():
            peak = int(np.nanargmax(curve))
            disagree |= bool(
                curve[peak] >= min_r and abs(int(lags[peak] - lags[best])) > tolerance_s
            )
    return SegmentLag(
        lag_s=float(lags[best]),
        r=float(pooled[best]),
        n_pairs=n,
        at_edge=abs(int(lags[best])) == search_s,
        pairs_disagree=disagree,
    )


def _curve(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    lags: npt.NDArray[np.int64],
    search_s: int,
    min_pairs: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return r and the paired seconds at every lag: ``a(t)`` against ``b(t - L)``.

    ``b`` starts ``search_s`` seconds before ``a``, so ``b(t - L)`` for the
    segment's seconds is the slice beginning ``search_s - L`` in.
    """
    r = np.full(len(lags), np.nan)
    count = np.zeros(len(lags), dtype=np.int64)
    a_ok = np.isfinite(a)
    for i, lag in enumerate(lags):
        start = search_s - int(lag)
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
