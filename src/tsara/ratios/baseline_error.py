"""The baseline's error across a member, as a covariance between its points (METHODS §5.4).

An enhancement is a reading minus a baseline, and the baseline has an error of
its own. The readings' sigma does not hold it, and a constant part of it is the
intercept's, which York fits; what moves a slope is how the error *changes*
across a member. So it is written as a covariance between a member's points
that depends only on how far apart in time they are, and the fits carry it
into the slope's sigma and the MSWD a single source is expected to read by the
same algebra as the readings' own persistence (``fits``). Two parts, per gas,
read from the data alone (``docs/METHODS.md`` §5.4, "The baseline's error,
entered"):

* the baseline's **wobble**: variance ``max(0, c^2 - sigma^2)``, ``c`` the
  events stage's clean spread and ``sigma`` the reading's random sigma,
  correlated as two baselines share their windows,
  ``max(0, 1 - |t_i - t_j| / W)``;
* the background's **wander**: the semivariogram of the enhancement on the
  pair cells outside every event of either gas at the window and quantile,
  less the readings' noise, which adds its
  variance at every lag; and for the pair the cross-semivariogram, since on
  real air two gases' backgrounds wander together and part of the error
  cancels in the residual.

Each gas's wander, as measured, is then made non-decreasing in lag (its
least-squares non-decreasing fit), and the cross-semivariogram is bounded by
the two (``|g_xy| <= sqrt(g_x g_y)``). Two errors cannot be more alike at a
longer gap than at a shorter one, but a semivariogram read from the fragments
of a record between its events can dip: on a real 2026 drive it rose to 4.5 ppb
of methane by 20 s, fell to 2.1 by 40 s and rose again. Used as measured, a
dip can make the slope's variance from the baseline's error negative for a
member, which the fits floor at zero; on real air with known plumes added
that left a fifth of the good members with no allowance at all (§5.4).

For a pair the residual's error is ``delta_y - b delta_x``, so the slow
covariance between two points ``dt`` apart is

    S(dt) = (w_y^2 + b^2 w_x^2) max(0, 1 - |dt| / W)
            - max(0, g_y + b^2 g_x - 2 b g_xy)(|dt|),

the semivariograms ``g`` read on :data:`LAGS_S`, in log lag between them and
held beyond their ends. This module measures the wander on a paired product
and evaluates ``S``; the fits do the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.naming import QUANTILE_DIM, TIME_COORD, WINDOW_DIM
from tsara.core.timebase import NS_PER_S

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import xarray as xr

__all__ = ["LAGS_S", "MIN_LAG_PAIRS", "Wander", "slow_covariance", "wander", "wobble"]

#: The lags (s) the wander's semivariograms are measured at: 40 steps evenly
#: spaced in log from 1 s to 2 h, past any member's span.
LAGS_S: npt.NDArray[np.float64] = np.geomspace(1.0, 7200.0, 40)

#: Fewer pairs of out-of-event cells than this at a lag, and the
#: semivariogram there is not read: a lag the record cannot describe.
MIN_LAG_PAIRS = 30


@dataclass(frozen=True)
class Wander:
    """The two gases' wander, per sweep point: semivariograms on :data:`LAGS_S`.

    Attributes
    ----------
    reference, numerator, cross : numpy.ndarray
        ``(window, quantile, lag)``, in the gases' units squared (``cross`` in
        their product), the readings' noise removed; at a lag the record could
        not describe, the nearest described lag's value; each gas's
        non-decreasing in lag, the cross within ``+/- sqrt(reference x
        numerator)``; zero throughout at a sweep point with no lag described.
    described : numpy.ndarray of bool
        ``(window, quantile)``: whether any lag was.
    """

    reference: npt.NDArray[np.float64]
    numerator: npt.NDArray[np.float64]
    cross: npt.NDArray[np.float64]
    described: npt.NDArray[np.bool_]


def wander(
    pairs: xr.Dataset,
    catalog: pd.DataFrame,
    deltas: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    sigmas: tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None],
    midpoints: npt.NDArray[np.int64],
    variables: tuple[str, str],
) -> Wander:
    """Measure the two gases' wander on the pair cells outside every event of either.

    Parameters
    ----------
    pairs : xarray.Dataset
        The paired product; its sweep coordinates order the result.
    catalog : pandas.DataFrame
        The event catalog: the two variables' events say which cells are inside.
    deltas : tuple of numpy.ndarray
        The numerator's and the reference's enhancements, ``(cell, window, quantile)``.
    sigmas : tuple of numpy.ndarray or None
        The numerator's and the reference's readings' random sigma per cell;
        without them the noise cannot be removed, and no lag is described.
    midpoints : numpy.ndarray of int64
        The pair cells' midpoints (ns), in time order.
    variables : tuple of str
        The numerator and the reference, as ``'<instrument>.<variable>'``.

    Returns
    -------
    Wander
    """
    windows = np.asarray(pairs[WINDOW_DIM].values, dtype=np.float64)
    quantiles = np.asarray(pairs[QUANTILE_DIM].values, dtype=np.float64)
    shape = (windows.size, quantiles.size, LAGS_S.size)
    tables = [np.zeros(shape) for _ in range(3)]
    described = np.zeros(shape[:2], dtype=bool)
    dy, dx = deltas
    sy, sx = sigmas
    if sx is None or sy is None:
        return Wander(
            reference=tables[0], numerator=tables[1], cross=tables[2], described=described
        )
    seconds = midpoints.astype(np.float64) / NS_PER_S
    for i, window in enumerate(windows):
        for j, quantile in enumerate(quantiles):
            inside = _inside(catalog, variables, window, quantile, midpoints)
            ok = ~inside & np.isfinite(dx[:, i, j]) & np.isfinite(dy[:, i, j])
            ok &= np.isfinite(sx) & np.isfinite(sy)
            measured = _semivariograms(seconds[ok], (dx[ok, i, j], dy[ok, i, j]), (sx[ok], sy[ok]))
            for table, values in zip(tables, measured, strict=True):
                if np.isfinite(values).any():
                    table[i, j] = _filled(values)
            tables[0][i, j], tables[1][i, j], tables[2][i, j] = _monotone(
                tables[0][i, j], tables[1][i, j], tables[2][i, j]
            )
            described[i, j] = bool(np.isfinite(measured[0]).any())
    return Wander(reference=tables[0], numerator=tables[1], cross=tables[2], described=described)


def wobble(
    pairs: xr.Dataset, column: str | None, sigma: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.float64] | None:
    """Return the baseline's wobble variance per pair cell, ``(cell, window, quantile)``.

    ``max(0, c^2 - sigma^2)``, ``c`` the clean spread carried onto the pair
    cells; None when no clean spread was carried or no sigma is declared.
    """
    if column is None or sigma is None:
        return None
    spread = np.asarray(
        pairs[column].transpose(TIME_COORD, WINDOW_DIM, QUANTILE_DIM).values, dtype=np.float64
    )
    # np.maximum, not fmax: a cell with no clean spread has an unknown wobble, not none.
    return np.maximum(0.0, spread**2 - sigma[:, None, None] ** 2)


def slow_covariance(
    dt_s: npt.NDArray[np.float64],
    amplitude: npt.NDArray[np.float64],
    window_s: npt.NDArray[np.float64],
    slope: npt.NDArray[np.float64],
    sweep: npt.NDArray[np.intp],
    wandering: Wander,
) -> npt.NDArray[np.float64]:
    """Return S(dt) for pairs of points, each pair with its member's parameters.

    Parameters
    ----------
    dt_s : numpy.ndarray
        The two points' separation (s), one per pair.
    amplitude : numpy.ndarray
        The member's ``w_y^2 + b^2 w_x^2`` (0 where no wobble is known).
    window_s : numpy.ndarray
        The member's baseline window (s).
    slope : numpy.ndarray
        The member's slope ``b``.
    sweep : numpy.ndarray of intp
        The member's sweep point, as ``window * n_quantiles + quantile``.
    wandering : Wander
        The two gases' wander at every sweep point.
    """
    gx, gy, gxy = (
        _at(table.reshape(-1, LAGS_S.size), sweep, dt_s)
        for table in (wandering.reference, wandering.numerator, wandering.cross)
    )
    wander_part = np.fmax(0.0, gy + slope * slope * gx - 2.0 * slope * gxy)
    wobble_part = amplitude * np.fmax(0.0, 1.0 - np.abs(dt_s) / window_s)
    return wobble_part - wander_part


def _inside(
    catalog: pd.DataFrame,
    variables: tuple[str, str],
    window: float,
    quantile: float,
    midpoints: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """Return which pair cells lie in an event of either variable at this window and quantile.

    Every entry and exit is read, which is the widest events' union: a higher
    entry keeps the same intervals, and a higher exit's lie inside a lower's
    (§6.8, "Candidates").
    """
    count = np.zeros(midpoints.size + 1, dtype=np.int64)
    for name in variables:
        instrument, _, variable = name.partition(".")
        rows = catalog[
            (catalog["instrument"] == instrument)
            & (catalog["species"] == variable)
            & (catalog["baseline_window"] == window)
            & (catalog["baseline_quantile"] == quantile)
        ]
        starts = pd.to_datetime(rows["start_time"]).to_numpy("datetime64[ns]").astype(np.int64)
        ends = pd.to_datetime(rows["end_time"]).to_numpy("datetime64[ns]").astype(np.int64)
        np.add.at(count, np.searchsorted(midpoints, starts, side="left"), 1)
        np.add.at(count, np.searchsorted(midpoints, ends, side="left"), -1)
    return np.cumsum(count)[:-1] > 0


def _semivariograms(
    seconds: npt.NDArray[np.float64],
    values: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    sigmas: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the x, y and cross semivariograms on :data:`LAGS_S`, the noise removed.

    Each cell is paired with the cell nearest ``lag`` later, accepted within
    half the median spacing plus 2 % of the lag, so a pair clock with holes
    still has its lags; a lag below half the spacing has no pairs.
    """
    gx, gy, gxy = (np.full(LAGS_S.size, np.nan) for _ in range(3))
    if seconds.size < 2:
        return gx, gy, gxy
    (x, y), (sx, sy) = values, sigmas
    spacing = float(np.median(np.diff(seconds)))
    for k, lag in enumerate(LAGS_S):
        target = seconds + lag
        later = np.clip(np.searchsorted(seconds, target), 1, seconds.size - 1)
        nearer = np.where(
            np.abs(seconds[later - 1] - target) < np.abs(seconds[later] - target), later - 1, later
        )
        keep = (np.abs(seconds[nearer] - target) <= 0.5 * spacing + 0.02 * lag) & (
            nearer > np.arange(seconds.size)
        )
        if keep.sum() < MIN_LAG_PAIRS:
            continue
        a, b = np.flatnonzero(keep), nearer[keep]
        noise_x = 0.5 * np.mean(sx[a] ** 2 + sx[b] ** 2)
        noise_y = 0.5 * np.mean(sy[a] ** 2 + sy[b] ** 2)
        ddx, ddy = x[b] - x[a], y[b] - y[a]
        gx[k] = max(0.0, 0.5 * float(np.mean(ddx * ddx)) - noise_x)
        gy[k] = max(0.0, 0.5 * float(np.mean(ddy * ddy)) - noise_y)
        gxy[k] = 0.5 * float(np.mean(ddx * ddy))
    return gx, gy, gxy


def _filled(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Fill the lags not described from the described ones, in log lag, held at the ends."""
    known = np.isfinite(values)
    logs = np.log(LAGS_S)
    return np.interp(logs, logs[known], values[known])


def _monotone(
    reference: npt.NDArray[np.float64],
    numerator: npt.NDArray[np.float64],
    cross: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Make each gas's semivariogram non-decreasing in lag and bound the cross by the two.

    Each gas's is replaced by its least-squares non-decreasing fit over the
    lags (pool-adjacent-violators, every lag weighted alike); the cross is
    clipped to ``+/- sqrt(reference x numerator)``, what two gases' errors
    can share at most (Cauchy-Schwarz).
    """
    reference, numerator = _non_decreasing(reference), _non_decreasing(numerator)
    bound = np.sqrt(reference * numerator)
    return reference, numerator, np.clip(cross, -bound, bound)


def _non_decreasing(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the least-squares non-decreasing fit to ``values`` (pool-adjacent-violators).

    Walking along the lags, a value below the block before it is pooled into
    that block, which takes the mean of its values, and pooling repeats
    backwards until the blocks rise.
    """
    sums: list[float] = []
    counts: list[int] = []
    for value in values:
        sums.append(float(value))
        counts.append(1)
        while len(sums) > 1 and sums[-2] * counts[-1] > sums[-1] * counts[-2]:
            # Pop first: in `sums[-2] += sums.pop()` the -2 would be read after the pop.
            last_sum, last_count = sums.pop(), counts.pop()
            sums[-1] += last_sum
            counts[-1] += last_count
    return np.repeat(np.asarray(sums) / np.asarray(counts), counts)


def _at(
    table: npt.NDArray[np.float64], sweep: npt.NDArray[np.intp], dt_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Read each pair's semivariogram, at its sweep point's row of ``table``, at its |dt|."""
    logs = np.log(LAGS_S)
    where = np.log(np.clip(np.abs(dt_s), LAGS_S[0], LAGS_S[-1]))
    upper = np.clip(np.searchsorted(logs, where), 1, LAGS_S.size - 1)
    lower = upper - 1
    share = (where - logs[lower]) / (logs[upper] - logs[lower])
    low, high = table[sweep, lower], table[sweep, upper]
    return np.where(dt_s == 0, 0.0, low + share * (high - low))
