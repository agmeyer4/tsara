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
* the background's **wander**: how different the enhancement is at two times
  a lag apart on the pair cells outside every event of either gas at the
  window and quantile (its semivariogram), less the readings' noise, which
  adds its variance at every lag; and for the pair the cross-semivariogram,
  since on real air two gases' backgrounds wander together and part of the
  error cancels in the residual.

The wander is measured at :data:`LAGS_S` and never used as measured. Read
from the fragments of a record between its events, a measured curve is
ragged (on a real 2026 drive the typical change in methane's, sqrt(2 g),
rose to 4.5 ppb by 20 s, fell to 2.1 by 40 s and rose again), and a ragged
curve can describe an error no background could have, one that changes more
over a gap than two half-gaps allow, which makes a member's slope variance
from it negative. So the three curves are fitted, per window and quantile,
by one model that is valid by construction (§5.4, "The wander as a fitted model"):

    g_x(L) = n_x + s_x (1 - exp(-L / tau)),  g_y and g_xy likewise, L > 0,

one decorrelation time ``tau`` shared by the two gases, each gas's nugget
``n`` and sill ``s`` at least zero, and the cross's within what the two
gases' allow (``|n_xy| <= sqrt(n_x n_y)``, ``|s_xy| <= sqrt(s_x s_y)``).
For a pair the residual's error is ``delta_y - b delta_x``, whose curve is
then ``(n_y + b^2 n_x - 2 b n_xy) + (s_y + b^2 s_x - 2 b s_xy)(1 - exp(-L / tau))``,
both coefficients at least zero for every ``b``: the curve of a white part and
an exponentially correlated part, a valid error whatever the slope. The slow
covariance between two points ``dt`` apart is

    S(dt) = (w_y^2 + b^2 w_x^2) max(0, 1 - |dt| / W) - g_r(|dt|),   g_r(0) = 0.

This module measures and fits the wander on a paired product and evaluates
``S``; the fits do the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import nnls

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

#: The decorrelation times the model is fitted at: 80 steps evenly spaced in
#: log from 0.5 s to ten baseline windows, the one with the least squared
#: error kept. Past ten windows the exponential is a straight line over every
#: lag a member at the window can span, so longer times add nothing a fit
#: could tell apart.
_TIMESCALE_STEPS = 80
_TIMESCALE_MIN_S = 0.5
_TIMESCALE_MAX_WINDOWS = 10.0

#: Fewer read lags than this within the window, and the model is fitted to
#: every read lag instead: two parameters per curve need more than two points.
_MIN_FIT_LAGS = 3

#: The order of the three curves along the last axis of the model's arrays.
_REFERENCE, _NUMERATOR, _CROSS = 0, 1, 2


@dataclass(frozen=True)
class Wander:
    """The two gases' wander, per sweep point: the fitted model and what it was fitted to.

    Attributes
    ----------
    nugget, sill : numpy.ndarray
        ``(window, quantile, 3)``: the model's white part and its exponential
        part's height for the reference, the numerator and the cross, in the
        gases' units squared (the cross in their product); zero at a sweep
        point with no lag read.
    timescale : numpy.ndarray
        ``(window, quantile)``: the exponential part's decorrelation time (s),
        shared by the three curves; NaN at a sweep point with no lag read.
    measured : numpy.ndarray
        ``(window, quantile, 3, lag)``: the three semivariograms as measured on
        :data:`LAGS_S`, the readings' noise removed; NaN at a lag not read.
        Kept to be looked at beside the model, never used.
    described : numpy.ndarray of bool
        ``(window, quantile)``: whether any lag was read.
    """

    nugget: npt.NDArray[np.float64]
    sill: npt.NDArray[np.float64]
    timescale: npt.NDArray[np.float64]
    measured: npt.NDArray[np.float64]
    described: npt.NDArray[np.bool_]

    def curves(self, lags_s: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the model's three curves at ``lags_s``, ``(window, quantile, 3, lag)``.

        Each is zero at a lag of zero, where a semivariogram is zero by definition.
        """
        lags = np.abs(np.asarray(lags_s, dtype=np.float64))
        shape = -np.expm1(-lags / self.timescale[..., None, None])
        values = self.nugget[..., None] + self.sill[..., None] * np.nan_to_num(shape)
        return np.where(lags > 0, values, 0.0)

    def residual(
        self, sweep: npt.NDArray[np.intp], slope: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the residual ``delta_y - b delta_x``'s nugget, sill and timescale, per entry.

        ``sweep`` is each entry's sweep point as ``window * n_quantiles +
        quantile``, ``slope`` its ``b``. Both coefficients are at least zero
        for every ``b`` by the model's bounds; the floor at zero only keeps
        rounding from turning an exact zero negative, and a slope that is NaN
        (a fit that failed) gives NaN.
        """
        nugget = self.nugget.reshape(-1, 3)[sweep]
        sill = self.sill.reshape(-1, 3)[sweep]

        def combined(part: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.maximum(
                0.0,
                part[:, _NUMERATOR]
                + slope * slope * part[:, _REFERENCE]
                - 2.0 * slope * part[:, _CROSS],
            )

        return combined(nugget), combined(sill), self.timescale.reshape(-1)[sweep]


def wander(
    pairs: xr.Dataset,
    catalog: pd.DataFrame,
    deltas: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    sigmas: tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None],
    midpoints: npt.NDArray[np.int64],
    variables: tuple[str, str],
) -> Wander:
    """Measure the two gases' wander outside every event of either, and fit its model.

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
        without them the noise cannot be removed, and no lag is read.
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
    nugget = np.zeros((windows.size, quantiles.size, 3))
    sill = np.zeros((windows.size, quantiles.size, 3))
    timescale = np.full((windows.size, quantiles.size), np.nan)
    measured = np.full((windows.size, quantiles.size, 3, LAGS_S.size), np.nan)
    dy, dx = deltas
    sy, sx = sigmas
    if sx is not None and sy is not None:
        seconds = midpoints.astype(np.float64) / NS_PER_S
        for i, window in enumerate(windows):
            for j, quantile in enumerate(quantiles):
                inside = _inside(catalog, variables, window, quantile, midpoints)
                ok = ~inside & np.isfinite(dx[:, i, j]) & np.isfinite(dy[:, i, j])
                ok &= np.isfinite(sx) & np.isfinite(sy)
                measured[i, j] = _semivariograms(
                    seconds[ok], (dx[ok, i, j], dy[ok, i, j]), (sx[ok], sy[ok])
                )
                if np.isfinite(measured[i, j, _REFERENCE]).any():
                    nugget[i, j], sill[i, j], timescale[i, j] = _fit_model(measured[i, j], window)
    return Wander(
        nugget=nugget,
        sill=sill,
        timescale=timescale,
        measured=measured,
        described=np.isfinite(timescale),
    )


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
    nugget, sill, timescale = wandering.residual(sweep, slope)
    lag = np.abs(dt_s)
    # A sweep point with no lag read has no timescale and a zero sill: no wander.
    shape = np.where(sill > 0, -np.expm1(-lag / np.where(sill > 0, timescale, 1.0)), 0.0)
    wander_part = np.where(lag > 0, nugget + sill * shape, 0.0)
    wobble_part = amplitude * np.fmax(0.0, 1.0 - lag / window_s)
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


def _fit_model(
    measured: npt.NDArray[np.float64], window: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Fit the three measured curves by the model, over the read lags up to the window.

    At each trial decorrelation time the model is linear in its nugget and
    sill: each gas's pair is the least-squares fit with both at least zero, the
    cross's the plain least-squares fit then held within what the two gases'
    allow, so that the model stays valid. The trial with the least summed
    squared error is kept, each curve's error divided by its own mean square
    so that methane does not drown ethane. Only the lags a member at this
    window can span are fitted (a longer member is the crowding gate's,
    METHODS §5.3), unless fewer than :data:`_MIN_FIT_LAGS` of them were read.

    Returns
    -------
    nugget, sill : numpy.ndarray
        ``(3,)``: reference, numerator, cross.
    timescale : float
        Seconds.
    """
    read = np.isfinite(measured[_REFERENCE])
    use = read & (LAGS_S <= window)
    if use.sum() < _MIN_FIT_LAGS:
        use = read
    lags = LAGS_S[use]
    curves = measured[:, use]
    scales = (
        float(np.mean(curves[_REFERENCE] ** 2)),
        float(np.mean(curves[_NUMERATOR] ** 2)),
        float(np.mean(np.abs(curves[_REFERENCE] * curves[_NUMERATOR]))),
    )
    best_error = np.inf
    best: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float] = (
        np.zeros(3),
        np.zeros(3),
        np.nan,
    )
    for timescale in np.geomspace(
        _TIMESCALE_MIN_S, _TIMESCALE_MAX_WINDOWS * window, _TIMESCALE_STEPS
    ):
        design = np.column_stack([np.ones(lags.size), -np.expm1(-lags / timescale)])
        nugget, sill = np.zeros(3), np.zeros(3)
        for part in (_REFERENCE, _NUMERATOR):
            (nugget[part], sill[part]), _ = nnls(design, curves[part])
        (nugget[_CROSS], sill[_CROSS]), *_ = np.linalg.lstsq(design, curves[_CROSS], rcond=None)
        for coefficient in (nugget, sill):
            bound = np.sqrt(coefficient[_REFERENCE] * coefficient[_NUMERATOR])
            coefficient[_CROSS] = np.clip(coefficient[_CROSS], -bound, bound)
        error = sum(
            float(np.sum((curves[part] - design @ (nugget[part], sill[part])) ** 2)) / scale
            for part, scale in enumerate(scales)
            if scale > 0
        )
        if error < best_error:
            best_error, best = error, (nugget, sill, float(timescale))
    return best
