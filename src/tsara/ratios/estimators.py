"""Straight lines through many groups of points at once: York's line and least squares.

An enhancement ratio is the slope of one gas's enhancement against another's
over a stretch of readings (``docs/METHODS.md`` §4). Both axes carry
measurement error, so the preferred line is York's (York et al. 2004): the
maximum-likelihood straight line given every point's sigma on both axes and,
where it is known, the correlation between the two errors. Ordinary least
squares is kept beside it for its familiar diagnostics (§4.1), and because
it is the only line there is when no sigma has been declared.

**Many groups at once.** Every row of the input belongs to the fit its group
label names, and a label below zero belongs to none. A reading may appear in
several rows, so fits may overlap (windows of several widths centred on one
event) and may have holes (a parent event with its children's readings left
out), and the events stage's per-reading event index, which is -1 outside any
event, can be passed as the labels directly. Every group is fitted on its own
rows only, in the order they arrive, and a group stops iterating the moment
it converges, so a group fitted beside thousands of others gets exactly the
numbers it gets alone.

**What a fit cannot do, it says.** Per group, :class:`FitStatus` records why
a fit holds no line: fewer than :data:`MIN_FIT_POINTS` points, no spread in
x, a point without a declared sigma (York only: it cannot weigh a point it was
not told the error of, and it does not guess, §2.3), or no convergence. The
line of such a group is NaN, never a fallback. A call that is wrong in itself
(mismatched shapes, a negative sigma, a correlation outside (-1, 1)) raises
:class:`TsaraRatioError` instead.

**What York reports, and what it leaves to the caller.** The slope's sigma is
York's analytic one, computed from the declared sigmas; the goodness of fit
is reported separately as the MSWD, the scatter about the line divided by
what the sigmas predict (1 when they explain it). Whether the slope's sigma
is to be widened by the MSWD is a decision about the ratio's combined
uncertainty (§5), not about the line, so it is left to the stage that makes
it: with the slope's sigma multiplied by the square root of the MSWD, a
common scale on every declared sigma cancels, and only their ratio across
the two axes matters (§4.2).

The estimators are registered by name, like the file readers, the baseline
methods and the clean-level estimators, and ``RegressionConfig.methods``
names them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

from tsara.core.exceptions import TsaraError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = [
    "FitStatus",
    "LineFit",
    "LineFitter",
    "MIN_FIT_POINTS",
    "TsaraRatioError",
    "available_line_fits",
    "get_line_fit",
    "ols",
    "register_line_fit",
    "york",
]

# ---------------------------------------------------------------------------
# Constants, types and the error class
# ---------------------------------------------------------------------------

#: The fewest points a line is fitted through. Two fix a line exactly; the
#: goodness of fit and least squares' own sigma both need at least one
#: degree of freedom beyond them.
MIN_FIT_POINTS = 3

#: York stops when the slope moves by less than this fraction of the larger
#: of its own size and the data's natural slope scale (see :func:`york`).
DEFAULT_RTOL = 1e-12

#: York gives up on a group after this many iterations. Most groups settle
#: in about ten; a group whose x noise is as large as its x spread starts
#: far from its answer (least squares reads it near zero) and creeps there
#: over a hundred or more (METHODS §4.2). A step touches only the rows of
#: groups still moving, so a generous limit costs only those groups.
DEFAULT_MAX_ITER = 1000


class FitStatus(IntEnum):
    """Why a fit holds a line, or why it does not; one per group.

    Attributes
    ----------
    OK
        A line was fitted.
    TOO_FEW_POINTS
        Fewer than :data:`MIN_FIT_POINTS` points with finite x and y.
    NO_SPREAD
        Every x is the same, so the line would be vertical.
    MISSING_SIGMA
        York only: a point with finite x and y but no finite sigma on an
        axis. York cannot weigh it, and does not guess.
    NOT_CONVERGED
        York only: the slope still moved after the iteration limit.
    """

    OK = 0
    TOO_FEW_POINTS = 1
    NO_SPREAD = 2
    MISSING_SIGMA = 3
    NOT_CONVERGED = 4


class TsaraRatioError(TsaraError):
    """Raised when a line cannot be fitted as asked.

    Its own type because the failures are about the call itself (rows,
    labels, sigmas and correlations that do not fit together) rather than
    about reading, joining, rolling or finding events. A group that merely
    holds too little to fit is not an error: its :class:`FitStatus` says so.
    """


@dataclass(frozen=True, eq=False)
class LineFit:
    """Straight lines fitted to many groups of points, one entry per group.

    Every array is shaped ``(n_groups,)``. A group whose ``status`` is not
    :attr:`FitStatus.OK` holds NaN in every float field.

    Attributes
    ----------
    method : str
        The registered name of the estimator that fitted the lines.
    slope, intercept : numpy.ndarray of float64
        The line ``y = intercept + slope * x``.
    slope_sigma, intercept_sigma : numpy.ndarray of float64
        Their standard errors: for York, from the declared sigmas (§4.2);
        for least squares, from the scatter about the line (§4.1).
    mswd : numpy.ndarray of float64
        York's goodness of fit, the weighted squared residuals over ``n - 2``:
        about 1 when the declared sigmas explain the scatter, far above it
        when something else (a second source, an understated sigma) adds
        scatter. NaN for least squares, which is told no sigmas.
    r_squared : numpy.ndarray of float64
        The squared correlation of the group's points, the same for every
        estimator. Reported because it is familiar, not because it judges a
        ratio: a single source whose ratio is 0 has an r² near 0 (§4.1).
    n : numpy.ndarray of int64
        Points the fit used: rows of the group with finite x and y.
    n_dropped : numpy.ndarray of int64
        Rows of the group dropped because x or y was not finite.
    iterations : numpy.ndarray of int64
        York's iterations until the slope stopped moving; 0 for least
        squares and for a group that was not fitted.
    status : numpy.ndarray of int8
        A :class:`FitStatus` value per group.
    """

    method: str
    slope: npt.NDArray[np.float64]
    intercept: npt.NDArray[np.float64]
    slope_sigma: npt.NDArray[np.float64]
    intercept_sigma: npt.NDArray[np.float64]
    mswd: npt.NDArray[np.float64]
    r_squared: npt.NDArray[np.float64]
    n: npt.NDArray[np.int64]
    n_dropped: npt.NDArray[np.int64]
    iterations: npt.NDArray[np.int64]
    status: npt.NDArray[np.int8]

    @property
    def ok(self) -> npt.NDArray[np.bool_]:
        """Per group, whether a line was fitted."""
        return np.asarray(self.status == FitStatus.OK)


class LineFitter(Protocol):
    """What a registered estimator is: rows and labels in, one line per group out."""

    def __call__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        sigma_x: npt.ArrayLike | None = None,
        sigma_y: npt.ArrayLike | None = None,
        group: npt.ArrayLike | None = None,
        *,
        n_groups: int | None = None,
    ) -> LineFit:
        """Fit one line per group."""


#: Registered estimators keyed by ``RegressionConfig.methods`` entries.
#: Module-private: mutation goes through :func:`register_line_fit`.
_ESTIMATORS: dict[str, LineFitter] = {}

#: Bound so the decorator returns the same function type it received.
F = TypeVar("F", bound=LineFitter)


@dataclass(frozen=True, eq=False)
class _Rows:
    """The rows a call fits: those in a group with finite x and y, in order."""

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    sigma_x: npt.NDArray[np.float64]
    sigma_y: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]
    group: npt.NDArray[np.intp]
    n_groups: int
    n_dropped: npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class _Moments:
    """Per group: the count, the means and the centred sums of the points."""

    n: npt.NDArray[np.int64]
    x_mean: npt.NDArray[np.float64]
    y_mean: npt.NDArray[np.float64]
    sxx: npt.NDArray[np.float64]
    syy: npt.NDArray[np.float64]
    sxy: npt.NDArray[np.float64]
    x_spread: npt.NDArray[np.bool_]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def register_line_fit(name: str, *, replace: bool = False) -> Callable[[F], F]:
    """Register a line estimator under its configuration name.

    Used as a decorator, as the baseline methods and clean-level estimators
    are; re-registering a name is refused unless ``replace=True`` says so,
    since a silent overwrite would make a ratio depend on import order.

    Parameters
    ----------
    name : str
        The ``RegressionConfig.methods`` entry.
    replace : bool, optional
        Replace an estimator already registered under ``name``.

    Returns
    -------
    callable
        Decorator returning its argument unchanged.

    Raises
    ------
    ValueError
        If ``name`` is blank, or already registered and ``replace`` is False.
    """
    if not name or not name.strip():
        raise ValueError("Line estimator name must be a non-empty string.")

    def decorator(func: F) -> F:
        if name in _ESTIMATORS and not replace:
            raise ValueError(
                f"A line estimator is already registered as '{name}'. Pass "
                "replace=True if you meant to override it."
            )
        if name in _ESTIMATORS:
            logger.warning("Replacing line estimator '%s'.", name)
        _ESTIMATORS[name] = func
        return func

    return decorator


def available_line_fits() -> tuple[str, ...]:
    """List the registered estimator names, sorted.

    Returns
    -------
    tuple of str
        e.g. ``('ols', 'york')``.
    """
    return tuple(sorted(_ESTIMATORS))


def get_line_fit(name: str) -> LineFitter:
    """Look up a line estimator by name.

    Parameters
    ----------
    name : str
        The ``RegressionConfig.methods`` entry.

    Returns
    -------
    callable
        The registered estimator.

    Raises
    ------
    TsaraRatioError
        If nothing is registered under ``name``; the message lists what is.
    """
    try:
        return _ESTIMATORS[name]
    except KeyError:
        raise TsaraRatioError(
            f"No line estimator registered as '{name}'. Available: {list(available_line_fits())}."
        ) from None


# ---------------------------------------------------------------------------
# York
# ---------------------------------------------------------------------------


@register_line_fit("york")
def york(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    sigma_x: npt.ArrayLike | None = None,
    sigma_y: npt.ArrayLike | None = None,
    group: npt.ArrayLike | None = None,
    *,
    n_groups: int | None = None,
    r: npt.ArrayLike = 0.0,
    rtol: float = DEFAULT_RTOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> LineFit:
    """Fit York's straight line to every group (York et al. 2004; METHODS §4.2).

    The maximum-likelihood line through points with errors on both axes:
    each point is weighted by ``W = 1 / (σy² + b²σx² − 2·b·r·σx·σy)``, the
    inverse of the variance of its distance from the line measured along y,
    and the slope
    ``b`` is iterated until it stops moving, starting from least squares.
    The weight is written in variances rather than York's ``ω = 1/σ²`` so
    that a sigma of exactly zero on one axis is allowed: with ``σx = 0``
    everywhere the line is weighted least squares of y on x, exactly.

    Parameters
    ----------
    x, y : array_like of float, shape (n_rows,)
        The points. A row with x or y not finite is dropped and counted.
    sigma_x, sigma_y : array_like of float, optional
        Each point's standard error on each axis, broadcast to the rows: a
        scalar declares one precision for every point. ``None``, or a NaN on
        a row that has finite x and y, means not declared, and that row's
        group is reported :attr:`FitStatus.MISSING_SIGMA` with no line.
    group : array_like of int, shape (n_rows,), optional
        The fit each row belongs to, numbered from 0; below 0 belongs to
        none. ``None`` puts every row in one fit.
    n_groups : int, optional
        How many fits there are, when some numbers hold no rows; by default
        one more than the largest label.
    r : array_like of float, optional
        The correlation between a point's x and y errors, broadcast to the
        rows, strictly between -1 and 1. 0, the default, when they are
        independent.
    rtol : float, optional
        The slope has converged when it moves by no more than ``rtol``
        times the larger of its own size and the group's natural slope
        scale (the spread of y, plus the typical σy, over the spread of x),
        so that a slope of 0 converges too.
    max_iter : int, optional
        Iterations after which a group still moving is reported
        :attr:`FitStatus.NOT_CONVERGED`.

    Returns
    -------
    LineFit
        One line per group, with York's analytic standard errors and the
        MSWD.

    Raises
    ------
    TsaraRatioError
        If the rows, labels, sigmas or correlations do not fit together, a
        sigma is negative, both sigmas of a point are zero, a correlation is
        not strictly between -1 and 1, or ``rtol``/``max_iter`` is not
        positive.
    """
    if not rtol > 0:
        raise TsaraRatioError(f"rtol must be positive, got {rtol}.")
    if max_iter < 1:
        raise TsaraRatioError(f"max_iter must be at least 1, got {max_iter}.")
    rows = _gather(x, y, group, n_groups, sigma_x=sigma_x, sigma_y=sigma_y, r=r)
    moments = _moments(rows)
    status = _shape_status(moments)

    # A point the caller did not declare a sigma for cannot be weighed: its
    # whole group gets no line rather than a line that ignores it.
    undeclared = ~(np.isfinite(rows.sigma_x) & np.isfinite(rows.sigma_y))
    missing = np.bincount(rows.group[undeclared], minlength=rows.n_groups) > 0
    status[(status == FitStatus.OK) & missing] = FitStatus.MISSING_SIGMA

    # From here on only the rows of groups that can be fitted take part. The
    # filter keeps their order, so every group's sums are the sums it gets
    # when fitted alone.
    fitted = status == FitStatus.OK
    keep = fitted[rows.group]
    lab = rows.group[keep]
    gx, gy = rows.x[keep], rows.y[keep]
    vx, vy = rows.sigma_x[keep] ** 2, rows.sigma_y[keep] ** 2
    cxy = rows.r[keep] * rows.sigma_x[keep] * rows.sigma_y[keep]
    n_groups_all = rows.n_groups

    def total(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sum per group, in row order."""
        return _group_sum(lab, values, n_groups_all)

    # Start from least squares, and measure "has the slope stopped moving"
    # against a scale the data set, so that a slope of exactly 0 (a source
    # with none of the numerator gas) converges as readily as any other.
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(fitted, moments.sxy / moments.sxx, np.nan)
        rms_sigma_y = np.sqrt(total(vy) / moments.n)
        scale = (np.sqrt(moments.syy / moments.n) + rms_sigma_y) / np.sqrt(moments.sxx / moments.n)
    iterations = np.zeros(n_groups_all, dtype=np.int64)
    moving = fitted.copy()
    for step in range(1, max_iter + 1):
        if not moving.any():
            break
        # Only the rows of groups still moving take part in a step, so a few
        # slow groups do not cost a pass over every row.
        live = moving[lab]
        new = _york_slope(
            slope, lab[live], gx[live], gy[live], vx[live], vy[live], cxy[live], n_groups_all
        )
        change = np.abs(new - slope)
        # A group that has converged keeps the slope it converged to: it
        # does not go on iterating while its neighbours finish.
        slope = np.where(moving, new, slope)
        iterations[moving] = step
        moving &= ~(change <= rtol * (np.abs(new) + scale))
    status[moving] = FitStatus.NOT_CONVERGED
    fitted = status == FitStatus.OK

    # York's intercept, standard errors and goodness of fit at the final slope.
    with np.errstate(divide="ignore", invalid="ignore"):
        b = slope[lab]
        weight = 1.0 / (vy + b * b * vx - 2.0 * b * cxy)
        sum_w = total(weight)
        x_bar = total(weight * gx) / sum_w
        y_bar = total(weight * gy) / sum_w
        u = gx - x_bar[lab]
        v = gy - y_bar[lab]
        beta = weight * (u * vy + b * v * vx - (b * u + v) * cxy)
        intercept = y_bar - slope * x_bar
        # The adjusted points sit at X̄ + β on the x axis; the slope's
        # variance is the inverse of their weighted spread.
        adjusted = x_bar[lab] + beta
        adjusted_mean = total(weight * adjusted) / sum_w
        spread = adjusted - adjusted_mean[lab]
        slope_var = 1.0 / total(weight * spread * spread)
        intercept_sigma = np.sqrt(1.0 / sum_w + adjusted_mean**2 * slope_var)
        residual = gy - slope[lab] * gx - intercept[lab]
        mswd = total(weight * residual * residual) / (moments.n - 2)

    _warn_about(status, rows.n_groups)
    return LineFit(
        method="york",
        slope=_blank(slope, fitted),
        intercept=_blank(intercept, fitted),
        slope_sigma=_blank(np.sqrt(slope_var), fitted),
        intercept_sigma=_blank(intercept_sigma, fitted),
        mswd=_blank(mswd, fitted),
        r_squared=_blank(_r_squared(moments), fitted),
        n=moments.n,
        n_dropped=rows.n_dropped,
        iterations=np.where(fitted, iterations, 0),
        status=status.astype(np.int8),
    )


def _group_sum(
    labels: npt.NDArray[np.intp], values: npt.NDArray[np.float64], n_groups: int
) -> npt.NDArray[np.float64]:
    """Sum values per group, adding each group's rows in the order they arrive."""
    return np.asarray(np.bincount(labels, weights=values, minlength=n_groups), dtype=np.float64)


def _gather(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    group: npt.ArrayLike | None,
    n_groups: int | None,
    *,
    sigma_x: npt.ArrayLike | None = None,
    sigma_y: npt.ArrayLike | None = None,
    r: npt.ArrayLike = 0.0,
) -> _Rows:
    """Check a call's arrays against each other and keep the rows to fit.

    Every array is checked on every row a group holds, finite or not, so a
    call that is wrong is refused whatever its data; then the rows outside
    any group, and those whose x or y is not finite, are set aside, the
    latter counted per group.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.ndim != 1 or ys.shape != xs.shape:
        raise TsaraRatioError(
            f"x and y must be one-dimensional and the same length, got shapes "
            f"{xs.shape} and {ys.shape}."
        )
    labels = _labels(group, xs.shape)
    count = _group_count(labels, n_groups)
    sx = _per_row("sigma_x", sigma_x, xs.shape)
    sy = _per_row("sigma_y", sigma_y, xs.shape)
    rr = _per_row("r", r, xs.shape)

    in_group = labels >= 0
    if np.any((sx < 0) | (sy < 0)):
        raise TsaraRatioError("A sigma is negative; a standard error cannot be.")
    if np.any(in_group & (sx == 0) & (sy == 0)):
        raise TsaraRatioError(
            "A point declares a sigma of zero on both axes, which would weigh it "
            "infinitely; at most one axis may be exact."
        )
    if not np.all(np.isfinite(rr) & (np.abs(rr) < 1)):
        raise TsaraRatioError(
            "Every correlation r between a point's x and y errors must be strictly "
            "between -1 and 1."
        )

    finite = np.isfinite(xs) & np.isfinite(ys)
    dropped = np.bincount(labels[in_group & ~finite], minlength=count).astype(np.int64)
    keep = in_group & finite
    return _Rows(
        x=xs[keep],
        y=ys[keep],
        sigma_x=sx[keep],
        sigma_y=sy[keep],
        r=rr[keep],
        group=labels[keep],
        n_groups=count,
        n_dropped=dropped,
    )


def _labels(group: npt.ArrayLike | None, shape: tuple[int, ...]) -> npt.NDArray[np.intp]:
    """Return the group label of every row: all 0 when no labels are given."""
    if group is None:
        return np.zeros(shape, dtype=np.intp)
    labels = np.asarray(group)
    if labels.shape != shape:
        raise TsaraRatioError(f"group must label every row: shape {labels.shape}, rows {shape}.")
    if labels.size and not np.issubdtype(labels.dtype, np.integer):
        raise TsaraRatioError(f"group labels must be integers, got dtype {labels.dtype}.")
    return labels.astype(np.intp)


def _group_count(labels: npt.NDArray[np.intp], n_groups: int | None) -> int:
    """Return how many fits there are, checking a stated count against the labels."""
    needed = int(labels.max()) + 1 if labels.size else 0
    needed = max(needed, 0)
    if n_groups is None:
        return needed
    if n_groups < needed:
        raise TsaraRatioError(
            f"n_groups is {n_groups}, but a row is labelled for group {needed - 1}."
        )
    return n_groups


def _per_row(
    name: str, value: npt.ArrayLike | None, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Broadcast a scalar or per-row quantity to the rows; ``None`` is NaN (not declared)."""
    if value is None:
        return np.full(shape, np.nan)
    try:
        return np.broadcast_to(np.asarray(value, dtype=np.float64), shape)
    except ValueError:
        raise TsaraRatioError(
            f"{name} must be a scalar or one value per row; shape "
            f"{np.shape(value)} does not broadcast to {shape}."
        ) from None


def _moments(rows: _Rows) -> _Moments:
    """Per group: the count, the means and the centred sums, in two passes.

    Centring on each group's mean before squaring keeps the sums accurate
    when the values sit far from zero (methane near 2000 ppb).
    """
    count = rows.n_groups
    n = np.bincount(rows.group, minlength=count).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_mean = _group_sum(rows.group, rows.x, count) / n
        y_mean = _group_sum(rows.group, rows.y, count) / n
    dx = rows.x - x_mean[rows.group]
    dy = rows.y - y_mean[rows.group]
    # "No spread" is asked of the values themselves, not of a sum that
    # rounding can leave a hair above zero.
    lowest = np.full(count, np.inf)
    highest = np.full(count, -np.inf)
    np.minimum.at(lowest, rows.group, rows.x)
    np.maximum.at(highest, rows.group, rows.x)
    return _Moments(
        n=n,
        x_mean=x_mean,
        y_mean=y_mean,
        sxx=_group_sum(rows.group, dx * dx, count),
        syy=_group_sum(rows.group, dy * dy, count),
        sxy=_group_sum(rows.group, dx * dy, count),
        x_spread=highest > lowest,
    )


def _shape_status(moments: _Moments) -> npt.NDArray[np.int8]:
    """Per group, whether its points can hold a line at all."""
    status = np.full(moments.n.shape, FitStatus.OK, dtype=np.int8)
    status[~moments.x_spread] = FitStatus.NO_SPREAD
    status[moments.n < MIN_FIT_POINTS] = FitStatus.TOO_FEW_POINTS
    return status


def _york_slope(
    slope: npt.NDArray[np.float64],
    lab: npt.NDArray[np.intp],
    gx: npt.NDArray[np.float64],
    gy: npt.NDArray[np.float64],
    vx: npt.NDArray[np.float64],
    vy: npt.NDArray[np.float64],
    cxy: npt.NDArray[np.float64],
    n_groups: int,
) -> npt.NDArray[np.float64]:
    """One step of York's iteration: the slope the current one implies, per group.

    York et al. (2004), steps 3-6, in variances: with weights at the current
    slope, the weighted centroid (X̄, Ȳ), and β the shift each point would
    take onto the line, the new slope is Σ W·β·V / Σ W·β·U. A group with no
    rows among those given comes back NaN.
    """

    def total(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _group_sum(lab, values, n_groups)

    with np.errstate(divide="ignore", invalid="ignore"):
        b = slope[lab]
        weight = 1.0 / (vy + b * b * vx - 2.0 * b * cxy)
        sum_w = total(weight)
        u = gx - (total(weight * gx) / sum_w)[lab]
        v = gy - (total(weight * gy) / sum_w)[lab]
        beta = weight * (u * vy + b * v * vx - (b * u + v) * cxy)
        return total(weight * beta * v) / total(weight * beta * u)


def _warn_about(status: npt.NDArray[np.int8], n_groups: int) -> None:
    """Say once per call how many fits York could not weigh or could not settle."""
    missing = int(np.count_nonzero(status == FitStatus.MISSING_SIGMA))
    if missing:
        logger.warning(
            "York fitted no line to %d of %d groups: a point had no declared sigma, and "
            "York cannot weigh a point whose error it was not told (METHODS §4.3). "
            "Declare the instruments' precision, or read the least-squares line.",
            missing,
            n_groups,
        )
    stuck = int(np.count_nonzero(status == FitStatus.NOT_CONVERGED))
    if stuck:
        logger.warning(
            "York's slope did not settle in %d of %d groups; they hold no line.",
            stuck,
            n_groups,
        )


def _blank(values: npt.NDArray[np.float64], keep: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    """Return the values where a line was fitted, NaN elsewhere."""
    return np.where(keep, values, np.nan)


def _r_squared(moments: _Moments) -> npt.NDArray[np.float64]:
    """Return the squared correlation of each group's points; NaN where y does not vary."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(moments.syy > 0, moments.sxy**2 / (moments.sxx * moments.syy), np.nan)


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------


@register_line_fit("ols")
def ols(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    sigma_x: npt.ArrayLike | None = None,
    sigma_y: npt.ArrayLike | None = None,
    group: npt.ArrayLike | None = None,
    *,
    n_groups: int | None = None,
) -> LineFit:
    """Fit an ordinary least-squares line of y on x to every group (METHODS §4.1).

    Unweighted, in closed form: ``slope = Sxy / Sxx`` from each group's
    centred sums, and standard errors from the scatter about the line,
    ``s² = Σ residual² / (n − 2)``, ``σ_slope = s / √Sxx`` and
    ``σ_intercept = s·√(1/n + x̄²/Sxx)``. With noise on x the slope reads low
    by the factor ``var(true x) / (var(true x) + σx²)``, which is why York is
    preferred; least squares is kept for its familiar diagnostics and as the
    line there is when no sigma is declared.

    Parameters
    ----------
    x, y : array_like of float, shape (n_rows,)
        The points. A row with x or y not finite is dropped and counted.
    sigma_x, sigma_y : array_like of float, optional
        Accepted so that every registered estimator is called the same way,
        and checked like York's; not used, since least squares is unweighted
        by definition.
    group : array_like of int, shape (n_rows,), optional
        The fit each row belongs to, numbered from 0; below 0 belongs to
        none. ``None`` puts every row in one fit.
    n_groups : int, optional
        How many fits there are; by default one more than the largest label.

    Returns
    -------
    LineFit
        One line per group; ``mswd`` is NaN and ``iterations`` 0.

    Raises
    ------
    TsaraRatioError
        If the rows, labels or sigmas do not fit together.
    """
    rows = _gather(x, y, group, n_groups, sigma_x=sigma_x, sigma_y=sigma_y)
    moments = _moments(rows)
    status = _shape_status(moments)
    fitted = status == FitStatus.OK
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = moments.sxy / moments.sxx
        intercept = moments.y_mean - slope * moments.x_mean
        # The residual about the line, from the centred values, so that a
        # large offset (methane near 2000 ppb) costs no precision.
        lab = rows.group
        residual = (rows.y - moments.y_mean[lab]) - slope[lab] * (rows.x - moments.x_mean[lab])
        scatter = _group_sum(lab, residual * residual, rows.n_groups)
        variance = scatter / (moments.n - 2)
        slope_sigma = np.sqrt(variance / moments.sxx)
        intercept_sigma = np.sqrt(variance * (1.0 / moments.n + moments.x_mean**2 / moments.sxx))
    nothing = np.full(rows.n_groups, np.nan)
    return LineFit(
        method="ols",
        slope=_blank(slope, fitted),
        intercept=_blank(intercept, fitted),
        slope_sigma=_blank(slope_sigma, fitted),
        intercept_sigma=_blank(intercept_sigma, fitted),
        mswd=nothing,
        r_squared=_blank(_r_squared(moments), fitted),
        n=moments.n,
        n_dropped=rows.n_dropped,
        iterations=np.zeros(rows.n_groups, dtype=np.int64),
        status=status,
    )
