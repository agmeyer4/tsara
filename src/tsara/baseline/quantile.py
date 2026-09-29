"""The weighted rolling quantile: one definition, a reference form, and a block form.

The definition (``docs/METHODS.md`` §6.3)
------------------------------------------
The baseline of a variable at window *w* is a low quantile of the readings
in a window of duration *w* centred on each reading, every reading weighted
by the share of its own cell lying inside the window. It is one weighted
quantile, defined once:

    Sort the contributing readings by value. Each reading's weight is its
    overlap with the window, and its *position* is the midpoint of its weight
    mass over the total: ``(S_k - w_k / 2) / S_N`` for the cumulative weight
    ``S_k``. The q-quantile is the value interpolated linearly at position q,
    clamped to the lowest and highest reading.

With equal weights the positions are ``(k - 0.5) / N``, Hazen's plotting
position, numpy's ``method="hazen"``; that is the convention the membership
measurement of §6.3 was made with. Numpy's default rule has no canonical
weighted form, and Hazen's is exactly "each reading's weight mass centred on
its value", so it is the one.

Two qualifiers come with every window, the same two every join writes: how
many readings contributed (a finite value and a positive overlap) and how
much of the window they covered. A window holding fewer readings than the
count rule asks for is blank *by the caller* (§6.4): this module applies
the definition and reports the count. It also reports whether a contributing
reading was at least :data:`~tsara.core.support.COPY_RATIO` times as wide as
the window, the width rule every join holds to (§11.2.4), for the caller to
blank. The window at a record's edge is no special case: it holds the
readings it holds, its count and coverage say so, and the count rule decides.

Uncertainty (§6.7)
------------------
The quantile's sampling uncertainty is Woodruff's: the same sorted window
read at positions ``q ± sqrt(q (1 - q) / N_eff)``, ``N_eff`` being Kish's
effective count ``(sum w)^2 / sum w^2``, and half that interval reported as a
one-sigma figure. It is distribution-free and costs two more interpolations
on a sort already done. It assumes the readings in a window are exchangeable
draws, which air is not, so it is a floor on the sampling error rather than
the whole of it; the spread across windows and quantiles is the sweep's.

Two forms, one arithmetic
-------------------------
:func:`weighted_quantile` is the definition on one window, written plainly,
and is the reference the block form is scored against. :func:`rolling_quantile`
evaluates every window of a record: for a block of windows it brackets the
candidate readings (contiguous, since readings are sorted), measures each
overlap exactly, sorts each window once, and reads every quantile from that
one sort. Both build the positions from the same cumulative sum and
interpolate with the same expression, so their values agree bitwise; the
qualifiers beside them (coverage, effective count) are summed in different
orders and agree to rounding.

Measured on a record shaped like the ten-drive Picarro (200 000 readings,
1 s cells every 2.3 s, dropouts, 2 % masked), all three quantiles at once:
2 min 0.9 s, 10 min 3.8 s, 60 min 26.5 s per species. Cost is linear in the
readings times the readings per window; memory is bounded by
:data:`MAX_BLOCK_ELEMENTS`, an element budget per block rather than a count
of windows, so that a 6 h window on a 10 Hz record (216 000 readings per
window) still fits. There the limit is time, not memory: 26 million windows
each sorting 216 000 values, which the sorted-sweep form (O(N log N),
maintaining one sorted window as it slides) would remove if that record
ever arrives. Not built; nothing in the archive is within a factor of a
hundred of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsara.baseline.windows import TsaraBaselineError
from tsara.core.support import COPY_RATIO, CellBounds, candidate_ranges, overlap_lengths

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = ["MAX_BLOCK_ELEMENTS", "RollingQuantile", "rolling_quantile", "weighted_quantile"]


#: Largest ``windows x readings-per-window`` block the rolling engine holds at
#: once. About ten float64 arrays of this many elements are live per block, so
#: two million elements is roughly 160 MB of working memory. A budget in
#: elements rather than in windows because the readings per window is what
#: varies: 52 at a 2 min window on a 2.3 s record, 216 000 at 6 h on 10 Hz.
MAX_BLOCK_ELEMENTS = 2_000_000


@dataclass(frozen=True, eq=False)
class RollingQuantile:
    """Every window's quantiles and the qualifiers that come with them.

    ``eq=False`` because the fields are arrays, matching
    :class:`~tsara.core.support.CellBounds`.

    Attributes
    ----------
    values : numpy.ndarray
        ``(windows, quantiles)``: the weighted quantile of each window at
        each requested quantile; ``nan`` where no reading contributed.
    sigma : numpy.ndarray
        ``(windows, quantiles)``: Woodruff's one-sigma sampling uncertainty
        of each value (§6.7); ``nan`` where the value is.
    n_readings : numpy.ndarray
        ``(windows,)``: readings that contributed, a finite value overlapping
        the window by a positive amount.
    coverage : numpy.ndarray
        ``(windows,)``: the contributing overlap over the window's duration.
        Exceeds 1 slightly where jittered fixed-width cells overlap each
        other, as a join's coverage does (§10.2).
    n_effective : numpy.ndarray
        ``(windows,)``: Kish's effective count of the contributing weights,
        equal to ``n_readings`` when every contributing reading lies wholly
        inside the window; ``nan`` where nothing contributed.
    too_wide : numpy.ndarray
        ``(windows,)``: whether a contributing reading was at least
        :data:`~tsara.core.support.COPY_RATIO` times as wide as the window,
        so that the window should not be stood on it (§6.3).
    carried : numpy.ndarray or None
        ``(windows, quantiles)``: the per-reading number handed in as
        ``carry``, sorted with the values and read at the same positions --
        the reading's systematic sigma at the quantile (§6.7). ``None`` when
        nothing was carried.
    """

    values: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    n_readings: npt.NDArray[np.int64]
    coverage: npt.NDArray[np.float64]
    n_effective: npt.NDArray[np.float64]
    too_wide: npt.NDArray[np.bool_]
    carried: npt.NDArray[np.float64] | None = None


def rolling_quantile(
    readings: CellBounds,
    values: npt.ArrayLike,
    windows: CellBounds,
    quantiles: npt.ArrayLike,
    *,
    carry: npt.ArrayLike | None = None,
    block_elements: int = MAX_BLOCK_ELEMENTS,
) -> RollingQuantile:
    """Evaluate the weighted quantile of a record over every window of a set.

    Parameters
    ----------
    readings : CellBounds
        The readings' cells, sorted by start time.
    values : array-like
        One value per reading; ``nan`` where masked.
    windows : CellBounds
        The windows, in any order: normally :func:`~tsara.baseline.windows.window_cells`
        centred on the readings themselves, but any cells will do: a
        statistic of other samples, such as the differences between
        consecutive readings, can be rolled over windows centred on readings.
    quantiles : array-like
        The quantiles wanted, each in [0, 1]; every one is read from the same
        sort.
    carry : array-like, optional
        One number per reading to carry through the sort and read at the same
        positions as the value: the reading's systematic sigma, which a
        quantile that falls between two readings inherits by the same
        interpolation (§6.7). ``nan`` where the reading is masked is fine;
        a masked reading takes no part.
    block_elements : int, optional
        The element budget per block, :data:`MAX_BLOCK_ELEMENTS` by default.
        Exposed so a test can force many small blocks; not a configuration.

    Returns
    -------
    RollingQuantile
        Values, sigmas and qualifiers, one row per window.

    Raises
    ------
    TsaraBaselineError
        If ``values`` is not one per reading, a quantile is outside [0, 1], a
        window has no width, or the budget is not positive.
    """
    q = _quantiles(quantiles)
    v = np.asarray(values, dtype=np.float64)
    if v.shape != (len(readings),):
        raise TsaraBaselineError(
            f"rolling_quantile got {v.size} value(s) for {len(readings)} reading(s); "
            "they must correspond one to one."
        )
    if block_elements < 1:
        raise TsaraBaselineError(f"block_elements must be positive, got {block_elements}.")
    carried_values = None if carry is None else np.asarray(carry, dtype=np.float64)
    if carried_values is not None and carried_values.shape != v.shape:
        raise TsaraBaselineError(
            f"carry has shape {carried_values.shape} but there are {v.size} reading(s); "
            "it is one number per reading."
        )
    n_windows = len(windows)
    out_values = np.full((n_windows, q.size), np.nan, dtype=np.float64)
    out_sigma = np.full((n_windows, q.size), np.nan, dtype=np.float64)
    n_readings = np.zeros(n_windows, dtype=np.int64)
    coverage = np.zeros(n_windows, dtype=np.float64)
    n_effective = np.full(n_windows, np.nan, dtype=np.float64)
    too_wide = np.zeros(n_windows, dtype=np.bool_)
    out_carried = (
        None if carried_values is None else np.full((n_windows, q.size), np.nan, dtype=np.float64)
    )
    result = RollingQuantile(
        out_values, out_sigma, n_readings, coverage, n_effective, too_wide, out_carried
    )
    if n_windows == 0 or len(readings) == 0:
        return result
    window_width = windows.width_ns.astype(np.float64)
    if np.any(window_width <= 0):
        raise TsaraBaselineError(
            "Every window must have a positive duration; a window of no duration holds no readings."
        )
    # 1. The bracket: which readings may touch each window, by index range.
    # Windows are contiguous ranges of the sorted readings, so a block of
    # windows is a (windows, longest range) matrix of candidate indices.
    lo, hi = candidate_ranges(readings, windows)
    k_max = int((hi - lo).max())
    if k_max == 0:
        return result
    # The width rule is asked per block only when some reading could fail it
    # on some window; on a dense record against long windows, none can.
    check_width = float(readings.width_ns.max()) >= COPY_RATIO * float(windows.width_ns.min())
    block = max(1, block_elements // k_max)
    offsets = np.arange(k_max, dtype=np.int64)
    last = len(readings) - 1
    for first in range(0, n_windows, block):
        rows = slice(first, min(n_windows, first + block))
        # 2. The candidates of every window in the block, and the exact
        # overlap of each: a candidate past its window's range is clipped to
        # a real index and given no weight, so it takes part in nothing.
        index = lo[rows, None] + offsets[None, :]
        inside = index < hi[rows, None]
        index = np.minimum(index, last)
        target_index = np.arange(rows.start, rows.stop, dtype=np.int64)[:, None]
        overlap = overlap_lengths(readings, windows, index, target_index)
        block_values = v[index]
        # 3. Contributing weight: the overlap where the reading holds a value,
        # zero where it is masked -- the join's definition (§11.2).
        weight = np.where(inside & np.isfinite(block_values), overlap, 0).astype(np.float64)
        contributing = weight > 0
        if check_width:
            ratio = readings.width_ns[index].astype(np.float64) / window_width[rows, None]
            too_wide[rows] = np.any(contributing & (ratio >= COPY_RATIO), axis=1)
        count = contributing.sum(axis=1)
        n_readings[rows] = count
        weight_sum = weight.sum(axis=1)
        coverage[rows] = weight_sum / window_width[rows]
        has = weight_sum > 0
        square_sum = np.where(has, (weight**2).sum(axis=1), 1.0)
        n_effective[rows] = np.where(has, weight_sum**2 / square_sum, np.nan)
        # 4. One sort per window, non-contributing candidates last, and the
        # positions of §6.3 from the sorted weights; a non-contributing
        # candidate sits at +inf so no quantile can reach it.
        key = np.where(contributing, block_values, np.inf)
        order = np.argsort(key, axis=1, kind="stable")
        sorted_values = np.take_along_axis(key, order, axis=1)
        sorted_weights = np.take_along_axis(weight, order, axis=1)
        sorted_carry = (
            None
            if carried_values is None or out_carried is None
            else np.take_along_axis(carried_values[index], order, axis=1)
        )
        # The total is the last cumulative sum, not the pairwise `weight_sum`
        # above: the reference form divides by its own last cumulative sum,
        # and only the same sequence of additions gives the same positions
        # to the last bit, which is what lets the two forms be compared
        # exactly rather than to a tolerance.
        cumulative = np.cumsum(sorted_weights, axis=1)
        total = np.where(has, cumulative[:, -1], 1.0)[:, None]
        positions = (cumulative - sorted_weights / 2) / total
        positions = np.where(sorted_weights > 0, positions, np.inf)
        # 5. Every quantile, and its Woodruff interval, from that one sort.
        for column, quantile in enumerate(q):
            out_values[rows, column] = np.where(
                has, _interpolate_rows(positions, sorted_values, count, quantile), np.nan
            )
            error = np.sqrt(quantile * (1.0 - quantile) / n_effective[rows])
            low = _interpolate_rows(
                positions, sorted_values, count, np.clip(quantile - error, 0, 1)
            )
            high = _interpolate_rows(
                positions, sorted_values, count, np.clip(quantile + error, 0, 1)
            )
            out_sigma[rows, column] = np.where(has, (high - low) / 2, np.nan)
            if sorted_carry is not None and out_carried is not None:
                # The carried number at the same bracket and fraction as the value.
                out_carried[rows, column] = np.where(
                    has, _interpolate_rows(positions, sorted_carry, count, quantile), np.nan
                )
    return result


def _quantiles(quantiles: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the requested quantiles as a 1-D float array, each in [0, 1]."""
    q = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
    if q.ndim != 1 or q.size == 0:
        raise TsaraBaselineError("quantiles must be a non-empty one-dimensional sequence.")
    if np.any(~np.isfinite(q)) or np.any(q < 0) or np.any(q > 1):
        raise TsaraBaselineError(f"Every quantile must lie in [0, 1]; got {q.tolist()}.")
    return q


def _interpolate_rows(
    positions: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    count: npt.NDArray[np.int64],
    quantile: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Read one quantile off every row of a block of sorted windows.

    ``positions`` and ``values`` are ``(windows, candidates)``, sorted by value
    with non-contributing candidates at +inf; ``count`` is each row's
    contributing readings; ``quantile`` is one number or one per row. Rows
    with no contributing reading return a number the caller masks.
    """
    wanted = np.broadcast_to(np.asarray(quantile, dtype=np.float64), (positions.shape[0],))
    # How many positions lie below the quantile: the index of the first at or above.
    below = (positions < wanted[:, None]).sum(axis=1)
    upper = np.minimum(below, np.maximum(count - 1, 0))
    lower = np.maximum(below - 1, 0)
    p_lo = np.take_along_axis(positions, lower[:, None], axis=1)[:, 0]
    p_hi = np.take_along_axis(positions, upper[:, None], axis=1)[:, 0]
    v_lo = np.take_along_axis(values, lower[:, None], axis=1)[:, 0]
    v_hi = np.take_along_axis(values, upper[:, None], axis=1)[:, 0]
    # A row with nothing contributing holds only +inf, and inf - inf is a
    # warning about a number the caller masks anyway; give it zeros instead.
    empty = count == 0
    p_lo, p_hi = np.where(empty, 0.0, p_lo), np.where(empty, 0.0, p_hi)
    v_lo, v_hi = np.where(empty, 0.0, v_lo), np.where(empty, 0.0, v_hi)
    return _between(p_lo, p_hi, v_lo, v_hi, wanted)


def _between(
    p_lo: npt.NDArray[np.float64],
    p_hi: npt.NDArray[np.float64],
    v_lo: npt.NDArray[np.float64],
    v_hi: npt.NDArray[np.float64],
    quantile: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Interpolate linearly between two bracketing positions.

    The one expression both forms use, so that the reference and the block
    form agree bitwise. Clamping needs no arithmetic of its own: a quantile
    below the lowest position or above the highest is handed the same index
    twice by its caller, the span is then zero, and the value at that end is
    returned; and whenever the two indices differ the quantile lies between
    their positions by construction.
    """
    span = p_hi - p_lo
    fraction = np.where(span > 0, (quantile - p_lo) / np.where(span > 0, span, 1.0), 0.0)
    return np.asarray(v_lo + fraction * (v_hi - v_lo), dtype=np.float64)


def weighted_quantile(
    values: npt.ArrayLike, weights: npt.ArrayLike, quantiles: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Return the weighted quantile of one window, written from the definition.

    The reference form: readable, one window at a time, and what
    :func:`rolling_quantile` is scored against by test. A caller with one set
    of readings and weights in hand can use it directly.

    Parameters
    ----------
    values : array-like
        The readings' values; ``nan`` where masked.
    weights : array-like
        Each reading's weight, its overlap with the window; zero or ``nan``
        excludes it.
    quantiles : array-like
        The quantiles wanted, each in [0, 1].

    Returns
    -------
    numpy.ndarray
        One value per quantile; ``nan`` for every quantile when nothing
        contributes.

    Raises
    ------
    TsaraBaselineError
        If the two arrays differ in shape or are not one-dimensional, a
        weight is negative, or a quantile is outside [0, 1].
    """
    q = _quantiles(quantiles)
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.ndim != 1 or v.shape != w.shape:
        raise TsaraBaselineError(
            f"values and weights must be one-dimensional and the same length; got shapes "
            f"{v.shape} and {w.shape}."
        )
    if np.any(w < 0):
        raise TsaraBaselineError("A weight cannot be negative; an overlap never is.")
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return np.full(q.shape, np.nan, dtype=np.float64)
    order = np.argsort(v[keep], kind="stable")
    sorted_values = v[keep][order]
    sorted_weights = w[keep][order]
    cumulative = np.cumsum(sorted_weights)
    positions = (cumulative - sorted_weights / 2) / cumulative[-1]
    # The index of the first position at or above each quantile, then the
    # same bracketing and the same expression as the block form.
    below = np.searchsorted(positions, q, side="left")
    upper = np.minimum(below, positions.size - 1)
    lower = np.maximum(below - 1, 0)
    return _between(
        positions[lower], positions[upper], sorted_values[lower], sorted_values[upper], q
    )
