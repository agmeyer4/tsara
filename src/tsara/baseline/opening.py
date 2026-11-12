"""The rolling opening: a low quantile, then a high one of those over each window.

The definition (``docs/METHODS.md`` §6.5, ``rolling_opening``)
---------------------------------------------------------------
Two passes of the one weighted quantile of §6.3, over the same windows, the
windows centred on the readings themselves:

1. **First pass.** At every reading, the q-quantile of the readings in the
   window centred on it: the rolling quantile, exactly. A window that fails
   the count rule (§6.4) or the width rule (§6.3) is blank *here*, before
   anything is spread, so that a window too thin to place its quantile cannot
   lend a too-high value to its neighbours.
2. **Second pass.** At every reading, the same weighted quantile again, of
   the first pass's values at the readings in its window, at 1 - q
   (``rank``) or at 1, the highest (``max``). Each first-pass value counts, as
   each reading did, in proportion to the share of its reading's cell inside
   the window. The reading's own value is always among them; where it is
   blank, so is the result.

A window centred on reading *j* contains the reading at *t* when *j* lies
within half a window of *t*, and those are the readings the second pass
gathers at *t* (to within the half cell by which a reading at the window's
edge overlaps it). The result is therefore a high one of the low quantiles of
every window that contains the reading: a greyscale morphological opening
with a flat window, the minimum and the maximum replaced by quantiles. With
``max`` and q towards 0 it is the textbook opening; with q > 0 the name is
borrowed, since applying it twice changes it and it may sit above some
readings. The forward/backward baseline (a one-sided quantile ahead and one
behind, each followed by a maximum over a window the other way) is the
``max`` form: each of its two lines is the highest low quantile among the
windows that contain the reading, so both equal this one, and each other,
except within one window of a record's ends.

Why the second pass
-------------------
A centred low quantile sits (1/2 - q) x window x slope below a steady ramp,
rising and falling alike, so a ramp longer than the noise can hide stands as
enhancement. Every window containing *t* sees the ramp; the one whose low
end sits highest is the one reaching furthest up it. On a straight ramp
without noise, ``max`` sits q x window x slope above the line, and ``rank``
lies on it: the first pass shifts the line down by (1/2 - q) x window x slope,
the second up by the same. A plume shorter than (1 - q) x window is left
whole, since every window containing a reading of it holds more than a share
q of air outside it; anything wider is taken into the baseline, as the
rolling quantile also does.

Uncertainty (§6.7)
------------------
The baseline at a reading is one window's first-pass value (``max``) or an
interpolation between two (``rank``). Its random sigma is that window's
Woodruff interval, carried through the second sort and read at the same
positions, as a reading's systematic sigma is carried through the first.
It remains a floor on the sampling error (§6.7), and it does not count the
choice the second pass made among overlapping windows.

Cost
----
The first pass reads every quantile from one sort per window; the second
pass sorts a different set of values for each quantile, so an opening costs
about 1 + (number of quantiles) times the sorts of a rolling quantile.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsara.baseline.quantile import MAX_BLOCK_ELEMENTS, rolling_quantile
from tsara.baseline.windows import TsaraBaselineError, window_cells

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    import numpy.typing as npt

    from tsara.config.analysis import Dilation
    from tsara.core.support import CellBounds

logger = logging.getLogger(__name__)

__all__ = ["RollingOpening", "rolling_opening"]


#: The second passes: ``rank`` reads the first pass's values at 1 - q, the
#: mirror of the first pass's position, so that a straight ramp is restored;
#: ``max`` reads the highest.
_DILATIONS = ("rank", "max")


@dataclass(frozen=True, eq=False)
class RollingOpening:
    """The opening at every reading, and what its first pass found.

    ``eq=False`` because the fields are arrays, matching
    :class:`~tsara.baseline.quantile.RollingQuantile`.

    Attributes
    ----------
    values : numpy.ndarray
        ``(readings, quantiles)``: the opening; ``nan`` where the window
        centred on the reading fails the count or width rule.
    sigma : numpy.ndarray
        ``(readings, quantiles)``: the Woodruff interval of the window whose
        first-pass value the opening took (§6.7); ``nan`` where the value is.
    carried : numpy.ndarray or None
        ``(readings, quantiles)``: the per-reading number handed in as
        ``carry``, read at the first pass's positions and then at the
        second's -- the reading's systematic sigma at the reading the
        baseline rests on. ``None`` when nothing was carried.
    n_readings, coverage, too_wide : numpy.ndarray
        ``(readings,)``: the window centred on each reading, as the first pass
        found it -- its contributing count, its covered share and the width
        flag. These are the qualifiers the count and width rules read, so
        the state applies the same rules to them as to a rolling quantile.
    """

    values: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    carried: npt.NDArray[np.float64] | None
    n_readings: npt.NDArray[np.int64]
    coverage: npt.NDArray[np.float64]
    too_wide: npt.NDArray[np.bool_]


def rolling_opening(
    readings: CellBounds,
    values: npt.ArrayLike,
    window_ns: int,
    quantiles: Sequence[float],
    *,
    dilation: Dilation,
    min_readings: Sequence[int],
    carry: npt.ArrayLike | None = None,
    block_elements: int = MAX_BLOCK_ELEMENTS,
) -> RollingOpening:
    """Open a record at one window: a low quantile, then a high one of those.

    Parameters
    ----------
    readings : CellBounds
        The readings' cells, sorted by start time.
    values : array-like
        One value per reading; ``nan`` where masked.
    window_ns : int
        The window's duration, in nanoseconds; every window is centred on a
        reading (:func:`~tsara.baseline.windows.window_cells`), since the
        second pass gathers the first pass's values at the readings.
    quantiles : sequence of float
        The first pass's quantiles, each in [0, 1].
    dilation : {"rank", "max"}
        The second pass: the (1 - q)-quantile of the first pass's values over
        each window, or the highest.
    min_readings : sequence of int
        For each quantile, the readings a window must hold for its first-pass
        value to count (§6.4); applied before the second pass.
    carry : array-like, optional
        One number per reading to carry through both passes: the reading's
        systematic sigma (§6.7).
    block_elements : int, optional
        The rolling engine's per-block budget; not a configuration.

    Returns
    -------
    RollingOpening
        The opening, its sigmas, and the first pass's qualifiers.

    Raises
    ------
    TsaraBaselineError
        If ``dilation`` is not ``rank`` or ``max``, or ``min_readings`` is not
        one count per quantile; and whatever the rolling engine refuses.
    """
    if dilation not in _DILATIONS:
        raise TsaraBaselineError(f"The opening's second pass is 'rank' or 'max', not {dilation!r}.")
    q = [float(quantile) for quantile in quantiles]
    if len(min_readings) != len(q):
        raise TsaraBaselineError(
            f"min_readings holds {len(min_readings)} count(s) for {len(q)} quantile(s); "
            "the count rule gives one per quantile."
        )
    windows = window_cells(readings, window_ns)
    # 1. The first pass: the rolling quantile, with the reading's systematic
    # sigma carried through its sort.
    first = rolling_quantile(
        readings, values, windows, q, carry=carry, block_elements=block_elements
    )
    # 2. The rules, before anything spreads: a first-pass value from a window
    # too thin for its quantile, or holding a reading too wide for it, takes no
    # part in the second pass, as a masked reading takes none in the first.
    count = first.n_readings[:, None] < np.asarray(min_readings, dtype=np.int64)[None, :]
    blank = count | first.too_wide[:, None]
    lows = np.where(blank, np.nan, first.values)
    n, n_quantiles = lows.shape
    opened = np.full((n, n_quantiles), np.nan, dtype=np.float64)
    sigma = np.full((n, n_quantiles), np.nan, dtype=np.float64)
    carried = None if first.carried is None else np.full((n, n_quantiles), np.nan)
    for column, quantile in enumerate(q):
        # 3. The second pass, once per quantile since its values differ: the
        # first pass's sigma, and the systematic sigma it read, ride through
        # this sort too, so that each is read off the window the result rests on.
        riders = [first.sigma[:, column]]
        if first.carried is not None:
            riders.append(first.carried[:, column])
        second = rolling_quantile(
            readings,
            lows[:, column],
            windows,
            [1.0 if dilation == "max" else 1.0 - quantile],
            carry=np.stack(riders, axis=1),
            block_elements=block_elements,
        )
        assert second.carried is not None  # carried columns were handed in
        # A reading whose own window is blank has no opening: the windows that
        # contain it may hold values, but the rule speaks for the reading's
        # own window, as it does for a rolling quantile.
        own = np.isfinite(lows[:, column])
        opened[:, column] = np.where(own, second.values[:, 0], np.nan)
        sigma[:, column] = np.where(own, second.carried[:, 0, 0], np.nan)
        if carried is not None:
            carried[:, column] = np.where(own, second.carried[:, 0, 1], np.nan)
    return RollingOpening(
        values=opened,
        sigma=sigma,
        carried=carried,
        n_readings=first.n_readings,
        coverage=first.coverage,
        too_wide=first.too_wide,
    )
