"""Windows as cells: the interval of air a rolling statistic looks at.

A rolling statistic at a reading is computed over a window of stated
duration centred on that reading's cell (``docs/METHODS.md`` §6.3). TSARA
represents the window as a cell, because that is what it is -- an interval
of time -- and because every question a rolling statistic asks of it is a
question a join already asks of a target cell: which readings touch it, by
how much, and whether one of them is so wide that it should not be stood on
it at all. Those answers live in :mod:`tsara.core.support`; this module only
builds the windows.

This module is the vocabulary of the subpackage, and holds its error class;
:mod:`tsara.rolling.quantile` is the operation spoken in it.
"""

from __future__ import annotations

import pandas as pd

from tsara.core.exceptions import TsaraError
from tsara.core.support import CellBounds

__all__ = ["TsaraRollingError", "duration_ns", "window_cells"]


class TsaraRollingError(TsaraError):
    """Raised when a rolling statistic cannot be computed as asked.

    Its own type because the failures are about *windows and sweeps* rather
    than about reading or joining data: a window of no duration, a quantile
    outside [0, 1], a value array that does not match its readings, a
    variable no method is registered for.
    """


def window_cells(cells: CellBounds, window_ns: int) -> CellBounds:
    """Return one window of ``window_ns`` centred on each cell's midpoint.

    Centred exactly as :meth:`~tsara.core.support.CellBounds.from_label`
    centres a cell on its timestamp -- ``start = midpoint - window // 2`` --
    so that an odd nanosecond falls on the same side there and here, and
    every window is exactly ``window_ns`` wide.

    Parameters
    ----------
    cells : CellBounds
        The cells to centre windows on: a stream's own, for the rolling state
        (§6.2), or any other cells a caller wants a statistic evaluated at.
    window_ns : int
        The window's duration in nanoseconds; strictly positive.

    Returns
    -------
    CellBounds
        One window per cell, in the cells' order.

    Raises
    ------
    TsaraRollingError
        If the window has no duration.
    """
    if window_ns <= 0:
        raise TsaraRollingError(
            f"A window must have a positive duration, got {window_ns} ns: a window of "
            "no duration holds no readings and a quantile over it is undefined."
        )
    start = cells.midpoint_ns - window_ns // 2
    return CellBounds(start_ns=start, stop_ns=start + window_ns)


def duration_ns(spec: str) -> int:
    """Return a configured duration such as ``'10min'`` in nanoseconds.

    The one place a window's spelling in the analysis configuration becomes
    a number, so that the sweep coordinate, the window cells and every
    attribute naming the window agree on it.

    Parameters
    ----------
    spec : str
        A pandas-parsable duration string, e.g. ``'2min'``, ``'90s'``.

    Returns
    -------
    int
        The duration in nanoseconds.

    Raises
    ------
    TsaraRollingError
        If the string is not a duration, or is not strictly positive.
    """
    try:
        value = pd.Timedelta(spec)
    except (ValueError, TypeError) as exc:
        raise TsaraRollingError(f"'{spec}' is not a duration: {exc}") from exc
    if pd.isna(value) or value <= pd.Timedelta(0):
        raise TsaraRollingError(f"A window must be a positive duration, got '{spec}'.")
    return int(value.value)
