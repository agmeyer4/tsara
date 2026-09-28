"""How many readings stand behind a product's cells, and the phase of a tiling.

The geometry a join needs *before* it averages anything -- a stream's cells,
their widths and spacing, the per-pair width ratio and the copy line, whether
two target cells overlap -- lives in :mod:`tsara.core.support`, because the
rolling stage asks the same questions of its windows and the stages never
import each other. What stays here is what only a *product* can be asked:
how many distinct readings stand behind a set of target cells and how many of
them formed more than one (``docs/METHODS.md`` §11.4.1), and whether a
same-width stream sits out of phase with the cells it is being put on
(§11.9.1). These are asked by the binner, by pairing and by the output grid,
and they must give the same answer to all three, which is why they live in
one module rather than as a private helper of any one caller. Nothing here
averages, propagates or logs: it measures in integer nanoseconds, without a
tolerance anywhere.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from tsara.core.support import CellBounds, overlap_pairs
from tsara.core.timebase import NS_PER_S

__all__ = [
    "phase_offset_s",
    "readings_behind",
    "shared_readings",
    "touched_readings",
]


# ---------------------------------------------------------------------------
# Readings against target cells
# ---------------------------------------------------------------------------


def readings_behind(
    stream: xr.Dataset, variable: str, readings: CellBounds, target: CellBounds
) -> int:
    """Return how many distinct readings of a variable the target cells draw on.

    Every finite reading overlapping at least one target cell by a positive
    amount — the same membership rule the binner uses to form a value, so the
    count describes the numbers actually reported. Fewer readings than occupied
    target cells means some reading appears in more than one row, which a fit
    or receptor model treating rows as independent would count more than once
    (§11.4.1, §11.7).

    Parameters
    ----------
    stream : xarray.Dataset
        The stream holding the variable.
    variable : str
        The variable's name in that stream.
    readings : CellBounds
        The stream's cells.
    target : CellBounds
        The cells whose readings are being counted.

    Returns
    -------
    int
        Distinct finite readings behind the target cells.
    """
    # A reading counts when it both overlaps a target cell and holds a value.
    finite = np.isfinite(np.asarray(stream[variable].values, dtype=np.float64))
    return int(np.count_nonzero(touched_readings(readings, target) & finite))


def touched_readings(readings: CellBounds, target: CellBounds) -> np.ndarray:
    """Return which readings overlap at least one target cell by a positive amount.

    The membership half of :func:`readings_behind`, separated so that a caller
    counting readings for many variables of one instrument — a canister
    carrying fifty VOCs on a campaign grid — finds the overlaps once rather
    than once per variable.

    Parameters
    ----------
    readings : CellBounds
        The instrument's cells.
    target : CellBounds
        The cells whose readings are being counted.

    Returns
    -------
    numpy.ndarray
        One boolean per reading.
    """
    links = overlap_pairs(readings, target)
    touched = np.zeros(len(readings), dtype=bool)
    # A pair touching only at a boundary (overlap 0) does not count, as in binning.
    touched[links.reading_index[links.overlap_ns > 0]] = True
    return touched


def shared_readings(
    stream: xr.Dataset, variable: str, readings: CellBounds, target: CellBounds
) -> int:
    """Return how many distinct finite readings of a variable formed more than one target cell.

    The independence question, asked exactly (§11.4.1). A reading that
    overlaps two target cells by a positive amount contributed to both of
    their values, so the two rows share its error and a fit treating them as
    independent is too confident. Neither of the other two numbers can see
    this. The borrowed share is 0.5 for a partner half a cell out of phase
    whether its readings feed one pair each or two: a *sparse* partner's
    pairs sit two or three cells apart, so no reading reaches two of them,
    while a *dense* partner puts every reading into two. And
    :func:`readings_behind` counts a reading once however many rows it
    formed. Same membership rule as the binner, so a reading touching a cell
    only at its boundary counts for nothing and a masked reading formed
    nothing.

    Parameters
    ----------
    stream : xarray.Dataset
        The stream holding the variable.
    variable : str
        The variable's name in that stream.
    readings : CellBounds
        The stream's cells.
    target : CellBounds
        The cells whose values the readings formed.

    Returns
    -------
    int
        Distinct finite readings overlapping two or more target cells.
    """
    finite = np.isfinite(np.asarray(stream[variable].values, dtype=np.float64))
    links = overlap_pairs(readings, target)
    # How many cells each reading formed, counting only positive overlaps.
    cells_formed = np.bincount(links.reading_index[links.overlap_ns > 0], minlength=len(readings))
    return int(np.count_nonzero((cells_formed > 1) & finite))


# ---------------------------------------------------------------------------
# The target cells among themselves
# ---------------------------------------------------------------------------


def phase_offset_s(readings: CellBounds, target: CellBounds) -> float | None:
    """Return how far a same-width stream sits out of phase with the target, or ``None``.

    The one blend that is both exact and actionable (§11.2.4, §11.9.1): when
    every reading is as wide as every target cell and the two tilings are
    offset, each target value blends two readings and each reading lends part
    of itself to two rows -- the borrowed share says how much, ``2f(1 - f)``
    -- and the remedy belongs to the caller, who can move the grid onto the
    instrument's boundaries or pair on a coarser common clock. ``None`` when
    the widths differ (cadence jitter included: a 1.023 s reading on a 1 s
    cell is narrowed, and not a question of phase), when either side is
    empty, or when the tilings coincide.

    Parameters
    ----------
    readings : CellBounds
        The stream's cells.
    target : CellBounds
        The cells it is being put on.

    Returns
    -------
    float or None
        The first non-zero offset of a reading's start from the target's
        tiling, in seconds; ``None`` when the situation does not arise.
    """
    if len(readings) == 0 or len(target) == 0:
        return None
    period = int(target.width_ns[0])
    if (
        period <= 0
        or not np.all(target.width_ns == period)
        or not np.all(readings.width_ns == period)
    ):
        return None
    # How far each reading starts past a target boundary; zero everywhere means in phase.
    offsets = (readings.start_ns - int(target.start_ns.min())) % period
    shifted = offsets[offsets != 0]
    if shifted.size == 0:
        return None
    return float(shifted[0]) / NS_PER_S
