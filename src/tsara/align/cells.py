"""A stream's cells, and how a set of readings meets a set of target cells.

Everything a join needs to know before it averages anything is a question
about cells rather than values: where a stream's cells are, how wide they
are and how far apart; for each reading that touches a target cell, how much
wider than that cell it is -- the width ratio, the one number that says which
kind of join a pair is (``docs/METHODS.md`` §11.2.4); how many distinct
readings stand behind a set of target cells and how many of them formed more
than one (§11.4.1); and, of the target cells themselves, whether a same-width
stream sits out of phase with them and whether any two of them overlap.

These are asked by the binner, by pairing and by the output grid, and they
must give the same answer to all three, which is why they live in one module
rather than as a private helper of any one caller. Nothing here averages,
propagates or logs: it measures geometry in integer nanoseconds, without a
tolerance anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from tsara.align.variables import TsaraAlignError
from tsara.core.naming import TIME_BOUNDS_VAR
from tsara.core.support import CellBounds, overlap_pairs
from tsara.core.timebase import NS_PER_S

if TYPE_CHECKING:  # pragma: no cover
    from tsara.core.support import OverlapPairs

__all__ = [
    "COPY_RATIO",
    "cadence_s",
    "median_width_s",
    "pair_width_ratios",
    "phase_offset_s",
    "readings_behind",
    "shared_readings",
    "stream_cells",
    "targets_overlap",
    "touched_readings",
]


#: Reading width over target width at which a join is a *copy*: one reading
#: filling two cells' worth of rows (§11.2.4). A named constant rather than a
#: knob. It is the line at which a reading holds less than one row's worth of
#: information, and it needs no tolerance: cadence jitter on the archive tops
#: out at 1.024 against a refusal at 2.
COPY_RATIO = 2.0


# ---------------------------------------------------------------------------
# A stream's own cells
# ---------------------------------------------------------------------------


def stream_cells(stream: xr.Dataset, instrument: str) -> CellBounds:
    """Return a stream's cells, or raise naming the instrument.

    Parameters
    ----------
    stream : xarray.Dataset
        The stream.
    instrument : str
        Its name, used only so the error says which one is at fault.

    Returns
    -------
    CellBounds
        The stream's cell boundaries.

    Raises
    ------
    TsaraAlignError
        If the stream has no bounds or no rows.
    """
    # Bounds are normally a coordinate; a stream may also carry them as a data variable.
    if TIME_BOUNDS_VAR not in stream.coords and TIME_BOUNDS_VAR not in stream.data_vars:
        raise TsaraAlignError(
            f"Stream '{instrument}' carries no '{TIME_BOUNDS_VAR}', so there is no "
            "interval to bin over. Streams gain cells at ingestion (METHODS §10); "
            "a bundle written before format 2 must be reloaded to acquire them."
        )
    # (time, 2) datetime64 edges -> int64 nanoseconds, the unit every overlap is measured in.
    bounds = np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]").astype(np.int64)
    if bounds.size == 0:
        raise TsaraAlignError(f"Stream '{instrument}' has no cells to bin from.")
    # Copies, so the CellBounds owns contiguous arrays rather than views into the dataset.
    return CellBounds(start_ns=bounds[:, 0].copy(), stop_ns=bounds[:, 1].copy())


def median_width_s(cells: CellBounds) -> float:
    """Return the median cell width in seconds."""
    return float(np.median(cells.width_ns)) / NS_PER_S


def cadence_s(cells: CellBounds) -> float:
    """Return the interval between consecutive cell starts, in seconds.

    The cadence in the sense a correlation correction needs — how far apart
    the samples are, not how wide each one is. Falls back to the cell width
    where there is no gap to measure, which happens for a single-cell record
    and for a file that nests one sample inside another.
    """
    if len(cells) >= 2:
        # Start-to-start steps; zero steps (two cells starting together) say nothing about spacing.
        deltas = np.diff(cells.start_ns)
        positive = deltas[deltas > 0]
        if positive.size:
            return float(np.median(positive)) / NS_PER_S
    return max(median_width_s(cells), 1.0 / NS_PER_S)


# ---------------------------------------------------------------------------
# Readings against target cells
# ---------------------------------------------------------------------------


def pair_width_ratios(readings: CellBounds, target: CellBounds, pairs: OverlapPairs) -> np.ndarray:
    """Return reading width over target width for every overlapping pair.

    The one number that says which kind of join a pair is (§11.2.4). At or
    below 1 the reading fits inside its target and is averaged, or straddles
    a boundary and is shared between rows. Above 1 the reading is wider than
    the cell it fills, so its value -- a mean over the whole reading -- stands
    for a shorter interval than it measured: the join *narrows* it. At
    :data:`COPY_RATIO` and beyond one reading fills two cells' worth of rows,
    which is the interpolation rule (§1.2) restated for a step function, and
    is refused unless a caller asks for it by name.

    Measured per pair rather than summed over a reading's rows, and that is
    load-bearing. The rule this replaced added up the time a reading shared
    with *all* target cells and refused at twice the widest, which assumed the
    targets do not overlap: sliding windows sixty seconds wide every ten
    seconds share every 30 s reading with six of them and were refused as a
    copy, while a 60 s mean stood on a single 15 s cell -- four times as wide
    as the cell it fills -- passed, because one narrow cell never adds up to
    two. A ratio per pair has neither hole, needs no tolerance (real jitter
    tops out at 1.024 on the archive against a refusal at 2), and reproduces
    every verdict the summed rule gave where that rule was right.

    Parameters
    ----------
    readings : CellBounds
        Cells being averaged.
    target : CellBounds
        Cells to average onto.
    pairs : OverlapPairs
        Their overlaps, from :func:`~tsara.core.support.overlap_pairs`.

    Returns
    -------
    numpy.ndarray
        One ratio per pair; zero for a pair whose target cell has no width,
        which overlaps nothing by a positive amount and weighs nothing.
    """
    reading_width = readings.width_ns[pairs.reading_index].astype(np.float64)
    target_width = target.width_ns[pairs.target_index].astype(np.float64)
    return np.divide(
        reading_width, target_width, out=np.zeros_like(reading_width), where=target_width > 0
    )


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


def targets_overlap(target: CellBounds) -> bool:
    """Say whether any two target cells overlap by a positive amount.

    Overlapping targets -- sliding windows, nested event windows -- share
    their readings by construction, so rows outnumbering readings is the
    question asked rather than a defect to warn about (§11.2.4). Exact in
    integer nanoseconds, no tolerance: a stream whose fixed-width cells
    overlap by jitter counts as overlapping too, and a join onto its cells
    keeps its per-column record while forgoing the sharing warning, which
    pairing asks again of its surviving rows.

    Parameters
    ----------
    target : CellBounds
        The cells.

    Returns
    -------
    bool
        True if some cell starts before an earlier cell stops.
    """
    if len(target) < 2:
        return False
    order = np.argsort(target.start_ns, kind="stable")
    start, stop = target.start_ns[order], target.stop_ns[order]
    return bool(np.any(start[1:] < np.maximum.accumulate(stop)[:-1]))
