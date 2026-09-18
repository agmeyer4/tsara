"""Temporal support: the interval of air that each value describes.

The idea, and why it is not just bookkeeping
---------------------------------------------
Until now TSARA treated every timestamp as an *instant*. Most real archives
label *intervals*: a 60 s mean, a ~15 s canister fill, or a 1 s sample whose
timestamp is a start time by ICARTT specification. Treating a 60 s mean as an
instant invents 59 s of resolution the instrument never had, which is the
same error as interpolating a gas, stated more generally. One principle
covers both:

    TSARA never evaluates a value on a finer support than it was
    delivered on.

This module is the arithmetic that makes that principle checkable. A **cell**
is one row of a stream: a value plus the time interval it describes. That
interval is its **support**; its length is the **width**; the **label** says
which point in the interval the file's timestamp names; the **method** says
whether the value is a mean over the interval or a sample taken inside it.

`point` versus `mean` is a claim about arithmetic, not about physics
--------------------------------------------------------------------
A cavity ring-down analyzer is not a point sampler: gas in the cavity is a
mixture of what entered over the previous seconds, species are measured
sequentially, and a ``_Sync`` log is the analyzer's own resampling onto a
grid. None of that makes it a *box-car mean* either. So the line TSARA draws
is operational:

* ``mean`` — an explicit averaging operation over a *known* interval was
  performed. The value times the width is the integral over the cell. This
  licenses interval arithmetic.
* ``point`` — the value is a sample, whatever instrumental smoothing lies
  beneath it. The bounds are a tiling convention, not an instrument claim.

CF says the same thing in its own words: for point data "the cell is
irrelevant to the data and the bounds are arbitrary. Nonetheless, the bounds
may still be included." Declaring ``point`` is therefore a *guard*: it is
what stops a later stage rescaling a 1 s precision figure onto a 2 s cell as
though averaging had happened. Instrument response and cavity residence are
deliberately not modelled (``docs/METHODS.md``); they are a deconvolution
problem, not a cell method.

Measured justification for the width rule
------------------------------------------
Width comes from one nominal cadence per file, applied to every row, never
from the distance to the next row. Measured over 303 files spanning the whole
2024 archive and the 2026 aligned stage, classifying every sampling interval:

===================================================  ==============
what the interval is                                 mean fraction
===================================================  ==============
jitter around the file's one nominal cadence         0.99218
a dropped row (an integer multiple of the cadence)   0.00570
genuinely something else                             0.00211
===================================================  ==============

So the design rule is:

    A cell's support is a property of that measurement, not of its
    neighbours.

A dropped row therefore leaves a *hole* in the tiling; it never produces a
wider cell. Under the rejected alternative, the widest single cell in that
sample would have been **23.3 days** (a 10 s nominal GPS record whose file
spans a whole campaign). The instruments where cadence inference is invalid
-- canisters, matching their modal cadence on as little as 3% of intervals --
are precisely the ones that declare their own start/stop columns, so
inference never has to cover the case it cannot do.

Everything here is integer nanoseconds
---------------------------------------
Bounds arithmetic is done in ``int64`` nanoseconds, never float seconds.
Float time loses nanosecond resolution above ~10^8 s of epoch and would make
an overlap of exactly one cell width compare unequal to itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.exceptions import TsaraError
from tsara.core.naming import (
    BOUNDS_ATTR,
    BOUNDS_DIM,
    CELL_METHODS_ATTR,
    SUPPORT_COVERAGE_ATTR,
    SUPPORT_LABEL_ATTR,
    SUPPORT_LABEL_PROVENANCE_ATTR,
    SUPPORT_METHOD_PROVENANCE_ATTR,
    SUPPORT_WIDENED_ATTR,
    SUPPORT_WIDTH_ATTR,
    SUPPORT_WIDTH_PROVENANCE_ATTR,
    TIME_BOUNDS_VAR,
    TIME_COORD,
    SupportLabel,
    SupportMethod,
    SupportProvenance,
    is_sigma_name,
)
from tsara.core.timebase import NS_PER_S, epoch_ns

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import xarray as xr

__all__ = [
    "BinnedOntoCells",
    "CellBounds",
    "NAT_NS",
    "OverlapPairs",
    "SupportLabel",
    "SupportMethod",
    "SupportProvenance",
    "TsaraSupportError",
    "attach_time_bounds",
    "bin_onto_cells",
    "borrowed_share",
    "cell_methods_value",
    "check_bounds_intact",
    "check_pairs_match",
    "contributing_weights",
    "declared_bounds_name",
    "ensure_time_bounds",
    "nominal_cadence_ns",
    "overlap_pairs",
    "support_attrs",
]

# `SupportLabel`, `SupportMethod` and `SupportProvenance` are re-exported above
# from `tsara.core.naming`, which is where they are defined and documented.
# They live there because the config layer needs the same vocabulary and must
# not pay for this module's NumPy import; they are re-exported here so that
# code doing support arithmetic can take the types from the module it is
# already importing. Same reasoning as `ingest.base.TIME_INDEX_NAME`.

#: ``int64`` value that ``datetime64[ns]`` NaT casts to.
#:
#: Named because it is otherwise an unrecognisable magic number, and because
#: a NaT that survives into bounds arithmetic is silent: it compares, sorts
#: and bins without complaint while placing a cell at an undefined point on
#: the timeline.
NAT_NS = int(np.iinfo(np.int64).min)


class TsaraSupportError(TsaraError):
    """Raised when temporal support cannot be determined or is incoherent.

    Distinct from a config error and from an ingest error: the manifest may
    be valid and the file perfectly readable while the *cells* it implies
    are impossible -- a stop before its start, a NaT boundary, or a width of
    zero, which would carry zero weight in every overlap calculation and so
    disappear from the analysis without a word.
    """


def cell_methods_value(method: SupportMethod) -> str:
    """Return the CF ``cell_methods`` string for a support method.

    Parameters
    ----------
    method : {'point', 'mean'}
        How the value relates to its cell.

    Returns
    -------
    str
        e.g. ``'time: mean'``. Built from
        :data:`~tsara.core.naming.TIME_COORD` rather than spelled out, so
        that renaming the time axis cannot leave a stale CF attribute
        pointing at a dimension that no longer exists.
    """
    return f"{TIME_COORD}: {method}"


def nominal_cadence_ns(times_ns: npt.NDArray[np.int64]) -> int | None:
    """Return one nominal sampling interval for a record, in nanoseconds.

    The **median** of the positive first differences, not the mode. The mode
    needs a rounding resolution chosen in advance, and that choice is wrong
    for at least one real instrument either way: exact-mode agreement is 100%
    on a Picarro but only 12% on a jittery 10 Hz GPS, while both have a
    perfectly well-defined median. The median needs no parameter and is
    unaffected by a minority of gaps, however large -- it is computed over
    *intervals*, so one 23-day hole moves it not at all.

    This is deliberately a per-record scalar. Applying it to every row is
    what makes a dropped sample a hole in the tiling rather than a wider
    cell (see the module docstring).

    Parameters
    ----------
    times_ns : numpy.ndarray
        Timestamps as int64 nanoseconds, sorted ascending.

    Returns
    -------
    int or None
        Median positive interval in nanoseconds, or ``None`` when the record
        is too short to have one. ``None`` rather than an exception because a
        single-row file is a legitimate thing to find in an archive; the
        caller decides whether to fall back to a declared width or refuse.
    """
    if times_ns.size < 2:
        return None
    deltas = np.diff(np.asarray(times_ns, dtype=np.int64))
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return None
    # Rounded rather than truncated: np.median interpolates between the two
    # central values on an even count, and truncation would bias every such
    # cadence low by up to a nanosecond.
    return int(round(float(np.median(positive))))


@dataclass(frozen=True, eq=False)
class CellBounds:
    """The start and stop of every cell in a stream, in epoch nanoseconds.

    Frozen and validated on construction, because an incoherent cell is the
    kind of defect that stays silent: a stop before its start yields a
    negative overlap that clips to zero, and the sample simply stops
    contributing to anything.

    ``eq=False`` because the fields are arrays; comparing two instances with
    ``==`` would raise rather than answer, and no caller needs it.

    Attributes
    ----------
    start_ns, stop_ns : numpy.ndarray
        int64 epoch nanoseconds, one entry per cell, ``start <= stop``.
    """

    start_ns: npt.NDArray[np.int64]
    stop_ns: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate the invariants every later stage relies on."""
        start = np.asarray(self.start_ns, dtype=np.int64)
        stop = np.asarray(self.stop_ns, dtype=np.int64)
        if start.ndim != 1 or stop.ndim != 1:
            raise TsaraSupportError(
                f"Cell bounds must be one-dimensional, got shapes {start.shape} and {stop.shape}."
            )
        if start.shape != stop.shape:
            raise TsaraSupportError(
                f"Cell bounds must have matching lengths, got {start.size} starts "
                f"and {stop.size} stops."
            )
        if np.any(start == NAT_NS) or np.any(stop == NAT_NS):
            raise TsaraSupportError(
                "Cell bounds contain NaT. A missing boundary places a cell at an "
                "undefined point on the timeline and must be dropped, not carried."
            )
        if np.any(stop < start):
            bad = int(np.count_nonzero(stop < start))
            raise TsaraSupportError(
                f"{bad} cell(s) have a stop earlier than their start. Check the "
                "stop column, and whether it needs the same epoch base as the "
                "independent variable."
            )
        object.__setattr__(self, "start_ns", start)
        object.__setattr__(self, "stop_ns", stop)

    def __len__(self) -> int:
        """Return the number of cells."""
        return int(self.start_ns.size)

    @property
    def width_ns(self) -> npt.NDArray[np.int64]:
        """Return each cell's width in nanoseconds."""
        return np.asarray(self.stop_ns - self.start_ns, dtype=np.int64)

    @property
    def midpoint_ns(self) -> npt.NDArray[np.int64]:
        """Return each cell's midpoint in epoch nanoseconds.

        Computed as ``start + width // 2`` rather than ``(start + stop) //
        2``. The obvious spelling overflows int64 for any cell after roughly
        the year 2116, which is inside the range ``datetime64[ns]`` can
        represent, so it is a latent wrong answer rather than an error.
        """
        return np.asarray(self.start_ns + self.width_ns // 2, dtype=np.int64)

    @property
    def span_ns(self) -> int:
        """Return the total extent covered, first start to last stop."""
        if len(self) == 0:
            return 0
        return int(self.stop_ns.max() - self.start_ns.min())

    @property
    def coverage_fraction(self) -> float:
        """Return the share of the record's extent that cells actually cover.

        The duty-cycle diagnostic. A value near 1 means the cells tile the
        record; a small value means most of the span is gap. Measured on the
        real archive the median is 1.0000, but 32 of 299 files fall below
        0.95 and one reaches 0.050 -- and it is precisely those records where
        treating the gap as a wide cell would have been catastrophic.

        Returns
        -------
        float
            Sum of widths divided by total span; ``nan`` when there is no
            span to divide by (an empty or single-instant record).
        """
        span = self.span_ns
        if span <= 0:
            return float("nan")
        return float(self.width_ns.sum() / span)

    @classmethod
    def from_label(
        cls,
        times_ns: npt.NDArray[np.int64],
        width_ns: int | npt.NDArray[np.int64],
        label: SupportLabel,
    ) -> CellBounds:
        """Build cells of a fixed width from timestamps and a label.

        Parameters
        ----------
        times_ns : numpy.ndarray
            The file's timestamps as int64 epoch nanoseconds.
        width_ns : int or numpy.ndarray
            Cell width, one value for every row or a single value for all of
            them. Must be strictly positive: a zero-width cell has zero
            measure, therefore zero weight in every overlap, and would vanish
            from the analysis silently. Real files do declare them, so this
            is a live case rather than a formality.

            The per-row form exists because one instrument's files can
            legitimately disagree about cadence -- measured, some met records
            run at 1 s in one file and 5 s in another -- so a campaign's cells
            are built from each file's own sampling interval rather than from
            a single number for the whole instrument.
        label : {'start', 'mid', 'end', 'unknown'}
            Which point of the cell the timestamp names. ``'unknown'`` is
            treated as ``'mid'``.

        Returns
        -------
        CellBounds
            Cells whose widths are all exactly ``width_ns``.

        Raises
        ------
        TsaraSupportError
            If ``width_ns`` is not strictly positive.
        """
        widths = np.asarray(width_ns, dtype=np.int64)
        if widths.ndim not in (0, 1):
            raise TsaraSupportError(
                f"Cell width must be a scalar or one value per row, got shape {widths.shape}."
            )
        if np.any(widths <= 0):
            smallest = int(widths.min()) if widths.size else 0
            raise TsaraSupportError(
                f"Cell width must be strictly positive, got {smallest} ns. A "
                "zero-width cell carries zero weight in every overlap and would "
                "disappear from the analysis without a word."
            )
        times = np.asarray(times_ns, dtype=np.int64)
        if widths.ndim == 1 and widths.shape != times.shape:
            raise TsaraSupportError(
                f"Got {widths.size} cell width(s) for {times.size} timestamp(s); "
                "a per-row width needs one value per row."
            )
        if label == "start":
            start = times
        elif label == "end":
            start = times - widths
        else:
            # Centred, and offset by floor(width/2) so that `start + width`
            # reproduces the width exactly for odd widths too. Splitting as
            # (w//2, w - w//2) instead would make the cell asymmetric by a
            # nanosecond, which is harmless but would break the "every width
            # is identical" invariant that makes tests exact.
            start = times - widths // 2
        return cls(start_ns=np.asarray(start, dtype=np.int64), stop_ns=start + widths)

    def floor_width(self, minimum_ns: int | npt.NDArray[np.int64]) -> tuple[CellBounds, int]:
        """Widen any cell narrower than ``minimum_ns``, keeping it centred.

        Exists for a measured case, not a hypothetical one: one airborne
        instrument in the archive declares stop equal to start, i.e. cells of
        literally zero width. Those cells are real measurements and must not
        be dropped, but zero measure means zero weight everywhere downstream.
        Widening them to the nominal cadence and *counting* how many were
        widened keeps the data and records the intervention.

        Parameters
        ----------
        minimum_ns : int or numpy.ndarray
            Smallest acceptable width, normally the record's nominal cadence.
            One value per row is accepted for the same reason
            :meth:`from_label` accepts one: an instrument's files can
            legitimately disagree about cadence.

        Returns
        -------
        CellBounds
            Cells with every width at least ``minimum_ns``.
        int
            How many cells were widened, for the caller to log or record.

        Raises
        ------
        TsaraSupportError
            If ``minimum_ns`` is not strictly positive.
        """
        minimum = np.asarray(minimum_ns, dtype=np.int64)
        if np.any(minimum <= 0):
            smallest = int(minimum.min()) if minimum.size else 0
            raise TsaraSupportError(
                f"Minimum cell width must be strictly positive, got {smallest} ns."
            )
        narrow = self.width_ns < minimum
        n_widened = int(np.count_nonzero(narrow))
        if n_widened == 0:
            return self, 0
        centre = self.midpoint_ns
        start = np.where(narrow, centre - minimum // 2, self.start_ns)
        stop = np.where(narrow, start + minimum, self.stop_ns)
        return CellBounds(
            start_ns=np.asarray(start, dtype=np.int64),
            stop_ns=np.asarray(stop, dtype=np.int64),
        ), n_widened


@dataclass(frozen=True)
class BinnedOntoCells:
    """One stream's values averaged onto another stream's cells.

    Attributes
    ----------
    values : numpy.ndarray
        Overlap-weighted mean per target cell; ``nan`` where nothing
        contributed. Never interpolated: a target cell with no overlapping
        readings stays ``nan`` rather than being bridged.
    n_readings : numpy.ndarray
        How many readings *contributed* to each target cell. The honest
        sample size, and what stops interpolated points posing as
        independent samples in a later regression.
    n_overlapping : numpy.ndarray
        How many readings overlapped it at all, masked ones included.
        The difference between this and ``n_readings`` is the number of
        samples QA/QC removed, which is what separates "there was no data
        here" from "the data here was rejected". Both leave a NaN, and a
        later stage diagnosing a dropped pair needs to tell them apart.
    coverage : numpy.ndarray
        Share of each target cell's width covered by contributing
        readings. The guard a later stage needs: a canister whose 15 s fill
        overlaps only 3 s of partner data is not comparable to one with full
        coverage, even though both produce a number.
    borrowed : numpy.ndarray
        Share of each target cell's value that rests on air *outside* the
        cell (:func:`borrowed_share`); ``nan`` where nothing contributed.
        The third qualifier beside the count and the coverage: how many
        readings, how much of the cell, and how much of the value was
        borrowed from beyond it.
    """

    values: npt.NDArray[np.float64]
    n_readings: npt.NDArray[np.int64]
    coverage: npt.NDArray[np.float64]
    n_overlapping: npt.NDArray[np.int64]
    borrowed: npt.NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class OverlapPairs:
    """Every (target cell, reading) pair that overlaps, and by how much.

    The seam between *finding* overlaps and *doing something with them*.
    Extracted when circular binning became the second caller: an angle cannot
    be averaged arithmetically, so it needs its own aggregation, but it must
    weight by exactly the same overlaps as a scalar does or the wind direction
    reported for a cell would describe a different interval than the methane.

    The arrays are "long form": one entry per overlapping pair, so
    ``target_index``, ``reading_index`` and ``overlap_ns`` are parallel and a
    caller aggregates with :func:`numpy.bincount` on ``target_index``.

    ``eq=False`` because the fields are arrays, matching :class:`CellBounds`.

    Attributes
    ----------
    target_index, reading_index : numpy.ndarray
        Indices into the target cells and readings, one pair per entry.
    overlap_ns : numpy.ndarray
        Nanoseconds of overlap for each pair, never negative. Zero entries
        can occur: the candidate search brackets a window and the exact test
        below decides membership, so a bracketed pair that touches only at a
        boundary is kept with weight zero rather than dropped, which keeps
        the "overlapped at all" and "contributed" counts distinguishable.
    n_target : int
        How many target cells there are, needed to size any aggregation.
    """

    target_index: npt.NDArray[np.int64]
    reading_index: npt.NDArray[np.int64]
    overlap_ns: npt.NDArray[np.int64]
    n_target: int


def overlap_pairs(readings: CellBounds, target: CellBounds) -> OverlapPairs:
    """Find every overlapping pair of reading and target cell.

    Vectorized rather than looped: a binary search brackets the candidate
    readings for each target cell, the (target, candidate) pairs are
    expanded without a Python loop, and the exact overlap decides membership.

    Parameters
    ----------
    readings : CellBounds
        Cells being averaged. Must be sorted by start time.
    target : CellBounds
        Cells to average onto.

    Returns
    -------
    OverlapPairs
        The pairs, possibly empty.

    Raises
    ------
    TsaraSupportError
        If the readings are not sorted by start time.
    """
    n_target = len(target)
    empty = np.empty(0, dtype=np.int64)
    if n_target == 0 or len(readings) == 0:
        return OverlapPairs(
            target_index=empty, reading_index=empty, overlap_ns=empty, n_target=n_target
        )

    if np.any(np.diff(readings.start_ns) < 0):
        raise TsaraSupportError(
            "Binning onto cells requires readings sorted by start time; the "
            "candidate search below is a binary search and would silently miss "
            "overlaps on an unsorted input."
        )

    # Candidate window per target cell. `running_stop` is a cumulative
    # maximum so that it is non-decreasing and therefore searchable, which
    # matters because jittered timestamps give fixed-width cells that can
    # overlap slightly -- their raw stops are then not sorted. Using the
    # running maximum only ever widens the candidate window, so the exact
    # overlap test below still decides membership.
    running_stop = np.maximum.accumulate(readings.stop_ns)
    lo = np.searchsorted(running_stop, target.start_ns, side="right")
    hi = np.searchsorted(readings.start_ns, target.stop_ns, side="left")
    counts = np.maximum(hi - lo, 0).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return OverlapPairs(
            target_index=empty, reading_index=empty, overlap_ns=empty, n_target=n_target
        )

    # Expand (target, candidate reading) pairs without a Python loop: repeat
    # each target index `counts` times, then walk 0..count-1 within each run.
    target_index = np.repeat(np.arange(n_target, dtype=np.int64), counts)
    run_start = np.repeat(np.cumsum(counts) - counts, counts)
    reading_index = np.repeat(lo, counts) + (np.arange(total, dtype=np.int64) - run_start)

    overlap = np.minimum(
        readings.stop_ns[reading_index], target.stop_ns[target_index]
    ) - np.maximum(readings.start_ns[reading_index], target.start_ns[target_index])
    return OverlapPairs(
        target_index=target_index,
        reading_index=reading_index,
        overlap_ns=np.maximum(overlap, 0),
        n_target=n_target,
    )


def check_pairs_match(pairs: OverlapPairs, readings: CellBounds, target: CellBounds) -> None:
    """Refuse overlap pairs that were found for other cells.

    A caller may hand :func:`bin_onto_cells` or
    :func:`~tsara.core.circular.bin_circular_onto_cells` pairs it found
    earlier, so that an instrument carrying a thousand columns on one clock
    is searched once rather than a thousand times. The saving is worth
    having; the failure mode is not. Pairs found for a different target
    index a different set of cells, and every aggregation below would then
    be silently wrong in a way no output could reveal, since the numbers
    would still be means of real readings over real intervals -- just not
    the intervals the product claims. So the two cheap facts that would
    expose the mismatch are checked on every call.

    Parameters
    ----------
    pairs : OverlapPairs
        Pairs a caller found earlier.
    readings, target : CellBounds
        The cells they must have been found for.

    Raises
    ------
    TsaraSupportError
        If the pairs are sized for a different target, or index a reading the
        readings do not have.
    """
    if pairs.n_target != len(target):
        raise TsaraSupportError(
            f"The overlap pairs were found for {pairs.n_target} target cell(s), but "
            f"{len(target)} were given. Pairs belong to the cells they were found for; "
            "find them again for these."
        )
    if pairs.reading_index.size and int(pairs.reading_index.max()) >= len(readings):
        raise TsaraSupportError(
            f"The overlap pairs index reading {int(pairs.reading_index.max())}, but only "
            f"{len(readings)} reading(s) were given. Pairs belong to the cells they were "
            "found for; find them again for these."
        )


def contributing_weights(pairs: OverlapPairs, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return each pair's weight in an overlap-weighted mean: its overlap, or zero.

    The one definition of "this reading contributes to this cell": the
    reading holds a value, and the pair's overlap is what it weighs. A masked
    reading weighs nothing, and a pair that touches only at a boundary
    already carries an overlap of zero. Written once and shared by the
    scalar mean, the circular mean, the borrowed share and the uncertainty
    propagation, so that a cell's methane, its wind direction, its sigma and
    its qualifiers are all formed from the same readings at the same
    weights. Three private spellings of this line once existed; they agreed,
    and nothing but a reader's patience guaranteed it.

    Parameters
    ----------
    pairs : OverlapPairs
        The overlapping pairs.
    values : array-like
        One value per reading, ``nan`` where masked.

    Returns
    -------
    numpy.ndarray
        One float64 weight per pair, in nanoseconds of overlap.
    """
    paired = np.asarray(values, dtype=np.float64)[pairs.reading_index]
    return np.where(np.isfinite(paired), pairs.overlap_ns, 0).astype(np.float64)


def borrowed_share(
    pairs: OverlapPairs, readings: CellBounds, weight: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Return, per target cell, how much of its value rests on air outside the cell.

    An overlap-weighted mean uses each reading's value, which is that
    reading's mean over its *own* cell, to describe only the part of that
    cell lying inside the target. For the rest of the reading's cell the
    formula silently assumes the air was the same as inside. This number
    says how much of each target cell's value carries that assumption:

    .. math::

        b_T = \frac{\sum_i w_i \,(1 - w_i / |R_i|)}{\sum_i w_i},

    the coverage-weighted mean over contributing readings of the fraction of
    each reading's cell that lies outside the target (``docs/METHODS.md``
    §11.2.4). It is exactly zero when every contributing reading sits wholly
    inside its target, which is pure averaging and assumes nothing. For two
    equal-width cells offset by a fraction ``f`` of a cell it is ``2f(1-f)``,
    half at half a cell. For a narrower cell wholly inside a wider reading it
    is ``1 - W/R``: three quarters of a 60 s mean stood on a 15 s cell is
    borrowed from the other 45 s.

    It is a magnitude, not a category. Measured, ordinary cadence jitter
    (a 1.023 s analyzer on a 1 s grid) gives about 0.34, a half-phase blend
    exactly 0.5 and allowed narrowing 0.41-0.56, so no threshold on it
    separates the three; the width ratio of a reading to its target is what
    says which case a join is. This says how much was borrowed.

    Parameters
    ----------
    pairs : OverlapPairs
        The overlapping pairs.
    readings : CellBounds
        The readings' cells, for their widths.
    weight : numpy.ndarray
        Each pair's contributing weight, from :func:`contributing_weights`.

    Returns
    -------
    numpy.ndarray
        One value per target cell in ``[0, 1]``; ``nan`` where nothing
        contributed.
    """
    n_target = pairs.n_target
    reading_width = readings.width_ns[pairs.reading_index].astype(np.float64)
    # The share of each reading's own cell lying inside its target. A
    # zero-width reading has no inside; its weight is zero too, so the
    # guarded division changes nothing it contributes.
    inside = np.divide(
        pairs.overlap_ns.astype(np.float64),
        reading_width,
        out=np.zeros_like(reading_width),
        where=reading_width > 0,
    )
    # An overlap equal to the reading's width divides to exactly 1.0, so a
    # reading wholly inside its target borrows exactly nothing -- which is
    # what lets "averaged" be an exact test rather than a tolerance.
    outside = 1.0 - inside
    weight_sum = np.bincount(pairs.target_index, weights=weight, minlength=n_target)
    borrowed_sum = np.bincount(pairs.target_index, weights=weight * outside, minlength=n_target)
    out = np.full(n_target, np.nan, dtype=np.float64)
    filled = weight_sum > 0
    out[filled] = borrowed_sum[filled] / weight_sum[filled]
    return out


def bin_onto_cells(
    readings: CellBounds,
    values: npt.NDArray[np.float64],
    target: CellBounds,
    *,
    pairs: OverlapPairs | None = None,
) -> BinnedOntoCells:
    """Average one stream onto another stream's cells, weighted by overlap.

    The arithmetic behind every join TSARA performs: cross-rate pairing, the
    output grid, and whatever cells a later stage asks for, all through
    :func:`tsara.align.binning.bin_streams_onto_cells`, which calls this for
    each scalar variable. Each target cell receives the mean of the readings
    that overlap it, each weighted by *how much* of the reading's cell falls
    inside the target cell. That is the operation the interval model exists
    to make well defined: a 60 s mean can only be compared with the mean of a
    faster stream over the same 60 s, and "the same 60 s" is exactly what
    bounds say.

    Gases are never interpolated here, only averaged, per METHODS §1.2. A
    target cell with no overlapping readings yields ``nan``.

    Parameters
    ----------
    readings : CellBounds
        Cells of the stream being averaged. Must be sorted by start time.
    values : numpy.ndarray
        One value per reading. ``nan`` entries contribute nothing and are
        excluded from ``coverage``, so a masked sample reduces coverage
        rather than silently passing as data.
    target : CellBounds
        Cells to average onto, typically the wider-supported stream's own
        cells.
    pairs : OverlapPairs, optional
        The overlaps between ``readings`` and ``target``, if the caller has
        found them already. The overlap search is the one cost that scales
        with the record and it depends only on the cells, so an instrument
        carrying many variables on one clock finds them once and passes them
        to every call. They are checked against the cells
        (:func:`check_pairs_match`) rather than trusted.

    Returns
    -------
    BinnedOntoCells
        Values, contributing counts, coverage fractions and borrowed shares.

    Raises
    ------
    TsaraSupportError
        If ``values`` does not have one entry per reading, the readings
        cells are not sorted by start time, or ``pairs`` were found for
        other cells.
    """
    reading_values = np.asarray(values, dtype=np.float64)
    if reading_values.shape != (len(readings),):
        raise TsaraSupportError(
            f"bin_onto_cells got {reading_values.size} value(s) for {len(readings)} "
            "reading(s); they must correspond one to one."
        )
    n_target = len(target)
    out_values = np.full(n_target, np.nan, dtype=np.float64)
    out_counts = np.zeros(n_target, dtype=np.int64)
    out_coverage = np.zeros(n_target, dtype=np.float64)
    out_overlapping = np.zeros(n_target, dtype=np.int64)
    out_borrowed = np.full(n_target, np.nan, dtype=np.float64)

    if pairs is None:
        pairs = overlap_pairs(readings, target)
    else:
        check_pairs_match(pairs, readings, target)
    if pairs.overlap_ns.size == 0:
        return BinnedOntoCells(
            values=out_values,
            n_readings=out_counts,
            coverage=out_coverage,
            n_overlapping=out_overlapping,
            borrowed=out_borrowed,
        )
    target_index = pairs.target_index
    reading_index = pairs.reading_index
    overlap = pairs.overlap_ns

    paired = reading_values[reading_index]
    finite = np.isfinite(paired)
    weight = contributing_weights(pairs, reading_values)
    # NaN values are zeroed *before* multiplying: 0 * nan is nan, so relying
    # on the weight alone would poison the sum.
    contribution = weight * np.where(finite, paired, 0.0)

    weight_sum = np.bincount(target_index, weights=weight, minlength=n_target)
    value_sum = np.bincount(target_index, weights=contribution, minlength=n_target)
    out_counts = np.bincount(
        target_index, weights=(weight > 0).astype(np.float64), minlength=n_target
    ).astype(np.int64)
    # Overlap alone, ignoring whether the value survived QA/QC.
    out_overlapping = np.bincount(
        target_index, weights=(overlap > 0).astype(np.float64), minlength=n_target
    ).astype(np.int64)

    contributing = weight_sum > 0
    out_values[contributing] = value_sum[contributing] / weight_sum[contributing]
    target_width = target.width_ns.astype(np.float64)
    # A zero-width target cell cannot be covered by anything; guarded rather
    # than divided, since CellBounds permits width 0 (floor_width is how a
    # caller opts into fixing it) and this must not raise on it.
    wide = target_width > 0
    # Coverage can exceed 1 slightly, and is left alone when it does.
    # Fixed-width cells centred on jittered timestamps overlap each other, so
    # their overlaps with one target cell can sum past its width (§10.2, where
    # the effect is documented as benign: the value is a weighted MEAN, so the
    # weights normalize). Clipping would hide a real property of the input
    # record behind a tidier number.
    out_coverage[wide] = weight_sum[wide] / target_width[wide]
    return BinnedOntoCells(
        values=out_values,
        n_readings=out_counts,
        coverage=out_coverage,
        n_overlapping=out_overlapping,
        borrowed=borrowed_share(pairs, readings, weight),
    )


def attach_time_bounds(
    dataset: xr.Dataset,
    bounds: CellBounds,
    method: SupportMethod,
) -> xr.Dataset:
    """Attach CF cell boundaries and cell methods to a stream, in place.

    Writes the representation both producers must agree on exactly: a
    ``time_bnds`` coordinate of shape ``(time, 2)``, the CF ``bounds``
    attribute pointing at it from the time coordinate, and a
    ``cell_methods`` attribute on every time-varying data variable.

    **Not on the sigma companions**, and the exclusion is a correctness
    matter rather than a fastidious one. ``cell_methods`` says what operation
    produced a value *from* its cell, so ``time: mean`` on ``sigma_rand_ch4``
    asserts the stored number is the mean of the random sigmas over the cell.
    It is not. A sigma describes the uncertainty *of the cell's value*, and
    where that value is an average the two differ by exactly the square root
    of N_eff (METHODS §3.4) -- the factor that makes averaging worth doing in
    the first place. Stamping ``time: mean`` on it would put a false claim in
    the file, off by the one quantity the two-component design exists to keep
    track of.

    The systematic companion happens to satisfy ``time: mean`` exactly, since
    a fully correlated error does not average down and every within-cell value
    is the same number. It is excluded anyway: the claim is true there only by
    coincidence, and stamping the same string on both invites a reader to
    treat two components that behave oppositely under averaging as though they
    were alike -- which is the one thing the two-component design exists to
    prevent. What the companions *are* is said by their names and their
    ``uncertainty_component`` attribute; a cell method is the wrong vocabulary
    for it.

    Parameters
    ----------
    dataset : xarray.Dataset
        Stream to modify. Must have a ``time`` dimension matching ``bounds``.
    bounds : CellBounds
        One cell per timestamp.
    method : {'point', 'mean'}
        How values relate to their cells.

    Returns
    -------
    xarray.Dataset
        The same object, for chaining.

    Raises
    ------
    TsaraSupportError
        If the dataset's time axis and the bounds have different lengths.
    """
    n_time = int(dataset.sizes.get(TIME_COORD, 0))
    if n_time != len(bounds):
        raise TsaraSupportError(
            f"Cannot attach {len(bounds)} cell(s) to a stream with {n_time} "
            "timestamp(s); bounds are per-timestamp."
        )
    stacked = np.stack(
        [
            bounds.start_ns.astype("datetime64[ns]"),
            bounds.stop_ns.astype("datetime64[ns]"),
        ],
        axis=1,
    )
    # A coordinate, not a data variable: it describes the time axis rather
    # than being measured on it, and xarray moves it into coords on load
    # anyway once the `bounds` attribute names it. Registering it as a
    # coordinate up front keeps a saved stream and a reloaded one the same
    # shape.
    dataset.coords[TIME_BOUNDS_VAR] = ((TIME_COORD, BOUNDS_DIM), stacked)
    dataset[TIME_COORD].attrs[BOUNDS_ATTR] = TIME_BOUNDS_VAR
    # Declaring the axis costs two attributes and is what lets a generic CF
    # tool find the time coordinate by role rather than by our choice of
    # name. Without them cf_xarray reports no axes at all for the stream,
    # which would make the CF compliance claim half true.
    dataset[TIME_COORD].attrs["standard_name"] = "time"
    dataset[TIME_COORD].attrs["axis"] = "T"
    cell_methods = cell_methods_value(method)
    for name in dataset.data_vars:
        if TIME_COORD in dataset[name].dims and not is_sigma_name(str(name)):
            dataset[name].attrs[CELL_METHODS_ATTR] = cell_methods
    return dataset


def declared_bounds_name(dataset: xr.Dataset) -> str | None:
    """Return the bounds variable a stream's time coordinate names, if any.

    Looks in ``encoding`` as well as ``attrs``, and that is not defensive
    padding: opening a file with ``decode_coords="all"`` -- which TSARA does,
    so that ``time_bnds`` comes back as a coordinate -- **moves** the CF
    ``bounds`` attribute out of ``attrs`` and into ``encoding``. Checking only
    ``attrs`` therefore reports "this stream has no cells" for every stream
    TSARA itself just loaded.

    That mistake was live and nearly invisible. The migration helper below
    would have re-attached *assumed* centred cells over perfectly good
    declared ones on every load, and the round-trip test could not see it
    because for a centred point stream the assumed cells are identical to the
    real ones. It only shows up on a start-labelled mean stream, where it
    silently moves every cell by half its width.

    Parameters
    ----------
    dataset : xarray.Dataset
        Stream to inspect.

    Returns
    -------
    str or None
        The declared bounds variable name, or None if the stream declares
        none. A declared name whose variable is absent is still returned;
        distinguishing that case is :func:`check_bounds_intact`'s job.
    """
    if TIME_COORD not in dataset.variables:
        return None
    time = dataset[TIME_COORD]
    name = time.attrs.get(BOUNDS_ATTR, time.encoding.get(BOUNDS_ATTR))
    return None if name is None else str(name)


def check_bounds_intact(dataset: xr.Dataset) -> None:
    """Verify a stream's cell boundaries are still coherent.

    Turns the "never resample a stream carrying bounds" rule from a note in
    the documentation into something the code can enforce. Measured, xarray's
    ``resample`` fails on a bounds-bearing stream in two ways and raises on
    neither: stored as a coordinate the bounds variable is *dropped* while
    ``time.attrs['bounds']`` goes on naming it, and stored as a data variable
    the boundary timestamps are *averaged* into cells no instrument measured.
    The first case is what this catches, and it is the one TSARA's own layout
    would produce.

    Cheap enough to call at every persistence boundary, which is where a
    corrupted stream would otherwise be written to disk and inherited by
    everything downstream.

    Parameters
    ----------
    dataset : xarray.Dataset
        Stream to check. A stream with no ``bounds`` attribute at all is
        fine: not every TSARA product carries cells, and this must not
        invent a requirement.

    Raises
    ------
    TsaraSupportError
        If the time coordinate declares a bounds variable that is missing,
        or one whose shape does not match the time axis.
    """
    declared = declared_bounds_name(dataset)
    if declared is None:
        return
    if declared not in dataset.variables:
        raise TsaraSupportError(
            f"The time coordinate names '{declared}' as its bounds variable, but "
            "no such variable is present. This is what an xarray resample/coarsen "
            "leaves behind; use TSARA's bounds-aware binning instead."
        )
    expected = (int(dataset.sizes[TIME_COORD]), 2)
    found = tuple(int(size) for size in dataset[declared].shape)
    if found != expected:
        raise TsaraSupportError(
            f"Bounds variable '{declared}' has shape {found}, but the time axis "
            f"requires {expected}: one start and one stop per timestamp."
        )


def support_attrs(
    *,
    label: SupportLabel,
    width_ns: int | None,
    coverage: float,
    label_provenance: SupportProvenance,
    width_provenance: SupportProvenance,
    method_provenance: SupportProvenance,
    n_widened: int = 0,
) -> dict[str, str | float]:
    """Build the stream attributes that describe temporal support.

    Everything here is a netCDF-safe scalar (str or float), the invariant
    both bundle writers rely on. Provenance is recorded per field rather than
    once per stream, because the three facts are established independently:
    a stationary analyzer with a stop column in its file and a manifest
    declaring ``method: mean`` is honestly reported / reported / declared,
    and one label could not say that.

    Parameters
    ----------
    label : {'start', 'mid', 'end', 'unknown'}
        The ORIGINAL label position, before ``time`` was moved to the cell
        midpoint. Recorded so the shift is recoverable and explicable.
    width_ns : int or None
        Nominal cell width in nanoseconds, or None when widths are per row
        and no single nominal value applies (a canister sampler).
    coverage : float
        Share of the record's extent that cells cover, the duty-cycle
        diagnostic from :attr:`CellBounds.coverage_fraction`.
    label_provenance, width_provenance, method_provenance : str
        Where each fact came from; see :data:`SupportProvenance`.

    Returns
    -------
    dict
        Attributes to merge into a stream's ``attrs``.
    """
    attrs: dict[str, str | float] = {
        SUPPORT_LABEL_ATTR: label,
        SUPPORT_COVERAGE_ATTR: float(coverage),
        SUPPORT_LABEL_PROVENANCE_ATTR: label_provenance,
        SUPPORT_WIDTH_PROVENANCE_ATTR: width_provenance,
        SUPPORT_METHOD_PROVENANCE_ATTR: method_provenance,
    }
    if width_ns is not None:
        attrs[SUPPORT_WIDTH_ATTR] = float(width_ns) / NS_PER_S
    if n_widened:
        attrs[SUPPORT_WIDENED_ATTR] = float(n_widened)
    return attrs


def ensure_time_bounds(dataset: xr.Dataset) -> bool:
    """Give a stream assumed cells if it has none, and say whether it did.

    The migration path for products written before cells existed, and the
    reason a version-1 bundle is readable rather than refused. The reading it
    applies is the weakest available and is labelled as such: cells of the
    record's own nominal cadence, centred on each timestamp, so no timestamp
    moves and nothing is claimed about averaging.

    Parameters
    ----------
    dataset : xarray.Dataset
        Stream to complete, modified in place.

    Returns
    -------
    bool
        True if cells were attached, False if the stream already had them.
        The caller decides whether that is worth logging; it is not an error.
    """
    if TIME_COORD not in dataset.variables:
        return False
    if declared_bounds_name(dataset) is not None:
        # Already has cells, or names cells that have gone missing. The second
        # case is a corruption rather than an absence, so it is left for
        # `check_bounds_intact` to report instead of being papered over with
        # assumed cells that would hide it.
        return False

    import pandas as pd

    stamps = epoch_ns(pd.DatetimeIndex(dataset[TIME_COORD].values))
    cadence = nominal_cadence_ns(stamps)
    if cadence is None:
        # A single-sample stream has no cadence to infer, and inventing one
        # would be a fabrication rather than a weak reading. Left alone.
        return False
    bounds = CellBounds.from_label(stamps, cadence, "unknown")
    attach_time_bounds(dataset, bounds, "point")
    dataset.attrs.update(
        support_attrs(
            label="unknown",
            width_ns=cadence,
            coverage=bounds.coverage_fraction,
            label_provenance="assumed",
            width_provenance="inferred",
            method_provenance="assumed",
        )
    )
    return True
