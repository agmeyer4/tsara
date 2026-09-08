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
from typing import TYPE_CHECKING, Literal

import numpy as np

from tsara.core.exceptions import TsaraError
from tsara.core.naming import (
    BOUNDS_ATTR,
    BOUNDS_DIM,
    CELL_METHODS_ATTR,
    TIME_BOUNDS_VAR,
    TIME_COORD,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import xarray as xr

__all__ = [
    "BinnedOntoCells",
    "CellBounds",
    "NAT_NS",
    "SupportLabel",
    "SupportMethod",
    "SupportSource",
    "TsaraSupportError",
    "attach_time_bounds",
    "bin_onto_cells",
    "cell_methods_value",
    "check_bounds_intact",
    "nominal_cadence_ns",
]

#: Where the timestamp sits inside its cell.
#:
#: ``unknown`` is a first-class value rather than an error: 402 files in the
#: target archive declare nothing at all, and refusing them would be worse
#: than admitting them with the assumption recorded. An unknown label is
#: treated as centred, which is the choice that minimises the worst-case
#: misplacement (half a cell rather than a whole one).
SupportLabel = Literal["start", "mid", "end", "unknown"]

#: Whether a value is an average over its cell or a sample inside it.
SupportMethod = Literal["point", "mean"]

#: Where a piece of support information came from, best evidence first.
#:
#: Recorded **per field** (label, width, method) rather than once per stream,
#: because the three are established independently: a stationary Picarro with
#: a stop column and a manifest declaring ``method: mean`` is honestly
#: described as reported / reported / declared. This mirrors the uncertainty
#: system, which already records ``random`` and ``systematic`` provenance
#: separately and reports ``mixed`` when they disagree (METHODS §2.4).
#:
#: * ``reported``  -- per-row start/stop columns in the file itself.
#: * ``declared``  -- the manifest states it.
#: * ``inferred``  -- TSARA read it from the file (column names, cadence).
#: * ``assumed``   -- nothing said; a default was applied and labelled.
SupportSource = Literal["reported", "declared", "inferred", "assumed"]

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
        width_ns: int,
        label: SupportLabel,
    ) -> CellBounds:
        """Build cells of a fixed width from timestamps and a label.

        Parameters
        ----------
        times_ns : numpy.ndarray
            The file's timestamps as int64 epoch nanoseconds.
        width_ns : int
            Cell width. Must be strictly positive: a zero-width cell has zero
            measure, therefore zero weight in every overlap, and would vanish
            from the analysis silently. Real files do declare them, so this
            is a live case rather than a formality.
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
        if width_ns <= 0:
            raise TsaraSupportError(
                f"Cell width must be strictly positive, got {width_ns} ns. A "
                "zero-width cell carries zero weight in every overlap and would "
                "disappear from the analysis without a word."
            )
        times = np.asarray(times_ns, dtype=np.int64)
        if label == "start":
            start = times
        elif label == "end":
            start = times - width_ns
        else:
            # Centred, and offset by floor(width/2) so that `start + width`
            # reproduces the width exactly for odd widths too. Splitting as
            # (w//2, w - w//2) instead would make the cell asymmetric by a
            # nanosecond, which is harmless but would break the "every width
            # is identical" invariant that makes tests exact.
            start = times - width_ns // 2
        return cls(start_ns=start, stop_ns=start + width_ns)

    def floor_width(self, minimum_ns: int) -> tuple[CellBounds, int]:
        """Widen any cell narrower than ``minimum_ns``, keeping it centred.

        Exists for a measured case, not a hypothetical one: one airborne
        instrument in the archive declares stop equal to start, i.e. cells of
        literally zero width. Those cells are real measurements and must not
        be dropped, but zero measure means zero weight everywhere downstream.
        Widening them to the nominal cadence and *counting* how many were
        widened keeps the data and records the intervention.

        Parameters
        ----------
        minimum_ns : int
            Smallest acceptable width, normally the record's nominal cadence.

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
        if minimum_ns <= 0:
            raise TsaraSupportError(
                f"Minimum cell width must be strictly positive, got {minimum_ns} ns."
            )
        narrow = self.width_ns < minimum_ns
        n_widened = int(np.count_nonzero(narrow))
        if n_widened == 0:
            return self, 0
        centre = self.midpoint_ns
        start = np.where(narrow, centre - minimum_ns // 2, self.start_ns)
        stop = np.where(narrow, start + minimum_ns, self.stop_ns)
        return CellBounds(start_ns=start, stop_ns=stop), n_widened


@dataclass(frozen=True)
class BinnedOntoCells:
    """One stream's values averaged onto another stream's cells.

    Attributes
    ----------
    values : numpy.ndarray
        Overlap-weighted mean per target cell; ``nan`` where nothing
        contributed. Never interpolated: a target cell with no overlapping
        source data stays ``nan`` rather than being bridged.
    n_source : numpy.ndarray
        How many source cells contributed to each target cell. The honest
        sample size, and what stops interpolated points posing as
        independent samples in a later regression.
    coverage : numpy.ndarray
        Share of each target cell's width covered by contributing source
        cells. The guard a later stage needs: a canister whose 15 s fill
        overlaps only 3 s of partner data is not comparable to one with full
        coverage, even though both produce a number.
    """

    values: npt.NDArray[np.float64]
    n_source: npt.NDArray[np.int64]
    coverage: npt.NDArray[np.float64]


def bin_onto_cells(
    source: CellBounds,
    values: npt.NDArray[np.float64],
    target: CellBounds,
) -> BinnedOntoCells:
    """Average one stream onto another stream's cells, weighted by overlap.

    The primitive behind cross-rate pairing and the output grid. Each target
    cell receives the mean of the source values that overlap it, each
    weighted by *how much* of the source cell falls inside the target cell.
    That is the operation the interval model exists to make well defined:
    a 60 s mean can only be compared with the mean of a faster stream over
    the same 60 s, and "the same 60 s" is exactly what bounds say.

    Gases are never interpolated here, only averaged, per METHODS §1.2. A
    target cell with no overlapping source data yields ``nan``.

    Parameters
    ----------
    source : CellBounds
        Cells of the stream being averaged. Must be sorted by start time.
    values : numpy.ndarray
        One value per source cell. ``nan`` entries contribute nothing and are
        excluded from ``coverage``, so a masked sample reduces coverage
        rather than silently passing as data.
    target : CellBounds
        Cells to average onto, typically the slower stream's own cells.

    Returns
    -------
    BinnedOntoCells
        Values, contributing counts, and coverage fractions.

    Raises
    ------
    TsaraSupportError
        If ``values`` does not have one entry per source cell, or the source
        cells are not sorted by start time.
    """
    source_values = np.asarray(values, dtype=np.float64)
    if source_values.shape != (len(source),):
        raise TsaraSupportError(
            f"bin_onto_cells got {source_values.size} value(s) for {len(source)} "
            "source cell(s); they must correspond one to one."
        )
    n_target = len(target)
    out_values = np.full(n_target, np.nan, dtype=np.float64)
    out_counts = np.zeros(n_target, dtype=np.int64)
    out_coverage = np.zeros(n_target, dtype=np.float64)
    if n_target == 0 or len(source) == 0:
        return BinnedOntoCells(values=out_values, n_source=out_counts, coverage=out_coverage)

    if np.any(np.diff(source.start_ns) < 0):
        raise TsaraSupportError(
            "bin_onto_cells requires source cells sorted by start time; the "
            "candidate search below is a binary search and would silently miss "
            "overlaps on an unsorted input."
        )

    # Candidate window per target cell. `running_stop` is a cumulative
    # maximum so that it is non-decreasing and therefore searchable, which
    # matters because jittered timestamps give fixed-width cells that can
    # overlap slightly -- their raw stops are then not sorted. Using the
    # running maximum only ever widens the candidate window, so the exact
    # overlap test below still decides membership.
    running_stop = np.maximum.accumulate(source.stop_ns)
    lo = np.searchsorted(running_stop, target.start_ns, side="right")
    hi = np.searchsorted(source.start_ns, target.stop_ns, side="left")
    counts = np.maximum(hi - lo, 0).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return BinnedOntoCells(values=out_values, n_source=out_counts, coverage=out_coverage)

    # Expand (target, candidate-source) pairs without a Python loop: repeat
    # each target index `counts` times, then walk 0..count-1 within each run.
    target_index = np.repeat(np.arange(n_target, dtype=np.int64), counts)
    run_start = np.repeat(np.cumsum(counts) - counts, counts)
    source_index = np.repeat(lo, counts) + (np.arange(total, dtype=np.int64) - run_start)

    overlap = np.minimum(source.stop_ns[source_index], target.stop_ns[target_index]) - np.maximum(
        source.start_ns[source_index], target.start_ns[target_index]
    )
    overlap = np.maximum(overlap, 0)

    paired = source_values[source_index]
    finite = np.isfinite(paired)
    weight = np.where(finite, overlap, 0).astype(np.float64)
    # NaN values are zeroed *before* multiplying: 0 * nan is nan, so relying
    # on the weight alone would poison the sum.
    contribution = weight * np.where(finite, paired, 0.0)

    weight_sum = np.bincount(target_index, weights=weight, minlength=n_target)
    value_sum = np.bincount(target_index, weights=contribution, minlength=n_target)
    out_counts = np.bincount(
        target_index, weights=(weight > 0).astype(np.float64), minlength=n_target
    ).astype(np.int64)

    contributing = weight_sum > 0
    out_values[contributing] = value_sum[contributing] / weight_sum[contributing]
    target_width = target.width_ns.astype(np.float64)
    # A zero-width target cell cannot be covered by anything; guarded rather
    # than divided, since CellBounds permits width 0 (floor_width is how a
    # caller opts into fixing it) and this must not raise on it.
    wide = target_width > 0
    out_coverage[wide] = weight_sum[wide] / target_width[wide]
    return BinnedOntoCells(values=out_values, n_source=out_counts, coverage=out_coverage)


def attach_time_bounds(
    dataset: xr.Dataset,
    bounds: CellBounds,
    method: SupportMethod,
) -> xr.Dataset:
    """Attach CF cell boundaries and cell methods to a stream, in place.

    Writes the representation both producers must agree on exactly: a
    ``time_bnds`` coordinate of shape ``(time, 2)``, the CF ``bounds``
    attribute pointing at it from the time coordinate, and a
    ``cell_methods`` attribute on every time-varying data variable. Sigma
    companions get it too, because a random error describes the same cell as
    the value it belongs to.

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
        if TIME_COORD in dataset[name].dims:
            dataset[name].attrs[CELL_METHODS_ATTR] = cell_methods
    return dataset


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
    if TIME_COORD not in dataset.variables:
        return
    declared = dataset[TIME_COORD].attrs.get(BOUNDS_ATTR)
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
