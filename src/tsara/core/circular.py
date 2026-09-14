r"""Averaging angles, which is not averaging numbers.

The problem in one line
------------------------
The arithmetic mean of 359° and 1° is 180°: due south, from two readings that
were both essentially due north. A direction is a point on a circle, not a
number on a line, and the mean of several directions is the angle of their
summed unit vectors.

This is not a corner case in the archive TSARA targets. Measured on the ten
2024 mobile-lab drive days, over the 3337 sixty-second cells of 1 Hz wind
direction that hold at least 30 readings, the arithmetic mean differs from the
vector mean by more than 45° in **26.7 %** of cells, and by up to a full 180°.

What a binned direction carries
--------------------------------
Summing unit vectors produces two numbers, not one: the **mean direction**
and the **mean resultant length** *R*, the length of the average vector.
*R* runs from 1 (every sample pointing the same way) to 0 (the samples cancel
completely, and there is no mean direction at all).

*R* is the primary quantity, and TSARA stores it. Every dispersion statistic
in the circular literature is a transform of it — the circular standard
deviation :math:`\sqrt{-2\ln R}` and the Yamartino (1984) approximation
alike — so choosing between them is choosing a presentation, not an
estimator. Storing *R* keeps the choice open and loses nothing.

The dispersion TSARA reports alongside it is the **exact circular standard
deviation**, which is unbounded: as the samples spread toward uniform it runs
to infinity, which is the honest description of a direction that has ceased
to exist. The Yamartino form saturates near 105° instead, and is deliberately
not implemented (``docs/METHODS.md`` §11.5): it is a single-pass
approximation built for 1980s dataloggers that could not hold the sample
vectors in memory, which is not a constraint TSARA operates under. Measured on
the same drive data the two agree to 0.16° in the median cell and diverge by
up to 66.5° — entirely in the cells where the direction is tumbling and the
unbounded answer is the correct one.

Two things *R* does not say
----------------------------
**How many readings it rests on.** *R* is biased high for a small sample:
uniformly random directions, which have no mean at all, average *R* = 0.64
from two readings, 0.40 from five and 0.23 from fifteen, and the dispersion
derived from it is understated to match (a true 40° spread reports about 34°
from five readings). By sixty readings the bias is a degree. Nothing here
corrects it, because a correction assumes a distribution; the contributing
count travels beside every binned direction so a reader can see what *R*
rests on (``docs/METHODS.md`` §11.5).

**How hard the wind blew.** These are *unit*-vector means: every reading
counts equally whatever its speed. The other established convention weights
each reading by speed and reports the direction the air moved in on average.
Measured on a 2024 drive the two differ by a median 2.2° per minute and by
more when the wind is light and variable. Unit vectors are used because this
module knows only angles; a speed-weighted direction is what binning the wind
components ``u`` and ``v`` as ordinary scalars gives.

Why this earns its own module
------------------------------
The overlap weighting must be *identical* to the scalar case, or the wind
direction reported for a cell would describe a different interval than the
methane in the same cell. So the overlap search lives in
:func:`tsara.core.support.overlap_pairs` and both aggregations call it; only
the arithmetic applied to the pairs differs. That shared seam is the reason
this is a sibling of :mod:`tsara.core.support` rather than a function inside
it: support knows about intervals and must not learn about angles.

No *scientific* threshold is applied
-------------------------------------
A direction with *R* = 0.001 is meaningless, and TSARA reports it anyway,
next to the *R* that says so. Picking a cut-off would put a magic number in
the library where the science belongs to the user, and the quality number is
right there.

There is one **numerical** threshold, and it is not a judgement about wind.
Directions that cancel mathematically do not cancel in floating point:
``sin(180°)`` is 1.22e-16, not zero, so a north/south pair leaves a residual
vector of length 6.1e-17 pointing due *east*, and ``atan2`` reports 90.000°
with complete confidence. Measured, the four compass points summed in three
different orders give 129.60°, 153.43° and 132.19° — a number that changes
when the samples are reordered is not a measurement. So a resultant length at
or below the rounding floor of the sum that produced it, :math:`N\epsilon`
for *N* contributing samples, is snapped to exactly zero and its direction
reported as ``nan``. The bound comes from float64, not from meteorology: it
is the level at which *R* is indistinguishable from cancellation given the
arithmetic used to compute it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.exceptions import TsaraError
from tsara.core.support import CellBounds, overlap_pairs

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = [
    "BinnedCircular",
    "CircularMean",
    "TsaraCircularError",
    "bin_circular_onto_cells",
    "circular_dispersion",
    "circular_mean",
    "wrap_degrees",
]

#: Full turn in degrees. Named so the wrapping arithmetic reads as intent.
FULL_TURN = 360.0

#: Rounding floor per contributing sample, used to decide that a resultant
#: length is cancellation rather than a direction.
#:
#: A weighted mean of *N* unit vectors accumulates rounding of order
#: :math:`N\epsilon`, so that is the level below which the residual vector is
#: made of nothing but arithmetic. See the module docstring for the measured
#: case that makes this necessary rather than fastidious.
RESULTANT_EPSILON = float(np.finfo(np.float64).eps)


class TsaraCircularError(TsaraError):
    """Raised when a circular statistic cannot be computed as asked.

    Its own type rather than a support error, because the failures are about
    angles: an array that does not match its weights, or a negative weight
    that would let one direction subtract another.
    """


@dataclass(frozen=True)
class CircularMean:
    r"""The vector average of a set of directions.

    Attributes
    ----------
    mean_deg : float
        Mean direction in ``[0, 360)``, or ``nan`` when the vectors cancel
        exactly and no direction exists.
    resultant_length : float
        Length of the mean unit vector, in ``[0, 1]``. The quality number:
        1 means every sample agreed, 0 means they cancelled. ``nan`` when
        nothing contributed at all, which is a different situation from 0 and
        must stay distinguishable from it.
    dispersion_deg : float
        Exact circular standard deviation, :math:`\\sqrt{-2\\ln R}` converted
        to degrees. Zero when all samples agree and ``inf`` when they cancel.
    n_source : int
        How many samples contributed, masked ones excluded.
    """

    mean_deg: float
    resultant_length: float
    dispersion_deg: float
    n_source: int


@dataclass(frozen=True)
class BinnedCircular:
    """One angular stream vector-averaged onto another stream's cells.

    The circular counterpart of
    :class:`~tsara.core.support.BinnedOntoCells`, with the same counts and
    coverage so that a caller handling both kinds of variable reads them the
    same way.

    Attributes
    ----------
    mean_deg : numpy.ndarray
        Mean direction per target cell, ``nan`` where nothing contributed or
        where the vectors cancelled exactly.
    resultant_length : numpy.ndarray
        Mean resultant length per cell, ``nan`` where nothing contributed.
    dispersion_deg : numpy.ndarray
        Exact circular standard deviation per cell, in degrees.
    n_source : numpy.ndarray
        Contributing source cells per target cell.
    coverage : numpy.ndarray
        Share of each target cell's width covered by contributing data.
    n_overlapping : numpy.ndarray
        Source cells overlapping at all, masked ones included. The difference
        from ``n_source`` separates "no data here" from "data rejected here".
    """

    mean_deg: npt.NDArray[np.float64]
    resultant_length: npt.NDArray[np.float64]
    dispersion_deg: npt.NDArray[np.float64]
    n_source: npt.NDArray[np.int64]
    coverage: npt.NDArray[np.float64]
    n_overlapping: npt.NDArray[np.int64]


def wrap_degrees(angles: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap angles into ``[0, 360)``.

    The compass convention, deliberately different from
    :func:`tsara.core.geodesy.wrap_longitude`, which wraps into
    ``[-180, 180)`` because a longitude is a signed offset from a meridian
    while a bearing is not. Two conventions, two functions, so neither has to
    be remembered as a special case of the other.

    Parameters
    ----------
    angles : array_like
        Angles in degrees, any magnitude or sign.

    Returns
    -------
    numpy.ndarray
        The same angles in ``[0, 360)``. ``nan`` passes through unchanged.

    Notes
    -----
    The modulo alone does not close the interval. A tiny negative angle wraps
    to a value a hair under a full turn, which rounds *up* to exactly 360.0 in
    float64 -- so ``-1e-17 % 360`` is ``360.0``, outside the range this
    function documents. Anything landing on a full turn is folded back to
    zero, which is the same direction.
    """
    wrapped = np.asarray(angles, dtype=np.float64) % FULL_TURN
    return np.asarray(np.where(wrapped >= FULL_TURN, 0.0, wrapped), dtype=np.float64)


def circular_dispersion(resultant_length: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""Return the exact circular standard deviation, in degrees.

    :math:`s = \sqrt{-2 \ln R}`, converted from radians. Exact for a wrapped
    normal distribution, whose resultant length is :math:`e^{-\sigma^2/2}`,
    and the standard definition (Mardia) otherwise.

    Parameters
    ----------
    resultant_length : array_like
        Mean resultant length(s) in ``[0, 1]``. Values marginally above 1 from
        floating-point rounding are clipped rather than producing ``nan``
        through the logarithm of a number greater than one.

    Returns
    -------
    numpy.ndarray
        Dispersion in degrees: 0 where ``R`` is 1, ``inf`` where it is 0, and
        ``nan`` where it is ``nan``. The infinity is meant literally — a
        uniform set of directions has no scale — and it is why this form is
        preferred to the bounded approximation it replaces.
    """
    r = np.asarray(resultant_length, dtype=np.float64)
    clipped = np.clip(r, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        # log(0) is -inf by design here, not an error: it is what an
        # arbitrarily wide direction distribution looks like.
        radians = np.sqrt(-2.0 * np.log(clipped))
    return np.asarray(np.degrees(radians), dtype=np.float64)


def _weighted_components(
    angles_deg: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Return the weighted sine, cosine and total weight of finite angles."""
    finite = np.isfinite(angles_deg) & (weights > 0)
    if not finite.any():
        return float("nan"), float("nan"), 0.0
    radians = np.radians(angles_deg[finite])
    w = weights[finite]
    total = float(w.sum())
    return float(np.sum(w * np.sin(radians))), float(np.sum(w * np.cos(radians))), total


def circular_mean(
    angles_deg: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> CircularMean:
    """Average directions as unit vectors.

    Each angle becomes a unit vector, the vectors are averaged with the given
    weights, and the mean direction is the angle of the result. The length of
    that result is the resultant length *R*, reported alongside because the
    direction alone does not say whether it means anything.

    Parameters
    ----------
    angles_deg : array_like
        Directions in degrees, any wrapping. ``nan`` entries are excluded
        along with their weights, which is what a QA/QC-masked sample looks
        like by the time it reaches here.
    weights : array_like, optional
        Non-negative weights, one per angle. ``None`` weights every angle
        equally. Normalized internally, so overlap in nanoseconds may be
        passed directly.

    Returns
    -------
    CircularMean
        Mean direction, resultant length, dispersion and contributing count.

    Raises
    ------
    TsaraCircularError
        If the arrays disagree in length or a weight is negative.
    """
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=np.float64))
    if angles.ndim != 1:
        raise TsaraCircularError(f"Angles must be one-dimensional, got shape {angles.shape}.")
    if weights is None:
        w = np.ones_like(angles)
    else:
        w = np.atleast_1d(np.asarray(weights, dtype=np.float64))
        if w.shape != angles.shape:
            raise TsaraCircularError(
                f"Got {w.size} weight(s) for {angles.size} angle(s); they must "
                "correspond one to one."
            )
        if np.any(w < 0):
            raise TsaraCircularError(
                "Weights must be non-negative; a negative weight would point one "
                "sample's direction backwards."
            )
    sin_sum, cos_sum, total = _weighted_components(angles, w)
    if total <= 0:
        return CircularMean(
            mean_deg=float("nan"),
            resultant_length=float("nan"),
            dispersion_deg=float("nan"),
            n_source=0,
        )
    n_source = int(np.count_nonzero(np.isfinite(angles) & (w > 0)))
    sin_mean, cos_mean = sin_sum / total, cos_sum / total
    resultant = float(min(np.hypot(sin_mean, cos_mean), 1.0))
    if resultant <= n_source * RESULTANT_EPSILON:
        # The vectors cancelled to within the rounding of the sum that added
        # them up. What is left points somewhere, confidently and
        # irreproducibly -- see the module docstring.
        return CircularMean(
            mean_deg=float("nan"),
            resultant_length=0.0,
            dispersion_deg=float("inf"),
            n_source=n_source,
        )
    return CircularMean(
        mean_deg=float(wrap_degrees(np.degrees(np.arctan2(sin_mean, cos_mean)))),
        resultant_length=resultant,
        dispersion_deg=float(circular_dispersion(resultant)),
        n_source=n_source,
    )


def bin_circular_onto_cells(
    source: CellBounds,
    angles_deg: npt.NDArray[np.float64],
    target: CellBounds,
) -> BinnedCircular:
    """Vector-average an angular stream onto another stream's cells.

    The circular counterpart of
    :func:`tsara.core.support.bin_onto_cells`, and weighted by the *same*
    overlaps — both call :func:`tsara.core.support.overlap_pairs`, so a cell's
    wind direction and its methane always describe the same interval.

    Parameters
    ----------
    source : CellBounds
        Cells of the angular stream. Must be sorted by start time.
    angles_deg : numpy.ndarray
        One direction per source cell, in degrees. ``nan`` contributes
        nothing and reduces coverage, exactly as a masked scalar does.
    target : CellBounds
        Cells to average onto.

    Returns
    -------
    BinnedCircular
        Per-cell mean direction, resultant length, dispersion, counts and
        coverage.

    Raises
    ------
    TsaraCircularError
        If ``angles_deg`` does not have one entry per source cell.
    """
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.shape != (len(source),):
        raise TsaraCircularError(
            f"bin_circular_onto_cells got {angles.size} angle(s) for {len(source)} "
            "source cell(s); they must correspond one to one."
        )
    n_target = len(target)
    mean_deg = np.full(n_target, np.nan, dtype=np.float64)
    resultant = np.full(n_target, np.nan, dtype=np.float64)
    dispersion = np.full(n_target, np.nan, dtype=np.float64)
    n_source = np.zeros(n_target, dtype=np.int64)
    coverage = np.zeros(n_target, dtype=np.float64)
    n_overlapping = np.zeros(n_target, dtype=np.int64)

    pairs = overlap_pairs(source, target)
    if pairs.overlap_ns.size == 0:
        return BinnedCircular(
            mean_deg=mean_deg,
            resultant_length=resultant,
            dispersion_deg=dispersion,
            n_source=n_source,
            coverage=coverage,
            n_overlapping=n_overlapping,
        )

    paired = angles[pairs.source_index]
    finite = np.isfinite(paired)
    weight = np.where(finite, pairs.overlap_ns, 0).astype(np.float64)
    # Zero the angle before taking sine and cosine: sin(nan) is nan, so a
    # zero weight alone would not keep a masked sample out of the sum.
    radians = np.radians(np.where(finite, paired, 0.0))

    sin_sum = np.bincount(pairs.target_index, weights=weight * np.sin(radians), minlength=n_target)
    cos_sum = np.bincount(pairs.target_index, weights=weight * np.cos(radians), minlength=n_target)
    weight_sum = np.bincount(pairs.target_index, weights=weight, minlength=n_target)
    n_source = np.bincount(
        pairs.target_index, weights=(weight > 0).astype(np.float64), minlength=n_target
    ).astype(np.int64)
    n_overlapping = np.bincount(
        pairs.target_index, weights=(pairs.overlap_ns > 0).astype(np.float64), minlength=n_target
    ).astype(np.int64)

    contributing = weight_sum > 0
    sin_mean = sin_sum[contributing] / weight_sum[contributing]
    cos_mean = cos_sum[contributing] / weight_sum[contributing]
    cell_resultant = np.minimum(np.hypot(sin_mean, cos_mean), 1.0)
    # A cell whose vectors cancelled to within the rounding of their own sum
    # has no direction: the residual points somewhere, confidently and
    # irreproducibly. Snapped to exactly zero so that the three reported
    # numbers agree with each other -- R of 0, a nan direction, and an
    # infinite dispersion all say the same thing.
    cancelled = cell_resultant <= n_source[contributing] * RESULTANT_EPSILON
    cell_resultant = np.where(cancelled, 0.0, cell_resultant)
    resultant[contributing] = cell_resultant
    dispersion[contributing] = circular_dispersion(cell_resultant)
    directions = np.full(sin_mean.shape, np.nan, dtype=np.float64)
    defined = ~cancelled
    directions[defined] = wrap_degrees(np.degrees(np.arctan2(sin_mean[defined], cos_mean[defined])))
    mean_deg[contributing] = directions

    target_width = target.width_ns.astype(np.float64)
    wide = target_width > 0
    coverage[wide] = weight_sum[wide] / target_width[wide]
    return BinnedCircular(
        mean_deg=mean_deg,
        resultant_length=resultant,
        dispersion_deg=dispersion,
        n_source=n_source,
        coverage=coverage,
        n_overlapping=n_overlapping,
    )
