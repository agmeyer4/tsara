"""The clean level and spread: the events stage's description of plume-free air.

Plumes only add, so an enhancement's plume-free readings are the ones below
its most common value (``docs/METHODS.md`` §6.8). Per variable, per record
and per point of the baseline sweep, the **clean level** *m* is that most
common value, the **clean spread** is

    s = 1.4826 · median(m − Δ  for the readings with Δ < m),

and the stage's thresholds are multiples of *s* above *m*. Neither is a
measurement uncertainty (§2.3): *s* holds the instrument's noise plus the
background's wobble at the window's scale, and *m* is measured rather than
modelled, since a longer window's low quantile reaches further
into that wobble (on the 07-18 methane, 0.8, 2.4 and 3.1 ppb at 2, 10 and
60 min).

Which estimator finds the most common value is registered by name, like the
file readers and the baseline methods, so a better one replaces it without a
redesign. One is registered: the **half-sample mode** (Bickel and Frühwirth
2006), chosen because the archive is plume-dense and robustness decided it.
The midpoint of the shortest half, steadier on clean records, was approved
and withdrawn the same day when it collapsed on a record where plumes are
most of the readings (§6.8's estimator table).

Two rules bound what is reported. The spread is floored at δ/√12, the
standard deviation of rounding to a step δ, so that a record written in
steps coarser than its noise cannot report a spread of zero (§2.5): δ is
the variable's declared ``quantization`` when it has one, else the smallest
positive gap between the record's distinct readings. And a record with fewer
than ``min_clean_readings`` readings below its level is blank: 100 is a
floor for having a scale at all, not a guarantee of the chance rate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from tsara.events.records import TsaraEventError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from tsara.events.records import Records

logger = logging.getLogger(__name__)

__all__ = [
    "CleanAir",
    "CleanDescription",
    "CleanLevelEstimator",
    "MAD_TO_SIGMA",
    "available_clean_level_estimators",
    "clean_air",
    "describe_clean_air",
    "get_clean_level_estimator",
    "half_sample_mode",
    "quantization_step",
    "register_clean_level_estimator",
]

#: The Gaussian consistency constant of a median absolute distance, 1/Φ⁻¹(3/4).
#: The synthetic generator's profiling carries the same number; the stages
#: never import each other, and a textbook constant does not drift.
MAD_TO_SIGMA = 1.4826

#: A clean-level estimator: finite enhancements in, their most common value out.
CleanLevelEstimator = Callable[["npt.NDArray[np.float64]"], float]

#: Registered estimators keyed by ``EventsConfig.clean_level_estimator``.
#: Module-private: mutation goes through :func:`register_clean_level_estimator`.
_ESTIMATORS: dict[str, CleanLevelEstimator] = {}

#: Bound so the decorator returns the same function type it received.
E = TypeVar("E", bound=CleanLevelEstimator)


@dataclass(frozen=True)
class CleanDescription:
    """Plume-free air in one sample: one record at one sweep point.

    Attributes
    ----------
    level : float
        The clean level, the sample's most common value; NaN for an empty
        sample.
    spread : float
        The clean spread, floored; NaN when nothing lies below the level.
    n_below : int
        How many readings lie strictly below the level: the ones the
        spread is measured on.
    floored : bool
        Whether the floor raised the spread.
    """

    level: float
    spread: float
    n_below: int
    floored: bool


@dataclass(frozen=True, eq=False)
class CleanAir:
    """A variable's clean level and spread, per record and sweep point.

    Each array is shaped ``(record, *sweep)``, the sweep being whatever
    trailing dimensions the enhancement had (the baseline's window and
    quantile).

    Attributes
    ----------
    level, spread : numpy.ndarray of float64
        The clean level and spread; NaN where ``blank``.
    n_below : numpy.ndarray of int64
        Readings below the level, reported blank or not, since it is the
        reason a blank record is blank.
    floored : numpy.ndarray of bool
        Where the quantization floor raised the spread.
    blank : numpy.ndarray of bool
        Where the record held fewer than ``min_clean_readings`` readings
        below its level.
    floor : numpy.ndarray of float64
        Per record, the floor δ/√12 the spread was held to.
    min_clean_readings : int
        The count the blanks were judged against.
    """

    level: npt.NDArray[np.float64]
    spread: npt.NDArray[np.float64]
    n_below: npt.NDArray[np.int64]
    floored: npt.NDArray[np.bool_]
    blank: npt.NDArray[np.bool_]
    floor: npt.NDArray[np.float64]
    min_clean_readings: int


# ---------------------------------------------------------------------------
# A whole variable
# ---------------------------------------------------------------------------


def clean_air(
    enhancement: npt.NDArray[np.float64],
    readings: npt.NDArray[np.float64],
    records: Records,
    *,
    estimator: str,
    min_clean_readings: int,
    quantization: float | None = None,
) -> CleanAir:
    """Describe a variable's plume-free air in every record at every sweep point.

    Parameters
    ----------
    enhancement : numpy.ndarray of float64
        Shaped ``(reading, *sweep)``, as the baseline state holds it.
    readings : numpy.ndarray of float64
        The variable's readings, shaped ``(reading,)``, for δ.
    records : Records
        The variable's records (:func:`~tsara.events.records.find_records`).
    estimator : str
        The registered clean-level estimator.
    min_clean_readings : int
        Fewer readings than this below its level, and a record is blank at
        that sweep point.
    quantization : float, optional
        The variable's declared ``quantization``.

    Returns
    -------
    CleanAir
        Level, spread, count, floor flag and blank flag, shaped
        ``(record, *sweep)``, and the floor per record.

    Raises
    ------
    TsaraEventError
        If the arrays do not match the records, or the estimator is not
        registered.
    """
    values = np.asarray(enhancement, dtype=np.float64)
    x = np.asarray(readings, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != records.index.size or x.shape != records.index.shape:
        raise TsaraEventError(
            f"Enhancement of shape {values.shape} and readings of shape {x.shape} do not match "
            f"{records.index.size} readings."
        )
    get_clean_level_estimator(estimator)  # refuse an unknown name before any work
    sweep = values.shape[1:]
    flat = values.reshape(values.shape[0], -1)
    shape = (records.n, flat.shape[1])
    level = np.full(shape, np.nan)
    spread = np.full(shape, np.nan)
    n_below = np.zeros(shape, dtype=np.int64)
    floored = np.zeros(shape, dtype=bool)
    floor = np.empty(records.n)
    for r in range(records.n):
        rows = records.index == r
        floor[r] = quantization_step(x[rows], quantization) / np.sqrt(12.0)
        for point in range(flat.shape[1]):
            found = describe_clean_air(flat[rows, point], estimator=estimator, floor=floor[r])
            level[r, point], spread[r, point] = found.level, found.spread
            n_below[r, point], floored[r, point] = found.n_below, found.floored
    blank = n_below < min_clean_readings
    level[blank] = np.nan
    spread[blank] = np.nan
    floored[blank] = False
    return CleanAir(
        level=level.reshape(records.n, *sweep),
        spread=spread.reshape(records.n, *sweep),
        n_below=n_below.reshape(records.n, *sweep),
        floored=floored.reshape(records.n, *sweep),
        blank=blank.reshape(records.n, *sweep),
        floor=floor,
        min_clean_readings=min_clean_readings,
    )


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------


def describe_clean_air(
    values: npt.NDArray[np.float64],
    *,
    estimator: str = "half_sample_mode",
    floor: float = 0.0,
) -> CleanDescription:
    """Describe the plume-free air in one sample of enhancements.

    Parameters
    ----------
    values : numpy.ndarray of float64
        The enhancements of one record at one sweep point; non-finite ones
        are ignored.
    estimator : str, optional
        The registered clean-level estimator.
    floor : float, optional
        The least spread to report, δ/√12 (:func:`quantization_step`).

    Returns
    -------
    CleanDescription
        The level, the spread, the count below the level and whether the
        floor acted. No count rule is applied here; :func:`clean_air` does
        that.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    level = get_clean_level_estimator(estimator)(finite)
    below = finite[finite < level]
    if below.size == 0:
        return CleanDescription(level=level, spread=float("nan"), n_below=0, floored=False)
    measured = MAD_TO_SIGMA * float(np.median(level - below))
    return CleanDescription(
        level=level,
        spread=max(measured, floor),
        n_below=int(below.size),
        floored=bool(measured < floor),
    )


def quantization_step(readings: npt.NDArray[np.float64], declared: float | None = None) -> float:
    """Return δ, the step a record's readings are written in.

    Parameters
    ----------
    readings : numpy.ndarray of float64
        One record's readings (not their enhancements, which a baseline
        between two steps moves off the grid).
    declared : float, optional
        The variable's declared ``quantization``, which wins when given.

    Returns
    -------
    float
        ``declared``, else the smallest positive gap between the distinct
        finite readings: exact for a record written in steps, negligible
        for a continuous one; 0.0 with fewer than two distinct readings.
    """
    if declared is not None:
        return float(declared)
    distinct = np.unique(readings[np.isfinite(readings)])
    if distinct.size < 2:
        return 0.0
    return float(np.min(np.diff(distinct)))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def register_clean_level_estimator(name: str, *, replace: bool = False) -> Callable[[E], E]:
    """Register a clean-level estimator under its configuration name.

    Used as a decorator, as the baseline methods are; re-registering a name
    is refused unless ``replace=True`` says so, since a silent overwrite
    would make the events depend on import order.

    Parameters
    ----------
    name : str
        The ``EventsConfig.clean_level_estimator`` value.
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
        raise ValueError("Clean-level estimator name must be a non-empty string.")

    def decorator(func: E) -> E:
        if name in _ESTIMATORS and not replace:
            raise ValueError(
                f"A clean-level estimator is already registered as '{name}'. Pass "
                "replace=True if you meant to override it."
            )
        if name in _ESTIMATORS:
            logger.warning("Replacing clean-level estimator '%s'.", name)
        _ESTIMATORS[name] = func
        return func

    return decorator


def available_clean_level_estimators() -> tuple[str, ...]:
    """List the registered estimator names, sorted.

    Returns
    -------
    tuple of str
        e.g. ``('half_sample_mode',)``.
    """
    return tuple(sorted(_ESTIMATORS))


def get_clean_level_estimator(name: str) -> CleanLevelEstimator:
    """Look up a clean-level estimator by name.

    Parameters
    ----------
    name : str
        The ``EventsConfig.clean_level_estimator`` value.

    Returns
    -------
    callable
        The registered estimator.

    Raises
    ------
    TsaraEventError
        If nothing is registered under ``name``; the message lists what is.
    """
    try:
        return _ESTIMATORS[name]
    except KeyError:
        raise TsaraEventError(
            f"No clean-level estimator registered as '{name}'. Available: "
            f"{list(available_clean_level_estimators())}."
        ) from None


# ---------------------------------------------------------------------------
# The estimators
# ---------------------------------------------------------------------------


@register_clean_level_estimator("half_sample_mode")
def half_sample_mode(values: npt.NDArray[np.float64]) -> float:
    """Return the half-sample mode of finite values (Bickel and Frühwirth 2006).

    Sorted, the values are narrowed to the shortest interval holding half of
    them, ``ceil(n / 2)``, then to the shortest holding half of those, until
    three or fewer remain. Of three, the mode is the mean of the closer pair,
    or the middle value when the two gaps are equal; of two, their mean; of
    one, itself. Where several intervals are equally short, the earliest
    (lowest) is taken, which on a record written in steps keeps a plateau of
    identical readings whole.

    Parameters
    ----------
    values : numpy.ndarray of float64
        Finite values, in any order.

    Returns
    -------
    float
        The mode; NaN for no values.
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0:
        return float("nan")
    while x.size > 3:
        half = (x.size + 1) // 2
        # Width of every interval of `half` consecutive sorted values.
        widths = x[half - 1 :] - x[: x.size - half + 1]
        first = int(np.argmin(widths))
        x = x[first : first + half]
    if x.size < 3:
        return float(np.mean(x))
    low, high = x[1] - x[0], x[2] - x[1]
    if low < high:
        return float((x[0] + x[1]) / 2)
    if high < low:
        return float((x[1] + x[2]) / 2)
    return float(x[1])
