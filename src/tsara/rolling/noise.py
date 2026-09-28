"""The noise scale of a variable at every reading: the sigma thresholds are quoted in.

Detection (Phase 6) finds plumes as enhancements above the noise, in units
of a sigma, and that sigma comes from the measurement-uncertainty system in
provenance order (``docs/METHODS.md`` §2.3, §6.8): the declared or reported
random sigma when the variable has one, otherwise an estimate from the data
itself by the estimator the configuration names (§2.5). This module is that
ladder, and the estimators, registered by name like the file readers and
the baseline methods.

The empirical estimators
------------------------
``diff_mad`` (the default) is the robust first-difference estimator: the
median absolute difference between consecutive readings, scaled to a sigma
(1.4826 for the MAD of a Gaussian, over √2 because a difference of two
independent readings has twice the variance). Differencing cancels
everything slow, so a broad plume barely moves it and a plume-dense record
does not inflate it; the price is that it sees only the random component,
which is what an empirical fallback is entitled to claim (§2.5). Each
difference is a value on the cell between the two readings' midpoints, and
it enters a window by overlap like any reading, through the one rolling
engine at q = 0.5. **A difference across a dropout is dropped**: two readings
farther apart than :data:`DROPOUT_SPACING_FACTOR` times the record's median
spacing between consecutive finite readings measure the air between them,
not the instrument. On spacing, not on cell width: the 07-18 drive's
analyzer reports every 2 or 3 s on 1 s cells (a median spacing of 2.00 s),
and a rule on width would drop every difference it has.

``mad`` is the rolling MAD of the signal about its rolling median, kept for
comparison: each reading's distance from the median of its own window, then
the median of those distances over the window. Taking the residual against
each reading's own window median removes what is broader than the window,
so it does not collapse outright on a plume; measured, it reads three to
four times the noise under a plume that fills most of a window, where the
difference estimator reads it within 15 to 60 % (§2.5).

Both are floored at the quantization scale δ/√12, δ being the reporting
resolution: a logger writing 0.01 ppm steps has a median absolute
difference of zero whenever more than half the window shares a value, and
a zero noise scale makes every reading a plume. δ is the variable's
declared ``quantization`` when the stream carries one, else the smallest
positive gap between its distinct values, which is exact for a quantized
record and negligible for a continuous one. A window holding fewer than
:data:`MIN_NOISE_SAMPLES` samples is blank: a median of a handful of
differences is a number, not a noise scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

from tsara.core.naming import sigma_rand_name
from tsara.core.support import CellBounds
from tsara.rolling.quantile import MAX_BLOCK_ELEMENTS, rolling_quantile
from tsara.rolling.windows import TsaraRollingError, window_cells

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt
    import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "DROPOUT_SPACING_FACTOR",
    "MAD_TO_SIGMA",
    "MIN_NOISE_SAMPLES",
    "NoiseEstimatorFunction",
    "NoiseRequest",
    "NoiseResult",
    "available_noise_estimators",
    "get_noise_estimator",
    "noise_by_diff_mad",
    "noise_by_mad",
    "noise_scale",
    "register_noise_estimator",
]


#: The MAD of a Gaussian is 0.6745 sigma; this is its reciprocal (§2.5).
MAD_TO_SIGMA = 1.4826

#: Two consecutive finite readings farther apart than this many times the
#: record's median spacing straddle a dropout, and their difference measures
#: the air between them rather than the instrument, so it is dropped. A
#: constant, not a knob: 1.5 keeps every jittered step (the archive's jitter
#: tops out at a few percent) and drops a single missing row (a step of 2).
DROPOUT_SPACING_FACTOR = 1.5

#: Fewest samples a noise window may hold and still report a scale. A MAD
#: is about 37 % efficient at the Gaussian, so ten differences already give
#: a figure jittering by a third; fewer is a number, not a noise scale.
MIN_NOISE_SAMPLES = 10


@dataclass(frozen=True, eq=False)
class NoiseRequest:
    """Everything an estimator is given for one variable.

    Attributes
    ----------
    readings : CellBounds
        The stream's cells.
    values : numpy.ndarray
        The variable's readings, ``nan`` where masked.
    window_ns : int
        The noise window, in nanoseconds.
    block_elements : int
        The rolling engine's per-block budget.
    """

    readings: CellBounds
    values: npt.NDArray[np.float64]
    window_ns: int
    block_elements: int


@dataclass(frozen=True, eq=False)
class NoiseResult:
    """The noise scale at every reading and where it came from.

    Attributes
    ----------
    sigma : numpy.ndarray
        One-sigma random noise scale per reading; ``nan`` where none could be
        given.
    provenance : str
        ``declared``, ``reported`` or ``empirical`` (§2.4).
    estimator : str or None
        The registered estimator used, or ``None`` for a declared or
        reported figure.
    resolution : float
        The reporting resolution δ the floor was built from; ``nan`` when
        no floor was applied (a declared figure is not floored).
    floor_fraction : float
        Share of readings whose estimate the floor raised.
    blank_fraction : float
        Share of readings with no scale.
    """

    sigma: npt.NDArray[np.float64]
    provenance: str
    estimator: str | None
    resolution: float
    floor_fraction: float
    blank_fraction: float


class NoiseEstimatorFunction(Protocol):
    """The signature every registered noise estimator has."""

    def __call__(self, request: NoiseRequest, /) -> npt.NDArray[np.float64]:
        """Return the one-sigma noise scale at every reading, before the floor."""
        ...  # pragma: no cover


#: Registered estimators keyed by the configuration's ``noise_estimator`` name.
_ESTIMATORS: dict[str, NoiseEstimatorFunction] = {}

#: Bound to the protocol so the decorator returns the same function type.
E = TypeVar("E", bound=NoiseEstimatorFunction)


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def noise_scale(
    stream: xr.Dataset,
    variable: str,
    readings: CellBounds,
    *,
    estimator: str,
    window_ns: int,
    block_elements: int = MAX_BLOCK_ELEMENTS,
) -> NoiseResult:
    """Return a variable's noise scale at every reading, by the provenance ladder.

    Parameters
    ----------
    stream : xarray.Dataset
        The instrument's stream.
    variable : str
        The variable.
    readings : CellBounds
        The stream's cells.
    estimator : str
        The registered estimator to use when the variable declares no random
        sigma (``DetectionConfig.noise_estimator``).
    window_ns : int
        The window for that estimator (``DetectionConfig.noise_window``).
    block_elements : int, optional
        The rolling engine's per-block budget.

    Returns
    -------
    NoiseResult
        The scale, its provenance, and the floor's record.

    Raises
    ------
    TsaraRollingError
        If no estimator is registered under ``estimator``.
    """
    declared = sigma_rand_name(variable)
    if declared in stream.data_vars:
        # The top of the ladder: the instrument said. Used as it stands,
        # provenance carried, no floor -- a declared figure is not an estimate.
        sigma = np.asarray(stream[declared].values, dtype=np.float64)
        provenance = str(stream[declared].attrs.get("uncertainty_provenance", "declared"))
        return NoiseResult(
            sigma=sigma,
            provenance=provenance,
            estimator=None,
            resolution=float("nan"),
            floor_fraction=0.0,
            blank_fraction=float(np.mean(~np.isfinite(sigma))) if sigma.size else 0.0,
        )
    values = np.asarray(stream[variable].values, dtype=np.float64)
    estimate = get_noise_estimator(estimator)(
        NoiseRequest(
            readings=readings, values=values, window_ns=window_ns, block_elements=block_elements
        )
    )
    resolution = _resolution(stream, variable, values)
    floor = resolution / np.sqrt(12.0)
    raised = np.isfinite(estimate) & (estimate < floor)
    floored = np.where(raised, floor, estimate)
    return NoiseResult(
        sigma=floored,
        provenance="empirical",
        estimator=estimator,
        resolution=resolution,
        floor_fraction=float(np.mean(raised)) if raised.size else 0.0,
        blank_fraction=float(np.mean(~np.isfinite(floored))) if floored.size else 0.0,
    )


def _resolution(stream: xr.Dataset, variable: str, values: npt.NDArray[np.float64]) -> float:
    """Return the reporting resolution δ: declared on the variable, else detected.

    Detected as the smallest positive gap between the record's distinct
    finite values: exact for a record written in steps, and a number far
    below any noise for a continuous one, where the floor then does nothing.
    Zero when the record holds fewer than two distinct values, so the floor
    is zero rather than a guess.
    """
    declared = stream[variable].attrs.get("quantization")
    if declared is not None:
        return float(declared)
    distinct = np.unique(values[np.isfinite(values)])
    if distinct.size < 2:
        return 0.0
    return float(np.min(np.diff(distinct)))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def register_noise_estimator(name: str, *, replace: bool = False) -> Callable[[E], E]:
    """Register a noise estimator under its configuration name.

    Used as a decorator, on the terms of
    :func:`~tsara.rolling.methods.register_baseline_method`: a name is
    registered once unless ``replace=True`` says so, and an override is
    logged.

    Parameters
    ----------
    name : str
        The estimator's name, the ``noise_estimator`` value of the
        configuration.
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
        raise ValueError("Noise estimator name must be a non-empty string.")

    def decorator(func: E) -> E:
        existing = _ESTIMATORS.get(name)
        if existing is not None:
            if not replace:
                raise ValueError(
                    f"A noise estimator is already registered as '{name}' "
                    f"({_describe(existing)}). Pass replace=True if you meant to override it."
                )
            logger.warning(
                "Replacing noise estimator '%s' (%s) with %s.",
                name,
                _describe(existing),
                _describe(func),
            )
        _ESTIMATORS[name] = func
        return func

    return decorator


def _describe(func: object) -> str:
    """Return ``module.qualname`` for an estimator, for messages."""
    return f"{getattr(func, '__module__', '?')}.{getattr(func, '__qualname__', '?')}"


def available_noise_estimators() -> tuple[str, ...]:
    """List the registered estimator names, sorted.

    Returns
    -------
    tuple of str
        e.g. ``('diff_mad', 'mad')``.
    """
    return tuple(sorted(_ESTIMATORS))


def get_noise_estimator(name: str) -> NoiseEstimatorFunction:
    """Look up a noise estimator by name.

    Parameters
    ----------
    name : str
        The ``noise_estimator`` value of the configuration.

    Returns
    -------
    NoiseEstimatorFunction
        The registered estimator.

    Raises
    ------
    TsaraRollingError
        If nothing is registered under ``name``; the message lists what is.
    """
    try:
        return _ESTIMATORS[name]
    except KeyError:
        raise TsaraRollingError(
            f"No noise estimator registered as '{name}'. Available: "
            f"{list(available_noise_estimators())}. An estimator must have been imported "
            "to be registered."
        ) from None


# ---------------------------------------------------------------------------
# The two estimators
# ---------------------------------------------------------------------------


@register_noise_estimator("diff_mad")
def noise_by_diff_mad(request: NoiseRequest, /) -> npt.NDArray[np.float64]:
    """Return the robust first-difference noise scale at every reading (§2.5).

    Parameters
    ----------
    request : NoiseRequest
        The readings, their values and the window.

    Returns
    -------
    numpy.ndarray
        One-sigma per reading; ``nan`` where the window held fewer than
        :data:`MIN_NOISE_SAMPLES` differences.
    """
    finite = np.flatnonzero(np.isfinite(request.values))
    windows = window_cells(request.readings, request.window_ns)
    blank = np.full(len(request.readings), np.nan, dtype=np.float64)
    if finite.size < 2:
        return blank
    midpoints = request.readings.midpoint_ns[finite]
    spacing = np.diff(midpoints)
    # Differences that straddle a dropout measure the air, and are dropped.
    # At least one is always kept: the median spacing is one of them.
    keep = spacing <= DROPOUT_SPACING_FACTOR * float(np.median(spacing))
    differences = np.abs(np.diff(request.values[finite]))[keep]
    # A difference is a value on the cell between the two readings' midpoints,
    # and enters a window by overlap like any reading.
    cells = CellBounds(start_ns=midpoints[:-1][keep], stop_ns=midpoints[1:][keep])
    rolled = rolling_quantile(
        cells, differences, windows, [0.5], block_elements=request.block_elements
    )
    sigma = MAD_TO_SIGMA * rolled.values[:, 0] / np.sqrt(2.0)
    return np.where(rolled.n_readings >= MIN_NOISE_SAMPLES, sigma, np.nan)


@register_noise_estimator("mad")
def noise_by_mad(request: NoiseRequest, /) -> npt.NDArray[np.float64]:
    """Return the rolling MAD of the signal about its rolling median (§2.5).

    Two passes of the rolling engine at q = 0.5: the median of the readings
    in each window, then the median of each reading's absolute distance from
    the median at its own cell. Kept for comparison with ``diff_mad``: under
    a plume that fills most of a window it reads three to four times the
    noise, measured (§2.5).

    Parameters
    ----------
    request : NoiseRequest
        The readings, their values and the window.

    Returns
    -------
    numpy.ndarray
        One-sigma per reading; ``nan`` where the window held fewer than
        :data:`MIN_NOISE_SAMPLES` readings.
    """
    windows = window_cells(request.readings, request.window_ns)
    centre = rolling_quantile(
        request.readings, request.values, windows, [0.5], block_elements=request.block_elements
    )
    residual = np.abs(request.values - centre.values[:, 0])
    rolled = rolling_quantile(
        request.readings, residual, windows, [0.5], block_elements=request.block_elements
    )
    sigma = MAD_TO_SIGMA * rolled.values[:, 0]
    return np.where(rolled.n_readings >= MIN_NOISE_SAMPLES, sigma, np.nan)
