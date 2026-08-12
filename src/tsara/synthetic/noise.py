r"""Injecting measurement error with a *known* two-component structure.

This module is the ground-truth counterpart to TSARA's uncertainty model
(METHODS.md §2). It manufactures error whose decomposition is known exactly,
so that every downstream claim about uncertainty — that random error averages
down as :math:`1/\sqrt{N_{\mathrm{eff}}}`, that systematic error does not,
that a reported confidence interval achieves its nominal coverage — becomes a
testable proposition rather than an assertion.

The two components differ in *how they are drawn*, which is the entire point:

* **random** — one independent draw per sample (optionally AR(1)-correlated
  with a known timescale tau). Averaging N samples shrinks it.
* **systematic** — drawn **once per species per run** as a pair of
  coefficients, then applied to every sample. This produces a rank-1 error
  structure with correlation exactly 1 between every pair of points, which is
  precisely METHODS.md §3.3's "fully correlated" case. Averaging a million
  samples does not reduce it by a single part per billion, and any pipeline
  that claims otherwise will be caught by data generated here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.timebase import epoch_s as _epoch_s
from tsara.synthetic.config import TrueComponent, TrueUncertainty

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

logger = logging.getLogger(__name__)

#: Tolerance (seconds) within which sampling is treated as uniform, enabling
#: the vectorized AR(1) path.
_UNIFORM_DT_TOLERANCE_S = 1e-9


@dataclass(frozen=True, eq=False)
class NoiseRealization:
    """One component's injected error and the sigma that generated it.

    Attributes
    ----------
    error : numpy.ndarray
        The error actually added to the signal.
    sigma : numpy.ndarray
        Per-point 1-sigma used to draw it — the *true* uncertainty.
    reported_sigma : numpy.ndarray or None
        What the instrument publishes, i.e. ``sigma * report_bias``. None
        when the component has no ``report_as`` column, meaning the error is
        real but undeclared and downstream code must fall back to empirical
        estimation.
    """

    error: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    reported_sigma: npt.NDArray[np.float64] | None


def draw_random_error(
    values: npt.NDArray[np.float64],
    component: TrueComponent,
    times: pd.DatetimeIndex,
    rng: np.random.Generator,
    decorrelation_timescale_s: float | None = None,
) -> NoiseRealization:
    r"""Draw the uncorrelated (or AR(1)-correlated) random error component.

    With no ``decorrelation_timescale_s`` the draws are i.i.d. Gaussian with
    the per-point sigma from
    :meth:`~tsara.synthetic.config.TrueComponent.sigma`. With one, the
    *standardized* error follows an AR(1) process with lag correlation
    :math:`\rho = e^{-\Delta t/\tau}` (METHODS.md §3.4) before being scaled
    to the per-point sigma — so the marginal variance is unchanged and only
    the correlation structure differs. That separation matters: it means a
    test can vary tau alone and watch an ``N_eff`` estimator respond, without
    the noise amplitude moving underneath it.

    Parameters
    ----------
    values : numpy.ndarray
        True (noise-free) signal, used for the relative sigma term.
    component : TrueComponent
        Random component parameters.
    times : pandas.DatetimeIndex
        Sample times, needed for the AR(1) correlation between samples.
    rng : numpy.random.Generator
        Source of randomness.
    decorrelation_timescale_s : float, optional
        AR(1) timescale in seconds. None gives white noise.

    Returns
    -------
    NoiseRealization
        Error, true sigma, and optionally the reported sigma.
    """
    sigma = np.asarray(component.sigma(values), dtype=np.float64)

    if decorrelation_timescale_s is None or values.size < 2:
        standardized = rng.normal(size=values.shape)
    else:
        standardized = _ar1_standardized(times, decorrelation_timescale_s, rng)

    error = sigma * standardized
    reported = None if component.report_as is None else sigma * component.report_bias
    return NoiseRealization(error=error, sigma=sigma, reported_sigma=reported)


def _ar1_standardized(
    times: pd.DatetimeIndex,
    tau_s: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    r"""Generate a unit-variance AR(1) series on (possibly irregular) ``times``.

    The recursion is

    .. math::

        e_i = \rho_i e_{i-1} + \sqrt{1-\rho_i^2}\,w_i,
        \qquad \rho_i = e^{-\Delta t_i/\tau}

    with :math:`e_0 \sim N(0,1)`, which is stationary with unit marginal
    variance at every point (so the caller can scale by any per-point sigma
    without distorting it).

    Two implementations, selected automatically:

    * **Uniform sampling** — rho is constant, so the recursion is a
      first-order IIR filter and ``scipy.signal.lfilter`` runs it vectorized.
      The filter's initial condition is seeded from a stationary draw so the
      series does not start at zero and "warm up" (which would leave the
      first few hundred samples with the wrong variance).
    * **Irregular sampling** — rho varies per step, so an explicit loop is
      used. This is the exact computation, deliberately preferred over
      approximating rho from a median interval: silently assuming regularity
      is exactly the class of hidden assumption this package refuses to make.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Sample times.
    tau_s : float
        Decorrelation timescale, seconds.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        Unit-variance AR(1) series, same length as ``times``.
    """
    from scipy.signal import lfilter

    epoch_s = _epoch_s(times)
    dt_s = np.diff(epoch_s)
    n = epoch_s.size

    white = rng.normal(size=n)
    initial = float(rng.normal())

    uniform = bool(np.all(np.abs(dt_s - dt_s[0]) < _UNIFORM_DT_TOLERANCE_S))
    if uniform:
        rho = math.exp(-float(dt_s[0]) / tau_s)
        innovation_scale = math.sqrt(max(0.0, 1.0 - rho**2))
        # y[n] = innovation_scale * w[n] + rho * y[n-1], started from a
        # stationary value so there is no burn-in transient.
        filtered, _ = lfilter([innovation_scale], [1.0, -rho], white, zi=np.array([rho * initial]))
        return np.asarray(filtered, dtype=np.float64)

    rho_all = np.exp(-dt_s / tau_s)
    innovation_scales = np.sqrt(np.clip(1.0 - rho_all**2, 0.0, None))
    out = np.empty(n, dtype=np.float64)
    out[0] = initial
    for i in range(1, n):
        out[i] = rho_all[i - 1] * out[i - 1] + innovation_scales[i - 1] * white[i]
    return out


def draw_systematic_error(
    values: npt.NDArray[np.float64],
    component: TrueComponent,
    rng: np.random.Generator,
) -> tuple[NoiseRealization, tuple[float, float]]:
    r"""Draw the fully correlated (systematic) error component.

    Two standard-normal coefficients are drawn **once** and applied to the
    whole record:

    .. math::

        e^{\mathrm{sys}}_i = a\,g_{\mathrm{abs}}
        + r\,x_i\,g_{\mathrm{rel}}

    This is a genuine calibration error — an offset plus a scale error — not
    a slowly varying noise process. Its per-point magnitude still matches
    :math:`\sqrt{a^2 + (r x_i)^2}`, so it is directly comparable with the
    random component, but every pair of points is perfectly correlated, so it
    survives averaging intact.

    The two coefficients are returned so the generator can record them in the
    stream metadata: a test that wants to verify a systematic error was
    correctly *propagated* (rather than merely present) needs to know which
    realization it got.

    Parameters
    ----------
    values : numpy.ndarray
        True (noise-free) signal, used for the relative term.
    component : TrueComponent
        Systematic component parameters.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    NoiseRealization
        Error, true per-point sigma, and optionally the reported sigma.
    tuple of float
        The realized ``(g_abs, g_rel)`` standard-normal coefficients.
    """
    g_abs = float(rng.normal())
    g_rel = float(rng.normal())

    error = component.absolute * g_abs + component.relative * values * g_rel
    sigma = np.asarray(component.sigma(values), dtype=np.float64)
    reported = None if component.report_as is None else sigma * component.report_bias

    return (
        NoiseRealization(
            error=np.asarray(error, dtype=np.float64), sigma=sigma, reported_sigma=reported
        ),
        (g_abs, g_rel),
    )


@dataclass(frozen=True, eq=False)
class AppliedUncertainty:
    """The result of injecting a complete error budget into one species.

    The three output channels are kept separate rather than merged into one
    dict because they land in different places in the emitted stream, and
    conflating them would let a reported column silently overwrite a truth
    array:

    Attributes
    ----------
    values : numpy.ndarray
        The observable signal, i.e. truth plus injected error. This is the
        only array the analysis pipeline is ever allowed to see.
    sigma_rand, sigma_sys : numpy.ndarray or None
        The *true* per-point sigmas of each component. Published under
        ``truth_``-prefixed names so they are trivially excluded from any
        pipeline-facing view, and never confused with what the instrument
        claims about itself.
    reported : dict of str to numpy.ndarray
        Instrument-published per-point sigma columns, keyed by the exact
        ``report_as`` name a manifest would reference. Deliberately *not*
        prefixed: these are raw-file columns, and they may be biased away
        from the true sigma via ``report_bias``.
    scalars : dict of str to float
        Provenance values for the variable's attrs — the realized
        standard-normal coefficients behind the systematic component, without
        which a test could confirm systematic error was *present* but not
        that it was correctly *propagated*.
    """

    values: npt.NDArray[np.float64]
    sigma_rand: npt.NDArray[np.float64] | None
    sigma_sys: npt.NDArray[np.float64] | None
    reported: dict[str, npt.NDArray[np.float64]]
    scalars: dict[str, float]


def apply_uncertainty(
    values: npt.NDArray[np.float64],
    uncertainty: TrueUncertainty | None,
    times: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> AppliedUncertainty:
    """Apply a complete error budget to a noise-free signal.

    Parameters
    ----------
    values : numpy.ndarray
        True (noise-free) signal.
    uncertainty : TrueUncertainty or None
        The budget to inject. None leaves the signal untouched — a
        deliberately available case, since a noise-free stream isolates
        algorithmic error from measurement error.
    times : pandas.DatetimeIndex
        Sample times (needed for AR(1) correlation).
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    AppliedUncertainty
        Observable signal plus the truth and reported channels.
    """
    import pandas as pd

    reported: dict[str, npt.NDArray[np.float64]] = {}
    scalars: dict[str, float] = {}
    if uncertainty is None:
        return AppliedUncertainty(
            values=values.copy(),
            sigma_rand=None,
            sigma_sys=None,
            reported=reported,
            scalars=scalars,
        )

    total = values.copy()
    sigma_rand: npt.NDArray[np.float64] | None = None
    sigma_sys: npt.NDArray[np.float64] | None = None

    if uncertainty.random is not None:
        tau_s: float | None = None
        if uncertainty.decorrelation_timescale is not None:
            tau_s = float(pd.Timedelta(uncertainty.decorrelation_timescale).total_seconds())
        realization = draw_random_error(
            values, uncertainty.random, times, rng, decorrelation_timescale_s=tau_s
        )
        total = total + realization.error
        sigma_rand = realization.sigma
        if realization.reported_sigma is not None:
            # Guarded: reported_sigma is non-None exactly when report_as is.
            reported[str(uncertainty.random.report_as)] = realization.reported_sigma

    if uncertainty.systematic is not None:
        sys_realization, (g_abs, g_rel) = draw_systematic_error(values, uncertainty.systematic, rng)
        total = total + sys_realization.error
        sigma_sys = sys_realization.sigma
        scalars["true_sys_abs_draw"] = g_abs
        scalars["true_sys_rel_draw"] = g_rel
        if sys_realization.reported_sigma is not None:
            reported[str(uncertainty.systematic.report_as)] = sys_realization.reported_sigma

    return AppliedUncertainty(
        values=total,
        sigma_rand=sigma_rand,
        sigma_sys=sigma_sys,
        reported=reported,
        scalars=scalars,
    )


def quantize(values: npt.NDArray[np.float64], resolution: float) -> npt.NDArray[np.float64]:
    r"""Round values to a fixed reporting resolution.

    Reproduces a logger that writes, say, 0.01 ppm steps. This is a required
    adversarial case (CLAUDE.md Phase 2): every median-based noise estimator
    collapses to *exactly zero* once more than half a window shares a single
    value, and a zero noise scale makes every point a detection. METHODS.md
    §2.5 specifies the guard (a floor at :math:`\delta/\sqrt{12}`, the
    standard deviation of uniform rounding error); data generated here is how
    that guard gets tested rather than assumed.

    Parameters
    ----------
    values : numpy.ndarray
        Values to round.
    resolution : float
        Reporting step. Must be positive.

    Returns
    -------
    numpy.ndarray
        Rounded values.

    Raises
    ------
    ValueError
        If ``resolution`` is not positive.
    """
    if resolution <= 0.0:
        raise ValueError(f"quantize() resolution must be positive; got {resolution}.")
    return np.asarray(np.round(values / resolution) * resolution, dtype=np.float64)


def quantization_floor(resolution: float) -> float:
    """Return the noise floor implied by a reporting resolution.

    The standard deviation of uniform rounding error on a step of width
    ``delta`` is ``delta / sqrt(12)`` (METHODS.md §2.5). Exposed here so that
    tests of the Phase 6 detection floor can compare against the same
    constant the generator used, rather than a re-derived literal.

    Parameters
    ----------
    resolution : float
        Reporting step.

    Returns
    -------
    float
        Implied 1-sigma floor.

    Raises
    ------
    ValueError
        If ``resolution`` is not positive.
    """
    if resolution <= 0.0:
        raise ValueError(f"quantization_floor() resolution must be positive; got {resolution}.")
    return resolution / math.sqrt(12.0)
