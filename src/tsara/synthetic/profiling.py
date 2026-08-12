"""Deriving synthetic-generator parameters from *real* measurements.

A synthetic dataset is only as useful as its resemblance to reality. This
module closes that gap: it measures the statistical shape of a real
timeseries — noise magnitude, noise autocorrelation, background level and
spread — and packages it as a :class:`RealDataProfile` that the generator
can either read parameters from or resample fluctuations out of
(:class:`~tsara.synthetic.config.BootstrapBackground`).

Naming
------
This is deliberately **not** called "calibration". In this domain that word
means referencing an instrument against gas standards — a completely
different operation that real campaign directory trees already use
(``04_calibrated/``, ``calibration_coefs.json``). "Profiling" is the
established, unambiguous term for summarizing a dataset's statistical shape.

Scope boundaries
----------------
* **No real data ships with TSARA, ever.** These functions take an
  already-loaded ``pandas.Series``; they never read files, never bundle
  data, and never cache profiles into the repository. Tests that exercise
  them against real measurements are skipped unless a live data mount is
  configured.
* **This is not the Phase 5 baseline engine.** The rolling low quantile
  computed here is a deliberately plain, unswept, single-parameter estimate
  used only to strip slow structure so the residual can be characterized.
  It shares an idea with the production baseline; it is not that code, has
  no uncertainty propagation, and must not grow into it.
* **This module does not depend on ingestion.** Phase 3 does not exist yet,
  so the input is a plain pandas object a user can produce with one
  ``read_parquet`` call. When real ingestion lands, an ``xarray`` variable
  satisfies the same minimal interface with no rework.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from tsara.core.exceptions import TsaraError
from tsara.core.timebase import NS_PER_S
from tsara.core.timebase import epoch_ns as _epoch_ns

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

logger = logging.getLogger(__name__)

#: Gaussian consistency constant: for normally distributed x,
#: 1.4826 * MAD(x) is an unbiased estimator of the standard deviation.
MAD_TO_SIGMA = 1.4826


class TsaraProfilingError(TsaraError):
    """Raised when a real timeseries cannot be profiled.

    Distinct from a config error: the configuration may be perfectly valid
    while the *data* is too short, too gappy, or entirely NaN to yield a
    usable profile.
    """


@dataclass(frozen=True, eq=False)
class RealDataProfile:
    """Statistical fingerprint of one real timeseries.

    Deliberately a plain frozen dataclass rather than a Pydantic model: this
    is a *computed artifact* holding numpy arrays, not user-authored
    configuration, and it must never be serialized into a config file (see
    the module docstring's scope note). ``eq=False`` because dataclass
    equality on an array field raises rather than returning a bool.

    Attributes
    ----------
    name : str
        Key by which a :class:`~tsara.synthetic.config.BootstrapBackground`
        refers to this profile.
    residual_blocks : numpy.ndarray
        Shape ``(n_blocks, block_length)``. Contiguous, gap-free, individually
        **mean-centred** chunks of the baseline-subtracted signal. Centring
        is what makes blocks stitchable without step discontinuities at the
        seams; the cost is that between-block (low-frequency) structure is
        discarded, which the parametric ``base`` term supplies instead.
    residual_sigma : float
        Robust standard deviation of the full residual (MAD-based). Includes
        any real plume energy that leaked through the profiling baseline.
    noise_sigma : float
        Robust *point-to-point* noise scale, via the same first-difference
        estimator TSARA uses in production (METHODS.md §2.5). Structurally
        immune to plume leakage, so ``residual_sigma`` >> ``noise_sigma`` is
        the signature of a plume-dense record rather than a noisy one.
    lag1_autocorr : float
        Lag-1 autocorrelation **of the residual**, in ``[-1, 1]``.

        Read this as a property of the bootstrap substrate, not of the
        instrument noise. In a plume-dense record the residual contains
        leaked plume structure, which is smooth and therefore strongly
        autocorrelated — a record whose noise is AR(1) with rho = 0.8 can
        easily profile at rho1 > 0.99 once a few broad plumes survive the
        baseline. That is the correct value *for its purpose* (it describes
        what block-bootstrap resampling will actually reproduce), but it is
        not an estimate of the noise decorrelation timescale, and it must
        not be fed to an N_eff calculation as though it were.
    decorrelation_timescale_s : float or None
        Implied AR(1) timescale ``tau`` from ``lag1_autocorr`` (METHODS.md
        §3.4), or None when the residual is uncorrelated or anticorrelated
        at lag 1 (no AR(1) fit exists). Inherits the caveat above: on a clean
        record it recovers the true noise timescale; on a plume-dense one it
        measures structure plus noise.
    background_median : float
        Median of the fitted background — the "typical clean-air level".
    background_iqr : float
        Interquartile range of the fitted background, i.e. how much the
        background itself moves.
    sample_period_s : float
        Median sampling interval of the source record, in seconds.
    n_source_points : int
        Number of finite samples the profile was built from.
    """

    name: str
    residual_blocks: npt.NDArray[np.float64]
    residual_sigma: float
    noise_sigma: float
    lag1_autocorr: float
    decorrelation_timescale_s: float | None
    background_median: float
    background_iqr: float
    sample_period_s: float
    n_source_points: int

    @property
    def n_blocks(self) -> int:
        """Number of resamplable residual blocks."""
        return int(self.residual_blocks.shape[0])

    @property
    def block_length(self) -> int:
        """Number of samples in each residual block."""
        return int(self.residual_blocks.shape[1])

    def summary(self) -> str:
        """Return a one-line human-readable summary, for logs and notebooks.

        Returns
        -------
        str
            Compact description of the profile's key statistics.
        """
        tau = (
            "none"
            if self.decorrelation_timescale_s is None
            else f"{self.decorrelation_timescale_s:.1f}s"
        )
        return (
            f"RealDataProfile({self.name!r}: {self.n_blocks} blocks x "
            f"{self.block_length} samples @ {self.sample_period_s:g}s, "
            f"noise_sigma={self.noise_sigma:.4g}, residual_sigma={self.residual_sigma:.4g}, "
            f"rho1={self.lag1_autocorr:.3f}, tau={tau})"
        )


def diff_mad_sigma(values: npt.NDArray[np.float64]) -> float:
    r"""Robust first-difference noise estimate (the ``diff_mad`` estimator).

    Implements METHODS.md §2.5:

    .. math::

        \hat\sigma = \frac{1.4826 \cdot
        \mathrm{median}_i(|x_{i+1} - x_i|)}{\sqrt{2}}

    The :math:`\sqrt{2}` corrects for differencing inflating the variance
    (:math:`\mathrm{Var}(x_{i+1}-x_i) = 2\sigma^2` for white noise). Working
    on differences rather than the signal is what makes the estimate immune
    to plumes: a broad enhancement has point-to-point differences of noise
    size no matter how tall it is.

    This is a small standalone helper, *not* the production Phase 6 estimator
    — that one is windowed, registered by name, and carries the quantization
    floor. This one exists so profiling can characterize noise using the same
    mathematics the pipeline will later apply.

    Parameters
    ----------
    values : numpy.ndarray
        Finite signal values in acquisition order.

    Returns
    -------
    float
        Estimated 1-sigma random noise, or ``nan`` if fewer than two values.
    """
    if values.size < 2:
        return float("nan")
    diffs = np.abs(np.diff(values))
    return float(MAD_TO_SIGMA * np.median(diffs) / math.sqrt(2.0))


def profile_series(
    series: pd.Series,
    *,
    name: str,
    baseline_window: str = "30min",
    baseline_quantile: float = 0.10,
    block_length: int = 512,
    max_blocks: int | None = 2000,
) -> RealDataProfile:
    """Measure the statistical shape of a real timeseries.

    The procedure is deliberately simple and inspectable:

    1. Fit a rolling low-quantile background (plain pandas, no sweep).
    2. Subtract it to obtain the residual fluctuation.
    3. Characterize the residual: robust spread, plume-immune noise scale,
       lag-1 autocorrelation and its implied AR(1) timescale.
    4. Cut the residual into contiguous, gap-free, mean-centred blocks for
       later block-bootstrap resampling.

    Step 4 splits on gaps first: blocks never straddle a data dropout, so a
    resampled substrate can never contain a fabricated jump that the real
    instrument did not produce.

    Parameters
    ----------
    series : pandas.Series
        Real measurements indexed by a monotonically increasing
        ``DatetimeIndex``. NaNs are permitted and are excluded from blocks.
    name : str
        Profile key, referenced by
        :class:`~tsara.synthetic.config.BootstrapBackground.profile`.
    baseline_window : str, default "30min"
        Centered rolling window for the background fit.
    baseline_quantile : float, default 0.10
        Quantile taken within each window, in ``(0, 0.5]``. Low quantiles
        track the background *underneath* plumes, since enhancements are
        one-sided.
    block_length : int, default 512
        Samples per bootstrap block. Should comfortably exceed the noise
        decorrelation length (so within-block autocorrelation is captured)
        while staying well under the shortest baseline window the generated
        data will be analyzed with.
    max_blocks : int or None, default 2000
        Cap on retained blocks, to bound memory when profiling a long
        record. None keeps every block.

    Returns
    -------
    RealDataProfile
        The fitted profile.

    Raises
    ------
    TsaraProfilingError
        If the index is not a sorted ``DatetimeIndex``, the quantile is out
        of range, or the record yields no usable block.
    """
    import pandas as pd

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TsaraProfilingError(
            f"profile_series({name!r}) requires a DatetimeIndex; got {type(series.index).__name__}."
        )
    if not series.index.is_monotonic_increasing:
        raise TsaraProfilingError(
            f"profile_series({name!r}) requires a monotonically increasing index; "
            "sort the series first."
        )
    if not 0.0 < baseline_quantile <= 0.5:
        raise TsaraProfilingError(
            f"baseline_quantile must be in (0, 0.5]; got {baseline_quantile}. "
            "A background estimate above the median is not a background."
        )
    if block_length < 2:
        raise TsaraProfilingError(f"block_length must be at least 2; got {block_length}.")

    values = pd.to_numeric(series, errors="coerce").astype(float)
    finite = values.dropna()
    if finite.size < block_length:
        raise TsaraProfilingError(
            f"profile_series({name!r}): only {finite.size} finite samples, which is "
            f"fewer than block_length={block_length}; nothing to resample."
        )

    # --- 1/2. Background and residual ------------------------------------
    # min_periods=2 keeps the quantile from degenerating to the identity at
    # the record's edges, where a centered window is half-empty.
    background = values.rolling(baseline_window, center=True, min_periods=2).quantile(
        baseline_quantile
    )
    residual = values - background

    # --- 3. Residual statistics ------------------------------------------
    residual_finite = residual.dropna()
    residual_values = residual_finite.to_numpy(dtype=float)
    if residual_values.size < 2:
        raise TsaraProfilingError(
            f"profile_series({name!r}): fewer than two finite residual samples."
        )

    residual_sigma = float(
        MAD_TO_SIGMA * np.median(np.abs(residual_values - np.median(residual_values)))
    )
    noise_sigma = diff_mad_sigma(residual_values)

    # Lag-1 autocorrelation of the residual. Two degenerate cases map to
    # "uncorrelated", which is the only defensible reading of each: fewer
    # than three samples gives a single lagged pair (no correlation is
    # estimable), and a constant residual has zero variance so np.corrcoef
    # divides by zero and returns nan. errstate silences the expected
    # warning rather than letting it reach the user as noise.
    lag1 = 0.0
    if residual_values.size >= 3:
        with np.errstate(invalid="ignore", divide="ignore"):
            lag1 = float(np.corrcoef(residual_values[:-1], residual_values[1:])[0, 1])
        if not math.isfinite(lag1):
            lag1 = 0.0

    sample_period_s = float(np.median(np.diff(_epoch_ns(series.index)))) / NS_PER_S

    # AR(1): rho1 = exp(-dt/tau)  =>  tau = -dt / ln(rho1). Only defined for
    # a positive, sub-unity correlation; an anticorrelated residual (rho1<=0,
    # typical when the record is dominated by quantization or by the
    # differencing artifacts of an over-tight baseline) has no AR(1) fit.
    tau_s: float | None = None
    if 0.0 < lag1 < 1.0 and sample_period_s > 0.0:
        tau_s = float(-sample_period_s / math.log(lag1))

    # --- 4. Contiguous, gap-free, mean-centred blocks --------------------
    blocks = _extract_blocks(
        residual_finite,
        sample_period_s=sample_period_s,
        block_length=block_length,
        max_blocks=max_blocks,
        name=name,
    )

    background_finite = background.dropna()
    profile = RealDataProfile(
        name=name,
        residual_blocks=blocks,
        residual_sigma=residual_sigma,
        noise_sigma=noise_sigma,
        lag1_autocorr=lag1,
        decorrelation_timescale_s=tau_s,
        background_median=float(background_finite.median()),
        background_iqr=float(background_finite.quantile(0.75) - background_finite.quantile(0.25)),
        sample_period_s=sample_period_s,
        n_source_points=int(finite.size),
    )
    logger.info("Profiled real data: %s", profile.summary())
    if residual_sigma > 3.0 * noise_sigma:
        # Not an error — this is the expected signature of the plume-dense
        # records this project actually has, and the reason the bootstrap
        # substrate doubles as an adversarial noise-estimation test case.
        logger.info(
            "Profile %r: residual_sigma (%.4g) far exceeds noise_sigma (%.4g), "
            "indicating substantial real plume energy leaked through the "
            "profiling baseline. Bootstrapped backgrounds will inherit it.",
            name,
            residual_sigma,
            noise_sigma,
        )
    return profile


def _extract_blocks(
    residual: pd.Series,
    *,
    sample_period_s: float,
    block_length: int,
    max_blocks: int | None,
    name: str,
) -> npt.NDArray[np.float64]:
    """Cut a residual series into contiguous, gap-free, mean-centred blocks.

    Segments are split wherever the sampling interval exceeds 1.5x the
    nominal period, so no block spans a dropout. Each retained block has its
    own mean removed (see :class:`RealDataProfile.residual_blocks` for why).

    Parameters
    ----------
    residual : pandas.Series
        Finite residual values with a DatetimeIndex.
    sample_period_s : float
        Nominal sampling period, seconds.
    block_length : int
        Samples per block.
    max_blocks : int or None
        Optional cap on the number of blocks returned.
    name : str
        Profile name, for error messages.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_blocks, block_length)``, dtype float64.

    Raises
    ------
    TsaraProfilingError
        If no gap-free run is long enough to yield a single block.
    """
    # The caller always passes a DatetimeIndex-backed series (profile_series
    # validates that up front); the cast tells mypy what the runtime already
    # guarantees, since Series.index is statically just Index.
    times = _epoch_ns(cast("pd.DatetimeIndex", residual.index))
    values = residual.to_numpy(dtype=float)

    # Gap detection. A zero/negative nominal period (a degenerate record with
    # duplicate timestamps) disables splitting rather than dividing by zero.
    if sample_period_s > 0.0:
        gap_threshold_ns = 1.5 * sample_period_s * 1e9
        breaks = np.flatnonzero(np.diff(times) > gap_threshold_ns) + 1
    else:  # pragma: no cover - degenerate records are rejected upstream
        breaks = np.array([], dtype=int)
    segments = np.split(values, breaks)

    collected: list[npt.NDArray[np.float64]] = []
    for segment in segments:
        n_full = segment.size // block_length
        if n_full == 0:
            continue
        trimmed = segment[: n_full * block_length].reshape(n_full, block_length)
        # Mean-centre each block independently.
        collected.append(trimmed - trimmed.mean(axis=1, keepdims=True))
        if max_blocks is not None and sum(b.shape[0] for b in collected) >= max_blocks:
            break

    if not collected:
        raise TsaraProfilingError(
            f"profile_series({name!r}): no gap-free run reached block_length="
            f"{block_length}; lower block_length or supply a less fragmented record."
        )

    blocks = np.concatenate(collected, axis=0)
    if max_blocks is not None and blocks.shape[0] > max_blocks:
        blocks = blocks[:max_blocks]
    return np.ascontiguousarray(blocks, dtype=np.float64)
