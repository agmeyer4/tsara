"""Rendering the plume-free part of a synthetic signal.

The background is everything a baseline estimator is *supposed* to track and
subtract: clean-air level, the diurnal boundary-layer cycle, slow drift, and
unstructured low-frequency wander. Getting it wrong in the easy direction (a
flat constant) would make Phase 5's rolling-quantile baseline look far better
than it is, so the parametric model deliberately offers non-stationary terms
with no closed-form inverse.

Two interchangeable sources, per the
:class:`~tsara.synthetic.config.BackgroundConfig` union:

* **parametric** — analytic terms with exactly known truth.
* **bootstrap** — fluctuations resampled in contiguous blocks from a real
  :class:`~tsara.synthetic.profiling.RealDataProfile`, layered over an
  optional parametric base. Reproduces real noise colour, skew, and
  instrument quirks that no analytic model captures.

Both return a plain ``float64`` array aligned to the caller's timestamps, so
the rest of the generator never needs to know which was used.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.exceptions import TsaraError
from tsara.core.timebase import NS_PER_S
from tsara.core.timebase import epoch_ns as _epoch_ns
from tsara.core.timebase import epoch_s as _epoch_s
from tsara.synthetic.config import (
    BackgroundConfig,
    BootstrapBackground,
    ParametricBackground,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

    from tsara.synthetic.profiling import RealDataProfile

logger = logging.getLogger(__name__)

#: Seconds in one day; the reference interval for drift and random-walk
#: parameters, so those knobs mean the same thing at any sampling rate.
SECONDS_PER_DAY = 86_400.0


class TsaraSyntheticError(TsaraError):
    """Raised when a synthetic dataset cannot be generated as configured.

    Covers runtime mismatches a static schema cannot catch: a referenced
    real-data profile that was not supplied, a time span that yields no
    samples, and similar.
    """


def render_background(
    config: BackgroundConfig,
    times: pd.DatetimeIndex,
    rng: np.random.Generator,
    profiles: Mapping[str, RealDataProfile] | None = None,
) -> npt.NDArray[np.float64]:
    """Render the background signal on ``times``.

    Parameters
    ----------
    config : BackgroundConfig
        Either a :class:`~tsara.synthetic.config.ParametricBackground` or a
        :class:`~tsara.synthetic.config.BootstrapBackground`.
    times : pandas.DatetimeIndex
        Timestamps to render on. Need not be uniformly spaced.
    rng : numpy.random.Generator
        Source of randomness for the stochastic terms.
    profiles : mapping of str to RealDataProfile, optional
        Real-data profiles available by name. Required only when ``config``
        is a bootstrap background.

    Returns
    -------
    numpy.ndarray
        Background values, shape ``(len(times),)``.

    Raises
    ------
    TsaraSyntheticError
        If a bootstrap background names a profile that was not supplied.
    """
    if isinstance(config, ParametricBackground):
        return _render_parametric(config, times, rng)
    return _render_bootstrap(config, times, rng, profiles)


def _render_parametric(
    config: ParametricBackground,
    times: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Render the analytic background: offset + diurnal + drift + wander.

    Parameters
    ----------
    config : ParametricBackground
        Term parameters.
    times : pandas.DatetimeIndex
        Timestamps to render on.
    rng : numpy.random.Generator
        Used only by the random-walk term.

    Returns
    -------
    numpy.ndarray
        Background values.
    """
    import pandas as pd

    epoch_s = _epoch_s(times)
    values = np.full(epoch_s.shape, float(config.offset), dtype=np.float64)

    # --- Diurnal term -----------------------------------------------------
    # Phased off the Unix epoch rather than off the record start, because the
    # epoch is midnight-aligned: this ties the cycle to real clock time, so
    # two instruments in the same run breathe in phase with each other (as
    # they physically must) regardless of when each stream begins.
    # -cos is used rather than sin so that phase 0 puts the *minimum* at
    # midnight UTC, matching the field's documented meaning.
    if config.diurnal_amplitude > 0.0:
        period_s = float(pd.Timedelta(config.diurnal_period).total_seconds())
        phase_s = config.diurnal_phase_hours * 3600.0
        values -= config.diurnal_amplitude * np.cos(2.0 * np.pi * (epoch_s - phase_s) / period_s)

    # --- Linear drift -----------------------------------------------------
    if config.drift_per_day != 0.0:
        days_since_start = (epoch_s - epoch_s[0]) / SECONDS_PER_DAY
        values += config.drift_per_day * days_since_start

    # --- Random-walk wander ----------------------------------------------
    # Increment variance scales with elapsed time so the walk's magnitude
    # after a day is `random_walk_std` regardless of sampling rate — without
    # this, a 10 Hz stream would wander ~10x further than a 1 Hz one for the
    # same configured value.
    if config.random_walk_std > 0.0:
        dt_days = np.diff(epoch_s) / SECONDS_PER_DAY
        # Guard against a non-positive interval from duplicate timestamps.
        dt_days = np.clip(dt_days, 0.0, None)
        steps = rng.normal(0.0, config.random_walk_std * np.sqrt(dt_days))
        walk = np.concatenate(([0.0], np.cumsum(steps)))
        values += walk

    return values


def _render_bootstrap(
    config: BootstrapBackground,
    times: pd.DatetimeIndex,
    rng: np.random.Generator,
    profiles: Mapping[str, RealDataProfile] | None,
) -> npt.NDArray[np.float64]:
    """Render a background whose fluctuations come from real data blocks.

    Parameters
    ----------
    config : BootstrapBackground
        Names the profile, the optional parametric base, and a scale factor.
    times : pandas.DatetimeIndex
        Timestamps to render on.
    rng : numpy.random.Generator
        Chooses which blocks are drawn.
    profiles : mapping of str to RealDataProfile or None
        Available profiles, keyed by name.

    Returns
    -------
    numpy.ndarray
        Background values.

    Raises
    ------
    TsaraSyntheticError
        If the named profile was not supplied.
    """
    if not profiles or config.profile not in profiles:
        available = sorted(profiles) if profiles else []
        raise TsaraSyntheticError(
            f"Bootstrap background references profile '{config.profile}', which was "
            f"not supplied to the generator; available profiles: {available}. "
            "Profiles are passed at call time (they hold real-data-derived arrays "
            "and are deliberately not serializable into a config file)."
        )
    profile = profiles[config.profile]

    # Sampling-rate sanity check. Blocks are replayed sample-for-sample, so a
    # profile built at 1 Hz stretched onto a 10 Hz stream would reproduce the
    # right *shape* at ten times the real duration — silently wrong noise
    # colour. Warn rather than fail: deliberately replaying a profile at a
    # different rate is a legitimate (if unusual) experiment.
    target_period_s = _median_period_s(times)
    if profile.sample_period_s > 0.0 and target_period_s > 0.0:
        ratio = target_period_s / profile.sample_period_s
        if ratio > 2.0 or ratio < 0.5:
            logger.warning(
                "Bootstrap profile %r was built at %.4g s sampling but is being "
                "replayed onto a %.4g s grid (%.2gx); the reproduced noise "
                "autocorrelation timescale will be stretched accordingly.",
                config.profile,
                profile.sample_period_s,
                target_period_s,
                ratio,
            )

    fluctuations = _stitch_blocks(profile.residual_blocks, len(times), rng)

    if config.base is not None:
        base = _render_parametric(config.base, times, rng)
    else:
        base = np.full(len(times), float(profile.background_median), dtype=np.float64)

    return base + config.scale * fluctuations


def _stitch_blocks(
    blocks: npt.NDArray[np.float64],
    n_samples: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Concatenate randomly drawn blocks (with replacement) to length ``n_samples``.

    Block resampling — rather than resampling individual points — is the
    whole point: drawing point-by-point would destroy the residual's
    autocorrelation and hand back white noise, defeating the purpose of using
    real data. Blocks are already mean-centred by
    :func:`~tsara.synthetic.profiling.profile_series`, so the seams between
    them carry no level jump.

    Parameters
    ----------
    blocks : numpy.ndarray
        Shape ``(n_blocks, block_length)``.
    n_samples : int
        Required output length.
    rng : numpy.random.Generator
        Source of block indices.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples,)``.
    """
    n_blocks, block_length = blocks.shape
    n_needed = int(np.ceil(n_samples / block_length))
    chosen = rng.integers(0, n_blocks, size=n_needed)
    return np.asarray(blocks[chosen].reshape(-1)[:n_samples], dtype=np.float64)


def _median_period_s(times: pd.DatetimeIndex) -> float:
    """Return the median sampling interval of ``times`` in seconds.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Timestamps, assumed sorted.

    Returns
    -------
    float
        Median interval in seconds; 0.0 for a single-sample index.
    """
    if len(times) < 2:
        return 0.0
    deltas = np.diff(_epoch_ns(times))
    return float(np.median(deltas)) / NS_PER_S
