"""Realizing the plume-free part of one field of the synthetic atmosphere.

The background is everything a baseline estimator is *supposed* to track and
subtract: clean-air level, the diurnal boundary-layer cycle, slow drift, and
unstructured low-frequency wander. Getting it wrong in the easy direction (a
flat constant) would make Phase 5's rolling-quantile baseline look far better
than it is, so the parametric model deliberately offers non-stationary terms
with no closed-form inverse.

Two interchangeable sources, per the
:class:`~tsara.synthetic.config.BackgroundConfig` union:

* **parametric** — analytic terms with exactly known truth, plus an optional
  random walk.
* **bootstrap** — fluctuations resampled in contiguous blocks from a real
  :class:`~tsara.synthetic.profiling.RealDataProfile`, layered over an
  optional parametric base. Reproduces real noise colour, skew, and
  instrument quirks that no analytic model captures.

Realized once, queried anywhere
-------------------------------
A background is a property of the air, not of whoever measures it, so it is
*realized* once per field per run -- every random draw made -- and then
evaluated at whatever times are asked for (:meth:`RealizedBackground.at`).
The analytic terms are evaluated exactly at those times. The stochastic terms
cannot be: a random walk has no formula to evaluate. So each is drawn once on
evenly spaced **nodes** across the campaign and is linear between them
(:class:`NodeSeries`), which makes the realized background a deterministic
function of time. Two instruments measuring one field then see the same
wander at the same instant, whatever their clocks.

That replaces rendering the background separately on every instrument's own
timestamps, and it removes two defects that design had. Two instruments
measuring one field drew independent walks, so they disagreed by about as
much as the background varied; and drift was measured from the first
timestamp each rendering happened to start at, so a mean-support instrument
(rendered on sub-samples starting half a sub-step early) and a point
instrument disagreed by a small constant. Drift is now measured from the
campaign start, for every instrument at once.

Outside the realized span a stochastic term holds its edge value. Instrument
cells can reach a fraction of a sampling interval beyond the campaign's ends,
and holding the value there keeps the atmosphere independent of which
instruments sample it; the analytic terms are unaffected.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.exceptions import TsaraError
from tsara.core.timebase import NS_PER_S
from tsara.synthetic.config import (
    BackgroundConfig,
    BootstrapBackground,
    ParametricBackground,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

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


@dataclass(frozen=True, eq=False)
class NodeSeries:
    """A stochastic term, drawn once on evenly spaced nodes and linear between them.

    Attributes
    ----------
    node_s : numpy.ndarray
        Node times, epoch seconds, strictly increasing. Computed from integer
        nanoseconds, so a query at exactly a node's instant lands on that node's
        value rather than a rounding away from it.
    values : numpy.ndarray
        The term's value at each node.
    """

    node_s: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]

    def at(self, epoch_s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the term at ``epoch_s``: linear between nodes, held beyond them.

        Parameters
        ----------
        epoch_s : numpy.ndarray
            Query times, epoch seconds, in any order.

        Returns
        -------
        numpy.ndarray
            The term's value at each query time.
        """
        return np.asarray(np.interp(epoch_s, self.node_s, self.values), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class RealizedBackground:
    """One field's background with every random draw made: a function of time.

    Attributes
    ----------
    offset : float
        Constant level: the parametric ``offset``, or a bootstrap profile's
        median level when no ``base`` was given.
    diurnal_amplitude, diurnal_period_s, diurnal_phase_s : float
        The cyclic term; amplitude 0 switches it off.
    drift_per_day : float
        Linear trend, in field units per day.
    origin_s : float
        Epoch seconds the drift is measured from: the campaign start.
    stochastic : tuple of NodeSeries
        Terms with no analytic form, added in order: a bootstrap's
        fluctuations, then a random walk.
    """

    offset: float
    diurnal_amplitude: float
    diurnal_period_s: float
    diurnal_phase_s: float
    drift_per_day: float
    origin_s: float
    stochastic: tuple[NodeSeries, ...]

    def at(self, epoch_s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the background at ``epoch_s``.

        The analytic terms are written exactly as they were when backgrounds
        were rendered per instrument, operation for operation. That is not
        style: it is what keeps every configuration without a stochastic term
        or drift producing byte-identical streams through the redesign.

        Parameters
        ----------
        epoch_s : numpy.ndarray
            Query times, epoch seconds, in any order.

        Returns
        -------
        numpy.ndarray
            Background values, same shape as ``epoch_s``.
        """
        values = np.full(epoch_s.shape, self.offset, dtype=np.float64)

        # --- Diurnal term -------------------------------------------------
        # Phased off the Unix epoch rather than off the record start, because
        # the epoch is midnight-aligned: this ties the cycle to real clock
        # time. -cos is used rather than sin so that phase 0 puts the
        # *minimum* at midnight UTC, matching the field's documented meaning.
        if self.diurnal_amplitude > 0.0:
            values -= self.diurnal_amplitude * np.cos(
                2.0 * np.pi * (epoch_s - self.diurnal_phase_s) / self.diurnal_period_s
            )

        # --- Linear drift, from the campaign start --------------------------
        if self.drift_per_day != 0.0:
            days_since_start = (epoch_s - self.origin_s) / SECONDS_PER_DAY
            values += self.drift_per_day * days_since_start

        # --- Stochastic terms, realized once on nodes ------------------------
        for series in self.stochastic:
            values += series.at(epoch_s)

        return values


def realize_background(
    config: BackgroundConfig,
    *,
    start_ns: int,
    end_ns: int,
    truth_resolution_ns: int,
    rng: np.random.Generator,
    profiles: Mapping[str, RealDataProfile] | None = None,
    field: str = "",
) -> RealizedBackground:
    """Make every random draw one field's background needs, once.

    Parameters
    ----------
    config : BackgroundConfig
        Either a :class:`~tsara.synthetic.config.ParametricBackground` or a
        :class:`~tsara.synthetic.config.BootstrapBackground`.
    start_ns, end_ns : int
        The campaign, epoch nanoseconds. Drift is measured from ``start_ns``,
        and stochastic terms are realized across ``[start_ns, end_ns]``.
    truth_resolution_ns : int
        Node spacing for a random walk.
    rng : numpy.random.Generator
        Source of randomness for the stochastic terms. A background with none
        draws nothing, which is what lets a configuration without one keep
        every later draw -- and so every noise realization -- where it was.
    profiles : mapping of str to RealDataProfile, optional
        Real-data profiles available by name. Required only for a bootstrap
        background.
    field : str, optional
        The field being realized, named in error messages.

    Returns
    -------
    RealizedBackground
        The background as a function of time.

    Raises
    ------
    TsaraSyntheticError
        If a bootstrap background names a profile that was not supplied, or a
        profile with no sampling period.
    """
    if isinstance(config, ParametricBackground):
        return _realize_parametric(
            config,
            start_ns=start_ns,
            end_ns=end_ns,
            truth_resolution_ns=truth_resolution_ns,
            rng=rng,
        )
    return _realize_bootstrap(
        config,
        start_ns=start_ns,
        end_ns=end_ns,
        truth_resolution_ns=truth_resolution_ns,
        rng=rng,
        profiles=profiles,
        field=field,
    )


def _realize_parametric(
    config: ParametricBackground,
    *,
    start_ns: int,
    end_ns: int,
    truth_resolution_ns: int,
    rng: np.random.Generator,
    prior: tuple[NodeSeries, ...] = (),
) -> RealizedBackground:
    """Realize the analytic background: offset + diurnal + drift + wander.

    ``prior`` carries stochastic terms realized before this one (a
    bootstrap's fluctuations) so they are added first.
    """
    import pandas as pd

    stochastic = prior
    if config.random_walk_std > 0.0:
        stochastic = (
            *prior,
            _realize_walk(config.random_walk_std, start_ns, end_ns, truth_resolution_ns, rng),
        )
    return RealizedBackground(
        offset=float(config.offset),
        diurnal_amplitude=float(config.diurnal_amplitude),
        diurnal_period_s=float(pd.Timedelta(config.diurnal_period).total_seconds()),
        diurnal_phase_s=config.diurnal_phase_hours * 3600.0,
        drift_per_day=float(config.drift_per_day),
        origin_s=start_ns / NS_PER_S,
        stochastic=stochastic,
    )


def _realize_walk(
    std_per_day: float,
    start_ns: int,
    end_ns: int,
    spacing_ns: int,
    rng: np.random.Generator,
) -> NodeSeries:
    """Draw a random walk on nodes ``spacing_ns`` apart, starting at zero.

    Each increment's variance is proportional to the node spacing, so the
    walk accumulates ``std_per_day`` of spread over a day whatever the
    spacing; a finer truth clock resolves the same wander more finely rather
    than wandering further. The walk is zero at the campaign start, so the
    configured offset is the level the record opens at.
    """
    n_steps = _n_steps(start_ns, end_ns, spacing_ns)
    node_ns = start_ns + spacing_ns * np.arange(n_steps + 1, dtype=np.int64)
    dt_days = spacing_ns / NS_PER_S / SECONDS_PER_DAY
    steps = rng.normal(0.0, std_per_day * math.sqrt(dt_days), size=n_steps)
    values = np.concatenate(([0.0], np.cumsum(steps)))
    return NodeSeries(node_s=node_ns / NS_PER_S, values=values)


def _realize_bootstrap(
    config: BootstrapBackground,
    *,
    start_ns: int,
    end_ns: int,
    truth_resolution_ns: int,
    rng: np.random.Generator,
    profiles: Mapping[str, RealDataProfile] | None,
    field: str,
) -> RealizedBackground:
    """Realize a background whose fluctuations come from real data blocks.

    The fluctuations are replayed at the profile's own sampling period, so
    the real record's correlation timescale survives whatever rate the field
    is later measured at. Blocks are drawn before a base's random walk, the
    same order the two were drawn in when backgrounds were rendered per
    instrument.
    """
    if not profiles or config.profile not in profiles:
        available = sorted(profiles) if profiles else []
        raise TsaraSyntheticError(
            f"Field '{field}' has a bootstrap background referencing profile "
            f"'{config.profile}', which was not supplied to the generator; available "
            f"profiles: {available}. Profiles are passed at call time (they hold "
            "real-data-derived arrays and are deliberately not serializable into a "
            "config file)."
        )
    profile = profiles[config.profile]

    spacing_ns = round(profile.sample_period_s * NS_PER_S)
    if spacing_ns <= 0:
        raise TsaraSyntheticError(
            f"Profile '{config.profile}' records a sampling period of "
            f"{profile.sample_period_s} s, so its fluctuations have no cadence to be "
            "replayed at."
        )
    n_steps = _n_steps(start_ns, end_ns, spacing_ns)
    node_ns = start_ns + spacing_ns * np.arange(n_steps + 1, dtype=np.int64)
    fluctuations = NodeSeries(
        node_s=node_ns / NS_PER_S,
        values=config.scale * _stitch_blocks(profile.residual_blocks, n_steps + 1, rng),
    )

    # With no base, the fluctuations sit on the profile's own median level, flat.
    base = config.base or ParametricBackground(
        kind="parametric", offset=float(profile.background_median)
    )
    return _realize_parametric(
        base,
        start_ns=start_ns,
        end_ns=end_ns,
        truth_resolution_ns=truth_resolution_ns,
        rng=rng,
        prior=(fluctuations,),
    )


def _n_steps(start_ns: int, end_ns: int, spacing_ns: int) -> int:
    """Return how many node spacings it takes to reach ``end_ns`` from ``start_ns``.

    At least one, so even a campaign shorter than one spacing has two nodes
    and a defined value at both ends.
    """
    return max(1, -(-(end_ns - start_ns) // spacing_ns))


def _stitch_blocks(
    blocks: npt.NDArray[np.float64],
    n_samples: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Concatenate randomly drawn blocks (with replacement) to length ``n_samples``.

    Block resampling — rather than resampling individual points — is the
    whole point: drawing point-by-point would destroy the residual's
    autocorrelation and hand back white noise, defeating the purpose of using
    real data.

    What the mean-centring in
    :func:`~tsara.synthetic.profiling.profile_series` buys, and what it does
    not: adjacent blocks share a mean, so no *level* discontinuity appears at
    a seam. The individual samples either side of a seam are still
    independent draws, so a seam does carry a sample-to-sample step of order
    the residual sigma — empirically about 4x the typical interior step.

    That is tolerable, and deliberately so. Seams occupy only
    ``1/block_length`` of all adjacent pairs (0.8 % at the default 128), and
    every noise estimator downstream is median-based, so the seam steps sit
    far out in the tail where they cannot move the answer: on a strongly
    autocorrelated substrate ``diff_mad`` differs by well under 1 % from its
    seam-free value. The alternative — overlap-blending seams — would smooth
    real high-frequency structure, which is the one thing the bootstrap
    exists to preserve.

    Worth knowing when reading a generated record: an isolated sharp step
    every ``block_length`` profile samples is an artifact of this stitching,
    not injected signal.

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
