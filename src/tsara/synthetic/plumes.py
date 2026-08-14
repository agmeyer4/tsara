"""Plume shapes, event scheduling, and the ground-truth record.

This module manufactures the *answer key*. Everything downstream in TSARA —
baselines, detection, regression, uncertainty quantification — is ultimately
scored against the :class:`GroundTruth` produced here, and because this
project has no controlled-release measurements to validate against
(CLAUDE.md §5), injected synthetic truth is the **only** arbiter of
detection and ratio correctness in v1. That makes the fidelity of this module
load-bearing.

Three concerns live here:

1. :class:`PlumeKernel` — a normalized temporal shape, precomputed once per
   configured shape, that knows its own peak location and finite support.
2. :func:`schedule_events` — draws *physical* events from a Poisson process
   and realizes their per-species amplitudes, ratios, lags, and nesting.
3. :class:`GroundTruth` — the catalog of what was injected, deliberately
   schema-compatible with a subset of the future Phase 6 ``PlumeCatalog`` so
   that scoring detection is a direct column-wise diff rather than a
   translation layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from tsara.core.timebase import to_utc_naive_stamp as _to_utc_naive_stamp
from tsara.synthetic.config import (
    AmplitudeSpec,
    GaussianShape,
    LognormalAmplitude,
    PlumeShape,
    RatioSpec,
    SourceSpec,
    SyntheticConfig,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

logger = logging.getLogger(__name__)

#: Gaussian support half-width, in sigmas. At 4 sigma the shape is at 3.4e-4
#: of its peak — far below any realistic detection threshold, so truncating
#: there costs nothing and keeps rendering O(events x window) instead of
#: O(events x record).
GAUSSIAN_SUPPORT_SIGMAS = 4.0

#: Exponential-tail support, in units of tau. e^-6 = 0.25 % of peak.
EMG_SUPPORT_TAUS = 6.0

#: Below this argument, ``erfcx`` overflows float64 (erfcx(z) ~ 2exp(z^2),
#: and exp(709) is the ceiling). The asymptotic branch takes over here; see
#: :func:`_emg_log_shape`.
_ERFCX_ASYMPTOTIC_CUTOFF = -26.0


# ---------------------------------------------------------------------------
# Plume shape kernels
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PlumeKernel:
    """A normalized plume shape with peak amplitude exactly 1.

    Precomputing the kernel once per configured shape (rather than
    re-deriving it per event) matters because the EMG's peak location and
    normalizing constant have no convenient closed form and are found on a
    dense reference grid.

    Attributes
    ----------
    sigma_s : float
        Gaussian width in seconds.
    tau_s : float
        Exponential tail timescale in seconds; 0 for a pure Gaussian.
    peak_offset_s : float
        Offset from the shape's center parameter (the Gaussian mu) to its
        actual maximum. Always 0 for a Gaussian; positive for an EMG, whose
        tail drags the mode later than mu.
    support_before_s, support_after_s : float
        Finite support around the center, outside which the kernel is
        exactly 0.
    log_peak : float
        Log of the unnormalized shape's maximum, used to normalize.
    """

    sigma_s: float
    tau_s: float
    peak_offset_s: float
    support_before_s: float
    support_after_s: float
    log_peak: float

    @property
    def duration_s(self) -> float:
        """Total supported width in seconds."""
        return self.support_before_s + self.support_after_s

    def evaluate(self, dt_s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the kernel at offsets ``dt_s`` from the shape center.

        Parameters
        ----------
        dt_s : numpy.ndarray
            Seconds relative to the shape's center parameter (not its peak).

        Returns
        -------
        numpy.ndarray
            Values in ``[0, 1]``, zero outside the kernel's support.
        """
        inside = (dt_s >= -self.support_before_s) & (dt_s <= self.support_after_s)
        out = np.zeros_like(dt_s, dtype=np.float64)
        if not np.any(inside):
            return out
        log_shape = self._log_shape(dt_s[inside])
        out[inside] = np.exp(log_shape - self.log_peak)
        return out

    def _log_shape(self, dt_s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Unnormalized log-shape; Gaussian and EMG dispatch on ``tau_s``."""
        u = dt_s / self.sigma_s
        if self.tau_s <= 0.0:
            return np.asarray(-0.5 * u**2, dtype=np.float64)
        return _emg_log_shape(u, self.sigma_s / self.tau_s)


def _emg_log_shape(u: npt.NDArray[np.float64], sigma_over_tau: float) -> npt.NDArray[np.float64]:
    r"""Log of the (unnormalized) exponentially-modified Gaussian shape.

    The EMG density is the convolution of a Gaussian (turbulent dispersion)
    with a decaying exponential (residence time). Its textbook form,

    .. math::

        f(t) \propto \exp\!\Big(\tfrac{\sigma^2}{2\tau^2}
        - \tfrac{t-\mu}{\tau}\Big)\,
        \mathrm{erfc}\!\Big(\tfrac{\sigma/\tau - u}{\sqrt 2}\Big),

    overflows catastrophically in float64: the exponential explodes while
    ``erfc`` underflows to 0, so their product evaluates to ``inf * 0 = nan``
    exactly in the tail where plume shapes matter most. Rewriting via the
    *scaled* complementary error function
    :math:`\mathrm{erfcx}(z) = e^{z^2}\mathrm{erfc}(z)` cancels the two
    divergences analytically:

    .. math::

        f(t) \propto e^{-u^2/2}\;\mathrm{erfcx}\!\Big(
        \tfrac{\sigma/\tau - u}{\sqrt 2}\Big),
        \qquad u = \tfrac{t-\mu}{\sigma}

    which is stable until ``erfcx`` itself overflows around ``z = -26``.
    Beyond that, :math:`\mathrm{erfcx}(z) \to 2e^{z^2}`, giving the closed
    form :math:`\log f \to \log 2 + \sigma^2/(2\tau^2) - u\sigma/\tau`
    — a pure exponential decay of timescale tau, which is exactly the
    physical tail behaviour the EMG is chosen for.

    Parameters
    ----------
    u : numpy.ndarray
        Standardized offset ``(t - mu) / sigma``.
    sigma_over_tau : float
        Ratio ``sigma / tau``, the shape's only dimensionless parameter.

    Returns
    -------
    numpy.ndarray
        Unnormalized log-shape values.
    """
    from scipy.special import erfcx

    z = (sigma_over_tau - u) / math.sqrt(2.0)
    result = np.empty_like(u, dtype=np.float64)

    asymptotic = z < _ERFCX_ASYMPTOTIC_CUTOFF
    stable = ~asymptotic

    if np.any(stable):
        result[stable] = -0.5 * u[stable] ** 2 + np.log(erfcx(z[stable]))
    if np.any(asymptotic):
        # log f = log(2) + z^2 - u^2/2, and z^2 - u^2/2 simplifies exactly to
        # (sigma/tau)^2/2 - u*(sigma/tau).
        result[asymptotic] = (
            math.log(2.0) + 0.5 * sigma_over_tau**2 - u[asymptotic] * sigma_over_tau
        )
    return result


def build_kernel(shape: PlumeShape) -> PlumeKernel:
    """Precompute the normalized kernel for a configured plume shape.

    For a Gaussian everything is analytic. For an EMG the peak location and
    normalizing constant are found on a dense reference grid (4001 points
    across the support), which is accurate to well under a sampling interval
    for any realistic configuration and avoids a root-find.

    Parameters
    ----------
    shape : PlumeShape
        A :class:`~tsara.synthetic.config.GaussianShape` or
        :class:`~tsara.synthetic.config.EMGShape`.

    Returns
    -------
    PlumeKernel
        Kernel normalized to unit peak.
    """
    import pandas as pd

    sigma_s = float(pd.Timedelta(shape.sigma).total_seconds())

    if isinstance(shape, GaussianShape):
        support = GAUSSIAN_SUPPORT_SIGMAS * sigma_s
        return PlumeKernel(
            sigma_s=sigma_s,
            tau_s=0.0,
            peak_offset_s=0.0,
            support_before_s=support,
            support_after_s=support,
            log_peak=0.0,
        )

    tau_s = float(pd.Timedelta(shape.tau).total_seconds())
    support_before = GAUSSIAN_SUPPORT_SIGMAS * sigma_s
    support_after = GAUSSIAN_SUPPORT_SIGMAS * sigma_s + EMG_SUPPORT_TAUS * tau_s

    grid = np.linspace(-support_before, support_after, 4001, dtype=np.float64)
    log_values = _emg_log_shape(grid / sigma_s, sigma_s / tau_s)
    peak_index = int(np.argmax(log_values))

    # Refine the peak between the coarse grid's neighbouring points. Without
    # this, `log_peak` is the maximum of a *sampled* shape and therefore
    # slightly below the true continuous maximum, so the normalized kernel can
    # exceed 1 by ~1e-6 at a well-placed sample — which would let a recorded
    # `sampled_peak_amplitude` exceed the `true_amplitude` it is supposed to
    # be bounded by. Cheap to fix, and it keeps that invariant exact enough to
    # assert on.
    lo = grid[max(peak_index - 1, 0)]
    hi = grid[min(peak_index + 1, grid.size - 1)]
    fine = np.linspace(lo, hi, 401, dtype=np.float64)
    fine_values = _emg_log_shape(fine / sigma_s, sigma_s / tau_s)
    fine_index = int(np.argmax(fine_values))

    return PlumeKernel(
        sigma_s=sigma_s,
        tau_s=tau_s,
        peak_offset_s=float(fine[fine_index]),
        support_before_s=support_before,
        support_after_s=support_after,
        log_peak=float(fine_values[fine_index]),
    )


# ---------------------------------------------------------------------------
# Realized events
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RealizedEvent:
    """One physical plume event, with every random draw already made.

    Created by :func:`schedule_events` before any rendering happens, because
    an event is shared across species and therefore across *instruments*:
    the same leak must appear on the 1 Hz analyzer and the 10 Hz analyzer at
    consistent amplitudes. Drawing per-instrument would silently destroy the
    cross-species covariance the whole package measures.

    Attributes
    ----------
    event_id : str
        Unique identifier, e.g. ``"well_pad_00007"``.
    source_name : str
        Key of the :class:`~tsara.synthetic.config.SourceSpec` that spawned it.
    center_time : pandas.Timestamp
        The shape's center parameter (Gaussian mu), *before* per-species lag.
    kernel : PlumeKernel
        Normalized temporal shape.
    reference_species : str
        Species whose amplitude was drawn directly.
    amplitudes : dict of str to float
        Species -> true peak enhancement.
    ratios : dict of str to float
        Species -> realized true ratio to the reference species (the
        reference species maps to exactly 1.0).
    lags_s : dict of str to float
        Species -> signed arrival offset in seconds.
    parent_event_id : str or None
        Set on nested child events; None for top-level events.
    """

    event_id: str
    source_name: str
    center_time: pd.Timestamp
    kernel: PlumeKernel
    reference_species: str
    amplitudes: dict[str, float]
    ratios: dict[str, float]
    lags_s: dict[str, float] = field(default_factory=dict)
    parent_event_id: str | None = None

    def species_center(self, species: str) -> pd.Timestamp:
        """Return the shape center for ``species``, including its lag.

        Parameters
        ----------
        species : str
            Canonical species name.

        Returns
        -------
        pandas.Timestamp
            Lag-adjusted center time.
        """
        import pandas as pd

        return self.center_time + pd.Timedelta(seconds=self.lags_s.get(species, 0.0))

    def species_peak_time(self, species: str) -> pd.Timestamp:
        """Return the true peak time for ``species`` (center + lag + mode offset).

        Parameters
        ----------
        species : str
            Canonical species name.

        Returns
        -------
        pandas.Timestamp
            Time of the shape's maximum for this species.
        """
        import pandas as pd

        return self.species_center(species) + pd.Timedelta(seconds=self.kernel.peak_offset_s)


def _draw_amplitude(spec: AmplitudeSpec, rng: np.random.Generator) -> float:
    """Draw one peak amplitude from the configured distribution.

    Parameters
    ----------
    spec : AmplitudeSpec
        Lognormal or uniform amplitude configuration.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    float
        Peak enhancement in the reference species' units.
    """
    if isinstance(spec, LognormalAmplitude):
        # Parameterized by the median, so exp(mu) = median directly.
        return float(spec.median * math.exp(spec.sigma_log * rng.normal()))
    return float(rng.uniform(spec.low, spec.high))


def _draw_ratio(spec: RatioSpec, rng: np.random.Generator) -> float:
    """Draw one realized enhancement ratio.

    Uses the lognormal parameterization from
    :meth:`~tsara.synthetic.config.RatioSpec.lognormal_parameters`, so the
    configured ``mean`` is the true arithmetic mean of the draws and
    ``relative_spread`` is their true relative standard deviation. A
    zero-spread spec returns the mean exactly.

    Parameters
    ----------
    spec : RatioSpec
        Ratio distribution.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    float
        Realized ratio for one event; always strictly positive.
    """
    mu, sigma_log = spec.lognormal_parameters()
    if sigma_log == 0.0:
        return float(spec.mean)
    return float(math.exp(mu + sigma_log * rng.normal()))


def schedule_events(config: SyntheticConfig, rng: np.random.Generator) -> list[RealizedEvent]:
    """Draw and realize every plume event in the run.

    Events arrive as a homogeneous Poisson process: the count over the record
    is ``Poisson(rate_per_hour x hours)`` and, conditional on that count,
    onset times are i.i.d. uniform over the span — the standard and exactly
    correct construction. No minimum spacing is imposed, so events *will*
    overlap at high rates. That is intentional: overlapping plumes are the
    realistic near-source condition and the case in which naive baselines and
    signal-MAD noise estimates both fail.

    Centers fall strictly inside ``[start, start + duration)``. An event
    landing within its own support of the end therefore has its tail cut off,
    which is realistic. The start edge is **not** symmetric: because no center
    is ever drawn before the record begins, a generated record can never open
    part-way through a plume whose peak already passed — a condition real
    records routinely contain.

    That asymmetry is a known limitation of the harness rather than a
    modelling claim. It matters most to stages with distinct edge behaviour —
    Phase 5's rolling baselines and Phase 6's detection both work with
    half-empty windows at the record boundary — so if either needs the
    already-in-progress case, widen the draw to
    ``[-support_before, span + support_after)`` and scale ``n_events`` for the
    wider window. Events falling entirely outside would then produce catalog
    rows with ``sampled_peak_amplitude`` NaN, the same flag already used for
    an event lost inside a data gap.

    Parameters
    ----------
    config : SyntheticConfig
        The full run configuration.
    rng : numpy.random.Generator
        Source of randomness, threaded through the whole build for
        reproducibility.

    Returns
    -------
    list of RealizedEvent
        All events, parents and nested children, sorted by center time.
    """
    import pandas as pd

    # Normalize to tz-naive UTC here, at the one point where a catalog
    # timestamp is born: every event time below is this value plus a
    # Timedelta, so fixing the representation once fixes the whole catalog.
    # It must match the streams' clocks (normalized identically in
    # `generator._build_times`), because the harness's central operation is
    # slicing a stream with a ground-truth event window — and pandas raises
    # TypeError on any comparison between an aware and a naive timestamp.
    start = _to_utc_naive_stamp(pd.Timestamp(config.start))
    span = pd.Timedelta(config.duration)
    span_hours = span.total_seconds() / 3600.0
    span_s = span.total_seconds()

    events: list[RealizedEvent] = []

    for source_name, source in config.sources.items():
        kernel = build_kernel(source.shape)
        expected = source.rate_per_hour * span_hours
        n_events = int(rng.poisson(expected))
        logger.debug(
            "Source %r: drew %d events (expected %.1f over %.2f h).",
            source_name,
            n_events,
            expected,
            span_hours,
        )
        if n_events == 0:
            continue

        offsets_s = np.sort(rng.uniform(0.0, span_s, size=n_events))
        for index, offset_s in enumerate(offsets_s):
            center = start + pd.Timedelta(seconds=float(offset_s))
            parent = _realize_event(
                event_id=f"{source_name}_{index:05d}",
                source_name=source_name,
                source=source,
                kernel=kernel,
                center=center,
                rng=rng,
            )
            events.append(parent)

            if source.nested is not None and rng.random() < source.nested.probability:
                events.append(_realize_child(parent, source, rng))

    events.sort(key=lambda event: event.center_time)
    return events


def _realize_event(
    *,
    event_id: str,
    source_name: str,
    source: SourceSpec,
    kernel: PlumeKernel,
    center: pd.Timestamp,
    rng: np.random.Generator,
) -> RealizedEvent:
    """Realize one parent event's amplitudes, ratios, and lags.

    Parameters
    ----------
    event_id : str
        Identifier to assign.
    source_name : str
        Owning source key.
    source : SourceSpec
        Source configuration.
    kernel : PlumeKernel
        Precomputed shape.
    center : pandas.Timestamp
        Shape center.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    RealizedEvent
        Fully realized parent event.
    """
    import pandas as pd

    reference_amplitude = _draw_amplitude(source.amplitude, rng)

    amplitudes = {source.reference_species: reference_amplitude}
    ratios = {source.reference_species: 1.0}
    for species, ratio_spec in source.ratios.items():
        ratio = _draw_ratio(ratio_spec, rng)
        ratios[species] = ratio
        # The defining relation: ratio = delta_species / delta_reference.
        amplitudes[species] = reference_amplitude * ratio

    lags_s = {
        species: float(pd.Timedelta(lag).total_seconds())
        for species, lag in source.inter_species_lag.items()
    }

    return RealizedEvent(
        event_id=event_id,
        source_name=source_name,
        center_time=center,
        kernel=kernel,
        reference_species=source.reference_species,
        amplitudes=amplitudes,
        ratios=ratios,
        lags_s=lags_s,
    )


def _realize_child(
    parent: RealizedEvent, source: SourceSpec, rng: np.random.Generator
) -> RealizedEvent:
    """Realize a nested child plume riding inside ``parent``.

    The child is placed uniformly within +/-1 sigma of the parent's center,
    guaranteeing it sits inside the parent's high-amplitude region — which is
    what makes it a genuinely *nested* detection problem rather than two
    adjacent plumes.

    By default the child inherits the parent's realized ratios (same source,
    finer temporal structure). When ``nested.ratios`` is configured it draws
    its own, making the child a chemically distinct source superimposed on
    the parent — the harder and more scientifically interesting case, since a
    regression that pools parent and child samples then recovers neither
    true ratio.

    Parameters
    ----------
    parent : RealizedEvent
        The event this child is nested inside.
    source : SourceSpec
        Parent's source configuration; ``source.nested`` must not be None.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    RealizedEvent
        The child event, with ``parent_event_id`` set.
    """
    import pandas as pd

    nested = source.nested
    assert nested is not None  # guarded by the caller

    child_kernel = build_kernel(nested.shape)
    jitter_s = float(rng.uniform(-parent.kernel.sigma_s, parent.kernel.sigma_s))
    child_center = parent.center_time + pd.Timedelta(seconds=jitter_s)

    child_reference_amplitude = (
        parent.amplitudes[parent.reference_species] * nested.amplitude_factor
    )

    if nested.ratios is None:
        child_ratios = dict(parent.ratios)
    else:
        child_ratios = {parent.reference_species: 1.0}
        for species, ratio_spec in nested.ratios.items():
            child_ratios[species] = _draw_ratio(ratio_spec, rng)
        # Species the parent emits but the child's ratio table omits still
        # participate, inheriting the parent's realized ratio; otherwise a
        # partial override would silently delete species from the child.
        for species, ratio in parent.ratios.items():
            child_ratios.setdefault(species, ratio)

    child_amplitudes = {
        species: child_reference_amplitude * ratio for species, ratio in child_ratios.items()
    }

    return RealizedEvent(
        event_id=f"{parent.event_id}_child",
        source_name=parent.source_name,
        center_time=child_center,
        kernel=child_kernel,
        reference_species=parent.reference_species,
        amplitudes=child_amplitudes,
        ratios=child_ratios,
        lags_s=dict(parent.lags_s),
        parent_event_id=parent.event_id,
    )


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

#: Column order of the ground-truth table. Fixed and explicit because this
#: schema is the contract that Phase 6's PlumeCatalog must remain
#: diff-compatible with; a silent column reordering would break scoring code
#: that positional-indexes the parquet file.
GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    "event_id",
    "parent_event_id",
    "source_name",
    "species",
    "instrument",
    "reference_species",
    "start_time",
    "peak_time",
    "end_time",
    "true_amplitude",
    "sampled_peak_amplitude",
    "true_baseline_at_peak",
    "true_ratio_to_reference",
    "latitude",
    "longitude",
)


@dataclass(frozen=True, eq=False)
class GroundTruthEvent:
    """One (event, species) row of the answer key.

    One physical event contributes one row per participating species, mirroring
    how the Phase 6 catalog will store per-species enhancements. Both a
    continuous and a sampled amplitude are recorded because they answer
    different questions: ``true_amplitude`` is what the *source* did, while
    ``sampled_peak_amplitude`` is the largest value the instrument could
    possibly have seen given its sampling rate. A detector cannot be faulted
    for missing the difference between them.

    Attributes
    ----------
    event_id : str
        Unique event identifier.
    parent_event_id : str or None
        Parent's ``event_id`` for nested children; None otherwise.
    source_name : str
        Source that spawned the event.
    species : str
        Canonical species name this row describes.
    instrument : str
        Instrument that measures this species.
    reference_species : str
        Denominator species of ``true_ratio_to_reference``.
    start_time, peak_time, end_time : pandas.Timestamp
        Support boundaries and true maximum, lag-adjusted for this species.
    true_amplitude : float
        Continuous peak enhancement of the injected shape.
    sampled_peak_amplitude : float
        Largest injected contribution actually landing on a sample time; NaN
        if the event fell entirely inside a data gap.
    true_baseline_at_peak : float
        Background value underneath the peak — the number a baseline
        estimator is trying to recover.
    true_ratio_to_reference : float
        Realized ratio for this event; exactly 1.0 for the reference species.
    latitude, longitude : float or None
        Platform position at the peak time, or the fixed site position.
    """

    event_id: str
    parent_event_id: str | None
    source_name: str
    species: str
    instrument: str
    reference_species: str
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    end_time: pd.Timestamp
    true_amplitude: float
    sampled_peak_amplitude: float
    true_baseline_at_peak: float
    true_ratio_to_reference: float
    latitude: float | None = None
    longitude: float | None = None


@dataclass(frozen=True, eq=False)
class GroundTruth:
    """The complete answer key for one synthetic dataset.

    Held as typed rows for readable assertions in tests, with
    :meth:`to_frame` / :meth:`from_frame` for Parquet persistence — the same
    format CLAUDE.md §5 fixes for the plume catalog, so the two are directly
    comparable on disk.

    Note that only *realized* per-event ratios are stored. The population
    parameters they were drawn from (``RatioSpec.mean`` and
    ``relative_spread``) live in the ``SyntheticConfig`` saved alongside in
    the same bundle, so both the point-estimate question ("did the fit find
    the right ratio?") and the coverage question ("does the reported
    confidence interval actually contain the truth at its nominal rate?")
    remain answerable without duplicating data.
    """

    events: tuple[GroundTruthEvent, ...]

    def __len__(self) -> int:
        """Return the number of (event, species) rows."""
        return len(self.events)

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Unique event identifiers, in first-appearance order."""
        seen: dict[str, None] = {}
        for event in self.events:
            seen.setdefault(event.event_id)
        return tuple(seen)

    def for_species(self, species: str) -> tuple[GroundTruthEvent, ...]:
        """Return all rows describing one species.

        Parameters
        ----------
        species : str
            Canonical species name.

        Returns
        -------
        tuple of GroundTruthEvent
            Matching rows, in catalog order.
        """
        return tuple(event for event in self.events if event.species == species)

    def to_frame(self) -> pd.DataFrame:
        """Render the catalog as a DataFrame with :data:`GROUND_TRUTH_COLUMNS`.

        Returns
        -------
        pandas.DataFrame
            One row per (event, species), columns in fixed order.
        """
        import pandas as pd

        records = [
            {column: getattr(event, column) for column in GROUND_TRUTH_COLUMNS}
            for event in self.events
        ]
        frame = pd.DataFrame.from_records(records, columns=list(GROUND_TRUTH_COLUMNS))

        # Pin the time columns on BOTH paths, not just the empty one. Left to
        # itself pandas infers the unit from the values it was handed — so a
        # populated catalog inherits whatever resolution `config.start` had
        # (microseconds, for a `datetime.datetime`) while an empty one gets
        # whatever `astype` defaults to. Two catalogs from the same package
        # would then differ in dtype purely by whether they contained events,
        # and could not be concatenated without an upcast. Nanoseconds match
        # the stream clocks (`generator._build_times`) and netCDF's storage
        # unit, so the whole bundle speaks one time representation.
        for column in ("start_time", "peak_time", "end_time"):
            frame[column] = frame[column].astype("datetime64[ns]")

        if frame.empty:
            # An empty frame from an empty record list has object dtypes for
            # everything else, which Parquet round-trips inconsistently.
            # Impose the rest of the schema so a plume-free control dataset
            # saves and loads like any other.
            for column in (
                "true_amplitude",
                "sampled_peak_amplitude",
                "true_baseline_at_peak",
                "true_ratio_to_reference",
                "latitude",
                "longitude",
            ):
                frame[column] = frame[column].astype(float)
            for column in (
                "event_id",
                "parent_event_id",
                "source_name",
                "species",
                "instrument",
                "reference_species",
            ):
                frame[column] = frame[column].astype(object)
        return frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> GroundTruth:
        """Rebuild a catalog from a DataFrame produced by :meth:`to_frame`.

        Parameters
        ----------
        frame : pandas.DataFrame
            Table with :data:`GROUND_TRUTH_COLUMNS`.

        Returns
        -------
        GroundTruth
            Reconstructed catalog.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        import pandas as pd

        missing = set(GROUND_TRUTH_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"Ground-truth frame is missing columns {sorted(missing)}; "
                f"expected {list(GROUND_TRUTH_COLUMNS)}."
            )
        # Records (rather than itertuples) because every cell arrives as a
        # loosely-typed object and each field needs its own explicit
        # narrowing anyway; going through dicts keeps that narrowing visible
        # instead of hiding it behind attribute access.
        events = tuple(
            GroundTruthEvent(
                event_id=str(row["event_id"]),
                # Parquet round-trips a None object column as NaN, so both
                # spellings of "no parent" must map back to None.
                parent_event_id=(
                    None if _is_missing(row["parent_event_id"]) else str(row["parent_event_id"])
                ),
                source_name=str(row["source_name"]),
                species=str(row["species"]),
                instrument=str(row["instrument"]),
                reference_species=str(row["reference_species"]),
                start_time=pd.Timestamp(row["start_time"]),
                peak_time=pd.Timestamp(row["peak_time"]),
                end_time=pd.Timestamp(row["end_time"]),
                true_amplitude=float(row["true_amplitude"]),
                sampled_peak_amplitude=float(row["sampled_peak_amplitude"]),
                true_baseline_at_peak=float(row["true_baseline_at_peak"]),
                true_ratio_to_reference=float(row["true_ratio_to_reference"]),
                latitude=None if _is_missing(row["latitude"]) else float(row["latitude"]),
                longitude=None if _is_missing(row["longitude"]) else float(row["longitude"]),
            )
            for row in frame.to_dict(orient="records")
        )
        return cls(events=events)


def _is_missing(value: object) -> bool:
    """Return True for None/NaN/pandas-NA, without assuming a dtype.

    Parameters
    ----------
    value : object
        Candidate value from a DataFrame cell.

    Returns
    -------
    bool
        True when the value represents "missing".
    """
    import pandas as pd

    if value is None:
        return True
    # pd.isna is typed against concrete scalar/array types; the value here
    # comes from an object-dtype cell, so the cast records that we are
    # deliberately handing it a scalar of unknown static type.
    return bool(pd.isna(cast("float", value)))
