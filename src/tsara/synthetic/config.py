"""Synthetic-dataset configuration schema: *what fake reality to manufacture*.

Where :mod:`tsara.config.manifest` says "here is how to read data that
exists" and :mod:`tsara.config.analysis` says "here is what to do with it",
this module says "here is a world to invent, and here is exactly what is
true about it". The generator (:mod:`tsara.synthetic.generator`) turns one
:class:`SyntheticConfig` into per-instrument ``xarray.Dataset`` streams plus
a :class:`~tsara.synthetic.plumes.GroundTruth` record of every fact the
analysis pipeline will later have to rediscover.

Design decisions embedded in this schema
-----------------------------------------
* **Same conventions as the manifest schema.** ``StrictModel`` (``extra=
  "forbid"``, frozen), ``kind``/``mode``-discriminated unions for
  polymorphism, tuples for sweepable/ordered collections, duration strings
  validated at parse time. A reader of the manifest schema should find
  nothing surprising here.
* **Sources are multi-species, not per-species.** A plume is a *physical
  event*: one leak emits CH4 and C2H6 together in a fixed ratio. Modelling
  each species' plumes independently would destroy the very covariance
  TSARA exists to measure, so :class:`SourceSpec` emits a correlated bundle
  of species at once and the true enhancement ratio is a first-class
  configured quantity (:class:`RatioSpec`).
* **"True" uncertainty is a generator concept, not a manifest one.**
  :class:`UncertaintySpec` (manifest) describes how to *read* an uncertainty
  that an instrument already reports. The generator needs the inverse: the
  parameters from which to *manufacture* error. :class:`TrueComponent`
  therefore mirrors ``DeclaredUncertainty``'s absolute/relative form but adds
  ``report_as`` — the raw-file column an instrument would publish its
  per-point sigma under. :meth:`TrueUncertainty.to_manifest_uncertainty`
  converts back, so the budget a synthetic instrument *has* and the budget a
  manifest would *declare* for it stay structurally in step: same components,
  same absolute/relative terms, same reported-column names. It is the same
  *declaration*, not the same *budget* — ``report_bias`` is deliberately not
  carried across, because a manifest cannot know that an instrument
  understates its own error. That gap is the point: it is what makes
  "does downstream UQ notice an optimistic instrument?" a testable question.
* **Real-data profiles are referenced by name, never embedded.** A
  :class:`BootstrapBackground` names a profile key; the actual numeric
  residual blocks (which are derived point-for-point from real
  measurements) are passed to the generator separately at call time. This
  keeps every ``SyntheticConfig`` losslessly YAML-round-trippable and
  guarantees that no real-data-derived numbers can ever end up committed
  inside a config file.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from tsara.config.base import StrictModel as _StrictModel
from tsara.config.base import validate_positive_timedelta as _validate_duration
from tsara.config.base import validate_signed_timedelta as _validate_signed
from tsara.config.manifest import (
    DeclaredUncertainty,
    ReportedUncertainty,
    UncertaintySpec,
    VariableRole,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import numpy.typing as npt

#: Prefix marking generator-emitted answer-key variables. Defined here, in the
#: schema layer, so validation can reserve it without importing the generator
#: (which imports this module); :mod:`tsara.synthetic.generator` re-exports it.
TRUTH_PREFIX = "truth_"


# ---------------------------------------------------------------------------
# Background models (tagged union on 'kind')
# ---------------------------------------------------------------------------


class ParametricBackground(_StrictModel):
    """Analytic background: constant + diurnal cycle + linear drift + wander.

    The four terms are deliberately separable so a test can switch on exactly
    one of them:

    * ``offset`` — the clean-air baseline level (e.g. ~1900 ppb CH4).
    * ``diurnal_amplitude`` — the boundary-layer breathing cycle that makes
      background estimation non-trivial: a rolling low quantile must track a
      *moving* background, not a constant one.
    * ``drift_per_day`` — slow instrument or seasonal drift.
    * ``random_walk_std`` — non-stationary low-frequency wander with no
      analytic form, the hardest case for a baseline estimator because it is
      unpredictable yet genuinely background (not enhancement).
    """

    kind: Literal["parametric"] = "parametric"
    offset: float = Field(description="Constant background level in the species' units.")
    diurnal_amplitude: float = Field(
        default=0.0, ge=0, description="Half-peak-to-peak amplitude of the diurnal cycle."
    )
    diurnal_period: str = Field(
        default="24h", description="Period of the cyclic term (a duration string)."
    )
    diurnal_phase_hours: float = Field(
        default=0.0, description="Phase shift in hours; 0 puts the minimum at midnight UTC."
    )
    drift_per_day: float = Field(
        default=0.0, description="Linear trend, in species units per day (may be negative)."
    )
    random_walk_std: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Standard deviation the random-walk term accumulates over one day. "
            "Increments are scaled as sqrt(dt/1day) so the wander magnitude is "
            "independent of the instrument's sampling rate."
        ),
    )

    @field_validator("diurnal_period")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"ParametricBackground.{info.field_name}")
        return value


class BootstrapBackground(_StrictModel):
    """Background whose *fluctuations* are block-bootstrapped from real data.

    The highest-realism option: rather than assuming a noise shape, resample
    contiguous blocks of real (baseline-subtracted) fluctuation from a
    :class:`~tsara.synthetic.profiling.RealDataProfile` built off a live data
    mount. Blocks preserve whatever short-range autocorrelation, skew, and
    instrument quirks the real record has — structure that no parametric
    noise model reproduces.

    Two deliberate limitations, documented rather than hidden (METHODS.md
    §8.3):

    * Blocks are **mean-centred**, so only *within-block* (high-frequency)
      structure survives; between-block low-frequency structure is discarded
      to avoid step discontinuities at the seams where blocks are stitched.
      Slow structure is therefore supplied by ``base`` instead.
    * Because the source record is plume-dense, some real plume energy leaks
      through the profiling baseline into the residual. This is treated as a
      *feature* — it is precisely the adversarial "is `diff_mad` really
      plume-immune on my instrument?" test case — but it means the substrate
      is not a pure noise realization.
    """

    kind: Literal["bootstrap"] = "bootstrap"
    profile: str = Field(
        min_length=1,
        description=(
            "Key into the profiles mapping passed to the generator at call "
            "time. Deliberately a name, not embedded data: real-derived "
            "numbers must never be serialized into a config file."
        ),
    )
    base: ParametricBackground | None = Field(
        default=None,
        description=(
            "Optional slow structure (level, diurnal, drift) added under the "
            "bootstrapped fluctuations. None uses the profile's own median "
            "level as a flat offset."
        ),
    )
    scale: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Multiplier on the bootstrapped fluctuations; 2.0 doubles the "
            "real noise amplitude for sensitivity testing without changing "
            "its correlation structure."
        ),
    )

    @field_validator("profile")
    @classmethod
    def _profile_key_not_blank(cls, value: str) -> str:
        """Reject a whitespace-only profile key.

        ``min_length=1`` already rejects the empty string, but a key of
        spaces would survive validation and fail much later inside the
        generator, reporting that an apparently-invisible profile name was
        never supplied. Catching it at the field keeps the error where the
        mistake is.
        """
        if not value.strip():
            raise ValueError("BootstrapBackground.profile must not be blank.")
        return value


#: Tagged union on 'kind', matching the manifest schema's convention.
BackgroundConfig = Annotated[
    ParametricBackground | BootstrapBackground, Field(discriminator="kind")
]


# ---------------------------------------------------------------------------
# True uncertainty (the generator's inverse of the manifest's UncertaintySpec)
# ---------------------------------------------------------------------------


class TrueComponent(_StrictModel):
    """One component (random or systematic) of a synthetic instrument's error.

    Mirrors :class:`~tsara.config.manifest.DeclaredUncertainty`'s two-term
    quadrature form, ``sigma_i = sqrt(absolute^2 + (relative * x_i)^2)``,
    because that is how instrument teams actually specify precision — and
    because keeping the forms identical is what makes
    :meth:`TrueUncertainty.to_manifest_uncertainty` exact rather than
    approximate.

    ``report_as`` is the one field with no manifest counterpart: setting it
    makes this synthetic instrument *publish* its per-point sigma as a named
    variable, reproducing the EM27-style ``reported`` mode. The published
    column may be deliberately imperfect via ``report_bias`` — a real
    instrument's self-reported error is an estimate, and a pipeline that
    silently trusts it should be testable against one that is 20 % optimistic.
    """

    absolute: float = Field(default=0.0, ge=0, description="Noise floor in the species' units.")
    relative: float = Field(
        default=0.0, ge=0, lt=1, description="Fraction of reading, e.g. 0.02 for 2 %."
    )
    report_as: str | None = Field(
        default=None,
        description=(
            "If set, the per-point sigma of this component is emitted as a "
            "variable of this name (the 'reported' uncertainty mode). If "
            "None, the component is real but undeclared — the case that "
            "forces the empirical fallback estimator downstream."
        ),
    )
    report_bias: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Multiplier between the sigma actually used to draw errors and "
            "the sigma reported in the 'report_as' column. 1.0 = honest "
            "instrument; 0.8 = an instrument that understates its own error "
            "by 20 %, which downstream UQ ought to expose (METHODS.md §5.1)."
        ),
    )

    @model_validator(mode="after")
    def _nonzero(self) -> TrueComponent:
        """Reject a component that generates nothing.

        Mirrors ``DeclaredUncertainty._nonzero``: a component with both terms
        zero adds no error, so declaring it is a config mistake.
        """
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError(
                "TrueComponent with absolute=0 and relative=0 injects no error; "
                "omit the component instead."
            )
        return self

    def sigma(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the per-point 1-sigma for ``values``.

        Kept here rather than in the noise module so the *definition* of the
        two-term quadrature form lives in exactly one place, next to the
        fields that parameterize it.

        numpy is imported inside the function (and only for typing at module
        scope) so that importing the config schema stays cheap: validating a
        YAML file should not pull in the numeric stack.

        Parameters
        ----------
        values : numpy.ndarray
            The true (noise-free) signal values.

        Returns
        -------
        numpy.ndarray
            Per-point standard deviation, same shape as ``values``.
        """
        import numpy

        arr = numpy.asarray(values, dtype=float)
        result: npt.NDArray[np.float64] = numpy.sqrt(self.absolute**2 + (self.relative * arr) ** 2)
        return result


class TrueUncertainty(_StrictModel):
    """The complete, *known* error budget injected into one synthetic species.

    Ground truth for the whole uncertainty system: the analysis pipeline's
    job is to recover these numbers (or to correctly report that it cannot),
    and Phase 7's combined-UQ machinery can only be scored because this
    object says what the answer was.

    The component semantics match METHODS.md §2.1 exactly — ``random`` is
    drawn independently per point (optionally AR(1)-correlated via
    ``decorrelation_timescale``) and averages down; ``systematic`` is drawn
    **once per species per run** and applied to every point, so it does not
    average down no matter how much data is aggregated.
    """

    random: TrueComponent | None = Field(
        default=None, description="Uncorrelated point-to-point noise component."
    )
    systematic: TrueComponent | None = Field(
        default=None,
        description=(
            "Correlated component, drawn once per run and applied to all "
            "points (a calibration scale/offset error)."
        ),
    )
    decorrelation_timescale: str | None = Field(
        default=None,
        description=(
            "If set, the random component is generated as an AR(1) process "
            "with rho = exp(-dt/tau) instead of white noise. This is the only "
            "way to obtain data with a *known* tau, which is what the open "
            "N_eff estimator question (METHODS.md §3.4) needs to be tested "
            "against."
        ),
    )

    @field_validator("decorrelation_timescale")
    @classmethod
    def _valid_decorrelation_timescale(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_duration(value, field="TrueUncertainty.decorrelation_timescale")
        return value

    @model_validator(mode="after")
    def _at_least_one_component(self) -> TrueUncertainty:
        if self.random is None and self.systematic is None:
            raise ValueError(
                "TrueUncertainty with no 'random' and no 'systematic' component "
                "injects nothing; omit the 'uncertainty' block instead."
            )
        if self.decorrelation_timescale is not None and self.random is None:
            raise ValueError(
                "TrueUncertainty.decorrelation_timescale describes the random "
                "component's autocorrelation, but no random component is declared."
            )
        return self

    def to_manifest_uncertainty(self) -> UncertaintySpec:
        """Convert to the :class:`UncertaintySpec` a manifest would declare.

        This is the seam that keeps the generator and the ingestion schema
        honest about each other. A component with ``report_as`` set becomes
        :class:`~tsara.config.manifest.ReportedUncertainty` naming that
        column; a component without becomes
        :class:`~tsara.config.manifest.DeclaredUncertainty` carrying the same
        absolute/relative terms it was generated from.

        Phase 3 will use this to emit a matching manifest alongside synthetic
        raw files, so ingestion can be round-trip tested against data whose
        true budget is known.

        Returns
        -------
        UncertaintySpec
            The manifest-side declaration of this budget.

        Raises
        ------
        ValueError
            If the budget declares nothing at all (guarded by the validator
            above, so unreachable through normal construction).
        """

        def _convert(
            component: TrueComponent | None,
        ) -> DeclaredUncertainty | ReportedUncertainty | None:
            if component is None:
                return None
            if component.report_as is not None:
                return ReportedUncertainty(column=component.report_as)
            return DeclaredUncertainty(absolute=component.absolute, relative=component.relative)

        return UncertaintySpec(
            random=_convert(self.random),
            systematic=_convert(self.systematic),
            decorrelation_timescale=self.decorrelation_timescale,
        )


# ---------------------------------------------------------------------------
# Species and instruments
# ---------------------------------------------------------------------------


class SpeciesSpec(_StrictModel):
    """One measured variable on a synthetic instrument.

    The variable's canonical name is its key in ``InstrumentSpec.species``.
    ``role`` is reused verbatim from the manifest schema so synthetic streams
    carry the same role vocabulary real ones will: only ``role="gas"``
    variables receive plumes, which is what makes a ``met`` wind-direction
    variable expressible here (needed to exercise circular statistics in
    Phase 4) without inventing a parallel concept.
    """

    background: BackgroundConfig = Field(description="How the plume-free signal is built.")
    role: VariableRole = Field(
        default="gas",
        description=(
            "Downstream handling category, same vocabulary as the manifest. "
            "Only role='gas' variables participate in plume events."
        ),
    )
    units: str = Field(default="", description="Units label, carried into the stream attrs.")
    uncertainty: TrueUncertainty | None = Field(
        default=None,
        description=(
            "The known error budget injected into this species. None means a "
            "noise-free variable (useful for isolating algorithm error from "
            "measurement error in a test)."
        ),
    )
    quantization: float | None = Field(
        default=None,
        gt=0,
        description=(
            "If set, values are rounded to this step, reproducing a logger "
            "that reports at coarse resolution. The required adversarial case "
            "for the median-based noise estimators, which collapse to zero "
            "when more than half a window shares one value (METHODS.md §2.5)."
        ),
    )
    circular: bool = Field(
        default=False,
        description=(
            "True for angular quantities (wind direction): values are wrapped "
            "into [0, 360) after generation."
        ),
    )

    @model_validator(mode="after")
    def _circular_only_for_met(self) -> SpeciesSpec:
        """Mirror the manifest's rule: circular is meaningful only for met."""
        if self.circular and self.role != "met":
            raise ValueError(
                f"circular=true is only valid for role='met' variables (got role='{self.role}')."
            )
        return self


class DropoutSpec(_StrictModel):
    """Random instrument outages that remove samples entirely.

    Rows are **dropped**, not NaN-filled, because that is what a real logger
    does when it stops writing — and because the resulting irregular native
    timestamps are exactly the condition METHODS.md §1.1 claims the rolling
    machinery handles. A generator that only ever produced perfectly regular
    grids would leave that claim untested.
    """

    rate_per_day: float = Field(gt=0, description="Mean number of outages per day (Poisson).")
    duration: str = Field(description="Mean outage length; actual lengths are exponential.")

    @field_validator("duration")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"DropoutSpec.{info.field_name}")
        return value


class InstrumentSpec(_StrictModel):
    """One synthetic instrument: a clock, and the species sharing it.

    Species on one instrument share a time axis (METHODS.md §1.1), so the
    native rate lives here rather than on the species. Multiple instruments
    with different ``native_rate`` values is how this generator produces the
    multi-rate streams the whole "synchronize late" architecture exists to
    handle.
    """

    native_rate: str = Field(
        description="Nominal sampling interval, e.g. '1s' for 1 Hz, '0.1s' for 10 Hz."
    )
    species: dict[str, SpeciesSpec] = Field(
        min_length=1, description="Mapping of canonical variable name -> spec."
    )
    timestamp_jitter: str | None = Field(
        default=None,
        description=(
            "If set, each timestamp is perturbed by a uniform draw in "
            "+/- this amount, producing a non-uniform clock. Must be under "
            "half the native rate so timestamps stay strictly increasing."
        ),
    )
    dropouts: DropoutSpec | None = Field(
        default=None, description="Optional random outages that delete samples."
    )

    @field_validator("native_rate")
    @classmethod
    def _valid_rate(cls, value: str) -> str:
        _validate_duration(value, field="InstrumentSpec.native_rate")
        return value

    @field_validator("timestamp_jitter")
    @classmethod
    def _valid_jitter(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_duration(value, field="InstrumentSpec.timestamp_jitter")
        return value

    @field_validator("species")
    @classmethod
    def _canonical_names_are_identifiers(
        cls, value: dict[str, SpeciesSpec]
    ) -> dict[str, SpeciesSpec]:
        """Require identifiers, as the manifest does: names become xarray variables."""
        for name in value:
            if not name.isidentifier():
                raise ValueError(
                    f"Canonical species name '{name}' must be a valid identifier "
                    "(letters, digits, underscores; not starting with a digit)."
                )
        return value

    @model_validator(mode="after")
    def _jitter_below_half_rate(self) -> InstrumentSpec:
        """Jitter must not be able to reorder samples.

        With a uniform draw in +/- j, two adjacent timestamps dt apart stay
        ordered as long as 2j < dt, i.e. j < dt/2. Enforcing it here means the
        generator never has to sort (and never silently produces a clock that
        runs backwards).
        """
        import pandas as pd

        if self.timestamp_jitter is None:
            return self
        jitter = pd.Timedelta(self.timestamp_jitter)
        rate = pd.Timedelta(self.native_rate)
        if jitter * 2 >= rate:
            raise ValueError(
                f"InstrumentSpec.timestamp_jitter ({self.timestamp_jitter}) must be "
                f"less than half native_rate ({self.native_rate}), otherwise "
                "timestamps could stop being strictly increasing."
            )
        return self

    @model_validator(mode="after")
    def _reported_columns_are_unique(self) -> InstrumentSpec:
        """Reject a 'report_as' column colliding with a species or another column.

        Reported-sigma variables share the instrument's namespace, so a
        collision would silently overwrite one variable with another. The
        ``truth_`` prefix is reserved for the same reason: those variables are
        the answer key, and a reported column shadowing one would corrupt the
        very record later phases are scored against.
        """
        seen: dict[str, str] = {}
        for name, spec in self.species.items():
            if spec.uncertainty is None:
                continue
            for label, component in (
                ("random", spec.uncertainty.random),
                ("systematic", spec.uncertainty.systematic),
            ):
                if component is None or component.report_as is None:
                    continue
                column = component.report_as
                if column.startswith(TRUTH_PREFIX):
                    raise ValueError(
                        f"Species '{name}' reports its {label} sigma as '{column}', "
                        f"but the '{TRUTH_PREFIX}' prefix is reserved for "
                        "generator-emitted ground-truth variables."
                    )
                if column in self.species:
                    raise ValueError(
                        f"Species '{name}' reports its {label} sigma as '{column}', "
                        "which collides with a declared species name."
                    )
                if column in seen:
                    raise ValueError(
                        f"Reported-sigma column '{column}' is claimed by both "
                        f"{seen[column]} and {name}.{label}."
                    )
                seen[column] = f"{name}.{label}"
        return self


# ---------------------------------------------------------------------------
# Plume shapes (tagged union on 'kind')
# ---------------------------------------------------------------------------


class GaussianShape(_StrictModel):
    """Symmetric Gaussian plume: pure turbulent dispersion, no residence time.

    The textbook case. Symmetric shapes are the easy problem for detection
    and integration; :class:`EMGShape` is the realistic one.
    """

    kind: Literal["gaussian"] = "gaussian"
    sigma: str = Field(description="Gaussian width as a duration, e.g. '20s'.")

    @field_validator("sigma")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"GaussianShape.{info.field_name}")
        return value


class EMGShape(_StrictModel):
    """Exponentially-modified Gaussian: dispersion convolved with residence time.

    The physically-motivated plume shape — a Gaussian (turbulent mixing)
    convolved with a decaying exponential (residence time in the source
    volume / inlet and cavity flushing). Produces the sharp rise and long
    trailing tail that real plume transects show, and it is that asymmetric
    tail that makes baseline placement genuinely hard: the tail decays
    asymptotically, so "where the plume ends" has no crisp answer.

    ``tau`` is the exponential decay timescale; ``tau`` -> 0 recovers a
    Gaussian.
    """

    kind: Literal["emg"] = "emg"
    sigma: str = Field(description="Gaussian (rise) width as a duration.")
    tau: str = Field(description="Exponential tail decay timescale as a duration.")

    @field_validator("sigma", "tau")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"EMGShape.{info.field_name}")
        return value


PlumeShape = Annotated[GaussianShape | EMGShape, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Amplitude distributions (tagged union on 'kind')
# ---------------------------------------------------------------------------


class LognormalAmplitude(_StrictModel):
    """Right-skewed peak amplitudes: many modest plumes, a few large ones.

    The realistic default. Emission-rate distributions across sources (and
    the distribution of how closely a mobile platform passes them) are both
    heavy-tailed, so the observed enhancement distribution is strongly
    right-skewed. Parameterized by the *median* rather than the mean because
    the median is the robust, intuitive "typical plume" for a skewed law.
    """

    kind: Literal["lognormal"] = "lognormal"
    median: float = Field(gt=0, description="Median peak enhancement in the species' units.")
    sigma_log: float = Field(
        gt=0,
        description=(
            "Standard deviation of log(amplitude); 0.5 gives a mild skew, 1.5 a very heavy tail."
        ),
    )


class UniformAmplitude(_StrictModel):
    """Flat amplitude range, for controlled detection-sensitivity sweeps.

    Deliberately unrealistic: an even spread of amplitudes across a known
    range is what you want when measuring "at what enhancement does detection
    start to fail?", because it puts equal numbers of events at every
    signal-to-noise level.
    """

    kind: Literal["uniform"] = "uniform"
    low: float = Field(gt=0, description="Minimum peak enhancement.")
    high: float = Field(gt=0, description="Maximum peak enhancement.")

    @model_validator(mode="after")
    def _ordered(self) -> UniformAmplitude:
        if self.low >= self.high:
            raise ValueError(f"UniformAmplitude.low ({self.low}) must be < high ({self.high}).")
        return self


AmplitudeSpec = Annotated[LognormalAmplitude | UniformAmplitude, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class RatioSpec(_StrictModel):
    """The true enhancement ratio of one species to the source's reference.

    The single most important number in the package: everything TSARA
    computes is ultimately an estimate of this. It is specified as a
    *distribution*, not a constant, because real emission ratios vary between
    encounters of the same source type (combustion conditions, mixing,
    which piece of equipment is venting).

    ``relative_spread`` = 0 collapses to a fixed ratio — the textbook case
    for asking "does the estimator find the right slope at all?" — while a
    non-zero spread creates genuine between-event scatter, which is what
    Phase 7's methodological-variance and error-budget-closure diagnostics
    (METHODS.md §5.1) are built to characterize. Without it, the Birge ratio
    has nothing to detect.
    """

    mean: float = Field(gt=0, description="Population mean of the true ratio (delta_s/delta_ref).")
    relative_spread: float = Field(
        default=0.0,
        ge=0,
        lt=2.0,
        description=(
            "Event-to-event standard deviation of the ratio, as a fraction of "
            "'mean'. 0.15 means 15 % scatter. Draws are lognormal, "
            "parameterized so the arithmetic mean is exactly 'mean' and the "
            "relative standard deviation is exactly this value."
        ),
    )

    def lognormal_parameters(self) -> tuple[float, float]:
        r"""Return ``(mu, sigma_log)`` of the underlying normal distribution.

        For a lognormal with median ``exp(mu)``, requiring an arithmetic mean
        of ``m`` and a relative standard deviation of ``s`` gives

        .. math::

            \sigma_{\log} = \sqrt{\ln(1 + s^2)}, \qquad
            \mu = \ln m - \tfrac{1}{2}\sigma_{\log}^2

        so the configured ``mean`` really is the mean (not the median), which
        is what makes "did the estimator recover the true ratio?" a
        well-posed question with an unambiguous target.

        Returns
        -------
        tuple of float
            ``(mu, sigma_log)``; ``sigma_log`` is 0 for a fixed ratio.
        """
        if self.relative_spread == 0.0:
            return math.log(self.mean), 0.0
        sigma_log = math.sqrt(math.log(1.0 + self.relative_spread**2))
        return math.log(self.mean) - 0.5 * sigma_log**2, sigma_log


class NestedSpec(_StrictModel):
    """A short, sharp child plume riding on top of a broader parent plume.

    The multi-scale case from CLAUDE.md: a natural-gas "blip" superimposed on
    a broad landfill plume. The child is a *distinct physical source* that
    happens to be encountered inside the parent, which is exactly why nesting
    matters scientifically — a regression that lumps child and parent samples
    together measures neither source's ratio.

    Setting ``ratios`` is what expresses that distinctness, and it may name
    species the parent never emits: the landfill above has no ethane, while
    the thermogenic blip inside it does. Leaving ``ratios`` as ``None``
    inherits the parent's chemistry instead, describing finer temporal
    structure within one source rather than a second source.

    Phase 6 records the parent-child link in the catalog; the area
    mathematics that would separate their masses is deferred (METHODS.md §7).
    """

    probability: float = Field(
        ge=0, le=1, description="Chance that any given parent event carries a nested child."
    )
    shape: PlumeShape = Field(
        description="Child shape; should be substantially narrower than the parent's."
    )
    amplitude_factor: float = Field(
        gt=0,
        description="Child peak amplitude as a multiple of its parent's peak amplitude.",
    )
    ratios: dict[str, RatioSpec] | None = Field(
        default=None,
        description=(
            "Child's own species ratios, relative to the parent's reference "
            "species. None inherits the parent's ratios (same source, finer "
            "structure). Setting them makes the child a genuinely different "
            "source — the harder and more realistic case — and may name "
            "species the parent does not emit; species left unmentioned still "
            "inherit the parent's ratio."
        ),
    )


class SourceSpec(_StrictModel):
    """One family of correlated, multi-species plume events.

    The source's name is its key in ``SyntheticConfig.sources``. Events arrive
    as a homogeneous Poisson process at ``rate_per_hour``; each event draws a
    reference-species amplitude from ``amplitude``, then every other
    participating species gets ``amplitude x ratio`` with its ratio drawn
    from that species' :class:`RatioSpec`.

    Turning ``rate_per_hour`` up until event supports routinely overlap is
    how the roadmap's required "plume-dense stretches" test case is produced:
    overlapping plumes are simultaneously the realistic condition near
    sources and the condition under which signal-MAD noise estimation and
    naive baselines both fail.
    """

    rate_per_hour: float = Field(
        gt=0, description="Mean event onsets per hour (Poisson process intensity)."
    )
    shape: PlumeShape = Field(description="Temporal shape shared by all species in the event.")
    reference_species: str = Field(
        min_length=1,
        description=(
            "Canonical name of the species whose amplitude is drawn directly; "
            "every ratio in 'ratios' is relative to this species."
        ),
    )
    amplitude: AmplitudeSpec = Field(
        description="Distribution of the reference species' peak enhancement."
    )
    ratios: dict[str, RatioSpec] = Field(
        default_factory=dict,
        description=(
            "Other participating species -> their true ratio to the reference "
            "species. Empty means a single-species source."
        ),
    )
    inter_species_lag: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional signed per-species arrival offsets, e.g. {'c2h6': "
            "'30s'}. Reproduces co-located sources whose plumes reach the "
            "inlet at different times (separate stacks at one refinery) — the "
            "condition the Phase 8 smoothing stage exists to repair. Negative "
            "values mean the species arrives *before* the reference."
        ),
    )
    nested: NestedSpec | None = Field(
        default=None, description="Optional nested child-plume structure."
    )

    @field_validator("inter_species_lag")
    @classmethod
    def _valid_lags(cls, value: dict[str, str]) -> dict[str, str]:
        """Lags are *signed* offsets, so they use the signed validator.

        This is the field whose arrival ``validate_positive_timedelta``'s
        docstring anticipated when it warned that signed time quantities are
        "a different physical animal".
        """
        for species, lag in value.items():
            _validate_signed(lag, field=f"SourceSpec.inter_species_lag['{species}']")
        return value

    @model_validator(mode="after")
    def _references_are_consistent(self) -> SourceSpec:
        """Cross-check the species names this source talks about.

        Catches the two mistakes that would otherwise fail silently: giving
        the reference species a ratio to itself (which would double-count it),
        and lagging a species the source does not actually emit (a typo that
        would simply do nothing).

        Note what is deliberately *not* checked here: a nested child may name
        species the parent does not emit. That is the CLAUDE.md multi-scale
        case — a sharp thermogenic blip (with ethane) encountered inside a
        broad landfill plume (without) — and forbidding it would make the
        package's own motivating example inexpressible. Those names are
        instead validated against the declared gas species campaign-wide by
        ``SyntheticConfig._sources_reference_declared_gas_species``, which is
        the check that actually catches typos.
        """
        if self.reference_species in self.ratios:
            raise ValueError(
                f"SourceSpec.ratios must not contain the reference species "
                f"'{self.reference_species}'; its ratio to itself is 1 by definition."
            )
        participating = {self.reference_species} | set(self.ratios)
        unknown_lag = set(self.inter_species_lag) - participating
        if unknown_lag:
            raise ValueError(
                f"SourceSpec.inter_species_lag names species {sorted(unknown_lag)} "
                f"that this source does not emit; participating species are "
                f"{sorted(participating)}."
            )
        if (
            self.nested is not None
            and self.nested.ratios is not None
            and self.reference_species in self.nested.ratios
        ):
            # Same rule, same reason as the parent's own ratios above: the
            # child's amplitudes are built as reference_amplitude x ratio, so
            # a ratio of the reference to itself would overwrite the implicit
            # 1.0 and double-count the reference species.
            raise ValueError(
                f"SourceSpec.nested.ratios must not contain the reference species "
                f"'{self.reference_species}'; its ratio to itself is 1 by definition."
            )
        return self


# ---------------------------------------------------------------------------
# Platforms (tagged union on 'kind')
# ---------------------------------------------------------------------------


class StationarySite(_StrictModel):
    """A fixed site: one static lat/lon attached to every stream as a scalar."""

    kind: Literal["stationary"] = "stationary"
    latitude: float = Field(ge=-90, le=90, description="Site latitude, decimal degrees N.")
    longitude: float = Field(ge=-180, le=180, description="Site longitude, decimal degrees E.")
    altitude_m: float | None = Field(default=None, description="Optional site elevation, meters.")


class MobileTrack(_StrictModel):
    """A moving platform: GPS emitted as its own stream at its own rate.

    The GPS stream is deliberately a *separate instrument* with a separate
    ``gps_rate``, because that is the canonical multi-rate alignment problem
    (METHODS.md §1.2): position is a smooth auxiliary field that may be
    interpolated onto gas timestamps, while the gases themselves may not be.
    A generator that put GPS on the gas clock would make that distinction
    untestable.

    Two track patterns are offered, both cheap: ``random_walk`` (a
    correlated wander, realistic for an unstructured survey drive) and
    ``circuit`` (a closed circle, which returns to the same coordinates
    repeatedly and therefore produces genuinely clusterable revisits for the
    Phase 8 source-complex stage).

    Known limitation (METHODS.md §8.5): plume *timing* is not derived from
    the track geometry — there is no dispersion model placing sources in
    space and computing when the vehicle drives through them. Event
    coordinates in the ground truth are the platform position at each event's
    peak time, which is what a real mobile catalog records anyway.
    """

    kind: Literal["mobile"] = "mobile"
    gps_instrument: str = Field(
        default="gps", description="Name of the emitted GPS stream in the streams mapping."
    )
    gps_rate: str = Field(default="1s", description="GPS sampling interval.")
    start_latitude: float = Field(ge=-90, le=90, description="Track start latitude.")
    start_longitude: float = Field(ge=-180, le=180, description="Track start longitude.")
    speed_m_s: float = Field(default=10.0, gt=0, description="Platform ground speed, m/s.")
    pattern: Literal["random_walk", "circuit"] = Field(
        default="random_walk", description="Track geometry."
    )
    radius_m: float = Field(
        default=1000.0, gt=0, description="Circuit radius in meters (ignored for random_walk)."
    )
    heading_volatility: float = Field(
        default=0.1,
        ge=0,
        description=(
            "Random-walk heading change in radians per second; 0 drives in a "
            "straight line, larger values wander more tightly."
        ),
    )

    @field_validator("gps_rate")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"MobileTrack.{info.field_name}")
        return value


PlatformSpec = Annotated[StationarySite | MobileTrack, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Top-level synthetic configuration
# ---------------------------------------------------------------------------


class SyntheticConfig(_StrictModel):
    """Complete specification of one synthetic dataset.

    Fully YAML-round-trippable by construction (no embedded arrays), so the
    exact configuration that produced a dataset is saved inside its bundle
    and a run is reproducible from the bundle alone given the same
    ``seed``.
    """

    name: str = Field(min_length=1, description="Short identifier, e.g. 'synth_basic'.")
    description: str = Field(default="", description="Free-text note for humans.")
    start: datetime = Field(description="First timestamp (UTC).")
    duration: str = Field(description="Total span, e.g. '24h'.")
    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "Seed for the single NumPy Generator threaded through the whole "
            "build. A given config plus seed always yields byte-identical "
            "output; changing the config anywhere may change all draws."
        ),
    )
    platform: PlatformSpec = Field(description="Stationary site or mobile track.")
    instruments: dict[str, InstrumentSpec] = Field(
        min_length=1, description="Mapping of instrument name -> spec."
    )
    sources: dict[str, SourceSpec] = Field(
        default_factory=dict,
        description=(
            "Mapping of source name -> spec. Empty produces a plume-free "
            "dataset, which is the right control case for measuring an "
            "algorithm's false-positive rate."
        ),
    )

    @field_validator("duration")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"SyntheticConfig.{info.field_name}")
        return value

    @model_validator(mode="after")
    def _species_unique_across_instruments(self) -> SyntheticConfig:
        """Canonical species names must be unique campaign-wide.

        Same rule (and same reason) as ``Manifest``: two instruments both
        producing 'ch4' would collide when the streams are combined, and a
        source naming 'ch4' would be ambiguous about which instrument it
        enhances.
        """
        owner: dict[str, str] = {}
        for inst_name, inst in self.instruments.items():
            for species in inst.species:
                if species in owner:
                    raise ValueError(
                        f"Species '{species}' is declared by both '{owner[species]}' "
                        f"and '{inst_name}'; names must be unique across instruments."
                    )
                owner[species] = inst_name
        return self

    @model_validator(mode="after")
    def _sources_reference_declared_gas_species(self) -> SyntheticConfig:
        """Every species a source emits must exist and be a gas.

        Fails fast on the two config errors that would otherwise produce a
        silently plume-free dataset: a typo'd species name, and pointing a
        source at a met/aux variable that the generator will never enhance.
        """
        gas_species = {
            name
            for inst in self.instruments.values()
            for name, spec in inst.species.items()
            if spec.role == "gas"
        }
        all_species = {name for inst in self.instruments.values() for name in inst.species}

        for source_name, source in self.sources.items():
            emitted = {source.reference_species} | set(source.ratios)
            # A nested child may introduce species of its own (see
            # SourceSpec._references_are_consistent), so its ratios are folded
            # in here: this is the single place every species name a source
            # can possibly emit is checked against what the instruments
            # actually declare.
            if source.nested is not None and source.nested.ratios is not None:
                emitted |= set(source.nested.ratios)
            for species in sorted(emitted):
                if species not in all_species:
                    raise ValueError(
                        f"Source '{source_name}' emits undeclared species '{species}'; "
                        f"declared species are {sorted(all_species)}."
                    )
                if species not in gas_species:
                    raise ValueError(
                        f"Source '{source_name}' emits '{species}', which is not a "
                        "role='gas' variable; only gases receive plumes."
                    )
        return self

    @model_validator(mode="after")
    def _gps_instrument_name_is_free(self) -> SyntheticConfig:
        """Reject a mobile GPS stream name that collides with a real instrument.

        The GPS stream is synthesized by the platform, not declared in
        ``instruments``; if both existed under one name the platform would
        silently overwrite the declared instrument.
        """
        if isinstance(self.platform, MobileTrack) and self.platform.gps_instrument in (
            self.instruments
        ):
            raise ValueError(
                f"platform.gps_instrument '{self.platform.gps_instrument}' collides "
                "with a declared instrument name; the GPS stream is generated "
                "separately and needs its own name."
            )
        return self

    @property
    def gas_species(self) -> tuple[str, ...]:
        """Canonical names of all role='gas' species, across instruments."""
        return tuple(
            name
            for inst in self.instruments.values()
            for name, spec in inst.species.items()
            if spec.role == "gas"
        )

    def instrument_of(self, species: str) -> str:
        """Return the instrument name that owns ``species``.

        Parameters
        ----------
        species : str
            Canonical species name.

        Returns
        -------
        str
            Owning instrument name.

        Raises
        ------
        KeyError
            If no instrument declares that species.
        """
        for inst_name, inst in self.instruments.items():
            if species in inst.species:
                return inst_name
        raise KeyError(f"No instrument declares species '{species}'.")
