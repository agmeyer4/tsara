"""Analysis configuration schema: *what to do with the ingested data*.

Where the manifest (:mod:`tsara.config.manifest`) describes the raw data,
this module describes the science: the master time grid, the baseline
parameter sweep, plume detection thresholds, smoothing, source-complex
clustering, and regression/UQ settings.

The sweep philosophy
--------------------
Several fields here are deliberately *lists* (baseline ``windows``,
``quantiles``, smoothing ``cutoffs``, detection ``enter_sigma``). Each list
becomes a named dimension of the parameter hypercube: the engine evaluates
every combination, and the spread of results across the cube *is* the
methodological uncertainty reported in Phase 7. A user who wants a single
fixed methodology simply supplies one-element lists — the hypercube then
collapses to a point and methodological variance is zero by construction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from tsara.config.base import StrictModel as _StrictModel
from tsara.config.base import validate_positive_timedelta as _validate_duration

# ---------------------------------------------------------------------------
# Output grid (continuous rolling state + PMF export matrix ONLY)
# ---------------------------------------------------------------------------


class OutputGridConfig(_StrictModel):
    """Definition of the uniform grid built for the continuous state + PMF matrix.

    REVISED 2026-07-09 ("synchronize late", see METHODS.md §1.1/§1.4): this
    grid is *not* what streams are synchronized onto for baselines,
    detection, or regression — those all run at each stream's own native
    rate. This grid exists only at the output boundary, for the two products
    that inherently need a common `(time x species)` cube: the continuous
    rolling state and the PMF export matrix.

    Construction is binning-only in both directions: gas species are *never*
    interpolated (a concentration inside a plume is not a smooth field), so
    a cell with zero native samples in its interval is NaN with `n_native =
    0` — never papered over with a straight line. Aux-field interpolation
    (GPS, met) is a separate concern, see :class:`AlignmentConfig`.
    """

    freq: str = Field(
        description="Grid spacing as a pandas offset alias, e.g. '1s', '5s', '1min'."
    )
    start: datetime | None = Field(
        default=None,
        description="Optional grid start (UTC). Default: first timestamp across streams.",
    )
    end: datetime | None = Field(
        default=None,
        description="Optional grid end (UTC). Default: last timestamp across streams.",
    )
    bin_statistic: Literal["mean", "median"] = Field(
        default="mean",
        description=(
            "Statistic for binning native samples into each grid cell. "
            "'median' is more robust to sub-grid spikes but slightly biases "
            "sharp plume peaks low (see METHODS.md §3.5 for the median "
            "standard-error inflation factor this implies)."
        ),
    )

    @field_validator("freq")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"OutputGridConfig.{info.field_name}")
        return value

    @model_validator(mode="after")
    def _start_before_end(self) -> OutputGridConfig:
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError(
                f"OutputGridConfig.start ({self.start}) must be before end ({self.end})."
            )
        return self


# ---------------------------------------------------------------------------
# Auxiliary-field alignment (GPS/met interpolation guard — NOT gas pairing)
# ---------------------------------------------------------------------------


class AlignmentConfig(_StrictModel):
    """Guard on interpolating smooth auxiliary fields onto gas timestamps.

    Per METHODS.md §1.2: quantified species (gases) are never interpolated —
    only bin-averaged — but smooth auxiliary fields (GPS position, met
    variables) vary slowly enough that interpolating them onto gas
    timestamps is physically justified, *as long as* the gap being bridged
    isn't so long that the interpolation would be inventing data across a
    real instrument dropout. This is unrelated to cross-species pairing for
    regression (METHODS.md §1.3), which has no free parameters of its own —
    it always uses the slower-of-the-pair's native clock.
    """

    max_interp_gap: str = Field(
        default="10s",
        description=(
            "Longest data gap that interpolation may bridge when aligning "
            "auxiliary fields (GPS, met) onto gas timestamps. Gaps longer "
            "than this remain NaN rather than being bridged."
        ),
    )

    @field_validator("max_interp_gap")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"AlignmentConfig.{info.field_name}")
        return value


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class BaselineConfig(_StrictModel):
    """Rolling low-quantile baseline estimation (the background signal).

    The baseline at each instant is a low quantile (e.g. the 5th percentile)
    of the signal within a centered rolling window. Low quantiles track the
    background *underneath* plumes because plumes only ever add mass —
    enhancements are one-sided — so the lower tail of a window is dominated
    by background air.

    ``windows`` and ``quantiles`` are sweep dimensions: every (window,
    quantile) pair is evaluated. Windows double as the *multi-scale
    hierarchy* used for nested-plume parent/child bookkeeping in Phase 6 —
    a sharp blip is an enhancement over the shortest window's baseline, a
    broad plume over the longest.
    """

    windows: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Rolling window lengths (timedelta strings), e.g. ['2min', '10min', "
            "'60min']. Each becomes a point along the 'baseline_window' sweep "
            "dimension, ordered short -> long for the plume hierarchy."
        ),
    )
    quantiles: tuple[float, ...] = Field(
        min_length=1,
        description=(
            "Quantiles in (0, 0.5], e.g. [0.01, 0.05, 0.10]. Each becomes a "
            "point along the 'baseline_quantile' sweep dimension."
        ),
    )
    min_valid_fraction: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description=(
            "Minimum fraction of non-NaN samples a window must contain for "
            "its baseline to be reported; sparser windows yield NaN rather "
            "than a baseline built on a handful of points."
        ),
    )

    @field_validator("windows")
    @classmethod
    def _valid_sorted_unique_windows(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Validate each window and require strictly increasing durations.

        Ordering is enforced (rather than silently sorted) because window
        order defines the micro->macro plume hierarchy; a manifest that
        lists them shuffled probably contains a typo'd unit.
        """
        import pandas as pd

        for w in value:
            _validate_duration(w, field="BaselineConfig.windows")
        tds = [pd.Timedelta(w) for w in value]
        if any(b <= a for a, b in zip(tds, tds[1:])):
            raise ValueError(
                f"BaselineConfig.windows must be strictly increasing (short -> long); "
                f"got {list(value)}."
            )
        return value

    @field_validator("quantiles")
    @classmethod
    def _valid_quantiles(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        for q in value:
            # Above the median, a "baseline" would sit inside the plumes
            # themselves — scientifically meaningless as a background.
            if not (0.0 < q <= 0.5):
                raise ValueError(
                    f"Baseline quantiles must be in (0, 0.5]; got {q}. "
                    "A background estimate above the median is not a background."
                )
        if len(set(value)) != len(value):
            raise ValueError("BaselineConfig.quantiles contains duplicates.")
        return value


# ---------------------------------------------------------------------------
# Plume detection
# ---------------------------------------------------------------------------


#: Registered empirical noise estimators (METHODS.md §2.5). Used only as the
#: *fallback* noise scale for a species with no declared/reported random
#: uncertainty (METHODS.md §2.3/§6) — the provenance order itself (declared
#: > reported > empirical) is automatic, not a user choice; this field only
#: picks the algorithm backing the empirical rung of that ladder.
NoiseEstimator = Literal["diff_mad", "mad"]


class DetectionConfig(_StrictModel):
    """Threshold + hysteresis segmentation of enhancements into plume events.

    A plume *starts* when the enhancement exceeds ``enter_sigma`` × the
    noise-sigma scale and *ends* only when it falls back below
    ``exit_sigma`` × noise. The two-threshold (hysteresis) design stops a
    plume that hovers near a single threshold from being chopped into dozens
    of fragments by noise crossings.

    The noise scale itself is *not* always the empirical estimator below: per
    METHODS.md §2.3/§6, it comes from the measurement-uncertainty system in
    provenance order (declared or reported ``sigma_rand`` when the species
    has one, else the empirical estimate) — thresholds are expressed in
    noise-sigma units precisely so they mean the same thing regardless of
    which rung of that ladder supplied sigma for a given species.
    """

    enter_sigma: tuple[float, ...] = Field(
        default=(3.0,),
        min_length=1,
        description=(
            "Entry thresholds in noise-sigma units; a sweep dimension "
            "('detection_enter_sigma') when more than one value is given."
        ),
    )
    exit_sigma: float = Field(
        default=1.0, gt=0, description="Exit threshold in noise-sigma units."
    )
    noise_estimator: NoiseEstimator = Field(
        default="diff_mad",
        description=(
            "Registered empirical noise estimator (METHODS.md §2.5), used "
            "only when a species has no declared/reported random "
            "uncertainty. 'diff_mad' (default) is the plume-immune robust "
            "first-difference estimator; 'mad' (rolling MAD of the signal "
            "itself) is kept for comparison but breaks down in plume-dense "
            "records."
        ),
    )
    noise_window: str = Field(
        default="10min",
        description=(
            "Rolling window for the empirical noise estimate of the "
            "enhancement signal (only used when noise_estimator applies)."
        ),
    )
    min_duration: str = Field(
        default="3s",
        description="Events shorter than this are discarded as noise blips.",
    )
    max_internal_gap: str = Field(
        default="5s",
        description=(
            "Sub-threshold dips shorter than this are bridged, keeping one "
            "physical plume from splitting into several events."
        ),
    )

    @field_validator("noise_window", "min_duration", "max_internal_gap")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"DetectionConfig.{info.field_name}")
        return value

    @model_validator(mode="after")
    def _hysteresis_ordering(self) -> DetectionConfig:
        """Every entry threshold must sit above the exit threshold.

        enter <= exit would invert the hysteresis and make event boundaries
        ill-defined.
        """
        bad = [e for e in self.enter_sigma if e <= self.exit_sigma]
        if bad:
            raise ValueError(
                f"DetectionConfig.enter_sigma values {bad} must all exceed "
                f"exit_sigma ({self.exit_sigma})."
            )
        return self


# ---------------------------------------------------------------------------
# Smoothing (optional stage)
# ---------------------------------------------------------------------------


class SmoothingConfig(_StrictModel):
    """Zero-phase low-pass filtering to align temporally disjoint plumes.

    Different species from one facility (e.g. separate refinery stacks) can
    arrive at the sensor minutes apart; low-pass filtering broadens both
    until they covary on the source scale, at the cost of temporal
    resolution. Cutoffs are a sweep dimension so that trade-off is
    quantified, not guessed. Filtering is zero-phase (forward-backward) so
    plume *timing* is never shifted — a phase lag would corrupt every
    ratio regression downstream.
    """

    enabled: bool = Field(default=False, description="Master switch for the smoothing stage.")
    cutoff_periods: tuple[str, ...] = Field(
        default=("60s",),
        min_length=1,
        description=(
            "Low-pass cutoff periods (timedelta strings); fluctuations faster "
            "than each cutoff are attenuated. Sweep dimension "
            "('smoothing_cutoff') when more than one value is given."
        ),
    )
    order: int = Field(
        default=4, ge=1, le=10, description="Butterworth filter order (steepness of rolloff)."
    )

    @field_validator("cutoff_periods")
    @classmethod
    def _valid_cutoffs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for c in value:
            _validate_duration(c, field="SmoothingConfig.cutoff_periods")
        return value


# ---------------------------------------------------------------------------
# Source-complex clustering (optional stage)
# ---------------------------------------------------------------------------


class ClusteringConfig(_StrictModel):
    """DBSCAN clustering of plume events into unified Source Complexes.

    Events are clustered in a scaled space-time metric: horizontal distance
    in meters plus time difference converted to meters via
    ``space_time_scale``. DBSCAN is chosen because the number of sources is
    unknown a priori and isolated events legitimately remain unclustered
    ("noise" in DBSCAN terms = a lone source encounter, not an error).
    """

    enabled: bool = Field(default=False, description="Master switch for clustering.")
    eps_m: float = Field(
        default=500.0,
        gt=0,
        description="DBSCAN neighborhood radius in scaled meters.",
    )
    space_time_scale: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Meters-per-second equivalence for the time axis. E.g. 2.0 means "
            "two events 100 s apart are 'as far apart' as two events 200 m "
            "apart — physically, an advection-speed scale."
        ),
    )
    min_samples: int = Field(
        default=2, ge=1, description="Minimum events to form a Source Complex core."
    )


# ---------------------------------------------------------------------------
# Regression & UQ
# ---------------------------------------------------------------------------


class RegressionConfig(_StrictModel):
    """Enhancement-ratio regression settings.

    Ratios are computed as species-vs-``reference_species`` slopes within
    each plume event (discrete mode) and over rolling windows (continuous
    mode). ``york`` (METHODS.md §4.2) is the scientifically preferred
    estimator: both axes carry measurement error — plain OLS attenuates
    slopes toward zero (regression dilution) when x is noisy — and, unlike
    ``scipy.odr``, York natively accepts per-point x-y error correlation,
    the expected case when numerator and denominator come from the same
    instrument. ``odr`` is retained as a numerical cross-check (they agree
    when all per-point correlations are zero) and ``ols`` for its cheap,
    familiar diagnostics.
    """

    reference_species: str = Field(
        min_length=1,
        description=(
            "Canonical name of the denominator species (e.g. 'ch4' or 'co2'). "
            "Must be a role='gas' variable in the manifest; cross-checked when "
            "manifest and analysis configs are combined."
        ),
    )
    methods: tuple[Literal["ols", "york", "odr"], ...] = Field(
        default=("ols", "york"),
        min_length=1,
        description=(
            "Regression estimators to run. 'york' is the default preferred "
            "errors-in-both-axes estimator (METHODS.md §4.2); 'odr' is an "
            "opt-in cross-check, not part of the default set."
        ),
    )
    min_points: int = Field(
        default=8,
        ge=3,
        description=(
            "Minimum synchronized samples within an event for a regression; "
            "events with fewer get NaN ratios flagged in the catalog."
        ),
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Confidence level for reported ratio intervals.",
    )

    @field_validator("methods")
    @classmethod
    def _no_duplicate_methods(
        cls, value: tuple[Literal["ols", "york", "odr"], ...]
    ) -> tuple[Literal["ols", "york", "odr"], ...]:
        if len(set(value)) != len(value):
            raise ValueError("RegressionConfig.methods contains duplicates.")
        return value


# ---------------------------------------------------------------------------
# Top-level analysis config
# ---------------------------------------------------------------------------


class AnalysisConfig(_StrictModel):
    """Top-level science configuration for one TSARA run.

    Optional stages (smoothing, clustering) default to disabled-but-present
    so that ``analysis.smoothing.enabled`` is always a safe attribute access
    — downstream code never needs None checks for whole stages.
    """

    output_grid: OutputGridConfig = Field(
        description="Uniform grid for the continuous state + PMF matrix only (METHODS.md §1.4)."
    )
    alignment: AlignmentConfig = Field(
        default_factory=AlignmentConfig,
        description="Auxiliary-field (GPS/met) interpolation guard (METHODS.md §1.2).",
    )
    baseline: BaselineConfig = Field(description="Rolling baseline sweep settings.")
    detection: DetectionConfig = Field(
        default_factory=DetectionConfig, description="Plume event segmentation settings."
    )
    smoothing: SmoothingConfig = Field(
        default_factory=SmoothingConfig, description="Optional low-pass alignment stage."
    )
    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig, description="Optional Source Complex clustering."
    )
    regression: RegressionConfig = Field(description="Enhancement-ratio regression settings.")

    @model_validator(mode="after")
    def _grid_finer_than_shortest_window(self) -> AnalysisConfig:
        """Require the output grid to resolve the shortest baseline window.

        A 5-minute grid with a 2-minute baseline window means windows hold
        zero or one samples — the quantile degenerates to the identity and
        the 'baseline' is just the signal. Require at least ~10 grid cells
        per shortest window so the quantile has something to chew on. Note
        this checks the *output* grid, not the native-rate streams the
        baseline itself actually rolls over — it is a sanity bound on the
        continuous-state/PMF product's resolution relative to the shortest
        swept window, not a synchronization requirement (METHODS.md §1.1).
        """
        import pandas as pd

        grid_dt = pd.Timedelta(self.output_grid.freq)
        shortest = pd.Timedelta(self.baseline.windows[0])
        if shortest < 10 * grid_dt:
            raise ValueError(
                f"Shortest baseline window ({self.baseline.windows[0]}) must be at "
                f"least 10x the output grid spacing ({self.output_grid.freq}); a "
                "rolling quantile over fewer samples is statistically meaningless."
            )
        return self
