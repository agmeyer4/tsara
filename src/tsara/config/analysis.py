"""Analysis configuration schema: *what to do with the ingested data*.

Where the manifest (:mod:`tsara.config.manifest`) describes the raw data,
this module describes the science: the master time grid, the baseline
parameter sweep, plume detection thresholds, smoothing, source-complex
clustering, and regression/UQ settings.

The sweep philosophy
--------------------
Several fields here are deliberately *lists* (baseline ``windows``,
``quantiles``, smoothing ``cutoffs``, detection ``enter_mads``). Each list
becomes a named dimension of the parameter hypercube: the engine evaluates
every combination, and the spread of results across the cube *is* the
methodological uncertainty reported in Phase 7. A user who wants a single
fixed methodology simply supplies one-element lists — the hypercube then
collapses to a point and methodological variance is zero by construction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tsara.config.base import StrictModel as _StrictModel
from tsara.config.base import validate_timedelta as _parse_timedelta

# ---------------------------------------------------------------------------
# Master time grid
# ---------------------------------------------------------------------------


class GridConfig(_StrictModel):
    """Definition of the master time grid all streams are synchronized onto.

    Streams *faster* than the grid are bin-averaged into grid cells (with
    per-bin std and count retained — they feed ODR weighting later);
    streams *slower* than the grid are linearly interpolated, but never
    across gaps longer than ``max_interp_gap``. That guard is what prevents
    an hour-long instrument dropout from being papered over with a straight
    line that plume detection would happily "discover".
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
    max_interp_gap: str = Field(
        default="10s",
        description=(
            "Longest data gap that interpolation may bridge when upsampling "
            "slow streams. Gaps longer than this remain NaN."
        ),
    )
    bin_statistic: Literal["mean", "median"] = Field(
        default="mean",
        description=(
            "Statistic for downsampling fast streams. 'median' is more robust "
            "to sub-grid spikes but slightly biases sharp plume peaks low."
        ),
    )

    @field_validator("freq", "max_interp_gap")
    @classmethod
    def _valid_durations(cls, value: str, info) -> str:
        _parse_timedelta(value, field=f"GridConfig.{info.field_name}")
        return value

    @model_validator(mode="after")
    def _start_before_end(self) -> GridConfig:
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError(f"GridConfig.start ({self.start}) must be before end ({self.end}).")
        return self


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
    hierarchy* used for nested-plume attribution in Phase 6 — a sharp blip
    is an enhancement over the shortest window's baseline, a broad plume
    over the longest.
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
            _parse_timedelta(w, field="BaselineConfig.windows")
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


class DetectionConfig(_StrictModel):
    """Threshold + hysteresis segmentation of enhancements into plume events.

    A plume *starts* when the enhancement exceeds ``enter_mad`` × the
    rolling MAD-based noise estimate and *ends* only when it falls back
    below ``exit_mad`` × noise. The two-threshold (hysteresis) design stops
    a plume that hovers near a single threshold from being chopped into
    dozens of fragments by noise crossings.
    """

    enter_mads: tuple[float, ...] = Field(
        default=(3.0,),
        min_length=1,
        description=(
            "Entry thresholds in noise-MAD units; a sweep dimension "
            "('detection_enter_mad') when more than one value is given."
        ),
    )
    exit_mad: float = Field(
        default=1.0, gt=0, description="Exit threshold in noise-MAD units."
    )
    noise_window: str = Field(
        default="10min",
        description="Rolling window for the MAD noise estimate of the enhancement signal.",
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
    def _valid_durations(cls, value: str, info) -> str:
        _parse_timedelta(value, field=f"DetectionConfig.{info.field_name}")
        return value

    @model_validator(mode="after")
    def _hysteresis_ordering(self) -> DetectionConfig:
        """Every entry threshold must sit above the exit threshold.

        enter <= exit would invert the hysteresis and make event boundaries
        ill-defined.
        """
        bad = [e for e in self.enter_mads if e <= self.exit_mad]
        if bad:
            raise ValueError(
                f"DetectionConfig.enter_mads values {bad} must all exceed "
                f"exit_mad ({self.exit_mad})."
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
            _parse_timedelta(c, field="SmoothingConfig.cutoff_periods")
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
    mode). ODR is the scientifically preferred estimator here because both
    axes carry measurement error — plain OLS attenuates slopes toward zero
    (regression dilution) when x is noisy — but OLS is retained for
    comparison and for its cheap, familiar diagnostics.
    """

    reference_species: str = Field(
        min_length=1,
        description=(
            "Canonical name of the denominator species (e.g. 'ch4' or 'co2'). "
            "Must be a role='gas' variable in the manifest; cross-checked when "
            "manifest and analysis configs are combined."
        ),
    )
    methods: tuple[Literal["ols", "odr"], ...] = Field(
        default=("ols", "odr"),
        min_length=1,
        description="Regression estimators to run.",
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
        cls, value: tuple[Literal["ols", "odr"], ...]
    ) -> tuple[Literal["ols", "odr"], ...]:
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

    grid: GridConfig = Field(description="Master time grid definition.")
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
        """The grid must resolve the shortest baseline window.

        A 5-minute grid with a 2-minute baseline window means windows hold
        zero or one samples — the quantile degenerates to the identity and
        the 'baseline' is just the signal. Require at least ~10 grid cells
        per shortest window so the quantile has something to chew on.
        """
        import pandas as pd

        grid_dt = pd.Timedelta(self.grid.freq)
        shortest = pd.Timedelta(self.baseline.windows[0])
        if shortest < 10 * grid_dt:
            raise ValueError(
                f"Shortest baseline window ({self.baseline.windows[0]}) must be at "
                f"least 10x the grid spacing ({self.grid.freq}); a rolling quantile "
                "over fewer samples is statistically meaningless."
            )
        return self
