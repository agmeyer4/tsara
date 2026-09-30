"""Analysis configuration schema: *what to do with the ingested data*.

Where the manifest (:mod:`tsara.config.manifest`) describes the raw data,
this module describes the science: the baseline parameter sweep and the
method each variable's baseline is built with, how plume events are found
(thresholds, records, triggers), smoothing, source-complex clustering,
regression/UQ settings, and an optional uniform output grid for the run that
wants to export a rectangular table.

The sweep philosophy
--------------------
Several fields here are deliberately *lists* (baseline ``windows``,
``quantiles``, plumes ``enter_multiple`` and ``exit_multiple``, smoothing
``cutoff_periods``). Each list
becomes a named dimension of the parameter hypercube: the engine evaluates
every combination, and the spread of results across the cube *is* the
methodological uncertainty reported in Phase 7. A user who wants a single
fixed methodology simply supplies one-element lists — the hypercube then
collapses to a point and methodological variance is zero by construction.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, ValidationInfo, field_validator, model_validator

from tsara.config.base import StrictModel as _StrictModel
from tsara.config.base import validate_positive_timedelta as _validate_duration

# ---------------------------------------------------------------------------
# Output grid (optional: a uniform tiling, one kind of cell set, for an export)
# ---------------------------------------------------------------------------


class OutputGridConfig(_StrictModel):
    """A uniform tiling of cells: one kind of cell set, built when an export wants one.

    A grid is not where any science happens. Baselines, detection and
    regression run per stream at native rate ("synchronize late", METHODS.md
    §1.1), and the continuous baseline state -- baseline and enhancement, with
    their uncertainties, for every reading at every sweep point -- lives beside each
    stream's own readings (§6.2), not on this grid: measured, native rate is
    thirteen times smaller than a grid spanning a campaign and loses nothing.
    What a tiling is for is a rectangular table at the end, a receptor
    model's input matrix being the usual case, and it is one cell set among
    the others a join may be asked to fill (§1.4). It is therefore optional
    in :class:`AnalysisConfig`: a run that never exports a table declares
    none, and a run that does chooses the period for that export.

    Construction is binning-only in both directions: gas species are *never*
    interpolated (a concentration inside a plume is not a smooth field), so
    a cell with zero native samples in its interval is NaN with
    `n_readings_<name> = 0` — never papered over with a straight line. The
    period is checked against the selected variables (METHODS.md §11.7): a
    reading at least twice as wide as a grid cell it touches would be copied
    across rows and is refused unless :class:`AlignmentConfig` says
    ``finer_support: allow``; everything narrower is recorded per column
    (§11.2.4). Aux-field interpolation (GPS, met) is the other knob on that
    class.

    There is deliberately no choice of binning statistic. A ``median``
    option was specified in Phase 1 and removed in Phase 4 before it ever
    had an implementation: its purpose is robustness to sub-grid spikes, and
    a sub-grid spike in this science *is the plume*. Measured on the real
    mobile-lab CH4 of the ten 2024 drive days, a 60 s median discards 19.6 %
    of the enhancement mass and up to 100 % of an individual cell's — a low
    bias that scales with how sharp each species' plumes are, so it corrupts
    exactly the between-species ratios TSARA exists to compute (METHODS.md
    §11.7.1). It is
    the same argument that deleted the QA/QC spike rule (§9.5).
    """

    freq: str = Field(description="Grid spacing as a pandas offset alias, e.g. '1s', '5s', '1min'.")
    start: datetime | None = Field(
        default=None,
        description="Optional grid start (UTC). Default: first timestamp across streams.",
    )
    end: datetime | None = Field(
        default=None,
        description="Optional grid end (UTC). Default: last timestamp across streams.",
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
    """How support may be changed: the copy policy, and the interpolation guard.

    Every join in TSARA averages readings onto target cells and records, per
    column, what that did to their support (METHODS.md §11.2.4): a reading
    wholly inside its cell is *averaged*, one lying across a boundary is
    *straddled* (part of it in each of two rows), one wider than the cell it
    fills is *narrowed*, and one at least twice as wide would be *copied*
    across rows. All but the
    last are allowed, recorded and warned about; the last is what
    ``finer_support`` decides. Interpolation is the other way a value can be
    placed on a support it was not measured on, and TSARA performs it only
    on smooth auxiliary fields (GPS position, met) onto gas cells, under
    ``max_interp_gap`` (§1.2, §11.6). Quantified species are never
    interpolated.

    Cross-species pairing (§1.3) reads neither knob by default: its clock is
    the wider-supported member's own cells, so nothing is copied there, and
    nothing is interpolated. It reads ``finer_support`` only for a clock
    whose cells vary in width (§11.2.4).
    """

    finer_support: Literal["refuse", "allow"] = Field(
        default="refuse",
        description=(
            "What a join does with a reading at least twice as wide as a "
            "target cell it touches, so that its value would be copied across "
            "rows. 'refuse' (the default) raises: sixty rows from one 60 s "
            "mean would enter a fit as sixty measurements, which is the "
            "interpolation rule for a step function (METHODS.md §1.2). "
            "'allow' makes the copies, labels every affected column 'copied', "
            "records how much of each value was borrowed from beyond its "
            "cell, and warns (§11.2.4). Nothing narrower than that line is "
            "governed here: narrowing and straddling are always allowed, always "
            "recorded, and named in the same warning."
        ),
    )
    max_interp_gap: str = Field(
        default="10s",
        description=(
            "Longest data gap that interpolation may bridge when aligning "
            "auxiliary fields (GPS, met) onto gas timestamps. Gaps longer "
            "than this remain NaN rather than being bridged. For a moving "
            "platform's position this is a statement about how far the track "
            "may stray from a straight line between fixes: on real urban "
            "driving at a median 13 m/s, 10 s costs about 10 m at the 90th "
            "percentile, 50 s about 110 m (METHODS.md §11.6). A record sampled "
            "more sparsely than this is almost entirely refused, with a warning."
        ),
    )

    @field_validator("max_interp_gap")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"AlignmentConfig.{info.field_name}")
        return value


# ---------------------------------------------------------------------------
# Cross-species pairing (the regression clock — NOT aux interpolation)
# ---------------------------------------------------------------------------


class PairingConfig(_StrictModel):
    """How much of a cell must be measured for the pair to count.

    Pairing two species measured at different rates has no free parameters
    of its own: the clock is always the cells of whichever stream has the
    *wider* support, and the other is averaged onto them weighted by overlap
    (METHODS.md §1.3). What it does have is a question of *sufficiency*.

    A canister that integrates for 15 s paired against a 60 s mean covers a
    quarter of that cell. That pair is real and it is not comparable to one
    with full coverage, and nothing in the arithmetic distinguishes them —
    both produce a number. So every paired value carries the fraction of its
    cell that contributing data actually covered, and this is the one knob
    that can act on it.

    Separate from :class:`AlignmentConfig`, which guards *interpolation* of
    smooth auxiliary fields and has nothing to do with cross-species pairing.
    """

    min_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of a pairing cell that must be covered by "
            "contributing partner data for the pair to be kept. The default "
            "of 0.0 drops nothing: coverage is always recorded alongside "
            "every pair, and how much is enough is a question about the "
            "science rather than about the arithmetic. Raise it to exclude "
            "thinly-covered pairs from regressions."
        ),
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class RollingQuantileMethod(_StrictModel):
    """The weighted low quantile of a variable's own readings over each window.

    The default for every variable. Each reading in a window counts in
    proportion to the share of its cell inside the window (METHODS.md §6.3),
    and a window holding fewer readings than the count rule asks for is
    blank rather than a baseline built on a handful of points (§6.4).
    """

    method: Literal["rolling_quantile"] = "rolling_quantile"


class FromFieldMethod(_StrictModel):
    """Another instrument's baseline of the same field, joined onto this one's cells.

    For an instrument too sparse to see a background at the windows that
    matter -- a canister filling every nine minutes has one fill in more than
    half of all ten-minute windows (METHODS.md §6.5) -- the atmosphere has one
    background per field, and a dense instrument beside it samples that
    background every second. Its baseline at the same window and quantile is
    averaged onto this instrument's cells by the ordinary join, and the
    product records which instrument it came from. It fails exactly when the
    two instruments disagree in calibration, which is why the enhancement's
    systematic uncertainty then carries both instruments' terms (§6.7).
    """

    method: Literal["from_field"] = "from_field"
    instrument: str = Field(
        min_length=1,
        description=(
            "The instrument whose baseline this variable adopts. It must declare a "
            "variable of the same `field` as this one, and its own baseline for "
            "that variable must not itself be `from_field` (no chains); checked "
            "when manifest and analysis configs are combined."
        ),
    )


class ConstantMethod(_StrictModel):
    """A declared number, zero included.

    At zero the enhancement is the concentration. A slope fitted inside an
    event loses nothing by it, since a baseline flat across the event moves
    the intercept and not the slope (METHODS.md §6.1), and a receptor model
    run on concentrations rather than enhancements is common practice, with
    the background appearing as a factor of its own.

    A non-zero value is a claim about the background and carries no
    uncertainty of its own: its error shifts every enhancement alike and
    enters no budget (METHODS.md §6.5). An optional sigma for it waits for a
    configuration that uses a non-zero constant.
    """

    method: Literal["constant"] = "constant"
    value: float = Field(
        description=(
            "The baseline, in the variable's canonical units (after the "
            "manifest's unit conversion). Zero makes the enhancement the concentration."
        ),
    )


#: Tagged union of the registered baseline methods, dispatched on ``method``
#: -- the same discriminator convention as the manifest's QA/QC rules, loaders
#: and platforms. Every name here is a registered method in
#: :mod:`tsara.baseline` and has a section in METHODS.md §6.5.
BaselineMethod = Annotated[
    RollingQuantileMethod | FromFieldMethod | ConstantMethod, Field(discriminator="method")
]


class BaselineConfig(_StrictModel):
    """Baselines: the window × quantile sweep, the count rule, and the method per variable.

    The baseline at each reading is a low quantile (e.g. the 5th percentile)
    of the signal within a window of stated duration centred on the reading.
    Low quantiles track the background *underneath* plumes because plumes
    only ever add mass -- enhancements are one-sided -- so the lower tail of
    a window is dominated by background air. There is no single true
    baseline: a window of length *w* follows everything slower than about
    *w* and leaves what is shorter standing as enhancement (METHODS.md
    §6.1), so the window is a sweep dimension rather than a setting to get
    right.

    ``windows`` and ``quantiles`` are sweep dimensions: every (window,
    quantile) pair is evaluated. Windows double as the *multi-scale
    hierarchy* of the plume catalog's parent-child tree (METHODS.md §6.8) --
    a sharp blip is an enhancement over the shortest window's baseline, a
    broad plume over the longest. ``min_readings`` is the validity rule, a
    count tied to the quantile (§6.4). ``methods`` chooses, per variable, how
    its baseline is built (§6.5); anything not named uses the rolling
    quantile of its own readings.
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
    min_readings: int | None = Field(
        default=None,
        ge=2,
        description=(
            "How many readings a window must hold for its baseline to be "
            "reported at a quantile; a thinner window is blank, with the count "
            "recorded beside it. `null` (the default) means ceil(1/q) for each "
            "quantile q: the count at which a window holds, on average, one "
            "reading below its q-quantile. With fewer it usually holds none that "
            "low, so what it reports is too high (METHODS.md §6.4). An integer "
            "applies to every quantile. "
            "A count rather than a fraction, because a fraction never said what "
            "it was a fraction of, and its three possible denominators disagree "
            "completely on real records."
        ),
    )
    methods: dict[str, BaselineMethod] = Field(
        default_factory=dict,
        description=(
            "Per-variable baseline method, keyed '<instrument>.<variable>' "
            "(variable names are unique per instrument, METHODS.md §1.6). A "
            "variable not named here uses `rolling_quantile`. Each entry is "
            "checked against the manifest when the two configs are combined."
        ),
    )

    def min_readings_for(self, quantile: float) -> int:
        """Return the readings a window must hold to report ``quantile``.

        Parameters
        ----------
        quantile : float
            One of ``quantiles``, in (0, 0.5].

        Returns
        -------
        int
            ``min_readings`` when set; otherwise the smallest integer not
            below 1/q, the count at which a window holds, on average, one
            reading below its q-quantile (METHODS.md §6.4). Rounded before
            the ceiling so that a quantile whose
            reciprocal is a whole number in arithmetic but not in float64
            (1/0.05 is 20.000000000000004) gives that whole number.
        """
        if self.min_readings is not None:
            return self.min_readings
        return math.ceil(round(1.0 / quantile, 9))

    def method_for(self, instrument: str, variable: str) -> BaselineMethod:
        """Return the baseline method configured for one variable.

        Parameters
        ----------
        instrument, variable : str
            The variable, as the manifest names it.

        Returns
        -------
        RollingQuantileMethod or FromFieldMethod or ConstantMethod
            The configured method, or :class:`RollingQuantileMethod` when
            ``methods`` does not name this variable.
        """
        return self.methods.get(f"{instrument}.{variable}", RollingQuantileMethod())

    @field_validator("methods")
    @classmethod
    def _keys_name_an_instrument_and_a_variable(
        cls, value: dict[str, BaselineMethod]
    ) -> dict[str, BaselineMethod]:
        """Require every key to be ``<instrument>.<variable>``, both identifiers.

        Whether the two names exist is the combined config's question
        (:class:`~tsara.config.loader.TsaraConfig`), since the analysis
        schema alone cannot see the manifest; the shape is checked here so
        that a key such as ``ch4`` fails at once with the spelling wanted
        rather than later as an instrument nobody declared.
        """
        for key in value:
            instrument, dot, variable = key.partition(".")
            if not dot or not instrument.isidentifier() or not variable.isidentifier():
                raise ValueError(
                    f"BaselineConfig.methods keys are '<instrument>.<variable>', e.g. "
                    f"'iwas.benzene'; got {key!r}."
                )
        return value

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
# Plumes
# ---------------------------------------------------------------------------


#: Clean-level estimators (METHODS.md §6.8), registered by name in the plumes
#: stage. One today: the half-sample mode, chosen over the midpoint of the
#: shortest half, which collapses once plumes are most of a record.
CleanLevelEstimator = Literal["half_sample_mode"]


class PlumesConfig(_StrictModel):
    """Finding plume events on the enhancements: thresholds, records, triggers.

    Per variable, at every point of the baseline sweep, the plumes stage
    describes plume-free air by its **clean level** (the most common
    enhancement) and its **clean spread** (1.4826 times the median distance
    below that level), and marks an event wherever the statistic
    z = (enhancement - level) / spread reaches an entry multiple and stays
    above an exit multiple (METHODS.md §6.8). Plumes only add, so the
    readings below the most common value are plume-free air; the spread is
    measured there. The two multiples are sweep dimensions, like the
    baseline's window and quantile, because where an event starts and ends
    moves with them, and that movement is methodological.

    The clean spread is not a measurement uncertainty: it includes the
    background's wobble at the window's scale, and a declared or reported
    sigma plays no part in any threshold (§2.3).

    Level and spread are computed per **record**: a stream split where
    consecutive readings are more than ``record_gap`` apart and cut into
    equal parts no longer than ``max_record_length``, so that days of
    different air are never described by one number. A record holding fewer
    than ``min_clean_readings`` readings below its level has no description
    and no events of its own, and says why; a sparse variable, such as a
    canister's, takes its events from a dense one named in ``triggers``.
    """

    enter_multiple: tuple[float, ...] = Field(
        default=(3.0,),
        min_length=1,
        description=(
            "Entry thresholds, in multiples of the clean spread above the clean "
            "level; a sweep dimension ('enter_multiple'). An event holds at least "
            "one reading this far up."
        ),
    )
    exit_multiple: tuple[float, ...] = Field(
        default=(1.0,),
        min_length=1,
        description=(
            "Exit thresholds, in the same units; a sweep dimension ('exit_multiple'). "
            "An event runs while its readings stay this far up, so a lower exit "
            "makes longer events. Every entry must exceed every exit."
        ),
    )
    max_internal_gap: str = Field(
        default="5s",
        description=(
            "A dip below the exit threshold shorter than this is bridged, so that "
            "noise does not split one plume in two. A dropout is never bridged: "
            "missing data is not turbulent air."
        ),
    )
    record_gap: str = Field(
        default="2h",
        description=(
            "A stream is split into records where consecutive finite readings are "
            "more than this apart; the clean level and spread are computed per record."
        ),
    )
    max_record_length: str = Field(
        default="6h",
        description=(
            "A record longer than this is cut into equal parts no longer than it, "
            "since a site or a van that logs for days without a gap would otherwise "
            "be described by one number (METHODS.md §6.8)."
        ),
    )
    min_clean_readings: int = Field(
        default=100,
        ge=1,
        description=(
            "Readings a record must hold below its clean level for its level and "
            "spread to be reported. Fewer, and the record is blank with the reason "
            "and defines no events. 100 is a floor for having a scale at all, not a "
            "guarantee of the chance rate (METHODS.md §6.8)."
        ),
    )
    clean_level_estimator: CleanLevelEstimator = Field(
        default="half_sample_mode",
        description="The registered estimator of the clean level (METHODS.md §6.8).",
    )
    triggers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Variables that take their events from another variable instead of "
            "finding their own, for an instrument too sparse for a clean level of "
            "its own, such as a canister. Keyed '<instrument>' (every gas variable "
            "it declares) or '<instrument>.<variable>' (which wins over its "
            "instrument's key), each naming its trigger as '<instrument>.<variable>', "
            "of any field. Checked against the manifest when the two configs are "
            "combined."
        ),
    )

    def trigger_for(self, instrument: str, variable: str) -> str | None:
        """Return the variable a variable takes its events from, if any.

        Parameters
        ----------
        instrument, variable : str
            The variable, as the manifest names it.

        Returns
        -------
        str or None
            The trigger as ``'<instrument>.<variable>'``: the variable's own
            key when ``triggers`` has one, else its instrument's, else
            ``None``, meaning it finds its own events.
        """
        return self.triggers.get(f"{instrument}.{variable}", self.triggers.get(instrument))

    @field_validator("max_internal_gap", "record_gap", "max_record_length")
    @classmethod
    def _valid_durations(cls, value: str, info: ValidationInfo) -> str:
        _validate_duration(value, field=f"PlumesConfig.{info.field_name}")
        return value

    @field_validator("enter_multiple", "exit_multiple")
    @classmethod
    def _positive_and_increasing(
        cls, value: tuple[float, ...], info: ValidationInfo
    ) -> tuple[float, ...]:
        """Require every multiple positive, and each list strictly increasing.

        Enforced rather than silently sorted, as the baseline's windows are:
        each list becomes a sweep coordinate, and a shuffled or repeated one
        more likely holds a typo than an intention.
        """
        if any(not math.isfinite(m) or m <= 0 for m in value):
            raise ValueError(
                f"PlumesConfig.{info.field_name} must be positive multiples of the clean "
                f"spread; got {list(value)}."
            )
        if any(b <= a for a, b in zip(value, value[1:])):
            raise ValueError(
                f"PlumesConfig.{info.field_name} must be strictly increasing; got {list(value)}."
            )
        return value

    @field_validator("triggers")
    @classmethod
    def _keys_and_triggers_are_spelled_right(cls, value: dict[str, str]) -> dict[str, str]:
        """Require every key and trigger to be spelled as a variable is named.

        A key is ``<instrument>`` or ``<instrument>.<variable>``, a trigger
        ``<instrument>.<variable>``, every part an identifier. Whether they
        exist is the combined config's question, since this schema cannot see
        the manifest; the shape is checked here so that a misspelling fails at
        once with the spelling wanted.
        """
        for key, trigger in value.items():
            parts = key.split(".")
            if len(parts) > 2 or not all(part.isidentifier() for part in parts):
                raise ValueError(
                    "PlumesConfig.triggers keys are '<instrument>' or "
                    f"'<instrument>.<variable>', e.g. 'iwas' or 'iwas.benzene'; got {key!r}."
                )
            instrument, dot, variable = trigger.partition(".")
            if not dot or not instrument.isidentifier() or not variable.isidentifier():
                raise ValueError(
                    f"PlumesConfig.triggers['{key}'] names its trigger as "
                    f"'<instrument>.<variable>', e.g. 'ptr.benzene'; got {trigger!r}."
                )
        return value

    @model_validator(mode="after")
    def _every_entry_above_every_exit(self) -> PlumesConfig:
        """Refuse a configuration in which any entry is at or below any exit.

        Both are swept, so every pairing is run; at entry <= exit the
        hysteresis inverts and an event's boundaries stop meaning anything.
        """
        if min(self.enter_multiple) <= max(self.exit_multiple):
            raise ValueError(
                f"Every PlumesConfig.enter_multiple must exceed every exit_multiple, since "
                f"both are swept; got enter {list(self.enter_multiple)} and exit "
                f"{list(self.exit_multiple)}."
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
            "The denominator species (e.g. 'ch4' or 'co2'), as a field: the "
            "physical quantity a manifest variable declares it measures, which "
            "is the variable's name unless it declares `field`. Must be the "
            "field of a role='gas' variable; cross-checked when manifest and "
            "analysis configs are combined."
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
    — downstream code never needs None checks for whole stages. The output
    grid is the one exception and is ``None`` when absent: a grid has no
    sensible default period, since the period is the export's choice, and an
    absent grid is a statement (nothing is exported on a tiling) rather than
    a stage switched off.
    """

    output_grid: OutputGridConfig | None = Field(
        default=None,
        description=(
            "Optional uniform tiling for an export (METHODS.md §1.4). Absent by "
            "default: the continuous baseline state lives per stream at native "
            "rate (§6.2), so a run that never exports a table needs no grid."
        ),
    )
    alignment: AlignmentConfig = Field(
        default_factory=AlignmentConfig,
        description=(
            "How support may be changed: the copy policy (METHODS.md §11.2.4) and "
            "the auxiliary-field interpolation guard (§1.2)."
        ),
    )
    pairing: PairingConfig = Field(
        default_factory=PairingConfig,
        description="Sufficiency guard on cross-species pairs (METHODS.md §1.3).",
    )
    baseline: BaselineConfig = Field(
        description="Baseline settings: the sweep, the count rule and the method per variable."
    )
    plumes: PlumesConfig = Field(
        default_factory=PlumesConfig,
        description="Plume events: thresholds, records and triggers (METHODS.md §6.8).",
    )
    smoothing: SmoothingConfig = Field(
        default_factory=SmoothingConfig, description="Optional low-pass alignment stage."
    )
    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig, description="Optional Source Complex clustering."
    )
    regression: RegressionConfig = Field(description="Enhancement-ratio regression settings.")
