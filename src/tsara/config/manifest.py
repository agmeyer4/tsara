"""Ingestion manifest schema: *what the data is and how to read it*.

A manifest YAML file fully describes a measurement campaign's raw data:
which instruments exist, where their files live (via path templates), how to
parse each file format, how columns map to canonical variable names, what
units they arrive in, what QA/QC masking to apply, and whether the platform
was stationary or mobile.

Design decisions embedded in this schema
-----------------------------------------
* **Species are data, not code.** Gas species appear only as keys in a
  ``variables:`` mapping. Adding a 40th VOC to a campaign is a YAML edit;
  no TSARA source file ever names a specific gas.
* **``extra="forbid"`` everywhere.** A typo like ``quantles:`` in a science
  config silently changing results is the worst failure mode a config-driven
  package can have. Every model rejects unknown keys loudly instead.
* **Discriminated unions for polymorphism.** Platform type (stationary vs
  mobile), loader format (csv vs icartt), and QA/QC rule kind are all tagged
  unions on a literal field. Pydantic then produces precise error messages
  ("unexpected field for kind='stationary'") rather than a wall of failed
  alternatives.
* **Cross-references are validated at parse time.** A mobile platform names
  the instrument that carries the GPS; the manifest refuses to validate if
  that instrument (or its lat/lon variables) doesn't exist. Errors surface
  in seconds at config load, not hours later mid-ingestion on the cluster.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    Field,
    field_validator,
    model_validator,
)

from tsara.config.base import StrictModel as _StrictModel
from tsara.config.base import validate_positive_timedelta as _validate_duration

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class UnitConversion(_StrictModel):
    """Linear conversion from a variable's native unit to a canonical unit.

    Applied as ``canonical = native * scale + offset``. A linear form covers
    the overwhelming majority of trace-gas needs (ppm→ppb, °C→K, mg→µg);
    it deliberately avoids a heavyweight unit library (pint) for now, though
    the interface — declared ``from_unit``/``to_unit`` strings — leaves room
    to plug one in later without touching manifests.

    Examples
    --------
    ppm to ppb::

        convert: {from_unit: ppm, to_unit: ppb, scale: 1000.0}

    Celsius to Kelvin::

        convert: {from_unit: degC, to_unit: K, offset: 273.15}
    """

    from_unit: str = Field(description="Unit the raw file reports (bookkeeping/metadata).")
    to_unit: str = Field(description="Canonical unit after conversion; stored in dataset attrs.")
    scale: float = Field(default=1.0, description="Multiplicative factor applied first.")
    offset: float = Field(default=0.0, description="Additive offset applied after scaling.")

    @model_validator(mode="after")
    def _must_actually_convert(self) -> UnitConversion:
        """Reject the identity conversion — if units already match, omit the block.

        This keeps manifests honest: a ``convert:`` block always means the
        numbers change.
        """
        if self.scale == 1.0 and self.offset == 0.0:
            raise ValueError(
                "UnitConversion with scale=1 and offset=0 is a no-op; "
                "omit the 'convert' block if no conversion is needed."
            )
        return self


# ---------------------------------------------------------------------------
# QA/QC rules (tagged union on 'kind')
# ---------------------------------------------------------------------------


class RangeRule(_StrictModel):
    """Mask samples outside a physically plausible range.

    Either bound may be omitted for one-sided checks (e.g., mixing ratios
    must simply be non-negative).
    """

    kind: Literal["range"] = "range"
    min: float | None = Field(default=None, description="Mask values strictly below this.")
    max: float | None = Field(default=None, description="Mask values strictly above this.")

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> RangeRule:
        if self.min is None and self.max is None:
            raise ValueError("RangeRule needs at least one of 'min' or 'max'.")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(f"RangeRule min ({self.min}) must be < max ({self.max}).")
        return self


class FlagRule(_StrictModel):
    """Mask samples according to an instrument status/QC flag column.

    Exactly one of ``good_values`` / ``bad_values`` must be given: listing
    both is ambiguous for flag values that appear in neither list. ICARTT
    files typically ship a numeric flag where 0 = good.
    """

    kind: Literal["flag"] = "flag"
    flag_column: str = Field(description="Column in the *raw file* holding the QC flag.")
    good_values: list[float | int | str] | None = Field(
        default=None, description="Keep only samples whose flag is in this list."
    )
    bad_values: list[float | int | str] | None = Field(
        default=None, description="Mask samples whose flag is in this list."
    )

    @model_validator(mode="after")
    def _exactly_one_list(self) -> FlagRule:
        if (self.good_values is None) == (self.bad_values is None):
            raise ValueError("FlagRule requires exactly one of 'good_values' or 'bad_values'.")
        return self


class SpikeRule(_StrictModel):
    """Mask isolated electronic spikes via a rolling-MAD outlier test.

    A sample is masked when it deviates from the rolling median by more than
    ``n_mad`` × the rolling median absolute deviation. MAD is used instead
    of standard deviation because the statistic must be robust to the very
    spikes it is hunting. NOTE: deliberately distinct from plume detection —
    this is for sub-second instrument glitches, and the window should be far
    shorter than any real atmospheric feature.
    """

    kind: Literal["spike"] = "spike"
    window: str = Field(description="Rolling window as a pandas timedelta string, e.g. '5s'.")
    n_mad: float = Field(default=6.0, gt=0, description="Deviation threshold in MADs.")

    @field_validator("window")
    @classmethod
    def _valid_timedelta(cls, value: str) -> str:
        _validate_duration(value, field="SpikeRule.window")
        return value


#: Tagged union — Pydantic dispatches on the 'kind' literal, giving precise
#: per-kind validation errors instead of trying every alternative blindly.
QAQCRule = Annotated[RangeRule | FlagRule | SpikeRule, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

#: Roles drive downstream behavior: 'gas' variables populate the species
#: dimension; 'met' variables may be circular (wind direction); 'gps_*'
#: variables feed mobile-platform coordinate alignment; 'aux' is carried
#: through untouched (e.g., cavity pressure used only for diagnostics).
VariableRole = Literal["gas", "met", "gps_lat", "gps_lon", "gps_alt", "aux"]


class DeclaredUncertainty(_StrictModel):
    """A component's 1-sigma uncertainty as a declared noise-floor + fraction.

    Total sigma is combined in quadrature: ``sigma = sqrt(absolute**2 +
    (relative * value)**2)``. This two-term form matches how instrument
    teams typically report precision (a noise floor plus a percent-of-
    reading term). Used for either the random or systematic component of
    :class:`UncertaintySpec` (METHODS.md §2.2).
    """

    mode: Literal["declared"] = "declared"
    absolute: float = Field(default=0.0, ge=0, description="Noise floor in canonical units.")
    relative: float = Field(
        default=0.0, ge=0, lt=1, description="Fraction of reading, e.g. 0.02 for 2 %."
    )

    @model_validator(mode="after")
    def _nonzero(self) -> DeclaredUncertainty:
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError(
                "DeclaredUncertainty with absolute=0 and relative=0 declares "
                "perfect measurement; omit this component instead."
            )
        return self


class ReportedUncertainty(_StrictModel):
    """A component's per-point 1-sigma uncertainty, read from the raw file.

    Some instruments (e.g. EM27 retrievals) report a per-sample uncertainty
    that changes every point rather than a fixed instrument spec — this
    mode names the raw-file column holding it (METHODS.md §2.2). Unit
    conversion for this column follows the parent variable's own
    ``convert.scale`` (a spread scales with the value's units) but never
    ``convert.offset`` (a spread has no origin to shift); that scaling is
    applied at ingestion (Phase 3), not by this schema.
    """

    mode: Literal["reported"] = "reported"
    column: str = Field(
        min_length=1, description="Column in the raw file holding this component's per-point sigma."
    )


#: Tagged union — dispatches on 'mode', matching the QAQCRule/LoaderConfig/
#: PlatformConfig discriminated-union convention used throughout this schema.
ComponentUncertainty = Annotated[
    DeclaredUncertainty | ReportedUncertainty, Field(discriminator="mode")
]


class UncertaintySpec(_StrictModel):
    """Per-sample measurement uncertainty, consumed by regression in Phase 7.

    Two components, carried separately because downstream operations treat
    them differently (METHODS.md §2.1, §3, §4.3):

    * ``random`` — uncorrelated point-to-point noise. Averages down with the
      number of effective samples; feeds regression point weights directly.
    * ``systematic`` — correlated error (calibration scale/offset, drift).
      Does *not* average down; propagated to the ratio analytically after
      the fit rather than folded into per-point weights.

    Each component is independently either :class:`DeclaredUncertainty`
    (absolute/relative) or :class:`ReportedUncertainty` (a per-point
    column) — a real instrument might declare a constant systematic
    calibration uncertainty while reporting a per-point random column, or
    any other combination. Omitting a component means "not modeled here":
    an omitted ``systematic`` is treated as zero; an omitted ``random``
    falls back to the empirical ``diff_mad`` estimator (METHODS.md §2.5) at
    runtime, with `uncertainty_source` provenance recorded either way.
    """

    random: ComponentUncertainty | None = Field(
        default=None,
        description=(
            "Random (uncorrelated) component. None = fall back to the "
            "empirical noise estimator at runtime (METHODS.md §2.3)."
        ),
    )
    systematic: ComponentUncertainty | None = Field(
        default=None,
        description=(
            "Systematic (correlated, non-averaging) component. None = not "
            "modeled (treated as zero)."
        ),
    )
    decorrelation_timescale: str | None = Field(
        default=None,
        description=(
            "Optional AR(1)-like decorrelation timescale tau for the random "
            "component (METHODS.md §3.4), covering errors correlated over "
            "minutes but not hours. None = points treated as independent."
        ),
    )

    @field_validator("decorrelation_timescale")
    @classmethod
    def _valid_decorrelation_timescale(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_duration(value, field="UncertaintySpec.decorrelation_timescale")
        return value

    @model_validator(mode="after")
    def _at_least_one_component(self) -> UncertaintySpec:
        if self.random is None and self.systematic is None:
            raise ValueError(
                "UncertaintySpec with no 'random' and no 'systematic' component "
                "declares nothing; omit the 'uncertainty' block instead."
            )
        return self


class VariableConfig(_StrictModel):
    """One variable measured by an instrument.

    The *canonical name* of the variable is the key under which this config
    appears in ``InstrumentConfig.variables`` — e.g. ``ch4:``. This model
    describes how to find it in the raw file and how to treat it.
    """

    column: str = Field(description="Column/variable name as it appears in the raw file.")
    role: VariableRole = Field(default="gas", description="Downstream handling category.")
    units: str = Field(description="Native units in the raw file, e.g. 'ppb'.")
    convert: UnitConversion | None = Field(
        default=None, description="Optional linear conversion to canonical units."
    )
    circular: bool = Field(
        default=False,
        description=(
            "True for angular quantities (wind direction, degrees 0-360): "
            "synchronization then uses vector averaging, never arithmetic means."
        ),
    )
    uncertainty: UncertaintySpec | None = Field(
        default=None,
        description=(
            "Measurement uncertainty budget (random/systematic components), "
            "consumed by plume detection's noise scale and by regression "
            "point weights (METHODS.md §2, §4.3)."
        ),
    )
    qaqc: tuple[QAQCRule, ...] = Field(
        default=(), description="Masking rules applied in order at ingestion."
    )
    description: str = Field(default="", description="Free-text note for humans.")

    @model_validator(mode="after")
    def _circular_only_for_met(self) -> VariableConfig:
        """Circular statistics only make sense for angular met variables.

        A 'circular gas concentration' is almost certainly a manifest editing
        mistake, so fail fast rather than silently vector-averaging ppb.
        """
        if self.circular and self.role != "met":
            raise ValueError(
                f"circular=true is only valid for role='met' variables (got role='{self.role}')."
            )
        return self


# ---------------------------------------------------------------------------
# File loaders (tagged union on 'format')
# ---------------------------------------------------------------------------

#: Path templates may contain {named} metadata fields, strftime date tokens,
#: and glob wildcards. The crawler (Phase 3) compiles these into a glob plus
#: a regex that extracts the metadata back out of each matched path.
_TEMPLATE_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class TimeParsing(_StrictModel):
    """How to build the datetime index from a delimited-text file.

    ICARTT files don't need this — their time axis is defined by the format
    specification itself (seconds from a header-declared date).

    Why a *list* of columns: loggers routinely split one timestamp across
    several fields. A Picarro ``DataLog_User`` file, for instance, carries
    ``DATE`` and ``TIME`` as separate whitespace-delimited columns; the
    timestamp only exists once they are rejoined. Such a file is unreadable
    if a schema can name only one column, and the workaround (depend on a
    redundant epoch column) fails on the loggers that don't emit one. The
    singular ``column:`` spelling remains valid shorthand for the common
    one-column case, exactly as ``path_template`` does for
    :attr:`_BaseLoader.path_templates`.
    """

    columns: tuple[str, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("columns", "column"),
        description=(
            "Column(s) holding the timestamp (or epoch seconds). A bare "
            "string is accepted for the single-column case; several columns "
            "are concatenated in the order given, separated by `join`, "
            "before parsing (e.g. ['DATE', 'TIME'])."
        ),
    )
    join: str = Field(
        default=" ",
        description=(
            "Separator used to concatenate multiple time columns before "
            "parsing. Ignored when a single column is given."
        ),
    )
    format: str | None = Field(
        default=None,
        description=(
            "strftime pattern (e.g. '%Y-%m-%d %H:%M:%S'), or the sentinels "
            "'unix' (epoch seconds) / 'iso8601'. None lets pandas infer — "
            "convenient but slower and riskier; prefer explicit formats. "
            "The pattern must describe the *joined* string when several "
            "columns are given."
        ),
    )
    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone of naive timestamps in the file. All TSARA "
            "processing is UTC internally; anything else is converted."
        ),
    )

    @field_validator("columns", mode="before")
    @classmethod
    def _coerce_single_column(cls, value: object) -> object:
        """Accept a bare string as shorthand for a one-element list."""
        if isinstance(value, str):
            return (value,)
        return value

    @field_validator("columns")
    @classmethod
    def _sane_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank and repeated column names."""
        for name in value:
            if not name.strip():
                raise ValueError("TimeParsing.columns entries must be non-empty column names.")
        if len(set(value)) != len(value):
            raise ValueError(
                "TimeParsing.columns contains duplicates; each entry must name "
                "a distinct column of the raw file."
            )
        return value

    @model_validator(mode="after")
    def _unix_is_single_column(self) -> TimeParsing:
        """Epoch seconds are one number, so 'unix' cannot span columns.

        Concatenating two numeric fields and reading the result as epoch
        seconds would silently produce a wildly wrong date rather than an
        error, so this combination is refused at config load.
        """
        if self.format == "unix" and len(self.columns) > 1:
            raise ValueError(
                "TimeParsing: format='unix' expects a single column of epoch "
                f"seconds, but {len(self.columns)} columns were given "
                f"({list(self.columns)}). Use an explicit strftime pattern to "
                "combine separate date and time fields."
            )
        return self


class _BaseLoader(_StrictModel):
    """Fields common to every file format.

    Why a *list* of templates: real archives are messy. The same
    instrument's files routinely live under several directory layouts and
    naming conventions at once (a reorganization mid-campaign, a NOAA
    archive mirror vs. a local copy, per-year restructuring). Each template
    describes one such convention; the crawler (Phase 3) searches all of
    them and merges the results, harvesting {field} metadata per template.
    Adding a new layout is a YAML edit, never a code change.
    """

    path_templates: tuple[str, ...] = Field(
        min_length=1,
        # Ergonomics: a manifest with one layout may write the singular
        # `path_template: "..."` with a plain string; both spellings land
        # in this field (the string is coerced to a 1-tuple below).
        validation_alias=AliasChoices("path_templates", "path_template"),
        description=(
            "One or more path patterns relative to Manifest.base_path, each "
            "describing a directory/naming convention where this "
            "instrument's files live. Supports {field} metadata "
            "placeholders, strftime tokens (%Y/%m/%d), and glob wildcards, "
            "e.g. '{institution}/{campaign}/%Y/%m/%d/*.ict'."
        ),
    )

    max_dropped_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Largest fraction of a file's rows that may be discarded (for a "
            "missing or unparseable timestamp, or a wrong field count) before "
            "the read is treated as a misparse and raises instead of warning. "
            "Readers have always refused a file where *every* row failed; this "
            "generalizes that to near-total loss, which is the case that "
            "actually hurts — a file yielding 2 rows of 10,000 looks downstream "
            "like a quiet instrument rather than a broken parse. Raise it "
            "toward 1.0 for archives where heavy row loss is genuinely "
            "expected; 1.0 restores warn-only behaviour."
        ),
    )

    @field_validator("path_templates", mode="before")
    @classmethod
    def _coerce_single_template(cls, value: object) -> object:
        """Accept a bare string as shorthand for a one-element list."""
        if isinstance(value, str):
            return (value,)
        return value

    @field_validator("path_templates")
    @classmethod
    def _sane_templates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for template in value:
            if not template or template.strip() == "":
                raise ValueError("path_templates entries must be non-empty path patterns.")
            if Path(template).is_absolute():
                raise ValueError(
                    "path_templates must be relative (they are joined to "
                    f"Manifest.base_path); got absolute path '{template}'."
                )
            # Reserved field names would collide with coordinates TSARA creates.
            reserved = {"time", "species"} & set(_TEMPLATE_FIELD_RE.findall(template))
            if reserved:
                raise ValueError(
                    f"path_templates metadata fields {sorted(reserved)} are reserved names."
                )
        if len(set(value)) != len(value):
            raise ValueError(
                "path_templates contains duplicates; each entry must describe "
                "a distinct layout or the same files would be ingested twice."
            )
        return value

    @property
    def template_fields(self) -> tuple[str, ...]:
        """Metadata field names across all templates, deduplicated in order."""
        seen: dict[str, None] = {}
        for template in self.path_templates:
            for field in _TEMPLATE_FIELD_RE.findall(template):
                seen.setdefault(field)
        return tuple(seen)


class CSVLoader(_BaseLoader):
    r"""Loader for delimited text files (CSV/TSV and friends).

    "CSV" is read loosely here: the same reader covers comma-, tab- and
    whitespace-delimited logger output, which in practice arrives with
    extensions like ``.dat`` and ``.txt`` as often as ``.csv``. Set
    ``delimiter: '\s+'`` for the fixed-width-looking, space-padded files
    that gas analyzers commonly write.

    Headerless files are supported via ``header_row: null`` plus an explicit
    ``column_names`` list — some instruments (e.g. Aeris Spectralite logs)
    begin emitting data on line 1 with the column meanings living only in a
    manual, which is precisely the knowledge a manifest exists to record.
    """

    format: Literal["csv"] = "csv"
    time: TimeParsing = Field(description="Datetime index construction (required for CSV).")
    delimiter: str = Field(
        default=",",
        description=(
            "Field separator. Accepts a regular expression, so '\\s+' handles "
            "space- or tab-padded logger output with runs of whitespace."
        ),
    )
    header_row: int | None = Field(
        default=0,
        ge=0,
        description=(
            "0-based index of the column-name line, counted AFTER blank and "
            "comment lines are discarded -- not its physical line number. A "
            "file whose header sits on physical line 4 with one blank line "
            "above it therefore needs `header_row: 2`. Use null for a file "
            "with no header at all (which then requires `column_names`)."
        ),
    )
    column_names: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Column names for a headerless file, in file order. Required "
            "when `header_row` is null, and forbidden otherwise."
        ),
    )
    na_values: tuple[str, ...] = Field(
        default=(), description="Extra strings to treat as missing, e.g. ['-9999', 'NULL']."
    )
    comment: str | None = Field(
        default=None, max_length=1, description="Comment-line prefix character to skip."
    )

    @field_validator("column_names")
    @classmethod
    def _sane_column_names(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Reject empty lists, blank names, and duplicates."""
        if value is None:
            return None
        if not value:
            raise ValueError("column_names must name at least one column when given.")
        for name in value:
            if not name.strip():
                raise ValueError("column_names entries must be non-empty column names.")
        if len(set(value)) != len(value):
            raise ValueError(
                "column_names contains duplicates; positional names must be "
                "distinct or later columns would be unreachable."
            )
        return value

    @model_validator(mode="after")
    def _header_and_names_agree(self) -> CSVLoader:
        """Tie ``header_row`` and ``column_names`` into one coherent statement.

        The two fields answer the same question — "where do column names come
        from?" — so exactly one of them must supply the answer. Allowing both
        would mean silently overriding a header that is present, which is
        indistinguishable at read time from a manifest whose author
        miscounted the header row; refusing the combination turns that into
        an immediate config error instead of a column mis-mapping that
        surfaces as nonsense concentrations.
        """
        if self.header_row is None and self.column_names is None:
            raise ValueError(
                "CSVLoader: header_row=null declares a file with no header "
                "line, so column_names must list the columns in file order."
            )
        if self.header_row is not None and self.column_names is not None:
            raise ValueError(
                "CSVLoader: column_names applies only to headerless files. "
                "Either set header_row=null (the file has no header) or drop "
                "column_names (the header line supplies the names)."
            )
        return self


class ICARTTLoader(_BaseLoader):
    """Loader for ICARTT (.ict) files per the NASA/NOAA FFI-1001 convention.

    TSARA ships its own FFI-1001 parser (Phase 3) rather than depending on
    the GPL-licensed, unmaintained ``icartt`` PyPI package. Owning the
    parser also lets us tolerate the spec-noncompliant files common in real
    campaign archives.

    Time handling is intentionally absent: the ICARTT format defines its own
    time axis (an independent variable of seconds since a date declared in
    the header), which the reader interprets directly.
    """

    format: Literal["icartt"] = "icartt"
    revision_policy: Literal["latest", "all"] = Field(
        default="latest",
        description=(
            "How to handle multiple revisions of the same data file. ICARTT "
            "filenames follow 'dataID_locationID_YYYYMMDD[_R#].ict', and "
            "archives routinely hold several revisions of one day (R0 "
            "preliminary, R1 final, ...). 'latest' keeps only the highest "
            "revision per (identifier, date, trailing comment) — the safe "
            "default, since ingesting all revisions double-counts the same "
            "air. The trailing comment is part of the key deliberately: "
            "'_L1'/'_L2' processing levels and '_Drive01'/'_Stationary01' "
            "are different products, not revisions of each other. "
            "'all' ingests everything (only for revision-comparison "
            "studies; expect duplicate timestamps downstream)."
        ),
    )


class ParquetLoader(_BaseLoader):
    """Loader for Apache Parquet files.

    Why parquet is a first-class format rather than an afterthought: a
    campaign's *processed* stages are commonly stored as parquet even when
    the instruments wrote text. The archive this package targets keeps its
    entire instrument-aligned stage that way, so parquet is the only route
    to that data.

    Why ``time`` is optional here but required for CSV
    -------------------------------------------------
    Parquet stores a dataframe's index as part of the file, so a parquet
    written by pandas usually *already has* its ``DatetimeIndex`` — there is
    nothing to parse and no format string to get wrong. That is the default:
    leave ``time`` unset and the file's own index is used. Set it only for
    files that store time as an ordinary column instead, which is the same
    :class:`TimeParsing` block CSV uses.

    A text format can never do this, which is why ``CSVLoader.time`` is
    mandatory: a CSV has no index, only columns.
    """

    format: Literal["parquet"] = "parquet"
    time: TimeParsing | None = Field(
        default=None,
        description=(
            "How to build the datetime index from column(s). Omit (the "
            "default) to use the index already stored in the file, which is "
            "the usual case for parquet written by pandas."
        ),
    )


#: Tagged union dispatched on 'format'. New formats extend this union and
#: register a reader; the manifest schema needs no other change.
LoaderConfig = Annotated[CSVLoader | ICARTTLoader | ParquetLoader, Field(discriminator="format")]


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


class InstrumentConfig(_StrictModel):
    """One instrument (one data stream) in the campaign.

    The instrument's *name* is its key in ``Manifest.instruments``.
    """

    description: str = Field(default="", description="Free-text note for humans.")
    loader: LoaderConfig = Field(description="How to find and parse this instrument's files.")
    variables: dict[str, VariableConfig] = Field(
        min_length=1,
        description=(
            "Mapping of canonical variable name -> config. Keys become names "
            "in the synchronized dataset (and, for role='gas', entries along "
            "the species dimension)."
        ),
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Fixed values for path-template {fields} that are not meant to be "
            "extracted from disk (e.g. institution: 'uutah'). Fields absent "
            "here are treated as wildcards and harvested from matched paths."
        ),
    )

    @field_validator("variables")
    @classmethod
    def _canonical_names_are_identifiers(
        cls, value: dict[str, VariableConfig]
    ) -> dict[str, VariableConfig]:
        """Canonical names become xarray variable names — keep them clean.

        Requiring valid Python identifiers guarantees they work with
        ``ds.ch4`` attribute access and never collide with path syntax.
        """
        for name in value:
            if not name.isidentifier():
                raise ValueError(
                    f"Canonical variable name '{name}' must be a valid identifier "
                    "(letters, digits, underscores; not starting with a digit)."
                )
        return value

    @model_validator(mode="after")
    def _no_duplicate_columns(self) -> InstrumentConfig:
        """Two canonical names must not read the same raw column.

        Mapping one column to two variables is almost always a copy-paste
        error in the manifest, and it would silently duplicate data.
        """
        seen: dict[str, str] = {}
        for name, var in self.variables.items():
            if var.column in seen:
                raise ValueError(
                    f"Variables '{seen[var.column]}' and '{name}' both read raw "
                    f"column '{var.column}'."
                )
            seen[var.column] = name
        return self

    @model_validator(mode="after")
    def _headerless_columns_are_declared(self) -> InstrumentConfig:
        """For a headerless CSV, every referenced column must be declared.

        Normally a mistyped column name is caught when the file is read and
        pandas reports the header it actually found. A headerless file has no
        such fallback: ``column_names`` *is* the schema, so a typo there
        produces a ``KeyError`` deep inside ingestion — after the crawler has
        already walked the archive — naming a column the user cannot see in
        any file. Checking the reference here fails in seconds at config
        load, which is the same bargain the mobile-GPS cross-reference makes.

        Only the headerless case can be checked: when a header line exists,
        the authoritative name list lives in the data files, not the manifest.
        """
        loader = self.loader
        if not isinstance(loader, CSVLoader) or loader.column_names is None:
            return self

        declared = set(loader.column_names)
        # Everything in the manifest that names a raw column of this file.
        referenced: dict[str, str] = {}
        for column in loader.time.columns:
            referenced[column] = "loader.time.columns"
        for name, var in self.variables.items():
            referenced[var.column] = f"variables.{name}.column"
            if var.uncertainty is not None:
                for component in ("random", "systematic"):
                    spec = getattr(var.uncertainty, component)
                    # Only ReportedUncertainty reads a column; declared does not.
                    if isinstance(spec, ReportedUncertainty):
                        referenced[spec.column] = f"variables.{name}.uncertainty.{component}.column"
            for rule in var.qaqc:
                if isinstance(rule, FlagRule):
                    referenced[rule.flag_column] = f"variables.{name}.qaqc flag_column"

        missing = {col: where for col, where in referenced.items() if col not in declared}
        if missing:
            detail = "; ".join(f"'{col}' ({where})" for col, where in sorted(missing.items()))
            raise ValueError(
                f"Headerless CSV declares column_names={list(loader.column_names)}, "
                f"but these referenced columns are not among them: {detail}."
            )
        return self


# ---------------------------------------------------------------------------
# Platforms (tagged union on 'kind')
# ---------------------------------------------------------------------------


class StationaryPlatform(_StrictModel):
    """A fixed site: one static lat/lon applied to every sample globally."""

    kind: Literal["stationary"] = "stationary"
    latitude: float = Field(ge=-90, le=90, description="Site latitude, decimal degrees N.")
    longitude: float = Field(ge=-180, le=180, description="Site longitude, decimal degrees E.")
    altitude_m: float | None = Field(
        default=None, description="Optional site elevation above sea level, meters."
    )


class MobilePlatform(_StrictModel):
    """A moving platform: coordinates come from a GPS instrument's timeseries.

    Rather than embedding GPS parsing here, this model *references* an
    instrument declared in the manifest and names which of its canonical
    variables carry the coordinates. Cross-references are validated by
    :class:`Manifest`.
    """

    kind: Literal["mobile"] = "mobile"
    gps_instrument: str = Field(description="Key in Manifest.instruments providing GPS.")
    lat_variable: str = Field(
        default="latitude", description="Canonical variable name with role='gps_lat'."
    )
    lon_variable: str = Field(
        default="longitude", description="Canonical variable name with role='gps_lon'."
    )
    alt_variable: str | None = Field(
        default=None, description="Optional canonical variable name with role='gps_alt'."
    )


PlatformConfig = Annotated[StationaryPlatform | MobilePlatform, Field(discriminator="kind")]


# ---------------------------------------------------------------------------
# The manifest itself
# ---------------------------------------------------------------------------


class Manifest(_StrictModel):
    """Top-level ingestion manifest for one campaign/deployment.

    One manifest = one platform deployment. A study with both a fixed site
    and a mobile van is two manifests processed as two TSARA runs (their
    catalogs can be merged downstream).
    """

    name: str = Field(min_length=1, description="Short campaign identifier, e.g. 'slv_2025'.")
    description: str = Field(default="", description="Free-text campaign summary.")
    base_path: Path = Field(
        description=(
            "Root directory that every instrument's path_template is joined "
            "to. May be relative; the YAML loader resolves it against the "
            "manifest file's own directory."
        )
    )
    platform: PlatformConfig = Field(description="Stationary or mobile platform metadata.")
    instruments: dict[str, InstrumentConfig] = Field(
        min_length=1, description="Mapping of instrument name -> config."
    )

    @model_validator(mode="after")
    def _validate_mobile_gps_references(self) -> Manifest:
        """Mobile platforms must reference a real GPS instrument and variables.

        We check three things, in order of increasing specificity, so error
        messages point at exactly what's wrong:

        1. the named instrument exists;
        2. the named lat/lon (and optional alt) variables exist on it;
        3. those variables carry the matching ``gps_*`` role, so the
           ingestion layer knows to treat them as coordinates, not data.
        """
        if not isinstance(self.platform, MobilePlatform):
            return self

        gps_name = self.platform.gps_instrument
        if gps_name not in self.instruments:
            raise ValueError(
                f"platform.gps_instrument '{gps_name}' is not a declared instrument; "
                f"available: {sorted(self.instruments)}."
            )

        gps_vars = self.instruments[gps_name].variables
        wanted: list[tuple[str, str]] = [
            (self.platform.lat_variable, "gps_lat"),
            (self.platform.lon_variable, "gps_lon"),
        ]
        if self.platform.alt_variable is not None:
            wanted.append((self.platform.alt_variable, "gps_alt"))

        for var_name, expected_role in wanted:
            if var_name not in gps_vars:
                raise ValueError(
                    f"GPS variable '{var_name}' not found on instrument '{gps_name}'; "
                    f"available: {sorted(gps_vars)}."
                )
            actual_role = gps_vars[var_name].role
            if actual_role != expected_role:
                raise ValueError(
                    f"Variable '{var_name}' on instrument '{gps_name}' must have "
                    f"role='{expected_role}' (got '{actual_role}')."
                )
        return self

    @model_validator(mode="after")
    def _no_duplicate_canonical_names_across_instruments(self) -> Manifest:
        """Canonical variable names must be unique campaign-wide.

        Two instruments both producing 'ch4' would collide in the merged
        synchronized dataset. If a campaign genuinely has redundant
        measurements, the manifest must name them distinctly (e.g.
        'ch4_picarro', 'ch4_lgr') — an explicit scientific choice.
        """
        owner: dict[str, str] = {}
        for inst_name, inst in self.instruments.items():
            for var_name in inst.variables:
                if var_name in owner:
                    raise ValueError(
                        f"Canonical variable '{var_name}' is declared by both "
                        f"'{owner[var_name]}' and '{inst_name}'; canonical names "
                        "must be unique across the whole manifest."
                    )
                owner[var_name] = inst_name
        return self

    @property
    def gas_species(self) -> tuple[str, ...]:
        """Canonical names of all role='gas' variables, across instruments.

        This tuple is what ultimately becomes the ``species`` dimension of
        the synchronized dataset — computed, never hardcoded.
        """
        return tuple(
            var_name
            for inst in self.instruments.values()
            for var_name, var in inst.variables.items()
            if var.role == "gas"
        )
