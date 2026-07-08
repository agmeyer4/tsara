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
from tsara.config.base import validate_timedelta as _parse_timedelta


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
        _parse_timedelta(value, field="SpikeRule.window")
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


class UncertaintySpec(_StrictModel):
    """Per-sample measurement uncertainty, consumed by ODR in Phase 7.

    Total 1-sigma uncertainty is combined in quadrature:
    ``sigma = sqrt(absolute**2 + (relative * value)**2)``. This two-term
    form matches how instrument teams typically report precision (a noise
    floor plus a percent-of-reading term).
    """

    absolute: float = Field(default=0.0, ge=0, description="Noise floor in canonical units.")
    relative: float = Field(
        default=0.0, ge=0, lt=1, description="Fraction of reading, e.g. 0.02 for 2 %."
    )

    @model_validator(mode="after")
    def _nonzero(self) -> UncertaintySpec:
        if self.absolute == 0.0 and self.relative == 0.0:
            raise ValueError(
                "UncertaintySpec with absolute=0 and relative=0 declares perfect "
                "measurement; omit the 'uncertainty' block instead."
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
        default=None, description="1-sigma measurement uncertainty for ODR weighting."
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
                f"circular=true is only valid for role='met' variables "
                f"(got role='{self.role}')."
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
    """

    column: str = Field(description="Column holding the timestamp (or epoch seconds).")
    format: str | None = Field(
        default=None,
        description=(
            "strftime pattern (e.g. '%Y-%m-%d %H:%M:%S'), or the sentinels "
            "'unix' (epoch seconds) / 'iso8601'. None lets pandas infer — "
            "convenient but slower and riskier; prefer explicit formats."
        ),
    )
    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone of naive timestamps in the file. All TSARA "
            "processing is UTC internally; anything else is converted."
        ),
    )


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
    """Loader for delimited text files (CSV/TSV and friends)."""

    format: Literal["csv"] = "csv"
    time: TimeParsing = Field(description="Datetime index construction (required for CSV).")
    delimiter: str = Field(default=",", description="Field separator.")
    header_row: int = Field(default=0, ge=0, description="0-based row index of column names.")
    na_values: tuple[str, ...] = Field(
        default=(), description="Extra strings to treat as missing, e.g. ['-9999', 'NULL']."
    )
    comment: str | None = Field(
        default=None, max_length=1, description="Comment-line prefix character to skip."
    )


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
            "revision per (dataID, locationID, date) — the safe default, "
            "since ingesting all revisions double-counts the same air. "
            "'all' ingests everything (only for revision-comparison "
            "studies; expect duplicate timestamps downstream)."
        ),
    )


#: Tagged union dispatched on 'format'. New formats (Phase 3+) extend this
#: union and register a reader; the manifest schema needs no other change.
LoaderConfig = Annotated[CSVLoader | ICARTTLoader, Field(discriminator="format")]


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
