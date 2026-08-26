"""Tests for the manifest schema (tsara.config.manifest)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from tsara.config.manifest import (
    CSVLoader,
    DeclaredUncertainty,
    InstrumentConfig,
    Manifest,
    MobilePlatform,
    ParquetLoader,
    ReportedUncertainty,
    StationaryPlatform,
    TimeParsing,
    UncertaintySpec,
    UnitConversion,
)

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_valid_stationary_manifest_parses(stationary_manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.model_validate(stationary_manifest_dict)
    assert manifest.name == "test_site"
    assert isinstance(manifest.platform, StationaryPlatform)
    assert manifest.platform.latitude == pytest.approx(40.766)
    # Discriminated union picked the right loader class.
    assert isinstance(manifest.instruments["picarro"].loader, CSVLoader)


def test_valid_mobile_manifest_parses(mobile_manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.model_validate(mobile_manifest_dict)
    assert isinstance(manifest.platform, MobilePlatform)
    assert manifest.platform.gps_instrument == "gps"


def test_gas_species_property_collects_across_instruments(
    mobile_manifest_dict: dict[str, Any],
) -> None:
    manifest = Manifest.model_validate(mobile_manifest_dict)
    # ch4 + co2 from picarro; GPS/met variables must NOT appear.
    assert set(manifest.gas_species) == {"ch4", "co2"}


def test_defaults_applied(stationary_manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.model_validate(stationary_manifest_dict)
    var = manifest.instruments["picarro"].variables["co2"]
    assert var.role == "gas"
    assert var.convert is None
    assert var.circular is False
    assert var.qaqc == ()


def test_manifest_is_frozen(stationary_manifest_dict: dict[str, Any]) -> None:
    """Configs are immutable facts about a run."""
    manifest = Manifest.model_validate(stationary_manifest_dict)
    with pytest.raises(ValidationError):
        manifest.name = "mutated"  # type: ignore[misc]  # read-only: that is the assertion


def test_template_fields_extracted() -> None:
    loader = CSVLoader(
        path_template="{institution}/{campaign}/%Y/*.csv",
        time={"column": "t", "format": "unix"},  # type: ignore[arg-type]
    )
    assert loader.template_fields == ("institution", "campaign")


# ---------------------------------------------------------------------------
# Multiple path templates (heterogeneous directory/naming conventions)
# ---------------------------------------------------------------------------


def test_single_template_string_coerced_to_tuple() -> None:
    """The singular `path_template: "..."` shorthand lands in the tuple field."""
    loader = CSVLoader(
        path_template="a/%Y/*.csv",
        time={"column": "t", "format": "unix"},  # type: ignore[arg-type]
    )
    assert loader.path_templates == ("a/%Y/*.csv",)


def test_multiple_templates_accepted() -> None:
    loader = CSVLoader(
        path_templates=["layout_a/{campaign}/%Y/*.csv", "layout_b/{site}/*.csv"],
        time={"column": "t", "format": "unix"},  # type: ignore[arg-type]
    )
    assert len(loader.path_templates) == 2
    # Fields are the deduplicated union across templates, in order.
    assert loader.template_fields == ("campaign", "site")


def test_duplicate_templates_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        CSVLoader(
            path_templates=["a/*.csv", "a/*.csv"],
            time={"column": "t", "format": "unix"},  # type: ignore[arg-type]
        )


def test_any_bad_template_in_list_rejected() -> None:
    """One absolute path among several relative ones must still fail."""
    with pytest.raises(ValidationError, match="relative"):
        CSVLoader(
            path_templates=["good/%Y/*.csv", "/bad/abs/*.csv"],
            time={"column": "t", "format": "unix"},  # type: ignore[arg-type]
        )


def test_empty_template_list_rejected() -> None:
    with pytest.raises(ValidationError):
        CSVLoader(path_templates=[], time={"column": "t", "format": "unix"})  # type: ignore[arg-type]


def test_blank_template_entry_rejected() -> None:
    """A whitespace-only entry is a distinct failure from an empty *list*."""
    with pytest.raises(ValidationError, match="non-empty"):
        CSVLoader(
            path_templates=["   ", "b/%Y/*.csv"],
            time={"column": "t", "format": "unix"},  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# ICARTT revision policy
# ---------------------------------------------------------------------------


def test_icartt_revision_policy_defaults_to_latest() -> None:
    """'latest' is the safe default: ingesting all revisions double-counts."""
    from tsara.config.manifest import ICARTTLoader

    loader = ICARTTLoader(path_template="voc/%Y/*.ict")
    assert loader.revision_policy == "latest"


def test_icartt_unknown_revision_policy_rejected() -> None:
    from tsara.config.manifest import ICARTTLoader

    with pytest.raises(ValidationError, match="revision_policy"):
        ICARTTLoader(path_template="voc/%Y/*.ict", revision_policy="newest")  # type: ignore[arg-type]


def test_csv_loader_has_no_revision_policy(stationary_manifest_dict: dict[str, Any]) -> None:
    """revision_policy is ICARTT-specific; CSV filenames aren't standardized."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["loader"]["revision_policy"] = "latest"
    with pytest.raises(ValidationError, match="revision_policy"):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Typo protection: unknown keys are rejected everywhere
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["basepath"] = "/oops"  # typo'd duplicate of base_path
    with pytest.raises(ValidationError, match="basepath"):
        Manifest.model_validate(bad)


def test_unknown_nested_key_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["unitts"] = "ppm"
    with pytest.raises(ValidationError, match="unitts"):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Platform validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [("latitude", 91.0), ("latitude", -91.0), ("longitude", 181.0), ("longitude", -181.0)],
)
def test_stationary_coordinates_bounds(
    stationary_manifest_dict: dict[str, Any], field: str, value: float
) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["platform"][field] = value
    with pytest.raises(ValidationError, match=field):
        Manifest.model_validate(bad)


def test_platform_kind_discriminates(stationary_manifest_dict: dict[str, Any]) -> None:
    """A stationary platform must not accept mobile-only fields."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["platform"]["gps_instrument"] = "gps"
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


def test_mobile_missing_gps_instrument_rejected(mobile_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(mobile_manifest_dict)
    bad["platform"]["gps_instrument"] = "nonexistent"
    with pytest.raises(ValidationError, match="nonexistent"):
        Manifest.model_validate(bad)


def test_mobile_missing_lat_variable_rejected(mobile_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(mobile_manifest_dict)
    del bad["instruments"]["gps"]["variables"]["latitude"]
    with pytest.raises(ValidationError, match="latitude"):
        Manifest.model_validate(bad)


def test_mobile_wrong_gps_role_rejected(mobile_manifest_dict: dict[str, Any]) -> None:
    """Referenced GPS variable exists but carries the wrong role."""
    bad = copy.deepcopy(mobile_manifest_dict)
    bad["instruments"]["gps"]["variables"]["latitude"]["role"] = "aux"
    with pytest.raises(ValidationError, match="gps_lat"):
        Manifest.model_validate(bad)


def test_mobile_altitude_variable_accepted_when_present(
    mobile_manifest_dict: dict[str, Any],
) -> None:
    """alt_variable is optional; when given it must pass the same
    exists-and-has-the-right-role check already covered for lat/lon."""
    ok = copy.deepcopy(mobile_manifest_dict)
    ok["platform"]["alt_variable"] = "altitude"
    ok["instruments"]["gps"]["variables"]["altitude"] = {
        "column": "alt",
        "role": "gps_alt",
        "units": "m",
    }
    manifest = Manifest.model_validate(ok)
    platform = manifest.platform
    assert isinstance(platform, MobilePlatform)  # narrows the union for alt_variable
    assert platform.alt_variable == "altitude"


def test_mobile_missing_alt_variable_rejected(mobile_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(mobile_manifest_dict)
    bad["platform"]["alt_variable"] = "elevation"  # never declared on gps
    with pytest.raises(ValidationError, match="elevation"):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Variable & instrument validation
# ---------------------------------------------------------------------------


def test_circular_gas_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    """circular=true is only physically meaningful for met variables."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["circular"] = True
    with pytest.raises(ValidationError, match="circular"):
        Manifest.model_validate(bad)


def test_non_identifier_canonical_name_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4-dry"] = {
        "column": "X",
        "units": "ppm",
    }
    with pytest.raises(ValidationError, match="ch4-dry"):
        Manifest.model_validate(bad)


def test_duplicate_raw_column_within_instrument_rejected(
    stationary_manifest_dict: dict[str, Any],
) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4_copy"] = {
        "column": "CH4_dry",  # same raw column as ch4
        "units": "ppm",
    }
    with pytest.raises(ValidationError, match="CH4_dry"):
        Manifest.model_validate(bad)


def test_duplicate_canonical_name_across_instruments_rejected(
    mobile_manifest_dict: dict[str, Any],
) -> None:
    bad = copy.deepcopy(mobile_manifest_dict)
    # GPS instrument also claims to produce 'ch4' — collides with picarro's.
    bad["instruments"]["gps"]["variables"]["ch4"] = {"column": "CH4", "units": "ppb"}
    with pytest.raises(ValidationError, match="ch4"):
        Manifest.model_validate(bad)


def test_instrument_requires_at_least_one_variable(
    stationary_manifest_dict: dict[str, Any],
) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"] = {}
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Unit conversion validation
# ---------------------------------------------------------------------------


def test_identity_conversion_rejected() -> None:
    """scale=1, offset=0 must be spelled as 'no convert block at all'."""
    with pytest.raises(ValidationError, match="no-op"):
        UnitConversion(from_unit="ppb", to_unit="ppb")


def test_offset_only_conversion_allowed() -> None:
    conv = UnitConversion(from_unit="degC", to_unit="K", offset=273.15)
    assert conv.scale == 1.0


# ---------------------------------------------------------------------------
# QA/QC rule validation
# ---------------------------------------------------------------------------


def test_range_rule_requires_a_bound(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [{"kind": "range"}]
    with pytest.raises(ValidationError, match="min.*max|at least one"):
        Manifest.model_validate(bad)


def test_range_rule_min_below_max(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [
        {"kind": "range", "min": 10.0, "max": 5.0}
    ]
    with pytest.raises(ValidationError, match="min"):
        Manifest.model_validate(bad)


def test_flag_rule_requires_exactly_one_list(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [
        {"kind": "flag", "flag_column": "QC", "good_values": [0], "bad_values": [1]}
    ]
    with pytest.raises(ValidationError, match="exactly one"):
        Manifest.model_validate(bad)


def test_empty_good_values_is_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    """An empty list validates trivially and then masks the entire record.

    Nothing can be a member of an empty list, so every sample is "not good".
    The schema already refuses the analogous no-op UnitConversion; a rule that
    cannot do anything sensible belongs in the same category.
    """
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [
        {"kind": "flag", "flag_column": "QC", "good_values": []}
    ]
    with pytest.raises(ValidationError, match="would mask every sample"):
        Manifest.model_validate(bad)


def test_empty_bad_values_is_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    """The mirror case: a rule that looks active in the manifest and does nothing."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [
        {"kind": "flag", "flag_column": "QC", "bad_values": []}
    ]
    with pytest.raises(ValidationError, match="would mask nothing"):
        Manifest.model_validate(bad)


def test_unknown_qaqc_kind_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [{"kind": "magic"}]
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Uncertainty validation
# ---------------------------------------------------------------------------


def test_declared_component_all_zero_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    """absolute=0 and relative=0 declares perfect measurement for that component."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "declared", "absolute": 0.0, "relative": 0.0}
    }
    with pytest.raises(ValidationError, match="perfect"):
        Manifest.model_validate(bad)


def test_uncertainty_with_no_components_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    """Neither random nor systematic declares nothing at all."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {}
    with pytest.raises(ValidationError, match="omit"):
        Manifest.model_validate(bad)


def test_declared_random_component_parses(stationary_manifest_dict: dict[str, Any]) -> None:
    ok = copy.deepcopy(stationary_manifest_dict)
    ok["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "declared", "absolute": 0.5, "relative": 0.02}
    }
    manifest = Manifest.model_validate(ok)
    spec = manifest.instruments["picarro"].variables["ch4"].uncertainty
    assert isinstance(spec, UncertaintySpec)
    assert isinstance(spec.random, DeclaredUncertainty)
    assert spec.random.absolute == 0.5
    assert spec.systematic is None


def test_reported_component_requires_column(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "reported"}
    }
    with pytest.raises(ValidationError, match="column"):
        Manifest.model_validate(bad)


def test_reported_component_parses(stationary_manifest_dict: dict[str, Any]) -> None:
    """EM27-style per-point sigma column, e.g. a per-retrieval error column."""
    ok = copy.deepcopy(stationary_manifest_dict)
    ok["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "reported", "column": "CH4_1SIGMA"}
    }
    manifest = Manifest.model_validate(ok)
    spec = manifest.instruments["picarro"].variables["ch4"].uncertainty
    assert isinstance(spec, UncertaintySpec)
    assert isinstance(spec.random, ReportedUncertainty)
    assert spec.random.column == "CH4_1SIGMA"


def test_both_components_and_decorrelation_timescale_parse(
    stationary_manifest_dict: dict[str, Any],
) -> None:
    """A per-point random column plus a declared, constant systematic term."""
    ok = copy.deepcopy(stationary_manifest_dict)
    ok["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "reported", "column": "CH4_1SIGMA"},
        "systematic": {"mode": "declared", "relative": 0.01},
        "decorrelation_timescale": "5min",
    }
    manifest = Manifest.model_validate(ok)
    spec = manifest.instruments["picarro"].variables["ch4"].uncertainty
    assert isinstance(spec, UncertaintySpec)
    assert isinstance(spec.systematic, DeclaredUncertainty)
    assert spec.systematic.relative == 0.01
    assert spec.decorrelation_timescale == "5min"


def test_explicit_null_decorrelation_timescale_accepted(
    stationary_manifest_dict: dict[str, Any],
) -> None:
    """Explicitly passing null is equivalent to omitting the field entirely."""
    ok = copy.deepcopy(stationary_manifest_dict)
    ok["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "declared", "absolute": 0.5},
        "decorrelation_timescale": None,
    }
    manifest = Manifest.model_validate(ok)
    spec = manifest.instruments["picarro"].variables["ch4"].uncertainty
    assert spec is not None
    assert spec.decorrelation_timescale is None


def test_bad_decorrelation_timescale_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "declared", "absolute": 0.5},
        "decorrelation_timescale": "not a duration",
    }
    with pytest.raises(ValidationError, match="timedelta"):
        Manifest.model_validate(bad)


def test_unknown_uncertainty_mode_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "magic", "absolute": 0.5}
    }
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Loader / path template validation
# ---------------------------------------------------------------------------


def test_absolute_path_template_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["loader"]["path_template"] = "/abs/path/*.dat"
    with pytest.raises(ValidationError, match="relative"):
        Manifest.model_validate(bad)


def test_reserved_template_field_rejected(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["loader"]["path_template"] = "{species}/%Y/*.dat"
    with pytest.raises(ValidationError, match="reserved"):
        Manifest.model_validate(bad)


def test_csv_loader_requires_time(stationary_manifest_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    del bad["instruments"]["picarro"]["loader"]["time"]
    with pytest.raises(ValidationError, match="time"):
        Manifest.model_validate(bad)


def test_icartt_loader_forbids_time(stationary_manifest_dict: dict[str, Any]) -> None:
    """ICARTT defines its own time axis; a time block is a user error."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["loader"] = {
        "format": "icartt",
        "path_template": "voc/%Y/*.ict",
        "time": {"column": "t", "format": "unix"},
    }
    with pytest.raises(ValidationError, match="time"):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# TimeParsing: one column or several
# ---------------------------------------------------------------------------


def test_singular_column_is_shorthand_for_a_one_tuple() -> None:
    """Back-compatible spelling, mirroring path_template -> path_templates."""
    spec = TimeParsing.model_validate({"column": "EPOCH_TIME", "format": "unix"})
    assert spec.columns == ("EPOCH_TIME",)


def test_plural_columns_are_accepted() -> None:
    """The Picarro DataLog shape: DATE and TIME as separate fields."""
    spec = TimeParsing.model_validate({"columns": ["DATE", "TIME"], "format": "%Y-%m-%d %H:%M:%S"})
    assert spec.columns == ("DATE", "TIME")
    assert spec.join == " "


def test_time_columns_may_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        TimeParsing.model_validate({"columns": []})


@pytest.mark.parametrize("name", ["", "   "])
def test_blank_time_column_name_rejected(name: str) -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        TimeParsing.model_validate({"columns": [name]})


def test_duplicate_time_columns_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        TimeParsing.model_validate({"columns": ["DATE", "DATE"]})


def test_unix_format_refuses_multiple_columns() -> None:
    """Joining two numbers and reading them as epoch seconds fails silently."""
    with pytest.raises(ValidationError, match="single column of epoch"):
        TimeParsing.model_validate({"columns": ["A", "B"], "format": "unix"})


def test_unix_format_allows_one_column() -> None:
    assert TimeParsing.model_validate({"column": "t", "format": "unix"}).columns == ("t",)


# ---------------------------------------------------------------------------
# CSVLoader: headerless files
# ---------------------------------------------------------------------------


def _csv(**kwargs: Any) -> CSVLoader:
    kwargs.setdefault("path_template", "*.csv")
    kwargs.setdefault("time", {"column": "t", "format": "unix"})
    return CSVLoader(**kwargs)


def test_header_row_defaults_to_zero() -> None:
    loader = _csv()
    assert loader.header_row == 0
    assert loader.column_names is None


def test_headerless_requires_column_names() -> None:
    with pytest.raises(ValidationError, match="column_names must list"):
        _csv(header_row=None)


def test_column_names_requires_headerless() -> None:
    """Both would mean silently overriding a header that is present."""
    with pytest.raises(ValidationError, match="applies only to headerless"):
        _csv(header_row=0, column_names=["t", "ch4"])


def test_headerless_with_names_is_valid() -> None:
    loader = _csv(header_row=None, column_names=["t", "ch4"])
    assert loader.header_row is None
    assert loader.column_names == ("t", "ch4")


def test_empty_column_names_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one column"):
        _csv(header_row=None, column_names=[])


@pytest.mark.parametrize("name", ["", "  "])
def test_blank_column_name_rejected(name: str) -> None:
    with pytest.raises(ValidationError, match="non-empty column names"):
        _csv(header_row=None, column_names=["t", name])


def test_duplicate_column_names_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        _csv(header_row=None, column_names=["t", "t"])


# ---------------------------------------------------------------------------
# Headerless cross-reference check (InstrumentConfig)
# ---------------------------------------------------------------------------


def _headerless_instrument(**loader_kwargs: Any) -> dict[str, Any]:
    """An instrument reading a headerless file, with one gas variable."""
    loader: dict[str, Any] = {
        "format": "csv",
        "path_template": "*.txt",
        "header_row": None,
        "column_names": ["t", "CH4"],
        "time": {"column": "t", "format": "unix"},
    }
    loader.update(loader_kwargs)
    return {
        "loader": loader,
        "variables": {"ch4": {"column": "CH4", "role": "gas", "units": "ppb"}},
    }


def test_headerless_instrument_with_consistent_columns_is_valid() -> None:
    assert InstrumentConfig.model_validate(_headerless_instrument()) is not None


def test_headerless_unknown_variable_column_is_rejected() -> None:
    """The typo that would otherwise surface as a KeyError mid-ingestion."""
    spec = _headerless_instrument()
    spec["variables"]["ch4"]["column"] = "CH_4"
    with pytest.raises(ValidationError, match=r"'CH_4' \(variables\.ch4\.column\)"):
        InstrumentConfig.model_validate(spec)


def test_headerless_unknown_time_column_is_rejected() -> None:
    spec = _headerless_instrument()
    spec["loader"]["time"] = {"column": "Timestamp", "format": "unix"}
    with pytest.raises(ValidationError, match=r"loader\.time\.columns"):
        InstrumentConfig.model_validate(spec)


def test_headerless_unknown_reported_sigma_column_is_rejected() -> None:
    spec = _headerless_instrument()
    spec["variables"]["ch4"]["uncertainty"] = {"random": {"mode": "reported", "column": "CH4_ERR"}}
    with pytest.raises(ValidationError, match=r"uncertainty\.random\.column"):
        InstrumentConfig.model_validate(spec)


def test_headerless_declared_uncertainty_needs_no_column() -> None:
    """Only ReportedUncertainty reads a column; declared must not be checked."""
    spec = _headerless_instrument()
    spec["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "declared", "absolute": 0.7},
        "systematic": {"mode": "declared", "relative": 0.01},
    }
    assert InstrumentConfig.model_validate(spec) is not None


def test_headerless_unknown_flag_column_is_rejected() -> None:
    spec = _headerless_instrument()
    spec["variables"]["ch4"]["qaqc"] = [{"kind": "flag", "flag_column": "QC", "good_values": [0]}]
    with pytest.raises(ValidationError, match="qaqc flag_column"):
        InstrumentConfig.model_validate(spec)


def test_headerless_known_flag_column_is_accepted() -> None:
    spec = _headerless_instrument(column_names=["t", "CH4", "QC"])
    spec["variables"]["ch4"]["qaqc"] = [{"kind": "flag", "flag_column": "QC", "good_values": [0]}]
    assert InstrumentConfig.model_validate(spec) is not None


def test_non_flag_qaqc_rules_are_not_column_checked() -> None:
    """A range rule names no column, so it cannot be inconsistent with one."""
    spec = _headerless_instrument()
    spec["variables"]["ch4"]["qaqc"] = [{"kind": "range", "min": 1700.0}]
    assert InstrumentConfig.model_validate(spec) is not None


def test_headerful_csv_columns_are_not_cross_checked() -> None:
    """With a header, the authoritative name list lives in the data files."""
    spec = _headerless_instrument()
    spec["loader"]["header_row"] = 0
    del spec["loader"]["column_names"]
    spec["variables"]["ch4"]["column"] = "anything_at_all"
    assert InstrumentConfig.model_validate(spec) is not None


def test_icartt_loader_is_not_column_checked() -> None:
    """The headerless check applies to CSV only; ICARTT declares its own names."""
    spec = {
        "loader": {"format": "icartt", "path_template": "*.ict"},
        "variables": {"ch4": {"column": "CH4", "role": "gas", "units": "ppb"}},
    }
    assert InstrumentConfig.model_validate(spec) is not None


def test_explicit_null_column_names_is_the_same_as_omitting_it() -> None:
    """YAML authors write `column_names: null` as often as they omit the key."""
    assert _csv(column_names=None).column_names is None


# ---------------------------------------------------------------------------
# ParquetLoader
# ---------------------------------------------------------------------------


def test_parquet_loader_time_is_optional() -> None:
    """Parquet stores its index, so there is usually nothing to parse."""
    loader = ParquetLoader(path_template="*.parquet")
    assert loader.format == "parquet"
    assert loader.time is None


def test_parquet_loader_accepts_a_time_block() -> None:
    """For files that store time as an ordinary column instead."""
    loader = ParquetLoader.model_validate(
        {"path_template": "*.parquet", "time": {"column": "TIMESTAMP", "format": "iso8601"}}
    )
    assert loader.time is not None
    assert loader.time.columns == ("TIMESTAMP",)


def test_parquet_loader_takes_part_in_the_format_union(
    stationary_manifest_dict: dict[str, Any],
) -> None:
    """A new format must reach the manifest through the discriminator alone."""
    spec = copy.deepcopy(stationary_manifest_dict)
    spec["instruments"]["picarro"]["loader"] = {
        "format": "parquet",
        "path_template": "picarro/%Y/%m/*.parquet",
    }
    manifest = Manifest.model_validate(spec)
    assert isinstance(manifest.instruments["picarro"].loader, ParquetLoader)


def test_parquet_loader_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ParquetLoader.model_validate({"path_template": "*.parquet", "delimiter": ","})
