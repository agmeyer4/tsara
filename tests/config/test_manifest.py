"""Tests for the manifest schema (tsara.config.manifest)."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from tsara.config.manifest import (
    CSVLoader,
    DeclaredUncertainty,
    Manifest,
    MobilePlatform,
    ReportedUncertainty,
    StationaryPlatform,
    UncertaintySpec,
    UnitConversion,
)

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_valid_stationary_manifest_parses(stationary_manifest_dict):
    manifest = Manifest.model_validate(stationary_manifest_dict)
    assert manifest.name == "test_site"
    assert isinstance(manifest.platform, StationaryPlatform)
    assert manifest.platform.latitude == pytest.approx(40.766)
    # Discriminated union picked the right loader class.
    assert isinstance(manifest.instruments["picarro"].loader, CSVLoader)


def test_valid_mobile_manifest_parses(mobile_manifest_dict):
    manifest = Manifest.model_validate(mobile_manifest_dict)
    assert isinstance(manifest.platform, MobilePlatform)
    assert manifest.platform.gps_instrument == "gps"


def test_gas_species_property_collects_across_instruments(mobile_manifest_dict):
    manifest = Manifest.model_validate(mobile_manifest_dict)
    # ch4 + co2 from picarro; GPS/met variables must NOT appear.
    assert set(manifest.gas_species) == {"ch4", "co2"}


def test_defaults_applied(stationary_manifest_dict):
    manifest = Manifest.model_validate(stationary_manifest_dict)
    var = manifest.instruments["picarro"].variables["co2"]
    assert var.role == "gas"
    assert var.convert is None
    assert var.circular is False
    assert var.qaqc == ()


def test_manifest_is_frozen(stationary_manifest_dict):
    """Configs are immutable facts about a run."""
    manifest = Manifest.model_validate(stationary_manifest_dict)
    with pytest.raises(ValidationError):
        manifest.name = "mutated"


def test_template_fields_extracted():
    loader = CSVLoader(
        path_template="{institution}/{campaign}/%Y/*.csv",
        time={"column": "t", "format": "unix"},
    )
    assert loader.template_fields == ("institution", "campaign")


# ---------------------------------------------------------------------------
# Multiple path templates (heterogeneous directory/naming conventions)
# ---------------------------------------------------------------------------


def test_single_template_string_coerced_to_tuple():
    """The singular `path_template: "..."` shorthand lands in the tuple field."""
    loader = CSVLoader(
        path_template="a/%Y/*.csv",
        time={"column": "t", "format": "unix"},
    )
    assert loader.path_templates == ("a/%Y/*.csv",)


def test_multiple_templates_accepted():
    loader = CSVLoader(
        path_templates=["layout_a/{campaign}/%Y/*.csv", "layout_b/{site}/*.csv"],
        time={"column": "t", "format": "unix"},
    )
    assert len(loader.path_templates) == 2
    # Fields are the deduplicated union across templates, in order.
    assert loader.template_fields == ("campaign", "site")


def test_duplicate_templates_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        CSVLoader(
            path_templates=["a/*.csv", "a/*.csv"],
            time={"column": "t", "format": "unix"},
        )


def test_any_bad_template_in_list_rejected():
    """One absolute path among several relative ones must still fail."""
    with pytest.raises(ValidationError, match="relative"):
        CSVLoader(
            path_templates=["good/%Y/*.csv", "/bad/abs/*.csv"],
            time={"column": "t", "format": "unix"},
        )


def test_empty_template_list_rejected():
    with pytest.raises(ValidationError):
        CSVLoader(path_templates=[], time={"column": "t", "format": "unix"})


def test_blank_template_entry_rejected():
    """A whitespace-only entry is a distinct failure from an empty *list*."""
    with pytest.raises(ValidationError, match="non-empty"):
        CSVLoader(path_templates=["   ", "b/%Y/*.csv"], time={"column": "t", "format": "unix"})


# ---------------------------------------------------------------------------
# ICARTT revision policy
# ---------------------------------------------------------------------------


def test_icartt_revision_policy_defaults_to_latest():
    """'latest' is the safe default: ingesting all revisions double-counts."""
    from tsara.config.manifest import ICARTTLoader

    loader = ICARTTLoader(path_template="voc/%Y/*.ict")
    assert loader.revision_policy == "latest"


def test_icartt_unknown_revision_policy_rejected():
    from tsara.config.manifest import ICARTTLoader

    with pytest.raises(ValidationError, match="revision_policy"):
        ICARTTLoader(path_template="voc/%Y/*.ict", revision_policy="newest")


def test_csv_loader_has_no_revision_policy(stationary_manifest_dict):
    """revision_policy is ICARTT-specific; CSV filenames aren't standardized."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["loader"]["revision_policy"] = "latest"
    with pytest.raises(ValidationError, match="revision_policy"):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Typo protection: unknown keys are rejected everywhere
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["basepath"] = "/oops"  # typo'd duplicate of base_path
    with pytest.raises(ValidationError, match="basepath"):
        Manifest.model_validate(bad)


def test_unknown_nested_key_rejected(stationary_manifest_dict):
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
def test_stationary_coordinates_bounds(stationary_manifest_dict, field, value):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["platform"][field] = value
    with pytest.raises(ValidationError, match=field):
        Manifest.model_validate(bad)


def test_platform_kind_discriminates(stationary_manifest_dict):
    """A stationary platform must not accept mobile-only fields."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["platform"]["gps_instrument"] = "gps"
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


def test_mobile_missing_gps_instrument_rejected(mobile_manifest_dict):
    bad = copy.deepcopy(mobile_manifest_dict)
    bad["platform"]["gps_instrument"] = "nonexistent"
    with pytest.raises(ValidationError, match="nonexistent"):
        Manifest.model_validate(bad)


def test_mobile_missing_lat_variable_rejected(mobile_manifest_dict):
    bad = copy.deepcopy(mobile_manifest_dict)
    del bad["instruments"]["gps"]["variables"]["latitude"]
    with pytest.raises(ValidationError, match="latitude"):
        Manifest.model_validate(bad)


def test_mobile_wrong_gps_role_rejected(mobile_manifest_dict):
    """Referenced GPS variable exists but carries the wrong role."""
    bad = copy.deepcopy(mobile_manifest_dict)
    bad["instruments"]["gps"]["variables"]["latitude"]["role"] = "aux"
    with pytest.raises(ValidationError, match="gps_lat"):
        Manifest.model_validate(bad)


def test_mobile_altitude_variable_accepted_when_present(mobile_manifest_dict):
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
    assert manifest.platform.alt_variable == "altitude"


def test_mobile_missing_alt_variable_rejected(mobile_manifest_dict):
    bad = copy.deepcopy(mobile_manifest_dict)
    bad["platform"]["alt_variable"] = "elevation"  # never declared on gps
    with pytest.raises(ValidationError, match="elevation"):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Variable & instrument validation
# ---------------------------------------------------------------------------


def test_circular_gas_rejected(stationary_manifest_dict):
    """circular=true is only physically meaningful for met variables."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["circular"] = True
    with pytest.raises(ValidationError, match="circular"):
        Manifest.model_validate(bad)


def test_non_identifier_canonical_name_rejected(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4-dry"] = {
        "column": "X",
        "units": "ppm",
    }
    with pytest.raises(ValidationError, match="ch4-dry"):
        Manifest.model_validate(bad)


def test_duplicate_raw_column_within_instrument_rejected(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4_copy"] = {
        "column": "CH4_dry",  # same raw column as ch4
        "units": "ppm",
    }
    with pytest.raises(ValidationError, match="CH4_dry"):
        Manifest.model_validate(bad)


def test_duplicate_canonical_name_across_instruments_rejected(mobile_manifest_dict):
    bad = copy.deepcopy(mobile_manifest_dict)
    # GPS instrument also claims to produce 'ch4' — collides with picarro's.
    bad["instruments"]["gps"]["variables"]["ch4"] = {"column": "CH4", "units": "ppb"}
    with pytest.raises(ValidationError, match="ch4"):
        Manifest.model_validate(bad)


def test_instrument_requires_at_least_one_variable(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"] = {}
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Unit conversion validation
# ---------------------------------------------------------------------------


def test_identity_conversion_rejected():
    """scale=1, offset=0 must be spelled as 'no convert block at all'."""
    with pytest.raises(ValidationError, match="no-op"):
        UnitConversion(from_unit="ppb", to_unit="ppb")


def test_offset_only_conversion_allowed():
    conv = UnitConversion(from_unit="degC", to_unit="K", offset=273.15)
    assert conv.scale == 1.0


# ---------------------------------------------------------------------------
# QA/QC rule validation
# ---------------------------------------------------------------------------


def test_range_rule_requires_a_bound(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [{"kind": "range"}]
    with pytest.raises(ValidationError, match="min.*max|at least one"):
        Manifest.model_validate(bad)


def test_range_rule_min_below_max(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [
        {"kind": "range", "min": 10.0, "max": 5.0}
    ]
    with pytest.raises(ValidationError, match="min"):
        Manifest.model_validate(bad)


def test_flag_rule_requires_exactly_one_list(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [
        {"kind": "flag", "flag_column": "QC", "good_values": [0], "bad_values": [1]}
    ]
    with pytest.raises(ValidationError, match="exactly one"):
        Manifest.model_validate(bad)


def test_spike_rule_validates_window(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [
        {"kind": "spike", "window": "not-a-duration"}
    ]
    with pytest.raises(ValidationError, match="timedelta"):
        Manifest.model_validate(bad)


def test_unknown_qaqc_kind_rejected(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["qaqc"] = [{"kind": "magic"}]
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Uncertainty validation
# ---------------------------------------------------------------------------


def test_declared_component_all_zero_rejected(stationary_manifest_dict):
    """absolute=0 and relative=0 declares perfect measurement for that component."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "declared", "absolute": 0.0, "relative": 0.0}
    }
    with pytest.raises(ValidationError, match="perfect"):
        Manifest.model_validate(bad)


def test_uncertainty_with_no_components_rejected(stationary_manifest_dict):
    """Neither random nor systematic declares nothing at all."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {}
    with pytest.raises(ValidationError, match="omit"):
        Manifest.model_validate(bad)


def test_declared_random_component_parses(stationary_manifest_dict):
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


def test_reported_component_requires_column(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "reported"}
    }
    with pytest.raises(ValidationError, match="column"):
        Manifest.model_validate(bad)


def test_reported_component_parses(stationary_manifest_dict):
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


def test_both_components_and_decorrelation_timescale_parse(stationary_manifest_dict):
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


def test_explicit_null_decorrelation_timescale_accepted(stationary_manifest_dict):
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


def test_bad_decorrelation_timescale_rejected(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "declared", "absolute": 0.5},
        "decorrelation_timescale": "not a duration",
    }
    with pytest.raises(ValidationError, match="timedelta"):
        Manifest.model_validate(bad)


def test_unknown_uncertainty_mode_rejected(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["variables"]["ch4"]["uncertainty"] = {
        "random": {"mode": "magic", "absolute": 0.5}
    }
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad)


# ---------------------------------------------------------------------------
# Loader / path template validation
# ---------------------------------------------------------------------------


def test_absolute_path_template_rejected(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["loader"]["path_template"] = "/abs/path/*.dat"
    with pytest.raises(ValidationError, match="relative"):
        Manifest.model_validate(bad)


def test_reserved_template_field_rejected(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["loader"]["path_template"] = "{species}/%Y/*.dat"
    with pytest.raises(ValidationError, match="reserved"):
        Manifest.model_validate(bad)


def test_csv_loader_requires_time(stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    del bad["instruments"]["picarro"]["loader"]["time"]
    with pytest.raises(ValidationError, match="time"):
        Manifest.model_validate(bad)


def test_icartt_loader_forbids_time(stationary_manifest_dict):
    """ICARTT defines its own time axis; a time block is a user error."""
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["instruments"]["picarro"]["loader"] = {
        "format": "icartt",
        "path_template": "voc/%Y/*.ict",
        "time": {"column": "t", "format": "unix"},
    }
    with pytest.raises(ValidationError, match="time"):
        Manifest.model_validate(bad)
