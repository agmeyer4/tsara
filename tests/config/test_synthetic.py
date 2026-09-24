"""Validation tests for the synthetic configuration schema."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from tsara.config.base import validate_signed_timedelta
from tsara.config.manifest import DeclaredUncertainty, ReportedUncertainty, SupportSpec
from tsara.config.synthetic import (
    TRUTH_PREFIX,
    AtmosphereSpec,
    BootstrapBackground,
    FieldSpec,
    InstrumentSpec,
    LognormalAmplitude,
    MeasurementSpec,
    MobileTrack,
    ParametricBackground,
    RatioSpec,
    SourceSpec,
    SyntheticConfig,
    TrueComponent,
    TrueSupport,
    TrueUncertainty,
    UniformAmplitude,
)

# ---------------------------------------------------------------------------
# Signed timedelta validator (added to config.base for inter_species_lag)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["30s", "-15s", "0s", "2min", "-1h"])
def test_signed_timedelta_accepts_both_signs(value: str) -> None:
    validate_signed_timedelta(value, field="test")


def test_signed_timedelta_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="not a valid timedelta"):
        validate_signed_timedelta("not-a-duration", field="test")


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------


def test_parametric_background_rejects_bad_period() -> None:
    with pytest.raises(ValidationError, match="diurnal_period"):
        ParametricBackground(kind="parametric", offset=1.0, diurnal_period="banana")


def test_bootstrap_background_requires_profile_key() -> None:
    with pytest.raises(ValidationError):
        BootstrapBackground(kind="bootstrap", profile="")


def test_bootstrap_background_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValidationError):
        BootstrapBackground(kind="bootstrap", profile="p", scale=0.0)


def test_whitespace_only_profile_key_rejected_at_the_field() -> None:
    # min_length=1 accepts "   "; the field validator is what rejects it, so
    # the error lands on BootstrapBackground itself rather than surfacing much
    # later as a confusing "profile not supplied" from the generator.
    with pytest.raises(ValidationError, match="must not be blank"):
        BootstrapBackground(kind="bootstrap", profile="   ")


def test_whitespace_only_profile_key_rejected_through_a_full_config(
    synthetic_dict: dict[str, Any],
) -> None:
    # Same rule, reached by nesting: field validators fire wherever the model
    # appears, which is the reason for moving the check down to the field.
    synthetic_dict["atmosphere"]["fields"]["ch4"]["background"] = {
        "kind": "bootstrap",
        "profile": "   ",
    }
    with pytest.raises(ValidationError, match="must not be blank"):
        SyntheticConfig.model_validate(synthetic_dict)


# ---------------------------------------------------------------------------
# True uncertainty
# ---------------------------------------------------------------------------


def test_true_component_rejects_empty_budget() -> None:
    with pytest.raises(ValidationError, match="injects no error"):
        TrueComponent()


def test_true_component_sigma_combines_in_quadrature() -> None:
    component = TrueComponent(absolute=3.0, relative=0.1)
    sigma = np.asarray(component.sigma(np.array([40.0])))
    assert sigma[0] == pytest.approx(math.sqrt(3.0**2 + 4.0**2))


def test_true_uncertainty_requires_a_component() -> None:
    with pytest.raises(ValidationError, match="injects nothing"):
        TrueUncertainty()


def test_decorrelation_without_random_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no random component"):
        TrueUncertainty(systematic=TrueComponent(absolute=1.0), decorrelation_timescale="10s")


def test_decorrelation_timescale_must_parse() -> None:
    with pytest.raises(ValidationError, match="decorrelation_timescale"):
        TrueUncertainty(random=TrueComponent(absolute=1.0), decorrelation_timescale="soon")


def test_to_manifest_uncertainty_maps_declared_and_reported() -> None:
    """The generator's budget and the manifest's declaration must agree."""
    spec = TrueUncertainty(
        random=TrueComponent(absolute=1.0, report_as="ch4_err"),
        systematic=TrueComponent(relative=0.02),
        decorrelation_timescale="30s",
    ).to_manifest_uncertainty()

    assert isinstance(spec.random, ReportedUncertainty)
    assert spec.random.column == "ch4_err"
    assert isinstance(spec.systematic, DeclaredUncertainty)
    assert spec.systematic.relative == pytest.approx(0.02)
    assert spec.decorrelation_timescale == "30s"


def test_to_manifest_uncertainty_omits_absent_component() -> None:
    spec = TrueUncertainty(random=TrueComponent(absolute=1.0)).to_manifest_uncertainty()
    assert isinstance(spec.random, DeclaredUncertainty)
    assert spec.systematic is None


# ---------------------------------------------------------------------------
# Fields and measurements
# ---------------------------------------------------------------------------


def test_circular_requires_met_role() -> None:
    with pytest.raises(ValidationError, match="circular=true"):
        FieldSpec(
            background=ParametricBackground(kind="parametric", offset=1.0),
            role="gas",
            circular=True,
        )


def test_a_field_cannot_take_a_gps_role() -> None:
    """Position belongs to the platform, which manufactures its own track."""
    with pytest.raises(ValidationError, match="role"):
        FieldSpec.model_validate(
            {"background": {"kind": "parametric", "offset": 40.0}, "role": "gps_lat"}
        )


def test_quantization_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        MeasurementSpec(quantization=0.0)


def test_an_empty_measurement_is_a_perfect_one_under_the_field_name() -> None:
    measurement = MeasurementSpec()
    assert (measurement.name, measurement.uncertainty, measurement.quantization) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize("name", ["ch4-dry", "2ch4", "ch4 aeris"])
def test_a_measurement_name_must_be_usable(name: str) -> None:
    with pytest.raises(ValidationError, match="valid identifier"):
        MeasurementSpec(name=name)


def _instrument(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"native_rate": "1s", "measures": {"ch4": {}}}
    base.update(overrides)
    return base


def test_instrument_rejects_bad_native_rate() -> None:
    with pytest.raises(ValidationError, match="native_rate"):
        InstrumentSpec.model_validate(_instrument(native_rate="fast"))


def test_instrument_rejects_bad_jitter() -> None:
    with pytest.raises(ValidationError, match="timestamp_jitter"):
        InstrumentSpec.model_validate(_instrument(timestamp_jitter="soonish"))


def test_instrument_rejects_a_non_identifier_field() -> None:
    with pytest.raises(ValidationError, match="valid identifier"):
        InstrumentSpec.model_validate(_instrument(measures={"2ch4": {}}))


def test_an_instrument_must_measure_something() -> None:
    with pytest.raises(ValidationError, match="measures"):
        InstrumentSpec.model_validate(_instrument(measures={}))


def test_jitter_must_be_under_half_the_native_rate() -> None:
    with pytest.raises(ValidationError, match="less than half native_rate"):
        InstrumentSpec.model_validate(_instrument(timestamp_jitter="0.6s"))


def test_jitter_under_half_rate_is_accepted() -> None:
    instrument = InstrumentSpec.model_validate(_instrument(timestamp_jitter="0.4s"))
    assert instrument.timestamp_jitter == "0.4s"


def test_a_variable_is_named_for_its_field_unless_renamed() -> None:
    instrument = InstrumentSpec.model_validate(
        _instrument(measures={"ch4": {}, "c2h6": {"name": "ethane_aeris"}})
    )
    assert instrument.variable_name("ch4") == "ch4"
    assert instrument.variable_name("c2h6") == "ethane_aeris"


def test_variable_name_of_an_unmeasured_field_raises() -> None:
    instrument = InstrumentSpec.model_validate(_instrument())
    with pytest.raises(KeyError):
        instrument.variable_name("co2")


def test_two_fields_may_not_be_written_as_one_variable() -> None:
    """A declared name can collide where field names cannot."""
    with pytest.raises(ValidationError, match="would both be written as variable 'c2h6'"):
        InstrumentSpec.model_validate(_instrument(measures={"ch4": {"name": "c2h6"}, "c2h6": {}}))


def test_a_variable_may_not_claim_the_truth_prefix() -> None:
    """The answer key's namespace is reserved for variables as for reported columns."""
    with pytest.raises(ValidationError, match="is reserved"):
        InstrumentSpec.model_validate(
            _instrument(measures={"ch4": {"name": f"{TRUTH_PREFIX}background_ch4"}})
        )


def test_reported_column_may_not_shadow_a_variable() -> None:
    spec = _instrument()
    spec["measures"]["ch4"]["uncertainty"] = {"random": {"absolute": 1.0, "report_as": "ch4"}}
    with pytest.raises(ValidationError, match="collides with a variable name"):
        InstrumentSpec.model_validate(spec)


def test_reported_column_may_not_shadow_a_renamed_variable() -> None:
    """The namespace is the variables' names, not their fields'."""
    spec = _instrument()
    spec["measures"]["ch4"] = {
        "name": "methane",
        "uncertainty": {"random": {"absolute": 1.0, "report_as": "methane"}},
    }
    with pytest.raises(ValidationError, match="collides with a variable name"):
        InstrumentSpec.model_validate(spec)


def test_reported_columns_must_be_unique() -> None:
    spec = _instrument()
    spec["measures"]["ch4"]["uncertainty"] = {
        "random": {"absolute": 1.0, "report_as": "err"},
        "systematic": {"absolute": 1.0, "report_as": "err"},
    }
    with pytest.raises(ValidationError, match="claimed by both"):
        InstrumentSpec.model_validate(spec)


def test_reported_column_may_not_claim_the_truth_prefix() -> None:
    """The answer key's namespace is reserved.

    A reported-sigma column named ``truth_*`` would be written into the same
    stream as the generator's ground-truth variables, silently corrupting the
    record every later phase is scored against.
    """
    spec = _instrument()
    spec["measures"]["ch4"]["uncertainty"] = {
        "random": {"absolute": 1.0, "report_as": f"{TRUTH_PREFIX}background_ch4"}
    }
    with pytest.raises(ValidationError, match="is reserved"):
        InstrumentSpec.model_validate(spec)


def test_instrument_without_uncertainty_passes_column_check() -> None:
    instrument = InstrumentSpec.model_validate(_instrument())
    assert instrument.measures["ch4"].uncertainty is None


def test_dropout_duration_must_parse() -> None:
    with pytest.raises(ValidationError, match="duration"):
        InstrumentSpec.model_validate(
            _instrument(dropouts={"rate_per_day": 1.0, "duration": "ages"})
        )


# ---------------------------------------------------------------------------
# Shapes and amplitudes
# ---------------------------------------------------------------------------


def test_shape_durations_must_parse() -> None:
    with pytest.raises(ValidationError, match="sigma"):
        SourceSpec.model_validate(
            {
                "rate_per_hour": 1.0,
                "shape": {"kind": "gaussian", "sigma": "wide"},
                "reference_species": "ch4",
                "amplitude": {"kind": "uniform", "low": 1.0, "high": 2.0},
            }
        )


def test_emg_requires_both_durations() -> None:
    with pytest.raises(ValidationError, match="tau"):
        SourceSpec.model_validate(
            {
                "rate_per_hour": 1.0,
                "shape": {"kind": "emg", "sigma": "10s", "tau": "slow"},
                "reference_species": "ch4",
                "amplitude": {"kind": "uniform", "low": 1.0, "high": 2.0},
            }
        )


def test_uniform_amplitude_requires_ordered_bounds() -> None:
    with pytest.raises(ValidationError, match="must be <"):
        UniformAmplitude(kind="uniform", low=5.0, high=1.0)


def test_lognormal_amplitude_requires_positive_median() -> None:
    with pytest.raises(ValidationError):
        LognormalAmplitude(kind="lognormal", median=0.0, sigma_log=0.5)


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------


def test_zero_spread_ratio_is_exact() -> None:
    mu, sigma_log = RatioSpec(mean=4.0).lognormal_parameters()
    assert sigma_log == 0.0
    assert math.exp(mu) == pytest.approx(4.0)


def test_lognormal_parameters_reproduce_mean_and_relative_spread() -> None:
    """The configured mean must be the *arithmetic* mean of realized draws."""
    spec = RatioSpec(mean=4.0, relative_spread=0.25)
    mu, sigma_log = spec.lognormal_parameters()
    draws = np.exp(mu + sigma_log * np.random.default_rng(0).normal(size=400_000))
    assert draws.mean() == pytest.approx(4.0, rel=1e-2)
    assert draws.std() / draws.mean() == pytest.approx(0.25, rel=2e-2)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_reference_species_may_not_appear_in_ratios(source_dict: dict[str, Any]) -> None:
    source_dict["ratios"]["ch4"] = {"mean": 1.0}
    with pytest.raises(ValidationError, match="must not contain the reference species"):
        SourceSpec.model_validate(source_dict)


def test_lag_species_must_be_emitted(source_dict: dict[str, Any]) -> None:
    source_dict["inter_species_lag"] = {"co2": "30s"}
    with pytest.raises(ValidationError, match="does not emit"):
        SourceSpec.model_validate(source_dict)


def test_negative_lag_is_accepted(source_dict: dict[str, Any]) -> None:
    source_dict["inter_species_lag"] = {"c2h6": "-20s"}
    source = SourceSpec.model_validate(source_dict)
    assert source.inter_species_lag["c2h6"] == "-20s"


def test_unparseable_lag_is_rejected(source_dict: dict[str, Any]) -> None:
    source_dict["inter_species_lag"] = {"c2h6": "whenever"}
    with pytest.raises(ValidationError, match="inter_species_lag"):
        SourceSpec.model_validate(source_dict)


def test_nested_ratios_may_add_species_the_parent_does_not_emit(
    source_dict: dict[str, Any],
) -> None:
    """The CLAUDE.md multi-scale case must be expressible.

    A broad landfill plume (methane only) carrying a sharp thermogenic blip
    (methane *and* ethane) is the package's own motivating example for nested
    events. The child is a distinct physical source, so it is entitled to
    chemistry the parent does not have.
    """
    source_dict["ratios"] = {}  # parent emits its reference species only
    source_dict["nested"] = {
        "probability": 0.5,
        "shape": {"kind": "gaussian", "sigma": "3s"},
        "amplitude_factor": 0.4,
        "ratios": {"c2h6": {"mean": 0.06}},
    }
    source = SourceSpec.model_validate(source_dict)
    assert source.ratios == {}
    assert source.nested is not None
    assert source.nested.ratios is not None
    assert set(source.nested.ratios) == {"c2h6"}


def test_nested_ratios_may_not_name_the_reference_species(
    source_dict: dict[str, Any],
) -> None:
    """A child's ratio to the reference species would double-count it.

    The child's amplitudes are built as ``reference_amplitude * ratio``, with
    the reference implicitly at 1.0; a declared entry would overwrite that.
    """
    source_dict["nested"] = {
        "probability": 0.5,
        "shape": {"kind": "gaussian", "sigma": "3s"},
        "amplitude_factor": 0.4,
        "ratios": {"ch4": {"mean": 2.0}},
    }
    with pytest.raises(ValidationError, match="must not contain the reference species"):
        SourceSpec.model_validate(source_dict)


def test_nested_species_must_still_be_a_declared_gas(synthetic_dict: dict[str, Any]) -> None:
    """Typo protection moved up a level, not away.

    Relaxing the parent-subset rule means a misspelled nested species is no
    longer caught by SourceSpec; the campaign-wide declared-gas check is what
    catches it instead.
    """
    synthetic_dict["atmosphere"]["sources"] = {
        "vent": {
            "rate_per_hour": 2.0,
            "shape": {"kind": "gaussian", "sigma": "20s"},
            "reference_species": "ch4",
            "amplitude": {"kind": "lognormal", "median": 100.0, "sigma_log": 0.5},
            "nested": {
                "probability": 0.5,
                "shape": {"kind": "gaussian", "sigma": "3s"},
                "amplitude_factor": 0.4,
                "ratios": {"c2h6_typo": {"mean": 0.06}},
            },
        }
    }
    with pytest.raises(ValidationError, match="undeclared species 'c2h6_typo'"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_nested_ratios_subset_is_accepted(source_dict: dict[str, Any]) -> None:
    source_dict["nested"] = {
        "probability": 0.5,
        "shape": {"kind": "gaussian", "sigma": "3s"},
        "amplitude_factor": 0.4,
        "ratios": {"c2h6": {"mean": 0.2}},
    }
    source = SourceSpec.model_validate(source_dict)
    assert source.nested is not None
    assert source.nested.ratios is not None


def test_nested_without_ratios_is_accepted(source_dict: dict[str, Any]) -> None:
    source_dict["nested"] = {
        "probability": 0.5,
        "shape": {"kind": "gaussian", "sigma": "3s"},
        "amplitude_factor": 0.4,
    }
    source = SourceSpec.model_validate(source_dict)
    assert source.nested is not None
    assert source.nested.ratios is None


# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------


def test_mobile_track_rejects_bad_gps_rate() -> None:
    with pytest.raises(ValidationError, match="gps_rate"):
        MobileTrack(kind="mobile", start_latitude=40.0, start_longitude=-111.0, gps_rate="quick")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


def test_duration_must_parse(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["duration"] = "a while"
    with pytest.raises(ValidationError, match="duration"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_two_instruments_may_measure_one_field(synthetic_dict: dict[str, Any]) -> None:
    """The point of an atmosphere: one methane, measured twice, under one name each.

    Refused until Phase 4.5, when a species belonged to its instrument and two
    of them would have been two different gases with the same name.
    """
    synthetic_dict["instruments"]["other"] = copy.deepcopy(
        synthetic_dict["instruments"]["analyzer"]
    )
    config = SyntheticConfig.model_validate(synthetic_dict)
    assert {name: list(spec.measures) for name, spec in config.instruments.items()} == {
        "analyzer": ["ch4"],
        "other": ["ch4"],
    }


def test_an_instrument_cannot_measure_an_undeclared_field(
    synthetic_dict: dict[str, Any],
) -> None:
    synthetic_dict["instruments"]["analyzer"]["measures"]["co2"] = {}
    with pytest.raises(ValidationError, match=r"measures undeclared field\(s\) \['co2'\]"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_an_unmeasured_field_is_allowed(synthetic_dict: dict[str, Any]) -> None:
    """The air may hold more than anyone measures; its truth is still queryable."""
    synthetic_dict["atmosphere"]["fields"]["co2"] = {
        "units": "ppm",
        "background": {"kind": "parametric", "offset": 420.0},
    }
    config = SyntheticConfig.model_validate(synthetic_dict)
    assert set(config.atmosphere.fields) == {"ch4", "co2"}


def test_the_atmosphere_needs_a_field(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["atmosphere"]["fields"] = {}
    with pytest.raises(ValidationError, match="fields"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_field_names_must_be_usable(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["atmosphere"]["fields"] = {
        "ch4 dry": synthetic_dict["atmosphere"]["fields"]["ch4"]
    }
    with pytest.raises(ValidationError, match="valid identifier"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_truth_resolution_must_parse() -> None:
    with pytest.raises(ValidationError, match="truth_resolution"):
        AtmosphereSpec.model_validate(
            {
                "truth_resolution": "fine",
                "fields": {"ch4": {"background": {"kind": "parametric", "offset": 1.0}}},
            }
        )


def test_source_cannot_emit_undeclared_species(
    synthetic_dict: dict[str, Any], source_dict: dict[str, Any]
) -> None:
    synthetic_dict["atmosphere"]["sources"] = {"pad": source_dict}
    with pytest.raises(ValidationError, match="undeclared species 'c2h6'"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_source_cannot_emit_a_non_gas_species(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["atmosphere"]["fields"]["temperature"] = {
        "background": {"kind": "parametric", "offset": 290.0},
        "role": "aux",
        "units": "K",
    }
    synthetic_dict["atmosphere"]["sources"] = {
        "pad": {
            "rate_per_hour": 1.0,
            "shape": {"kind": "gaussian", "sigma": "10s"},
            "reference_species": "temperature",
            "amplitude": {"kind": "uniform", "low": 1.0, "high": 2.0},
        }
    }
    with pytest.raises(ValidationError, match="not a role='gas' field"):
        SyntheticConfig.model_validate(synthetic_dict)


@pytest.mark.parametrize(
    "name",
    [
        "03_instrument_aligned/WYO_picarro",  # a path pasted in as a name
        "WYO picarro",
        "picarro.v2",
        "..",
    ],
)
def test_instrument_name_must_be_usable_as_a_filename(
    synthetic_dict: dict[str, Any], name: str
) -> None:
    """Rejected at validation, not at save.

    Instrument names become `streams/<name>.nc`. Before this rule a name
    containing a separator survived a whole `generate()` and then died in the
    netCDF backend with an error naming only a path.
    """
    synthetic_dict["instruments"] = {name: synthetic_dict["instruments"]["analyzer"]}
    with pytest.raises(ValidationError, match="valid identifier"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_gps_instrument_name_must_be_usable_as_a_filename(
    synthetic_dict: dict[str, Any],
) -> None:
    """The GPS stream is written as a file under its own name too."""
    synthetic_dict["platform"] = {
        "kind": "mobile",
        "gps_instrument": "gps/primary",
        "start_latitude": 40.0,
        "start_longitude": -111.0,
    }
    with pytest.raises(ValidationError, match="valid identifier"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_gps_instrument_name_may_not_collide(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["platform"] = {
        "kind": "mobile",
        "gps_instrument": "analyzer",
        "start_latitude": 40.0,
        "start_longitude": -111.0,
    }
    with pytest.raises(ValidationError, match="collides with a declared instrument"):
        SyntheticConfig.model_validate(synthetic_dict)


def test_mobile_with_free_gps_name_is_accepted(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["platform"] = {
        "kind": "mobile",
        "start_latitude": 40.0,
        "start_longitude": -111.0,
    }
    config = SyntheticConfig.model_validate(synthetic_dict)
    assert isinstance(config.platform, MobileTrack)


def test_gas_fields_are_listed_in_declaration_order(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["atmosphere"]["fields"]["c2h6"] = {
        "units": "ppb",
        "background": {"kind": "parametric", "offset": 2.0},
    }
    config = SyntheticConfig.model_validate(synthetic_dict)
    assert config.atmosphere.gas_fields == ("ch4", "c2h6")


def test_gas_fields_leave_out_met_and_aux(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["atmosphere"]["fields"]["wind_dir"] = {
        "role": "met",
        "circular": True,
        "background": {"kind": "parametric", "offset": 180.0},
    }
    config = SyntheticConfig.model_validate(synthetic_dict)
    assert config.atmosphere.gas_fields == ("ch4",)


def test_config_is_frozen_and_rejects_extra_keys(synthetic_dict: dict[str, Any]) -> None:
    synthetic_dict["unexpected"] = 1
    with pytest.raises(ValidationError):
        SyntheticConfig.model_validate(synthetic_dict)


# ---------------------------------------------------------------------------
# TrueSupport -> SupportSpec (the seam an exported archive travels through)
# ---------------------------------------------------------------------------


def test_support_converts_to_the_manifest_declaration() -> None:
    declared = TrueSupport(method="mean", label="start", width="15s").to_manifest_support("60s")
    assert isinstance(declared, SupportSpec)
    assert declared.method == "mean"
    assert declared.label == "start"
    assert declared.width == "15s"


def test_an_implicit_width_becomes_the_native_rate() -> None:
    """A manifest cannot say "the cadence, whatever that is" about a width it
    is declaring, so the seam resolves it to the number the generator used."""
    assert TrueSupport().to_manifest_support("2s").width == "2s"


def test_the_default_support_declares_a_centred_point_cell() -> None:
    declared = TrueSupport().to_manifest_support("1s")
    assert (declared.method, declared.label) == ("point", "mid")
