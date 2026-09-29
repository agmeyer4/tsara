"""Tests for the analysis schema (tsara.config.analysis)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from tsara.config.analysis import (
    AlignmentConfig,
    AnalysisConfig,
    BaselineConfig,
    ConstantMethod,
    DetectionConfig,
    FromFieldMethod,
    OutputGridConfig,
    RollingQuantileMethod,
)

# ---------------------------------------------------------------------------
# Happy paths & defaults
# ---------------------------------------------------------------------------


def test_minimal_analysis_parses(analysis_dict: dict[str, Any]) -> None:
    config = AnalysisConfig.model_validate(analysis_dict)
    assert config.output_grid is not None
    assert config.output_grid.freq == "1s"
    assert config.baseline.windows == ("2min", "10min")
    assert config.baseline.min_readings is None
    assert config.baseline.methods == {}
    # Optional stages exist with safe defaults instead of being None.
    assert config.alignment.max_interp_gap == "10s"
    assert config.detection.exit_sigma == 1.0
    assert config.detection.noise_estimator == "diff_mad"
    assert config.smoothing.enabled is False
    assert config.clustering.enabled is False
    assert config.regression.methods == ("ols", "york")


def test_sweep_lists_accepted(analysis_dict: dict[str, Any]) -> None:
    full = copy.deepcopy(analysis_dict)
    full["baseline"]["quantiles"] = [0.01, 0.05, 0.10]
    full["detection"] = {"enter_sigma": [3.0, 5.0], "exit_sigma": 1.0}
    full["smoothing"] = {"enabled": True, "cutoff_periods": ["30s", "60s"]}
    config = AnalysisConfig.model_validate(full)
    assert len(config.baseline.quantiles) == 3
    assert len(config.detection.enter_sigma) == 2
    assert len(config.smoothing.cutoff_periods) == 2


# ---------------------------------------------------------------------------
# Output grid validation
# ---------------------------------------------------------------------------


def test_bad_grid_freq_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["output_grid"]["freq"] = "one second"
    with pytest.raises(ValidationError, match="timedelta"):
        AnalysisConfig.model_validate(bad)


def test_negative_duration_rejected() -> None:
    with pytest.raises(ValidationError, match="positive"):
        OutputGridConfig(freq="-5s")


def test_grid_start_must_precede_end(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["output_grid"]["start"] = "2025-06-02T00:00:00"
    bad["output_grid"]["end"] = "2025-06-01T00:00:00"
    with pytest.raises(ValidationError, match="before"):
        AnalysisConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# Alignment (aux-field interpolation guard) validation
# ---------------------------------------------------------------------------


def test_alignment_defaults_present_without_being_specified(analysis_dict: dict[str, Any]) -> None:
    """alignment is optional-but-present, like smoothing/clustering/detection."""
    config = AnalysisConfig.model_validate(analysis_dict)
    assert config.alignment.max_interp_gap == "10s"


def test_bad_alignment_gap_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["alignment"] = {"max_interp_gap": "not a duration"}
    with pytest.raises(ValidationError, match="timedelta"):
        AnalysisConfig.model_validate(bad)


def test_negative_alignment_gap_rejected() -> None:
    with pytest.raises(ValidationError, match="positive"):
        AlignmentConfig(max_interp_gap="-10s")


# ---------------------------------------------------------------------------
# Baseline validation
# ---------------------------------------------------------------------------


def test_unsorted_windows_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["windows"] = ["10min", "2min"]
    with pytest.raises(ValidationError, match="increasing"):
        AnalysisConfig.model_validate(bad)


def test_duplicate_windows_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["windows"] = ["2min", "2min"]
    with pytest.raises(ValidationError, match="increasing"):
        AnalysisConfig.model_validate(bad)


def test_duplicate_quantiles_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["quantiles"] = [0.05, 0.05]
    with pytest.raises(ValidationError, match="duplicate"):
        AnalysisConfig.model_validate(bad)


@pytest.mark.parametrize("quantile", [0.0, 0.51, 0.95, 1.0, -0.05])
def test_out_of_range_quantiles_rejected(analysis_dict: dict[str, Any], quantile: float) -> None:
    """Baseline quantiles above the median are not backgrounds."""
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["quantiles"] = [quantile]
    with pytest.raises(ValidationError, match="quantile"):
        AnalysisConfig.model_validate(bad)


def test_median_quantile_allowed(analysis_dict: dict[str, Any]) -> None:
    ok = copy.deepcopy(analysis_dict)
    ok["baseline"]["quantiles"] = [0.5]
    AnalysisConfig.model_validate(ok)  # must not raise


def test_empty_windows_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["windows"] = []
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(bad)


def test_the_grid_is_optional_and_absent_by_default(analysis_dict: dict[str, Any]) -> None:
    """The rolling state lives per stream at native rate (METHODS §6.2), so a run
    that exports nothing on a tiling declares no grid, and the field is None
    rather than a disabled stage: a grid has no default period to fall back on."""
    without = copy.deepcopy(analysis_dict)
    del without["output_grid"]
    config = AnalysisConfig.model_validate(without)
    assert config.output_grid is None


def test_a_coarse_grid_no_longer_constrains_the_shortest_window(
    analysis_dict: dict[str, Any],
) -> None:
    """The baseline never rolled over the grid, so the grid cannot invalidate it.

    Before Phase 5 a 5 min grid with a 2 min window was refused with a message
    about the quantile having nothing to chew on -- a rule about a product the
    baseline does not use. Validity is now a count of readings in the window,
    asked of the stream itself (METHODS §6.4).
    """
    ok = copy.deepcopy(analysis_dict)
    ok["output_grid"]["freq"] = "5min"
    config = AnalysisConfig.model_validate(ok)
    assert config.output_grid is not None
    assert config.output_grid.freq == "5min"


# ---------------------------------------------------------------------------
# Detection validation
# ---------------------------------------------------------------------------


def test_enter_must_exceed_exit() -> None:
    """Inverted hysteresis makes event boundaries ill-defined."""
    with pytest.raises(ValidationError, match="exceed"):
        DetectionConfig(enter_sigma=(2.0,), exit_sigma=3.0)


def test_any_enter_below_exit_rejected() -> None:
    with pytest.raises(ValidationError, match="exceed"):
        DetectionConfig(enter_sigma=(5.0, 0.5), exit_sigma=1.0)


def test_unknown_noise_estimator_rejected() -> None:
    with pytest.raises(ValidationError):
        DetectionConfig(noise_estimator="qn")  # type: ignore[arg-type]  # not a registered name


def test_mad_noise_estimator_rejected() -> None:
    """'mad' was measured and rejected (METHODS §2.5): never better than diff_mad."""
    with pytest.raises(ValidationError):
        DetectionConfig(noise_estimator="mad")  # type: ignore[arg-type]  # rejected 2026-09-29


# ---------------------------------------------------------------------------
# Regression validation
# ---------------------------------------------------------------------------


def test_york_is_the_default_preferred_method(analysis_dict: dict[str, Any]) -> None:
    """York (errors-in-both-axes, correlation-aware) is default; odr is opt-in."""
    config = AnalysisConfig.model_validate(analysis_dict)
    assert config.regression.methods == ("ols", "york")
    assert "odr" not in config.regression.methods


def test_all_three_methods_accepted(analysis_dict: dict[str, Any]) -> None:
    ok = copy.deepcopy(analysis_dict)
    ok["regression"]["methods"] = ["ols", "york", "odr"]
    config = AnalysisConfig.model_validate(ok)
    assert config.regression.methods == ("ols", "york", "odr")


def test_unknown_method_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["regression"]["methods"] = ["ols", "wls"]  # 'wls' is not a registered estimator
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(bad)


def test_duplicate_methods_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["regression"]["methods"] = ["ols", "ols"]
    with pytest.raises(ValidationError, match="duplicate"):
        AnalysisConfig.model_validate(bad)


def test_min_points_floor(analysis_dict: dict[str, Any]) -> None:
    """A 2-point regression always fits perfectly — meaningless."""
    bad = copy.deepcopy(analysis_dict)
    bad["regression"]["min_points"] = 2
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(bad)


def test_unknown_key_rejected(analysis_dict: dict[str, Any]) -> None:
    bad = copy.deepcopy(analysis_dict)
    bad["baselines"] = bad.pop("baseline")  # plural typo
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# Alignment: the copy policy (METHODS §11.2.4)
# ---------------------------------------------------------------------------


def test_finer_support_defaults_to_refuse(analysis_dict: dict[str, Any]) -> None:
    """The interpolation rule's guarantee for a step function is the default."""
    config = AnalysisConfig.model_validate(analysis_dict)
    assert config.alignment.finer_support == "refuse"


def test_finer_support_may_be_allowed_by_name() -> None:
    assert AlignmentConfig(finer_support="allow").finer_support == "allow"


def test_finer_support_has_no_third_value() -> None:
    """A copy is refused or made; there is no 'warn' that makes it silently."""
    with pytest.raises(ValidationError, match="finer_support"):
        AlignmentConfig(finer_support="warn")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Baseline: the count rule (METHODS §6.4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantile", "expected"),
    [(0.01, 100), (0.05, 20), (0.10, 10), (0.03, 34), (0.07, 15), (0.5, 2)],
)
def test_min_readings_defaults_to_one_over_the_quantile(quantile: float, expected: int) -> None:
    """ceil(1/q) per quantile, and a whole number where 1/q is one in arithmetic.

    1/0.05 is 20.000000000000004 in float64; a naive ceiling would demand 21
    readings of a window that needs 20.
    """
    config = BaselineConfig(windows=("2min",), quantiles=(quantile,))
    assert config.min_readings_for(quantile) == expected


def test_an_explicit_min_readings_applies_to_every_quantile() -> None:
    config = BaselineConfig(windows=("2min",), quantiles=(0.01, 0.05), min_readings=30)
    assert config.min_readings_for(0.01) == 30
    assert config.min_readings_for(0.05) == 30


def test_min_readings_below_two_rejected() -> None:
    """A quantile of one reading is that reading."""
    with pytest.raises(ValidationError, match="min_readings"):
        BaselineConfig(windows=("2min",), quantiles=(0.05,), min_readings=1)


def test_min_valid_fraction_is_gone(analysis_dict: dict[str, Any]) -> None:
    """The retired knob is an unknown key now, refused like any typo (METHODS §6.4)."""
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["min_valid_fraction"] = 0.5
    with pytest.raises(ValidationError, match="min_valid_fraction"):
        AnalysisConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# Baseline: the method per variable (METHODS §6.5)
# ---------------------------------------------------------------------------


def test_baseline_methods_dispatch_on_the_method_key(analysis_dict: dict[str, Any]) -> None:
    full = copy.deepcopy(analysis_dict)
    full["baseline"]["methods"] = {
        "iwas.benzene": {"method": "from_field", "instrument": "ptr"},
        "iwas.toluene": {"method": "constant", "value": 0.0},
        "ptr.benzene": {"method": "rolling_quantile"},
    }
    config = AnalysisConfig.model_validate(full)
    assert config.baseline.method_for("iwas", "benzene") == FromFieldMethod(instrument="ptr")
    assert config.baseline.method_for("iwas", "toluene") == ConstantMethod(value=0.0)
    assert config.baseline.method_for("ptr", "benzene") == RollingQuantileMethod()


def test_a_variable_not_named_in_methods_uses_the_rolling_quantile() -> None:
    config = BaselineConfig(windows=("2min",), quantiles=(0.05,))
    assert config.method_for("picarro", "ch4") == RollingQuantileMethod()


@pytest.mark.parametrize("key", ["ch4", "picarro.", ".ch4", "a.b.c", "pic arro.ch4", "1x.ch4"])
def test_method_keys_must_be_instrument_dot_variable(key: str) -> None:
    """The shape is checked here; whether the names exist is the combined config's job."""
    with pytest.raises(ValidationError, match="'<instrument>.<variable>'"):
        _baseline({key: {"method": "constant", "value": 0}})


def test_an_unregistered_baseline_method_is_rejected() -> None:
    with pytest.raises(ValidationError, match="method"):
        _baseline({"a.b": {"method": "lowess"}})


def test_from_field_needs_an_instrument_and_constant_needs_a_value() -> None:
    with pytest.raises(ValidationError, match="instrument"):
        _baseline({"a.b": {"method": "from_field"}})
    with pytest.raises(ValidationError, match="value"):
        _baseline({"a.b": {"method": "constant"}})


def test_a_method_rejects_a_key_of_another_method() -> None:
    """A constant with an `instrument` is a pasted-together entry, refused like any unknown key."""
    with pytest.raises(ValidationError, match="instrument"):
        _baseline({"a.b": {"method": "constant", "value": 0.0, "instrument": "c"}})


def _baseline(methods: dict[str, dict[str, Any]]) -> BaselineConfig:
    """Validate a one-window baseline config carrying ``methods`` as YAML would deliver it."""
    return BaselineConfig.model_validate(
        {"windows": ["2min"], "quantiles": [0.05], "methods": methods}
    )
