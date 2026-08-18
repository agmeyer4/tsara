"""Tests for the analysis schema (tsara.config.analysis)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from tsara.config.analysis import (
    AlignmentConfig,
    AnalysisConfig,
    DetectionConfig,
    OutputGridConfig,
)

# ---------------------------------------------------------------------------
# Happy paths & defaults
# ---------------------------------------------------------------------------


def test_minimal_analysis_parses(analysis_dict: dict[str, Any]) -> None:
    config = AnalysisConfig.model_validate(analysis_dict)
    assert config.output_grid.freq == "1s"
    assert config.baseline.windows == ("2min", "10min")
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


def test_grid_must_resolve_shortest_window(analysis_dict: dict[str, Any]) -> None:
    """A 5-min grid cannot support a 2-min rolling window."""
    bad = copy.deepcopy(analysis_dict)
    bad["output_grid"]["freq"] = "5min"
    with pytest.raises(ValidationError, match="10x"):
        AnalysisConfig.model_validate(bad)


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


def test_mad_noise_estimator_accepted() -> None:
    """'mad' is registered (kept for comparison against the diff_mad default)."""
    config = DetectionConfig(noise_estimator="mad")
    assert config.noise_estimator == "mad"


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
