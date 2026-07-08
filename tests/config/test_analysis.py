"""Tests for the analysis schema (tsara.config.analysis)."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from tsara.config.analysis import AnalysisConfig, DetectionConfig, GridConfig

# ---------------------------------------------------------------------------
# Happy paths & defaults
# ---------------------------------------------------------------------------


def test_minimal_analysis_parses(analysis_dict):
    config = AnalysisConfig.model_validate(analysis_dict)
    assert config.grid.freq == "1s"
    assert config.baseline.windows == ("2min", "10min")
    # Optional stages exist with safe defaults instead of being None.
    assert config.detection.exit_mad == 1.0
    assert config.smoothing.enabled is False
    assert config.clustering.enabled is False
    assert config.regression.methods == ("ols", "odr")


def test_sweep_lists_accepted(analysis_dict):
    full = copy.deepcopy(analysis_dict)
    full["baseline"]["quantiles"] = [0.01, 0.05, 0.10]
    full["detection"] = {"enter_mads": [3.0, 5.0], "exit_mad": 1.0}
    full["smoothing"] = {"enabled": True, "cutoff_periods": ["30s", "60s"]}
    config = AnalysisConfig.model_validate(full)
    assert len(config.baseline.quantiles) == 3
    assert len(config.detection.enter_mads) == 2
    assert len(config.smoothing.cutoff_periods) == 2


# ---------------------------------------------------------------------------
# Grid validation
# ---------------------------------------------------------------------------


def test_bad_grid_freq_rejected(analysis_dict):
    bad = copy.deepcopy(analysis_dict)
    bad["grid"]["freq"] = "one second"
    with pytest.raises(ValidationError, match="timedelta"):
        AnalysisConfig.model_validate(bad)


def test_negative_duration_rejected():
    with pytest.raises(ValidationError, match="positive"):
        GridConfig(freq="-5s")


def test_grid_start_must_precede_end(analysis_dict):
    bad = copy.deepcopy(analysis_dict)
    bad["grid"]["start"] = "2025-06-02T00:00:00"
    bad["grid"]["end"] = "2025-06-01T00:00:00"
    with pytest.raises(ValidationError, match="before"):
        AnalysisConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# Baseline validation
# ---------------------------------------------------------------------------


def test_unsorted_windows_rejected(analysis_dict):
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["windows"] = ["10min", "2min"]
    with pytest.raises(ValidationError, match="increasing"):
        AnalysisConfig.model_validate(bad)


def test_duplicate_windows_rejected(analysis_dict):
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["windows"] = ["2min", "2min"]
    with pytest.raises(ValidationError, match="increasing"):
        AnalysisConfig.model_validate(bad)


@pytest.mark.parametrize("quantile", [0.0, 0.51, 0.95, 1.0, -0.05])
def test_out_of_range_quantiles_rejected(analysis_dict, quantile):
    """Baseline quantiles above the median are not backgrounds."""
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["quantiles"] = [quantile]
    with pytest.raises(ValidationError, match="quantile"):
        AnalysisConfig.model_validate(bad)


def test_median_quantile_allowed(analysis_dict):
    ok = copy.deepcopy(analysis_dict)
    ok["baseline"]["quantiles"] = [0.5]
    AnalysisConfig.model_validate(ok)  # must not raise


def test_empty_windows_rejected(analysis_dict):
    bad = copy.deepcopy(analysis_dict)
    bad["baseline"]["windows"] = []
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(bad)


def test_grid_must_resolve_shortest_window(analysis_dict):
    """A 5-min grid cannot support a 2-min rolling window."""
    bad = copy.deepcopy(analysis_dict)
    bad["grid"]["freq"] = "5min"
    with pytest.raises(ValidationError, match="10x"):
        AnalysisConfig.model_validate(bad)


# ---------------------------------------------------------------------------
# Detection validation
# ---------------------------------------------------------------------------


def test_enter_must_exceed_exit():
    """Inverted hysteresis makes event boundaries ill-defined."""
    with pytest.raises(ValidationError, match="exceed"):
        DetectionConfig(enter_mads=(2.0,), exit_mad=3.0)


def test_any_enter_below_exit_rejected():
    with pytest.raises(ValidationError, match="exceed"):
        DetectionConfig(enter_mads=(5.0, 0.5), exit_mad=1.0)


# ---------------------------------------------------------------------------
# Regression validation
# ---------------------------------------------------------------------------


def test_unknown_method_rejected(analysis_dict):
    bad = copy.deepcopy(analysis_dict)
    bad["regression"]["methods"] = ["ols", "york"]
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(bad)


def test_duplicate_methods_rejected(analysis_dict):
    bad = copy.deepcopy(analysis_dict)
    bad["regression"]["methods"] = ["ols", "ols"]
    with pytest.raises(ValidationError, match="duplicate"):
        AnalysisConfig.model_validate(bad)


def test_min_points_floor(analysis_dict):
    """A 2-point regression always fits perfectly — meaningless."""
    bad = copy.deepcopy(analysis_dict)
    bad["regression"]["min_points"] = 2
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(bad)


def test_unknown_key_rejected(analysis_dict):
    bad = copy.deepcopy(analysis_dict)
    bad["baselines"] = bad.pop("baseline")  # plural typo
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(bad)
