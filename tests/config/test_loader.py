"""Tests for YAML loading (tsara.config.loader)."""

from __future__ import annotations

import copy

import pytest

from tsara import load_analysis, load_config, load_manifest
from tsara.exceptions import TsaraConfigError

# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(write_yaml, stationary_manifest_dict):
    path = write_yaml(stationary_manifest_dict, "manifest.yaml")
    manifest = load_manifest(path)
    assert manifest.name == "test_site"
    assert manifest.gas_species == ("ch4", "co2")


def test_analysis_roundtrip(write_yaml, analysis_dict):
    path = write_yaml(analysis_dict, "analysis.yaml")
    analysis = load_analysis(path)
    assert analysis.baseline.quantiles == (0.05,)


def test_combined_roundtrip(write_yaml, mobile_manifest_dict, analysis_dict):
    path = write_yaml(
        {"manifest": mobile_manifest_dict, "analysis": analysis_dict}, "run.yaml"
    )
    config = load_config(path)
    assert config.manifest.name == "test_mobile"
    assert config.analysis.output_grid.freq == "1s"


def test_example_configs_are_valid():
    """The shipped examples double as documentation — they must always load.

    (The combined cross-check isn't exercised here because the examples are
    separate files; each half validates independently.)
    """
    from pathlib import Path

    examples = Path(__file__).parents[2] / "examples" / "configs"
    manifest = load_manifest(examples / "manifest_mobile_example.yaml")
    analysis = load_analysis(examples / "analysis_example.yaml")
    assert "benzene" in manifest.gas_species
    assert analysis.regression.reference_species in manifest.gas_species


def test_stationary_example_config_is_valid():
    """The stationary counterpart to the mobile example above: no GPS
    instrument, a single static coordinate applied to every sample."""
    from pathlib import Path

    from tsara.config.manifest import StationaryPlatform

    examples = Path(__file__).parents[2] / "examples" / "configs"
    manifest = load_manifest(examples / "manifest_stationary_example.yaml")
    assert isinstance(manifest.platform, StationaryPlatform)
    assert set(manifest.gas_species) == {"ch4", "co2"}


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(TsaraConfigError, match="not found"):
        load_manifest(tmp_path / "nope.yaml")


def test_yaml_syntax_error_names_the_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(TsaraConfigError, match="broken.yaml"):
        load_manifest(path)


def test_non_mapping_top_level_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(TsaraConfigError, match="mapping"):
        load_manifest(path)


def test_validation_error_names_the_file(write_yaml, stationary_manifest_dict):
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["platform"]["latitude"] = 200.0
    path = write_yaml(bad, "bad_manifest.yaml")
    with pytest.raises(TsaraConfigError, match="bad_manifest.yaml"):
        load_manifest(path)


def test_combined_missing_section_rejected(write_yaml, stationary_manifest_dict):
    path = write_yaml({"manifest": stationary_manifest_dict}, "half.yaml")
    with pytest.raises(TsaraConfigError, match="analysis"):
        load_config(path)


# ---------------------------------------------------------------------------
# Cross-validation between manifest and analysis
# ---------------------------------------------------------------------------


def test_reference_species_must_be_declared_gas(
    write_yaml, stationary_manifest_dict, analysis_dict
):
    bad_analysis = copy.deepcopy(analysis_dict)
    bad_analysis["regression"]["reference_species"] = "sf6"  # not in manifest
    path = write_yaml(
        {"manifest": stationary_manifest_dict, "analysis": bad_analysis}, "run.yaml"
    )
    with pytest.raises(TsaraConfigError, match="sf6"):
        load_config(path)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_relative_base_path_resolved_against_manifest_dir(
    write_yaml, stationary_manifest_dict, tmp_path
):
    rel = copy.deepcopy(stationary_manifest_dict)
    rel["base_path"] = "../data/raw"
    path = write_yaml(rel, "manifest.yaml")
    manifest = load_manifest(path)
    assert manifest.base_path.is_absolute()
    assert manifest.base_path == (tmp_path / "../data/raw").resolve()


def test_absolute_base_path_untouched(write_yaml, stationary_manifest_dict):
    manifest = load_manifest(write_yaml(stationary_manifest_dict, "manifest.yaml"))
    assert str(manifest.base_path) == "/data/raw"


def test_combined_config_resolves_relative_base_path(
    write_yaml, stationary_manifest_dict, analysis_dict, tmp_path
):
    rel_manifest = copy.deepcopy(stationary_manifest_dict)
    rel_manifest["base_path"] = "data"
    path = write_yaml({"manifest": rel_manifest, "analysis": analysis_dict}, "run.yaml")
    config = load_config(path)
    assert config.manifest.base_path == (tmp_path / "data").resolve()
