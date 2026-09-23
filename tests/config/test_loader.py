"""Tests for YAML loading (tsara.config.loader)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tsara import load_analysis, load_config, load_manifest
from tsara.config.loader import read_yaml
from tsara.core.exceptions import TsaraConfigError

# Spelled out rather than imported from conftest: `from tests.conftest import ...`
# only resolves when pytest is invoked from the repo root, so running this
# file by absolute path from anywhere else would fail at collection.
WriteYaml = Callable[..., Path]

# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(
    write_yaml: WriteYaml, stationary_manifest_dict: dict[str, Any]
) -> None:
    path = write_yaml(stationary_manifest_dict, "manifest.yaml")
    manifest = load_manifest(path)
    assert manifest.name == "test_site"
    assert manifest.gas_species == ("ch4", "co2")


def test_analysis_roundtrip(write_yaml: WriteYaml, analysis_dict: dict[str, Any]) -> None:
    path = write_yaml(analysis_dict, "analysis.yaml")
    analysis = load_analysis(path)
    assert analysis.baseline.quantiles == (0.05,)


def test_combined_roundtrip(
    write_yaml: WriteYaml, mobile_manifest_dict: dict[str, Any], analysis_dict: dict[str, Any]
) -> None:
    path = write_yaml({"manifest": mobile_manifest_dict, "analysis": analysis_dict}, "run.yaml")
    config = load_config(path)
    assert config.manifest.name == "test_mobile"
    assert config.analysis.output_grid.freq == "1s"


def test_example_configs_are_valid() -> None:
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


def test_stationary_example_config_is_valid() -> None:
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


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(TsaraConfigError, match="not found"):
        load_manifest(tmp_path / "nope.yaml")


def test_yaml_syntax_error_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(TsaraConfigError, match="broken.yaml"):
        load_manifest(path)


def test_non_mapping_top_level_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(TsaraConfigError, match="mapping"):
        load_manifest(path)


# ---------------------------------------------------------------------------
# Duplicate keys
# ---------------------------------------------------------------------------
#
# The YAML specification requires unique keys; PyYAML's safe_load keeps the
# LAST value silently. Measured before the loader refused it: a variable
# pasted twice under one instrument loaded as one variable reading the second
# column. These pin the refusal and its message on every level a duplicate can
# occur at, and the two shapes that must NOT be mistaken for one.


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_duplicate_top_level_key_is_refused_naming_both_lines(tmp_path: Path) -> None:
    path = _write(tmp_path, "twice.yaml", "name: first\nbase_path: /data\nname: second\n")
    with pytest.raises(TsaraConfigError) as info:
        load_manifest(path)
    message = str(info.value)
    assert "twice.yaml" in message
    assert "duplicate key 'name'" in message
    # First definition first, then the repeat: the order a reader fixes them in.
    assert "line 1" in message and "line 3" in message
    assert message.index("line 1") < message.index("line 3")


def test_a_variable_pasted_twice_under_one_instrument_is_refused(tmp_path: Path) -> None:
    """The motivating case: variable names are unique per instrument (METHODS
    §1.6), and with safe_load the mapping enforced that by keeping the second
    `ch4` and dropping the first without a word."""
    path = _write(
        tmp_path,
        "manifest.yaml",
        "instruments:\n"
        "  picarro:\n"
        "    variables:\n"
        "      ch4: {column: CH4_A}\n"
        "      co2: {column: CO2}\n"
        "      ch4: {column: CH4_B}\n",
    )
    with pytest.raises(TsaraConfigError) as info:
        load_manifest(path)
    message = str(info.value)
    assert "duplicate key 'ch4'" in message
    assert "line 4" in message and "line 6" in message


def test_a_duplicate_key_inside_a_sequence_item_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "rules.yaml", "qaqc:\n  - kind: range\n    kind: flag\n")
    with pytest.raises(TsaraConfigError, match="duplicate key 'kind'"):
        read_yaml(path)


def test_the_same_key_under_different_parents_is_not_a_duplicate(tmp_path: Path) -> None:
    """Two instruments each declaring `ch4` is the everyday case, not an error."""
    path = _write(tmp_path, "two.yaml", "picarro:\n  ch4: 1\naeris:\n  ch4: 2\n")
    assert read_yaml(path) == {"picarro": {"ch4": 1}, "aeris": {"ch4": 2}}


def test_overriding_a_merged_default_is_not_a_duplicate(tmp_path: Path) -> None:
    """A YAML merge key (`<<: *defaults`) plus an override of one merged entry
    is the legitimate use of merge keys. PyYAML implements the merge by
    flattening the anchor's entries INTO the mapping's key list, after which
    the overridden key appears twice -- so the refusal has to look before that
    flattening, and this is the test that would catch it looking after."""
    path = _write(
        tmp_path,
        "merge.yaml",
        "defaults: &defaults\n  units: ppb\n  role: gas\nch4:\n  <<: *defaults\n  units: ppm\n",
    )
    assert read_yaml(path)["ch4"] == {"units": "ppm", "role": "gas"}


def test_an_unhashable_key_is_left_to_pyyaml_to_report(tmp_path: Path) -> None:
    """A sequence used as a key (`? [a, b]`) cannot be a dict key at all. The
    duplicate check steps over it rather than crashing on `key in seen`, and
    PyYAML's own 'found unhashable key' arrives wrapped like any other error."""
    path = _write(tmp_path, "complex_key.yaml", "? [a, b]\n: value\n")
    with pytest.raises(TsaraConfigError, match="unhashable key"):
        read_yaml(path)


def test_the_reader_is_still_a_safe_loader(tmp_path: Path) -> None:
    """The refusal is a SafeLoader subclass; a config file still cannot build
    arbitrary Python objects, which is the property `safe_load` gave."""
    path = _write(tmp_path, "unsafe.yaml", "x: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(TsaraConfigError, match="Invalid YAML"):
        read_yaml(path)


def test_validation_error_names_the_file(
    write_yaml: WriteYaml, stationary_manifest_dict: dict[str, Any]
) -> None:
    bad = copy.deepcopy(stationary_manifest_dict)
    bad["platform"]["latitude"] = 200.0
    path = write_yaml(bad, "bad_manifest.yaml")
    with pytest.raises(TsaraConfigError, match="bad_manifest.yaml"):
        load_manifest(path)


def test_combined_missing_section_rejected(
    write_yaml: WriteYaml, stationary_manifest_dict: dict[str, Any]
) -> None:
    path = write_yaml({"manifest": stationary_manifest_dict}, "half.yaml")
    with pytest.raises(TsaraConfigError, match="analysis"):
        load_config(path)


# ---------------------------------------------------------------------------
# Cross-validation between manifest and analysis
# ---------------------------------------------------------------------------


def test_reference_species_must_be_declared_gas(
    write_yaml: WriteYaml, stationary_manifest_dict: dict[str, Any], analysis_dict: dict[str, Any]
) -> None:
    bad_analysis = copy.deepcopy(analysis_dict)
    bad_analysis["regression"]["reference_species"] = "sf6"  # not in manifest
    path = write_yaml({"manifest": stationary_manifest_dict, "analysis": bad_analysis}, "run.yaml")
    with pytest.raises(TsaraConfigError, match="sf6"):
        load_config(path)


def _two_methane_analyzers(manifest: dict[str, Any]) -> dict[str, Any]:
    """The picarro's methane renamed, plus a second analyzer, both field ch4."""
    both = copy.deepcopy(manifest)
    picarro = both["instruments"]["picarro"]["variables"]
    picarro["ch4_picarro"] = {**picarro.pop("ch4"), "field": "ch4"}
    both["instruments"]["lgr"] = {
        "loader": {
            "format": "csv",
            "path_template": "lgr/*.csv",
            "time": {"column": "t", "format": "unix"},
        },
        "variables": {"ch4_lgr": {"column": "CH4", "units": "ppb", "field": "ch4"}},
    }
    return both


def test_reference_species_is_checked_against_fields(
    write_yaml: WriteYaml, stationary_manifest_dict: dict[str, Any], analysis_dict: dict[str, Any]
) -> None:
    """No variable is called 'ch4', and 'ch4' is still the gas both measure."""
    manifest = _two_methane_analyzers(stationary_manifest_dict)
    path = write_yaml({"manifest": manifest, "analysis": analysis_dict}, "run.yaml")
    config = load_config(path)
    assert config.analysis.regression.reference_species == "ch4"
    assert config.manifest.gas_species == ("ch4", "co2")


def test_reference_species_is_not_a_variable_name_that_declares_another_field(
    write_yaml: WriteYaml, stationary_manifest_dict: dict[str, Any], analysis_dict: dict[str, Any]
) -> None:
    """A denominator names a gas; 'ch4_lgr' is one analyzer's record of it."""
    manifest = _two_methane_analyzers(stationary_manifest_dict)
    analysis = copy.deepcopy(analysis_dict)
    analysis["regression"]["reference_species"] = "ch4_lgr"
    path = write_yaml({"manifest": manifest, "analysis": analysis}, "run.yaml")
    with pytest.raises(TsaraConfigError, match="not the field of any role='gas' variable"):
        load_config(path)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_relative_base_path_resolved_against_manifest_dir(
    write_yaml: WriteYaml, stationary_manifest_dict: dict[str, Any], tmp_path: Path
) -> None:
    rel = copy.deepcopy(stationary_manifest_dict)
    rel["base_path"] = "../data/raw"
    path = write_yaml(rel, "manifest.yaml")
    manifest = load_manifest(path)
    assert manifest.base_path.is_absolute()
    assert manifest.base_path == (tmp_path / "../data/raw").resolve()


def test_absolute_base_path_untouched(
    write_yaml: WriteYaml, stationary_manifest_dict: dict[str, Any]
) -> None:
    manifest = load_manifest(write_yaml(stationary_manifest_dict, "manifest.yaml"))
    assert str(manifest.base_path) == "/data/raw"


def test_combined_config_resolves_relative_base_path(
    write_yaml: WriteYaml,
    stationary_manifest_dict: dict[str, Any],
    analysis_dict: dict[str, Any],
    tmp_path: Path,
) -> None:
    rel_manifest = copy.deepcopy(stationary_manifest_dict)
    rel_manifest["base_path"] = "data"
    path = write_yaml({"manifest": rel_manifest, "analysis": analysis_dict}, "run.yaml")
    config = load_config(path)
    assert config.manifest.base_path == (tmp_path / "data").resolve()


# ---------------------------------------------------------------------------
# Synthetic configs
# ---------------------------------------------------------------------------


def test_load_synthetic_reads_a_valid_file(
    synthetic_dict: dict[str, Any], write_yaml: WriteYaml
) -> None:
    from tsara.config.loader import load_synthetic

    config = load_synthetic(write_yaml(synthetic_dict, "synthetic.yaml"))
    assert config.name == "cfg"
    assert "analyzer" in config.instruments


def test_load_synthetic_reports_the_file_on_invalid_config(
    synthetic_dict: dict[str, Any], write_yaml: WriteYaml
) -> None:
    from tsara.config.loader import load_synthetic

    synthetic_dict["duration"] = "not-a-duration"
    path = write_yaml(synthetic_dict, "broken.yaml")
    with pytest.raises(TsaraConfigError, match="broken.yaml"):
        load_synthetic(path)


def test_load_synthetic_reports_a_missing_file(tmp_path: Path) -> None:
    from tsara.config.loader import load_synthetic

    with pytest.raises(TsaraConfigError, match="not found"):
        load_synthetic(tmp_path / "absent.yaml")


def test_shipped_synthetic_example_is_valid_and_generates() -> None:
    """The example config must stay runnable, not just parseable."""
    from tsara.config.loader import load_synthetic
    from tsara.synthetic import generate

    path = Path(__file__).parents[2] / "examples" / "configs" / "synthetic_example.yaml"
    config = load_synthetic(path)
    dataset = generate(config)

    # Four streams: three instruments plus the platform's own GPS.
    assert set(dataset.streams) == {"aeris", "picarro", "met", "gps"}
    # Instruments really are on different clocks.
    assert len(dataset.streams["aeris"].time) > len(dataset.streams["picarro"].time)
    assert len(dataset.ground_truth) > 0
    # Both source types fired, including nested children.
    frame = dataset.ground_truth.to_frame()
    assert set(frame["source_name"].unique()) == {"well_pad", "compressor"}
    assert frame["parent_event_id"].notna().any()


def test_shipped_bootstrap_example_is_valid_and_generates() -> None:
    """The bootstrap example must run against a supplied profile.

    It cannot be generated without one — that is the point of the design, so
    the test builds a stand-in profile exactly as the file's header documents,
    from synthetic numbers rather than a live mount.
    """
    import numpy as np
    import pandas as pd

    from tsara.config.loader import load_synthetic
    from tsara.synthetic import generate
    from tsara.synthetic.background import TsaraSyntheticError
    from tsara.synthetic.profiling import profile_series

    path = Path(__file__).parents[2] / "examples" / "configs" / "synthetic_bootstrap.yaml"
    config = load_synthetic(path)

    # Without the named profile the generator must refuse rather than invent.
    with pytest.raises(TsaraSyntheticError, match="picarro_ch4"):
        generate(config)

    rng = np.random.default_rng(0)
    n = 8000
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + rng.normal(0.0, 0.7)
    stand_in = pd.Series(
        1950.0 + ar,
        index=pd.date_range("2026-02-03T17:00:00", periods=n, freq="2s"),
    )
    profile = profile_series(stand_in, name="picarro_ch4", block_length=512)

    dataset = generate(config, profiles={"picarro_ch4": profile})
    assert set(dataset.streams) == {"picarro"}
    # One instrument, two species, one bootstrapped and one parametric.
    assert {"ch4", "co2"} <= set(dataset.streams["picarro"].data_vars)
    assert len(dataset.ground_truth) > 0
