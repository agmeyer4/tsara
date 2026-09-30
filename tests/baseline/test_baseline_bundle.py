"""Tests for saving and reloading the baseline state (tsara.baseline.bundle)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tsara.align import bin_streams_onto_cells
from tsara.baseline import baseline_state, load_state, save_state
from tsara.config.analysis import AnalysisConfig, BaselineConfig
from tsara.config.loader import read_yaml
from tsara.core.bundle import (
    BUNDLE_ANALYSIS_CONFIG,
    BUNDLE_BASELINE_DIR,
    TsaraBundleError,
    pin_time_encoding,
)
from tsara.core.support import CellBounds, declared_bounds_name
from tsara.core.timebase import SECOND_NS as SECOND


def cells(start_s: float, width_s: float, n: int, step_s: float | None = None) -> CellBounds:
    step = width_s if step_s is None else step_s
    start = (np.arange(n, dtype=np.int64) * int(step * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def make_stream(bounds: CellBounds, values: np.ndarray) -> xr.Dataset:
    n = values.size
    dataset = xr.Dataset(
        data_vars={
            "ch4": (
                "time",
                values,
                {"units": "ppb", "role": "gas", "field": "ch4", "cell_methods": "time: point"},
            ),
            "sigma_rand_ch4": (
                "time",
                np.full(n, 0.7),
                {
                    "units": "ppb",
                    "uncertainty_component": "random",
                    "uncertainty_provenance": "declared",
                },
            ),
            "sigma_sys_ch4": (
                "time",
                0.01 * np.abs(values),
                {"units": "ppb", "uncertainty_component": "systematic"},
            ),
        },
        coords={
            "time": bounds.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
            "latitude": 40.77,
            "longitude": -111.85,
        },
        attrs={"tsara_stage": "synthetic", "tsara_support_label": "mid"},
    )
    dataset["time"].attrs["bounds"] = "time_bnds"
    return dataset


@pytest.fixture()
def analysis() -> AnalysisConfig:
    return AnalysisConfig.model_validate(
        {
            "baseline": {"windows": ["2min", "10min"], "quantiles": [0.01, 0.05]},
            "regression": {"reference_species": "ch4"},
        }
    )


@pytest.fixture()
def states(analysis: AnalysisConfig) -> dict[str, xr.Dataset]:
    rng = np.random.default_rng(21)
    n = 900
    values = 1900 + rng.normal(0, 0.7, n)
    values[400:420] = np.nan
    stream = make_stream(cells(0.0, 1.0, n, step_s=2.0), values)
    other = make_stream(cells(3.0, 1.0, n, step_s=2.0), values + 5)
    return {
        "van": baseline_state(stream, instrument="van", baseline=analysis.baseline),
        "aeris": baseline_state(other, instrument="aeris", baseline=analysis.baseline),
    }


def assert_identical(a: xr.Dataset, b: xr.Dataset) -> None:
    """Every variable and coordinate exactly, every attribute exactly, arrays included.

    One documented exception: opening a file with ``decode_coords="all"``
    moves the CF ``bounds`` attribute of ``time`` into its encoding
    (:func:`~tsara.core.support.declared_bounds_name`), so that attribute is
    compared through the function that reads both places.
    """
    assert sorted(map(str, a.variables)) == sorted(map(str, b.variables))
    assert declared_bounds_name(a) == declared_bounds_name(b) == "time_bnds"
    for name in a.variables:
        assert a[name].dims == b[name].dims, name
        assert np.array_equal(a[name].values, b[name].values, equal_nan=True), name
        attrs_a = {k: v for k, v in a[name].attrs.items() if not (name == "time" and k == "bounds")}
        attrs_b = {k: v for k, v in b[name].attrs.items() if not (name == "time" and k == "bounds")}
        assert sorted(attrs_a) == sorted(attrs_b), name
        for key, value in attrs_a.items():
            got = attrs_b[key]
            if isinstance(value, np.ndarray):
                assert np.array_equal(value, got), (name, key)
            else:
                assert value == got, (name, key)
    assert a.attrs == b.attrs


def test_the_round_trip_is_exact_and_brings_the_config_back(
    tmp_path: Path, states: dict[str, xr.Dataset], analysis: AnalysisConfig
) -> None:
    target = save_state(states, tmp_path / "bundle", baseline=analysis.baseline)
    assert target == tmp_path / "bundle" / BUNDLE_BASELINE_DIR
    assert sorted(p.name for p in target.iterdir()) == [
        "aeris.nc",
        BUNDLE_ANALYSIS_CONFIG,
        "van.nc",
    ]
    back = load_state(tmp_path / "bundle")
    assert sorted(back.states) == ["aeris", "van"]
    for name, state in states.items():
        assert_identical(state, back.states[name])
        assert "time_bnds" in back.states[name].coords
    assert back.baseline == analysis.baseline
    # The baseline directory itself is also a valid path to load from.
    assert sorted(load_state(target).states) == ["aeris", "van"]


def test_a_state_rolled_before_phase_6_loses_its_unkept_empirical(
    tmp_path: Path, analysis: AnalysisConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """The reading and its enhancement copy the stream's labels; Woodruff's stays true.

    The stream is shaped as ingestion wrote a variable with no budget before
    Phase 6: no random sigma, and `empirical` promising an estimate no stage
    makes. The baseline's own sampling sigma is `empirical` and was estimated
    (METHODS §6.7), so it must come back unchanged.
    """
    stream = make_stream(cells(0.0, 1.0, 600), 1900 + np.zeros(600)).drop_vars(["sigma_rand_ch4"])
    stream["ch4"].attrs.update(
        uncertainty_provenance="mixed",
        uncertainty_provenance_random="empirical",
        uncertainty_provenance_systematic="declared",
    )
    state = baseline_state(stream, instrument="van", baseline=analysis.baseline)
    assert state["enhancement_ch4"].attrs["uncertainty_provenance_random"] == "empirical"
    save_state({"van": state}, tmp_path / "bundle", baseline=analysis.baseline)

    with caplog.at_level(logging.INFO, logger="tsara.baseline.bundle"):
        back = load_state(tmp_path / "bundle").states["van"]
    assert "promised an estimate nothing makes" in caplog.text
    assert back["ch4"].attrs["uncertainty_provenance_random"] == "unknown"
    assert back["ch4"].attrs["uncertainty_provenance"] == "mixed"
    assert back["enhancement_ch4"].attrs["uncertainty_provenance_random"] == "unknown"
    assert back["sigma_rand_baseline_ch4"].attrs["uncertainty_provenance"] == "empirical"


def test_the_per_sweep_point_records_survive_as_arrays(
    tmp_path: Path, states: dict[str, xr.Dataset]
) -> None:
    """netCDF stores a numeric array attribute; the record must come back as one."""
    save_state(states, tmp_path)
    attrs = load_state(tmp_path).states["van"]["baseline_ch4"].attrs
    assert np.array_equal(attrs["tsara_baseline_min_readings"], np.array([100, 20]))
    assert attrs["tsara_baseline_blank_fraction"].shape == (4,)
    assert attrs["tsara_baseline_too_wide_fraction"].tolist() == [0.0, 0.0]


def test_a_reloaded_state_is_joined_like_a_stream(
    tmp_path: Path, states: dict[str, xr.Dataset]
) -> None:
    """The point of persistence: what comes back is what the binner takes (METHODS §6.2)."""
    save_state(states, tmp_path)
    back = load_state(tmp_path).states["van"]
    joined = bin_streams_onto_cells({"van": back}, cells(0.0, 60.0, 30), [("van", "baseline_ch4")])
    assert joined["baseline_ch4"].dims == ("time", "baseline_window", "baseline_quantile")
    fresh = bin_streams_onto_cells(
        {"van": states["van"]}, cells(0.0, 60.0, 30), [("van", "baseline_ch4")]
    )
    assert np.array_equal(
        joined["baseline_ch4"].values, fresh["baseline_ch4"].values, equal_nan=True
    )


def test_compression_changes_the_size_and_nothing_else(
    tmp_path: Path, states: dict[str, xr.Dataset]
) -> None:
    plain = save_state(states, tmp_path / "plain")
    small = save_state(states, tmp_path / "small", compression=4)
    assert (small / "van.nc").stat().st_size < 0.7 * (plain / "van.nc").stat().st_size
    assert_identical(
        load_state(tmp_path / "plain").states["van"], load_state(tmp_path / "small").states["van"]
    )
    with pytest.raises(TsaraBundleError, match="zlib level"):
        save_state(states, tmp_path / "bad", compression=0)
    with pytest.raises(TsaraBundleError, match="zlib level"):
        save_state(states, tmp_path / "bad", compression=True)


def test_saving_without_a_config_writes_none_and_says_so(
    tmp_path: Path, states: dict[str, xr.Dataset], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="tsara.baseline.bundle"):
        target = save_state(states, tmp_path)
    assert not (target / BUNDLE_ANALYSIS_CONFIG).exists()
    assert "without their baseline configuration" in caplog.text
    assert load_state(tmp_path).baseline is None


def test_a_stale_state_file_is_removed_on_save(
    tmp_path: Path, states: dict[str, xr.Dataset], caplog: pytest.LogCaptureFixture
) -> None:
    """A bundle is the record of what ran; a file from an earlier, wider run is not."""
    save_state(states, tmp_path)
    (tmp_path / BUNDLE_BASELINE_DIR / "notes.txt").write_text("kept")
    with caplog.at_level(logging.INFO, logger="tsara.baseline.bundle"):
        save_state({"van": states["van"]}, tmp_path)
    names = sorted(p.name for p in (tmp_path / BUNDLE_BASELINE_DIR).iterdir())
    assert names == ["notes.txt", "van.nc"]
    assert "Removing stale baseline state file" in caplog.text


def test_what_is_not_a_baseline_state_is_refused_at_save_and_at_load(
    tmp_path: Path, states: dict[str, xr.Dataset]
) -> None:
    with pytest.raises(TsaraBundleError, match="No baseline states"):
        save_state({}, tmp_path)
    stream = make_stream(cells(0.0, 1.0, 10), np.arange(10.0))
    with pytest.raises(TsaraBundleError, match="tsara_stage 'synthetic'"):
        save_state({"raw": stream}, tmp_path)
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(TsaraBundleError, match="not a directory"):
        save_state(states, blocker)
    with pytest.raises(TsaraBundleError, match="not an existing directory"):
        load_state(tmp_path / "missing")
    (tmp_path / BUNDLE_BASELINE_DIR).mkdir()
    with pytest.raises(TsaraBundleError, match="holds no baseline state file"):
        load_state(tmp_path)
    pin_time_encoding(stream)
    stream.to_netcdf(tmp_path / BUNDLE_BASELINE_DIR / "raw.nc")
    with pytest.raises(TsaraBundleError, match="written by the 'synthetic' stage"):
        load_state(tmp_path)


def test_a_state_that_lost_its_bounds_is_refused_before_writing(
    tmp_path: Path, states: dict[str, xr.Dataset]
) -> None:
    broken = states["van"].drop_vars("time_bnds")
    assert (
        broken["time"].attrs["bounds"] == "time_bnds"
    )  # the attribute dangles, as after a resample
    with pytest.raises(Exception, match="bounds"):
        save_state({"van": broken}, tmp_path)


def test_a_corrupt_config_beside_the_states_is_refused_by_name(
    tmp_path: Path, states: dict[str, xr.Dataset], analysis: AnalysisConfig
) -> None:
    """Valid in every respect but a key written twice, which only the one YAML door
    refuses; a plain load would keep the last value and say nothing (config.loader)."""
    target = save_state(states, tmp_path, baseline=analysis.baseline)
    (target / BUNDLE_ANALYSIS_CONFIG).write_text(
        "baseline:\n"
        "  windows: [2min, 10min]\n"
        "  quantiles: [0.01, 0.05]\n"
        "  quantiles: [0.5]\n"
        "regression:\n"
        "  reference_species: ch4\n"
    )
    with pytest.raises(
        TsaraBundleError, match="(?s)Could not read the baseline configuration.*duplicate key"
    ):
        load_state(tmp_path)


def test_only_the_baseline_section_is_saved_beside_the_states(
    tmp_path: Path, states: dict[str, xr.Dataset], analysis: AnalysisConfig
) -> None:
    """The states were rolled from a BaselineConfig, and that is what is saved.

    The whole AnalysisConfig was saved until Phase 6, when deleting the
    detection settings made every such file unreadable (the next test): a
    stage records the section it read, so another stage's schema can change.
    """
    target = save_state(states, tmp_path, baseline=analysis.baseline)
    written = read_yaml(target / BUNDLE_ANALYSIS_CONFIG)
    assert list(written) == ["baseline"]
    assert BaselineConfig.model_validate(written["baseline"]) == analysis.baseline


def test_a_phase_5_bundle_with_the_whole_configuration_still_loads(
    tmp_path: Path, states: dict[str, xr.Dataset], analysis: AnalysisConfig
) -> None:
    """Phase 5 wrote every section, the retired ``detection`` one included.

    Validated whole, that file is now refused, since ``detection`` became
    ``plumes`` with other fields and unknown keys are refused; read by its
    baseline section alone, it loads as it was written.
    """
    target = save_state(states, tmp_path)
    (target / BUNDLE_ANALYSIS_CONFIG).write_text(
        "output_grid: null\n"
        "baseline:\n"
        "  windows: [2min, 10min]\n"
        "  quantiles: [0.01, 0.05]\n"
        "  min_readings: null\n"
        "  methods: {}\n"
        "detection:\n"
        "  enter_sigma: [3.0]\n"
        "  exit_sigma: 1.0\n"
        "  noise_estimator: diff_mad\n"
        "  noise_window: 10min\n"
        "  min_duration: 3s\n"
        "  max_internal_gap: 5s\n"
        "regression:\n"
        "  reference_species: ch4\n"
    )
    with pytest.raises(Exception, match="(?s)detection.*Extra inputs"):
        AnalysisConfig.model_validate(read_yaml(target / BUNDLE_ANALYSIS_CONFIG))
    assert load_state(tmp_path).baseline == analysis.baseline


def test_a_configuration_without_a_baseline_section_is_refused(
    tmp_path: Path, states: dict[str, xr.Dataset]
) -> None:
    target = save_state(states, tmp_path)
    (target / BUNDLE_ANALYSIS_CONFIG).write_text("regression:\n  reference_species: ch4\n")
    with pytest.raises(TsaraBundleError, match="no 'baseline' section"):
        load_state(tmp_path)
