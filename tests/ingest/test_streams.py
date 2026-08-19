"""Tests for stream assembly.

The load-bearing property is substitutability: a stream built from an
archive must be shaped like one the synthetic generator manufactures, since
every later phase consumes both through one code path and synthetic truth is
the only correctness arbiter available. Several tests below therefore assert
against :mod:`tsara.core.naming` rather than against literal strings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.config.manifest import (
    InstrumentConfig,
    MobilePlatform,
    StationaryPlatform,
)
from tsara.core.naming import sigma_rand_name, sigma_sys_name
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.streams import build_stream

SITE = StationaryPlatform(latitude=40.77, longitude=-111.85, altitude_m=1300.0)
MOBILE = MobilePlatform(gps_instrument="gps", lat_variable="latitude", lon_variable="longitude")


def _frame(*, n: int = 4, **columns: Any) -> pd.DataFrame:
    index = pd.date_range("2026-02-03 17:00", periods=n, freq="2s", name="time")
    base = {"CH4_dry": np.linspace(1.9, 1.93, n)}
    base.update(columns)
    return pd.DataFrame(base, index=index)


def _instrument(**variables: Any) -> InstrumentConfig:
    """Build an instrument reading a CSV, with the given variable configs."""
    if not variables:
        variables = {"ch4": {"column": "CH4_dry", "role": "gas", "units": "ppm"}}
    return InstrumentConfig.model_validate(
        {
            "loader": {
                "format": "csv",
                "path_template": "*.dat",
                "time": {"column": "t", "format": "unix"},
            },
            "variables": variables,
        }
    )


def _build(frame: pd.DataFrame, instrument: InstrumentConfig, **kw: Any) -> Any:
    kw.setdefault("name", "picarro")
    kw.setdefault("platform", SITE)
    return build_stream(frame, instrument, **kw)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_variables_take_their_canonical_names() -> None:
    """The raw column name never reaches the dataset."""
    stream = _build(_frame(), _instrument())
    assert "ch4" in stream.data_vars
    assert "CH4_dry" not in stream.data_vars
    assert stream["ch4"].attrs["raw_column"] == "CH4_dry"


def test_time_is_the_dimension() -> None:
    stream = _build(_frame(), _instrument())
    assert stream["ch4"].dims == ("time",)
    assert stream.sizes["time"] == 4
    assert stream["time"].dtype == "datetime64[ns]"


def test_values_are_converted_to_canonical_units() -> None:
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "units": "ppm",
            "convert": {"from_unit": "ppm", "to_unit": "ppb", "scale": 1000.0},
        }
    )
    stream = _build(_frame(), instrument)
    assert stream["ch4"].values[0] == pytest.approx(1900.0)
    assert stream["ch4"].attrs["units"] == "ppb"


def test_units_attr_describes_the_stored_numbers() -> None:
    """Not the raw file's units, which no longer describe anything stored."""
    stream = _build(_frame(), _instrument())
    assert stream["ch4"].attrs["units"] == "ppm"


def test_role_and_circular_are_recorded() -> None:
    instrument = _instrument(
        wdir={"column": "WD", "role": "met", "units": "degrees", "circular": True}
    )
    stream = _build(_frame(WD=np.array([10.0, 20.0, 350.0, 5.0])), instrument)
    assert stream["wdir"].attrs["role"] == "met"
    # netCDF has no boolean type; 0/1 matches the generator's convention.
    assert stream["wdir"].attrs["circular"] == 1


def test_many_species_scale_without_code_changes() -> None:
    """Species are data, not code — a 40-VOC instrument is a YAML edit."""
    columns: dict[str, Any] = {f"VOC{i}": np.zeros(4) for i in range(40)}
    variables = {f"voc{i}": {"column": f"VOC{i}", "role": "gas", "units": "ppb"} for i in range(40)}
    stream = _build(_frame(**columns), _instrument(**variables))
    assert len([v for v in stream.data_vars if str(v).startswith("voc")]) == 40


# ---------------------------------------------------------------------------
# QA/QC and uncertainty pass through
# ---------------------------------------------------------------------------


def test_qaqc_masks_and_records_counts() -> None:
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "units": "ppm",
            "qaqc": [{"kind": "range", "min": 1.91}],
        }
    )
    stream = _build(_frame(), instrument)
    assert np.isnan(stream["ch4"].values[0])
    assert stream["ch4"].attrs["qaqc_masked"] == "range:1"
    assert stream["ch4"].attrs["masked_fraction"] == pytest.approx(0.25)


def test_range_bounds_are_read_in_canonical_units() -> None:
    """The whole reason conversion precedes masking."""
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "units": "ppm",
            "convert": {"from_unit": "ppm", "to_unit": "ppb", "scale": 1000.0},
            "qaqc": [{"kind": "range", "min": 1700.0, "max": 3000.0}],
        }
    )
    stream = _build(_frame(), instrument)
    assert np.isfinite(stream["ch4"].values).all()


def test_declared_uncertainty_becomes_sigma_variables() -> None:
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "units": "ppm",
            "uncertainty": {
                "random": {"mode": "declared", "absolute": 0.002},
                "systematic": {"mode": "declared", "relative": 0.01},
            },
        }
    )
    stream = _build(_frame(), instrument)

    assert sigma_rand_name("ch4") in stream.data_vars
    assert sigma_sys_name("ch4") in stream.data_vars
    assert stream[sigma_rand_name("ch4")].values[0] == pytest.approx(0.002)
    assert stream[sigma_rand_name("ch4")].attrs["units"] == "ppm"


def test_sigma_names_compose_with_the_generators_truth_prefix() -> None:
    """'truth_' + the ingest name must equal the generator's answer-key name."""
    from tsara.synthetic.generator import TRUTH_PREFIX

    assert f"{TRUTH_PREFIX}{sigma_rand_name('ch4')}" == "truth_sigma_rand_ch4"
    assert f"{TRUTH_PREFIX}{sigma_sys_name('ch4')}" == "truth_sigma_sys_ch4"


def test_undeclared_uncertainty_emits_no_sigma_but_labels_it() -> None:
    """METHODS §2.3: nothing invented, and the obligation is recorded."""
    stream = _build(_frame(), _instrument())

    assert sigma_rand_name("ch4") not in stream.data_vars
    assert stream["ch4"].attrs["uncertainty_source"] == "empirical"
    assert stream["ch4"].attrs["uncertainty_source_random"] == "empirical"
    assert stream["ch4"].attrs["uncertainty_source_systematic"] == "unknown"


def test_omitted_systematic_is_labelled_zero_not_unknown() -> None:
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "units": "ppm",
            "uncertainty": {"random": {"mode": "declared", "absolute": 0.002}},
        }
    )
    stream = _build(_frame(), instrument)
    assert stream["ch4"].attrs["uncertainty_source_systematic"] == "zero"


def test_mixed_modes_are_labelled_mixed() -> None:
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "units": "ppm",
            "uncertainty": {
                "random": {"mode": "reported", "column": "CH4_SIG"},
                "systematic": {"mode": "declared", "relative": 0.01},
            },
        }
    )
    stream = _build(_frame(CH4_SIG=np.full(4, 0.001)), instrument)
    assert stream["ch4"].attrs["uncertainty_source"] == "mixed"


def test_decorrelation_timescale_is_carried() -> None:
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "units": "ppm",
            "uncertainty": {
                "random": {"mode": "declared", "absolute": 0.002},
                "decorrelation_timescale": "5min",
            },
        }
    )
    stream = _build(_frame(), _instrument()) if False else _build(_frame(), instrument)
    assert stream["ch4"].attrs["decorrelation_timescale"] == "5min"


# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------


def test_stationary_platform_gets_scalar_coordinates() -> None:
    stream = _build(_frame(), _instrument())
    assert stream.coords["latitude"].shape == ()
    assert float(stream.coords["latitude"]) == pytest.approx(40.77)
    assert float(stream.coords["altitude"]) == pytest.approx(1300.0)


def test_stationary_altitude_is_optional() -> None:
    platform = StationaryPlatform(latitude=40.0, longitude=-111.0)
    stream = _build(_frame(), _instrument(), platform=platform)
    assert "altitude" not in stream.coords


def test_mobile_platform_gets_no_coordinates_yet() -> None:
    """Attaching a track to this clock is interpolation, guarded in Phase 4."""
    stream = _build(_frame(), _instrument(), platform=MOBILE)
    assert "latitude" not in stream.coords
    assert "longitude" not in stream.coords


def test_mobile_binding_is_recorded_for_phase_4() -> None:
    stream = _build(_frame(), _instrument(), platform=MOBILE)
    assert stream.attrs["platform_gps_instrument"] == "gps"
    assert stream.attrs["platform_lat_variable"] == "latitude"
    assert stream.attrs["platform_kind"] == "mobile"


def test_mobile_altitude_variable_is_recorded_when_present() -> None:
    platform = MobilePlatform(
        gps_instrument="gps",
        lat_variable="latitude",
        lon_variable="longitude",
        alt_variable="altitude",
    )
    stream = _build(_frame(), _instrument(), platform=platform)
    assert stream.attrs["platform_alt_variable"] == "altitude"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_stream_describes_itself() -> None:
    """A file found on disk must explain what produced it."""
    stream = _build(
        _frame(), _instrument(), campaign="slv_2026", sources=[Path("a.dat"), Path("b.dat")]
    )
    assert stream.attrs["tsara_stage"] == "ingest"
    assert stream.attrs["instrument"] == "picarro"
    assert stream.attrs["campaign"] == "slv_2026"
    assert stream.attrs["n_source_files"] == 2
    assert stream.attrs["loader_format"] == "csv"
    assert stream.attrs["tsara_version"]


def test_instrument_metadata_is_carried_as_attrs() -> None:
    instrument = InstrumentConfig.model_validate(
        {
            "loader": {
                "format": "csv",
                "path_template": "*.dat",
                "time": {"column": "t", "format": "unix"},
            },
            "variables": {"ch4": {"column": "CH4_dry", "units": "ppm"}},
            "metadata": {"institution": "uutah"},
        }
    )
    assert _build(_frame(), instrument).attrs["meta_institution"] == "uutah"


def test_instrument_description_is_carried() -> None:
    instrument = InstrumentConfig.model_validate(
        {
            "description": "Picarro G2401",
            "loader": {
                "format": "csv",
                "path_template": "*.dat",
                "time": {"column": "t", "format": "unix"},
            },
            "variables": {"ch4": {"column": "CH4_dry", "units": "ppm"}},
        }
    )
    assert _build(_frame(), instrument).attrs["description"] == "Picarro G2401"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_missing_column_names_what_is_present() -> None:
    instrument = _instrument(ch4={"column": "ABSENT", "units": "ppm"})
    with pytest.raises(TsaraIngestError, match=r"reads column 'ABSENT'.*CH4_dry"):
        _build(_frame(), instrument)


def test_non_datetime_index_is_refused() -> None:
    frame = pd.DataFrame({"CH4_dry": [1.9, 2.0]})
    with pytest.raises(TsaraIngestError, match="DatetimeIndex-ed frame"):
        _build(frame, _instrument())


def test_unsorted_index_is_refused() -> None:
    """Concatenated files arrive in path order, which is not time order."""
    frame = _frame().iloc[np.array([0, 2, 1, 3])]
    with pytest.raises(TsaraIngestError, match="not monotonically"):
        _build(frame, _instrument())


def test_non_numeric_values_become_nan_rather_than_failing() -> None:
    frame = _frame()
    frame["CH4_dry"] = ["1.9", "bad", "1.92", "1.93"]
    stream = _build(frame, _instrument())
    assert np.isnan(stream["ch4"].values[1])
    assert stream["ch4"].values[0] == pytest.approx(1.9)


def test_microsecond_index_is_pinned_to_nanoseconds() -> None:
    """netCDF stores ns; an unpinned axis changes dtype on save/load."""
    frame = _frame()
    frame.index = pd.DatetimeIndex(frame.index).astype("datetime64[us]")
    assert _build(frame, _instrument())["time"].dtype == "datetime64[ns]"


def test_timezone_aware_index_is_refused() -> None:
    frame = _frame()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize("UTC")
    with pytest.raises(TsaraIngestError, match="timezone-aware"):
        _build(frame, _instrument())


def test_variable_description_is_carried() -> None:
    instrument = _instrument(
        ch4={"column": "CH4_dry", "units": "ppm", "description": "Dry-air methane"}
    )
    assert _build(_frame(), instrument)["ch4"].attrs["description"] == "Dry-air methane"


def test_single_source_file_is_named_in_messages(caplog: pytest.LogCaptureFixture) -> None:
    """With one contributing file, diagnostics name that file rather than a count."""
    import logging

    instrument = _instrument(
        ch4={"column": "CH4_dry", "units": "ppm", "qaqc": [{"kind": "range", "min": 99.0}]}
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.qaqc"):
        _build(_frame(), instrument, sources=[Path("only.dat")])
    assert "only.dat" in caplog.text


def test_many_source_files_are_summarised_in_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After concatenation there is no single file to blame."""
    import logging

    instrument = _instrument(
        ch4={"column": "CH4_dry", "units": "ppm", "qaqc": [{"kind": "range", "min": 99.0}]}
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.qaqc"):
        _build(_frame(), instrument, sources=[Path("a.dat"), Path("b.dat")])
    assert "2 files starting a.dat" in caplog.text


def test_no_sources_still_builds() -> None:
    stream = _build(_frame(), _instrument(), sources=[])
    assert stream.attrs["n_source_files"] == 0
