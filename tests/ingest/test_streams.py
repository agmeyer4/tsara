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
from tsara.core.naming import (
    CELL_METHODS_ATTR,
    LOD_COUNT_KEY,
    RAW_TIME_START_COLUMN,
    RAW_TIME_STOP_COLUMN,
    SUPPORT_COVERAGE_ATTR,
    SUPPORT_LABEL_ATTR,
    TIME_BOUNDS_VAR,
    TIME_SHIFT_ATTR,
    sigma_rand_name,
    sigma_sys_name,
)
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.streams import build_stream
from tsara.ingest.support import ResolvedSupport

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


def test_every_variable_records_the_field_it_measures() -> None:
    """Declared or defaulted, the attribute is written, so no reader needs the rule."""
    instrument = _instrument(
        ch4={"column": "CH4_dry", "role": "gas", "units": "ppm"},
        ch4_backup={"column": "CH4_b", "role": "gas", "units": "ppm", "field": "ch4"},
    )
    stream = _build(_frame(CH4_b=np.linspace(1.9, 1.93, 4)), instrument)
    assert stream["ch4"].attrs["field"] == "ch4"
    assert stream["ch4_backup"].attrs["field"] == "ch4"


def test_role_and_circular_are_recorded() -> None:
    instrument = _instrument(
        wdir={"column": "WD", "role": "met", "units": "degrees", "circular": True}
    )
    stream = _build(_frame(WD=np.array([10.0, 20.0, 350.0, 5.0])), instrument)
    assert stream["wdir"].attrs["role"] == "met"
    # netCDF has no boolean type; 0/1 matches the generator's convention.
    assert stream["wdir"].attrs["circular"] == 1


def test_a_converted_direction_is_wrapped_before_qaqc_masks_it() -> None:
    """A magnetic-to-true offset must not delete the sector just west of north.

    +10.3 degrees takes 352 to 362.3. Unwrapped, the natural bound
    `range: [0, 360]` masks it, and every other reading within 10.3 degrees
    west of north with it -- a whole sector of the wind rose gone, silently.
    Wrapped first, 362.3 is 2.3 and nothing is lost.
    """
    instrument = _instrument(
        wdir={
            "column": "WD",
            "role": "met",
            "units": "degrees",
            "circular": True,
            "convert": {
                "from_unit": "degrees_magnetic",
                "to_unit": "degrees",
                "scale": 1.0,
                "offset": 10.3,
            },
            "qaqc": [{"kind": "range", "min": 0.0, "max": 360.0}],
        }
    )
    stream = _build(_frame(WD=np.array([352.0, 358.0, 1.0, 180.0])), instrument)
    assert stream["wdir"].values == pytest.approx([2.3, 8.3, 11.3, 190.3])
    assert stream["wdir"].attrs["masked_fraction"] == 0.0


def test_a_direction_already_past_a_full_turn_is_wrapped_without_a_conversion() -> None:
    """Loggers write -5 and 365 too; the wrap does not depend on a conversion."""
    instrument = _instrument(
        wdir={"column": "WD", "role": "met", "units": "degrees", "circular": True}
    )
    stream = _build(_frame(WD=np.array([-5.0, 365.0, 720.0, np.nan])), instrument)
    assert stream["wdir"].values[:3] == pytest.approx([355.0, 5.0, 0.0])
    assert np.isnan(stream["wdir"].values[3])


def test_a_scalar_past_360_is_not_wrapped() -> None:
    """Only a variable that declares itself circular is an angle."""
    instrument = _instrument(temp={"column": "T", "role": "met", "units": "K"})
    stream = _build(_frame(T=np.array([365.0, 370.0, 375.0, 380.0])), instrument)
    assert stream["temp"].values == pytest.approx([365.0, 370.0, 375.0, 380.0])


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
    assert stream["ch4"].attrs["uncertainty_provenance"] == "empirical"
    assert stream["ch4"].attrs["uncertainty_provenance_random"] == "empirical"
    assert stream["ch4"].attrs["uncertainty_provenance_systematic"] == "unknown"


def test_omitted_systematic_is_labelled_zero_not_unknown() -> None:
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "units": "ppm",
            "uncertainty": {"random": {"mode": "declared", "absolute": 0.002}},
        }
    )
    stream = _build(_frame(), instrument)
    assert stream["ch4"].attrs["uncertainty_provenance_systematic"] == "zero"


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
    assert stream["ch4"].attrs["uncertainty_provenance"] == "mixed"


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
        _frame(), _instrument(), campaign="slv_2026", files=[Path("a.dat"), Path("b.dat")]
    )
    assert stream.attrs["tsara_stage"] == "ingest"
    assert stream.attrs["instrument"] == "picarro"
    assert stream.attrs["campaign"] == "slv_2026"
    assert stream.attrs["n_files"] == 2
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


def test_single_input_file_is_named_in_messages(caplog: pytest.LogCaptureFixture) -> None:
    """With one contributing file, diagnostics name that file rather than a count."""
    import logging

    instrument = _instrument(
        ch4={"column": "CH4_dry", "units": "ppm", "qaqc": [{"kind": "range", "min": 99.0}]}
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.qaqc"):
        _build(_frame(), instrument, files=[Path("only.dat")])
    assert "only.dat" in caplog.text


def test_many_input_files_are_summarised_in_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After concatenation there is no single file to blame."""
    import logging

    instrument = _instrument(
        ch4={"column": "CH4_dry", "units": "ppm", "qaqc": [{"kind": "range", "min": 99.0}]}
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.qaqc"):
        _build(_frame(), instrument, files=[Path("a.dat"), Path("b.dat")])
    assert "2 files starting a.dat" in caplog.text


def test_no_files_still_builds() -> None:
    stream = _build(_frame(), _instrument(), files=[])
    assert stream.attrs["n_files"] == 0


# ---------------------------------------------------------------------------
# Below-detection counts travel to the variable they censor
# ---------------------------------------------------------------------------


def test_lod_count_is_attached_to_the_variable_it_censors() -> None:
    """Keyed on the raw column name, which is the only name a reader knows;
    stream assembly is where it becomes a fact about a canonical species."""
    stream = _build(_frame(), _instrument(), file_attrs={LOD_COUNT_KEY: {"CH4_dry": 17}})
    assert stream["ch4"].attrs["n_lod_masked"] == 17


def test_no_lod_count_means_no_attr() -> None:
    """Absence must not read as a claim that nothing was below detection."""
    assert "n_lod_masked" not in _build(_frame(), _instrument())["ch4"].attrs


def test_a_zero_lod_count_is_not_recorded() -> None:
    stream = _build(_frame(), _instrument(), file_attrs={LOD_COUNT_KEY: {"CH4_dry": 0}})
    assert "n_lod_masked" not in stream["ch4"].attrs


def test_lod_counts_for_other_columns_are_ignored() -> None:
    stream = _build(_frame(), _instrument(), file_attrs={LOD_COUNT_KEY: {"Other": 4}})
    assert "n_lod_masked" not in stream["ch4"].attrs


def test_declared_file_attrs_reach_the_dataset() -> None:
    """A stream found on disk has to explain itself (CLAUDE.md 5)."""
    stream = _build(_frame(), _instrument(), file_attrs={"icartt_pi": "Hu, Lu"})
    assert stream.attrs["icartt_pi"] == "Hu, Lu"


# ---------------------------------------------------------------------------
# Cells (Phase 3.5)
# ---------------------------------------------------------------------------


def _resolved(**overrides: Any) -> ResolvedSupport:
    fields: dict[str, Any] = {
        "label": "start",
        "method": "mean",
        "width_ns": 2_000_000_000,
        "label_provenance": "declared",
        "width_provenance": "declared",
        "method_provenance": "declared",
    }
    fields.update(overrides)
    return ResolvedSupport(**fields)


def _bounded_frame() -> pd.DataFrame:
    frame = _frame()
    frame[RAW_TIME_START_COLUMN] = frame.index
    frame[RAW_TIME_STOP_COLUMN] = frame.index + pd.Timedelta("2s")
    return frame


def test_a_stream_carries_its_cells_and_their_provenance() -> None:
    stream = build_stream(
        _bounded_frame(),
        _instrument(),
        name="picarro",
        platform=StationaryPlatform(latitude=40.0, longitude=-111.0),
        support=_resolved(),
    )
    assert TIME_BOUNDS_VAR in stream.coords
    assert stream["ch4"].attrs[CELL_METHODS_ATTR] == "time: mean"
    assert stream.attrs[SUPPORT_LABEL_ATTR] == "start"
    assert stream.attrs["tsara_support_label_provenance"] == "declared"
    assert stream.attrs[SUPPORT_COVERAGE_ATTR] == pytest.approx(1.0)


def test_a_sigma_carries_no_cell_method() -> None:
    """A sigma shares its species' cell but not its cell *method*.

    `cell_methods` says what operation produced a value FROM its cell, so
    "time: mean" on `sigma_rand_ch4` asserts the stored number is the mean of
    the random sigmas over that cell. It is not — it is the standard error of
    the cell mean, smaller by exactly the square root of N_eff, a factor the
    same stream records in `uncertainty_n_eff`. Measured on a 60 s cell of
    1 s data with a 2 s decorrelation time: 0.130 ppb stored against 0.5
    declared, a ratio of 3.83.

    The systematic companion happens to satisfy "time: mean" exactly, because
    a fully correlated error does not average down. It is excluded anyway —
    true by coincidence is not a reason to assert it, and one string on both
    invites a reader to treat two components that behave oppositely under
    averaging as though they were alike.
    """
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "role": "gas",
            "units": "ppm",
            "uncertainty": {
                "random": {"mode": "declared", "absolute": 0.5},
                "systematic": {"mode": "declared", "absolute": 0.2},
            },
        }
    )
    stream = build_stream(
        _bounded_frame(),
        instrument,
        name="picarro",
        platform=StationaryPlatform(latitude=40.0, longitude=-111.0),
        support=_resolved(),
    )
    # The species says how it relates to its cell; its companions do not.
    assert stream["ch4"].attrs[CELL_METHODS_ATTR] == "time: mean"
    assert CELL_METHODS_ATTR not in stream["sigma_rand_ch4"].attrs
    assert CELL_METHODS_ATTR not in stream["sigma_sys_ch4"].attrs
    # They are still identified, by the seam both producers share.
    assert stream["sigma_rand_ch4"].attrs["uncertainty_component"] == "random"


def test_a_stream_whose_cells_could_not_be_determined_still_builds() -> None:
    """One sample implies no interval, and the stream says so rather than
    refusing to exist."""
    stream = build_stream(
        _frame(),
        _instrument(),
        name="picarro",
        platform=StationaryPlatform(latitude=40.0, longitude=-111.0),
        support=_resolved(width_ns=None, width_provenance="assumed"),
    )
    assert TIME_BOUNDS_VAR not in stream.coords
    assert stream.attrs["tsara_support_width_provenance"] == "assumed"
    assert np.isnan(stream.attrs[SUPPORT_COVERAGE_ATTR])


def test_a_clock_correction_is_recorded_but_not_reapplied() -> None:
    """Assembly receives a frame whose axis is already final; shifting again
    here would move timestamps out from under the ordering that ran on them."""
    frame = _bounded_frame()
    stream = build_stream(
        frame,
        _instrument(),
        name="picarro",
        platform=StationaryPlatform(latitude=40.0, longitude=-111.0),
        support=_resolved(),
        time_shift="-9s",
    )
    assert stream.attrs[TIME_SHIFT_ATTR] == "-9s"
    assert np.array_equal(
        np.asarray(stream["time"].values, dtype="datetime64[ns]"),
        np.asarray(frame.index, dtype="datetime64[ns]"),
    )


def test_a_quoted_interval_is_written_into_the_saved_product() -> None:
    """ "This sigma describes a different interval from its own cells" cannot
    be re-derived from the numbers, so it is written down. Ingestion does not
    reconcile the two -- that needs a decorrelation timescale and an averaging
    model (METHODS 10.8) -- which is exactly why the record has to travel."""
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "role": "gas",
            "units": "ppm",
            "uncertainty": {
                "random": {"mode": "declared", "absolute": 1.0, "at_width": "1s"},
                "decorrelation_timescale": "1ns",
            },
        }
    )
    stream = build_stream(
        _bounded_frame(),
        instrument,
        name="picarro",
        platform=StationaryPlatform(latitude=40.0, longitude=-111.0),
        support=_resolved(),
    )
    assert stream["ch4"].attrs["uncertainty_at_width"] == "1s"
    # The fixture's cells are 2 s wide, so two 1 s intervals fit in each.
    assert stream["ch4"].attrs["uncertainty_at_width_ratio"] == pytest.approx(2.0)
    # And the figure itself is untouched, which is the claim that matters.
    assert float(stream["sigma_rand_ch4"].values[0]) == pytest.approx(1.0)


def test_a_systematic_quoted_interval_reaches_the_product_too() -> None:
    """It can never be acted on, but a manifest that states one has said
    something, and the product should not be quieter than the manifest."""
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "role": "gas",
            "units": "ppm",
            "uncertainty": {
                "random": {"mode": "declared", "absolute": 1.0},
                "systematic": {"mode": "declared", "relative": 0.01, "at_width": "1s"},
            },
        }
    )
    stream = build_stream(
        _bounded_frame(),
        instrument,
        name="picarro",
        platform=StationaryPlatform(latitude=40.0, longitude=-111.0),
        support=_resolved(),
    )
    assert stream["ch4"].attrs["uncertainty_systematic_at_width"] == "1s"
    assert "uncertainty_at_width" not in stream["ch4"].attrs


def test_a_stream_without_cells_still_carries_the_declaration() -> None:
    instrument = _instrument(
        ch4={
            "column": "CH4_dry",
            "role": "gas",
            "units": "ppm",
            "uncertainty": {"random": {"mode": "declared", "absolute": 1.0, "at_width": "1s"}},
        }
    )
    stream = build_stream(
        _frame(),
        instrument,
        name="picarro",
        platform=StationaryPlatform(latitude=40.0, longitude=-111.0),
    )
    assert stream["ch4"].attrs["uncertainty_at_width"] == "1s"
    assert "uncertainty_at_width_ratio" not in stream["ch4"].attrs
