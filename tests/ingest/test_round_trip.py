"""The round-trip harness: generate → export → ingest → compare to truth.

This is the only test in the suite that can say anything about whether
ingestion is *correct* rather than merely self-consistent. Every other test
checks one stage against expectations written by the same person who wrote
the stage. Here the expectations come from the synthetic generator, which
knows the true values, the true error components and the true event times,
and which had no part in writing the reader, the crawler or the assembler.

What a failure here means, and what it does not: these tests exercise
crawl → read → convert → mask → resolve uncertainty → assemble → persist as
one path, so a failure localizes poorly. That is the point. The per-stage
tests localize; this one notices.

Conversion and masking reach that path only because the exporter is *asked*
to put them there. A default export writes every species in its own
canonical units under its own name, so there is nothing to convert and
nothing to mask; measured against five injected ingestion bugs, a round trip
in that shape noticed one. The tests below that pass ``raw_units`` and
``qaqc_bounds`` are the ones covering the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from tsara.config.loader import load_manifest
from tsara.config.manifest import CSVLoader, SupportSpec
from tsara.core.naming import (
    TIME_BOUNDS_VAR,
    TIME_COORD,
    sigma_rand_name,
    sigma_sys_name,
)
from tsara.ingest import ingest_campaign, load_streams, save_streams
from tsara.synthetic import generate
from tsara.synthetic.background import TsaraSyntheticError
from tsara.synthetic.config import SyntheticConfig
from tsara.synthetic.export import (
    EXPORT_MANIFEST,
    EXPORT_RAW_DIR,
    START_COLUMN,
    STOP_COLUMN,
    RawUnits,
    export_raw,
)

#: Ingestion recovers values through a text file, and pandas' default CSV
#: parser is not round-trip exact — it lands within about one unit in the
#: last place. That is far tighter than any real measurement and still tight
#: enough to catch a wrong column, a missing conversion or an off-by-one.
RTOL = 1e-12


def _config(**overrides: Any) -> SyntheticConfig:
    """A small two-instrument stationary campaign with a mixed error budget."""
    spec: dict[str, Any] = {
        "name": "round_trip",
        "seed": 7,
        "start": "2026-01-01T00:00:00Z",
        "duration": "20min",
        "platform": {"kind": "stationary", "latitude": 40.77, "longitude": -111.85},
        "atmosphere": {
            "fields": {
                "ch4": {
                    "background": {"kind": "parametric", "offset": 1900.0},
                    "role": "gas",
                    "units": "ppb",
                },
                "c2h6": {
                    "background": {"kind": "parametric", "offset": 2.0},
                    "role": "gas",
                    "units": "ppb",
                },
                "wind_dir": {
                    "background": {"kind": "parametric", "offset": 180.0},
                    "role": "met",
                    "units": "degrees",
                    "circular": True,
                },
            },
        },
        "instruments": {
            "analyzer": {
                "native_rate": "2s",
                "measures": {
                    "ch4": {
                        "uncertainty": {
                            "random": {"absolute": 0.7},
                            "systematic": {"relative": 0.005},
                        },
                    },
                    "c2h6": {
                        "uncertainty": {"random": {"absolute": 0.05, "report_as": "c2h6_err"}},
                    },
                },
            },
            "met": {"native_rate": "10s", "measures": {"wind_dir": {}}},
        },
    }
    spec.update(overrides)
    return SyntheticConfig.model_validate(spec)


def _round_trip(tmp_path: Path, config: SyntheticConfig | None = None) -> tuple[Any, Any]:
    """Generate, export, ingest. Returns (generated, ingested)."""
    generated = generate(config or _config())
    manifest_path = export_raw(generated, tmp_path / "export")
    ingested = ingest_campaign(load_manifest(manifest_path))
    return generated, ingested


# ---------------------------------------------------------------------------
# The loop closes
# ---------------------------------------------------------------------------


def test_export_writes_a_loadable_manifest(tmp_path: Path) -> None:
    generated = generate(_config())
    manifest_path = export_raw(generated, tmp_path / "export")

    assert manifest_path.name == EXPORT_MANIFEST
    assert (tmp_path / "export" / EXPORT_RAW_DIR / "analyzer.csv").is_file()
    assert load_manifest(manifest_path).name == "round_trip"


def test_an_archive_exported_to_a_relative_path_ingests_from_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README's quickstart, which exports to "demo_campaign".

    The manifest wrote the path it was given as `base_path`, and the loader
    resolves a relative `base_path` against the manifest's own directory, so
    a relative export looked for its files in demo_campaign/demo_campaign/raw.
    Every other test exports to an absolute temporary path, which is why the
    two conventions never met. The archive is also moved after writing: a
    manifest naming its files relative to itself travels with them.
    """
    monkeypatch.chdir(tmp_path)
    generated = generate(_config())
    manifest_path = export_raw(generated, "demo_campaign")
    assert yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["base_path"] == "raw"

    moved = tmp_path / "elsewhere" / "campaign"
    moved.parent.mkdir()
    (tmp_path / "demo_campaign").rename(moved)
    monkeypatch.chdir(moved.parent)
    ingested = ingest_campaign(load_manifest(moved / EXPORT_MANIFEST))
    assert set(ingested.streams) == set(generated.streams)


def test_every_instrument_survives_the_trip(tmp_path: Path) -> None:
    generated, ingested = _round_trip(tmp_path)
    assert set(ingested.streams) == set(generated.streams)


def test_sample_counts_are_preserved(tmp_path: Path) -> None:
    """Nothing dropped, nothing duplicated, on either instrument's clock."""
    generated, ingested = _round_trip(tmp_path)
    for name in generated.streams:
        assert ingested[name].sizes["time"] == generated.streams[name].sizes["time"]


def test_timestamps_are_recovered_exactly(tmp_path: Path) -> None:
    """Jitter puts real information in the nanosecond digits."""
    generated, ingested = _round_trip(tmp_path)
    for name in generated.streams:
        np.testing.assert_array_equal(
            ingested[name]["time"].values, generated.streams[name]["time"].values
        )
        assert ingested[name]["time"].dtype == "datetime64[ns]"


def test_values_are_recovered(tmp_path: Path) -> None:
    """The measurement itself, through crawl, read, convert and assemble."""
    generated, ingested = _round_trip(tmp_path)
    for name, stream in generated.streams.items():
        for species in ("ch4", "c2h6", "wind_dir"):
            if species not in stream.data_vars:
                continue
            np.testing.assert_allclose(
                ingested[name][species].values, stream[species].values, rtol=RTOL
            )


def test_units_and_roles_survive(tmp_path: Path) -> None:
    generated, ingested = _round_trip(tmp_path)
    assert ingested["analyzer"]["ch4"].attrs["units"] == "ppb"
    assert ingested["analyzer"]["ch4"].attrs["role"] == "gas"
    assert ingested["met"]["wind_dir"].attrs["role"] == "met"
    assert ingested["met"]["wind_dir"].attrs["circular"] == 1


def _assert_both_producers_record_the_same_fields(generated: Any, ingested: Any) -> None:
    """Every ingested variable's field is the one its generated original names.

    Walked from the ingested side, because ingestion writes the attribute on
    every variable its manifest declares, so that side lists exactly the
    measurements; walked from the generated side, a variable the generator
    forgot to label would simply be skipped. Counted, because a walk that
    found no field would pass.
    """
    checked = 0
    for name, stream in ingested.streams.items():
        for variable in stream.data_vars:
            if "field" in stream[variable].attrs:
                original = generated.streams[name][variable].attrs.get("field")
                assert original == stream[variable].attrs["field"], (name, variable)
                checked += 1
    assert checked > 0


def test_both_producers_record_the_same_fields(tmp_path: Path) -> None:
    """Substitutability for the identity attribute (METHODS §1.6, §9.7)."""
    generated, ingested = _round_trip(tmp_path)
    _assert_both_producers_record_the_same_fields(generated, ingested)


def test_two_instruments_measuring_one_field_round_trip_with_its_identity(
    tmp_path: Path,
) -> None:
    """The generator's `measures` and the manifest's `field` are one idea.

    A second analyzer measures the same methane under its own name. The
    exported manifest must declare `field: ch4` for exactly that variable and
    leave the default to speak for the other, and ingestion must hand both
    streams back naming one field, with the values the generator wrote.
    """
    spec = _config().model_dump(mode="json")
    spec["instruments"]["aeris"] = {
        "native_rate": "1s",
        "measures": {"ch4": {"name": "ch4_aeris", "uncertainty": {"random": {"absolute": 2.0}}}},
    }
    generated, ingested = _round_trip(tmp_path, SyntheticConfig.model_validate(spec))

    manifest = load_manifest(tmp_path / "export" / EXPORT_MANIFEST)
    assert manifest.instruments["aeris"].variables["ch4_aeris"].field == "ch4"
    assert manifest.instruments["analyzer"].variables["ch4"].field is None
    assert manifest.gas_species == ("ch4", "c2h6")

    assert ingested["aeris"]["ch4_aeris"].attrs["field"] == "ch4"
    assert ingested["analyzer"]["ch4"].attrs["field"] == "ch4"
    np.testing.assert_allclose(
        ingested["aeris"]["ch4_aeris"].values,
        generated.streams["aeris"]["ch4_aeris"].values,
        rtol=RTOL,
    )
    _assert_both_producers_record_the_same_fields(generated, ingested)


# ---------------------------------------------------------------------------
# The uncertainty budget, which is the part with a right answer
# ---------------------------------------------------------------------------


def test_declared_uncertainty_matches_the_true_budget(tmp_path: Path) -> None:
    """The generator drew noise from these sigmas; the manifest must rebuild them.

    This is the single strongest assertion in the suite. The generator wrote
    ``truth_sigma_rand_ch4`` from its own ``TrueUncertainty``; ingestion
    rebuilt ``sigma_rand_ch4`` from the manifest declaration produced by
    ``to_manifest_uncertainty()``. Agreement means the two schemas describe
    the same quantity, which is the thing that seam exists to guarantee.
    """
    generated, ingested = _round_trip(tmp_path)
    truth = generated.streams["analyzer"]

    np.testing.assert_allclose(
        ingested["analyzer"][sigma_rand_name("ch4")].values,
        truth["truth_sigma_rand_ch4"].values,
        rtol=RTOL,
    )
    # The systematic component is deliberately NOT compared at RTOL, and the
    # reason is a real difference the round trip exposed rather than a bug:
    # a relative term is a fraction of *something*, and the two sides
    # necessarily choose differently. The generator scales the TRUE signal,
    # because that is what actually produced the error it injected.
    # Ingestion can only scale the READING, because a manifest describes a
    # file and the true value is exactly what is unavailable. The two agree
    # to the fractional size of the error itself — second order, and the
    # standard reading of "percent of reading" in an instrument spec.
    np.testing.assert_allclose(
        ingested["analyzer"][sigma_sys_name("ch4")].values,
        truth["truth_sigma_sys_ch4"].values,
        rtol=0.02,
    )


def test_reported_uncertainty_is_read_from_its_column(tmp_path: Path) -> None:
    """An instrument that publishes its own per-point sigma (the EM27 case)."""
    generated, ingested = _round_trip(tmp_path)

    assert ingested["analyzer"]["c2h6"].attrs["uncertainty_provenance_random"] == "reported"
    np.testing.assert_allclose(
        ingested["analyzer"][sigma_rand_name("c2h6")].values,
        generated.streams["analyzer"]["truth_sigma_rand_c2h6"].values,
        rtol=RTOL,
    )


def test_provenance_labels_match_what_was_declared(tmp_path: Path) -> None:
    _, ingested = _round_trip(tmp_path)

    ch4 = ingested["analyzer"]["ch4"].attrs
    assert ch4["uncertainty_provenance_random"] == "declared"
    assert ch4["uncertainty_provenance_systematic"] == "declared"

    # c2h6 declares only a random component, so its systematic is a
    # deliberate zero rather than unknown.
    c2h6 = ingested["analyzer"]["c2h6"].attrs
    assert c2h6["uncertainty_provenance_systematic"] == "zero"

    # wind_dir declares no budget at all: random falls back to the empirical
    # estimator and systematic is genuinely unknown.
    wind = ingested["met"]["wind_dir"].attrs
    assert wind["uncertainty_provenance"] == "empirical"
    assert wind["uncertainty_provenance_systematic"] == "unknown"
    assert sigma_rand_name("wind_dir") not in ingested["met"].data_vars


def test_the_answer_key_is_not_exported(tmp_path: Path) -> None:
    """A leaked truth column would make every assertion above meaningless."""
    generated = generate(_config())
    export_raw(generated, tmp_path / "export")

    for csv in (tmp_path / "export" / EXPORT_RAW_DIR).glob("*.csv"):
        header = csv.read_text(encoding="utf-8").splitlines()[0]
        assert "truth_" not in header, csv.name


# ---------------------------------------------------------------------------
# Platforms
# ---------------------------------------------------------------------------


def test_stationary_position_survives(tmp_path: Path) -> None:
    _, ingested = _round_trip(tmp_path)
    assert float(ingested["analyzer"].coords["latitude"]) == pytest.approx(40.77)


def test_mobile_campaign_round_trips_with_a_gps_instrument(tmp_path: Path) -> None:
    """The generator already emits GPS as its own stream at its own rate.

    That is the canonical multi-rate case, and it means a mobile campaign
    round-trips with no special handling: the track is written like any
    other instrument and only needs declaring in the manifest.
    """
    config = _config(
        platform={
            "kind": "mobile",
            "start_latitude": 40.77,
            "start_longitude": -111.85,
            "speed_m_s": 15.0,
        }
    )
    generated, ingested = _round_trip(tmp_path, config)

    assert "gps" in ingested.streams
    assert ingested["gps"]["latitude"].attrs["role"] == "gps_lat"
    _assert_both_producers_record_the_same_fields(generated, ingested)
    # The gas streams get no coordinates: attaching a track to their clocks
    # is interpolation, which belongs to Phase 4.
    assert "latitude" not in ingested["analyzer"].coords
    assert ingested["analyzer"].attrs["platform_gps_instrument"] == "gps"


# ---------------------------------------------------------------------------
# All the way to disk and back
# ---------------------------------------------------------------------------


def test_full_loop_through_a_bundle(tmp_path: Path) -> None:
    """Generate → export → ingest → save → load, still matching truth."""
    generated, ingested = _round_trip(tmp_path)
    save_streams(ingested, tmp_path / "bundle")
    reloaded = load_streams(tmp_path / "bundle")

    truth = generated.streams["analyzer"]
    np.testing.assert_allclose(reloaded["analyzer"]["ch4"].values, truth["ch4"].values, rtol=RTOL)
    np.testing.assert_array_equal(reloaded["analyzer"]["time"].values, truth["time"].values)
    assert reloaded["analyzer"]["ch4"].attrs["uncertainty_provenance"] == "declared"


def test_quantized_species_round_trip(tmp_path: Path) -> None:
    """Quantization survives a text round trip to within the parser's precision.

    Not asserted bitwise: a quantized value like 1905.3 is not exactly
    representable in binary, and pandas' default CSV parser does not
    guarantee the same nearest double the writer chose. The recovered values
    land about one unit in the last place away — far below any measurement
    resolution, but not zero.
    """
    config = _config()
    spec = config.model_dump(mode="json")
    spec["instruments"]["analyzer"]["measures"]["ch4"]["quantization"] = 0.1
    generated, ingested = _round_trip(tmp_path, SyntheticConfig.model_validate(spec))

    np.testing.assert_allclose(
        ingested["analyzer"]["ch4"].values,
        generated.streams["analyzer"]["ch4"].values,
        rtol=RTOL,
    )


def test_site_altitude_survives_the_trip(tmp_path: Path) -> None:
    """An optional field must be carried across the seam, not quietly dropped."""
    config = _config(
        platform={
            "kind": "stationary",
            "latitude": 40.77,
            "longitude": -111.85,
            "altitude_m": 1300.0,
        }
    )
    _, ingested = _round_trip(tmp_path, config)
    assert float(ingested["analyzer"].coords["altitude"]) == pytest.approx(1300.0)


# ---------------------------------------------------------------------------
# Conversion and masking, which the default export cannot reach
# ---------------------------------------------------------------------------
#
# Measured before these were written: a round trip with no declared
# conversion caught 1 of 5 injected ingestion bugs. Three of the four misses
# were conversion-related, so the exporter learned to write a species in
# non-canonical units and declare the conversion back.
#
# The offset is not decorative. With offset = 0, `value * scale + offset` and
# `(value + offset) * scale` are the same function, and an ordering bug in
# convert_values passes unnoticed; with a non-zero offset it does not.

RAW_PPM = RawUnits(from_unit="ppm", scale=1000.0, offset=7.5)


def _converted(tmp_path: Path, **kw: Any) -> tuple[Any, Any]:
    generated = generate(_config())
    manifest_path = export_raw(generated, tmp_path / "export", raw_units={"ch4": RAW_PPM}, **kw)
    return generated, ingest_campaign(load_manifest(manifest_path))


def test_values_survive_a_unit_conversion(tmp_path: Path) -> None:
    """Ingestion must undo exactly what the exporter did."""
    generated, ingested = _converted(tmp_path)
    np.testing.assert_allclose(
        ingested["analyzer"]["ch4"].values,
        generated.streams["analyzer"]["ch4"].values,
        rtol=RTOL,
    )


def test_the_file_really_is_in_other_units(tmp_path: Path) -> None:
    """Guards the test above from passing because nothing was converted."""
    generated = generate(_config())
    export_raw(generated, tmp_path / "export", raw_units={"ch4": RAW_PPM})
    written = pd.read_csv(tmp_path / "export" / EXPORT_RAW_DIR / "analyzer.csv")["ch4"]
    # ~1900 ppb becomes ~1.89 ppm, which no unconverted path could produce.
    assert written.max() < 10.0


def test_canonical_units_are_recorded_after_conversion(tmp_path: Path) -> None:
    _, ingested = _converted(tmp_path)
    assert ingested["analyzer"]["ch4"].attrs["units"] == "ppb"


def test_a_reported_sigma_is_converted_but_a_declared_one_is_not(tmp_path: Path) -> None:
    """The asymmetry METHODS 2.2 requires, checked against truth rather than
    against a hand-written expectation.

    `absolute` is declared in canonical units and must survive untouched; a
    reported sigma column is in the file's units and must be scaled. Both
    are compared to the generator's own sigmas.
    """
    generated = generate(_config())
    manifest_path = export_raw(
        generated,
        tmp_path / "export",
        raw_units={"ch4": RAW_PPM, "c2h6": RawUnits(from_unit="ppm", scale=1000.0, offset=0.25)},
    )
    ingested = ingest_campaign(load_manifest(manifest_path))
    truth = generated.streams["analyzer"]

    np.testing.assert_allclose(
        ingested["analyzer"][sigma_rand_name("ch4")].values,
        truth["truth_sigma_rand_ch4"].values,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        ingested["analyzer"][sigma_rand_name("c2h6")].values,
        truth["truth_sigma_rand_c2h6"].values,
        rtol=RTOL,
    )


def test_qaqc_bounds_are_applied_after_conversion(tmp_path: Path) -> None:
    """A bound stated in canonical units must be compared against canonical
    values (METHODS 9.4).

    The bound here is meaningless in the file's own units -- 1900 ppm is not
    a number that appears anywhere in a file written in ppm -- so masking
    the right samples is only possible if conversion ran first.
    """
    generated = generate(_config())
    truth = generated.streams["analyzer"]["ch4"].values
    cutoff = float(np.median(truth))

    manifest_path = export_raw(
        generated,
        tmp_path / "export",
        raw_units={"ch4": RAW_PPM},
        qaqc_bounds={"ch4": (cutoff, None)},
    )
    ingested = ingest_campaign(load_manifest(manifest_path))["analyzer"]

    expected_mask = truth < cutoff
    np.testing.assert_array_equal(np.isnan(ingested["ch4"].values), expected_mask)
    assert int(expected_mask.sum()) > 0, "the bound must actually mask something"


def test_qaqc_without_conversion_still_masks(tmp_path: Path) -> None:
    """The simple case, so a failure above localizes to conversion."""
    generated = generate(_config())
    truth = generated.streams["analyzer"]["ch4"].values
    cutoff = float(np.median(truth))
    manifest_path = export_raw(generated, tmp_path / "export", qaqc_bounds={"ch4": (cutoff, None)})
    ingested = ingest_campaign(load_manifest(manifest_path))["analyzer"]
    np.testing.assert_array_equal(np.isnan(ingested["ch4"].values), truth < cutoff)


# ---------------------------------------------------------------------------
# The exporter refuses what it cannot honour
# ---------------------------------------------------------------------------


def test_exporting_onto_a_file_is_a_tsara_error(tmp_path: Path) -> None:
    target = tmp_path / "not_a_dir"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(TsaraSyntheticError, match="not a directory"):
        export_raw(generate(_config()), target)


def test_an_unknown_species_is_refused(tmp_path: Path) -> None:
    """Silently ignoring a typo would let the harness check less than it claims."""
    with pytest.raises(TsaraSyntheticError, match="ch5"):
        export_raw(
            generate(_config()),
            tmp_path / "export",
            raw_units={"ch5": RAW_PPM},
        )


def test_an_unknown_qaqc_species_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TsaraSyntheticError, match="nope"):
        export_raw(generate(_config()), tmp_path / "export", qaqc_bounds={"nope": (0.0, 1.0)})


# ---------------------------------------------------------------------------
# Float precision is a declared guarantee, not an accident
# ---------------------------------------------------------------------------


def test_exact_float_precision_recovers_values_bitwise(tmp_path: Path) -> None:
    """Measured: pandas' default parser returns ~41% of noisy synthetic
    values one unit in the last place away. 'exact' returns all of them."""
    generated = generate(_config())
    manifest_path = export_raw(generated, tmp_path / "export")

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for instrument in payload["instruments"].values():
        instrument["loader"]["float_precision"] = "exact"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    ingested = ingest_campaign(load_manifest(manifest_path))
    np.testing.assert_array_equal(
        ingested["analyzer"]["ch4"].values,
        generated.streams["analyzer"]["ch4"].values,
    )


def test_fast_is_the_default(tmp_path: Path) -> None:
    generated = generate(_config())
    manifest = load_manifest(export_raw(generated, tmp_path / "export"))
    loader = manifest.instruments["analyzer"].loader
    assert isinstance(loader, CSVLoader)
    assert loader.float_precision == "fast"


def test_an_upper_bound_alone_is_declarable(tmp_path: Path) -> None:
    """Either half of a range rule is optional, and a max-only bound is the
    natural spelling for 'reject the spikes above the calibration range'."""
    generated = generate(_config())
    truth = generated.streams["analyzer"]["ch4"].values
    cutoff = float(np.median(truth))
    manifest_path = export_raw(generated, tmp_path / "export", qaqc_bounds={"ch4": (None, cutoff)})
    ingested = ingest_campaign(load_manifest(manifest_path))["analyzer"]
    np.testing.assert_array_equal(np.isnan(ingested["ch4"].values), truth > cutoff)


# ---------------------------------------------------------------------------
# Temporal support: what the exported archive says about its own cells
# ---------------------------------------------------------------------------


def _cell_config(**support: Any) -> SyntheticConfig:
    """One 60 s instrument whose support the test chooses."""
    return SyntheticConfig.model_validate(
        {
            "name": "cells",
            "start": "2026-01-01T00:00:00Z",
            "duration": "20min",
            "seed": 4,
            "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1900.0}}
                }
            },
            "instruments": {
                "slow": {"native_rate": "60s", "support": support, "measures": {"ch4": {}}}
            },
        }
    )


def _export(tmp_path: Path, config: SyntheticConfig, **kwargs: Any) -> tuple[Any, Any]:
    """Export and return (manifest, the slow stream's frame)."""
    manifest_path = export_raw(generate(config), tmp_path / "archive", **kwargs)
    manifest = load_manifest(manifest_path)
    frame = pd.read_csv(tmp_path / "archive" / EXPORT_RAW_DIR / "slow.csv")
    return manifest, frame


def test_declared_support_restates_what_the_generator_did(tmp_path: Path) -> None:
    manifest, frame = _export(
        tmp_path, _cell_config(method="mean", label="start"), support_declaration="declared"
    )
    support = manifest.instruments["slow"].loader.support
    assert (support.label, support.width, support.method) == ("start", "60s", "mean")
    assert STOP_COLUMN not in frame.columns


def test_reported_support_names_columns_the_file_actually_carries(tmp_path: Path) -> None:
    manifest, frame = _export(
        tmp_path, _cell_config(method="mean", label="start"), support_declaration="reported"
    )
    support = manifest.instruments["slow"].loader.support
    assert support.stop_column == STOP_COLUMN
    assert support.start_column is None, "the time column is already the cell start"
    assert support.label is None and support.width == "cadence"
    assert STOP_COLUMN in frame.columns
    stop = pd.to_datetime(frame[STOP_COLUMN])
    start = pd.to_datetime(frame[TIME_COORD])
    assert ((stop - start) == pd.Timedelta("60s")).all()


def test_a_mid_labelled_stream_needs_both_boundary_columns(tmp_path: Path) -> None:
    """The time axis is not the start, so the start has to be given too."""
    manifest, frame = _export(
        tmp_path, _cell_config(method="mean", label="mid"), support_declaration="reported"
    )
    support = manifest.instruments["slow"].loader.support
    assert support.start_column == START_COLUMN
    assert support.stop_column == STOP_COLUMN
    middle = pd.to_datetime(frame[TIME_COORD])
    assert (middle - pd.to_datetime(frame[START_COLUMN]) == pd.Timedelta("30s")).all()


def test_no_declaration_leaves_the_archive_silent(tmp_path: Path) -> None:
    """The negative control: start times written, nothing saying so."""
    manifest, frame = _export(
        tmp_path, _cell_config(method="mean", label="start"), support_declaration="none"
    )
    support = manifest.instruments["slow"].loader.support
    assert (support.label, support.method, support.stop_column) == (None, None, None)
    assert STOP_COLUMN not in frame.columns


def test_the_time_column_carries_the_labelled_instant_not_the_midpoint(
    tmp_path: Path,
) -> None:
    """A start-labelled product writes start times.

    Writing midpoints would hand ingestion the answer, and the label would
    stop being something it has to be told.
    """
    config = _cell_config(method="mean", label="start")
    dataset = generate(config)
    export_raw(dataset, tmp_path / "archive", support_declaration="declared")
    frame = pd.read_csv(tmp_path / "archive" / EXPORT_RAW_DIR / "slow.csv")

    written = np.asarray(pd.to_datetime(frame[TIME_COORD]), dtype="datetime64[ns]")
    stream = dataset.streams["slow"]
    starts = np.asarray(stream[TIME_BOUNDS_VAR].values[:, 0], dtype="datetime64[ns]")
    midpoints = np.asarray(stream[TIME_COORD].values, dtype="datetime64[ns]")
    assert np.array_equal(written, starts)
    assert not np.array_equal(written, midpoints)


def test_a_clock_offset_is_written_wrong_and_declared_right(tmp_path: Path) -> None:
    """Same bargain as raw_units: break the archive, declare the fix.

    The manifest carries the correction a real one would, and the file
    carries timestamps moved the other way, so ingestion applying the
    declared shift has to land back on the generator's truth.
    """
    config = _cell_config()
    dataset = generate(config)
    export_raw(dataset, tmp_path / "plain", support_declaration="declared")
    export_raw(
        dataset, tmp_path / "shifted", support_declaration="declared", time_shift={"slow": "-4s"}
    )
    plain = pd.to_datetime(
        pd.read_csv(tmp_path / "plain" / EXPORT_RAW_DIR / "slow.csv")[TIME_COORD]
    )
    shifted = pd.to_datetime(
        pd.read_csv(tmp_path / "shifted" / EXPORT_RAW_DIR / "slow.csv")[TIME_COORD]
    )
    assert ((shifted - plain) == pd.Timedelta("4s")).all()
    manifest = load_manifest(tmp_path / "shifted" / EXPORT_MANIFEST)
    assert manifest.instruments["slow"].time_shift == "-4s"


def test_the_offset_moves_the_boundaries_too(tmp_path: Path) -> None:
    """A clock offset moves a cell; it does not resize one."""
    _, frame = _export(
        tmp_path,
        _cell_config(method="mean", label="start"),
        support_declaration="reported",
        time_shift={"slow": "-4s"},
    )
    width = pd.to_datetime(frame[STOP_COLUMN]) - pd.to_datetime(frame[TIME_COORD])
    assert (width == pd.Timedelta("60s")).all()


def test_shifting_an_unknown_instrument_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TsaraSyntheticError, match="Cannot export"):
        export_raw(generate(_cell_config()), tmp_path / "a", time_shift={"nope": "1s"})


def test_a_variable_colliding_with_a_boundary_column_is_refused(tmp_path: Path) -> None:
    """`time_stop` is a legal variable name, and writing a cell boundary over a
    measurement would leave the round trip comparing the wrong column. Reached
    here through a renamed measurement, the route a field name cannot take."""
    config = SyntheticConfig.model_validate(
        {
            "name": "clash",
            "start": "2026-01-01T00:00:00Z",
            "duration": "5min",
            "seed": 1,
            "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1.0}}
                }
            },
            "instruments": {
                "slow": {"native_rate": "60s", "measures": {"ch4": {"name": STOP_COLUMN}}}
            },
        }
    )
    with pytest.raises(TsaraSyntheticError, match="would be overwritten"):
        export_raw(generate(config), tmp_path / "a", support_declaration="reported")
    # The other declarations write no such column, so they are unaffected.
    export_raw(generate(config), tmp_path / "b", support_declaration="declared")


def test_a_mobile_archive_can_also_declare_nothing(tmp_path: Path) -> None:
    """The GPS track is manufactured from the platform rather than listed as
    an instrument, so its support declaration takes a separate path and needs
    its own check that 'none' really writes nothing."""
    config = _config(
        platform={
            "kind": "mobile",
            "start_latitude": 40.77,
            "start_longitude": -111.85,
            "speed_m_s": 15.0,
        }
    )
    manifest_path = export_raw(generate(config), tmp_path / "archive", support_declaration="none")
    manifest = load_manifest(manifest_path)
    assert manifest.instruments["gps"].loader.support == SupportSpec()


def test_a_mobile_archive_declares_its_track_support(tmp_path: Path) -> None:
    config = _config(
        platform={
            "kind": "mobile",
            "start_latitude": 40.77,
            "start_longitude": -111.85,
            "speed_m_s": 15.0,
        }
    )
    manifest_path = export_raw(
        generate(config), tmp_path / "archive", support_declaration="declared"
    )
    support = load_manifest(manifest_path).instruments["gps"].loader.support
    assert (support.method, support.label) == ("point", "mid")


def test_a_declared_label_is_read_back_exactly(tmp_path: Path) -> None:
    """The loop closes: what the archive declares, ingestion recovers.

    This replaces a characterization test that pinned the gap while the
    reading side was being built. It asserted the timestamps came back half a
    cell late; the same fixture now asserts they come back exactly.
    """
    config = _cell_config(method="mean", label="start")
    dataset = generate(config)
    manifest_path = export_raw(dataset, tmp_path / "archive", support_declaration="declared")
    ingested = ingest_campaign(load_manifest(manifest_path))

    generated = dataset.streams["slow"]
    stream = ingested["slow"]
    assert np.array_equal(
        np.asarray(stream[TIME_COORD].values, dtype="datetime64[ns]"),
        np.asarray(generated[TIME_COORD].values, dtype="datetime64[ns]"),
    )
    assert np.array_equal(
        np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]"),
        np.asarray(generated[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]"),
    )
    assert stream["ch4"].attrs["cell_methods"] == "time: mean"
    assert stream.attrs["tsara_support_label"] == "start"
    assert stream.attrs["tsara_support_label_provenance"] == "declared"


def test_a_file_that_states_its_cells_is_read_back_exactly(tmp_path: Path) -> None:
    """The strongest rung: boundary columns, so nothing is inferred at all."""
    dataset = generate(_cell_config(method="mean", label="start"))
    manifest_path = export_raw(dataset, tmp_path / "archive", support_declaration="reported")
    stream = ingest_campaign(load_manifest(manifest_path))["slow"]
    assert np.array_equal(
        np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]"),
        np.asarray(dataset.streams["slow"][TIME_BOUNDS_VAR].values, dtype="datetime64[ns]"),
    )
    assert stream.attrs["tsara_support_label_provenance"] == "reported"
    assert stream.attrs["tsara_support_width_provenance"] == "reported"


def test_an_archive_that_says_nothing_is_read_back_wrong_and_says_so(
    tmp_path: Path,
) -> None:
    """The negative control, and the reason the ladder is worth having.

    The same start-labelled product, exported with no declaration, comes back
    half a cell late. Nothing about the data changed; only what the archive
    said about it did. Every field of the stream's provenance admits the
    guess, which is the difference between being wrong and being wrong
    silently.
    """
    dataset = generate(_cell_config(method="mean", label="start"))
    manifest_path = export_raw(dataset, tmp_path / "archive", support_declaration="none")
    stream = ingest_campaign(load_manifest(manifest_path))["slow"]

    generated = np.asarray(dataset.streams["slow"][TIME_COORD].values, dtype="datetime64[ns]")
    read_back = np.asarray(stream[TIME_COORD].values, dtype="datetime64[ns]")
    offset = (generated - read_back) / np.timedelta64(1, "s")
    assert np.all(offset == 30.0), "half a cell, exactly the label being unknown"
    assert stream.attrs["tsara_support_label_provenance"] == "assumed"
    assert stream.attrs["tsara_support_method_provenance"] == "assumed"
    assert stream["ch4"].attrs["cell_methods"] == "time: point"
    # The width is still right, because it is measured from the file's own
    # cadence rather than guessed. This is the ONLY path that exercises that
    # measurement -- the other two declarations take the width from the
    # manifest or from columns -- so without this assertion a wrong cadence
    # would go unnoticed.
    bounds = np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]")
    assert np.all(bounds[:, 1] - bounds[:, 0] == np.timedelta64(60, "s"))


def test_a_declared_clock_offset_is_applied_and_recorded(tmp_path: Path) -> None:
    """The archive is written wrong and declares the fix; ingestion applies
    it and says it did, which is the guard against correcting twice."""
    dataset = generate(_cell_config(method="mean", label="start"))
    manifest_path = export_raw(
        dataset,
        tmp_path / "archive",
        support_declaration="declared",
        time_shift={"slow": "-4s"},
    )
    stream = ingest_campaign(load_manifest(manifest_path))["slow"]
    assert np.array_equal(
        np.asarray(stream[TIME_COORD].values, dtype="datetime64[ns]"),
        np.asarray(dataset.streams["slow"][TIME_COORD].values, dtype="datetime64[ns]"),
    )
    assert stream.attrs["tsara_time_shift"] == "-4s"


def test_an_archive_written_in_local_time_round_trips(tmp_path: Path) -> None:
    """Boundaries are parsed on the same convention as the axis they bound.

    Added after a mutation test: dropping the declared timezone from
    cell-boundary parsing changed nothing any test could see, because the
    exporter only ever wrote UTC. Written in a real zone, a boundary parsed
    on the wrong convention lands hours from the start it belongs to.
    """
    dataset = generate(_cell_config(method="mean", label="start"))
    manifest_path = export_raw(
        dataset,
        tmp_path / "archive",
        support_declaration="reported",
        timezone="Etc/GMT-2",
    )
    # The file really is in local time, or this proves nothing.
    written = pd.read_csv(tmp_path / "archive" / EXPORT_RAW_DIR / "slow.csv")
    first = pd.Timestamp(written[TIME_COORD].iloc[0])
    generated = dataset.streams["slow"]
    truth = np.asarray(generated[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]")
    assert first - pd.Timestamp(truth[0, 0].item()) == pd.Timedelta("2h")

    stream = ingest_campaign(load_manifest(manifest_path))["slow"]
    assert np.array_equal(
        np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]"),
        np.asarray(generated[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]"),
    )


def test_files_of_one_instrument_keep_their_own_cadences(tmp_path: Path) -> None:
    """The decision the archive census makes load-bearing, now round-trip
    testable: some met records run at 1 s in one file and 5 s in another, and
    one instrument-wide cadence would give a whole file the wrong width.

    Written as two files, the second taking every third row.
    """
    dataset = generate(_cell_config(method="mean"))
    manifest_path = export_raw(
        dataset,
        tmp_path / "archive",
        support_declaration="none",
        split={"slow": (1, 3)},
    )
    written = sorted(p.name for p in (tmp_path / "archive" / EXPORT_RAW_DIR).glob("slow*.csv"))
    assert written == ["slow_000.csv", "slow_001.csv"]

    stream = ingest_campaign(load_manifest(manifest_path))["slow"]
    bounds = np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]")
    widths = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
    # 60 s cells from the first file, 180 s from the decimated second.
    assert set(np.unique(widths)) == {60.0, 180.0}
    # And so the instrument has no single nominal width to report.
    assert "tsara_nominal_cell_width_s" not in stream.attrs


def test_a_zero_width_cell_in_a_real_file_is_repaired(tmp_path: Path) -> None:
    """Round-trip cover for the repair, which was previously checked only
    against expectations written by the same person who wrote it."""
    dataset = generate(_cell_config(method="mean", label="start"))
    manifest_path = export_raw(
        dataset,
        tmp_path / "archive",
        support_declaration="reported",
        zero_width_cells={"slow": 3},
    )
    frame = pd.read_csv(tmp_path / "archive" / EXPORT_RAW_DIR / "slow.csv")
    degenerate = (pd.to_datetime(frame[STOP_COLUMN]) == pd.to_datetime(frame[TIME_COORD])).sum()
    assert degenerate == 3, "the fixture must really contain them"

    stream = ingest_campaign(load_manifest(manifest_path))["slow"]
    bounds = np.asarray(stream[TIME_BOUNDS_VAR].values, dtype="datetime64[ns]")
    assert np.all(bounds[:, 1] > bounds[:, 0]), "none left with zero duration"
    assert stream.attrs["tsara_cells_widened"] == 3.0


def test_exporting_for_an_instrument_that_does_not_exist_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(TsaraSyntheticError, match="Cannot export"):
        export_raw(generate(_cell_config()), tmp_path / "a", split={"nope": (1, 2)})
