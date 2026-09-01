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
from tsara.config.manifest import CSVLoader
from tsara.core.naming import sigma_rand_name, sigma_sys_name
from tsara.ingest import ingest_campaign, load_streams, save_streams
from tsara.synthetic import generate
from tsara.synthetic.background import TsaraSyntheticError
from tsara.synthetic.config import SyntheticConfig
from tsara.synthetic.export import (
    EXPORT_MANIFEST,
    EXPORT_RAW_DIR,
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
        "instruments": {
            "analyzer": {
                "native_rate": "2s",
                "species": {
                    "ch4": {
                        "background": {"kind": "parametric", "offset": 1900.0},
                        "role": "gas",
                        "units": "ppb",
                        "uncertainty": {
                            "random": {"absolute": 0.7},
                            "systematic": {"relative": 0.005},
                        },
                    },
                    "c2h6": {
                        "background": {"kind": "parametric", "offset": 2.0},
                        "role": "gas",
                        "units": "ppb",
                        "uncertainty": {"random": {"absolute": 0.05, "report_as": "c2h6_err"}},
                    },
                },
            },
            "met": {
                "native_rate": "10s",
                "species": {
                    "wind_dir": {
                        "background": {"kind": "parametric", "offset": 180.0},
                        "role": "met",
                        "units": "degrees",
                        "circular": True,
                    }
                },
            },
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

    assert ingested["analyzer"]["c2h6"].attrs["uncertainty_source_random"] == "reported"
    np.testing.assert_allclose(
        ingested["analyzer"][sigma_rand_name("c2h6")].values,
        generated.streams["analyzer"]["truth_sigma_rand_c2h6"].values,
        rtol=RTOL,
    )


def test_provenance_labels_match_what_was_declared(tmp_path: Path) -> None:
    _, ingested = _round_trip(tmp_path)

    ch4 = ingested["analyzer"]["ch4"].attrs
    assert ch4["uncertainty_source_random"] == "declared"
    assert ch4["uncertainty_source_systematic"] == "declared"

    # c2h6 declares only a random component, so its systematic is a
    # deliberate zero rather than unknown.
    c2h6 = ingested["analyzer"]["c2h6"].attrs
    assert c2h6["uncertainty_source_systematic"] == "zero"

    # wind_dir declares no budget at all: random falls back to the empirical
    # estimator and systematic is genuinely unknown.
    wind = ingested["met"]["wind_dir"].attrs
    assert wind["uncertainty_source"] == "empirical"
    assert wind["uncertainty_source_systematic"] == "unknown"
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
    assert reloaded["analyzer"]["ch4"].attrs["uncertainty_source"] == "declared"


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
    spec["instruments"]["analyzer"]["species"]["ch4"]["quantization"] = 0.1
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
