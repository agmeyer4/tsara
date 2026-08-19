"""Writing a synthetic dataset back out as raw files plus a matching manifest.

Why this exists
---------------
Ingestion has no other honest way to be checked. The campaign this package
is built for has no plume-free clean stretches and no controlled releases,
so there is no measured record whose right answer is known independently
(CLAUDE.md §5). Injected synthetic truth is the only arbiter available — but
it can only arbitrate *ingestion* if the synthetic data is made to travel
the same road real data does: written to files, discovered by a crawler,
parsed by a reader, converted, masked, and reassembled.

This module closes that loop. It takes a
:class:`~tsara.synthetic.generator.SyntheticDataset` — whose true values,
true error components and true event times are all known — and writes the
archive a real campaign would have produced, together with the manifest that
describes it. Ingesting that archive and comparing against the generator's
answer key exercises every stage of :mod:`tsara.ingest` at once, which no
unit test of an individual stage can do.

It is public rather than a test fixture because the same trick is useful to
users: point this at a generated dataset, ingest it, and you have a working
manifest to copy and a demonstration that your reading of the schema matches
TSARA's.

A mobile platform needs no special handling: the generator already emits
its GPS track as a separate stream on its own clock — the canonical
multi-rate case — so it is written like any other instrument and only needs
*declaring*, since the platform is not listed among the configured
instruments.

Why CSV, and only CSV
---------------------
A round trip proves something about the *reader* only when the writer is
trivially, unarguably correct. Delimited text is: pandas writes it, and the
mapping from value to text and back involves no convention TSARA invented.
An ICARTT writer would be a second piece of TSARA-authored code, and a round
trip through it would demonstrate that the writer and reader agree with each
other — not that either matches the FFI-1001 specification. The ICARTT
reader is instead checked against hand-written fixtures reproducing the
quirks real archives contain, which is the evidence that actually bears on
its correctness.

The one place exactness needs care
----------------------------------
Timestamps are written in ISO 8601 with full precision, because the
generator's jitter puts real information in the nanosecond digits and a
``%f``-style format truncates at microseconds. Floating-point values are
written at ``repr`` precision, which is round-trip exact; note that pandas'
*default* CSV parser is not, so values recovered by ingestion sit within
about one unit in the last place of the originals rather than being bitwise
identical.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tsara.config.manifest import Manifest
from tsara.core.naming import LATITUDE_COORD, LONGITUDE_COORD, TIME_COORD
from tsara.synthetic.config import MobileTrack, StationarySite

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

    from tsara.synthetic.config import SyntheticConfig
    from tsara.synthetic.generator import SyntheticDataset

logger = logging.getLogger(__name__)

__all__ = ["EXPORT_MANIFEST", "EXPORT_RAW_DIR", "export_raw"]

#: Manifest written beside the raw files it describes.
EXPORT_MANIFEST = "manifest.yaml"

#: Subdirectory holding the generated raw files.
EXPORT_RAW_DIR = "raw"


def export_raw(dataset: SyntheticDataset, path: str | Path) -> Path:
    """Write a synthetic dataset as raw CSV files with a matching manifest.

    Parameters
    ----------
    dataset : SyntheticDataset
        Generated data. Only its observable variables are written; the
        ``truth_``-prefixed answer key stays behind, which is what makes the
        result a fair test rather than a leak.
    path : str or pathlib.Path
        Directory to write into. Created if absent.

    Returns
    -------
    pathlib.Path
        Path of the written manifest, ready to hand to
        :func:`tsara.config.loader.load_manifest`.
    """
    root = Path(path)
    raw_dir = root / EXPORT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in dataset.streams:
        _write_csv(dataset.observable(name), raw_dir / f"{name}.csv")

    manifest = _build_manifest(dataset.config, raw_dir)
    manifest_path = root / EXPORT_MANIFEST
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json", exclude_none=False), sort_keys=False),
        encoding="utf-8",
    )

    logger.info(
        "Exported %d synthetic stream(s) as raw CSV to %s with manifest %s.",
        len(dataset.streams),
        raw_dir,
        manifest_path,
    )
    return manifest_path


def _write_csv(stream: xr.Dataset, target: Path) -> None:
    """Write one observable stream as a CSV with an ISO 8601 time column."""
    frame = stream.to_dataframe()
    # `to_dataframe` brings coordinates along as columns. Platform position
    # attached to a gas stream is a coordinate, not a measurement, and would
    # otherwise appear as an undeclared column. Dropped by asking the dataset
    # what its coordinates ARE rather than by name: on the GPS stream itself
    # `latitude`/`longitude` are the measurements, and dropping those by name
    # would export an empty file.
    frame = frame.drop(
        columns=[str(c) for c in stream.coords if c != TIME_COORD and c in frame.columns]
    )
    # Full precision, not `%f`: the generator's timestamp jitter puts real
    # information in the nanosecond digits, and strftime truncates at µs.
    frame.insert(0, TIME_COORD, [stamp.isoformat() for stamp in frame.index])
    frame.to_csv(target, index=False)


def _build_manifest(config: SyntheticConfig, raw_dir: Path) -> Manifest:
    """Build the manifest that describes the files just written."""
    instruments: dict[str, Any] = {}
    for name, instrument in config.instruments.items():
        variables: dict[str, Any] = {}
        for species_name, species in instrument.species.items():
            variable: dict[str, Any] = {
                # Columns are written under their canonical names, so no
                # renaming is exercised here; unit conversion and renaming
                # are covered by the ingest unit tests.
                "column": species_name,
                "role": species.role,
                "units": species.units,
                "circular": species.circular,
            }
            if species.uncertainty is not None:
                # The seam Phase 2 built for exactly this: the generator's
                # true budget expressed as the manifest declaration that
                # should reproduce it.
                variable["uncertainty"] = species.uncertainty.to_manifest_uncertainty().model_dump(
                    mode="json", exclude_none=True
                )
            variables[species_name] = variable
        instruments[name] = {
            "description": f"Synthetic instrument '{name}' at {instrument.native_rate}.",
            "loader": {
                "format": "csv",
                "path_template": f"{name}.csv",
                "time": {"column": TIME_COORD, "format": "iso8601"},
            },
            "variables": variables,
        }

    if isinstance(config.platform, MobileTrack):
        # The generator already emits the track as its own stream at its own
        # rate — the canonical multi-rate case — so it has been written like
        # any other. What is missing is only its *declaration*: the platform
        # is not in `config.instruments`, so nothing above described it.
        gps_instrument = config.platform.gps_instrument
        instruments[gps_instrument] = {
            "description": f"Synthetic GPS track at {config.platform.gps_rate}.",
            "loader": {
                "format": "csv",
                "path_template": f"{gps_instrument}.csv",
                "time": {"column": TIME_COORD, "format": "iso8601"},
            },
            "variables": {
                LATITUDE_COORD: {
                    "column": LATITUDE_COORD,
                    "role": "gps_lat",
                    "units": "degrees_north",
                },
                LONGITUDE_COORD: {
                    "column": LONGITUDE_COORD,
                    "role": "gps_lon",
                    "units": "degrees_east",
                },
            },
        }
        platform: dict[str, Any] = {
            "kind": "mobile",
            "gps_instrument": gps_instrument,
            "lat_variable": LATITUDE_COORD,
            "lon_variable": LONGITUDE_COORD,
        }
    else:
        site = config.platform
        assert isinstance(site, StationarySite)  # noqa: S101 - the only other union member
        platform = {
            "kind": "stationary",
            "latitude": site.latitude,
            "longitude": site.longitude,
        }
        if site.altitude_m is not None:
            platform["altitude_m"] = site.altitude_m

    return Manifest.model_validate(
        {
            "name": config.name,
            "description": f"Manifest for synthetic dataset '{config.name}'.",
            "base_path": str(raw_dir),
            "platform": platform,
            "instruments": instruments,
        }
    )
