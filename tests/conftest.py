"""Shared fixtures for the TSARA test suite.

The dict fixtures below are *known-good* configurations. Tests that probe
validation failures copy and corrupt them (one field at a time), which keeps
each failure test readable: everything is valid except the one thing under
test.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

import pytest
import yaml

#: The callable `rewrite_as_format` hands back: bundle directory and an older
#: format version in, count of attributes rewritten out.
RewriteBundle: TypeAlias = Callable[[Path, int], int]

#: The callable `write_yaml` hands back. Named rather than spelled inline at
#: every use site so that the tests reading it stay about configuration, and
#: so the signature has one place to change.
WriteYaml: TypeAlias = Callable[..., Path]


@pytest.fixture()
def stationary_manifest_dict() -> dict[str, Any]:
    """Minimal valid manifest: one CSV instrument at a fixed site."""
    return {
        "name": "test_site",
        "base_path": "/data/raw",
        "platform": {
            "kind": "stationary",
            "latitude": 40.766,
            "longitude": -111.847,
            "altitude_m": 1436.0,
        },
        "instruments": {
            "picarro": {
                "loader": {
                    "format": "csv",
                    "path_template": "picarro/%Y/%m/*.dat",
                    "time": {"column": "EPOCH_TIME", "format": "unix"},
                },
                "variables": {
                    "ch4": {
                        "column": "CH4_dry",
                        "role": "gas",
                        "units": "ppm",
                        "convert": {
                            "from_unit": "ppm",
                            "to_unit": "ppb",
                            "scale": 1000.0,
                        },
                        "qaqc": [{"kind": "range", "min": 1700.0}],
                    },
                    "co2": {"column": "CO2_dry", "role": "gas", "units": "ppm"},
                },
            }
        },
    }


@pytest.fixture()
def mobile_manifest_dict(stationary_manifest_dict: dict[str, Any]) -> dict[str, Any]:
    """Valid mobile manifest: gas instrument + GPS instrument, cross-referenced."""
    manifest = copy.deepcopy(stationary_manifest_dict)
    manifest["name"] = "test_mobile"
    manifest["platform"] = {
        "kind": "mobile",
        "gps_instrument": "gps",
        "lat_variable": "latitude",
        "lon_variable": "longitude",
    }
    manifest["instruments"]["gps"] = {
        "loader": {
            "format": "csv",
            "path_template": "gps/%Y/*.csv",
            "time": {"column": "timestamp", "format": "%Y-%m-%d %H:%M:%S"},
        },
        "variables": {
            "latitude": {"column": "lat", "role": "gps_lat", "units": "degrees_north"},
            "longitude": {"column": "lon", "role": "gps_lon", "units": "degrees_east"},
            "wind_dir": {
                "column": "wdir",
                "role": "met",
                "units": "degrees",
                "circular": True,
            },
        },
    }
    return manifest


@pytest.fixture()
def analysis_dict() -> dict[str, Any]:
    """Minimal valid analysis configuration."""
    return {
        "output_grid": {"freq": "1s"},
        "baseline": {"windows": ["2min", "10min"], "quantiles": [0.05]},
        "regression": {"reference_species": "ch4"},
    }


@pytest.fixture()
def write_yaml(tmp_path: Path) -> WriteYaml:
    """Return a helper that writes a dict to a YAML file and returns its path.

    Centralizing this keeps loader tests focused on behavior, not file
    plumbing.
    """

    def _write(data: dict[str, Any], filename: str = "config.yaml") -> Path:
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture()
def synthetic_dict() -> dict[str, Any]:
    """Minimal valid synthetic-dataset configuration: one field, measured once."""
    return {
        "name": "cfg",
        "start": "2026-01-01T00:00:00Z",
        "duration": "1h",
        "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
        "atmosphere": {
            "fields": {
                "ch4": {
                    "units": "ppb",
                    "background": {"kind": "parametric", "offset": 1900.0},
                }
            },
        },
        "instruments": {
            "analyzer": {
                "native_rate": "1s",
                "measures": {"ch4": {}},
            }
        },
    }


@pytest.fixture()
def source_dict() -> dict[str, Any]:
    """Raw dict for a valid source, for validation-failure tests."""
    return {
        "rate_per_hour": 2.0,
        "shape": {"kind": "gaussian", "sigma": "20s"},
        "reference_species": "ch4",
        "amplitude": {"kind": "lognormal", "median": 100.0, "sigma_log": 0.5},
        "ratios": {"c2h6": {"mean": 0.05}},
    }


@pytest.fixture()
def rewrite_as_format() -> RewriteBundle:
    """Return a helper that rewrites a saved stream bundle as an older format wrote it.

    Neither later format changed a value's number, only its labels, so an
    honest older bundle is a current one with those labels put back and the
    version number lowered. Before format 4 an unstated random component
    said `empirical` where it now says `unknown` (and a species with no
    budget at all said `empirical` too); before format 3 some attributes had
    the names in `RETIRED_ATTR_NAMES`, which this helper reads, so a name
    added to that map is exercised without editing it. The count it returns
    lets a test insist that something was rewritten, since a migration test
    over a bundle carrying none of the old labels passes vacuously.
    """
    import json

    import xarray as xr

    from tsara.core.bundle import (
        BUNDLE_MANIFEST,
        BUNDLE_STREAMS_DIR,
        BUNDLE_VERSION_WITH_READINGS_AND_PROVENANCE,
        BUNDLE_VERSION_WITHOUT_PROMISED_ESTIMATES,
        RETIRED_ATTR_NAMES,
    )
    from tsara.core.naming import sigma_rand_name

    current_to_retired = {new: old for old, new in RETIRED_ATTR_NAMES.items()}

    def _rewrite(bundle: Path, version: int) -> int:
        count = 0
        for target in sorted((bundle / BUNDLE_STREAMS_DIR).glob("*.nc")):
            with xr.open_dataset(target, engine="netcdf4", decode_coords="all") as opened:
                stream = opened.load()
            if version < BUNDLE_VERSION_WITHOUT_PROMISED_ESTIMATES:
                # Ingestion never wrote a random `unknown` before format 4, so
                # every one of them was an `empirical` then.
                for name in map(str, stream.data_vars):
                    attrs = stream.variables[name].attrs
                    if sigma_rand_name(name) in stream.variables:
                        continue
                    if attrs.get("uncertainty_provenance_random") != "unknown":
                        continue
                    attrs["uncertainty_provenance_random"] = "empirical"
                    if attrs.get("uncertainty_provenance") == "unknown":
                        attrs["uncertainty_provenance"] = "empirical"
                    count += 1
            if version < BUNDLE_VERSION_WITH_READINGS_AND_PROVENANCE:
                holders = [
                    stream.attrs,
                    *(stream.variables[name].attrs for name in stream.variables),
                ]
                for attrs in holders:
                    for new, old in current_to_retired.items():
                        if new in attrs:
                            attrs[old] = attrs.pop(new)
                            count += 1
            stream.to_netcdf(target, engine="netcdf4")
        descriptor = json.loads((bundle / BUNDLE_MANIFEST).read_text())
        descriptor["bundle_format_version"] = version
        (bundle / BUNDLE_MANIFEST).write_text(json.dumps(descriptor))
        return count

    return _rewrite
