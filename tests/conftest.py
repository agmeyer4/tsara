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
    """Minimal valid synthetic-dataset configuration."""
    return {
        "name": "cfg",
        "start": "2026-01-01T00:00:00Z",
        "duration": "1h",
        "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
        "instruments": {
            "analyzer": {
                "native_rate": "1s",
                "species": {
                    "ch4": {
                        "background": {"kind": "parametric", "offset": 1900.0},
                        "units": "ppb",
                    }
                },
            }
        },
    }
