"""Persisting a synthetic dataset as a TSARA bundle directory.

Implements the bundle convention fixed in CLAUDE.md §5 for the products
Phase 2 introduces, and establishes the on-disk layout that later phases
extend rather than replace::

    <bundle>/
        bundle.json          # manifest of what this bundle contains
        config.yaml          # the SyntheticConfig that produced it
        ground_truth.parquet # the answer key (catalog-shaped)
        streams/
            <instrument>.nc  # native-rate per-instrument Datasets

Why each format
---------------
* **netCDF4** for streams — the atmospheric-community interchange standard,
  self-describing, and what ``xarray`` round-trips losslessly including
  attrs and coordinates.
* **Parquet** for the catalog — columnar, typed, and the format CLAUDE.md
  fixes for the Phase 6 ``PlumeCatalog``. Using it here means ground truth
  and detections will be directly comparable on disk with no conversion.
* **YAML** for the config — human-readable and diffable, the same format
  every other TSARA config uses.

Why save/load ships now rather than with the Phase 9 pipeline: every stage
product gains persistence in the phase that introduces it (CLAUDE.md §5), so
that intermediates are inspectable in a notebook, long HPC runs can resume
after a crash, and a generated dataset can be handed to a collaborator as a
single directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from tsara import __version__
from tsara.core.exceptions import TsaraError
from tsara.synthetic.config import SyntheticConfig
from tsara.synthetic.plumes import GroundTruth

if TYPE_CHECKING:  # pragma: no cover
    from tsara.synthetic.generator import SyntheticDataset

logger = logging.getLogger(__name__)

#: Filenames inside a bundle. Centralized so readers and writers cannot drift.
BUNDLE_MANIFEST = "bundle.json"
BUNDLE_CONFIG = "config.yaml"
BUNDLE_GROUND_TRUTH = "ground_truth.parquet"
BUNDLE_STREAMS_DIR = "streams"

#: Bumped only when the layout changes incompatibly, so a future reader can
#: refuse (or migrate) an old bundle rather than misinterpreting it.
BUNDLE_FORMAT_VERSION = 1


class TsaraBundleError(TsaraError):
    """Raised when a TSARA bundle cannot be written or read.

    Distinct from a config error: the configuration may be valid while the
    *directory* is missing, incomplete, or written by an incompatible
    version.
    """


def save_synthetic(dataset: SyntheticDataset, path: str | Path) -> Path:
    """Write a :class:`~tsara.synthetic.generator.SyntheticDataset` to disk.

    Parameters
    ----------
    dataset : SyntheticDataset
        The dataset to persist.
    path : str or pathlib.Path
        Bundle directory. Created if absent; existing stream files with the
        same names are overwritten.

    Returns
    -------
    pathlib.Path
        The bundle directory.

    Raises
    ------
    TsaraBundleError
        If ``path`` exists but is not a directory.
    """
    bundle = Path(path)
    if bundle.exists() and not bundle.is_dir():
        raise TsaraBundleError(f"Bundle path '{bundle}' exists and is not a directory.")
    streams_dir = bundle / BUNDLE_STREAMS_DIR
    streams_dir.mkdir(parents=True, exist_ok=True)

    # Config: mode="json" so datetimes and Paths become YAML-safe scalars.
    config_payload = dataset.config.model_dump(mode="json", exclude_none=False)
    (bundle / BUNDLE_CONFIG).write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )

    # engine is left at pandas' "auto", which resolves to pyarrow — a
    # declared hard dependency of this package (pyproject).
    dataset.ground_truth.to_frame().to_parquet(bundle / BUNDLE_GROUND_TRUTH, index=False)

    for name, stream in dataset.streams.items():
        # Stream names are validated as Python identifiers by the config
        # layer, so they are safe as filenames: no separators, no '..', no
        # spaces. That validation is what lets this be a bare f-string.
        target = streams_dir / f"{name}.nc"
        # Invariant relied on here: every attr the generator emits is a
        # netCDF-safe scalar (str, int, or float). Booleans and None are not
        # valid netCDF attribute types, which is why `circular` is written as
        # int(...) and why no optional field is emitted as a bare None. A new
        # attr that breaks the invariant fails here, at save, with a backend
        # error that does not name the offending key — so keep new attrs to
        # those three types.
        stream.to_netcdf(target, engine="netcdf4")

    (bundle / BUNDLE_MANIFEST).write_text(
        json.dumps(
            {
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "tsara_version": __version__,
                "stage": "synthetic",
                "config_name": dataset.config.name,
                "streams": sorted(dataset.streams),
                "n_ground_truth_rows": len(dataset.ground_truth),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Wrote synthetic bundle to %s (%d streams, %d truth rows).",
        bundle,
        len(dataset.streams),
        len(dataset.ground_truth),
    )
    return bundle


def load_synthetic(path: str | Path) -> SyntheticDataset:
    """Read a bundle written by :func:`save_synthetic`.

    Streams are loaded eagerly (``load()``) rather than left lazily bound to
    open file handles, so the returned object stays valid after the files
    move or the process changes directory — the behaviour a notebook user
    expects.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory.

    Returns
    -------
    SyntheticDataset
        The round-tripped dataset.

    Raises
    ------
    TsaraBundleError
        If the directory is missing, incomplete, or written by an
        incompatible bundle format version.
    """
    import pandas as pd
    import xarray as xr

    from tsara.synthetic.generator import SyntheticDataset

    bundle = Path(path)
    if not bundle.is_dir():
        raise TsaraBundleError(f"Bundle directory '{bundle}' does not exist.")

    manifest_path = bundle / BUNDLE_MANIFEST
    if not manifest_path.is_file():
        raise TsaraBundleError(f"'{bundle}' is not a TSARA bundle: no {BUNDLE_MANIFEST} found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    found_version = manifest.get("bundle_format_version")
    if found_version != BUNDLE_FORMAT_VERSION:
        raise TsaraBundleError(
            f"Bundle '{bundle}' has format version {found_version!r}, but this "
            f"TSARA understands version {BUNDLE_FORMAT_VERSION}."
        )

    config_path = bundle / BUNDLE_CONFIG
    if not config_path.is_file():
        raise TsaraBundleError(f"Bundle '{bundle}' is missing {BUNDLE_CONFIG}.")
    config = SyntheticConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    truth_path = bundle / BUNDLE_GROUND_TRUTH
    if not truth_path.is_file():
        raise TsaraBundleError(f"Bundle '{bundle}' is missing {BUNDLE_GROUND_TRUTH}.")
    ground_truth = GroundTruth.from_frame(pd.read_parquet(truth_path))

    streams: dict[str, xr.Dataset] = {}
    for name in manifest.get("streams", []):
        stream_path = bundle / BUNDLE_STREAMS_DIR / f"{name}.nc"
        if not stream_path.is_file():
            raise TsaraBundleError(
                f"Bundle '{bundle}' lists stream '{name}' but '{stream_path.name}' is missing."
            )
        with xr.open_dataset(stream_path, engine="netcdf4") as opened:
            streams[name] = opened.load()

    logger.info("Loaded synthetic bundle from %s (%d streams).", bundle, len(streams))
    return SyntheticDataset(streams=streams, ground_truth=ground_truth, config=config)
