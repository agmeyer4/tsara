"""Persisting ingested streams as a TSARA bundle.

Layout (the shared parts come from :mod:`tsara.core.bundle`)::

    <bundle>/
        bundle.json        # what this bundle contains
        manifest.yaml      # the resolved manifest that produced it
        streams/
            <instrument>.nc

Why this ships now rather than with the Phase-9 pipeline: every stage
product gains persistence in the phase that introduces it (CLAUDE.md §5).
Ingesting a few thousand files is the slowest step in the whole workflow, so
being able to do it once and reload in a notebook — or resume after a
cluster job dies — is worth more here than anywhere else.

Why the manifest is written alongside
-------------------------------------
A stream without its manifest is uninterpretable in the ways that matter: it
cannot tell you which QA/QC rules were applied, what a variable's units were
before conversion, or why a species carries no sigma. Saving the *resolved*
manifest — after ``base_path`` has been made absolute — also makes the
bundle a record of what actually ran, not of what a relative path happened
to mean in someone's working directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import xarray as xr
import yaml

from tsara import __version__
from tsara.config.manifest import Manifest
from tsara.core.bundle import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_MANIFEST,
    BUNDLE_STREAMS_DIR,
    TsaraBundleError,
)
from tsara.ingest.campaign import StreamCollection

logger = logging.getLogger(__name__)

__all__ = ["BUNDLE_MANIFEST_CONFIG", "load_streams", "save_streams"]

#: The resolved manifest, written beside the streams it produced.
BUNDLE_MANIFEST_CONFIG = "manifest.yaml"

#: Value of ``stage`` in ``bundle.json`` for bundles this module writes.
_STAGE = "ingest"


def save_streams(collection: StreamCollection, path: str | Path) -> Path:
    """Write a :class:`~tsara.ingest.campaign.StreamCollection` to disk.

    Parameters
    ----------
    collection : StreamCollection
        Streams and the manifest that produced them.
    path : str or pathlib.Path
        Bundle directory. Created if absent; stream files with the same
        names are overwritten.

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

    # mode="json" so Paths and any datetimes become YAML-safe scalars.
    payload = collection.manifest.model_dump(mode="json", exclude_none=False)
    (bundle / BUNDLE_MANIFEST_CONFIG).write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    for name, stream in collection.streams.items():
        # Instrument names are validated as Python identifiers by the config
        # layer, so they are safe as filenames: no separators, no '..', no
        # spaces. That validation is what lets this be a bare f-string.
        #
        # Invariant relied on here: every attr written by stream assembly is
        # a netCDF-safe scalar (str, int or float). Booleans and None are not
        # valid netCDF attribute types, which is why `circular` is stored as
        # int and why optional attrs are omitted rather than written as None.
        stream.to_netcdf(streams_dir / f"{name}.nc", engine="netcdf4")

    (bundle / BUNDLE_MANIFEST).write_text(
        json.dumps(
            {
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "tsara_version": __version__,
                "stage": _STAGE,
                "campaign": collection.manifest.name,
                "streams": sorted(collection.streams),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("Wrote ingest bundle to %s (%d streams).", bundle, len(collection.streams))
    return bundle


def load_streams(path: str | Path) -> StreamCollection:
    """Read a bundle written by :func:`save_streams`.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory.

    Returns
    -------
    StreamCollection
        The streams and their manifest.

    Raises
    ------
    TsaraBundleError
        If the directory is missing, incomplete, written by an incompatible
        bundle format version, or written by a different stage.
    """
    bundle = Path(path)
    if not bundle.is_dir():
        raise TsaraBundleError(f"Bundle path '{bundle}' is not an existing directory.")

    descriptor = _read_descriptor(bundle)
    manifest = _read_manifest(bundle)

    streams: dict[str, xr.Dataset] = {}
    for name in descriptor.get("streams", []):
        target = bundle / BUNDLE_STREAMS_DIR / f"{name}.nc"
        if not target.is_file():
            raise TsaraBundleError(
                f"Bundle '{bundle}' lists stream '{name}' but '{target}' is missing."
            )
        # Loaded eagerly rather than lazily: a StreamCollection is handed
        # around and sliced freely, and a lazily-opened file that closes
        # underneath it fails far from here.
        with xr.open_dataset(target, engine="netcdf4") as stream:
            streams[name] = stream.load()

    logger.info("Loaded ingest bundle from %s (%d streams).", bundle, len(streams))
    return StreamCollection(streams=streams, manifest=manifest)


def _read_descriptor(bundle: Path) -> dict[str, Any]:
    """Read and check ``bundle.json``."""
    target = bundle / BUNDLE_MANIFEST
    if not target.is_file():
        raise TsaraBundleError(f"'{target}' is missing; this is not a TSARA bundle.")
    try:
        descriptor = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TsaraBundleError(f"'{target}' is not valid JSON: {exc}") from exc

    version = descriptor.get("bundle_format_version")
    if version != BUNDLE_FORMAT_VERSION:
        # The whole point of writing a version is to refuse rather than
        # misinterpret; a layout change that silently half-loads would be
        # worse than not loading at all.
        raise TsaraBundleError(
            f"Bundle '{bundle}' has format version {version!r}, but this TSARA "
            f"reads version {BUNDLE_FORMAT_VERSION}."
        )
    stage = descriptor.get("stage")
    if stage != _STAGE:
        raise TsaraBundleError(
            f"Bundle '{bundle}' was written by the '{stage}' stage, not "
            f"'{_STAGE}'. Use that stage's loader instead."
        )
    return dict(descriptor)


def _read_manifest(bundle: Path) -> Manifest:
    """Read and validate the manifest stored beside the streams."""
    target = bundle / BUNDLE_MANIFEST_CONFIG
    if not target.is_file():
        raise TsaraBundleError(f"'{target}' is missing; the bundle is incomplete.")
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
        return Manifest.model_validate(payload)
    except Exception as exc:
        raise TsaraBundleError(f"Could not read the manifest in '{target}': {exc}") from exc
