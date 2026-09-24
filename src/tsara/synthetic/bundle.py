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

The atmosphere is rebuilt, not stored
-------------------------------------
A generated dataset carries the :class:`~tsara.synthetic.atmosphere.Atmosphere`
its streams sampled, and a loaded one gets it back without a file of its own:
the generator draws the atmosphere first, before any platform or instrument,
so replaying :func:`~tsara.synthetic.atmosphere.realize_atmosphere` on a fresh
generator seeded from the saved config reproduces it exactly. The one thing a
config cannot supply is a bootstrap background's real-data profile, which is
never serialized (METHODS.md §8.4); pass it to the loader, or the dataset
loads with its streams and answer key and no atmosphere.

A bundle written before Phase 4.5 describes its campaign in a schema with no
atmosphere at all, and is refused with a message saying so rather than a
wall of validation errors. There is deliberately no converter: such a bundle
is regenerated from a current config instead.

Why ``load_bundle`` and not ``load_synthetic``
----------------------------------------------
That name is already taken by :func:`tsara.config.loader.load_synthetic`,
which reads a *config* — the YAML description of a dataset to manufacture —
while this one reads a *dataset*, the manufactured result. Both would accept
a ``str | Path`` and return something plausible-looking, so a notebook line
like ``load_synthetic("run/")`` would have meant entirely different things
depending on which import was in scope. "Bundle" is the noun this directory
convention already uses everywhere else (:class:`TsaraBundleError`, the
``BUNDLE_*`` constants, CLAUDE.md §5), and it generalizes: later phases will
save bundles that are not synthetic at all.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tsara import __version__
from tsara.config.loader import read_yaml
from tsara.config.synthetic import BootstrapBackground, SyntheticConfig
from tsara.core.bundle import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_MANIFEST,
    BUNDLE_STAGE_KEY,
    BUNDLE_STREAMS_DIR,
    BUNDLE_VERSION_WITH_CELLS,
    BUNDLE_VERSION_WITH_READINGS_AND_PROVENANCE,
    SUPPORTED_BUNDLE_VERSIONS,
    TsaraBundleError,
    pin_time_encoding,
    rename_retired_attrs,
)
from tsara.core.support import check_bounds_intact, ensure_time_bounds
from tsara.synthetic.plumes import GroundTruth

if TYPE_CHECKING:  # pragma: no cover
    from tsara.synthetic.atmosphere import Atmosphere
    from tsara.synthetic.generator import SyntheticDataset
    from tsara.synthetic.profiling import RealDataProfile

logger = logging.getLogger(__name__)

#: Re-exported so that callers of this module see one bundle vocabulary
#: rather than having to know which names are shared with other stages.
__all__ = [
    "BUNDLE_CONFIG",
    "BUNDLE_FORMAT_VERSION",
    "BUNDLE_GROUND_TRUTH",
    "BUNDLE_MANIFEST",
    "BUNDLE_STAGE_KEY",
    "BUNDLE_STREAMS_DIR",
    "TsaraBundleError",
    "load_bundle",
    "save_bundle",
]

#: Files this stage adds on top of the shared layout (tsara.core.bundle).
BUNDLE_CONFIG = "config.yaml"
BUNDLE_GROUND_TRUTH = "ground_truth.parquet"

#: Value of the shared ``stage`` key for bundles this module writes.
_STAGE = "synthetic"


def save_bundle(dataset: SyntheticDataset, path: str | Path) -> Path:
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
        # Pinned before writing so `time` and `time_bnds` cannot end up with
        # different reference epochs, and checked so that a stream whose
        # bounds were destroyed upstream is caught here rather than
        # inherited by everything downstream.
        check_bounds_intact(stream)
        pin_time_encoding(stream)
        stream.to_netcdf(target, engine="netcdf4")

    (bundle / BUNDLE_MANIFEST).write_text(
        json.dumps(
            {
                "bundle_format_version": BUNDLE_FORMAT_VERSION,
                "tsara_version": __version__,
                BUNDLE_STAGE_KEY: _STAGE,
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


def load_bundle(
    path: str | Path, profiles: Mapping[str, RealDataProfile] | None = None
) -> SyntheticDataset:
    """Read a bundle written by :func:`save_bundle`.

    Streams are loaded eagerly (``load()``) rather than left lazily bound to
    open file handles, so the returned object stays valid after the files
    move or the process changes directory — the behaviour a notebook user
    expects.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory.
    profiles : mapping of str to RealDataProfile, optional
        Real-data profiles for the campaign's bootstrap backgrounds, if it
        has any. Needed only to rebuild the atmosphere; without them the
        dataset loads with ``atmosphere`` None.

    Returns
    -------
    SyntheticDataset
        The round-tripped dataset, with its atmosphere rebuilt from the config.

    Raises
    ------
    TsaraBundleError
        If the directory is missing, incomplete, written by an incompatible
        bundle format version, or describes its campaign in the schema that
        predates the atmosphere.
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
    if found_version not in SUPPORTED_BUNDLE_VERSIONS:
        raise TsaraBundleError(
            f"Bundle '{bundle}' has format version {found_version!r}, but this "
            f"TSARA reads versions {list(SUPPORTED_BUNDLE_VERSIONS)}."
        )

    # Checked before the stage-specific files, so that pointing this loader
    # at another stage's bundle says so plainly instead of reporting a
    # missing 'config.yaml' -- a message that reads like a corrupt synthetic
    # bundle rather than a perfectly good bundle of the wrong kind.
    found_stage = manifest.get(BUNDLE_STAGE_KEY)
    if found_stage != _STAGE:
        raise TsaraBundleError(
            f"Bundle '{bundle}' was written by the '{found_stage}' stage, not "
            f"'{_STAGE}'. Use that stage's loader instead."
        )

    config_path = bundle / BUNDLE_CONFIG
    if not config_path.is_file():
        raise TsaraBundleError(f"Bundle '{bundle}' is missing {BUNDLE_CONFIG}.")
    # The same door `load_synthetic` uses for a user's file, so a hand-edited
    # copy with a key written twice is refused rather than read last-wins.
    payload = read_yaml(config_path)
    _refuse_the_schema_before_the_atmosphere(payload, bundle)
    config = SyntheticConfig.model_validate(payload)

    truth_path = bundle / BUNDLE_GROUND_TRUTH
    if not truth_path.is_file():
        raise TsaraBundleError(f"Bundle '{bundle}' is missing {BUNDLE_GROUND_TRUTH}.")
    ground_truth = GroundTruth.from_frame(pd.read_parquet(truth_path))

    # See the same gate in `tsara.ingest.bundle`: an absent `time_bnds` means
    # "the format could not record one" in version 1 and "nothing could be
    # known" from version 2 on, and only the first is safe to complete.
    predates_cells = int(found_version) < BUNDLE_VERSION_WITH_CELLS
    # Older bundles spell some attributes the way TSARA used to; see
    # `RETIRED_ATTR_NAMES`. Respelled before the cells migration so it sees
    # current names only.
    predates_vocabulary = int(found_version) < BUNDLE_VERSION_WITH_READINGS_AND_PROVENANCE
    respelled: list[str] = []

    streams: dict[str, xr.Dataset] = {}
    migrated: list[str] = []
    for name in manifest.get("streams", []):
        stream_path = bundle / BUNDLE_STREAMS_DIR / f"{name}.nc"
        if not stream_path.is_file():
            raise TsaraBundleError(
                f"Bundle '{bundle}' lists stream '{name}' but '{stream_path.name}' is missing."
            )
        # decode_coords="all" so the CF `bounds` attribute is honoured and
        # `time_bnds` returns as a coordinate, the shape it was saved in.
        with xr.open_dataset(stream_path, engine="netcdf4", decode_coords="all") as opened:
            streams[name] = opened.load()
        if predates_vocabulary and rename_retired_attrs(streams[name]):
            respelled.append(name)
        if predates_cells and ensure_time_bounds(streams[name]):
            migrated.append(name)

    if respelled:
        logger.info(
            "Renamed attributes to the current vocabulary in %d stream(s) from a "
            "format-%d bundle: %s.",
            len(respelled),
            int(found_version),
            ", ".join(respelled),
        )
    if migrated:
        # Said out loud rather than applied quietly: the reading is a weak one
        # and a user comparing results against a freshly generated bundle
        # deserves to know which streams got assumed cells.
        logger.info(
            "Attached assumed cells (cadence width, centred) to %d stream(s) "
            "written before cell boundaries existed: %s.",
            len(migrated),
            ", ".join(migrated),
        )
    logger.info("Loaded synthetic bundle from %s (%d streams).", bundle, len(streams))
    return SyntheticDataset(
        streams=streams,
        ground_truth=ground_truth,
        config=config,
        atmosphere=_rebuild_atmosphere(config, profiles),
    )


def _refuse_the_schema_before_the_atmosphere(payload: dict[str, Any], bundle: Path) -> None:
    """Say plainly that a pre-Phase-4.5 bundle is from another schema.

    Left to validation, such a config fails with one error per misplaced key
    -- ``atmosphere`` missing, ``sources`` and every instrument's ``species``
    unexpected -- which reads like a corrupt file rather than an old one.
    The payload is a mapping by construction: ``read_yaml`` refused anything
    else before this is reached.
    """
    if "atmosphere" in payload:
        return
    instruments = payload.get("instruments")
    old_instruments = isinstance(instruments, dict) and any(
        isinstance(spec, dict) and "species" in spec for spec in instruments.values()
    )
    if "sources" in payload or old_instruments:
        raise TsaraBundleError(
            f"Bundle '{bundle}' was written before the synthetic atmosphere existed: its "
            f"{BUNDLE_CONFIG} gives each instrument its own 'species' and lists 'sources' "
            "at the top level. TSARA no longer reads that schema. Regenerate the dataset "
            "from a config in the current form, where the fields and sources sit under "
            "'atmosphere' and each instrument lists the fields it 'measures'."
        )


def _rebuild_atmosphere(
    config: SyntheticConfig, profiles: Mapping[str, RealDataProfile] | None
) -> Atmosphere | None:
    """Replay the atmosphere's draws on a fresh generator from the saved seed.

    Exact, because the generator makes those draws before any other. A
    campaign with a bootstrap background whose profile was not supplied has
    no way to be rebuilt, and gets None with a log line saying why rather
    than an error: its streams and answer key are complete without it.
    """
    import numpy as np

    from tsara.synthetic.atmosphere import realize_atmosphere

    needed = sorted(
        {
            spec.background.profile
            for spec in config.atmosphere.fields.values()
            if isinstance(spec.background, BootstrapBackground)
        }
        - set(profiles or {})
    )
    if needed:
        logger.info(
            "Loaded without its atmosphere: bootstrap profile(s) %s were not supplied, "
            "and a real-data profile is never saved in a bundle. Pass profiles= to "
            "rebuild it.",
            needed,
        )
        return None
    return realize_atmosphere(config, np.random.default_rng(config.seed), profiles)
