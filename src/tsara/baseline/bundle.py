"""Saving and reloading the baseline state.

Each instrument's state is one netCDF file, ``baseline/<instrument>.nc``,
inside a bundle directory beside whatever else that bundle holds; the
analysis configuration that produced them is written beside the files as
``baseline/analysis.yaml``, as the resolved manifest is written beside the
streams. ``bundle.json`` is not touched, for the grid's reason: that
descriptor records which stage created the bundle and what streams it
wrote, and the baseline state is a later stage's product arriving in the
same directory (:mod:`tsara.core.bundle`).

Why this ships now rather than with the Phase-9 pipeline: every stage
product gains persistence in the phase that introduces it (CLAUDE.md §5).
Computing a campaign's baselines is the second slowest step after ingesting it, and a
notebook that inspects the baselines, or a cluster job that fits ratios
from them, should not have to roll again.

What a reload preserves
-----------------------
Everything: every variable and coordinate exactly, including the CF cell
boundaries (restored as a coordinate), every attribute including the
per-sweep-point records that are numeric arrays, and the sweep coordinates
by value. A reloaded state is joined like a stream by :mod:`tsara.align`,
which is what the round-trip test checks last.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
import yaml

from tsara.config.analysis import AnalysisConfig
from tsara.config.loader import read_yaml
from tsara.core.bundle import (
    BUNDLE_ANALYSIS_CONFIG,
    BUNDLE_BASELINE_DIR,
    TsaraBundleError,
    pin_time_encoding,
)
from tsara.core.naming import BASELINE_STAGE
from tsara.core.support import check_bounds_intact

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = ["BaselineStates", "load_state", "save_state"]


@dataclass(frozen=True)
class BaselineStates:
    """What a bundle's baseline directory holds, reloaded.

    Attributes
    ----------
    states : dict of str to xarray.Dataset
        One baseline state per instrument, keyed by instrument name.
    analysis : AnalysisConfig or None
        The analysis configuration written beside them, or ``None`` when the
        states were saved without one.
    """

    states: dict[str, xr.Dataset]
    analysis: AnalysisConfig | None


def save_state(
    states: Mapping[str, xr.Dataset],
    path: str | Path,
    *,
    analysis: AnalysisConfig | None = None,
    compression: int | None = None,
) -> Path:
    """Write baseline states into a bundle directory.

    Parameters
    ----------
    states : mapping of str to xarray.Dataset
        Baseline states keyed by instrument, as :func:`~tsara.baseline.state.baseline_states`
        returns them.
    path : str or pathlib.Path
        Bundle directory. Created if absent. State files for instruments not
        in ``states`` are removed, so the directory never contradicts the
        run that wrote it; anything that is not a ``.nc`` file is left alone.
    analysis : AnalysisConfig, optional
        The configuration the states were computed under, written beside
        them. Omitted, nothing is written and a note is logged: a state
        without its configuration cannot say which sweep it is, beyond what
        its own attributes record.
    compression : int, optional
        zlib level, 1 (fastest) to 9 (smallest), applied to every array in
        each file, the time axis and bounds included; ``None`` (the default)
        writes uncompressed. Worth it: a state is mostly float64 over a sweep.

    Returns
    -------
    pathlib.Path
        The ``baseline`` directory written.

    Raises
    ------
    TsaraBundleError
        If ``states`` is empty, a dataset is not a baseline state, ``path``
        exists and is not a directory, a state has lost its cell boundaries,
        or ``compression`` is not a level from 1 to 9.
    """
    if not states:
        raise TsaraBundleError("No baseline states to save.")
    for instrument, state in states.items():
        stage = state.attrs.get("tsara_stage")
        if stage != BASELINE_STAGE:
            # Refused here for the reason `save_grid` refuses a paired product:
            # a file this function writes must be one `load_state` reads.
            raise TsaraBundleError(
                f"save_state writes baseline states, and '{instrument}' has tsara_stage "
                f"'{stage}'. Build one with baseline_state, or write this dataset with "
                "to_netcdf."
            )
    if compression is not None and (
        isinstance(compression, bool)
        or not isinstance(compression, int)
        or not 1 <= compression <= 9
    ):
        raise TsaraBundleError(
            f"compression must be a zlib level from 1 to 9, or None for none; got {compression!r}."
        )
    bundle = Path(path)
    if bundle.exists() and not bundle.is_dir():
        raise TsaraBundleError(f"Bundle path '{bundle}' exists and is not a directory.")
    target = bundle / BUNDLE_BASELINE_DIR
    target.mkdir(parents=True, exist_ok=True)
    for instrument, state in states.items():
        # Instrument names are identifiers (the config layer validates them),
        # so they are safe as file names. Checked and pinned before writing,
        # as every bundle writer does: a state whose bounds were destroyed
        # upstream must be caught here, not inherited by everything after.
        check_bounds_intact(state)
        pin_time_encoding(state)
        encoding = (
            {
                str(name): {**variable.encoding, "zlib": True, "complevel": compression}
                for name, variable in state.variables.items()
                if variable.dims
            }
            if compression is not None
            else None
        )
        state.to_netcdf(target / f"{instrument}.nc", engine="netcdf4", encoding=encoding)
    _remove_orphan_states(target, set(states))
    if analysis is not None:
        payload = analysis.model_dump(mode="json", exclude_none=False)
        (target / BUNDLE_ANALYSIS_CONFIG).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    else:
        logger.info(
            "Baseline states written to %s without an analysis configuration beside them.",
            target,
        )
    logger.info("Wrote %d baseline state(s) to %s.", len(states), target)
    return target


def _remove_orphan_states(target: Path, keep: set[str]) -> None:
    """Delete state files for instruments the run being saved did not roll.

    The ingest bundle does the same for streams: a bundle is the record of
    what ran, and a file left over from an earlier run with more instruments
    would contradict that record while looking like part of it.
    """
    for existing in target.glob("*.nc"):
        if existing.is_file() and existing.stem not in keep:
            logger.info("Removing stale baseline state file %s.", existing)
            existing.unlink()


def load_state(path: str | Path) -> BaselineStates:
    """Read the baseline states written by :func:`save_state`.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory, or its ``baseline`` directory.

    Returns
    -------
    BaselineStates
        The states, keyed by instrument, and the analysis configuration
        beside them if one was written.

    Raises
    ------
    TsaraBundleError
        If the directory is missing or holds no state file, a file holds a
        product from a different stage, or the configuration beside them
        does not validate.
    """
    candidate = Path(path)
    target = candidate if candidate.name == BUNDLE_BASELINE_DIR else candidate / BUNDLE_BASELINE_DIR
    if not target.is_dir():
        raise TsaraBundleError(
            f"'{target}' is not an existing directory; this bundle holds no baseline state."
        )
    files = sorted(target.glob("*.nc"))
    if not files:
        raise TsaraBundleError(f"'{target}' holds no baseline state file.")
    states: dict[str, xr.Dataset] = {}
    for file in files:
        # `decode_coords="all"` brings `time_bnds` back as a coordinate, where
        # TSARA keeps it; loaded eagerly, since a state is sliced freely.
        with xr.open_dataset(file, engine="netcdf4", decode_coords="all") as opened:
            state = opened.load()
        stage = state.attrs.get("tsara_stage")
        if stage != BASELINE_STAGE:
            raise TsaraBundleError(
                f"'{file}' was written by the '{stage}' stage, not '{BASELINE_STAGE}'. "
                "Refusing rather than misreading it as a baseline state."
            )
        states[file.stem] = state
    analysis: AnalysisConfig | None = None
    config_file = target / BUNDLE_ANALYSIS_CONFIG
    if config_file.is_file():
        try:
            # The same door every configuration comes through, so a
            # hand-edited copy with a key written twice is refused here too.
            analysis = AnalysisConfig.model_validate(read_yaml(config_file))
        except Exception as exc:
            raise TsaraBundleError(
                f"Could not read the analysis configuration in '{config_file}': {exc}"
            ) from exc
    logger.info("Loaded %d baseline state(s) from %s.", len(states), target)
    return BaselineStates(states=states, analysis=analysis)
