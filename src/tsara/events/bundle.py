"""Saving and reloading the event states and the catalog.

Each instrument's event state is one netCDF file, ``events/<instrument>.nc``,
inside a bundle directory beside whatever else that bundle holds; the catalog
of every event is ``events/catalog.parquet`` beside them; and the events
section of the analysis configuration that found them is
``events/analysis.yaml``. Only that section, for the reason the baseline
state gives (``docs/METHODS.md`` §6.6): it is all this stage read, and a copy
of the whole would stop loading the first time another stage's settings
changed. ``bundle.json`` is not touched, for the grid's reason
(:mod:`tsara.core.bundle`).

Why this ships now rather than with the Phase-9 pipeline: every stage product
gains persistence in the phase that introduces it (CLAUDE.md §5), so that a
notebook can inspect the events and a later job can fit ratios over them
without searching again.

What a reload preserves
-----------------------
Everything: every variable and coordinate of each state exactly, the CF cell
boundaries restored as a coordinate, every attribute, and the catalog column
for column and type for type (the catalog declares its types, so Parquet has
nothing to infer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr
import yaml

from tsara.config.analysis import EventsConfig
from tsara.config.loader import read_yaml
from tsara.core.bundle import (
    BUNDLE_ANALYSIS_CONFIG,
    BUNDLE_CATALOG_FILE,
    BUNDLE_EVENTS_DIR,
    TsaraBundleError,
    pin_time_encoding,
)
from tsara.core.support import check_bounds_intact
from tsara.events.catalog import CATALOG_COLUMNS
from tsara.events.state import EVENTS_STAGE

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = ["EventStates", "load_events", "save_events"]


@dataclass(frozen=True, eq=False)
class EventStates:
    """What a bundle's events directory holds, reloaded.

    Attributes
    ----------
    states : dict of str to xarray.Dataset
        One event state per instrument, keyed by instrument name.
    catalog : pandas.DataFrame or None
        The catalog written beside them, or ``None`` when none was.
    events : EventsConfig or None
        The events configuration written beside them, or ``None``.
    """

    states: dict[str, xr.Dataset]
    catalog: pd.DataFrame | None
    events: EventsConfig | None


def save_events(
    states: Mapping[str, xr.Dataset],
    path: str | Path,
    *,
    catalog: pd.DataFrame | None = None,
    events: EventsConfig | None = None,
    compression: int | None = None,
) -> Path:
    """Write event states, and the catalog of their events, into a bundle directory.

    Parameters
    ----------
    states : mapping of str to xarray.Dataset
        Event states keyed by instrument, as
        :func:`~tsara.events.state.event_states` returns them.
    path : str or pathlib.Path
        Bundle directory, created if absent. State files for instruments not
        in ``states`` are removed, and so is a catalog left by an earlier run
        when none is given, so the directory never contradicts the run that
        wrote it; anything else is left alone.
    catalog : pandas.DataFrame, optional
        The catalog (:func:`~tsara.events.catalog.event_catalog`).
    events : EventsConfig, optional
        The configuration the events were found under, written beside them
        under a ``events`` key. Omitted, nothing is written and a note is
        logged.
    compression : int, optional
        zlib level, 1 (fastest) to 9 (smallest), applied to every array of
        each state file, the time axis and bounds included; ``None`` (the
        default) writes uncompressed. Worth it: a state's event index is
        mostly -1 (METHODS.md §6.8 measures by how much).

    Returns
    -------
    pathlib.Path
        The ``events`` directory written.

    Raises
    ------
    TsaraBundleError
        If ``states`` is empty, a dataset is not an event state, a state has
        lost its cell boundaries, the catalog does not have the catalog's
        columns, ``path`` exists and is not a directory, or ``compression``
        is not a level from 1 to 9.
    """
    if not states:
        raise TsaraBundleError("No event states to save.")
    for instrument, state in states.items():
        stage = state.attrs.get("tsara_stage")
        if stage != EVENTS_STAGE:
            # A file this function writes must be one `load_events` reads.
            raise TsaraBundleError(
                f"save_events writes event states, and '{instrument}' has tsara_stage "
                f"'{stage}'. Build one with event_state, or write this dataset with to_netcdf."
            )
    if catalog is not None:
        _check_catalog(catalog, "save_events was handed a catalog")
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
    target = bundle / BUNDLE_EVENTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    for instrument, state in states.items():
        # Checked and pinned before writing, as every bundle writer does.
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
    _remove_stale(target, set(states), keep_catalog=catalog is not None)
    if catalog is not None:
        catalog.to_parquet(target / BUNDLE_CATALOG_FILE, index=False)
    if events is not None:
        payload = {"events": events.model_dump(mode="json", exclude_none=False)}
        (target / BUNDLE_ANALYSIS_CONFIG).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
    else:
        logger.info(
            "Event states written to %s without their events configuration beside them.", target
        )
    logger.info("Wrote %d event state(s) to %s.", len(states), target)
    return target


def load_events(path: str | Path) -> EventStates:
    """Read the event states and catalog written by :func:`save_events`.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory, or its ``events`` directory.

    Returns
    -------
    EventStates
        The states keyed by instrument, and the catalog and events
        configuration beside them if they were written.

    Raises
    ------
    TsaraBundleError
        If the directory is missing or holds no state file, a file holds a
        product from a different stage, the catalog does not have the
        catalog's columns, or the configuration beside them has no
        ``events`` section or does not validate.
    """
    candidate = Path(path)
    target = candidate if candidate.name == BUNDLE_EVENTS_DIR else candidate / BUNDLE_EVENTS_DIR
    if not target.is_dir():
        raise TsaraBundleError(
            f"'{target}' is not an existing directory; this bundle holds no event state."
        )
    files = sorted(target.glob("*.nc"))
    if not files:
        raise TsaraBundleError(f"'{target}' holds no event state file.")
    states: dict[str, xr.Dataset] = {}
    for file in files:
        # `decode_coords="all"` brings `time_bnds` back as a coordinate.
        with xr.open_dataset(file, engine="netcdf4", decode_coords="all") as opened:
            state = opened.load()
        stage = state.attrs.get("tsara_stage")
        if stage != EVENTS_STAGE:
            raise TsaraBundleError(
                f"'{file}' was written by the '{stage}' stage, not '{EVENTS_STAGE}'. "
                "Refusing rather than misreading it as an event state."
            )
        states[file.stem] = state
    catalog: pd.DataFrame | None = None
    catalog_file = target / BUNDLE_CATALOG_FILE
    if catalog_file.is_file():
        catalog = pd.read_parquet(catalog_file)
        _check_catalog(catalog, f"'{catalog_file}'")
    events: EventsConfig | None = None
    config_file = target / BUNDLE_ANALYSIS_CONFIG
    if config_file.is_file():
        try:
            # The one YAML door, so a key written twice is refused here too.
            section = read_yaml(config_file).get("events")
            if section is None:
                raise TsaraBundleError("it has no 'events' section")
            events = EventsConfig.model_validate(section)
        except Exception as exc:
            raise TsaraBundleError(
                f"Could not read the events configuration in '{config_file}': {exc}"
            ) from exc
    logger.info("Loaded %d event state(s) from %s.", len(states), target)
    return EventStates(states=states, catalog=catalog, events=events)


def _check_catalog(catalog: pd.DataFrame, what: str) -> None:
    """Refuse a table that does not have the catalog's columns, in order."""
    if tuple(catalog.columns) != CATALOG_COLUMNS:
        raise TsaraBundleError(
            f"{what} whose columns are not the catalog's; build it with event_catalog. "
            f"Columns: {list(catalog.columns)[:8]}..."
        )


def _remove_stale(target: Path, keep: set[str], *, keep_catalog: bool) -> None:
    """Delete state files, and a catalog, that the run being saved did not write.

    A bundle is the record of what ran; a file left from an earlier run would
    contradict it while looking like part of it.
    """
    for existing in target.glob("*.nc"):
        if existing.is_file() and existing.stem not in keep:
            logger.info("Removing stale event state file %s.", existing)
            existing.unlink()
    stale = target / BUNDLE_CATALOG_FILE
    if not keep_catalog and stale.is_file():
        logger.info("Removing a catalog left by an earlier run, %s.", stale)
        stale.unlink()
