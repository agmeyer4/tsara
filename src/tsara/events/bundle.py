"""Saving and reloading the event states, the catalog and its candidates.

Each instrument's event state is one netCDF file, ``events/<instrument>.nc``,
inside a bundle directory beside whatever else that bundle holds; the catalog
of every event is ``events/catalog.parquet`` beside them, and the table of
its candidates ``events/candidates.parquet``, built from that catalog as it
is written so that the two cannot disagree. The catalog must be the whole
one :func:`~tsara.events.catalog.event_catalog` made from these states,
since a candidate is numbered and counted over every event of its window: a
table built from part of a catalog names different candidates, or counts
them wrongly, under the ids the catalog carries. A filtered or edited
catalog is therefore refused; filter after loading. And the events section of the
analysis configuration that found them is ``events/analysis.yaml``. Only
that section, for the reason the baseline state gives (``docs/METHODS.md``
§6.6): it is all this stage read, and a copy of the whole would stop loading
the first time another stage's settings changed. ``bundle.json`` is not
touched, for the grid's reason (:mod:`tsara.core.bundle`).

Why this ships now rather than with the Phase-9 pipeline: every stage product
gains persistence in the phase that introduces it (CLAUDE.md §5), so that a
notebook can inspect the events and a later job can fit ratios over them
without searching again.

What a reload preserves
-----------------------
Everything: every variable and coordinate of each state exactly, the CF cell
boundaries restored as a coordinate, every attribute, and the catalog and the
candidate table column for column and type for type (both declare their
types, so Parquet has nothing to infer).

A catalog written before candidates existed (Phase 6) has no
``candidate_id`` and no candidate table beside it. It is not refused: its
candidates are linked on load, by the same rule, since they are a function of
the catalog alone and the event states beside it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from tsara.config.analysis import EventsConfig
from tsara.config.loader import read_yaml
from tsara.core.bundle import (
    BUNDLE_ANALYSIS_CONFIG,
    BUNDLE_CANDIDATES_FILE,
    BUNDLE_CATALOG_FILE,
    BUNDLE_EVENTS_DIR,
    TsaraBundleError,
    pin_time_encoding,
)
from tsara.core.naming import QUANTILE_DIM, WINDOW_DIM
from tsara.core.support import check_bounds_intact
from tsara.events.candidates import CANDIDATE_COLUMNS, event_candidates, link_candidates
from tsara.events.catalog import CATALOG_COLUMNS
from tsara.events.records import TsaraEventError
from tsara.events.state import ENTER_DIM, EVENTS_STAGE, EXIT_DIM

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = ["EventStates", "load_events", "save_events"]

#: The catalog's columns before each event named its candidate (Phase 6): a
#: catalog with exactly these is linked on load rather than refused.
_COLUMNS_BEFORE_CANDIDATES = tuple(c for c in CATALOG_COLUMNS if c != "candidate_id")

#: An event state's count of the events of a variable at each sweep point is
#: this prefix and the variable's name (:mod:`tsara.events.state`).
_COUNT_PREFIX = "n_events_"


@dataclass(frozen=True, eq=False)
class EventStates:
    """What a bundle's events directory holds, reloaded.

    Attributes
    ----------
    states : dict of str to xarray.Dataset
        One event state per instrument, keyed by instrument name.
    catalog : pandas.DataFrame or None
        The catalog written beside them, or ``None`` when none was.
    candidates : pandas.DataFrame or None
        The table of the catalog's candidates, or ``None`` when no catalog
        was written.
    events : EventsConfig or None
        The events configuration written beside them, or ``None``.
    """

    states: dict[str, xr.Dataset]
    catalog: pd.DataFrame | None
    candidates: pd.DataFrame | None
    events: EventsConfig | None


def save_events(
    states: Mapping[str, xr.Dataset],
    path: str | Path,
    *,
    catalog: pd.DataFrame | None = None,
    events: EventsConfig | None = None,
    compression: int | None = None,
) -> Path:
    """Write event states, the catalog of their events and its candidates into a bundle.

    Parameters
    ----------
    states : mapping of str to xarray.Dataset
        Event states keyed by instrument, as
        :func:`~tsara.events.state.event_states` returns them.
    path : str or pathlib.Path
        Bundle directory, created if absent. State files for instruments not
        in ``states`` are removed, and so is a catalog left by an earlier run,
        with its candidates, when none is given, so the directory never
        contradicts the run that wrote it; anything else is left alone.
    catalog : pandas.DataFrame, optional
        The catalog (:func:`~tsara.events.catalog.event_catalog`), whole: every
        event of these states, under the candidate ids it was given. Its
        candidate table (:func:`~tsara.events.candidates.event_candidates`)
        is built from it and the states, and written beside it.
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
        columns, names an instrument whose state is not being saved, is not
        every event of the states (filtered) or names candidates other than
        linking it gives (edited), ``path`` exists and is not a directory, or
        ``compression`` is not a level from 1 to 9. Nothing is written then.
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
    candidates: pd.DataFrame | None = None
    if catalog is not None:
        _check_catalog(catalog, "save_events was handed a catalog")
        # Built before anything is written, so a catalog that does not belong
        # to these states leaves the directory as it was.
        candidates = _candidates_of(catalog, states, "save_events was handed a catalog")
        _check_whole(catalog, states)
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
    if catalog is not None and candidates is not None:
        catalog.to_parquet(target / BUNDLE_CATALOG_FILE, index=False)
        candidates.to_parquet(target / BUNDLE_CANDIDATES_FILE, index=False)
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
    """Read the event states, catalog and candidates written by :func:`save_events`.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory, or its ``events`` directory.

    Returns
    -------
    EventStates
        The states keyed by instrument, and the catalog, its candidates and
        the events configuration beside them if they were written. A catalog
        written before candidates existed is given its candidates here.

    Raises
    ------
    TsaraBundleError
        If the directory is missing or holds no state file, a file holds a
        product from a different stage, the catalog or the candidate table
        does not have its columns, or the configuration beside them has no
        ``events`` section or does not validate.
    """
    given = Path(path)
    target = given if given.name == BUNDLE_EVENTS_DIR else given / BUNDLE_EVENTS_DIR
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
    catalog, candidates = _read_catalog(target, states)
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
    return EventStates(states=states, catalog=catalog, candidates=candidates, events=events)


def _candidates_of(
    catalog: pd.DataFrame, states: Mapping[str, xr.Dataset], what: str
) -> pd.DataFrame:
    """Build a catalog's candidate table, refusing a catalog these states did not make."""
    try:
        return event_candidates(catalog, states)
    except TsaraEventError as exc:
        raise TsaraBundleError(f"{what} that does not belong to these event states: {exc}") from exc


def _check_whole(catalog: pd.DataFrame, states: Mapping[str, xr.Dataset]) -> None:
    """Refuse a catalog that is not every event of these states, as it was linked.

    Two tests, because a filter can pass either one alone: the rows at each
    sweep point of each variable must number the events its state counts
    there (keeping one entry threshold leaves every id as linking gives it,
    while each candidate's found and absent counts would describe the part),
    and every row must name the candidate linking the catalog gives (an
    edited id keeps every row).
    """
    keys = ["instrument", "species", WINDOW_DIM, QUANTILE_DIM, ENTER_DIM, EXIT_DIM]
    counted = pd.concat(
        [
            state[name]
            .to_series()
            .rename("events")
            .reset_index()
            .assign(instrument=instrument, species=str(name)[len(_COUNT_PREFIX) :])
            for instrument, state in states.items()
            for name in state.data_vars
            if str(name).startswith(_COUNT_PREFIX)
        ]
    )
    rows = catalog.groupby(keys).size().rename("rows").reset_index()
    points = counted.merge(rows, on=keys, how="outer").fillna({"events": 0, "rows": 0})
    differ = points[points["events"] != points["rows"]]
    if len(differ):
        first = differ.iloc[0]
        raise TsaraBundleError(
            "save_events was handed a catalog that is not every event of these states: at "
            f"{len(differ)} of {len(points)} sweep points its rows differ from the events the "
            f"state counts (first, '{first['instrument']}.{first['species']}' at window "
            f"{first[WINDOW_DIM]:g} s, quantile {first[QUANTILE_DIM]:g}, entry "
            f"{first[ENTER_DIM]:g}, exit {first[EXIT_DIM]:g}: {int(first['rows'])} rows, "
            f"{int(first['events'])} events). A candidate is numbered and counted over the "
            "whole catalog, so a filtered one cannot be saved; save the catalog event_catalog "
            "returned, and filter it after load_events."
        )
    linked = link_candidates(catalog)["candidate_id"].to_numpy()
    edited = int(np.sum(linked != catalog["candidate_id"].to_numpy()))
    if edited:
        raise TsaraBundleError(
            f"save_events was handed a catalog whose candidate ids are not its events': {edited} "
            f"of {len(catalog)} rows name another candidate than linking the catalog gives. "
            "Save the catalog event_catalog returned, and filter it after load_events."
        )


def _read_catalog(
    target: Path, states: Mapping[str, xr.Dataset]
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Read the catalog and its candidate table, linking those of a Phase-6 catalog."""
    catalog_file = target / BUNDLE_CATALOG_FILE
    if not catalog_file.is_file():
        return None, None
    catalog = pd.read_parquet(catalog_file)
    if tuple(catalog.columns) == _COLUMNS_BEFORE_CANDIDATES:
        logger.info(
            "The catalog in %s was written before events named their candidates; "
            "linking them on load.",
            catalog_file,
        )
        catalog = link_candidates(catalog)[list(CATALOG_COLUMNS)]
    _check_catalog(catalog, f"'{catalog_file}'")
    candidates_file = target / BUNDLE_CANDIDATES_FILE
    if not candidates_file.is_file():
        return catalog, _candidates_of(catalog, states, f"'{catalog_file}' is a catalog")
    candidates = pd.read_parquet(candidates_file)
    if tuple(candidates.columns) != CANDIDATE_COLUMNS:
        raise TsaraBundleError(
            f"'{candidates_file}' is a table whose columns are not the candidates'; "
            f"build it with event_candidates. Columns: {list(candidates.columns)[:8]}..."
        )
    return catalog, candidates


def _check_catalog(catalog: pd.DataFrame, what: str) -> None:
    """Refuse a table that does not have the catalog's columns, in order."""
    if tuple(catalog.columns) != CATALOG_COLUMNS:
        raise TsaraBundleError(
            f"{what} whose columns are not the catalog's; build it with event_catalog. "
            f"Columns: {list(catalog.columns)[:8]}..."
        )


def _remove_stale(target: Path, keep: set[str], *, keep_catalog: bool) -> None:
    """Delete state files, and a catalog with its candidates, the run being saved did not write.

    A bundle is the record of what ran; a file left from an earlier run would
    contradict it while looking like part of it.
    """
    for existing in target.glob("*.nc"):
        if existing.is_file() and existing.stem not in keep:
            logger.info("Removing stale event state file %s.", existing)
            existing.unlink()
    for name in (BUNDLE_CATALOG_FILE, BUNDLE_CANDIDATES_FILE):
        stale = target / name
        if not keep_catalog and stale.is_file():
            logger.info("Removing a catalog or its candidates left by an earlier run, %s.", stale)
            stale.unlink()
