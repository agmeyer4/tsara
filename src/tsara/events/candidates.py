"""Candidates: the events of one window that share a peak across the rest of the sweep.

An event is what one sweep point sees (``docs/METHODS.md`` §6.8). At one
baseline window, the other three sweep dimensions mostly see the same things
again, and how they do was measured rather than assumed: a higher entry keeps
the same intervals, those whose peak reaches it, because the detector bridges
dips before it tests entry; a higher exit splits an event into pieces lying
inside it; another quantile moves an event a little. A **candidate** gathers
them: for one variable at one window, the events across quantile, entry and
exit that share a peak. It is a description, as the parent-child tree is, not
a verdict: whether a candidate is a plume is judged later, from its height and
its ratio.

**The rule** (:func:`link_candidates`). Order the window's events by peak z,
the tallest first. The tallest event not yet taken is a seed. At every other
sweep point of that window, the event whose interval holds the seed's peak
joins the seed's candidate when no candidate has taken it and the seed's
interval holds its peak in turn. Repeat until every event is taken. So a
candidate has at most one event per sweep point, every event belongs to one
candidate, and the seed is its tallest. Where a candidate has no event, the
point is **absorbed** (an event there holds the seed's peak but belongs to
another candidate, or peaks outside the seed) or **absent** (nothing there
holds it). The rule has no settings.

It runs one window at a time because a long window's event over a broad plume
peaks on the blip riding on it, so that across windows a blip and its landfill
would share a peak (§6.1: scale is not source). Across windows the tree links
them instead: a candidate's parent is the candidate of its seed's parent.

A variable that takes its events from a trigger takes the trigger's
candidates too, through ``trigger_event_id``, since its rows are the
trigger's intervals.

The candidate table (:func:`event_candidates`) holds one row per candidate,
with its columns spelled as the answer key spells them where the meaning is
the same, and a ``candidate_id`` spelled from the instrument, the variable,
the window and the candidate's number in time order, ``van.ch4/w600s/c17``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tsara.core.naming import QUANTILE_DIM, WINDOW_DIM
from tsara.events.records import TsaraEventError
from tsara.events.state import ENTER_DIM, EVENTS_STAGE, EXIT_DIM

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import numpy.typing as npt
    import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "CANDIDATE_COLUMNS",
    "event_candidates",
    "link_candidates",
]

#: The candidate table's columns, in order. Those the answer key also has are
#: spelled as it spells them.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "parent_candidate_id",
    "instrument",
    "species",
    "field",
    "baseline_window",
    "candidate_number",
    "seed_event_id",
    "start_time",
    "peak_time",
    "end_time",
    "duration_s",
    "n_found",
    "n_absorbed",
    "n_absent",
    "max_peak_z",
    "min_peak_z",
    "peak_enhancement",
    "latitude",
    "longitude",
)

#: Each column's type, declared so that a table of any size, an empty one
#: included, has one shape and survives a Parquet round trip, as the event
#: catalog's does. Missing text is NaN.
_DTYPES: dict[str, str] = {
    **dict.fromkeys(
        (
            "candidate_id",
            "parent_candidate_id",
            "instrument",
            "species",
            "field",
            "seed_event_id",
        ),
        "str",
    ),
    **dict.fromkeys(("candidate_number", "n_found", "n_absorbed", "n_absent"), "int64"),
    **dict.fromkeys(("start_time", "peak_time", "end_time"), "datetime64[ns]"),
    **dict.fromkeys(
        (
            "baseline_window",
            "duration_s",
            "max_peak_z",
            "min_peak_z",
            "peak_enhancement",
            "latitude",
            "longitude",
        ),
        "float64",
    ),
}

#: The columns naming one window: candidates never cross them.
_WINDOW_KEYS = ["instrument", "species", WINDOW_DIM]

#: The columns naming a sweep point within a window, in sweep order.
_POINT_KEYS = [QUANTILE_DIM, ENTER_DIM, EXIT_DIM]

#: The catalog columns the rule reads.
_RULE_COLUMNS = (
    "event_id",
    *_WINDOW_KEYS,
    *_POINT_KEYS,
    "start_time",
    "peak_time",
    "end_time",
    "peak_z",
    "trigger_event_id",
)

#: The catalog columns the table reads besides those.
_TABLE_COLUMNS = ("field", "parent_event_id", "peak_enhancement", "latitude", "longitude")


@dataclass(frozen=True, eq=False)
class _Window:
    """The candidates of one window, as the rule found them.

    ``candidate`` gives each row its candidate's number, 0, 1, ... in the time
    order of the seeds' peaks; the per-candidate arrays are in that order.
    """

    candidate: npt.NDArray[np.int64]
    seed: npt.NDArray[np.int64]
    found: npt.NDArray[np.int64]
    absorbed: npt.NDArray[np.int64]
    min_z: npt.NDArray[np.float64]


def link_candidates(catalog: pd.DataFrame) -> pd.DataFrame:
    """Give every event of a catalog the candidate it belongs to.

    Parameters
    ----------
    catalog : pandas.DataFrame
        An event catalog (:func:`~tsara.events.catalog.event_catalog`), or
        any rows holding the columns the rule reads: ``event_id``,
        ``instrument``, ``species``, the four sweep coordinates,
        ``start_time``, ``peak_time``, ``end_time``, ``peak_z`` and
        ``trigger_event_id``.

    Returns
    -------
    pandas.DataFrame
        A copy with ``candidate_id``, on the caller's index. A row with a
        ``trigger_event_id`` takes its trigger's candidate.

    Raises
    ------
    TsaraEventError
        If a column the rule reads is missing, or a row's trigger event is
        not in the catalog.
    """
    _require(catalog, _RULE_COLUMNS, "link_candidates")
    out = catalog.reset_index(drop=True)
    candidate_id = np.full(len(out), None, dtype=object)
    own = out["trigger_event_id"].isna().to_numpy()
    for key, rows in out[own].groupby(_WINDOW_KEYS, sort=False):
        found = _group_window(rows)
        candidate_id[rows.index.to_numpy()] = _ids(key, found.candidate)
    by_event = dict(zip(out.loc[own, "event_id"], candidate_id[own], strict=True))
    candidate_id[~own] = [_triggered(t, by_event) for t in out.loc[~own, "trigger_event_id"]]
    out["candidate_id"] = candidate_id
    out.index = catalog.index
    return out


def event_candidates(catalog: pd.DataFrame, event_states: Mapping[str, xr.Dataset]) -> pd.DataFrame:
    """Return one row per candidate, from the catalog its events are in.

    The rows are rebuilt from the catalog by the rule itself, so a catalog
    and its candidate table cannot disagree.

    Parameters
    ----------
    catalog : pandas.DataFrame
        An event catalog (:func:`~tsara.events.catalog.event_catalog`).
    event_states : mapping of str to xarray.Dataset
        The event states the catalog was built from, keyed by instrument,
        for the number of sweep points at each window: a point where no event
        holds a candidate's peak is absent, whether or not it found anything.

    Returns
    -------
    pandas.DataFrame
        :data:`CANDIDATE_COLUMNS`, sorted by instrument, variable, window and
        candidate number. Of the seed (the candidate's tallest event): its
        id, peak time, field, largest enhancement and position at the peak;
        over its events: the span they cover from the earliest start to the
        latest end, the sweep points where it has an event (found), where an
        event holds its peak but belongs elsewhere (absorbed) and where none
        does (absent), and its highest and lowest peak z. Its parent is the
        candidate of its seed's parent.

    Raises
    ------
    TsaraEventError
        If a column the table reads is missing, or an instrument's event
        state is missing or is not one.
    """
    _require(catalog, (*_RULE_COLUMNS, *_TABLE_COLUMNS), "event_candidates")
    rows = catalog.reset_index(drop=True)
    rows = rows[rows["trigger_event_id"].isna()]
    frames: list[pd.DataFrame] = []
    candidate_id = pd.Series(index=rows.index, dtype=object)
    for key, window in rows.groupby(_WINDOW_KEYS, sort=False):
        points = _sweep_points(str(key[0]), event_states)
        found = _group_window(window)
        ids = _ids(key, found.candidate)
        candidate_id[window.index] = ids
        frames.append(_frame(window, found, ids, points))
    frames = [frame for frame in frames if len(frame)]
    table = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=[*CANDIDATE_COLUMNS, "parent_event_id"])
    )
    by_event = dict(zip(rows["event_id"], candidate_id, strict=True))
    table["parent_candidate_id"] = table["parent_event_id"].map(by_event)
    table = table.sort_values([*_WINDOW_KEYS, "candidate_number"], kind="stable")
    logger.debug("%d candidates from %d events.", len(table), len(rows))
    return table[list(CANDIDATE_COLUMNS)].reset_index(drop=True).astype(_DTYPES)


# ---------------------------------------------------------------------------
# Checking what was handed over
# ---------------------------------------------------------------------------


def _require(catalog: pd.DataFrame, columns: tuple[str, ...], caller: str) -> None:
    """Refuse rows missing a column the caller reads."""
    missing = [c for c in columns if c not in catalog.columns]
    if missing:
        raise TsaraEventError(
            f"{caller} reads an event catalog; these columns are missing: {missing}."
        )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def _group_window(rows: pd.DataFrame) -> _Window:
    """Apply the rule to the events of one window, in whatever order they come.

    Seeds are taken tallest first; a tie goes to the earlier peak, then to the
    earlier sweep point (quantile, then entry, then exit), so that the result
    does not depend on the order of the rows.
    """
    start = _ns(rows["start_time"])
    stop = _ns(rows["end_time"])
    peak = _ns(rows["peak_time"])
    z = rows["peak_z"].to_numpy(dtype=np.float64)
    point = np.asarray(rows.groupby(_POINT_KEYS, sort=True).ngroup(), dtype=np.int64)
    holder = _holders(point, start, stop, peak)

    owner = np.full(len(rows), -1, dtype=np.int64)
    seeds: list[int] = []
    found: list[int] = []
    absorbed: list[int] = []
    for s in np.lexsort((point, peak, -z)).tolist():
        if owner[s] >= 0:
            continue
        owner[s] = len(seeds)
        seeds.append(s)
        n_found, n_absorbed = 1, 0
        for p, j in enumerate(holder[:, s].tolist()):
            if p == point[s] or j < 0:
                # The seed's own point, or one where nothing holds its peak.
                continue
            if owner[j] < 0 and start[s] <= peak[j] < stop[s]:
                owner[j] = owner[s]
                n_found += 1
            else:
                n_absorbed += 1
        found.append(n_found)
        absorbed.append(n_absorbed)

    # Number the candidates in the time order of their seeds' peaks, the
    # seed's sweep point breaking a tie.
    seed = np.asarray(seeds, dtype=np.int64)
    rank = np.empty(seed.size, dtype=np.int64)
    rank[np.lexsort((point[seed], peak[seed]))] = np.arange(seed.size)
    order = np.argsort(rank)
    min_z = np.full(seed.size, np.inf)
    np.minimum.at(min_z, owner, z)
    return _Window(
        candidate=rank[owner],
        seed=seed[order],
        found=np.asarray(found, dtype=np.int64)[order],
        absorbed=np.asarray(absorbed, dtype=np.int64)[order],
        min_z=min_z[order],
    )


def _ns(column: pd.Series) -> npt.NDArray[np.int64]:
    """Return a time column as integer nanoseconds."""
    return np.asarray(column.to_numpy(dtype="datetime64[ns]").astype(np.int64))


def _holders(
    point: npt.NDArray[np.int64],
    start: npt.NDArray[np.int64],
    stop: npt.NDArray[np.int64],
    peak: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Return, per sweep point and row, the row at that point whose interval holds the row's peak.

    An interval holds a peak when start <= peak < end, as a cell holds its
    midpoint; -1 where none does. The events of one sweep point do not
    overlap, so at most one holds a time; where jittered cells would let two
    touch it, the later-starting one is taken, as a triggered variable's
    reading is given to the later event.
    """
    holder = np.full((int(point.max()) + 1, point.size), -1, dtype=np.int64)
    for p in range(holder.shape[0]):
        at = np.flatnonzero(point == p)
        at = at[np.argsort(start[at], kind="stable")]
        i = np.searchsorted(start[at], peak, side="right") - 1
        held = (i >= 0) & (peak < stop[at[np.maximum(i, 0)]])
        holder[p, held] = at[i[held]]
    return holder


def _ids(key: tuple[Any, ...], candidate: npt.NDArray[np.int64]) -> list[str]:
    """Spell each row's candidate key from its window and the candidate's number."""
    instrument, species, window = key
    stem = f"{instrument}.{species}/w{float(window):g}s"
    return [f"{stem}/c{k}" for k in candidate.tolist()]


def _triggered(trigger_event_id: object, by_event: dict[object, object]) -> object:
    """Return the candidate of a triggered row's trigger event, refusing a missing one."""
    if trigger_event_id not in by_event:
        raise TsaraEventError(
            f"A row takes its events from '{trigger_event_id}', which is not in the catalog; "
            "pass the trigger's events too."
        )
    return by_event[trigger_event_id]


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def _sweep_points(instrument: str, event_states: Mapping[str, xr.Dataset]) -> int:
    """Return the number of sweep points at each window of an instrument's event state."""
    state = event_states.get(instrument)
    if state is None or state.attrs.get("tsara_stage") != EVENTS_STAGE:
        raise TsaraEventError(
            f"event_candidates needs the event state of '{instrument}' the catalog was built from."
        )
    return int(state.sizes[QUANTILE_DIM] * state.sizes[ENTER_DIM] * state.sizes[EXIT_DIM])


def _frame(rows: pd.DataFrame, found: _Window, ids: list[str], points: int) -> pd.DataFrame:
    """Assemble one window's candidate rows, each described by its seed and its events."""
    seed = rows.iloc[found.seed]
    candidate = pd.Series(found.candidate, index=rows.index)
    span = pd.DataFrame(
        {
            "start_time": rows["start_time"].groupby(candidate).min(),
            "end_time": rows["end_time"].groupby(candidate).max(),
        }
    )
    frame = pd.DataFrame(
        {
            "candidate_id": [ids[i] for i in found.seed.tolist()],
            "parent_event_id": seed["parent_event_id"].to_numpy(),
            "instrument": seed["instrument"].to_numpy(),
            "species": seed["species"].to_numpy(),
            "field": seed["field"].to_numpy(),
            WINDOW_DIM: seed[WINDOW_DIM].to_numpy(),
            "candidate_number": np.arange(found.seed.size),
            "seed_event_id": seed["event_id"].to_numpy(),
            "start_time": span["start_time"].to_numpy(),
            "peak_time": seed["peak_time"].to_numpy(),
            "end_time": span["end_time"].to_numpy(),
            "n_found": found.found,
            "n_absorbed": found.absorbed,
            "n_absent": points - found.found - found.absorbed,
            "max_peak_z": seed["peak_z"].to_numpy(),
            "min_peak_z": found.min_z,
            "peak_enhancement": seed["peak_enhancement"].to_numpy(),
            "latitude": seed["latitude"].to_numpy(),
            "longitude": seed["longitude"].to_numpy(),
        }
    )
    frame["duration_s"] = (frame["end_time"] - frame["start_time"]).dt.total_seconds()
    return frame
