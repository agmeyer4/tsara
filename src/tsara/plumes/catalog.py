"""The plume catalog: one row per event, variable and sweep point, and the tree.

The plume state holds events as a number at every reading; the catalog holds
them as rows (``docs/METHODS.md`` §6.8): one long table keyed by
``event_id``, the sweep coordinates as columns, and every column that means
what a column of the generator's answer key means spelled as the answer key
spells it (``tsara.synthetic.plumes.GROUND_TRUTH_COLUMNS``: ``event_id``,
``parent_event_id``, ``instrument``, ``species``, ``field``, ``start_time``,
``peak_time``, ``end_time``, ``latitude``, ``longitude``), so that scoring a
run against its truth is a join. Later stages add tables keyed by
``event_id`` rather than editing this one; that key is the room §7 keeps
for integration.

**The parent–child tree** (:func:`link_parents`). Along the baseline's
window, at a fixed quantile and fixed thresholds, an event links to the
event at the nearest longer window whose interval holds its peak, skipping a
window at which nothing holds it. The link records containment, not origin:
scale is not source (§6.1). It carries both durations, and it is the tree
both of §6.1's candidate readings of a parent's ratio need.

An ``event_id`` is spelled from the instrument, the variable, the sweep point
and the event's number in time order, ``van.ch4/w600s/q0.05/e3/x1/17``, so a
run repeated on the same data and configuration gives the same keys.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tsara.core.naming import (
    BASELINE_STAGE,
    LATITUDE_COORD,
    LONGITUDE_COORD,
    QUANTILE_DIM,
    TIME_COORD,
    WINDOW_DIM,
    enhancement_name,
)
from tsara.core.support import stream_cells
from tsara.core.timebase import NS_PER_S
from tsara.plumes.hysteresis import describe_events
from tsara.plumes.records import TsaraPlumeError
from tsara.plumes.state import ENTER_DIM, EXIT_DIM, PLUMES_STAGE, TRIGGER_ATTR

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import numpy.typing as npt
    import xarray as xr

    from tsara.core.support import CellBounds
    from tsara.plumes.hysteresis import Events

logger = logging.getLogger(__name__)

__all__ = ["CATALOG_COLUMNS", "link_parents", "plume_catalog"]

#: The catalog's columns, in order. The ten the answer key also has are
#: spelled as it spells them.
CATALOG_COLUMNS: tuple[str, ...] = (
    "event_id",
    "parent_event_id",
    "instrument",
    "species",
    "field",
    "baseline_window",
    "baseline_quantile",
    "enter_multiple",
    "exit_multiple",
    "event_number",
    "record",
    "start_time",
    "peak_time",
    "end_time",
    "duration_s",
    "parent_duration_s",
    "n_readings",
    "covered",
    "peak_enhancement",
    "peak_z",
    "clean_level",
    "clean_spread",
    "events_at_point",
    "chance_events_at_point",
    "trigger",
    "trigger_event_id",
    "latitude",
    "longitude",
)

#: Each column's type. Declared rather than inferred so that a catalog of any
#: size, an empty one included, has one shape, and so that it survives a
#: Parquet round trip exactly: left to inference, a column holding only None
#: is pandas' `object` and comes back from Parquet as its string type. Missing
#: text is NaN, as the answer key's own string columns have it.
_DTYPES: dict[str, str] = {
    **dict.fromkeys(
        (
            "event_id",
            "parent_event_id",
            "instrument",
            "species",
            "field",
            "trigger",
            "trigger_event_id",
        ),
        "str",
    ),
    **dict.fromkeys(("event_number", "record", "n_readings", "events_at_point"), "int64"),
    **dict.fromkeys(("start_time", "peak_time", "end_time"), "datetime64[ns]"),
    **dict.fromkeys(
        (
            "baseline_window",
            "baseline_quantile",
            "enter_multiple",
            "exit_multiple",
            "duration_s",
            "parent_duration_s",
            "covered",
            "peak_enhancement",
            "peak_z",
            "clean_level",
            "clean_spread",
            "chance_events_at_point",
            "latitude",
            "longitude",
        ),
        "float64",
    ),
}

#: The columns naming one tree: an event's parent is sought only among
#: events sharing all of these, at longer windows.
_TREE_KEYS = ["instrument", "species", "baseline_quantile", "enter_multiple", "exit_multiple"]

#: A trigger's intervals by sweep point and event number, and its event ids.
_Intervals = dict[
    tuple[str, int, int, int, int],
    tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]", "npt.NDArray[np.int64]", list[str]],
]


def plume_catalog(
    plume_states: Mapping[str, xr.Dataset], baseline_states: Mapping[str, xr.Dataset]
) -> pd.DataFrame:
    """Return every event of a campaign as one long table, its tree linked.

    Parameters
    ----------
    plume_states : mapping of str to xarray.Dataset
        The campaign's plume states, keyed by instrument.
    baseline_states : mapping of str to xarray.Dataset
        The baseline states they were found on, for each event's largest
        enhancement and each variable's field.

    Returns
    -------
    pandas.DataFrame
        One row per event, variable and sweep point, with
        :data:`CATALOG_COLUMNS`, sorted by instrument, variable, sweep point
        and event number.

    Raises
    ------
    TsaraPlumeError
        If a product is not the one expected, an instrument's plume state has
        no baseline state on the same readings, or a triggered variable's
        trigger is not in the catalog.
    """
    frames: list[pd.DataFrame] = []
    intervals: _Intervals = {}
    adopted: list[tuple[str, str]] = []
    for instrument, plume in plume_states.items():
        baseline = _checked_baseline(instrument, plume, baseline_states)
        cells = stream_cells(plume, instrument)
        for variable in _event_variables(plume):
            if TRIGGER_ATTR in plume[f"event_{variable}"].attrs:
                adopted.append((instrument, variable))
                continue
            frames.append(_own_rows(instrument, variable, plume, baseline, cells, intervals))
    for instrument, variable in adopted:
        plume, baseline = plume_states[instrument], baseline_states[instrument]
        cells = stream_cells(plume, instrument)
        frames.append(_adopted_rows(instrument, variable, plume, baseline, cells, intervals))
    # Typed on the way out, below, so an empty catalog needs only its columns.
    frames = [frame for frame in frames if len(frame)]
    catalog = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CATALOG_COLUMNS)
    )
    catalog = catalog.sort_values(
        ["instrument", "species", "baseline_window", *_TREE_KEYS[2:], "event_number"],
        kind="stable",
    ).reset_index(drop=True)
    return link_parents(catalog)[list(CATALOG_COLUMNS)].astype(_DTYPES)


def link_parents(catalog: pd.DataFrame) -> pd.DataFrame:
    """Link each event to the event at the nearest longer window holding its peak.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Rows with at least ``event_id``, ``baseline_window``, ``start_time``,
        ``peak_time``, ``end_time``, ``duration_s`` and the tree's keys
        (instrument, species, quantile, entry and exit).

    Returns
    -------
    pandas.DataFrame
        A copy with ``parent_event_id`` (missing where no longer window holds
        the peak, and at the longest window, as the answer key's own column
        is missing for an event with no parent) and ``parent_duration_s``. A
        window at which nothing holds the peak is skipped for the next
        longer one; an interval holds a peak when start <= peak < end, as a
        cell holds its midpoint.
    """
    # Rows are addressed by position below, whatever the caller's index is.
    out = catalog.reset_index(drop=True)
    parent = np.full(len(out), None, dtype=object)
    parent_duration = np.full(len(out), np.nan)
    if len(out):
        for _, group in out.groupby(_TREE_KEYS, sort=False):
            _link_group(group, parent, parent_duration)
    out["parent_event_id"] = parent
    out["parent_duration_s"] = parent_duration
    out.index = catalog.index
    return out


# ---------------------------------------------------------------------------
# Checking what was handed over
# ---------------------------------------------------------------------------


def _checked_baseline(
    instrument: str, plume: xr.Dataset, baseline_states: Mapping[str, xr.Dataset]
) -> xr.Dataset:
    """Return the instrument's baseline state, refusing a mismatched pair."""
    if plume.attrs.get("tsara_stage") != PLUMES_STAGE:
        raise TsaraPlumeError(
            f"plume_catalog reads plume states, and '{instrument}' has tsara_stage "
            f"'{plume.attrs.get('tsara_stage')}'."
        )
    baseline = baseline_states.get(instrument)
    if baseline is None or baseline.attrs.get("tsara_stage") != BASELINE_STAGE:
        raise TsaraPlumeError(
            f"The plume state of '{instrument}' needs the baseline state it was found on."
        )
    if not np.array_equal(baseline[TIME_COORD].values, plume[TIME_COORD].values):
        raise TsaraPlumeError(
            f"The baseline and plume states of '{instrument}' are not on the same readings."
        )
    return baseline


def _event_variables(plume: xr.Dataset) -> list[str]:
    """Return every variable the plume state holds events of, in order."""
    return [str(n)[len("event_") :] for n in plume.data_vars if str(n).startswith("event_")]


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _own_rows(
    instrument: str,
    variable: str,
    plume: xr.Dataset,
    baseline: xr.Dataset,
    cells: CellBounds,
    intervals: _Intervals,
) -> pd.DataFrame:
    """Return the rows of a variable that found its own events, and keep its intervals."""
    event = _swept(plume[f"event_{variable}"], 5)
    z = _swept(plume[f"z_{variable}"], 3)
    enhancement = _swept(baseline[enhancement_name(variable)], 3)
    level = _swept(plume[f"clean_level_{variable}"], 3)
    spread = _swept(plume[f"clean_spread_{variable}"], 3)
    record = plume[f"record_{variable}"].values
    counts = _swept(plume[f"n_events_{variable}"], 4)
    chance = _swept(plume[f"chance_events_{variable}"], 4)
    frames = []
    for point in np.ndindex(*event.shape[1:]):
        w, q, _, _ = point
        found = describe_events(event[(slice(None), *point)], z[:, w, q], cells)
        ids = _ids(instrument, variable, plume, point, found.number)
        intervals[(f"{instrument}.{variable}", *point)] = (
            found.number,
            found.start_ns,
            found.stop_ns,
            ids,
        )
        frames.append(
            _frame(
                instrument,
                variable,
                plume,
                baseline,
                point,
                found,
                ids,
                record=record[found.peak],
                peak_z=found.peak_score,
                peak_enhancement=enhancement[found.peak, w, q],
                clean_level=level[found.peak, w, q],
                clean_spread=spread[found.peak, w, q],
                events_at_point=counts[point],
                chance_events_at_point=chance[point],
            )
        )
    return _joined(frames)


def _adopted_rows(
    instrument: str,
    variable: str,
    plume: xr.Dataset,
    baseline: xr.Dataset,
    cells: CellBounds,
    intervals: _Intervals,
) -> pd.DataFrame:
    """Return the rows of a variable that took its events from a trigger.

    Each row describes this variable's own readings inside the trigger's
    interval: their count, the share of the interval their cells cover, and
    the largest enhancement among them; no z, no record, no clean air.
    """
    trigger = str(plume[f"event_{variable}"].attrs[TRIGGER_ATTR])
    event = _swept(plume[f"event_{variable}"], 5)
    enhancement = _swept(baseline[enhancement_name(variable)], 3)
    counts = _swept(plume[f"n_events_{variable}"], 4)
    frames = []
    for point in np.ndindex(*event.shape[1:]):
        w, q, _, _ = point
        taken = intervals.get((trigger, *point))
        if taken is None:
            raise TsaraPlumeError(
                f"'{instrument}.{variable}' takes its events from '{trigger}', which is not "
                "in the catalog; pass its plume state too."
            )
        numbers, starts, stops, trigger_ids = taken
        found = describe_events(
            event[(slice(None), *point)],
            enhancement[:, w, q],
            cells,
            intervals=(numbers, starts, stops),
        )
        by_number = dict(zip(numbers.tolist(), trigger_ids, strict=True))
        frame = _frame(
            instrument,
            variable,
            plume,
            baseline,
            point,
            found,
            _ids(instrument, variable, plume, point, found.number),
            record=np.full(found.n, -1),
            peak_z=np.full(found.n, np.nan),
            peak_enhancement=found.peak_score,
            clean_level=np.full(found.n, np.nan),
            clean_spread=np.full(found.n, np.nan),
            events_at_point=counts[point],
            chance_events_at_point=np.nan,
        )
        frame["trigger"] = trigger
        frame["trigger_event_id"] = [by_number[k] for k in found.number.tolist()]
        frames.append(frame)
    return _joined(frames)


def _joined(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate one variable's sweep points, leaving out those with no event."""
    kept = [frame for frame in frames if len(frame)]
    return pd.concat(kept, ignore_index=True) if kept else frames[0]


def _swept(variable: xr.DataArray, ndim: int) -> npt.NDArray[Any]:
    """Return a state variable's values in the catalog's axis order.

    Per reading (time first) for three or five dimensions; per sweep point,
    without time, for four.
    """
    axes = (TIME_COORD, WINDOW_DIM, QUANTILE_DIM, ENTER_DIM, EXIT_DIM)
    order = axes[1:] if ndim == 4 else axes[:ndim]
    return np.asarray(variable.transpose(*order).values)


def _ids(
    instrument: str,
    variable: str,
    plume: xr.Dataset,
    point: tuple[int, ...],
    numbers: npt.NDArray[np.int64],
) -> list[str]:
    """Spell each event's key from its variable, sweep point and number."""
    w, q, e, x = point
    stem = (
        f"{instrument}.{variable}/w{plume[WINDOW_DIM].values[w]:g}s"
        f"/q{plume[QUANTILE_DIM].values[q]:g}/e{plume[ENTER_DIM].values[e]:g}"
        f"/x{plume[EXIT_DIM].values[x]:g}"
    )
    return [f"{stem}/{k}" for k in numbers.tolist()]


def _frame(
    instrument: str,
    variable: str,
    plume: xr.Dataset,
    baseline: xr.Dataset,
    point: tuple[int, ...],
    found: Events,
    ids: list[str],
    **columns: object,
) -> pd.DataFrame:
    """Assemble one sweep point's rows."""
    w, q, e, x = point
    times = plume[TIME_COORD].values
    frame = pd.DataFrame(
        {
            "event_id": ids,
            "instrument": instrument,
            "species": variable,
            "field": str(baseline[variable].attrs.get("field", variable)),
            "baseline_window": float(plume[WINDOW_DIM].values[w]),
            "baseline_quantile": float(plume[QUANTILE_DIM].values[q]),
            "enter_multiple": float(plume[ENTER_DIM].values[e]),
            "exit_multiple": float(plume[EXIT_DIM].values[x]),
            "event_number": found.number,
            "start_time": pd.to_datetime(found.start_ns, unit="ns"),
            "peak_time": pd.to_datetime(times[found.peak]),
            "end_time": pd.to_datetime(found.stop_ns, unit="ns"),
            "duration_s": (found.stop_ns - found.start_ns) / NS_PER_S,
            "n_readings": found.n_readings,
            "covered": found.covered,
            **columns,
            "trigger": None,
            "trigger_event_id": None,
            "latitude": _position(plume, LATITUDE_COORD, found.peak),
            "longitude": _position(plume, LONGITUDE_COORD, found.peak),
        },
        index=pd.RangeIndex(found.n),
    )
    return frame


def _position(plume: xr.Dataset, name: str, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Return a position at each row: a track's value there, a site's everywhere, or NaN.

    A stream carries a position as a site's scalar or a track along time and
    in no other shape, so those are the two read here.
    """
    if name not in plume.coords:
        return np.full(rows.size, np.nan)
    coordinate = plume.coords[name]
    if coordinate.ndim == 0:
        return np.full(rows.size, float(coordinate.values))
    return np.asarray(coordinate.transpose(TIME_COORD).values, dtype=np.float64)[rows]


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


def _link_group(
    group: pd.DataFrame,
    parent: npt.NDArray[np.object_],
    parent_duration: npt.NDArray[np.float64],
) -> None:
    """Link the events of one tree, window by window, writing into the arrays given."""
    windows = np.sort(group["baseline_window"].unique())
    for i, window in enumerate(windows):
        children = group[(group["baseline_window"] == window) & group["peak_time"].notna()]
        waiting = children.index.to_numpy()
        peaks = children["peak_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        for longer in windows[i + 1 :]:
            if waiting.size == 0:
                break
            candidates = group[group["baseline_window"] == longer].sort_values("start_time")
            starts = candidates["start_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            ends = candidates["end_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
            at = np.searchsorted(starts, peaks, side="right") - 1
            safe = np.clip(at, 0, None)
            held = (at >= 0) & (peaks < ends[safe]) if starts.size else np.zeros(peaks.size, bool)
            parent[waiting[held]] = candidates["event_id"].to_numpy()[safe[held]]
            parent_duration[waiting[held]] = candidates["duration_s"].to_numpy()[safe[held]]
            waiting, peaks = waiting[~held], peaks[~held]
