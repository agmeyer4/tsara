"""The event state: records, clean air, the statistic and events, per stream.

For every gas variable of a baseline state (``docs/METHODS.md`` §6.2) this
stage finds the variable's records (:mod:`tsara.events.records`), describes
its plume-free air in each at every sweep point (:mod:`tsara.events.clean`),
forms the statistic z = (Δ − clean level) / clean spread at every reading,
and finds the events at every combination of the baseline's window and
quantile with the events stage's entry and exit multiples
(:mod:`tsara.events.hysteresis`). Everything lands on the stream's own
cells, beside the readings, as the baseline state does (§6.8): one Dataset
per instrument, ``tsara_stage = "events"``.

Where ``events.platform_state`` names a variable of the instrument's stream
(its speed), the stream is handed over too, and records also split where the
platform stays parked for at least ``record_gap`` (§6.8); a moving stretch
never splits, and a reading whose speed is missing takes the state of the
reading before it.

A variable named in ``events.triggers`` finds no events of its own: at each
sweep point it takes the trigger's event intervals, a reading of it
belonging to an event when its cell midpoint lies in that event's interval
(§6.8). Its events keep the trigger's numbers, so the two stay linked.

What the state holds per variable ``x`` found on its own:

- ``record_x`` (time): the record of each reading, -1 for none;
- ``clean_level_x``, ``clean_spread_x``, ``n_clean_readings_x``
  (time, window, quantile): the description of the reading's record at each
  point, and how many readings lay below its level; level and spread blank
  where the record held fewer than ``min_clean_readings``;
- ``z_x`` (time, window, quantile): the statistic;
- ``event_x`` (time, window, quantile, entry, exit): the event each reading
  belongs to at each point, numbered from 0 per point, -1 for none;
- ``n_events_x`` and ``chance_events_x`` (window, quantile, entry, exit):
  the events found, and the events chance alone would make on as many
  independent Gaussian readings: a reference, not a bound, since a record
  whose clean air is described wrongly crosses more often (§6.8);
- ``sigma_rand_x`` when the reading declares or reports one, beside the
  spread for comparison and nothing else: the spread is not a measurement
  uncertainty (§2.3).

A variable with a trigger holds ``event_x`` and ``n_events_x``, and names
its trigger.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from tsara._version import __version__
from tsara.core.bundle import pin_time_encoding
from tsara.core.naming import (
    BASELINE_STAGE,
    ENHANCEMENT_PREFIX,
    QUANTILE_DIM,
    TIME_BOUNDS_VAR,
    TIME_COORD,
    WINDOW_DIM,
    enhancement_name,
    sigma_rand_name,
)
from tsara.core.support import stream_cells
from tsara.events.clean import clean_air
from tsara.events.hysteresis import expected_chance_rate, find_events
from tsara.events.records import TsaraEventError, find_records

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from tsara.config.analysis import EventsConfig
    from tsara.core.support import CellBounds

logger = logging.getLogger(__name__)

__all__ = [
    "CHANCE_ASSUMPTION_ATTR",
    "ENTER_DIM",
    "EVENTS_STAGE",
    "EXIT_DIM",
    "PLATFORM_STATE_ATTR",
    "TRIGGER_ATTR",
    "event_state",
    "event_states",
]

#: What an event state's ``tsara_stage`` says.
EVENTS_STAGE = "events"

#: The events stage's own two sweep dimensions, beside the baseline's.
ENTER_DIM = "enter_multiple"
EXIT_DIM = "exit_multiple"

#: On a variable that takes its events from another: which one.
TRIGGER_ATTR = "tsara_events_trigger"

#: On ``chance_events_x``: the assumption its closed form rests on.
CHANCE_ASSUMPTION_ATTR = "tsara_chance_assumption"

#: On a state whose records split at the platform's state: the rule, as
#: ``'<variable> > <value>'`` (moving).
PLATFORM_STATE_ATTR = "tsara_events_platform_state"

#: The settings a state was found under, as dataset attributes (§6.8).
_SETTING_ATTRS = {
    "record_gap": "tsara_events_record_gap",
    "max_record_length": "tsara_events_max_record_length",
    "min_clean_readings": "tsara_events_min_clean_readings",
    "clean_level_estimator": "tsara_events_clean_level_estimator",
    "max_internal_gap": "tsara_events_max_internal_gap",
    "max_bridged_dropout": "tsara_events_max_bridged_dropout",
}

_CHANCE_ASSUMPTION = (
    "independent Gaussian readings described correctly by the record's clean level "
    "and spread, no dip bridged (METHODS 6.8): oversampled, autocorrelated readings "
    "cross less often and bridging only merges events, while a misdescribed record "
    "or air wandering at the window's scale crosses more often, so this is a "
    "reference, not a bound"
)

#: At most this many variables or patterns are named in one warning.
_MAX_NAMED = 8


def event_state(
    state: xr.Dataset,
    *,
    instrument: str,
    events: EventsConfig,
    variables: Sequence[str] | None = None,
    triggers: Mapping[str, xr.Dataset] | None = None,
    stream: xr.Dataset | None = None,
) -> xr.Dataset:
    """Find one stream's events at every sweep point.

    Parameters
    ----------
    state : xarray.Dataset
        The instrument's baseline state.
    instrument : str
        The stream's name in the campaign, as ``events.triggers`` spells it.
    events : EventsConfig
        Thresholds, records and triggers.
    variables : sequence of str, optional
        Which variables to find events for. ``None`` takes every variable
        the baseline state holds an enhancement of.
    triggers : mapping of str to xarray.Dataset, optional
        For a variable whose trigger lies on another instrument, that
        instrument's event state, keyed by instrument name. A trigger on
        this instrument is taken from this call's own results.
    stream : xarray.Dataset, optional
        The instrument's stream, needed when ``events.platform_state`` names
        this instrument: the state is read from its variable at the
        baseline state's readings, which the stream must all hold.

    Returns
    -------
    xarray.Dataset
        The event state, on the baseline state's ``time`` and ``time_bnds``,
        with the baseline's two sweep dimensions and ``enter_multiple`` and
        ``exit_multiple``; per variable the columns the module docstring
        lists.

    Raises
    ------
    TsaraEventError
        If ``state`` is not a baseline state, a named variable has no
        enhancement, a trigger's event state is missing or was found on
        another sweep, or the platform's state cannot be read from
        ``stream``.
    """
    if state.attrs.get("tsara_stage") != BASELINE_STAGE:
        raise TsaraEventError(
            f"event_state reads a baseline state, and '{instrument}' has tsara_stage "
            f"'{state.attrs.get('tsara_stage')}'. Build one with baseline_state first."
        )
    selection = _select(state, instrument, variables)
    cells = stream_cells(state, instrument)
    enter = np.asarray(events.enter_multiple, dtype=np.float64)
    exit_ = np.asarray(events.exit_multiple, dtype=np.float64)
    own = [v for v in selection if events.trigger_for(instrument, v) is None]
    adopted = [v for v in selection if events.trigger_for(instrument, v) is not None]

    parked = _parked(state, instrument, events, stream) if own else None

    data_vars: dict[str, tuple[tuple[str, ...], npt.NDArray[np.generic], dict[str, object]]] = {}
    blank_everywhere: dict[str, list[tuple[str, int]]] = {}
    for variable in own:
        columns, blank = _own_events(state, variable, cells, events, enter, exit_, parked)
        data_vars.update(columns)
        if blank:
            blank_everywhere[variable] = blank
    for variable in adopted:
        trigger = str(events.trigger_for(instrument, variable))
        membership, source_cells = _trigger_events(
            trigger, instrument, cells, data_vars, triggers, state, events
        )
        data_vars.update(
            _adopted_events(state, variable, cells, trigger, membership, source_cells, instrument)
        )

    coords: dict[str, object] = {
        str(name): state.coords[name] for name in state.coords if name != TIME_BOUNDS_VAR
    }
    coords[ENTER_DIM] = ((ENTER_DIM,), enter, {"units": "1", "long_name": "entry multiple"})
    coords[EXIT_DIM] = ((EXIT_DIM,), exit_, {"units": "1", "long_name": "exit multiple"})
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            **state.attrs,
            "tsara_version": __version__,
            "tsara_stage": EVENTS_STAGE,
            "tsara_instrument": instrument,
            **{attr: str(getattr(events, field)) for field, attr in _SETTING_ATTRS.items()},
        },
    )
    spec = events.platform_state.get(instrument)
    if parked is not None and spec is not None:
        dataset.attrs[PLATFORM_STATE_ATTR] = f"{spec.variable} > {spec.moving_above:g}"
    dataset.coords[TIME_BOUNDS_VAR] = state[TIME_BOUNDS_VAR]
    dataset[TIME_COORD].attrs.update(state[TIME_COORD].attrs)
    if blank_everywhere:
        _warn_blank_everywhere(instrument, blank_everywhere, events.min_clean_readings)
    pin_time_encoding(dataset)
    return dataset


def event_states(
    states: Mapping[str, xr.Dataset],
    events: EventsConfig,
    *,
    streams: Mapping[str, xr.Dataset] | None = None,
) -> dict[str, xr.Dataset]:
    """Find the events of every baseline state of a campaign.

    In two passes, because a trigger may lie on another instrument: first
    every variable that finds its own events, on every instrument; then every
    variable that takes them from a trigger, which is never itself triggered
    (the configuration refuses a chain), so the first pass holds them all.

    Parameters
    ----------
    states : mapping of str to xarray.Dataset
        The campaign's baseline states, keyed by instrument.
    events : EventsConfig
        Thresholds, records and triggers.
    streams : mapping of str to xarray.Dataset, optional
        The campaign's streams, keyed by instrument; read only for an
        instrument ``events.platform_state`` names.

    Returns
    -------
    dict of str to xarray.Dataset
        One event state per instrument that had a variable to find events for.
    """
    first: dict[str, xr.Dataset] = {}
    later: dict[str, list[str]] = {}
    for instrument, state in states.items():
        names = _default_variables(state)
        own = [v for v in names if events.trigger_for(instrument, v) is None]
        later[instrument] = [v for v in names if v not in own]
        if own:
            first[instrument] = event_state(
                state,
                instrument=instrument,
                events=events,
                variables=own,
                stream=(streams or {}).get(instrument),
            )
    result = dict(first)
    for instrument, adopted in later.items():
        if not adopted:
            continue
        taken = event_state(
            states[instrument],
            instrument=instrument,
            events=events,
            variables=adopted,
            triggers=first,
        )
        result[instrument] = (
            first[instrument].assign(taken.data_vars) if instrument in first else taken
        )
    return result


# ---------------------------------------------------------------------------
# Selecting the variables
# ---------------------------------------------------------------------------


def _select(state: xr.Dataset, instrument: str, variables: Sequence[str] | None) -> list[str]:
    """Return the variables to find events for, refusing one with no enhancement."""
    if variables is None:
        return _default_variables(state)
    missing = [v for v in variables if enhancement_name(v) not in state.data_vars]
    if missing:
        raise TsaraEventError(
            f"The baseline state of '{instrument}' holds no enhancement of {missing}; "
            f"it holds enhancements of {_default_variables(state)}."
        )
    return list(variables)


def _default_variables(state: xr.Dataset) -> list[str]:
    """Return every variable the baseline state holds an enhancement of, in order."""
    return [
        str(name)[len(ENHANCEMENT_PREFIX) :]
        for name in state.data_vars
        if str(name).startswith(ENHANCEMENT_PREFIX)
    ]


# ---------------------------------------------------------------------------
# A variable that finds its own events
# ---------------------------------------------------------------------------

_Column = tuple[tuple[str, ...], "npt.NDArray[np.generic]", dict[str, object]]


def _own_events(
    state: xr.Dataset,
    variable: str,
    cells: CellBounds,
    events: EventsConfig,
    enter: npt.NDArray[np.float64],
    exit_: npt.NDArray[np.float64],
    parked: npt.NDArray[np.bool_] | None,
) -> tuple[dict[str, _Column], list[tuple[str, int]]]:
    """Return one variable's columns, and the sweep points blank at every record.

    Each blank point comes with the most readings any record held below its
    level there, which is what the count rule was short of.
    """
    reading = state[variable]
    x = np.asarray(reading.values, dtype=np.float64)
    enhancement = np.asarray(
        state[enhancement_name(variable)].transpose(TIME_COORD, WINDOW_DIM, QUANTILE_DIM).values,
        dtype=np.float64,
    )
    records = find_records(
        cells,
        np.isfinite(x),
        gap_ns=_ns(events.record_gap),
        max_length_ns=_ns(events.max_record_length),
        parked=parked,
    )
    declared = reading.attrs.get("quantization")
    air = clean_air(
        enhancement,
        x,
        records,
        estimator=events.clean_level_estimator,
        min_clean_readings=events.min_clean_readings,
        quantization=None if declared is None else float(declared),
    )

    # The record's description, at each of its readings; blank off record.
    on_record = records.index >= 0
    at = np.where(on_record, records.index, 0)
    level = np.where(on_record[:, None, None], air.level[at], np.nan)
    spread = np.where(on_record[:, None, None], air.spread[at], np.nan)
    n_clean = np.where(on_record[:, None, None], air.n_below[at], 0).astype(np.int32)
    z = (enhancement - level) / spread

    windows, quantiles = enhancement.shape[1:]
    sweep = (windows, quantiles, enter.size, exit_.size)
    event = np.full((x.size, *sweep), -1, dtype=np.int32)
    n_events = np.zeros(sweep, dtype=np.int64)
    chance = np.zeros(sweep, dtype=np.float64)
    gap_ns = _ns(events.max_internal_gap)
    hole_ns = _ns(events.max_bridged_dropout)
    for w, q in np.ndindex(windows, quantiles):
        scored = int(np.count_nonzero(np.isfinite(z[:, w, q])))
        for e, x_ in np.ndindex(enter.size, exit_.size):
            found = find_events(
                z[:, w, q],
                cells,
                records,
                enter=float(enter[e]),
                exit_=float(exit_[x_]),
                max_internal_gap_ns=gap_ns,
                max_bridged_dropout_ns=hole_ns,
            )
            event[:, w, q, e, x_] = found.membership
            n_events[w, q, e, x_] = found.n
            chance[w, q, e, x_] = scored * expected_chance_rate(float(enter[e]), float(exit_[x_]))

    units = str(reading.attrs.get("units", ""))
    swept = (TIME_COORD, WINDOW_DIM, QUANTILE_DIM)
    points = (WINDOW_DIM, QUANTILE_DIM, ENTER_DIM, EXIT_DIM)
    columns: dict[str, _Column] = {
        f"record_{variable}": (
            (TIME_COORD,),
            records.index.astype(np.int32),
            {"long_name": f"record of each {variable} reading; -1 for none"},
        ),
        f"clean_level_{variable}": (
            swept,
            level,
            {"units": units, "long_name": f"clean level of {variable}'s enhancement"},
        ),
        f"clean_spread_{variable}": (
            swept,
            spread,
            {
                "units": units,
                "long_name": (
                    f"clean spread of {variable}'s enhancement: 1.4826 x the median distance "
                    "below the clean level; not a measurement uncertainty"
                ),
            },
        ),
        f"n_clean_readings_{variable}": (
            swept,
            n_clean,
            {"long_name": "readings of the record below its clean level"},
        ),
        f"z_{variable}": (
            swept,
            z,
            {
                "units": "1",
                "long_name": f"(enhancement - clean level) / clean spread of {variable}",
            },
        ),
        f"event_{variable}": (
            (*swept, ENTER_DIM, EXIT_DIM),
            event,
            {
                "long_name": (
                    f"event of each {variable} reading, numbered per sweep point; -1 for none"
                )
            },
        ),
        f"n_events_{variable}": (points, n_events, {"long_name": f"events of {variable} found"}),
        f"chance_events_{variable}": (
            points,
            chance,
            {
                "long_name": f"events chance alone would make on as many readings of {variable}",
                CHANCE_ASSUMPTION_ATTR: _CHANCE_ASSUMPTION,
            },
        ),
    }
    sigma = sigma_rand_name(variable)
    if sigma in state.data_vars:
        columns[sigma] = ((TIME_COORD,), state[sigma].values, dict(state[sigma].attrs))
    blank = [
        (
            f"{state[WINDOW_DIM].values[w]:g} s, q {state[QUANTILE_DIM].values[q]:g}",
            int(air.n_below[:, w, q].max(initial=0)),
        )
        for w, q in np.ndindex(windows, quantiles)
        if air.blank[:, w, q].all()
    ]
    return columns, blank


def _parked(
    state: xr.Dataset, instrument: str, events: EventsConfig, stream: xr.Dataset | None
) -> npt.NDArray[np.bool_] | None:
    """Return True where the platform is parked, per reading; None when not asked.

    Parked means the speed is not above ``moving_above``.

    Read from the instrument's own stream at the baseline state's readings,
    which the stream must all hold (a baseline state built on a stream's
    finite readings of one variable holds a subset). A missing value takes
    the state before it, and the first ones the state after.
    """
    spec = events.platform_state.get(instrument)
    if spec is None:
        return None
    if stream is None:
        raise TsaraEventError(
            f"events.platform_state reads '{instrument}.{spec.variable}', so the events of "
            f"'{instrument}' need its stream: pass stream= (or streams= to event_states)."
        )
    if spec.variable not in stream.data_vars:
        raise TsaraEventError(
            f"events.platform_state reads '{spec.variable}', which the stream of "
            f"'{instrument}' does not hold."
        )
    times = state[TIME_COORD].values
    if not np.isin(times, stream[TIME_COORD].values).all():
        raise TsaraEventError(
            f"The stream handed over for '{instrument}' does not hold every reading of its "
            "baseline state; pass the stream the baseline state was built on."
        )
    value = np.asarray(stream[spec.variable].sel({TIME_COORD: times}).values, dtype=np.float64)
    moving = pd.Series(np.where(np.isfinite(value), value > spec.moving_above, np.nan))
    filled = moving.ffill().bfill()
    if filled.isna().all():
        raise TsaraEventError(
            f"events.platform_state reads '{instrument}.{spec.variable}', which has no finite "
            "value at these readings."
        )
    return np.asarray(filled.to_numpy() == 0, dtype=np.bool_)


def _ns(spec: str) -> int:
    """Return a configured duration, already validated positive, in nanoseconds."""
    return int(pd.Timedelta(spec).value)


def _warn_blank_everywhere(
    instrument: str, blank: Mapping[str, list[tuple[str, int]]], min_clean_readings: int
) -> None:
    """Warn once per stream about sweep points at which every record is blank.

    Variables blank at the same points are named together, at most
    :data:`_MAX_NAMED` of them, as the baseline stage's warning does, and the
    count quoted per point is the largest over the variables named.
    """
    patterns: dict[tuple[str, ...], list[str]] = {}
    held: dict[str, int] = {}
    for variable, where in blank.items():
        patterns.setdefault(tuple(point for point, _ in where), []).append(variable)
        for point, count in where:
            held[point] = max(held.get(point, 0), count)
    groups = []
    for pattern, variables in list(patterns.items())[:_MAX_NAMED]:
        named = ", ".join(variables[:_MAX_NAMED])
        if len(variables) > _MAX_NAMED:
            named += f" and {len(variables) - _MAX_NAMED} more"
        points = "; ".join(f"{point} (at most {held[point]})" for point in pattern)
        groups.append(f"{named}: {points}")
    if len(patterns) > _MAX_NAMED:
        groups.append(f"and {len(patterns) - _MAX_NAMED} more pattern(s)")
    logger.warning(
        "Event state of '%s': sweep points at which every record is blank, so no event is "
        "found -- %s. A record needs %d readings below its clean level (METHODS 6.8); a "
        "sparse variable can take its events from a dense one named in events.triggers.",
        instrument,
        " | ".join(groups),
        min_clean_readings,
    )


# ---------------------------------------------------------------------------
# A variable that takes its events from a trigger
# ---------------------------------------------------------------------------


def _trigger_events(
    trigger: str,
    instrument: str,
    cells: CellBounds,
    found: Mapping[str, _Column],
    triggers: Mapping[str, xr.Dataset] | None,
    state: xr.Dataset,
    events: EventsConfig,
) -> tuple[npt.NDArray[np.integer], CellBounds]:
    """Return a trigger's event membership, shaped (time, *sweep), and its cells.

    From this call's own results when the trigger is a variable of this
    instrument found here; otherwise from the event state handed over for
    its instrument, which must have been found on this same sweep.
    """
    trigger_instrument, _, trigger_variable = trigger.partition(".")
    name = f"event_{trigger_variable}"
    if trigger_instrument == instrument and name in found:
        return found[name][1].astype(np.int64), cells
    source = None if triggers is None else triggers.get(trigger_instrument)
    if source is None or name not in source:
        raise TsaraEventError(
            f"'{instrument}' takes events from '{trigger}', whose event state was not "
            "handed over; find its events first and pass its state in triggers=."
        )
    for dim, wanted in (
        (WINDOW_DIM, state[WINDOW_DIM].values),
        (QUANTILE_DIM, state[QUANTILE_DIM].values),
        (ENTER_DIM, np.asarray(events.enter_multiple, dtype=np.float64)),
        (EXIT_DIM, np.asarray(events.exit_multiple, dtype=np.float64)),
    ):
        if not np.array_equal(source[dim].values, wanted):
            raise TsaraEventError(
                f"The event state of '{trigger_instrument}' was found on another sweep "
                f"({dim} {source[dim].values.tolist()}, here {list(wanted)}); a trigger's "
                "events can only be taken at the same sweep points."
            )
    membership = source[name].transpose(TIME_COORD, WINDOW_DIM, QUANTILE_DIM, ENTER_DIM, EXIT_DIM)
    return membership.values.astype(np.int64), stream_cells(source, trigger_instrument)


def _adopted_events(
    state: xr.Dataset,
    variable: str,
    cells: CellBounds,
    trigger: str,
    membership: npt.NDArray[np.integer],
    source_cells: CellBounds,
    instrument: str,
) -> dict[str, _Column]:
    """Return a triggered variable's columns: the trigger's events, on its own readings."""
    finite = np.isfinite(np.asarray(state[variable].values, dtype=np.float64))
    midpoints = cells.midpoint_ns
    sweep = membership.shape[1:]
    event = np.full((midpoints.size, *sweep), -1, dtype=np.int32)
    n_events = np.zeros(sweep, dtype=np.int64)
    for point in np.ndindex(*sweep):
        starts, stops, numbers = _intervals(membership[(slice(None), *point)], source_cells)
        taken = _containing(midpoints, starts, stops, numbers)
        taken[~finite] = -1
        event[(slice(None), *point)] = taken
        n_events[point] = np.unique(taken[taken >= 0]).size
    attrs: dict[str, object] = {
        "long_name": (
            f"event of each {variable} reading, taken from {trigger} and numbered as there; "
            "-1 for none"
        ),
        TRIGGER_ATTR: trigger,
    }
    points = (WINDOW_DIM, QUANTILE_DIM, ENTER_DIM, EXIT_DIM)
    logger.info("'%s.%s' takes its events from '%s'.", instrument, variable, trigger)
    return {
        f"event_{variable}": ((TIME_COORD, *points), event, attrs),
        f"n_events_{variable}": (
            points,
            n_events,
            {
                "long_name": f"events of {trigger} holding a {variable} reading",
                TRIGGER_ATTR: trigger,
            },
        ),
    }


def _intervals(
    membership: npt.NDArray[np.integer], cells: CellBounds
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return each event's interval, first cell start to last cell stop, and its number."""
    rows = np.flatnonzero(membership >= 0)
    numbers, first = np.unique(membership[rows], return_index=True)
    last = rows.size - 1 - np.unique(membership[rows][::-1], return_index=True)[1]
    return cells.start_ns[rows[first]], cells.stop_ns[rows[last]], numbers.astype(np.int64)


def _containing(
    midpoints: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
    stops: npt.NDArray[np.int64],
    numbers: npt.NDArray[np.int64],
) -> npt.NDArray[np.int32]:
    """Return, per midpoint, the number of the interval holding it, -1 for none.

    Events at one sweep point are in time order, so the candidate is the last
    interval starting at or before the midpoint. Two intervals overlap only
    where cells at least as wide as the gap between two events touch both,
    and there a midpoint goes to the later one.
    """
    candidate = np.searchsorted(starts, midpoints, side="right") - 1
    safe = np.clip(candidate, 0, None)
    inside = (
        (candidate >= 0) & (midpoints < stops[safe])
        if starts.size
        else np.zeros(midpoints.size, dtype=bool)
    )
    out = np.full(midpoints.size, -1, dtype=np.int32)
    out[inside] = numbers[safe[inside]]
    return out
