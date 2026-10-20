"""METHODS §6.8's candidate tables, re-run through the package (opt-in, slow).

The tables in §6.8's "Candidates" were measured while 7b was planned, with a
scratch copy of the rule. This module re-runs every number of them through
the package (:func:`~tsara.events.link_candidates` and
:func:`~tsara.events.event_candidates` on a whole generated chain) under
each table's stated rule, and requires each to reproduce at the precision
METHODS prints it.

The records are notebook 06's, configuration for configuration, so that a
seed draws the same air: its landfill recipe at 6 h and 3 h (seed 5) and
with no plumes (seeds 1 and 2), and twelve hours of weak plumes in flat and
in wandering air (seed 1). Every record is described by the example sweep:
the baseline at 2, 10 and 60 min × q 0.01, 0.05 and 0.10, events at entry
3, 4 and 5 × exit 1, 1.5 and 2, the events configuration's defaults
otherwise.

About two minutes, so opt-in:

    TSARA_SLOW=1 pytest tests/events/test_candidate_tables.py
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.config.synthetic import SyntheticConfig
from tsara.events import event_candidates, event_catalog, event_state
from tsara.synthetic import SyntheticDataset, generate

pytestmark = pytest.mark.skipif(
    not os.environ.get("TSARA_SLOW"),
    reason="Set TSARA_SLOW=1 to re-run METHODS §6.8's candidate tables (about two minutes).",
)

SIGMA = 0.7
WINDOWS = (120.0, 600.0, 3600.0)
BASELINE = {"windows": ["2min", "10min", "60min"], "quantiles": [0.01, 0.05, 0.10]}
EVENTS = {"enter_multiple": [3.0, 4.0, 5.0], "exit_multiple": [1.0, 1.5, 2.0]}


# ---------------------------------------------------------------------------
# Notebook 06's records, configuration for configuration
# ---------------------------------------------------------------------------


def _campaign(
    name: str, seed: int, start: str, duration: str, background: dict[str, float], sources: Any
) -> SyntheticDataset:
    """One stationary 1 s methane analyzer, `analyzer`, with 0.7 ppb of white noise."""
    return generate(
        SyntheticConfig.model_validate(
            {
                "name": name,
                "seed": seed,
                "start": start,
                "duration": duration,
                "platform": {"kind": "stationary", "latitude": 40.77, "longitude": -111.89},
                "atmosphere": {
                    "truth_resolution": "1s",
                    "fields": {
                        "ch4": {
                            "role": "gas",
                            "units": "ppb",
                            "background": {"kind": "parametric", "offset": 1900.0, **background},
                        }
                    },
                    "sources": sources,
                },
                "instruments": {
                    "analyzer": {
                        "native_rate": "1s",
                        "measures": {"ch4": {"uncertainty": {"random": {"absolute": SIGMA}}}},
                    }
                },
            }
        )
    )


def landfill(seed: int, duration: str, *, plumes: bool = True) -> SyntheticDataset:
    """Notebook 06's landfill recipe: a landfill (0.6 an hour, Gaussian sigma 8 min,
    median 60 ppb) and a gas line (10 blips an hour, EMG 6 s / 12 s, median 120 ppb)."""
    sources = {
        "landfill": {
            "rate_per_hour": 0.6,
            "reference_species": "ch4",
            "shape": {"kind": "gaussian", "sigma": "8min"},
            "amplitude": {"kind": "lognormal", "median": 60.0, "sigma_log": 0.3},
        },
        "gas_line": {
            "rate_per_hour": 10.0,
            "reference_species": "ch4",
            "shape": {"kind": "emg", "sigma": "6s", "tau": "12s"},
            "amplitude": {"kind": "lognormal", "median": 120.0, "sigma_log": 0.4},
        },
    }
    return _campaign(
        "landfill_demo", seed, "2026-06-15T10:00:00Z", duration, {}, sources if plumes else {}
    )


def weak(*, wander: bool) -> SyntheticDataset:
    """Twelve hours of weak plumes (6 an hour, Gaussian sigma 20 s, peaks lognormal
    with median 5 ppb and sigma_log 0.8), in flat air or a 30 ppb/day random walk."""
    sources = {
        "weak": {
            "rate_per_hour": 6.0,
            "reference_species": "ch4",
            "shape": {"kind": "gaussian", "sigma": "20s"},
            "amplitude": {"kind": "lognormal", "median": 5.0, "sigma_log": 0.8},
        }
    }
    background = {"random_walk_std": 30.0} if wander else {}
    name = "weak" if wander else "weakflat"
    return _campaign(name, 1, "2026-06-15T00:00:00Z", "12h", background, sources)


RECORDS = {
    "landfill 6 h": lambda: landfill(5, "6h"),
    "landfill 3 h": lambda: landfill(5, "3h"),
    "weak, flat": lambda: weak(wander=False),
    "weak, wander": lambda: weak(wander=True),
    "no plumes, seed 1": lambda: landfill(1, "6h", plumes=False),
    "no plumes, seed 2": lambda: landfill(2, "6h", plumes=False),
}


@dataclass(frozen=True)
class Run:
    """A record described by the whole chain, and its answer key."""

    events: xr.Dataset
    catalog: pd.DataFrame
    candidates: pd.DataFrame
    times: npt.NDArray[np.datetime64]
    truth: npt.NDArray[np.float64]
    plumes: pd.DataFrame
    hours: float


@cache
def run(record: str) -> Run:
    """Generate a record and run the chain over it once per session."""
    campaign = RECORDS[record]()
    stream = campaign.observable("analyzer")
    baseline = baseline_state(
        stream, instrument="analyzer", baseline=BaselineConfig.model_validate(BASELINE)
    )
    events = event_state(
        baseline, instrument="analyzer", events=EventsConfig.model_validate(EVENTS)
    )
    catalog = event_catalog({"analyzer": events}, {"analyzer": baseline})
    truth = campaign.ground_truth.to_frame()
    times = stream["time"].values
    return Run(
        events=events,
        catalog=catalog,
        candidates=event_candidates(catalog, {"analyzer": events}),
        times=times,
        truth=campaign.streams["analyzer"]["truth_enhancement_ch4"].values,
        plumes=truth[(truth["instrument"] == "analyzer") & (truth["species"] == "ch4")],
        hours=float((times[-1] - times[0]) / np.timedelta64(1, "h")),
    )


# ---------------------------------------------------------------------------
# The scoring rules
# ---------------------------------------------------------------------------


def holds(start: Any, end: Any, t: Any) -> Any:
    """An interval holds a time when start <= t < end."""
    return (start <= t) & (t < end)


def scored(r: Run, window: float) -> pd.DataFrame:
    """One window's candidates, each with its kind and the number of quantiles it
    reaches. A candidate is false when no reading inside its seed's interval
    carries a true plume of 1 sigma; a fragment when one does but the seed's
    interval holds no true plume's peak; otherwise a plume."""
    table = r.candidates[r.candidates["baseline_window"] == window].copy()
    seeds = r.catalog.set_index("event_id").loc[table["seed_event_id"]]
    kinds = []
    for start, end in zip(seeds["start_time"], seeds["end_time"], strict=True):
        inside = holds(np.datetime64(start), np.datetime64(end), r.times)
        carries = bool(inside.any() and r.truth[inside].max() >= SIGMA)
        peak = bool(holds(start, end, r.plumes["peak_time"]).any())
        kinds.append("plume" if peak else ("fragment" if carries else "false"))
    table["kind"] = kinds
    quantiles = r.catalog.groupby("candidate_id")["baseline_quantile"].nunique()
    table["n_quantiles"] = quantiles.loc[table["candidate_id"]].to_numpy()
    return table


Gate = Callable[[pd.DataFrame], npt.NDArray[np.bool_]]

GATES: dict[str, Gate] = {
    "none": lambda t: np.ones(len(t), dtype=bool),
    "max z >= 4": lambda t: (t["max_peak_z"] >= 4).to_numpy(),
    "max z >= 5": lambda t: (t["max_peak_z"] >= 5).to_numpy(),
    "min z >= 4": lambda t: (t["min_peak_z"] >= 4).to_numpy(),
    "all 3 quantiles": lambda t: (t["n_quantiles"] == 3).to_numpy(),
}


def found(r: Run, window: float, kept: pd.Series) -> npt.NDArray[np.bool_]:
    """Per true plume: whether an event of a kept candidate at this window holds
    its true peak time."""
    members = r.catalog[
        (r.catalog["baseline_window"] == window) & r.catalog["candidate_id"].isin(kept)
    ]
    return np.array(
        [
            bool(holds(members["start_time"], members["end_time"], peak).any())
            for peak in r.plumes["peak_time"]
        ],
        dtype=bool,
    )


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def test_the_sweep_inside_one_window_on_the_landfill_record() -> None:
    """The three facts, on the 6 h landfill record. A higher entry keeps the same
    intervals, those whose peak z reaches it (54 comparisons: 3 windows x 3
    quantiles x 3 exits x entries 4 and 5); every event at a higher exit lies
    inside one event at the next lower exit (2,851 events, none outside); and
    at entry 3, the share of (event, other quantile at the same exit) pairs in
    which an event there holds the event's peak and the event holds its peak in
    turn is 86, 72 and 89 % at 2, 10 and 60 min."""
    r = run("landfill 6 h")
    columns = ["start_time", "peak_time", "end_time"]
    same = 0
    for _, point in r.catalog.groupby(["baseline_window", "baseline_quantile", "exit_multiple"]):
        low = point[point["enter_multiple"] == 3.0]
        for entry in (4.0, 5.0):
            high = point[point["enter_multiple"] == entry][columns].reset_index(drop=True)
            same += high.equals(low[low["peak_z"] >= entry][columns].reset_index(drop=True))
    assert same == 54

    index = (
        r.events["event_ch4"]
        .transpose(
            "time", "baseline_window", "baseline_quantile", "enter_multiple", "exit_multiple"
        )
        .values
    )
    checked = outside = 0
    for x in range(1, index.shape[4]):
        for w, q, e in np.ndindex(*index.shape[1:4]):
            upper, lower = index[:, w, q, e, x], index[:, w, q, e, x - 1]
            for k in np.unique(upper[upper >= 0]).tolist():
                parents = np.unique(lower[upper == k])
                checked += 1
                outside += not (parents.size == 1 and parents[0] >= 0)
    assert (checked, outside) == (2851, 0)

    shares = []
    entry3 = r.catalog[r.catalog["enter_multiple"] == 3.0]
    for window in WINDOWS:
        pairs = partnered = 0
        for _, at_exit in entry3[entry3["baseline_window"] == window].groupby("exit_multiple"):
            by_q = dict(tuple(at_exit.groupby("baseline_quantile")))
            for q, events in by_q.items():
                for _, e in events.iterrows():
                    for other_q, others in by_q.items():
                        if other_q == q:
                            continue
                        pairs += 1
                        mutual = holds(others["start_time"], others["end_time"], e["peak_time"]) & (
                            holds(e["start_time"], e["end_time"], others["peak_time"])
                        )
                        partnered += bool(mutual.any())
        shares.append(round(100 * partnered / pairs))
    assert shares == [86, 72, 89]


def test_candidates_and_the_plumes_they_hold() -> None:
    """Per window, no gate: the candidates of each plume record, how many true plumes
    an event of some candidate holds, and how many are held by events of more
    than one candidate."""
    expected = {
        "landfill 6 h": [(101, 73, 73, 4), (107, 73, 73, 5), (89, 73, 73, 3)],
        "landfill 3 h": [(121, 35, 35, 1), (69, 35, 35, 2), (36, 35, 35, 2)],
        "weak, wander": [(338, 72, 72, 2), (171, 67, 72, 3), (95, 50, 72, 5)],
    }
    for record, rows in expected.items():
        r = run(record)
        for window, row in zip(WINDOWS, rows, strict=True):
            members = r.catalog[r.catalog["baseline_window"] == window]
            holders = [
                members.loc[
                    holds(members["start_time"], members["end_time"], peak), "candidate_id"
                ].nunique()
                for peak in r.plumes["peak_time"]
            ]
            n = int((r.candidates["baseline_window"] == window).sum())
            got = (n, sum(h > 0 for h in holders), len(holders), sum(h > 1 for h in holders))
            assert got == row, (record, window)


def test_false_candidates_an_hour_under_each_gate() -> None:
    """False candidates per hour of record, at 2, 10 and 60 min, as METHODS prints
    them (two decimals)."""
    expected = {
        "landfill 6 h": {
            "none": [2.67, 8.83, 6.67],
            "max z >= 4": [0.00, 0.50, 0.17],
            "max z >= 5": [0.00, 0.00, 0.00],
            "min z >= 4": [0.00, 0.00, 0.00],
            "all 3 quantiles": [0.00, 0.83, 3.33],
        },
        "landfill 3 h": {
            "none": [16.67, 11.33, 3.00],
            "max z >= 4": [1.00, 0.67, 0.00],
            "max z >= 5": [0.00, 0.00, 0.00],
            "min z >= 4": [0.00, 0.00, 0.00],
            "all 3 quantiles": [0.00, 0.33, 1.67],
        },
        "weak, flat": {
            "none": [3.42, 6.83, 6.00],
            "max z >= 4": [0.17, 0.33, 0.17],
            "max z >= 5": [0.00, 0.00, 0.00],
            "min z >= 4": [0.00, 0.00, 0.00],
            "all 3 quantiles": [1.33, 1.92, 1.92],
        },
        "weak, wander": {
            "none": [21.00, 8.50, 4.42],
            "max z >= 4": [2.67, 0.92, 0.75],
            "max z >= 5": [0.25, 0.08, 0.17],
            "min z >= 4": [0.33, 0.08, 0.50],
            "all 3 quantiles": [3.50, 2.00, 0.17],
        },
        "no plumes, seed 1": {
            "none": [13.83, 3.50, 6.83],
            "max z >= 4": [1.17, 0.00, 0.33],
            "max z >= 5": [0.00, 0.00, 0.00],
            "min z >= 4": [0.00, 0.00, 0.00],
            "all 3 quantiles": [1.50, 0.83, 4.33],
        },
        "no plumes, seed 2": {
            "none": [18.50, 10.17, 8.17],
            "max z >= 4": [0.50, 0.50, 0.17],
            "max z >= 5": [0.00, 0.00, 0.00],
            "min z >= 4": [0.17, 0.17, 0.00],
            "all 3 quantiles": [3.67, 4.00, 5.00],
        },
    }
    for record, gates in expected.items():
        r = run(record)
        tables = [scored(r, window) for window in WINDOWS]
        for gate, rates in gates.items():
            got = [
                round(int((t["kind"][GATES[gate](t)] == "false").sum()) / r.hours, 2)
                for t in tables
            ]
            assert got == rates, (record, gate)


def test_true_plumes_found_under_each_gate_by_their_height() -> None:
    """Of the weak records' true plumes, by their sampled peak in noise sigmas (12
    at 3-5, 26 at 5-10, 22 at 10-30), the percent an event of a gated candidate
    holds, at 2, 10 and 60 min."""
    bands = {"3-5": (3, 5), "5-10": (5, 10), "10-30": (10, 30)}
    expected = {
        ("weak, flat", "3-5"): {
            "none": [100, 100, 100],
            "max z >= 4": [83, 100, 100],
            "max z >= 5": [25, 83, 83],
            "min z >= 4": [58, 83, 83],
        },
        ("weak, flat", "5-10"): {g: [100, 100, 100] for g in GATES if g != "all 3 quantiles"},
        ("weak, flat", "10-30"): {g: [100, 100, 100] for g in GATES if g != "all 3 quantiles"},
        ("weak, wander", "3-5"): {
            "none": [100, 100, 92],
            "max z >= 4": [100, 67, 50],
            "max z >= 5": [75, 17, 42],
            "min z >= 4": [75, 33, 42],
        },
        ("weak, wander", "5-10"): {
            "none": [100, 100, 62],
            "max z >= 4": [100, 96, 38],
            "max z >= 5": [100, 69, 27],
            "min z >= 4": [100, 77, 15],
        },
        ("weak, wander", "10-30"): {
            "none": [100, 100, 73],
            "max z >= 4": [100, 100, 68],
            "max z >= 5": [100, 100, 59],
            "min z >= 4": [100, 100, 59],
        },
    }
    for record in ("weak, flat", "weak, wander"):
        r = run(record)
        height = (r.plumes["sampled_peak_amplitude"] / SIGMA).to_numpy()
        sizes = [int(((height >= lo) & (height < hi)).sum()) for lo, hi in bands.values()]
        assert sizes == [12, 26, 22]
        tables = [scored(r, window) for window in WINDOWS]
        for (name, band), gates in expected.items():
            if name != record:
                continue
            lo, hi = bands[band]
            in_band = (height >= lo) & (height < hi)
            for gate, percents in gates.items():
                got = [
                    round(100 * found(r, w, t["candidate_id"][GATES[gate](t)])[in_band].mean())
                    for w, t in zip(WINDOWS, tables, strict=True)
                ]
                assert got == percents, (record, band, gate)


def test_tall_plumes_and_fragments_under_the_height_gate() -> None:
    """On the landfill records every true plume is 30 sigma or taller. Under
    max z >= 5, an event of a gated candidate holds every one at every window
    but one, at 2 min on the 3 h record (34 of 35), where the landfill's top is
    held only by a low event; and fragments (a seed carrying a true plume of 1
    sigma but holding no true peak) pass the gate at most 1.67 an hour, there."""
    rates = []
    for record, misses in (("landfill 6 h", [0, 0, 0]), ("landfill 3 h", [1, 0, 0])):
        r = run(record)
        assert (r.plumes["sampled_peak_amplitude"] / SIGMA >= 30).all()
        tables = [scored(r, window) for window in WINDOWS]
        got = []
        for w, t in zip(WINDOWS, tables, strict=True):
            gate = GATES["max z >= 5"](t)
            got.append(int((~found(r, w, t["candidate_id"][gate])).sum()))
            rates.append(round(int((t["kind"][gate] == "fragment").sum()) / r.hours, 2))
        assert got == misses, record
    assert max(rates) == 1.67
