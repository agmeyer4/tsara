"""METHODS §6.8's generated-data tables, re-run through the package (opt-in, slow).

The tables in §6.8 were measured at the Phase-6 scoping with throwaway
scripts that used TSARA's generator and baseline state and a detector of
their own. This module re-runs every column that the package now implements
(the clean level and spread of meaning C, the detector, the catalog) under
each table's stated rule, and requires every number to reproduce at the
precision METHODS prints it: the package does what was decided, measured
rather than argued. Columns of rejected alternatives (meanings A and B, the
shortest half and tenth) are not package code and are not re-run here.

Each rule is the table's own: the campaigns below are the scoping's,
configuration for configuration, so that a seed draws the same air; events
are found with nothing bridged (the scoping's detector bridged nothing), and
each generated record is one record (the scoping described 12 h at once).

About twelve minutes, so opt-in:

    TSARA_SLOW=1 pytest tests/events/test_methods_tables.py
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.config.synthetic import SyntheticConfig
from tsara.core.support import stream_cells
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import event_catalog, event_state, find_events, find_records
from tsara.synthetic import SyntheticDataset, generate

pytestmark = pytest.mark.skipif(
    not os.environ.get("TSARA_SLOW"),
    reason="Set TSARA_SLOW=1 to re-run METHODS §6.8's generated tables (about 12 minutes).",
)

SIGMA = 0.7
HOURS = 12
WINDOWS = ("2min", "10min", "60min")
HOUR = 3600 * SECOND


def as_scoped(**more: Any) -> EventsConfig:
    """Events as the scoping found them: nothing bridged, the whole 12 h one record."""
    return EventsConfig.model_validate(
        {"max_internal_gap": "1ns", "max_record_length": "13h", **more}
    )


BACKGROUNDS: dict[str, dict[str, float]] = {
    "flat": {},
    "rw30": {"random_walk_std": 30.0},
    "rw100": {"random_walk_std": 100.0},
    "diurnal+rw30": {"diurnal_amplitude": 25.0, "random_walk_std": 30.0},
}


def printed(value: float, expected: float, decimals: int) -> bool:
    """Whether ``value`` prints as ``expected`` at the precision METHODS uses."""
    return bool(abs(value - expected) <= 0.5 * 10.0**-decimals + 1e-9)


# ---------------------------------------------------------------------------
# The scoping's campaigns, configuration for configuration
# ---------------------------------------------------------------------------


def _analyzer(
    name: str, seed: int, start: str, background: dict[str, float], sources: dict[str, Any]
) -> SyntheticDataset:
    """One stationary 1 s methane analyzer, 0.7 ppb of white noise, 12 h."""
    return generate(
        SyntheticConfig.model_validate(
            {
                "name": name,
                "seed": seed,
                "start": start,
                "duration": f"{HOURS}h",
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
                    **({"sources": sources} if sources else {}),
                },
                "instruments": {
                    "a": {
                        "native_rate": "1s",
                        "measures": {"ch4": {"uncertainty": {"random": {"absolute": SIGMA}}}},
                    }
                },
            }
        )
    )


def _blips(rate: float, median: float) -> dict[str, Any]:
    return {
        "rate_per_hour": rate,
        "reference_species": "ch4",
        "shape": {"kind": "emg", "sigma": "6s", "tau": "12s"},
        "amplitude": {"kind": "lognormal", "median": median, "sigma_log": 0.4},
    }


def _landfill(rate: float) -> dict[str, Any]:
    return {
        "rate_per_hour": rate,
        "reference_species": "ch4",
        "shape": {"kind": "gaussian", "sigma": "8min"},
        "amplitude": {"kind": "lognormal", "median": 60.0, "sigma_log": 0.3},
    }


DENSITIES: dict[str, dict[str, Any]] = {
    "sparse": {"gas": _blips(10, 120)},
    "landfill": {"land": _landfill(0.6), "gas": _blips(10, 120)},
    "dense": {"land": _landfill(1.5), "gas": _blips(40, 40)},
}
WEAK = {
    "weak": {
        "rate_per_hour": 6.0,
        "reference_species": "ch4",
        "shape": {"kind": "gaussian", "sigma": "20s"},
        "amplitude": {"kind": "lognormal", "median": 5.0, "sigma_log": 0.8},
    }
}


@cache
def _rolled(
    kind: str, name: str, seed: int, quantiles: tuple[float, ...]
) -> tuple[SyntheticDataset, xr.Dataset]:
    """A campaign and its baseline state, rolled once for every test that needs it."""
    if kind == "none":
        campaign = _analyzer("fa", seed, "2026-06-16T00:00:00Z", BACKGROUNDS[name], {})
    elif kind == "weak":
        campaign = _analyzer("weak", seed, "2026-06-15T00:00:00Z", BACKGROUNDS[name], WEAK)
    else:
        campaign = _analyzer(name, seed, "2026-06-15T00:00:00Z", {}, DENSITIES[name])
    stream = campaign.observable("a").drop_vars(["sigma_rand_ch4"], errors="ignore")
    config = BaselineConfig.model_validate({"windows": list(WINDOWS), "quantiles": list(quantiles)})
    return campaign, baseline_state(stream, instrument="a", baseline=config, variables=["ch4"])


def _true_noise_events(campaign: SyntheticDataset) -> Any:
    """The detector that knows the truth: z = (reading - true background) / true sigma."""
    full = campaign.streams["a"]
    cells = stream_cells(full, "a")
    z = (full["ch4"].values - full["truth_background_ch4"].values) / SIGMA
    records = find_records(cells, np.isfinite(z), gap_ns=2 * HOUR, max_length_ns=13 * HOUR)
    found = find_events(
        z, cells, records, enter=3.0, exit_=1.0, max_internal_gap_ns=0, max_bridged_dropout_ns=0
    )
    return found, cells


# ---------------------------------------------------------------------------
# No plumes: every event is false
# ---------------------------------------------------------------------------


def test_the_no_plume_table() -> None:
    """Rule (§6.8): the four backgrounds, seeds 1-6, the 3 x 3 sweep, entry 3, exit 1;
    each cell the range over the nine sweep points of the mean over seeds. Also the
    true-noise detector's 5.1 an hour, the half-sample mode's seed-to-seed spread on
    flat air at q = 0.05 (12.3 / 4.6 / 3.8 an hour), and, from the same random walk,
    3.1 false events an hour at 60 min and exit 1 against 16.4 at exit 2."""
    table = {  # background: ((C low, high, decimals), (spread / sigma low, high, decimals))
        "flat": ((5, 11, 0), (0.98, 1.04, 2)),
        "rw30": ((3, 16, 0), (1.1, 2.8, 1)),
        "rw100": ((0.9, 18, 1), (1.8, 8.9, 1)),
        "diurnal+rw30": ((2, 14, 0), (1.1, 3.0, 1)),
    }
    config = as_scoped(exit_multiple=(1.0, 2.0))
    oracle = []
    for name, ((low, high, dp), (r_low, r_high, r_dp)) in table.items():
        rates, ratios, exits = [], [], []
        for seed in range(1, 7):
            campaign, state = _rolled("none", name, seed, (0.01, 0.05, 0.10))
            found = event_state(state, instrument="a", events=config)
            rates.append(found["n_events_ch4"].values[:, :, 0, 0] / HOURS)
            ratios.append(found["clean_spread_ch4"].values[0] / SIGMA)
            exits.append(found["n_events_ch4"].values[2, 1, 0, :] / HOURS)
            if name == "flat":
                oracle.append(_true_noise_events(campaign)[0].n / HOURS)
        mean_rate, mean_ratio = np.mean(rates, axis=0), np.mean(ratios, axis=0)
        low_dp = 1 if low < 1 else dp
        assert printed(mean_rate.min(), low, low_dp) and printed(mean_rate.max(), high, 0), name
        assert printed(mean_ratio.min(), r_low, r_dp) and printed(mean_ratio.max(), r_high, r_dp)
        if name == "flat":
            spread = np.std(rates, axis=0)[:, 1]
            assert [round(float(s), 1) for s in spread] == [12.3, 4.6, 3.8]
            assert printed(float(np.mean(oracle)), 5.1, 1)
        if name == "rw30":
            at_exits = np.mean(exits, axis=0)
            assert printed(at_exits[0], 3.1, 1) and printed(at_exits[1], 16.4, 1)


# ---------------------------------------------------------------------------
# Weak plumes: recall against the answer key
# ---------------------------------------------------------------------------


def _scored(catalog: pd.DataFrame, truth: pd.DataFrame, window_s: float) -> tuple[np.ndarray, int]:
    """Found (per true event) and the count of false events, at one window.

    Found: a detected interval holds the true peak time. False: a detected
    interval overlapping no true event's [start, end]. Both ends inclusive, as
    the scoping scored them.
    """
    rows = catalog[catalog["baseline_window"] == window_s]
    lo, hi = rows["start_time"].to_numpy(), rows["end_time"].to_numpy()
    peaks = truth["peak_time"].to_numpy()
    found = np.array([bool(np.any((lo <= p) & (p <= hi))) for p in peaks])
    t0, t1 = truth["start_time"].to_numpy(), truth["end_time"].to_numpy()
    false = sum(1 for a, b in zip(lo, hi, strict=True) if not np.any((t0 <= b) & (a <= t1)))
    return found, false


def test_the_weak_plume_recall_table() -> None:
    """Rule (§6.8): Gaussian plumes of sigma 20 s, 6 an hour, peaks lognormal median
    5 ppb and sigma_log 0.8; seeds 1-3, 196 events; q = 0.05, entry 3, exit 1;
    recall pooled over seeds by the peak's true size in sigmas; false events an hour
    over each record's span, averaged over seeds."""
    table = {  # background: per size bin, recall at 2 / 10 / 60 min (a METHODS cell); false/h
        "flat": ([(95, 100, 100), (100, 100, 100), (100, 100, 100)], (3.3, 4.9, 5.0)),
        "rw30": ([(93, 84, 26), (100, 100, 48), (100, 100, 98)], (7.5, 9.1, 0.9)),
    }
    bins = [(3, 5), (5, 10), (10, 30)]
    config = as_scoped()
    for name, (recall, false_rate) in table.items():
        found: dict[float, list[np.ndarray]] = {}
        false: dict[float, list[float]] = {}
        sizes, oracle_found = [], []
        for seed in (1, 2, 3):
            campaign, state = _rolled("weak", name, seed, (0.05,))
            catalog = event_catalog(
                {"a": event_state(state, instrument="a", events=config)}, {"a": state}
            )
            truth = campaign.ground_truth.to_frame()
            truth = truth[truth["sampled_peak_amplitude"].notna()]
            sizes.append(truth["sampled_peak_amplitude"].to_numpy() / SIGMA)
            span = np.ptp(state["time"].values.astype("int64")) / HOUR
            for window in (120.0, 600.0, 3600.0):
                hit, wrong = _scored(catalog, truth, window)
                found.setdefault(window, []).append(hit)
                false.setdefault(window, []).append(wrong / span)
            events, cells = _true_noise_events(campaign)
            lo, hi = cells.start_ns[events.first], cells.stop_ns[events.last]
            peaks = truth["peak_time"].to_numpy().astype("datetime64[ns]").astype(np.int64)
            oracle_found.append(np.array([np.any((lo <= p) & (p <= hi)) for p in peaks]))
        size = np.concatenate(sizes)
        assert size.size == 196
        oracle = np.concatenate(oracle_found)
        for b0, b1 in bins:
            assert oracle[(size >= b0) & (size < b1)].all()
        for w, window in enumerate((120.0, 600.0, 3600.0)):
            hit = np.concatenate(found[window])
            got = [100 * hit[(size >= b0) & (size < b1)].mean() for b0, b1 in bins]
            expected = [recall[b][w] for b in range(3)]
            assert [round(g) for g in got] == expected, (name, window, got)
            assert printed(float(np.mean(false[window])), false_rate[w], 1), (name, window)


# ---------------------------------------------------------------------------
# Plume-dense records: the clean level against truth
# ---------------------------------------------------------------------------


def test_the_plume_density_table() -> None:
    """Rule (§6.8): three plume densities, seeds 1-3, q = 0.05; truth is the median
    and 1.4826 x MAD of the enhancement over readings whose true enhancement is
    under 0.1 sigma; the half-sample mode's level error in true spreads at 2 / 10 /
    60 min, mean over seeds. On the dense record the half-sample mode puts 56-77 %
    of readings in events where the true-noise detector puts 82 %."""
    table = {
        "sparse": (0.02, 0.14, 0.16),
        "landfill": (0.21, 0.13, 0.14),
        "dense": (0.29, 0.33, 0.20),
    }
    config = as_scoped()
    for name, expected in table.items():
        errors, shares, oracle = [], [], []
        for seed in (1, 2, 3):
            campaign, state = _rolled("density", name, seed, (0.05,))
            found = event_state(state, instrument="a", events=config)
            enhancement = campaign.streams["a"]["truth_enhancement_ch4"].values
            clean = enhancement < 0.1 * SIGMA
            per_window = []
            for w in range(3):
                delta = state["enhancement_ch4"].values[:, w, 0]
                usable = np.isfinite(delta) & clean
                m0 = float(np.median(delta[usable]))
                s0 = 1.4826 * float(np.median(np.abs(delta[usable] - m0)))
                level = float(found["clean_level_ch4"].values[0, w, 0])
                per_window.append((level - m0) / s0)
            errors.append(per_window)
            shares.append((found["event_ch4"].values[:, :, 0, 0, 0] >= 0).mean(axis=0))
            events, _ = _true_noise_events(campaign)
            oracle.append((events.membership >= 0).mean())
        mean = np.mean(errors, axis=0)
        assert all(printed(float(m), e, 2) for m, e in zip(mean, expected, strict=True)), (
            name,
            mean,
        )
        if name == "dense":
            share = 100 * np.mean(shares, axis=0)
            assert printed(float(share.min()), 56, 0) and printed(float(share.max()), 77, 0)
            assert printed(100 * float(np.mean(oracle)), 82, 0)
