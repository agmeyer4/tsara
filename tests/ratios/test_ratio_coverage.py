"""The ratio stage's coverage on generated air, re-run through the stage (opt-in, slow).

METHODS §5.4 and §5.5 quote what the stage's intervals cover, measured while
7c2 was built: the widened σ of single-source members, what the one-source
gate turns away and lets through, and a candidate's combined interval. This
module is the Monte Carlo §5.5 names as the decisive check: it re-runs every
one of those numbers through the stage itself (:func:`~tsara.pipeline.ratio_fits`
and :func:`~tsara.ratios.candidate_ratios`, on the whole chain from the
baseline state) under the rule stated there, and requires each to reproduce
at the precision METHODS prints it. A miss is a finding about the stage, not a
tolerance to widen.

The rule (§5.3's records). One 1 s analyzer of methane and ethane, six hours,
seeds 1-3, in two airs: **wandering**, each background a random walk
wandering 30 ppb (methane) and 0.6 ppb (ethane) a day, and **flat**, both
constant. Backgrounds 1950 and 2 ppb. Two sources, A at an ethane/methane
ratio of 0.05 and B at 0, twenty plumes an hour each, Gaussian in time with
σ_t uniform in 5-60 s, methane peaks lognormal with median 15 ppb and σ_log
0.8, centres uniform. White noise 0.7 and 0.05 ppb, declared exactly. The
example sweep: the baseline at 2, 10 and 60 min × q 0.01, 0.05 and 0.10,
events at entry 3, 4 and 5 × exit 1, 1.5 and 2. Ethane against methane, York,
the native clock, the default gates.

A member's truth is the share of source A in the true methane enhancement
over its interval (the readings whose midpoints fall in it); it is **one
source** when that share is at least 0.999 or at most 0.001, and a
**mixture** between 0.05 and 0.95. A candidate is scored when every member
passing the gates is one source, its truth A's ratio when its members'
median share is at least one half.

About five minutes, so opt-in:

    TSARA_SLOW=1 pytest tests/ratios/test_ratio_coverage.py
"""

from __future__ import annotations

import logging
import math
import os
from functools import cache

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig, GatesConfig, RatiosConfig
from tsara.core.timebase import SECOND_NS
from tsara.events import event_candidates, event_catalog, event_states
from tsara.pipeline import ratio_fits
from tsara.ratios import GATE_NAMES, candidate_ratios, gate_passes, gate_settings

pytestmark = pytest.mark.skipif(
    not os.environ.get("TSARA_SLOW"),
    reason="Set TSARA_SLOW=1 to re-run METHODS §5.4-5.5's coverage (about five minutes).",
)

AIRS = {"wandering": (30.0, 0.6), "flat": (0.0, 0.0)}
SEEDS = (1, 2, 3)
SIGMA = {"ch4": 0.7, "c2h6": 0.05}
RATIO_A = 0.05
Z95 = 1.959963984540054


def _record(
    seed: int, walk_ch4: float, walk_c2h6: float
) -> tuple[xr.Dataset, np.ndarray, np.ndarray]:
    """One record of the rule, and source A's and B's true methane enhancements."""
    rng = np.random.default_rng(seed)
    n = 6 * 3600
    t = np.arange(n, dtype=float)
    true_a, true_b = np.zeros(n), np.zeros(n)
    for true in (true_a, true_b):
        for _ in range(20 * 6):
            centre, width = rng.uniform(0, n), rng.uniform(5, 60)
            height = rng.lognormal(math.log(15), 0.8)
            lo, hi = int(max(0, centre - 6 * width)), int(min(n, centre + 6 * width))
            true[lo:hi] += height * np.exp(-0.5 * ((t[lo:hi] - centre) / width) ** 2)

    def walk(per_day: float) -> np.ndarray:
        return np.cumsum(rng.normal(0, per_day / math.sqrt(86400), n))

    background_ch4 = 1950 + walk(walk_ch4)
    background_c2h6 = 2.0 + walk(walk_c2h6)
    ch4 = background_ch4 + true_a + true_b + rng.normal(0, SIGMA["ch4"], n)
    c2h6 = background_c2h6 + RATIO_A * true_a + rng.normal(0, SIGMA["c2h6"], n)
    start = np.arange(n, dtype=np.int64) * SECOND_NS
    declared = {
        "uncertainty_provenance": "declared",
        "uncertainty_provenance_random": "declared",
        "uncertainty_provenance_systematic": "zero",
    }
    data: dict[str, tuple[str, np.ndarray, dict[str, object]]] = {}
    for name, values in (("ch4", ch4), ("c2h6", c2h6)):
        data[name] = (
            "time",
            values,
            {"units": "ppb", "role": "gas", "cell_methods": "time: mean", **declared},
        )
        data["sigma_rand_" + name] = (
            "time",
            np.full(n, SIGMA[name]),
            {
                "units": "ppb",
                "uncertainty_component": "random",
                "uncertainty_provenance": "declared",
            },
        )
    stream = xr.Dataset(
        data,
        coords={
            "time": (start + SECOND_NS // 2).astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([start, start + SECOND_NS], 1).astype("datetime64[ns]"),
            ),
        },
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    return stream, true_a, true_b


@cache
def _tables(air: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stage's fits, each with its share of source A, and its candidates' ratios."""
    stream, true_a, true_b = _record(seed, *AIRS[air])
    logging.disable(logging.WARNING)
    try:
        states = {
            "van": baseline_state(
                stream,
                instrument="van",
                baseline=BaselineConfig(
                    windows=("2min", "10min", "60min"), quantiles=(0.01, 0.05, 0.10)
                ),
            )
        }
        events = event_states(
            states, EventsConfig(enter_multiple=(3.0, 4.0, 5.0), exit_multiple=(1.0, 1.5, 2.0))
        )
        catalog = event_catalog(events, states)
        ratios = RatiosConfig.model_validate(
            {"sets": [{"reference": "van.ch4", "numerators": ["van.c2h6"]}], "methods": ["york"]}
        )
        fits = ratio_fits(states, events, catalog, ratios)
        table = candidate_ratios(fits, event_candidates(catalog, events), ratios.gates)
    finally:
        logging.disable(logging.NOTSET)
    midpoints = stream.time.values.astype(np.int64)
    sum_a = np.concatenate([[0.0], np.cumsum(true_a)])
    sum_b = np.concatenate([[0.0], np.cumsum(true_b)])
    lo = np.searchsorted(midpoints, fits.start_time.to_numpy().astype(np.int64))
    hi = np.searchsorted(midpoints, fits.end_time.to_numpy().astype(np.int64))
    a, b = sum_a[hi] - sum_a[lo], sum_b[hi] - sum_b[lo]
    with np.errstate(invalid="ignore"):
        share = np.where(a + b > 0, a / (a + b), np.nan)
    return fits.assign(share=share), table


def _members(air: str) -> pd.DataFrame:
    """Every fit of an air's three records, with which gates it passes at the default setting."""
    frames: list[pd.DataFrame] = []
    (setting,) = gate_settings(GatesConfig())
    for seed in SEEDS:
        fits, _ = _tables(air, seed)
        each = gate_passes(fits, setting)
        frames.append(
            fits.assign(
                seed=seed,
                # Every gate but the one-source gate, so that it can be judged alone.
                others=np.logical_and.reduce(
                    [each[name] for name in GATE_NAMES if name != "one_source"]
                ),
                one_source=each["one_source"],
                single=(fits.share >= 0.999) | (fits.share <= 0.001),
                mixture=(fits.share > 0.05) & (fits.share < 0.95),
            )
        )
    return pd.concat(frames, ignore_index=True)


def _candidates(air: str) -> pd.DataFrame:
    """The candidates of an air's three records whose passing members are all one source."""
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        fits, table = _tables(air, seed)
        (setting,) = gate_settings(GatesConfig())
        each = gate_passes(fits, setting)
        passing = fits[np.logical_and.reduce(list(each.values()))]
        single = (passing.share >= 0.999) | (passing.share <= 0.001)
        all_single = single.groupby(passing.candidate_id).all()
        truth = (
            passing.groupby("candidate_id")
            .share.median()
            .map(lambda s: RATIO_A if s >= 0.5 else 0.0)
        )
        scored = table[table.n_members > 0].set_index("candidate_id")
        scored = scored[all_single.reindex(scored.index, fill_value=False).to_numpy()]
        frames.append(
            scored.assign(
                covered=np.abs(scored.ratio - truth.reindex(scored.index))
                <= Z95 * scored.sigma_combined
            )
        )
    return pd.concat(frames)


@pytest.mark.parametrize(("air", "coverage"), [("flat", 0.962), ("wandering", 0.940)])
def test_single_source_members_widened_sigma_covers_the_true_ratio(
    air: str, coverage: float
) -> None:
    """§5.4 Built and re-measured: members of one source passing every gate but the
    one-source gate (the crowding gate among them), covered at 2σ with the widened σ
    (the baseline allowance in it)."""
    members = _members(air)
    kept = members[(members.status == "OK") & members.others & members.single]
    truth = np.where(kept.share >= 0.5, RATIO_A, 0.0)
    covered = np.abs(kept.slope - truth) <= 2 * kept.slope_sigma_widened
    assert round(float(covered.mean()), 3) == coverage


@pytest.mark.parametrize(
    ("air", "turned_away", "let_through"), [("flat", 0.056, 0.075), ("wandering", 0.065, 0.101)]
)
def test_what_the_one_source_gate_turns_away_and_lets_through(
    air: str, turned_away: float, let_through: float
) -> None:
    """§5.4 Built: the share of single sources the gate turns away, and of mixtures it lets
    through, among members passing every other gate."""
    members = _members(air)
    judged = members[(members.status == "OK") & members.others]
    assert round(1 - float(judged[judged.single].one_source.mean()), 3) == turned_away
    assert round(float(judged[judged.mixture].one_source.mean()), 3) == let_through


@pytest.mark.parametrize(
    ("air", "count", "coverage"), [("flat", 111, 0.964), ("wandering", 119, 0.975)]
)
def test_a_candidates_combined_interval_covers_the_true_ratio(
    air: str, count: int, coverage: float
) -> None:
    """§5.5 Built: candidates of one source, read from the median member, covered by
    the combined interval at 95 %; and at every window at least 0.893."""
    scored = _candidates(air)
    assert len(scored) == count
    assert round(float(scored.covered.mean()), 3) == coverage


def test_every_window_covers_at_least_0893() -> None:
    worst = min(
        float(group.covered.mean())
        for air in AIRS
        for _, group in _candidates(air).groupby("baseline_window")
    )
    assert round(worst, 3) == 0.893
