"""Tests for candidates: one window's events that share a peak (tsara.events.candidates)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import (
    CANDIDATE_COLUMNS,
    TsaraEventError,
    event_candidates,
    event_catalog,
    event_state,
    link_candidates,
)
from tsara.synthetic.plumes import GROUND_TRUTH_COLUMNS

BASE = pd.Timestamp("2026-01-01")

# A hand-made event: (event_id, window, quantile, entry, exit, start, peak, end, z), times in s.
Row = tuple[str, float, float, float, float, float, float, float, float]


def catalog_of(rows: Sequence[Row], **extra: Any) -> pd.DataFrame:
    """A hand-made catalog holding every column the candidate table reads."""
    return pd.DataFrame(
        {
            "event_id": [r[0] for r in rows],
            "parent_event_id": None,
            "instrument": "van",
            "species": "ch4",
            "field": "ch4",
            "baseline_window": [r[1] for r in rows],
            "baseline_quantile": [r[2] for r in rows],
            "enter_multiple": [r[3] for r in rows],
            "exit_multiple": [r[4] for r in rows],
            "start_time": [BASE + pd.Timedelta(seconds=r[5]) for r in rows],
            "peak_time": [BASE + pd.Timedelta(seconds=r[6]) for r in rows],
            "end_time": [BASE + pd.Timedelta(seconds=r[7]) for r in rows],
            "peak_z": [r[8] for r in rows],
            "peak_enhancement": [10.0 * r[8] for r in rows],
            "trigger_event_id": None,
            "latitude": 40.77,
            "longitude": -111.89,
            **extra,
        }
    )


def states_of(
    quantiles: Sequence[float] = (0.05,),
    entries: Sequence[float] = (3.0,),
    exits: Sequence[float] = (1.0, 2.0),
    instrument: str = "van",
) -> dict[str, xr.Dataset]:
    """A stand-in event state: only its sweep matters to the candidate table."""
    state = xr.Dataset(
        coords={
            "baseline_quantile": list(quantiles),
            "enter_multiple": list(entries),
            "exit_multiple": list(exits),
        },
        attrs={"tsara_stage": "events"},
    )
    return {instrument: state}


def counts(table: pd.DataFrame, seed: str) -> list[int]:
    """The (found, absorbed, absent) counts of the candidate a seed made."""
    row = table[table["seed_event_id"] == seed].iloc[0]
    return [int(row["n_found"]), int(row["n_absorbed"]), int(row["n_absent"])]


def members(linked: pd.DataFrame) -> dict[str, set[str]]:
    """Each candidate's events, by candidate id."""
    return {str(k): set(map(str, g["event_id"])) for k, g in linked.groupby("candidate_id")}


# ---------------------------------------------------------------------------
# The rule, case by case
# ---------------------------------------------------------------------------


def test_two_blips_merged_at_a_low_exit_are_two_candidates_one_absorbed_there() -> None:
    """At exit 1 one event holds both blips and peaks on the taller; at exit 2 they
    are apart. The taller blip's candidate holds the merged event; the shorter is a
    candidate of its own, absorbed at exit 1, where its peak lies in the taller's
    event."""
    catalog = catalog_of(
        [
            ("merged", 120, 0.05, 3, 1, 0, 30, 100, 10.0),
            ("tall", 120, 0.05, 3, 2, 20, 30, 40, 10.0),
            ("short", 120, 0.05, 3, 2, 60, 70, 80, 6.0),
        ]
    )
    assert sorted(members(link_candidates(catalog)).values(), key=len) == [
        {"short"},
        {"merged", "tall"},
    ]
    table = event_candidates(catalog, states_of())
    assert counts(table, "merged") == [2, 0, 0]
    assert counts(table, "short") == [1, 1, 0]


def test_a_peak_that_moves_with_the_quantile_stays_one_candidate() -> None:
    catalog = catalog_of(
        [
            ("q1", 120, 0.01, 3, 1, 0, 25, 50, 8.0),
            ("q5", 120, 0.05, 3, 1, 5, 27, 55, 7.5),
        ]
    )
    assert list(members(link_candidates(catalog)).values()) == [{"q1", "q5"}]


def test_a_point_where_nothing_holds_the_peak_is_absent() -> None:
    catalog = catalog_of([("only", 120, 0.01, 3, 1, 0, 25, 50, 8.0)])
    table = event_candidates(catalog, states_of(quantiles=(0.01, 0.05), exits=(1.0,)))
    assert table[["n_found", "n_absorbed", "n_absent"]].iloc[0].tolist() == [1, 0, 1]


def test_an_event_peaking_outside_the_seed_does_not_join_it() -> None:
    """At q 0.05 a long event holds the blip's peak, but its own peak lies far
    outside the blip: the blip is absorbed there, and the long event, shorter in z,
    seeds its own candidate, absent at q 0.01 where nothing holds its peak."""
    catalog = catalog_of(
        [
            ("blip", 120, 0.01, 3, 1, 10, 15, 20, 9.0),
            ("long", 120, 0.05, 3, 1, 0, 80, 100, 5.0),
        ]
    )
    table = event_candidates(catalog, states_of(quantiles=(0.01, 0.05), exits=(1.0,)))
    assert counts(table, "blip") == [1, 1, 0]
    assert counts(table, "long") == [1, 0, 1]


def test_a_taller_candidate_keeps_an_event_a_shorter_one_also_reaches() -> None:
    """The long event at q 0.05 peaks inside both blips' reach only for the taller
    one, which takes it first; the shorter blip finds it taken and is absorbed."""
    catalog = catalog_of(
        [
            ("tall", 120, 0.01, 3, 1, 40, 50, 60, 12.0),
            ("short", 120, 0.01, 3, 1, 10, 15, 20, 6.0),
            ("long", 120, 0.05, 3, 1, 0, 50, 100, 11.0),
        ]
    )
    linked = members(link_candidates(catalog))
    assert sorted(linked.values(), key=len) == [{"short"}, {"tall", "long"}]
    table = event_candidates(catalog, states_of(quantiles=(0.01, 0.05), exits=(1.0,)))
    assert counts(table, "short")[:2] == [1, 1]


def test_an_interval_holds_a_peak_from_its_start_up_to_its_end() -> None:
    """A peak at an event's start is held; at its end it is not."""
    catalog = catalog_of(
        [
            ("seed", 120, 0.01, 3, 1, 100, 100, 110, 9.0),
            ("ends_on_it", 120, 0.05, 3, 1, 90, 95, 100, 4.0),
            ("other", 120, 0.10, 3, 1, 100, 105, 120, 5.0),
        ]
    )
    linked = members(link_candidates(catalog))
    assert {"seed", "other"} in linked.values()
    assert {"ends_on_it"} in linked.values()


def test_where_two_events_of_one_point_touch_the_peak_the_later_starting_one_holds_it() -> None:
    """Cells overlapping by a little can make two events of one sweep point touch
    one time: the seed's peak at 10 s lies in both events at q 0.05, and the
    later-starting one joins it, as a triggered variable's reading goes to the
    later event."""
    catalog = catalog_of(
        [
            ("seed", 120, 0.01, 3, 1, 0, 10, 30, 9.0),
            ("earlier", 120, 0.05, 3, 1, 0, 5, 10.5, 4.0),
            ("later", 120, 0.05, 3, 1, 9.5, 12, 30, 5.0),
        ]
    )
    assert sorted(members(link_candidates(catalog)).values(), key=len) == [
        {"earlier"},
        {"seed", "later"},
    ]


def test_entry_copies_join_the_same_candidate() -> None:
    """A higher entry keeps the same interval when the peak reaches it: one
    candidate holds both copies."""
    catalog = catalog_of(
        [
            ("e3", 120, 0.05, 3, 1, 0, 25, 50, 4.5),
            ("e4", 120, 0.05, 4, 1, 0, 25, 50, 4.5),
        ]
    )
    assert list(members(link_candidates(catalog)).values()) == [{"e3", "e4"}]


def test_the_seed_is_the_tallest_and_a_tie_goes_to_the_earlier_peak_then_sweep_point() -> None:
    """Three events tie in z. The earliest peak seeds; of two sharing a peak, the
    earlier sweep point (quantile before exit) seeds."""
    catalog = catalog_of(
        [
            ("late", 120, 0.01, 3, 1, 50, 60, 70, 7.0),
            ("x2", 120, 0.05, 3, 2, 0, 10, 20, 7.0),
            ("x1", 120, 0.05, 3, 1, 0, 10, 30, 7.0),
        ]
    )
    table = event_candidates(catalog, states_of(quantiles=(0.01, 0.05)))
    assert table["seed_event_id"].tolist() == ["x1", "late"]
    assert table["max_peak_z"].tolist() == [7.0, 7.0]


def test_the_highest_and_lowest_peak_z_and_the_span_of_a_candidate() -> None:
    catalog = catalog_of(
        [
            ("a", 120, 0.01, 3, 1, 0, 25, 50, 8.0),
            ("b", 120, 0.05, 3, 1, 10, 26, 70, 6.5),
        ]
    )
    row = event_candidates(catalog, states_of(quantiles=(0.01, 0.05), exits=(1.0,))).iloc[0]
    assert (row["max_peak_z"], row["min_peak_z"]) == (8.0, 6.5)
    assert row["start_time"] == BASE
    assert row["end_time"] == BASE + pd.Timedelta(seconds=70)
    assert row["duration_s"] == 70.0
    assert row["peak_time"] == BASE + pd.Timedelta(seconds=25)
    assert row["peak_enhancement"] == 80.0


def test_windows_do_not_mix_and_the_parent_is_the_candidate_of_the_seeds_parent() -> None:
    """The same peak at two windows makes two candidates; the short window's seed
    has the long window's event as its parent, so its candidate has that one's."""
    catalog = catalog_of(
        [
            ("short", 120, 0.05, 3, 1, 20, 30, 40, 9.0),
            ("long", 600, 0.05, 3, 1, 0, 30, 100, 9.0),
        ]
    )
    catalog["parent_event_id"] = ["long", None]
    linked = link_candidates(catalog).set_index("event_id")
    assert linked.loc["short", "candidate_id"] == "van.ch4/w120s/c0"
    assert linked.loc["long", "candidate_id"] == "van.ch4/w600s/c0"
    table = event_candidates(catalog, states_of(exits=(1.0,))).set_index("candidate_id")
    assert table.loc["van.ch4/w120s/c0", "parent_candidate_id"] == "van.ch4/w600s/c0"
    assert pd.isna(table.loc["van.ch4/w600s/c0", "parent_candidate_id"])


def test_candidates_are_numbered_in_time_order_and_spelled_from_their_window() -> None:
    catalog = catalog_of(
        [
            ("later_taller", 600, 0.05, 3, 1, 50, 60, 70, 9.0),
            ("earlier", 600, 0.05, 3, 1, 0, 10, 20, 5.0),
        ]
    )
    linked = link_candidates(catalog).set_index("event_id")
    assert linked.loc["earlier", "candidate_id"] == "van.ch4/w600s/c0"
    assert linked.loc["later_taller", "candidate_id"] == "van.ch4/w600s/c1"
    table = event_candidates(catalog, states_of(exits=(1.0,)))
    assert table["candidate_number"].tolist() == [0, 1]


def test_the_result_does_not_depend_on_the_order_or_index_of_the_rows() -> None:
    rng = np.random.default_rng(3)
    catalog = random_catalog(rng)
    shuffled = catalog.sample(frac=1.0, random_state=1)
    shuffled.index = pd.Index(rng.permutation(len(shuffled)) * 7)
    linked = link_candidates(shuffled)
    assert linked.index.tolist() == shuffled.index.tolist()
    expect = link_candidates(catalog).set_index("event_id")["candidate_id"]
    assert linked.set_index("event_id")["candidate_id"].sort_index().equals(expect.sort_index())


def test_a_triggered_variable_takes_its_triggers_candidates() -> None:
    catalog = catalog_of(
        [
            ("trigger", 120, 0.05, 3, 1, 0, 25, 50, 8.0),
            ("canister", 120, 0.05, 3, 1, 0, 25, 50, np.nan),
        ]
    )
    catalog.loc[1, ["instrument", "species", "trigger_event_id"]] = ["can", "benzene", "trigger"]
    linked = link_candidates(catalog).set_index("event_id")
    assert linked.loc["canister", "candidate_id"] == linked.loc["trigger", "candidate_id"]
    table = event_candidates(catalog, states_of(exits=(1.0,)))
    assert table["species"].tolist() == ["ch4"]


def test_a_triggered_row_whose_trigger_is_missing_is_refused() -> None:
    catalog = catalog_of([("canister", 120, 0.05, 3, 1, 0, 25, 50, np.nan)])
    catalog["trigger_event_id"] = "gone"
    with pytest.raises(TsaraEventError, match="'gone', which is not in the catalog"):
        link_candidates(catalog)


def test_what_is_not_a_catalog_or_its_state_is_refused() -> None:
    catalog = catalog_of([("a", 120, 0.05, 3, 1, 0, 25, 50, 8.0)])
    with pytest.raises(TsaraEventError, match=r"link_candidates .* missing: \['peak_z'\]"):
        link_candidates(catalog.drop(columns="peak_z"))
    with pytest.raises(TsaraEventError, match=r"event_candidates .* missing: \['field'\]"):
        event_candidates(catalog.drop(columns="field"), states_of())
    with pytest.raises(TsaraEventError, match="needs the event state of 'van'"):
        event_candidates(catalog, states_of(instrument="other"))
    wrong = states_of()
    wrong["van"].attrs["tsara_stage"] = "baseline"
    with pytest.raises(TsaraEventError, match="needs the event state of 'van'"):
        event_candidates(catalog, wrong)


def test_an_empty_catalog_makes_an_empty_table_with_every_column() -> None:
    catalog = catalog_of([])
    assert link_candidates(catalog)["candidate_id"].tolist() == []
    table = event_candidates(catalog, states_of())
    assert list(table.columns) == list(CANDIDATE_COLUMNS)
    assert len(table) == 0
    assert str(table["start_time"].dtype) == "datetime64[ns]"
    assert str(table["n_found"].dtype) == "int64"


def test_the_columns_shared_with_the_answer_key_are_spelled_as_it_spells_them() -> None:
    shared = {"instrument", "species", "field", "start_time", "peak_time", "end_time"}
    assert shared | {"latitude", "longitude"} <= set(GROUND_TRUTH_COLUMNS) & set(CANDIDATE_COLUMNS)


# ---------------------------------------------------------------------------
# Against a reference written from the rule's paragraph
# ---------------------------------------------------------------------------


def reference(catalog: pd.DataFrame) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    """The rule, one event at a time, as METHODS states it.

    Returns each event's seed and each seed's (found, absorbed) counts.
    """
    events = catalog[catalog["trigger_event_id"].isna()].to_dict("records")
    seed_of: dict[str, str] = {}
    counts: dict[str, tuple[int, int]] = {}
    for key in {(e["instrument"], e["species"], e["baseline_window"]) for e in events}:
        window = [e for e in events if (e["instrument"], e["species"], e["baseline_window"]) == key]
        points = sorted(
            {(e["baseline_quantile"], e["enter_multiple"], e["exit_multiple"]) for e in window}
        )

        def point(e: Any) -> tuple[float, float, float]:
            return (e["baseline_quantile"], e["enter_multiple"], e["exit_multiple"])

        tallest_first = sorted(window, key=lambda e: (-e["peak_z"], e["peak_time"], point(e)))
        for seed in tallest_first:
            if seed["event_id"] in seed_of:
                continue
            seed_of[seed["event_id"]] = seed["event_id"]
            found, absorbed = 1, 0
            for p in points:
                if p == point(seed):
                    continue
                holders = [
                    e
                    for e in window
                    if point(e) == p and e["start_time"] <= seed["peak_time"] < e["end_time"]
                ]
                if not holders:
                    continue
                held = max(holders, key=lambda e: e["start_time"])
                joins = seed["start_time"] <= held["peak_time"] < seed["end_time"]
                if held["event_id"] not in seed_of and joins:
                    seed_of[held["event_id"]] = seed["event_id"]
                    found += 1
                else:
                    absorbed += 1
            counts[seed["event_id"]] = (found, absorbed)
    return seed_of, counts


def random_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """Two windows, two quantiles, two entries and two exits; at each point a few
    disjoint events over ten minutes, with peaks inside them and z drawn from five
    values so that ties happen."""
    rows: list[Row] = []
    for window in (120.0, 600.0):
        for q in (0.01, 0.05):
            for e in (3.0, 4.0):
                for x in (1.0, 2.0):
                    edges = np.sort(rng.choice(600, size=2 * rng.integers(1, 7), replace=False))
                    for k, (a, b) in enumerate(edges.reshape(-1, 2).tolist()):
                        peak = float(rng.integers(a, b))
                        z = float(rng.choice([3.0, 4.0, 4.5, 6.0, 9.0]))
                        name = f"w{window:g}/q{q}/e{e}/x{x}/{k}"
                        rows.append((name, window, q, e, x, a, peak, b, z))
    return catalog_of(rows)


@pytest.mark.parametrize("seed", range(40))
def test_the_rule_agrees_with_its_reference_on_random_catalogs(seed: int) -> None:
    catalog = random_catalog(np.random.default_rng(seed))
    seed_of, counted = reference(catalog)
    linked = link_candidates(catalog)
    table = event_candidates(catalog, states_of(quantiles=(0.01, 0.05), entries=(3.0, 4.0)))
    seed_of_candidate = dict(zip(table["candidate_id"], table["seed_event_id"], strict=True))
    package = {
        str(e): seed_of_candidate[c]
        for e, c in zip(linked["event_id"], linked["candidate_id"], strict=True)
    }
    assert package == seed_of
    found = zip(table["seed_event_id"], table["n_found"], table["n_absorbed"], strict=True)
    assert {s: (int(f), int(a)) for s, f, a in found} == counted
    assert (table["n_found"] + table["n_absorbed"] + table["n_absent"] == 8).all()


# ---------------------------------------------------------------------------
# On a real run: the facts the rule rests on, and what it promises
# ---------------------------------------------------------------------------


def plumy(n: int, seed: int) -> npt.NDArray[np.float64]:
    """Methane at 1900 ppb with 0.7 ppb of noise and plumes of 3 to 40 ppb, some
    close enough together to merge at a low exit."""
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    plumes = np.zeros(n)
    for centre in rng.uniform(100, n - 100, size=40):
        height = rng.choice([3.0, 5.0, 10.0, 40.0])
        plumes += height * np.exp(-0.5 * ((t - centre) / rng.uniform(5, 30)) ** 2)
    values: npt.NDArray[np.float64] = 1900 + rng.normal(0, 0.7, n) + plumes
    return values


@pytest.fixture(scope="module")
def run() -> tuple[xr.Dataset, pd.DataFrame]:
    """Three hours of 1 s methane: its event state and its catalog with candidates,
    over 2 windows x 2 quantiles x 3 entries x 3 exits."""
    n = 3 * 3600
    mid = np.arange(n) * SECOND
    bounds = CellBounds(start_ns=mid - SECOND // 2, stop_ns=mid + SECOND // 2)
    stream = xr.Dataset(
        {"ch4": ("time", plumy(n, 4), {"units": "ppb", "role": "gas", "field": "ch4"})},
        coords={
            "time": mid.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
        attrs={"tsara_stage": "synthetic"},
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    baseline = baseline_state(
        stream,
        instrument="van",
        baseline=BaselineConfig(windows=("2min", "10min"), quantiles=(0.01, 0.05)),
    )
    events = event_state(
        baseline,
        instrument="van",
        events=EventsConfig(enter_multiple=(3.0, 4.0, 5.0), exit_multiple=(1.0, 1.5, 2.0)),
    )
    catalog = link_candidates(event_catalog({"van": events}, {"van": baseline}))
    return events, catalog


def test_a_higher_entry_keeps_the_same_intervals_whose_peak_reaches_it(
    run: tuple[xr.Dataset, pd.DataFrame],
) -> None:
    """The detector bridges dips before it tests entry, so at a fixed window,
    quantile and exit, the events at a higher entry are those at a lower one whose
    peak z reaches it, interval for interval."""
    _, catalog = run
    columns = ["start_time", "peak_time", "end_time", "peak_z"]
    compared = 0
    for _, point in catalog.groupby(["baseline_window", "baseline_quantile", "exit_multiple"]):
        low = point[point["enter_multiple"] == 3.0]
        for entry in (4.0, 5.0):
            high = point[point["enter_multiple"] == entry][columns].reset_index(drop=True)
            kept = low[low["peak_z"] >= entry][columns].reset_index(drop=True)
            assert high.equals(kept)
            compared += len(high)
    assert compared > 0


def test_every_event_at_a_higher_exit_lies_inside_one_at_a_lower_exit(
    run: tuple[xr.Dataset, pd.DataFrame],
) -> None:
    events, _ = run
    event = events["event_ch4"].transpose(
        "time", "baseline_window", "baseline_quantile", "enter_multiple", "exit_multiple"
    )
    values = event.values
    checked = 0
    for x in range(1, values.shape[4]):
        high, low = values[..., x], values[..., x - 1]
        for w, q, e in np.ndindex(*high.shape[1:]):
            h, lo = high[:, w, q, e], low[:, w, q, e]
            for k in np.unique(h[h >= 0]).tolist():
                assert np.unique(lo[h == k]).size == 1
                assert lo[h == k][0] >= 0
                checked += 1
    assert checked > 0


def test_a_candidate_has_at_most_one_event_per_point_each_sharing_its_seeds_peak(
    run: tuple[xr.Dataset, pd.DataFrame],
) -> None:
    events, catalog = run
    table = event_candidates(catalog, {"van": events}).set_index("candidate_id")
    seeds = catalog.set_index("event_id").loc[table["seed_event_id"]]
    seeds.index = table.index
    point = ["baseline_quantile", "enter_multiple", "exit_multiple"]
    assert not catalog.duplicated(["candidate_id", *point]).any()
    joined = catalog.join(seeds, on="candidate_id", rsuffix="_seed")
    assert (joined["start_time"] <= joined["peak_time_seed"]).all()
    assert (joined["peak_time_seed"] < joined["end_time"]).all()
    assert (joined["start_time_seed"] <= joined["peak_time"]).all()
    assert (joined["peak_time"] < joined["end_time_seed"]).all()
    assert (joined["peak_z"] <= joined["peak_z_seed"]).all()
    assert (table["max_peak_z"] == seeds["peak_z"]).all()
    assert (table["n_found"] == catalog.groupby("candidate_id").size().loc[table.index]).all()
    assert (table["n_found"] + table["n_absorbed"] + table["n_absent"] == 18).all()


def test_every_event_belongs_to_one_candidate_of_its_own_window(
    run: tuple[xr.Dataset, pd.DataFrame],
) -> None:
    _, catalog = run
    assert catalog["candidate_id"].notna().all()
    window = catalog["candidate_id"].str.extract(r"/w(\d+)s/")[0].astype(float)
    assert (window == catalog["baseline_window"]).all()
    windows = catalog.groupby("candidate_id")["baseline_window"].nunique()
    assert (windows == 1).all()
