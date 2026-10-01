"""Tests for the event catalog and its tree (tsara.events.catalog)."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.core.support import CellBounds, stream_cells
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import (
    CATALOG_COLUMNS,
    TsaraEventError,
    event_catalog,
    event_state,
    event_states,
    find_events,
    find_records,
    link_parents,
)
from tsara.synthetic import SyntheticDataset
from tsara.synthetic.plumes import GROUND_TRUTH_COLUMNS

HOUR = 3600 * SECOND
ExampleChain: TypeAlias = tuple[SyntheticDataset, xr.Dataset, xr.Dataset]
BASELINE = BaselineConfig(windows=("2min", "10min"), quantiles=(0.05,))
EVENTS = EventsConfig(enter_multiple=(3.0, 4.0), exit_multiple=(1.0,))


def make_stream(
    midpoints_s: npt.NDArray[np.float64],
    values: dict[str, npt.NDArray[np.float64]],
    *,
    width_s: float = 1.0,
    coords: dict[str, object] | None = None,
) -> xr.Dataset:
    """A stream of gas variables on cells of ``width_s`` centred on the midpoints."""
    mid = np.round(midpoints_s * SECOND).astype(np.int64)
    half = int(width_s * SECOND) // 2
    bounds = CellBounds(start_ns=mid - half, stop_ns=mid + half)
    stream = xr.Dataset(
        {
            name: ("time", v, {"units": "ppb", "role": "gas", "field": f"{name}_field"})
            for name, v in values.items()
        },
        coords={
            "time": bounds.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
            **(coords or {}),
        },
        attrs={"tsara_stage": "synthetic"},
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    return stream


def plumy(n: int, seed: int) -> npt.NDArray[np.float64]:
    """Methane at 1900 ppb with 0.7 ppb of noise and a 40 ppb plume every ten minutes."""
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    events = np.zeros(n)
    for centre in np.arange(300, t[-1], 600):
        events += 40 * np.exp(-0.5 * ((t - centre) / 15) ** 2)
    values: npt.NDArray[np.float64] = 1900 + rng.normal(0, 0.7, n) + events
    return values


@pytest.fixture(scope="module")
def van() -> tuple[xr.Dataset, xr.Dataset, pd.DataFrame]:
    """Two hours of 1 s methane: its baseline state, event state and catalog."""
    stream = make_stream(np.arange(7200.0), {"ch4": plumy(7200, 1)})
    baseline = baseline_state(stream, instrument="van", baseline=BASELINE)
    state = event_state(baseline, instrument="van", events=EVENTS)
    return baseline, state, event_catalog({"van": state}, {"van": baseline})


# ---------------------------------------------------------------------------
# Shape and keys
# ---------------------------------------------------------------------------


def test_the_columns_are_the_catalogs_and_the_shared_ones_are_spelled_as_the_answer_key(
    van: tuple[xr.Dataset, xr.Dataset, pd.DataFrame],
    example_chain: ExampleChain,
) -> None:
    _, _, catalog = van
    assert tuple(catalog.columns) == CATALOG_COLUMNS
    shared = set(CATALOG_COLUMNS) & set(GROUND_TRUTH_COLUMNS)
    assert shared == {
        "event_id",
        "parent_event_id",
        "instrument",
        "species",
        "field",
        "start_time",
        "peak_time",
        "end_time",
        "latitude",
        "longitude",
    }
    truth = example_chain[0].ground_truth.to_frame()
    for column in sorted(shared):
        assert catalog[column].dtype == truth[column].dtype, column
    assert (catalog["field"] == "ch4_field").all()


def test_one_row_per_event_and_sweep_point_each_with_a_readable_unique_key(
    van: tuple[xr.Dataset, xr.Dataset, pd.DataFrame],
) -> None:
    _, state, catalog = van
    assert catalog["event_id"].is_unique
    assert catalog["event_id"].iloc[0] == "van.ch4/w120s/q0.05/e3/x1/0"
    for w, e in np.ndindex(2, 2):
        at = (catalog["baseline_window"] == state["baseline_window"].values[w]) & (
            catalog["enter_multiple"] == state["enter_multiple"].values[e]
        )
        assert at.sum() == state["n_events_ch4"].values[w, 0, e, 0]
    assert (
        catalog["events_at_point"]
        == catalog.groupby(["baseline_window", "enter_multiple"])["event_id"].transform("size")
    ).all()


def test_each_row_is_the_detectors_event(van: tuple[xr.Dataset, xr.Dataset, pd.DataFrame]) -> None:
    """Rows rebuilt from the saved membership agree with the detector run again."""
    baseline, state, catalog = van
    cells = stream_cells(state, "van")
    x = baseline["ch4"].values
    records = find_records(cells, np.isfinite(x), gap_ns=2 * HOUR, max_length_ns=6 * HOUR)
    for w, e in np.ndindex(2, 2):
        z = state["z_ch4"].values[:, w, 0]
        found = find_events(
            z,
            cells,
            records,
            enter=EVENTS.enter_multiple[e],
            exit_=1.0,
            max_internal_gap_ns=5 * SECOND,
        )
        rows = catalog[
            (catalog["baseline_window"] == state["baseline_window"].values[w])
            & (catalog["enter_multiple"] == EVENTS.enter_multiple[e])
        ]
        assert rows["event_number"].tolist() == found.number.tolist()
        assert rows["start_time"].astype("int64").tolist() == found.start_ns.tolist()
        assert rows["end_time"].astype("int64").tolist() == found.stop_ns.tolist()
        assert rows["n_readings"].tolist() == found.n_readings.tolist()
        assert rows["covered"].tolist() == found.covered.tolist()
        assert rows["peak_z"].tolist() == found.peak_score.tolist()
        assert (rows["peak_time"].to_numpy() == state["time"].values[found.peak]).all()
        assert rows["peak_enhancement"].tolist() == (
            baseline["enhancement_ch4"].values[found.peak, w, 0].tolist()
        )
        assert rows["duration_s"].tolist() == ((found.stop_ns - found.start_ns) / 1e9).tolist()
        for column in ("clean_level", "clean_spread"):
            at_peak = state[f"{column}_ch4"].values[found.peak, w, 0]
            assert rows[column].tolist() == at_peak.tolist(), column
        assert rows["record"].tolist() == state["record_ch4"].values[found.peak].tolist()
    assert catalog["trigger"].isna().all() and catalog["trigger_event_id"].isna().all()


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


def tree_rows(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    """A hand-made catalog: (event_id, window, start, peak, end, quantile), times in s."""
    base = pd.Timestamp("2026-01-01")
    return pd.DataFrame(
        {
            "event_id": [r[0] for r in rows],
            "instrument": "van",
            "species": "ch4",
            "baseline_window": [r[1] for r in rows],
            "baseline_quantile": [r[5] for r in rows],
            "enter_multiple": 3.0,
            "exit_multiple": 1.0,
            "start_time": [base + pd.Timedelta(seconds=r[2]) for r in rows],
            "peak_time": [
                pd.NaT if np.isnan(r[3]) else base + pd.Timedelta(seconds=r[3]) for r in rows
            ],
            "end_time": [base + pd.Timedelta(seconds=r[4]) for r in rows],
            "duration_s": [r[4] - r[2] for r in rows],
        }
    )


def test_an_event_links_to_the_nearest_longer_window_holding_its_peak_skipping_empty_ones() -> None:
    """Three windows. The blip's peak lies inside the 600 s and the 3600 s events:
    its parent is the nearer. The spike's lies only inside the 3600 s event: the
    600 s window is skipped. The 3600 s event, at the longest window, has none."""
    catalog = tree_rows(
        [
            ("blip", 120, 100, 105, 110, 0.05),
            ("spike", 120, 400, 405, 410, 0.05),
            ("mid", 600, 50, 120, 200, 0.05),
            ("broad", 3600, 0, 300, 1000, 0.05),
        ]
    )
    linked = link_parents(catalog).set_index("event_id")
    assert linked.loc["blip", "parent_event_id"] == "mid"
    assert linked.loc["blip", "parent_duration_s"] == 150.0
    assert linked.loc["spike", "parent_event_id"] == "broad"
    assert linked.loc["mid", "parent_event_id"] == "broad"
    assert pd.isna(linked.loc["broad", "parent_event_id"])
    assert np.isnan(linked.loc["broad", "parent_duration_s"])


def test_an_interval_holds_a_peak_from_its_start_up_to_its_end() -> None:
    catalog = tree_rows(
        [
            ("at_start", 120, 90, 100, 105, 0.05),
            ("at_end", 120, 195, 200, 205, 0.05),
            ("parent", 600, 100, 150, 200, 0.05),
        ]
    )
    linked = link_parents(catalog).set_index("event_id")
    assert linked.loc["at_start", "parent_event_id"] == "parent"
    assert pd.isna(linked.loc["at_end", "parent_event_id"])


def test_trees_do_not_cross_quantiles_and_a_peakless_event_has_no_parent() -> None:
    catalog = tree_rows(
        [
            ("other_q", 120, 100, 105, 110, 0.10),
            ("no_peak", 120, 100, np.nan, 110, 0.05),
            ("parent", 600, 0, 150, 200, 0.05),
        ]
    )
    linked = link_parents(catalog).set_index("event_id")
    assert pd.isna(linked.loc["other_q", "parent_event_id"])
    assert pd.isna(linked.loc["no_peak", "parent_event_id"])


def test_the_links_do_not_depend_on_the_callers_index() -> None:
    catalog = tree_rows([("blip", 120, 100, 105, 110, 0.05), ("mid", 600, 50, 120, 200, 0.05)])
    catalog.index = pd.Index([7, 3])
    linked = link_parents(catalog)
    assert linked.index.tolist() == [7, 3]
    assert linked.loc[7, "parent_event_id"] == "mid"


def test_every_linked_event_lies_inside_its_parent(
    van: tuple[xr.Dataset, xr.Dataset, pd.DataFrame],
) -> None:
    """On a real run: a parent is at a longer window and holds its child's peak."""
    _, _, catalog = van
    child = catalog[catalog["parent_event_id"].notna()]
    assert len(child) > 0
    parents = catalog.set_index("event_id").loc[child["parent_event_id"]]
    assert (parents["baseline_window"].to_numpy() > child["baseline_window"].to_numpy()).all()
    assert (parents["start_time"].to_numpy() <= child["peak_time"].to_numpy()).all()
    assert (child["peak_time"].to_numpy() < parents["end_time"].to_numpy()).all()
    assert (parents["duration_s"].to_numpy() == child["parent_duration_s"].to_numpy()).all()
    assert catalog.loc[catalog["baseline_window"] == 600, "parent_event_id"].isna().all()


# ---------------------------------------------------------------------------
# Triggered variables
# ---------------------------------------------------------------------------


def test_a_canister_row_is_the_triggers_interval_with_its_own_readings() -> None:
    """Fills of 15 s every 5 min, at plume centres and between them, taking a 1 s
    analyzer's events; its baseline a constant zero, so its enhancement is its value."""
    stream = make_stream(np.arange(7200.0), {"ch4": plumy(7200, 2)})
    fills = np.arange(300.0, 7200.0, 300.0)
    canister = make_stream(fills, {"benzene": np.arange(1.0, fills.size + 1)}, width_s=15.0)
    baseline_config = BaselineConfig.model_validate(
        {
            "windows": ["2min", "10min"],
            "quantiles": [0.05],
            "methods": {"iwas.benzene": {"method": "constant", "value": 0.0}},
        }
    )
    baselines = {
        "van": baseline_state(stream, instrument="van", baseline=baseline_config),
        "iwas": baseline_state(canister, instrument="iwas", baseline=baseline_config),
    }
    events = event_states(baselines, EventsConfig(triggers={"iwas": "van.ch4"}))
    catalog = event_catalog(events, baselines)
    rows = catalog[catalog["instrument"] == "iwas"]
    assert len(rows) > 0
    assert (rows["trigger"] == "van.ch4").all()
    # One type per column whatever a column holds: text, missing for the analyzer.
    assert catalog["trigger"].dtype == catalog["trigger_event_id"].dtype == "str"
    assert catalog.loc[catalog["instrument"] == "van", "trigger"].isna().all()
    triggers = catalog.set_index("event_id").loc[rows["trigger_event_id"]]
    assert (rows["start_time"].to_numpy() == triggers["start_time"].to_numpy()).all()
    assert (rows["end_time"].to_numpy() == triggers["end_time"].to_numpy()).all()
    assert rows["event_id"].str.startswith("iwas.benzene/").all()
    assert (rows["record"] == -1).all() and rows["peak_z"].isna().all()
    assert rows["clean_level"].isna().all() and rows["chance_events_at_point"].isna().all()
    for _, row in rows.iterrows():
        mids = pd.to_datetime(fills * 1e9)
        inside = (mids >= row["start_time"]) & (mids < row["end_time"])
        assert row["n_readings"] == inside.sum()
        assert row["peak_enhancement"] == np.arange(1.0, fills.size + 1)[inside].max()
        overlap = sum(
            max(
                0.0,
                min(f + 7.5, row["end_time"].value / 1e9)
                - max(f - 7.5, row["start_time"].value / 1e9),
            )
            for f in fills[inside]
        )
        assert row["covered"] == pytest.approx(overlap / row["duration_s"], rel=1e-12)


def test_a_triggered_variable_whose_trigger_is_missing_is_refused() -> None:
    fills = np.arange(300.0, 7200.0, 300.0)
    stream = make_stream(np.arange(7200.0), {"ch4": plumy(7200, 2)})
    canister = make_stream(fills, {"benzene": np.ones(fills.size)}, width_s=15.0)
    baselines = {
        "van": baseline_state(stream, instrument="van", baseline=BASELINE),
        "iwas": baseline_state(canister, instrument="iwas", baseline=BASELINE),
    }
    events = event_states(baselines, EventsConfig(triggers={"iwas": "van.ch4"}))
    with pytest.raises(TsaraEventError, match="not in the catalog"):
        event_catalog({"iwas": events["iwas"]}, baselines)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def test_a_track_gives_each_event_its_position_at_the_peak_and_a_site_its_own() -> None:
    lat = np.linspace(40.0, 41.0, 7200)
    track = make_stream(
        np.arange(7200.0),
        {"ch4": plumy(7200, 3)},
        coords={"latitude": ("time", lat), "longitude": ("time", -lat)},
    )
    site = make_stream(
        np.arange(7200.0), {"ch4": plumy(7200, 3)}, coords={"latitude": 40.5, "longitude": -111.8}
    )
    for stream, check in (
        (track, lambda peak: (lat[peak], -lat[peak])),
        (site, lambda peak: (np.full(peak.size, 40.5), np.full(peak.size, -111.8))),
    ):
        baseline = baseline_state(stream, instrument="van", baseline=BASELINE)
        state = event_state(baseline, instrument="van", events=EVENTS)
        catalog = event_catalog({"van": state}, {"van": baseline})
        peak = np.searchsorted(state["time"].values, catalog["peak_time"].to_numpy())
        want_lat, want_lon = check(peak)
        assert np.array_equal(catalog["latitude"].to_numpy(), want_lat)
        assert np.array_equal(catalog["longitude"].to_numpy(), want_lon)


def test_a_stream_without_positions_gives_none(
    van: tuple[xr.Dataset, xr.Dataset, pd.DataFrame],
) -> None:
    _, _, catalog = van
    assert catalog["latitude"].isna().all() and catalog["longitude"].isna().all()


# ---------------------------------------------------------------------------
# Refusals, and a catalog of nothing
# ---------------------------------------------------------------------------


def test_what_does_not_pair_up_is_refused(van: tuple[xr.Dataset, xr.Dataset, pd.DataFrame]) -> None:
    baseline, state, _ = van
    with pytest.raises(TsaraEventError, match="reads event states"):
        event_catalog({"van": baseline}, {"van": baseline})
    with pytest.raises(TsaraEventError, match="needs the baseline state"):
        event_catalog({"van": state}, {})
    with pytest.raises(TsaraEventError, match="not on the same readings"):
        event_catalog({"van": state}, {"van": baseline.isel(time=slice(1, None))})


def test_quiet_air_makes_an_empty_catalog_with_every_column() -> None:
    rng = np.random.default_rng(4)
    stream = make_stream(np.arange(3600.0), {"ch4": 1900 + rng.normal(0, 0.7, 3600)})
    baseline = baseline_state(stream, instrument="van", baseline=BASELINE)
    state = event_state(baseline, instrument="van", events=EventsConfig(enter_multiple=(9.0,)))
    catalog = event_catalog({"van": state}, {"van": baseline})
    assert len(catalog) == 0
    assert tuple(catalog.columns) == CATALOG_COLUMNS
    assert catalog["start_time"].dtype == "datetime64[ns]"
    assert catalog["event_id"].dtype == "str" and catalog["n_readings"].dtype == "int64"


# ---------------------------------------------------------------------------
# Scoring is a join: the whole chain against the answer key
# ---------------------------------------------------------------------------


def test_the_example_campaign_scores_as_the_scoping_measured(
    example_chain: ExampleChain,
) -> None:
    """METHODS §6.8. The example chain (the fixture states the rule), scored against
    its answer key by joining on instrument and species: a true event is found when
    a detected interval holds its peak time, and a detected interval overlapping no
    true event's [start_time, end_time] is false. All 59 true events are found at
    every window, and the false events are 2.50, 4.00 and 7.67 an hour at 2, 10 and
    60 min over the record's 6 h."""
    campaign, baseline, state = example_chain
    catalog = event_catalog({"picarro": state}, {"picarro": baseline})
    truth = campaign.ground_truth.to_frame()
    truth = truth[truth["sampled_peak_amplitude"].notna()]
    pairs = catalog.merge(truth, on=["instrument", "species"], suffixes=("", "_true"))
    found = (pairs["start_time"] <= pairs["peak_time_true"]) & (
        pairs["peak_time_true"] <= pairs["end_time"]
    )
    overlaps = (pairs["start_time_true"] <= pairs["end_time"]) & (
        pairs["start_time"] <= pairs["end_time_true"]
    )
    per_hour = []
    for window in (120.0, 600.0, 3600.0):
        at = pairs["baseline_window"] == window
        assert pairs.loc[at & found, "event_id_true"].nunique() == 59
        touching = pairs.loc[at & overlaps, "event_id"].unique()
        detected = catalog.loc[catalog["baseline_window"] == window, "event_id"]
        per_hour.append(round(int((~detected.isin(touching)).sum()) / 6.0, 2))
    assert per_hour == [2.50, 4.00, 7.67]
