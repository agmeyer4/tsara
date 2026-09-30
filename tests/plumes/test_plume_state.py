"""Tests for the plume state (tsara.plumes.state)."""

from __future__ import annotations

import logging
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, PlumesConfig
from tsara.core.support import CellBounds, stream_cells
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.plumes import (
    CHANCE_ASSUMPTION_ATTR,
    PLUMES_STAGE,
    TRIGGER_ATTR,
    TsaraPlumeError,
    clean_air,
    expected_chance_rate,
    find_events,
    find_records,
    plume_state,
    plume_states,
)
from tsara.synthetic import SyntheticDataset
from tsara.synthetic.noise import quantization_floor

HOUR = 3600 * SECOND
ExampleChain: TypeAlias = tuple[SyntheticDataset, xr.Dataset, xr.Dataset]
BASELINE = BaselineConfig(windows=("2min", "10min"), quantiles=(0.05, 0.1))
PLUMES = PlumesConfig(enter_multiple=(3.0, 4.0), exit_multiple=(1.0,))


def make_stream(
    midpoints_s: npt.NDArray[np.float64],
    values: dict[str, npt.NDArray[np.float64]],
    *,
    width_s: float = 1.0,
    sigma: float | None = None,
) -> xr.Dataset:
    """A stream of gas variables on cells of ``width_s`` centred on the midpoints."""
    mid = np.round(midpoints_s * SECOND).astype(np.int64)
    half = int(width_s * SECOND) // 2
    bounds = CellBounds(start_ns=mid - half, stop_ns=mid + half)
    data_vars: dict[str, tuple[str, npt.NDArray[np.float64], dict[str, object]]] = {
        name: ("time", v, {"units": "ppb", "role": "gas", "field": name})
        for name, v in values.items()
    }
    if sigma is not None:
        for name in values:
            data_vars[f"sigma_rand_{name}"] = (
                "time",
                np.full(mid.size, sigma),
                {"units": "ppb", "uncertainty_component": "random"},
            )
    stream = xr.Dataset(
        data_vars,
        coords={
            "time": bounds.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
        attrs={"tsara_stage": "synthetic"},
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    return stream


def plumy(n: int, seed: int, *, every_s: float = 1.0) -> npt.NDArray[np.float64]:
    """Methane at 1900 ppb with 0.7 ppb of noise and a 40 ppb plume every ten minutes."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * every_s
    plumes = np.zeros(n)
    for centre in np.arange(300, t[-1], 600):
        plumes += 40 * np.exp(-0.5 * ((t - centre) / 15) ** 2)
    values: npt.NDArray[np.float64] = 1900 + rng.normal(0, 0.7, n) + plumes
    return values


@pytest.fixture(scope="module")
def dense() -> tuple[xr.Dataset, xr.Dataset]:
    """Two hours of 1 s methane, then a baseline state of it."""
    stream = make_stream(np.arange(7200.0), {"ch4": plumy(7200, 1)}, sigma=0.7)
    return stream, baseline_state(stream, instrument="van", baseline=BASELINE)


# ---------------------------------------------------------------------------
# What the state holds
# ---------------------------------------------------------------------------


def test_the_state_lives_on_the_readings_cells_with_two_more_sweep_dimensions(
    dense: tuple[xr.Dataset, xr.Dataset],
) -> None:
    _, state = dense
    found = plume_state(state, instrument="van", plumes=PLUMES)
    assert found.attrs["tsara_stage"] == PLUMES_STAGE
    assert found.attrs["tsara_instrument"] == "van"
    assert found.attrs["tsara_plumes_record_gap"] == "2h"
    assert found.attrs["tsara_plumes_min_clean_readings"] == "100"
    assert np.array_equal(found["time_bnds"].values, state["time_bnds"].values)
    assert found["enter_multiple"].values.tolist() == [3.0, 4.0]
    assert found["exit_multiple"].values.tolist() == [1.0]
    assert found["record_ch4"].dims == ("time",)
    assert found["z_ch4"].dims == ("time", "baseline_window", "baseline_quantile")
    assert found["event_ch4"].dims == (
        "time",
        "baseline_window",
        "baseline_quantile",
        "enter_multiple",
        "exit_multiple",
    )
    assert found["n_events_ch4"].shape == (2, 2, 2, 1)
    assert found["event_ch4"].dtype == np.int32
    assert CHANCE_ASSUMPTION_ATTR in found["chance_events_ch4"].attrs
    assert "not a measurement uncertainty" in found["clean_spread_ch4"].attrs["long_name"]
    # The reading's declared sigma rides along for comparison, and only it.
    assert np.array_equal(found["sigma_rand_ch4"].values, state["sigma_rand_ch4"].values)
    assert "ch4" not in found and "enhancement_ch4" not in found


def test_the_state_is_the_three_steps_called_by_hand(
    dense: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """Records, clean air, z and events at every sweep point, as stages 2 and 3
    compute them: the state only wires them, and must not transpose or shift."""
    _, state = dense
    found = plume_state(state, instrument="van", plumes=PLUMES)
    cells = stream_cells(state, "van")
    x = state["ch4"].values
    records = find_records(cells, np.isfinite(x), gap_ns=2 * HOUR, max_length_ns=6 * HOUR)
    assert (records.index >= 0).all()  # so the indexing below never wraps a -1
    enhancement = state["enhancement_ch4"].values
    air = clean_air(enhancement, x, records, estimator="half_sample_mode", min_clean_readings=100)
    level = air.level[records.index]
    z = (enhancement - level) / air.spread[records.index]
    assert np.array_equal(found["clean_level_ch4"].values, level, equal_nan=True)
    assert np.array_equal(found["z_ch4"].values, z, equal_nan=True)
    assert np.array_equal(found["record_ch4"].values, records.index)
    for w, q, e in np.ndindex(2, 2, 2):
        enter = PLUMES.enter_multiple[e]
        events = find_events(
            z[:, w, q], cells, records, enter=enter, exit_=1.0, max_internal_gap_ns=5 * SECOND
        )
        assert np.array_equal(found["event_ch4"].values[:, w, q, e, 0], events.membership)
        assert found["n_events_ch4"].values[w, q, e, 0] == events.n
        scored = np.count_nonzero(np.isfinite(z[:, w, q]))
        expected = scored * expected_chance_rate(enter, 1.0)
        assert found["chance_events_ch4"].values[w, q, e, 0] == pytest.approx(expected, rel=1e-15)


def test_every_plume_is_found_and_the_quiet_air_mostly_is_not(
    dense: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """Twelve 40 ppb plumes (57 sigma) in two hours: each peak lies in an event at
    every sweep point, and at entry 4 events are few beyond the plumes."""
    _, state = dense
    found = plume_state(state, instrument="van", plumes=PLUMES)
    peaks = np.arange(300, 7200, 600)
    event = found["event_ch4"].values
    assert np.all(event[peaks] >= 0)
    assert np.all(found["n_events_ch4"].values[:, :, 1, 0] < 12 + 10)


def test_a_stream_of_two_records_is_described_record_by_record() -> None:
    """Two hours of air at 0 ppb over its baseline, then, three hours later, two hours
    whose noise is three times larger: the spreads differ record by record."""
    rng = np.random.default_rng(3)
    t = np.r_[np.arange(7200.0), 5 * 3600 + np.arange(7200.0)]
    ch4 = 1900 + np.r_[rng.normal(0, 0.5, 7200), rng.normal(0, 1.5, 7200)]
    state = baseline_state(make_stream(t, {"ch4": ch4}), instrument="van", baseline=BASELINE)
    found = plume_state(state, instrument="van", plumes=PLUMES)
    assert set(np.unique(found["record_ch4"].values)) == {0, 1}
    spread = found["clean_spread_ch4"].values[:, 1, 0]
    assert 0.4 < spread[0] < 0.6 < 1.2 < spread[-1] < 1.8
    assert "sigma_rand_ch4" not in found


def test_a_reading_in_no_record_has_no_description_no_z_and_no_chance() -> None:
    """A hundred masked readings belong to no record: they get no level, no z, no
    event, and do not count toward the readings chance could make events on."""
    ch4 = plumy(3600, 11)
    ch4[1000:1100] = np.nan
    state = baseline_state(
        make_stream(np.arange(3600.0), {"ch4": ch4}), instrument="van", baseline=BASELINE
    )
    found = plume_state(state, instrument="van", plumes=PLUMES)
    off = np.isnan(ch4)
    assert (found["record_ch4"].values[off] == -1).all()
    assert np.isnan(found["clean_level_ch4"].values[off]).all()
    assert np.isnan(found["z_ch4"].values[off]).all()
    assert (found["n_clean_readings_ch4"].values[off] == 0).all()
    assert (found["event_ch4"].values[off] == -1).all()
    scored = np.count_nonzero(np.isfinite(found["z_ch4"].values[:, 0, 0]))
    assert scored <= 3500
    assert found["chance_events_ch4"].values[0, 0, 0, 0] == pytest.approx(
        scored * expected_chance_rate(3.0, 1.0), rel=1e-15
    )


def test_the_configured_gap_bridges_a_dip_and_a_tiny_one_does_not() -> None:
    """Two 50 ppb readings two seconds apart with one at the baseline between."""
    ch4 = plumy(3600, 12)
    ch4[2000], ch4[2001], ch4[2002] = 1950.0, 1899.0, 1950.0
    state = baseline_state(
        make_stream(np.arange(3600.0), {"ch4": ch4}), instrument="van", baseline=BASELINE
    )
    bridged = plume_state(state, instrument="van", plumes=PLUMES)["event_ch4"].values
    apart = plume_state(state, instrument="van", plumes=PlumesConfig(max_internal_gap="1ns"))[
        "event_ch4"
    ].values
    assert bridged[2001, 0, 0, 0, 0] == bridged[2000, 0, 0, 0, 0] >= 0
    assert apart[2001, 0, 0, 0, 0] == -1
    assert apart[2000, 0, 0, 0, 0] != apart[2002, 0, 0, 0, 0]


def test_a_declared_quantization_floors_the_spread() -> None:
    """Readings declared as written in 5 ppb steps: the spread is held at 5/sqrt 12."""
    stream = make_stream(np.arange(3600.0), {"ch4": plumy(3600, 13)})
    stream["ch4"].attrs["quantization"] = 5.0
    state = baseline_state(stream, instrument="van", baseline=BASELINE)
    spread = plume_state(state, instrument="van", plumes=PLUMES)["clean_spread_ch4"].values
    assert np.all(spread == quantization_floor(5.0))


# ---------------------------------------------------------------------------
# Blanks
# ---------------------------------------------------------------------------


def test_a_short_record_is_blank_and_finds_nothing_while_its_neighbour_does(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A record of 120 readings has about 60 below its level, under the 100 asked."""
    t = np.r_[np.arange(3600.0), 4 * 3600 + np.arange(120.0)]
    state = baseline_state(
        make_stream(t, {"ch4": plumy(3720, 5)}), instrument="van", baseline=BASELINE
    )
    with caplog.at_level(logging.WARNING, logger="tsara.plumes.state"):
        found = plume_state(state, instrument="van", plumes=PLUMES)
    short = found["record_ch4"].values == 1
    assert np.isnan(found["clean_level_ch4"].values[short]).all()
    assert np.isnan(found["z_ch4"].values[short]).all()
    assert (found["event_ch4"].values[short] == -1).all()
    assert (found["n_clean_readings_ch4"].values[short] < 100).all()
    assert (found["n_clean_readings_ch4"].values[~short] >= 100).all()
    # Blank somewhere is ordinary; blank everywhere is what the warning is for.
    assert "every record is blank" not in caplog.text


def test_a_variable_blank_at_every_record_is_named_in_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    t = np.arange(150.0)
    stream = make_stream(t, {"ch4": plumy(150, 6), "co2": plumy(150, 7)})
    state = baseline_state(stream, instrument="van", baseline=BASELINE)
    with caplog.at_level(logging.WARNING, logger="tsara.plumes.state"):
        found = plume_state(state, instrument="van", plumes=PLUMES)
    assert (found["event_ch4"].values == -1).all()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "ch4, co2:" in warnings[0].getMessage()
    assert "120 s, q 0.05 (at most" in warnings[0].getMessage()
    assert "A record needs 100 readings below its clean level" in warnings[0].getMessage()


def test_the_warning_names_eight_and_counts_the_rest(caplog: pytest.LogCaptureFixture) -> None:
    """Asked of the formatter directly, as the baseline stage's warning is: nine
    variables sharing one pattern, and nine patterns, are rare enough in a stream
    that building one would test the fixture more than the rule. A canister's 56
    VOCs sharing one blank pattern is the case the first half is for."""
    from tsara.plumes.state import _warn_blank_everywhere

    shared = {f"voc{k}": [("120 s, q 0.05", 3)] for k in range(9)}
    distinct = {f"v{k}": [(f"{k + 1} s, q 0.05", 3)] for k in range(9)}
    with caplog.at_level(logging.WARNING, logger="tsara.plumes.state"):
        _warn_blank_everywhere("iwas", shared, 100)
        _warn_blank_everywhere("lab", distinct, 100)
    first, second = (r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "voc7 and 1 more: 120 s, q 0.05 (at most 3)" in first
    assert "voc8" not in first
    assert "and 1 more pattern(s)" in second and "v8:" not in second


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_what_is_not_a_baseline_state_or_holds_no_enhancement_is_refused(
    dense: tuple[xr.Dataset, xr.Dataset],
) -> None:
    stream, state = dense
    with pytest.raises(TsaraPlumeError, match="reads a baseline state"):
        plume_state(stream, instrument="van", plumes=PLUMES)
    with pytest.raises(TsaraPlumeError, match="no enhancement of \\['co2'\\]"):
        plume_state(state, instrument="van", plumes=PLUMES, variables=["co2"])


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


def test_a_sparse_variable_takes_the_events_of_a_sibling_on_its_own_instrument(
    dense: tuple[xr.Dataset, xr.Dataset],
) -> None:
    """Benzene written once a minute on the van's clock takes methane's events: a
    finite benzene reading belongs to the methane event whose interval holds its
    cell midpoint, under that event's number."""
    stream, _ = dense
    benzene = np.full(7200, np.nan)
    benzene[::60] = 0.2
    both = stream.assign(benzene=("time", benzene, {"units": "ppb", "role": "gas"}))
    state = baseline_state(both, instrument="van", baseline=BASELINE)
    plumes = PlumesConfig(
        enter_multiple=(3.0, 4.0), exit_multiple=(1.0,), triggers={"van.benzene": "van.ch4"}
    )
    found = plume_state(state, instrument="van", plumes=plumes)
    assert "z_benzene" not in found and "record_benzene" not in found
    assert found["event_benzene"].attrs[TRIGGER_ATTR] == "van.ch4"
    ch4, taken = found["event_ch4"].values, found["event_benzene"].values
    cells = stream_cells(state, "van")
    for point in np.ndindex(*ch4.shape[1:]):
        membership = ch4[(slice(None), *point)]
        expected = np.full(7200, -1)
        for k in np.unique(membership[membership >= 0]):
            rows = np.flatnonzero(membership == k)
            inside = (cells.midpoint_ns >= cells.start_ns[rows[0]]) & (
                cells.midpoint_ns < cells.stop_ns[rows[-1]]
            )
            expected[inside & np.isfinite(benzene)] = k
        assert np.array_equal(taken[(slice(None), *point)], expected), point
        assert found["n_events_benzene"].values[point] == np.unique(expected[expected >= 0]).size
    assert (taken[np.isnan(benzene)] == -1).all()
    assert (taken >= 0).any()


def test_a_canister_takes_its_events_from_an_analyzer_on_another_instrument() -> None:
    """A canister filling for 15 s every 5 min beside a 1 s analyzer: plume_states
    finds the analyzer's events first, then hands them to the canister. Fills at
    the plumes' centres (300 + 600 k s) lie in the analyzer's events and take their
    numbers; fills midway between plumes (600 k s, 20 plume widths away) lie in
    none, bar chance."""
    van = make_stream(np.arange(7200.0), {"ch4": plumy(7200, 2)})
    fills = np.arange(300.0, 7200.0, 300.0)
    canister = make_stream(fills, {"benzene": np.full(fills.size, 0.3)}, width_s=15.0)
    states = {
        "van": baseline_state(van, instrument="van", baseline=BASELINE),
        "iwas": baseline_state(canister, instrument="iwas", baseline=BASELINE),
    }
    plumes = PlumesConfig(triggers={"iwas": "van.ch4"})
    found = plume_states(states, plumes)
    assert sorted(found) == ["iwas", "van"]
    assert found["iwas"]["event_benzene"].attrs[TRIGGER_ATTR] == "van.ch4"
    assert found["van"].identical(plume_state(states["van"], instrument="van", plumes=plumes))
    taken = found["iwas"]["event_benzene"].values[:, 0, 0, 0, 0]
    van_events = found["van"]["event_ch4"].values[:, 0, 0, 0, 0]
    in_plume = (fills - 300) % 600 == 0
    assert (taken[in_plume] >= 0).all()
    assert (taken[~in_plume] == -1).mean() > 0.8
    for fill, k in zip(fills, taken, strict=True):
        if k >= 0:
            rows = np.flatnonzero(van_events == k)
            assert rows[0] - 0.5 <= fill < rows[-1] + 0.5


def test_an_instrument_with_both_kinds_of_variable_keeps_both() -> None:
    """One instrument, two gases: carbon dioxide finds its own events, ethane is
    told to take methane's from another instrument, and the merged state holds
    both kinds of column."""
    van = make_stream(np.arange(7200.0), {"ch4": plumy(7200, 2)})
    lab = make_stream(np.arange(7200.0), {"co2": plumy(7200, 4), "c2h6": plumy(7200, 5)})
    states = {
        "van": baseline_state(van, instrument="van", baseline=BASELINE),
        "lab": baseline_state(lab, instrument="lab", baseline=BASELINE),
    }
    found = plume_states(states, PlumesConfig(triggers={"lab.c2h6": "van.ch4"}))
    assert "z_co2" in found["lab"] and "z_c2h6" not in found["lab"]
    assert found["lab"]["event_c2h6"].attrs[TRIGGER_ATTR] == "van.ch4"


def test_a_triggered_reading_belongs_from_an_events_start_up_to_its_stop() -> None:
    """Half-open, as a cell is. A reading of 1 s cells centred on the half-seconds
    has its midpoint exactly on a trigger cell's edge: the one on an event's
    first cell start belongs to it, the one on its last cell stop does not."""
    ch4 = plumy(3600, 14)
    ch4[1000:1003] = 1950.0
    ch4[999] = ch4[1003] = 1899.0
    van = make_stream(np.arange(3600.0), {"ch4": ch4})
    edges = np.arange(3600.0) - 0.5
    lab = make_stream(edges, {"c2h6": np.full(3600, 2.0)})
    states = {
        "van": baseline_state(van, instrument="van", baseline=BASELINE),
        "lab": baseline_state(lab, instrument="lab", baseline=BASELINE),
    }
    # Nothing bridged, so that noise crossings near the edges stay apart.
    plumes = PlumesConfig(max_internal_gap="1ns", triggers={"lab": "van.ch4"})
    found = plume_states(states, plumes)
    k = found["van"]["event_ch4"].values[1000, 0, 0, 0, 0]
    assert (found["van"]["event_ch4"].values[999:1004, 0, 0, 0, 0] == [-1, k, k, k, -1]).all()
    taken = found["lab"]["event_c2h6"].values[:, 0, 0, 0, 0]
    # lab reading r has its midpoint at r - 0.5 s: 1000 sits on the event's start
    # (999.5 s) and 1003 on its stop (1002.5 s).
    assert taken[1000] == k and taken[1003] == -1


def test_a_trigger_state_missing_or_found_on_another_sweep_is_refused(
    dense: tuple[xr.Dataset, xr.Dataset],
) -> None:
    _, state = dense
    other = make_stream(np.arange(7200.0), {"co2": plumy(7200, 9)})
    lab = baseline_state(other, instrument="lab", baseline=BASELINE)
    plumes = PlumesConfig(triggers={"lab": "van.ch4"})
    with pytest.raises(TsaraPlumeError, match="was not handed over"):
        plume_state(lab, instrument="lab", plumes=plumes)
    elsewhere = plume_state(state, instrument="van", plumes=PlumesConfig(enter_multiple=(5.0,)))
    with pytest.raises(TsaraPlumeError, match="found on another sweep"):
        plume_state(lab, instrument="lab", plumes=plumes, triggers={"van": elsewhere})


# ---------------------------------------------------------------------------
# The whole chain, against the scoping measurement
# ---------------------------------------------------------------------------


def test_the_example_campaign_reproduces_the_scoping_measurement(
    example_chain: ExampleChain,
) -> None:
    """METHODS §6.8: through the whole chain (the fixture states the rule) the clean
    spread is 1.01, 1.02 and 1.45 times the true sigma of 0.6 ppb, and 31, 33 and
    42 % of readings lie in events."""
    _, _, plume = example_chain
    spread = plume["clean_spread_ch4"].values[:, :, 0]
    event = plume["event_ch4"].values[:, :, 0, 0, 0]
    assert [round(float(np.nanmedian(spread[:, w])) / 0.6, 2) for w in range(3)] == [
        1.01,
        1.02,
        1.45,
    ]
    assert [round(100 * float(np.mean(event[:, w] >= 0))) for w in range(3)] == [31, 33, 42]
