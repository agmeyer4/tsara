"""Tests for the synthetic atmosphere: one truth, sampled by every instrument.

The assertions that carry the redesign are the exact ones. A noise-free
instrument is computed *by* the atmosphere rather than beside it, so its
values must equal the atmosphere's answer to the last bit -- not to a
tolerance -- and two instruments measuring one field must agree wherever
their instants coincide, random wander included.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.core.support import CellBounds
from tsara.synthetic import generate
from tsara.synthetic.atmosphere import CellGrid, realize_atmosphere
from tsara.synthetic.config import SyntheticConfig
from tsara.synthetic.generator import TRUTH_PREFIX

SECOND = 1_000_000_000


def _wandering_spec(**instruments: Any) -> dict[str, Any]:
    """A campaign whose truth has every term: diurnal, drift, walk, EMG plumes."""
    return {
        "name": "atmosphere",
        "start": "2026-06-15T14:00:00Z",
        "duration": "1h",
        "seed": 3,
        "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
        "atmosphere": {
            "fields": {
                "ch4": {
                    "units": "ppb",
                    "background": {
                        "kind": "parametric",
                        "offset": 1900.0,
                        "diurnal_amplitude": 30.0,
                        "drift_per_day": 60.0,
                        "random_walk_std": 20.0,
                    },
                },
                "wind_dir": {
                    "role": "met",
                    "circular": True,
                    "units": "degrees",
                    "background": {"kind": "parametric", "offset": 200.0, "random_walk_std": 500.0},
                },
            },
            "sources": {
                "pad": {
                    "rate_per_hour": 40.0,
                    "reference_species": "ch4",
                    "shape": {"kind": "emg", "sigma": "8s", "tau": "20s"},
                    "amplitude": {"kind": "lognormal", "median": 100.0, "sigma_log": 0.5},
                }
            },
        },
        "instruments": instruments or {"fast": {"native_rate": "1s", "measures": {"ch4": {}}}},
    }


def _cells_of(stream: Any) -> CellBounds:
    bounds = stream["time_bnds"].values.astype("datetime64[ns]").astype(np.int64)
    return CellBounds(start_ns=bounds[:, 0], stop_ns=bounds[:, 1])


# ---------------------------------------------------------------------------
# The two identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["mid", "start", "end"])
def test_a_perfect_point_instrument_reads_exactly_the_atmosphere(label: str) -> None:
    """At its own timestamps -- the cell midpoints -- whatever its label."""
    spec = _wandering_spec(
        a={
            "native_rate": "1s",
            "timestamp_jitter": "0.3s",
            "support": {"label": label},
            "measures": {"ch4": {}},
        }
    )
    dataset = generate(SyntheticConfig.model_validate(spec))
    assert dataset.atmosphere is not None
    stream = dataset.streams["a"]
    assert np.array_equal(
        stream["ch4"].values, dataset.atmosphere.value("ch4", stream["time"].values)
    )


def test_a_perfect_mean_instrument_reads_exactly_the_atmosphere_over_its_cells() -> None:
    spec = _wandering_spec(
        minute={
            "native_rate": "60s",
            "support": {"method": "mean", "label": "start", "subsamples": 256},
            "measures": {"ch4": {}},
        }
    )
    dataset = generate(SyntheticConfig.model_validate(spec))
    assert dataset.atmosphere is not None
    stream = dataset.streams["minute"]
    expected = dataset.atmosphere.mean_over("ch4", _cells_of(stream), subsamples=256)
    assert np.array_equal(stream["ch4"].values, expected)


def test_the_truth_columns_are_the_atmosphere_decomposed_over_the_cells() -> None:
    spec = _wandering_spec(
        minute={
            "native_rate": "60s",
            "support": {"method": "mean", "subsamples": 32},
            "measures": {"ch4": {"name": "methane"}},
        }
    )
    dataset = generate(SyntheticConfig.model_validate(spec))
    assert dataset.atmosphere is not None
    stream = dataset.streams["minute"]
    truth = dataset.atmosphere.over_cells("ch4", CellGrid.build(_cells_of(stream), 32))
    assert np.array_equal(stream[f"{TRUTH_PREFIX}background_methane"].values, truth.background)
    assert np.array_equal(stream[f"{TRUTH_PREFIX}enhancement_methane"].values, truth.enhancement)


# ---------------------------------------------------------------------------
# One atmosphere, many instruments
# ---------------------------------------------------------------------------


def test_two_instruments_on_one_field_see_the_same_wander() -> None:
    """The defect the redesign exists to remove.

    Before Phase 4.5 each instrument drew its own random walk, and two
    analyzers configured identically disagreed by as much as the background
    varied. Here a 1 s and a 10 s instrument land on the same instants every
    ten seconds and must read identical truth there.
    """
    spec = _wandering_spec(
        fast={"native_rate": "1s", "measures": {"ch4": {}}},
        slow={"native_rate": "10s", "measures": {"ch4": {}}},
    )
    dataset = generate(SyntheticConfig.model_validate(spec))
    fast = dataset.streams["fast"]
    slow = dataset.streams["slow"]
    assert np.array_equal(fast["time"].values[::10], slow["time"].values)
    assert np.array_equal(fast["ch4"].values[::10], slow["ch4"].values)


def test_a_mean_and_a_point_instrument_agree_on_a_drift_exactly() -> None:
    """The drift-origin quirk, gone.

    Drift used to be measured from the first instant each instrument
    rendered, so a mean instrument (whose first sub-sample precedes its first
    midpoint) read a constant offset from a point instrument. Averaging a
    linear function over a cell is its value at the midpoint, so with drift
    measured from the campaign start the two agree at every midpoint.
    """
    spec = _wandering_spec(
        point={"native_rate": "60s", "measures": {"ch4": {}}},
        mean={"native_rate": "60s", "support": {"method": "mean"}, "measures": {"ch4": {}}},
    )
    spec["atmosphere"]["fields"]["ch4"]["background"] = {
        "kind": "parametric",
        "offset": 1900.0,
        "drift_per_day": 60.0,
    }
    spec["atmosphere"]["sources"] = {}
    dataset = generate(SyntheticConfig.model_validate(spec))
    point = dataset.streams["point"]["ch4"].values
    mean = dataset.streams["mean"]["ch4"].values
    assert np.allclose(point, mean, rtol=0, atol=1e-9)


def test_the_atmosphere_does_not_depend_on_who_measures_it() -> None:
    """Drawn before the platform and the instruments, so neither can move it."""
    lone = SyntheticConfig.model_validate(_wandering_spec())
    crowded_spec = _wandering_spec(
        a={"native_rate": "0.5s", "timestamp_jitter": "0.1s", "measures": {"ch4": {}}},
        b={
            "native_rate": "10s",
            "dropouts": {"rate_per_day": 500.0, "duration": "60s"},
            "measures": {"ch4": {"uncertainty": {"random": {"absolute": 3.0}}}, "wind_dir": {}},
        },
    )
    crowded_spec["platform"] = {"kind": "mobile", "start_latitude": 40.0, "start_longitude": -111.0}
    crowded = SyntheticConfig.model_validate(crowded_spec)

    first = generate(lone).atmosphere
    second = generate(crowded).atmosphere
    assert first is not None and second is not None
    instants = pd.date_range("2026-06-15T14:00:00", "2026-06-15T15:00:00", freq="3100ms")
    for field in ("ch4", "wind_dir"):
        assert np.array_equal(first.value(field, instants), second.value(field, instants))


def test_realizing_the_atmosphere_alone_reproduces_the_generated_one() -> None:
    """What a loaded bundle relies on: the air is the generator's first draws."""
    config = SyntheticConfig.model_validate(_wandering_spec())
    generated = generate(config).atmosphere
    rebuilt = realize_atmosphere(config, np.random.default_rng(config.seed))
    assert generated is not None
    instants = pd.date_range("2026-06-15T14:00:00", periods=500, freq="7s")
    assert np.array_equal(rebuilt.value("ch4", instants), generated.value("ch4", instants))


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def _atmosphere(spec: dict[str, Any] | None = None) -> Any:
    config = SyntheticConfig.model_validate(spec or _wandering_spec())
    return realize_atmosphere(config, np.random.default_rng(config.seed))


def test_value_is_background_plus_enhancement() -> None:
    atmosphere = _atmosphere()
    instants = pd.date_range("2026-06-15T14:00:00", periods=3600, freq="1s")
    assert np.array_equal(
        atmosphere.value("ch4", instants),
        atmosphere.background("ch4", instants) + atmosphere.enhancement("ch4", instants),
    )


def test_one_event_is_its_amplitude_times_its_kernel() -> None:
    spec = _wandering_spec()
    spec["atmosphere"]["sources"]["pad"]["rate_per_hour"] = 1.0
    spec["seed"] = 0  # a Poisson(1) draw of exactly one event
    atmosphere = _atmosphere(spec)
    assert len(atmosphere.events) == 1
    (event,) = atmosphere.events
    instants = pd.date_range(event.center_time - pd.Timedelta("30s"), periods=400, freq="500ms")
    offsets = (instants - event.species_center("ch4")).total_seconds().to_numpy()
    expected = event.amplitudes["ch4"] * event.kernel.evaluate(offsets)
    assert np.allclose(atmosphere.enhancement("ch4", instants), expected, rtol=1e-12, atol=0)
    assert atmosphere.enhancement("ch4", instants).max() > 0.0


def test_queries_do_not_depend_on_the_order_instants_are_asked_in() -> None:
    atmosphere = _atmosphere()
    instants = pd.date_range("2026-06-15T14:00:00", periods=2000, freq="1700ms").to_numpy()
    shuffled = np.random.default_rng(0).permutation(instants.size)
    assert np.array_equal(
        atmosphere.value("ch4", instants[shuffled]), atmosphere.value("ch4", instants)[shuffled]
    )


def test_timezone_aware_and_naive_instants_are_one_instant() -> None:
    atmosphere = _atmosphere()
    naive = pd.date_range("2026-06-15T14:00:00", periods=100, freq="13s")
    aware = naive.tz_localize("UTC").tz_convert("America/Denver")
    assert np.array_equal(atmosphere.value("ch4", aware), atmosphere.value("ch4", naive))
    assert np.array_equal(atmosphere.value("ch4", list(naive)), atmosphere.value("ch4", naive))


def test_a_field_without_plumes_has_no_enhancement() -> None:
    atmosphere = _atmosphere()
    instants = pd.date_range("2026-06-15T14:00:00", periods=3600, freq="1s")
    assert np.all(atmosphere.enhancement("wind_dir", instants) == 0.0)
    assert (
        atmosphere.over_cells(
            "wind_dir",
            CellGrid.build(CellBounds(start_ns=np.array([0]), stop_ns=np.array([SECOND])), 4),
        ).peaks
        == ()
    )


@pytest.mark.parametrize("query", ["background", "enhancement", "value"])
def test_an_unknown_field_is_named_not_answered_with_zeros(query: str) -> None:
    atmosphere = _atmosphere()
    with pytest.raises(KeyError, match=r"no field 'co2'.*\['ch4', 'wind_dir'\]"):
        getattr(atmosphere, query)("co2", pd.date_range("2026-06-15T14:00:00", periods=3))


def test_an_event_no_cell_overlaps_has_no_sampled_peak() -> None:
    atmosphere = _atmosphere()
    far = int(pd.Timestamp("2030-01-01").value)
    cells = CellBounds(start_ns=np.array([far]), stop_ns=np.array([far + 60 * SECOND]))
    truth = atmosphere.over_cells("ch4", CellGrid.build(cells, 8))
    assert len(truth.peaks) == len(atmosphere.events)
    assert all(np.isnan(peak.sampled_peak) for peak in truth.peaks)
    assert truth.enhancement[0] == 0.0


# ---------------------------------------------------------------------------
# The cell grid
# ---------------------------------------------------------------------------


def test_one_subsample_sits_on_each_cell_midpoint() -> None:
    cells = CellBounds(
        start_ns=np.array([0, 10 * SECOND]), stop_ns=np.array([4 * SECOND, 20 * SECOND])
    )
    grid = CellGrid.build(cells, 1)
    assert np.array_equal(grid.fine_ns[:, 0], cells.midpoint_ns)
    assert grid.subsamples == 1


def test_subsamples_divide_cells_of_different_widths_evenly() -> None:
    cells = CellBounds(start_ns=np.array([0, 100]), stop_ns=np.array([8, 140]))
    grid = CellGrid.build(cells, 4)
    assert grid.fine_ns.tolist() == [[1, 3, 5, 7], [105, 115, 125, 135]]


def test_a_cell_mean_needs_a_subsample() -> None:
    cells = CellBounds(start_ns=np.array([0]), stop_ns=np.array([SECOND]))
    with pytest.raises(ValueError, match="at least 1 subsample"):
        CellGrid.build(cells, 0)


def test_mean_over_converges_to_the_closed_form_dilution() -> None:
    """A Gaussian wholly inside a wide cell averages to A sigma sqrt(2 pi) / W."""
    spec = copy.deepcopy(_wandering_spec())
    spec["atmosphere"]["fields"]["ch4"]["background"] = {"kind": "parametric", "offset": 0.0}
    spec["atmosphere"]["sources"] = {
        "blip": {
            "rate_per_hour": 1.0,
            "reference_species": "ch4",
            "shape": {"kind": "gaussian", "sigma": "3s"},
            "amplitude": {"kind": "uniform", "low": 100.0, "high": 100.001},
        }
    }
    spec["seed"] = 0  # a Poisson(1) draw of exactly one event
    atmosphere = _atmosphere(spec)
    (event,) = atmosphere.events
    centre = int(event.species_center("ch4").value)
    cells = CellBounds(
        start_ns=np.array([centre - 30 * SECOND]), stop_ns=np.array([centre + 30 * SECOND])
    )
    closed_form = event.amplitudes["ch4"] * 3.0 * np.sqrt(2.0 * np.pi) / 60.0
    # The kernel is truncated at 4 sigma, which removes 6.3e-5 of its area.
    truncation = 6.4e-5 * closed_form
    assert atmosphere.mean_over("ch4", cells, 4096)[0] == pytest.approx(closed_form, abs=truncation)


# ---------------------------------------------------------------------------
# Locating events: two cases no generated campaign reaches
# ---------------------------------------------------------------------------


def _one_event_atmosphere(center_ns: int) -> Any:
    """An atmosphere holding exactly one Gaussian event, built by hand.

    Hand-built because both cases below need an event at a chosen instant,
    and a scheduled event lands wherever its Poisson draw puts it.
    """
    from tsara.synthetic.atmosphere import Atmosphere
    from tsara.synthetic.background import RealizedBackground
    from tsara.synthetic.config import FieldSpec, GaussianShape, ParametricBackground
    from tsara.synthetic.plumes import RealizedEvent, build_kernel

    center = pd.Timestamp(center_ns)
    event = RealizedEvent(
        event_id="pad_00000",
        source_name="pad",
        center_time=center,
        kernel=build_kernel(GaussianShape(kind="gaussian", sigma="3s")),
        reference_species="ch4",
        amplitudes={"ch4": 100.0},
        ratios={"ch4": 1.0},
    )
    flat = RealizedBackground(
        offset=0.0,
        diurnal_amplitude=0.0,
        diurnal_period_s=86_400.0,
        diurnal_phase_s=0.0,
        drift_per_day=0.0,
        origin_s=center_ns / 1e9,
        stochastic=(),
    )
    return Atmosphere(
        start=center - pd.Timedelta("1h"),
        end=center + pd.Timedelta("1h"),
        fields={"ch4": FieldSpec(background=ParametricBackground(kind="parametric", offset=0.0))},
        backgrounds={"ch4": flat},
        events=(event,),
    )


def test_cells_of_different_widths_cannot_hide_an_event_from_the_search() -> None:
    """Why the search runs over the running maximum of the cell stops.

    A generated instrument's cells share one width, so their stops are sorted
    whenever their starts are and the running maximum changes nothing. The
    public `mean_over` takes any cells. Here the second cell nests inside the
    first and ends before the event's window opens, so the raw stops descend,
    and a binary search over them skips the first cell -- which overlaps the
    window and holds part of the event.
    """
    center_ns = int(pd.Timestamp("2026-06-15T14:30:00").value)
    atmosphere = _one_event_atmosphere(center_ns)
    (event,) = atmosphere.events
    window_start = int(event.species_window("ch4")[0].value)
    cells = CellBounds(
        start_ns=np.array([window_start - 5 * SECOND, window_start - 4 * SECOND, window_start]),
        stop_ns=np.array(
            [window_start + 5 * SECOND, window_start - SECOND, window_start + 30 * SECOND]
        ),
    )
    grid = CellGrid.build(cells, 64)
    # Every instant evaluated, no search at all.
    offsets = grid.fine_ns / 1e9 - center_ns / 1e9
    expected = (100.0 * event.kernel.evaluate(offsets.reshape(-1))).reshape(3, 64).mean(axis=1)
    produced = atmosphere.over_cells("ch4", grid).enhancement
    assert expected[0] > 0.0, "the first cell must hold part of the event, or this proves nothing"
    assert np.allclose(produced, expected, rtol=0, atol=1e-12)


def test_an_instant_rounding_puts_inside_the_support_is_evaluated_at_any_query() -> None:
    """Why the instant path widens its window by a millisecond.

    The window is judged in integer nanoseconds and the kernel's support in
    float seconds, where one step at epoch scale is ~0.24 us. At this center
    (found by search) the instant 1 ns before the window opens is still inside
    the support by the kernel's arithmetic, so a cell centred on it sees the
    event's edge. Without the margin the instant query would not, and a point
    instrument would no longer read exactly the atmosphere.
    """
    center_ns = 1_781_533_800_850_624_225
    atmosphere = _one_event_atmosphere(center_ns)
    (event,) = atmosphere.events
    instant = int(event.species_window("ch4")[0].value) - 1
    cell = CellBounds(
        start_ns=np.array([instant - SECOND // 2]), stop_ns=np.array([instant + SECOND // 2])
    )
    at_the_cell = atmosphere.mean_over("ch4", cell, subsamples=1)
    at_the_instant = atmosphere.enhancement("ch4", np.array([instant], dtype="datetime64[ns]"))
    assert at_the_cell[0] > 0.0, "the case must reach the kernel's edge, or this proves nothing"
    assert np.array_equal(at_the_instant, at_the_cell)
