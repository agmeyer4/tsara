"""Tests for cross-species pairing.

This is the stage that combines measurements, so the evidence follows METHODS
§11.1 rather than checking the code against itself. In order of strength:

* a two-cell fixture whose paired value can be worked out on paper;
* an independent slow reimplementation of the whole pairing, written from the
  definition, scored against the vectorized one on random cells;
* two closed forms -- the mean of a linear ramp over a cell is its value at
  the cell midpoint, and a species paired against a constant multiple of
  itself must return that multiple exactly;
* invariants that need no ground truth -- binning a stream onto its own cells
  is the identity, and a pair is never fabricated where data is absent.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.align import PairedSpecies, TsaraAlignError, pair_species
from tsara.core.naming import sigma_rand_name, sigma_sys_name
from tsara.core.propagation import propagate_random, propagate_systematic
from tsara.core.timebase import SECOND_NS as SECOND


def make_stream(
    start_s: float,
    width_s: float,
    n: int,
    variables: dict[str, np.ndarray],
    *,
    attrs: dict[str, dict[str, object]] | None = None,
    method: str = "point",
) -> xr.Dataset:
    """Build a minimal stream: abutting cells of a fixed width, with bounds."""
    start = (np.arange(n, dtype=np.int64) * int(width_s * SECOND)) + int(start_s * SECOND)
    stop = start + int(width_s * SECOND)
    variable_attrs = attrs or {}
    data = {
        name: ("time", values, dict(variable_attrs.get(name, {"units": "ppb"})))
        for name, values in variables.items()
    }
    dataset = xr.Dataset(
        data_vars=data,
        coords={
            "time": (start + (stop - start) // 2).astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([start, stop], axis=1).astype("datetime64[ns]"),
            ),
        },
    )
    dataset["time"].attrs["bounds"] = "time_bnds"
    for name in variables:
        dataset[name].attrs.setdefault("cell_methods", f"time: {method}")
    return dataset


def slow_pair(
    readings: xr.Dataset, readings_name: str, target: xr.Dataset, target_name: str
) -> list[float]:
    """Bin one species onto another's cells with a Python loop.

    Written from the definition in METHODS §1.3 and deliberately naive: for
    every target cell, walk every reading, compute the overlap by hand and
    accumulate. The production path uses a binary search, an index expansion
    and four ``bincount`` calls; this uses none of that, which is what makes
    the comparison worth anything.
    """
    s_start = readings["time_bnds"].values[:, 0].astype("int64")
    s_stop = readings["time_bnds"].values[:, 1].astype("int64")
    t_start = target["time_bnds"].values[:, 0].astype("int64")
    t_stop = target["time_bnds"].values[:, 1].astype("int64")
    values = readings[readings_name].values
    out = []
    for t in range(len(t_start)):
        total = weighted = 0.0
        for s in range(len(s_start)):
            if not np.isfinite(values[s]):
                continue
            overlap = max(min(s_stop[s], t_stop[t]) - max(s_start[s], t_start[t]), 0)
            if overlap == 0:
                continue
            total += overlap
            weighted += overlap * values[s]
        out.append(weighted / total if total else float("nan"))
    return out


@pytest.fixture()
def two_rates() -> dict[str, xr.Dataset]:
    """A 1 s stream and a 4 s stream over the same ten seconds."""
    fast = make_stream(0.0, 1.0, 10, {"ch4": np.arange(10.0)})
    slow = make_stream(0.0, 4.0, 2, {"co2": np.array([100.0, 200.0])})
    return {"fast": fast, "slow": slow}


# ---------------------------------------------------------------------------
# Which clock, and why
# ---------------------------------------------------------------------------


def test_the_wider_supported_stream_is_the_clock(two_rates: dict[str, xr.Dataset]) -> None:
    paired = pair_species(two_rates, "ch4", "co2")
    assert paired.clock == "slow"
    assert "wider cells (4 s vs 1 s)" in paired.dataset.attrs["tsara_pairing_clock_reason"]


def test_the_clock_does_not_depend_on_argument_order(two_rates: dict[str, xr.Dataset]) -> None:
    """Which species is numerator cannot change which air was compared."""
    assert pair_species(two_rates, "ch4", "co2").clock == "slow"
    assert pair_species(two_rates, "co2", "ch4").clock == "slow"


def test_width_beats_rate_the_way_a_canister_does() -> None:
    """The measured archive case that made 'slower' the wrong word.

    An iWAS canister fills for ~15 s every ~530 s. Against a 60 s stationary
    mean it is about nine times slower by rate and four times narrower by
    support. Pairing on the canister's clock would evaluate a 60 s mean over
    15 s, which the interval model forbids.
    """
    canister = make_stream(0.0, 15.0, 1, {"benzene": np.array([1.0])})
    # one 15 s cell, then nothing for the rest of the hour
    means = make_stream(0.0, 60.0, 60, {"ch4": np.linspace(2000.0, 2100.0, 60)})
    paired = pair_species({"iwas": canister, "picarro": means}, "benzene", "ch4")
    assert paired.clock == "picarro"
    assert paired.n_pairs == 1
    assert paired.dataset["coverage_benzene"].values[0] == pytest.approx(0.25)


def half_phase_pair(
    *, n: int = 40, sparse_every: int = 2, dense_offset_s: float = 0.5
) -> dict[str, xr.Dataset]:
    """Two 1 s instruments half a cell apart, the 2024-07-18 drive in miniature.

    ``lif`` fills every row on cells centred on the second. ``picarro`` has a
    row every second on cells starting on the second, but a *value* in only
    every ``sparse_every``-th row -- the others are NaN, as in a merged file.
    Named so that alphabetical order alone would choose the wrong clock.
    """
    co2 = np.full(n, np.nan)
    co2[::sparse_every] = 420.0 + np.arange(n)[::sparse_every]
    picarro = make_stream(0.0, 1.0, n, {"co2": co2})
    lif = make_stream(dense_offset_s, 1.0, n, {"noy": 2.0 + np.arange(n) * 0.1})
    return {"picarro": picarro, "lif": lif}


def test_a_tie_in_width_goes_to_the_sparser_member_in_either_order() -> None:
    """METHODS §11.4.1: on the denser clock every sparse reading became two pairs.

    Both instruments have 1 s cells, so width cannot decide. The sparse one
    must be the clock whichever species is named first, so that each of its
    readings is one pair; on the other clock every reading straddles two
    cells and is counted twice.
    """
    streams = half_phase_pair()
    for y, x in (("noy", "co2"), ("co2", "noy")):
        paired = pair_species(streams, y, x)
        assert paired.clock == "picarro"
        assert paired.n_pairs == 20
        assert paired.dataset["co2"].attrs["tsara_readings"] == 20
        reason = paired.dataset.attrs["tsara_pairing_clock_reason"]
        assert "fewer measured values where the records overlap (20 vs 40)" in reason


def test_sparseness_is_counted_where_the_two_records_overlap() -> None:
    """A long sparse record beside a short dense one is still the sparse one.

    Over their whole records the sparse analyzer has more readings (100 of 200
    rows) than the dense instrument (30), and counting that way would put the
    dense one on the clock -- where, inside the thirty seconds they share,
    every sparse reading straddles two of its cells.
    """
    co2 = np.full(200, np.nan)
    co2[::2] = 420.0
    streams = {
        "picarro": make_stream(0.0, 1.0, 200, {"co2": co2}),
        "lif": make_stream(50.5, 1.0, 30, {"noy": np.linspace(2.0, 5.0, 30)}),
    }
    paired = pair_species(streams, "noy", "co2")
    assert paired.clock == "picarro"
    assert paired.dataset["co2"].attrs["tsara_readings"] == paired.n_pairs


def test_a_full_tie_goes_to_the_first_instrument_by_name_in_either_order() -> None:
    """Equal widths and equal counts leave nothing physical to choose by.

    The choice is arbitrary, and it is recorded as arbitrary; what it must not
    be is dependent on argument order.
    """
    streams = half_phase_pair(sparse_every=1)
    clocks = {pair_species(streams, y, x).clock for y, x in (("noy", "co2"), ("co2", "noy"))}
    assert clocks == {"lif"}
    reason = pair_species(streams, "co2", "noy").dataset.attrs["tsara_pairing_clock_reason"]
    assert "first instrument by name" in reason


# ---------------------------------------------------------------------------
# How many readings stand behind the pairs
# ---------------------------------------------------------------------------


def test_each_species_records_the_readings_behind_its_pairs(
    two_rates: dict[str, xr.Dataset],
) -> None:
    """Ten 1 s samples onto two 4 s cells: eight samples are used, two cells.

    Samples 8 and 9 overlap no surviving cell, and sample 8's cell merely
    touches the second one at 8 s, which is an edge rather than an overlap.
    """
    paired = pair_species(two_rates, "ch4", "co2")
    assert (paired.y_readings, paired.x_readings) == (8, 2)
    assert paired.dataset["ch4"].attrs["tsara_readings"] == 8
    assert paired.dataset["co2"].attrs["tsara_readings"] == 2


def test_a_paired_product_may_not_be_paired_again(two_rates: dict[str, xr.Dataset]) -> None:
    """Pairing is the binner with a clock chosen, so it inherits the refusal.

    The pairs of a first call are rows, not readings: each already averages
    whatever fell in its cell, and pairing them against a third species would
    weight those rows as measurements (METHODS §11.2.3).
    """
    paired = pair_species(two_rates, "ch4", "co2")
    assert paired.dataset.attrs["tsara_stage"] == "paired"
    # Named by instrument, because the product carries both species and a bare
    # name would be refused as ambiguous before the stage was ever looked at.
    with pytest.raises(TsaraAlignError) as refused:
        pair_species(
            {"pairs": paired.dataset, "fast": two_rates["fast"]},
            ("pairs", "co2"),
            ("fast", "ch4"),
        )
    assert "paired" in str(refused.value)
    assert "native streams" in str(refused.value)


def test_a_masked_sample_is_not_a_reading() -> None:
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.array([1.0, np.nan, 3.0, 4.0])})
    slow = make_stream(0.0, 4.0, 1, {"co2": np.array([10.0])})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert paired.y_readings == 3


def test_a_cell_bracketed_but_not_overlapped_is_not_a_reading() -> None:
    """Nested readings make the overlap search return a zero-weight pair.

    A 10 s cell holding a 1 s cell (a file whose per-row bounds nest one
    sample inside another, which `stream_cells` deliberately accepts) raises
    the running maximum of cell stops, so the candidate window for the 6-12 s
    target still brackets the 1-2 s cell. It overlaps by nothing, and the
    binner gives it no weight; counting it as a reading would describe a pair
    the value was not formed from. The 0-6 s target is masked so that the
    short cell has no legitimate pair to be counted through instead.
    """
    start = np.array([0, 1], dtype=np.int64) * SECOND
    stop = np.array([10, 2], dtype=np.int64) * SECOND
    nested = xr.Dataset(
        {"benzene": ("time", np.array([1.0, 5.0]), {"units": "ppb"})},
        coords={
            "time": (start + (stop - start) // 2).astype("datetime64[ns]"),
            "time_bnds": (("time", "nv"), np.stack([start, stop], axis=1).astype("datetime64[ns]")),
        },
    )
    nested["time"].attrs["bounds"] = "time_bnds"
    wide = make_stream(0.0, 6.0, 2, {"ch4": np.array([np.nan, 2000.0])})
    paired = pair_species({"iwas": nested, "picarro": wide}, "benzene", "ch4")
    assert paired.clock == "picarro"
    assert paired.n_pairs == 1
    assert paired.y_readings == 1


def test_species_sharing_an_instrument_have_one_reading_per_pair() -> None:
    stream = make_stream(0.0, 1.0, 6, {"ch4": np.arange(6.0), "c2h6": np.arange(6.0)})
    paired = pair_species({"em27": stream}, "ch4", "c2h6")
    assert paired.y_readings == paired.x_readings == paired.n_pairs == 6


def test_a_fill_straddling_two_cells_is_one_reading_and_is_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The duplication the tie-break cannot remove, made visible instead.

    Three 60 s cells. One 15 s fill sits wholly inside the first; a second
    runs from 112.5 s to 127.5 s, across the boundary at 120 s, so it is the
    only partner of both the second and third cells. Three pairs, two
    readings -- and a fit counting three independent points would count that
    second fill's error twice.
    """
    minute = make_stream(0.0, 60.0, 3, {"ch4": np.array([2000.0, 2010.0, 2020.0])})
    fills = xr.Dataset(
        {"benzene": ("time", np.array([1.0, 2.0]), {"units": "ppb"})},
        coords={
            "time": np.array([27_500, 120_000], dtype=np.int64)
            .astype("datetime64[ms]")
            .astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                (np.array([[20.0, 35.0], [112.5, 127.5]]) * SECOND)
                .astype(np.int64)
                .astype("datetime64[ns]"),
            ),
        },
    )
    fills["time"].attrs["bounds"] = "time_bnds"
    with caplog.at_level("WARNING", logger="tsara.align"):
        paired = pair_species({"iwas": fills, "picarro": minute}, "benzene", "ch4")
    assert paired.clock == "picarro"
    assert paired.n_pairs == 3
    assert (paired.y_readings, paired.x_readings) == (2, 3)
    # Said once, by the binner over the candidate cells, and not repeated by
    # pairing over the surviving ones (METHODS §11.2.4).
    assert "'benzene' (3 rows from 2 readings" in caplog.text
    assert "rest on only" not in caplog.text


def test_pairing_speaks_where_dropping_rows_first_makes_readings_shared(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rows outnumber readings only after the drop, so the binner had nothing to say.

    Three minutes. One fill straddles the first boundary and sits in rows one
    and two; two more fills sit together in row three, where the analyzer has
    no value, so row three is dropped. Over the candidate cells three rows
    rest on three readings; over the surviving pairs, two rows rest on one.
    """
    minute = make_stream(0.0, 60.0, 3, {"ch4": np.array([2000.0, 2010.0, np.nan])})
    starts = (np.array([50.0, 130.0, 150.0]) * SECOND).astype(np.int64)
    stops = (np.array([70.0, 140.0, 160.0]) * SECOND).astype(np.int64)
    fills = xr.Dataset(
        {"benzene": ("time", np.array([1.0, 2.0, 3.0]), {"units": "ppb"})},
        coords={
            "time": ((starts + stops) // 2).astype("datetime64[ns]"),
            "time_bnds": (("time", "nv"), np.stack([starts, stops], 1).astype("datetime64[ns]")),
        },
    )
    fills["time"].attrs["bounds"] = "time_bnds"
    with caplog.at_level("WARNING", logger="tsara.align"):
        paired = pair_species({"iwas": fills, "picarro": minute}, "benzene", "ch4")
    assert paired.n_pairs == 2
    assert paired.y_readings == 1
    assert "rest on air" not in caplog.text
    assert "rest on only 1 distinct readings of benzene" in caplog.text


def test_no_warning_when_every_reading_is_one_pair(
    caplog: pytest.LogCaptureFixture, two_rates: dict[str, xr.Dataset]
) -> None:
    with caplog.at_level("WARNING", logger="tsara.align.pairing"):
        pair_species(two_rates, "ch4", "co2")
    assert "distinct readings" not in caplog.text


# ---------------------------------------------------------------------------
# A case small enough to check on paper
# ---------------------------------------------------------------------------


def test_worked_example_checkable_by_hand(two_rates: dict[str, xr.Dataset]) -> None:
    """Ten 1 s samples 0..9 onto two 4 s cells.

    The first cell spans 0-4 s and holds samples 0, 1, 2, 3, each for one
    second, so its mean is 1.5. The second spans 4-8 s and holds 4, 5, 6, 7,
    so its mean is 5.5. Samples 8 and 9 fall outside both cells and are
    dropped -- the slow stream simply stops there.
    """
    paired = pair_species(two_rates, "ch4", "co2")
    assert paired.dataset["ch4"].values == pytest.approx([1.5, 5.5])
    assert paired.dataset["co2"].values == pytest.approx([100.0, 200.0])
    assert paired.dataset["n_readings_ch4"].values.tolist() == [4, 4]
    assert paired.dataset["coverage_ch4"].values == pytest.approx([1.0, 1.0])


# ---------------------------------------------------------------------------
# Independent reimplementation
# ---------------------------------------------------------------------------


def test_pairing_matches_a_slow_reimplementation() -> None:
    """Random values, offset cells and a masked sample, against a loop."""
    rng = np.random.default_rng(20260914)
    values = rng.normal(1900.0, 40.0, size=300)
    values[::37] = np.nan
    fast = make_stream(0.0, 1.0, 300, {"ch4": values})
    slow = make_stream(2.5, 7.0, 40, {"co2": rng.normal(420.0, 5.0, size=40)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    expected = slow_pair(fast, "ch4", slow, "co2")
    surviving = [e for e in expected if np.isfinite(e)]
    assert paired.dataset["ch4"].values == pytest.approx(surviving, rel=1e-12)


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------


def test_the_mean_of_a_ramp_is_its_midpoint_value() -> None:
    """A closed form owing nothing to this module.

    The average of a linear function over an interval is its value at the
    interval's midpoint, exactly. So binning a linear ramp onto wider cells
    must reproduce the ramp evaluated at those cells' midpoints.
    """
    fast = make_stream(0.0, 1.0, 120, {"ramp": np.arange(120.0) + 0.5})
    slow = make_stream(0.0, 10.0, 12, {"anchor": np.ones(12)})
    paired = pair_species({"fast": fast, "slow": slow}, "ramp", "anchor")
    midpoints = np.arange(12) * 10.0 + 5.0
    assert paired.dataset["ramp"].values == pytest.approx(midpoints)


def test_a_species_paired_against_a_multiple_of_itself_returns_that_multiple() -> None:
    """The seed of the acceptance test: a known ratio must come back.

    Two species with identical time behaviour and a fixed ratio, measured on
    different clocks. Binning is linear, so the ratio must survive it exactly
    -- if it does not, the two members were not averaged over the same air.
    """
    rng = np.random.default_rng(20260915)
    signal = rng.normal(2000.0, 50.0, size=600)
    fast = make_stream(0.0, 1.0, 600, {"ch4": signal})
    slow = make_stream(0.0, 60.0, 10, {"tracer": np.zeros(10)})
    # Give the slow stream the same signal, averaged over its own cells.
    slow["tracer"].values = signal.reshape(10, 60).mean(axis=1) * 3.0
    paired = pair_species({"fast": fast, "slow": slow}, "tracer", "ch4")
    ratio = paired.dataset["tracer"].values / paired.dataset["ch4"].values
    assert ratio == pytest.approx(np.full(10, 3.0), rel=1e-12)


# ---------------------------------------------------------------------------
# Invariants that need no ground truth
# ---------------------------------------------------------------------------


def test_same_instrument_species_are_not_binned_at_all() -> None:
    """Several gases retrieved from one spectrum is the commonest case.

    They already share a clock, so the values must pass through untouched --
    not merely to within rounding. Averaging a cell onto itself is the
    identity mathematically and not in floating point, which is why this path
    exists rather than falling through the general one.
    """
    values = {"ch4": np.linspace(1900.0, 2100.0, 50), "c2h6": np.linspace(1.0, 5.0, 50)}
    stream = make_stream(0.0, 1.0, 50, values)
    paired = pair_species({"em27": stream}, "ch4", "c2h6")
    assert paired.clock == "em27"
    assert np.array_equal(paired.dataset["ch4"].values, values["ch4"])
    assert np.array_equal(paired.dataset["c2h6"].values, values["c2h6"])
    assert paired.dataset["ch4"].attrs["tsara_binned"] == 0
    assert paired.dataset["coverage_ch4"].values == pytest.approx(np.ones(50))


def test_binning_a_stream_onto_matching_cells_returns_it() -> None:
    """Two instruments with identical cells: neither is really binned.

    The identity has to survive going through the general path, since the
    fast path only triggers for one instrument.
    """
    values = np.linspace(10.0, 20.0, 30)
    a = make_stream(0.0, 1.0, 30, {"ch4": values})
    b = make_stream(0.0, 1.0, 30, {"co2": np.arange(30.0)})
    paired = pair_species({"a": a, "b": b}, "ch4", "co2")
    assert paired.dataset["ch4"].values == pytest.approx(values, rel=1e-12)
    assert paired.dataset["n_readings_ch4"].values.tolist() == [1] * 30


def test_a_pair_is_never_fabricated_across_a_gap() -> None:
    """A slow cell with no fast data yields no pair, rather than an interpolation."""
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    slow = make_stream(0.0, 4.0, 5, {"co2": np.arange(5.0)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert paired.n_pairs == 1
    assert paired.dataset.attrs["tsara_pairing_cells_considered"] == 5
    assert paired.dataset.attrs["tsara_pairing_cells_dropped"] == 4


def test_a_masked_sample_reduces_coverage_rather_than_the_value() -> None:
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.array([1.0, np.nan, 3.0, 4.0])})
    slow = make_stream(0.0, 4.0, 1, {"co2": np.array([10.0])})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert paired.dataset["ch4"].values[0] == pytest.approx((1.0 + 3.0 + 4.0) / 3)
    assert paired.dataset["n_readings_ch4"].values[0] == 3
    assert paired.dataset["coverage_ch4"].values[0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def test_propagated_uncertainty_matches_the_scalar_form() -> None:
    """The vectorized path against the per-cell one, which Monte Carlo checked.

    Stage 1 verified the scalar propagation against an experiment; this binds
    the binned form to it, so the chain from measured scatter to a paired
    sigma is unbroken.
    """
    sigma = np.full(12, 2.0)
    fast = make_stream(
        0.0,
        1.0,
        12,
        {"ch4": np.arange(12.0), sigma_rand_name("ch4"): sigma, sigma_sys_name("ch4"): sigma / 4},
    )
    slow = make_stream(0.0, 4.0, 3, {"co2": np.arange(3.0)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    expected_random = [propagate_random(sigma[i : i + 4], np.ones(4)).sigma for i in (0, 4, 8)]
    expected_systematic = [
        propagate_systematic(sigma[i : i + 4] / 4, np.ones(4)).sigma for i in (0, 4, 8)
    ]
    assert paired.dataset[sigma_rand_name("ch4")].values == pytest.approx(expected_random)
    assert paired.dataset[sigma_sys_name("ch4")].values == pytest.approx(expected_systematic)


def test_the_two_components_still_move_in_opposite_directions() -> None:
    """Averaging four samples shrinks the random component and not the other."""
    sigma = np.full(12, 2.0)
    fast = make_stream(
        0.0,
        1.0,
        12,
        {"ch4": np.arange(12.0), sigma_rand_name("ch4"): sigma, sigma_sys_name("ch4"): sigma},
    )
    slow = make_stream(0.0, 4.0, 3, {"co2": np.arange(3.0)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert paired.dataset[sigma_rand_name("ch4")].values == pytest.approx(np.full(3, 1.0))
    assert paired.dataset[sigma_sys_name("ch4")].values == pytest.approx(np.full(3, 2.0))


def test_a_declared_sigma_is_moved_onto_its_cells_at_the_point_of_use() -> None:
    """The arithmetic Phase 3.5 removed from ingestion, done here instead.

    A figure quoted at 1 s on a stream delivering 4 s cells describes a
    different interval from the cells it sits on. With a decorrelation
    timescale declared, it is moved; the sigma that reaches the pair is
    smaller than the declared one and larger than the naive root-N.
    """
    sigma = np.full(12, 2.0)
    attrs: dict[str, dict[str, object]] = {
        "ch4": {
            "units": "ppb",
            "uncertainty_at_width": "1s",
            "decorrelation_timescale": "2s",
            "uncertainty_provenance_random": "declared",
        },
        sigma_rand_name("ch4"): {"units": "ppb"},
    }
    fast = make_stream(
        0.0, 4.0, 12, {"ch4": np.arange(12.0), sigma_rand_name("ch4"): sigma}, attrs=attrs
    )
    slow = make_stream(0.0, 8.0, 6, {"co2": np.arange(6.0)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    companion = paired.dataset[sigma_rand_name("ch4")]
    assert companion.attrs["tsara_sigma_at_support"] == "ar1_neff"
    assert companion.attrs["uncertainty_provenance"] == "declared"


def test_a_declared_sigma_without_a_timescale_is_left_alone_and_says_so() -> None:
    """The refusal, carried through to the product.

    METHODS §10.8: with no timescale the naive root-N is not merely imprecise
    but confidently wrong, so the honest answer is the unscaled figure plus a
    label saying it was not scaled.
    """
    attrs: dict[str, dict[str, object]] = {"ch4": {"units": "ppb", "uncertainty_at_width": "1s"}}
    fast = make_stream(
        0.0, 4.0, 8, {"ch4": np.arange(8.0), sigma_rand_name("ch4"): np.full(8, 2.0)}, attrs=attrs
    )
    slow = make_stream(0.0, 8.0, 4, {"co2": np.arange(4.0)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert paired.dataset[sigma_rand_name("ch4")].attrs["tsara_sigma_at_support"] == "unscaled"


def test_a_species_with_no_declared_budget_gets_no_sigma_column() -> None:
    """Absence is not zero: an undeclared budget must not become a claim."""
    fast = make_stream(0.0, 1.0, 8, {"ch4": np.arange(8.0)})
    slow = make_stream(0.0, 4.0, 2, {"co2": np.arange(2.0)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert sigma_rand_name("ch4") not in paired.dataset.data_vars


# ---------------------------------------------------------------------------
# Coverage guard and interval restriction
# ---------------------------------------------------------------------------


def test_min_coverage_drops_thin_pairs_and_the_default_drops_nothing() -> None:
    fast = make_stream(0.0, 1.0, 5, {"ch4": np.arange(5.0)})
    slow = make_stream(0.0, 4.0, 2, {"co2": np.arange(2.0)})
    lenient = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert lenient.n_pairs == 2
    assert lenient.dataset["coverage_ch4"].values == pytest.approx([1.0, 0.25])
    strict = pair_species({"fast": fast, "slow": slow}, "ch4", "co2", min_coverage=0.5)
    assert strict.n_pairs == 1
    assert strict.dataset.attrs["tsara_pairing_min_coverage"] == 0.5


def test_an_interval_restricts_pairing_to_cells_that_overlap_it() -> None:
    """Overlap, not containment: an event boundary rarely lands on a cell edge.

    Requiring containment would silently shorten every event by up to one
    cell at each end.
    """
    fast = make_stream(0.0, 1.0, 40, {"ch4": np.arange(40.0)})
    slow = make_stream(0.0, 10.0, 4, {"co2": np.arange(4.0)})
    window = (pd.Timestamp("1970-01-01 00:00:09"), pd.Timestamp("1970-01-01 00:00:21"))
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2", interval=window)
    assert paired.n_pairs == 3


def test_an_interval_with_no_overlap_is_an_error_naming_it() -> None:
    fast = make_stream(0.0, 1.0, 10, {"ch4": np.arange(10.0)})
    slow = make_stream(0.0, 5.0, 2, {"co2": np.arange(2.0)})
    window = (pd.Timestamp("1999-01-01"), pd.Timestamp("1999-01-02"))
    with pytest.raises(TsaraAlignError, match="overlap the interval"):
        pair_species({"fast": fast, "slow": slow}, "ch4", "co2", interval=window)


def test_a_backwards_interval_is_rejected() -> None:
    fast = make_stream(0.0, 1.0, 10, {"ch4": np.arange(10.0)})
    slow = make_stream(0.0, 5.0, 2, {"co2": np.arange(2.0)})
    window = (pd.Timestamp("1970-01-01 00:00:05"), pd.Timestamp("1970-01-01 00:00:01"))
    with pytest.raises(TsaraAlignError, match="positive duration"):
        pair_species({"fast": fast, "slow": slow}, "ch4", "co2", interval=window)


# ---------------------------------------------------------------------------
# Resolving which species is meant
# ---------------------------------------------------------------------------


def test_a_species_measured_twice_must_be_named_by_instrument() -> None:
    """Two analyzers measuring one species is how a campaign compares them.

    Choosing the first silently would answer a different question from the
    one asked.
    """
    a = make_stream(0.0, 1.0, 10, {"ch4": np.arange(10.0)})
    b = make_stream(0.0, 2.0, 5, {"ch4": np.arange(5.0), "co2": np.arange(5.0)})
    with pytest.raises(TsaraAlignError, match="more than one instrument"):
        pair_species({"a": a, "b": b}, "ch4", "co2")
    paired = pair_species({"a": a, "b": b}, ("a", "ch4"), ("b", "co2"))
    assert paired.clock == "b"


def test_an_unknown_species_lists_what_is_available() -> None:
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    with pytest.raises(TsaraAlignError, match="No stream measures 'ozone'"):
        pair_species({"fast": fast}, "ozone", "ch4")


def test_an_unknown_instrument_is_named() -> None:
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    with pytest.raises(TsaraAlignError, match="No stream named 'nope'"):
        pair_species({"fast": fast}, ("nope", "ch4"), "ch4")


def test_an_unknown_variable_on_a_known_instrument_is_named() -> None:
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    with pytest.raises(TsaraAlignError, match="has no variable 'ozone'"):
        pair_species({"fast": fast}, ("fast", "ozone"), "ch4")


def test_pairing_a_species_with_itself_is_refused() -> None:
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    with pytest.raises(TsaraAlignError, match="with itself"):
        pair_species({"fast": fast}, "ch4", "ch4")


def test_no_streams_at_all() -> None:
    with pytest.raises(TsaraAlignError, match="No streams to pair"):
        pair_species({}, "ch4", "co2")


def test_a_stream_without_cells_is_refused_by_name() -> None:
    """A bundle written before cells existed must be reloaded, not guessed at."""
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    bare = xr.Dataset(
        {"co2": ("time", np.arange(4.0))},
        coords={"time": np.arange(4).astype("datetime64[s]").astype("datetime64[ns]")},
    )
    with pytest.raises(TsaraAlignError, match="carries no 'time_bnds'"):
        pair_species({"fast": fast, "bare": bare}, "ch4", "co2")


def test_records_that_never_meet_produce_a_message_naming_both_reasons() -> None:
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    slow = make_stream(1000.0, 4.0, 2, {"co2": np.arange(2.0)})
    with pytest.raises(TsaraAlignError, match="no usable pairs"):
        pair_species({"fast": fast, "slow": slow}, "ch4", "co2")


# ---------------------------------------------------------------------------
# What the product says about itself
# ---------------------------------------------------------------------------


def test_the_product_carries_cells_and_correct_cell_methods(
    two_rates: dict[str, xr.Dataset],
) -> None:
    """A paired value describes an interval, and says which operation made it.

    The binned species is a mean over the cell. The species already on the
    clock was not averaged by this stage at all, so it keeps its own stream's
    method -- stamping `time: mean` on a point sample would assert an
    averaging that never happened.
    """
    paired = pair_species(two_rates, "ch4", "co2")
    assert "time_bnds" in paired.dataset.coords
    assert paired.dataset["time"].attrs["bounds"] == "time_bnds"
    assert paired.dataset["ch4"].attrs["cell_methods"] == "time: mean"
    assert paired.dataset["co2"].attrs["cell_methods"] == "time: point"


def test_sigma_companions_carry_no_cell_method() -> None:
    """A sigma describes the uncertainty OF a cell's value, not a mean over it.

    Where that value is an average the two differ by exactly sqrt(N_eff), so
    `time: mean` on a sigma would be a false claim wrong by the one quantity
    the two-component design exists to track (METHODS §10.2).
    """
    fast = make_stream(
        0.0, 1.0, 8, {"ch4": np.arange(8.0), sigma_rand_name("ch4"): np.full(8, 1.0)}
    )
    slow = make_stream(0.0, 4.0, 2, {"co2": np.arange(2.0)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert "cell_methods" not in paired.dataset[sigma_rand_name("ch4")].attrs


def test_counts_are_sums_and_coverage_is_neither(two_rates: dict[str, xr.Dataset]) -> None:
    """A count is a sum over the cell; a coverage fraction is not a statistic
    of the data inside the cell at all, so it gets no cell method."""
    paired = pair_species(two_rates, "ch4", "co2")
    assert paired.dataset["n_readings_ch4"].attrs["cell_methods"] == "time: sum"
    assert "cell_methods" not in paired.dataset["coverage_ch4"].attrs


def test_the_product_records_how_it_was_made(two_rates: dict[str, xr.Dataset]) -> None:
    paired = pair_species(two_rates, "ch4", "co2")
    attrs = paired.dataset.attrs
    assert attrs["tsara_stage"] == "paired"
    assert attrs["tsara_pairing_clock"] == "slow"
    # Inherited from the binner: the form is recorded per propagated sigma, and
    # nowhere on the product itself (METHODS §11.2).
    assert "tsara_propagation_form" not in attrs
    assert paired.dataset["ch4"].attrs["tsara_instrument"] == "fast"
    assert paired.dataset["co2"].attrs["tsara_binned"] == 0


def test_the_paired_product_round_trips_through_netcdf(
    tmp_path: Path, two_rates: dict[str, xr.Dataset]
) -> None:
    """It is an ordinary self-describing Dataset, so saving it needs no bundle.

    The phase's bundle product is the output grid; a paired series is a
    per-call intermediate. It still has to survive being written, because a
    user inspecting one in a notebook will write it.
    """
    paired = pair_species(two_rates, "ch4", "co2")
    path = tmp_path / "paired.nc"
    with warnings.catch_warnings():
        # Any CF warning here is a defect, not noise: without the pinned time
        # encoding xarray chooses different reference epochs for `time` and
        # `time_bnds` and says so. This test caught exactly that.
        warnings.simplefilter("error", UserWarning)
        paired.dataset.to_netcdf(path)
    with xr.open_dataset(path, decode_coords="all") as reloaded:
        assert reloaded["ch4"].values == pytest.approx(paired.dataset["ch4"].values)
        assert reloaded.attrs["tsara_pairing_clock"] == "slow"
        assert "time_bnds" in reloaded.coords


def test_len_is_the_pair_count(two_rates: dict[str, xr.Dataset]) -> None:
    paired = pair_species(two_rates, "ch4", "co2")
    assert isinstance(paired, PairedSpecies)
    assert len(paired) == paired.n_pairs == 2


# ---------------------------------------------------------------------------
# Edges that only a deliberate case reaches
# ---------------------------------------------------------------------------


def test_an_empty_stream_is_refused_by_name() -> None:
    """A stream with bounds but no rows has no interval to pair over."""
    fast = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    empty = xr.Dataset(
        {"co2": ("time", np.empty(0))},
        coords={
            "time": np.empty(0, dtype="datetime64[ns]"),
            "time_bnds": (("time", "nv"), np.empty((0, 2), dtype="datetime64[ns]")),
        },
    )
    empty["time"].attrs["bounds"] = "time_bnds"
    with pytest.raises(TsaraAlignError, match="no cells to bin from"):
        pair_species({"fast": fast, "empty": empty}, "ch4", "co2")


def test_a_clock_species_with_no_declared_cell_method_gets_none() -> None:
    """Absence carries through rather than becoming an assertion.

    A stream that never said how its values relate to their cells must not
    acquire a claim by passing through pairing.
    """
    fast = make_stream(0.0, 1.0, 8, {"ch4": np.arange(8.0)})
    slow = make_stream(0.0, 4.0, 2, {"co2": np.arange(2.0)})
    del slow["co2"].attrs["cell_methods"]
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    assert "cell_methods" not in paired.dataset["co2"].attrs
    assert paired.dataset["ch4"].attrs["cell_methods"] == "time: mean"


def test_the_clock_species_carries_its_own_sigma_through_unbinned() -> None:
    """The species on the pairing clock was not averaged, so its sigma is its own."""
    sigma = np.linspace(1.0, 2.0, 3)
    slow = make_stream(0.0, 4.0, 3, {"co2": np.arange(3.0), sigma_rand_name("co2"): sigma})
    fast = make_stream(0.0, 1.0, 12, {"ch4": np.arange(12.0)})
    paired = pair_species({"fast": fast, "slow": slow}, "ch4", "co2")
    companion = paired.dataset[sigma_rand_name("co2")]
    assert companion.values == pytest.approx(sigma)
    assert companion.attrs["tsara_propagation_form"] == "native"


def test_a_single_cell_stream_has_no_gap_to_measure() -> None:
    """One cell has no spacing between samples, so the width stands in for it.

    Reached by a canister-like stream: one fill, paired onto a wider mean,
    with a declared timescale that would otherwise need a cadence.
    """
    attrs: dict[str, dict[str, object]] = {
        "benzene": {"units": "ppb", "decorrelation_timescale": "5s"}
    }
    canister = make_stream(
        0.0,
        15.0,
        1,
        {"benzene": np.array([2.0]), sigma_rand_name("benzene"): np.array([0.5])},
        attrs=attrs,
    )
    means = make_stream(0.0, 60.0, 1, {"ch4": np.array([2000.0])})
    paired = pair_species({"iwas": canister, "picarro": means}, "benzene", "ch4")
    assert paired.n_pairs == 1
    # One contributing sample cannot average down, whatever the timescale.
    assert paired.dataset[sigma_rand_name("benzene")].values[0] == pytest.approx(0.5)


def test_cells_that_all_start_together_have_no_cadence_to_measure() -> None:
    """Degenerate but expressible: several cells sharing one start time.

    A file declaring per-row bounds can nest one sample inside another, and
    then the gap between consecutive starts is zero. Falling back to the cell
    width keeps the correlation correction defined instead of raising from
    inside the propagation module about a spacing the caller never chose.
    """
    start = np.zeros(2, dtype=np.int64)
    stop = np.array([4 * SECOND, 8 * SECOND], dtype=np.int64)
    nested = xr.Dataset(
        {
            "benzene": (
                "time",
                np.array([1.0, 2.0]),
                {"units": "ppb", "decorrelation_timescale": "3s"},
            ),
            sigma_rand_name("benzene"): ("time", np.array([0.4, 0.4]), {"units": "ppb"}),
        },
        coords={
            "time": (start + (stop - start) // 2).astype("datetime64[ns]"),
            "time_bnds": (("time", "nv"), np.stack([start, stop], axis=1).astype("datetime64[ns]")),
        },
    )
    nested["time"].attrs["bounds"] = "time_bnds"
    means = make_stream(0.0, 60.0, 1, {"ch4": np.array([2000.0])})
    paired = pair_species({"iwas": nested, "picarro": means}, "benzene", "ch4")
    assert paired.n_pairs == 1
    assert np.isfinite(paired.dataset[sigma_rand_name("benzene")].values[0])


# ---------------------------------------------------------------------------
# A clock whose cells vary in width (METHODS §11.2.4)
# ---------------------------------------------------------------------------


def test_a_clock_cell_narrower_than_half_a_partner_reading_refuses_the_pair() -> None:
    """Canister fills of 14, 15 and 1.8 s against a 10 s analyzer.

    The canister is the wider-supported member by median, so it is the clock;
    but a 10 s reading on its 1.8 s fill is five times as wide as the cell it
    fills, which the per-pair rule refuses and the summed rule it replaced
    let through. Allowed by name, the partner column reads 'copied'.
    """
    analyzer = make_stream(0.0, 10.0, 60, {"ch4": np.arange(60.0)})
    starts = (np.array([20.0, 200.0, 400.0]) * SECOND).astype(np.int64)
    stops = (np.array([34.0, 215.0, 401.8]) * SECOND).astype(np.int64)
    fills = xr.Dataset(
        {"benzene": ("time", np.array([1.0, 2.0, 3.0]), {"units": "ppb"})},
        coords={
            "time": ((starts + stops) // 2).astype("datetime64[ns]"),
            "time_bnds": (("time", "nv"), np.stack([starts, stops], 1).astype("datetime64[ns]")),
        },
    )
    fills["time"].attrs["bounds"] = "time_bnds"
    streams = {"iwas": fills, "lgr": analyzer}
    with pytest.raises(TsaraAlignError, match="times as wide as a cell it fills"):
        pair_species(streams, "benzene", "ch4")
    paired = pair_species(streams, "benzene", "ch4", finer_support="allow")
    assert paired.clock == "iwas"
    assert paired.n_pairs == 3
    assert paired.dataset["ch4"].attrs["tsara_support_transform"] == "copied"
    assert paired.dataset["ch4"].attrs["tsara_width_ratio_max"] == pytest.approx(10 / 1.8)
