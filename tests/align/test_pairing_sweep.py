"""Tests for pairing swept members, and variables carried beside a pair.

The ratio stage pairs enhancements, which carry the baseline's sweep: one
version per (window, quantile) point (METHODS §6.2, §5.2). Pairing a swept
member must give, at every sweep point, exactly what pairing that sweep
point alone gives through the unswept path, which earlier phases verified
against a slow reimplementation, closed forms and Monte Carlo. So the
evidence here is, in order of strength:

* a two-version fixture whose pairs and counts can be worked out on paper;
* the swept product against the unswept path, sweep point by sweep point, on
  real baseline states of two instruments of different widths and of one
  instrument, on the clock rule's cells and on a uniform period: bit for bit
  on every value and companion, and the per-point pair and reading counts;
* the refusals and the carried variables' promises.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

from tsara.align import TsaraAlignError, bin_streams_onto_cells, pair_species
from tsara.align.pairing import PAIRS_PER_SWEEP_POINT_ATTR, SHARED_READINGS_ATTR
from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig
from tsara.core.naming import READINGS_ATTR, coverage_name, sigma_rand_name
from tsara.core.support import CellBounds, stream_cells
from tsara.core.timebase import SECOND_NS as SECOND

SWEEP = ("baseline_window", "baseline_quantile")


def bounds(start_s: float, width_s: float, n: int) -> CellBounds:
    start = (np.arange(n, dtype=np.int64) * int(width_s * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def stream_on(
    cells: CellBounds,
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]],
    coords: dict[str, np.ndarray] | None = None,
) -> xr.Dataset:
    """Build a stream on ``cells``, each variable with a 0.5 declared random sigma."""
    data: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
    for name, (dims, values) in variables.items():
        data[name] = (dims, values, {"units": "ppb", "role": "gas", "cell_methods": "time: mean"})
        data[sigma_rand_name(name)] = (
            ("time",),
            np.full(len(cells), 0.5),
            {
                "units": "ppb",
                "uncertainty_component": "random",
                "uncertainty_provenance": "declared",
            },
        )
    dataset = xr.Dataset(
        data_vars=data,
        coords={
            "time": cells.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([cells.start_ns, cells.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
            **(coords or {}),
        },
    )
    dataset["time"].attrs["bounds"] = "time_bnds"
    return dataset


# ---------------------------------------------------------------------------
# On paper
# ---------------------------------------------------------------------------


@pytest.fixture()
def by_hand() -> dict[str, xr.Dataset]:
    """Two versions ('k' = 0, 1). y on four 1 s cells, x on two 2 s cells over them.

    x is the wider, so the pairs are on x's two cells. Version 0 is complete;
    version 1 lacks y's second reading and x's second.
    """
    y = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, 3.0], [4.0, 4.0]])
    x = np.array([[10.0, 10.0], [20.0, np.nan]])
    k = {"k": np.array([0, 1])}
    return {
        "a": stream_on(bounds(0, 1, 4), {"y": (("time", "k"), y)}, k),
        "b": stream_on(bounds(0, 2, 2), {"x": (("time", "k"), x)}, k),
    }


def test_each_sweep_point_is_paired_on_its_own_and_a_row_kept_if_any_is(
    by_hand: dict[str, xr.Dataset],
) -> None:
    """Version 0: (1.5, 10) and (3.5, 20). Version 1: (1.0, 10) only; its second row,
    a pair at version 0, holds neither member at version 1."""
    paired = pair_species(by_hand, ("a", "y"), ("b", "x"))
    data = paired.dataset
    assert paired.clock == "b"
    assert paired.n_pairs == 2
    np.testing.assert_array_equal(data["y"].values, [[1.5, 1.0], [3.5, np.nan]])
    np.testing.assert_array_equal(data["x"].values, [[10.0, 10.0], [20.0, np.nan]])
    assert data.attrs[PAIRS_PER_SWEEP_POINT_ATTR] == [2, 1]
    # Version 0 draws on all four readings of y; version 1 on one (its first
    # cell's other reading is missing, its second cell is no pair there).
    assert data["y"].attrs[READINGS_ATTR] == 4
    assert data["x"].attrs[READINGS_ATTR] == 2
    # A blanked value keeps the attributes a value carries.
    assert data["y"].attrs["units"] == "ppb"


def test_min_coverage_is_asked_at_each_sweep_point(by_hand: dict[str, xr.Dataset]) -> None:
    """At version 1, y covers the first cell 0.5, below 0.6; that row stays, a pair
    at version 0 only."""
    paired = pair_species(by_hand, ("a", "y"), ("b", "x"), min_coverage=0.6)
    data = paired.dataset
    assert data.attrs[PAIRS_PER_SWEEP_POINT_ATTR] == [2, 0]
    # Both members are blanked where the row is no pair, x's 10 at version 1 too.
    np.testing.assert_array_equal(data["y"].values, [[1.5, np.nan], [3.5, np.nan]])
    np.testing.assert_array_equal(data["x"].values, [[10.0, np.nan], [20.0, np.nan]])
    # Either member's coverage counts: named the other way round, the same.
    swapped = pair_species(by_hand, ("b", "x"), ("a", "y"), min_coverage=0.6)
    assert swapped.dataset.attrs[PAIRS_PER_SWEEP_POINT_ATTR] == [2, 0]
    np.testing.assert_array_equal(data[coverage_name("y")].values, [[1.0, 0.5], [1.0, 1.0]])


def test_a_tie_in_width_counts_a_swept_member_measured_where_any_point_is() -> None:
    """Equal 1 s cells half a second apart: the sparser member is the clock (§11.4.1).

    a holds four readings at version 0 and two at version 1; b holds three at
    both. Read as measured where any version is, a has four and b three, so b
    is the clock, as it would be for the readings themselves; read as measured
    where every version is, a would have two and take the clock.
    """
    k = {"k": np.array([0, 1])}
    a = np.array([[1.0, 1.0], [2.0, np.nan], [3.0, np.nan], [4.0, 4.0]])
    b = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0], [4.0, 4.0]])
    streams = {
        "a": stream_on(bounds(0, 1, 4), {"y": (("time", "k"), a)}, k),
        "b": stream_on(bounds(0.5, 1, 4), {"x": (("time", "k"), b)}, k),
    }
    assert pair_species(streams, ("a", "y"), ("b", "x")).clock == "b"
    assert pair_species(streams, ("b", "x"), ("a", "y")).clock == "b"


def test_pairing_speaks_at_the_first_sweep_point_where_readings_become_shared(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The unswept case of test_pairing.py, at two sweep points, the first named.

    Four minutes. Fill A straddles the first boundary and fill D the third;
    fills E and F sit in the fourth minute beside D's second half. Over the
    candidate cells each version's four rows rest on four readings, so the
    binner says nothing. At version 0 the analyzer has its first two minutes:
    two pairs rest on A alone. At version 1 it has three: three pairs on A
    and D.
    """
    k = {"k": np.array([0, 1])}
    ch4 = np.array([[2000.0, 2000.0], [2010.0, 2010.0], [np.nan, 2020.0], [np.nan, np.nan]])
    minute = stream_on(bounds(0, 60, 4), {"ch4": (("time", "k"), ch4)}, k)
    starts = (np.array([50.0, 170.0, 200.0, 220.0]) * SECOND).astype(np.int64)
    widths = (np.array([20.0, 20.0, 10.0, 10.0]) * SECOND).astype(np.int64)
    benzene = np.repeat(np.array([[1.0], [2.0], [3.0], [4.0]]), 2, axis=1)
    fills = stream_on(
        CellBounds(start_ns=starts, stop_ns=starts + widths),
        {"benzene": (("time", "k"), benzene)},
        k,
    )
    with caplog.at_level("WARNING", logger="tsara.align"):
        paired = pair_species(
            {"iwas": fills, "picarro": minute}, ("iwas", "benzene"), ("picarro", "ch4")
        )
    assert paired.dataset.attrs[PAIRS_PER_SWEEP_POINT_ATTR] == [2, 3]
    assert "rows from" not in caplog.text
    assert (
        "2 pairs of benzene vs ch4 rest on only 1 distinct readings of benzene "
        "at sweep point 0 (of 2)" in caplog.text
    )


def test_an_unswept_pair_records_no_sweep_counts() -> None:
    streams = {
        "a": stream_on(bounds(0, 1, 4), {"y": (("time",), np.arange(4.0))}),
        "b": stream_on(bounds(0, 2, 2), {"x": (("time",), np.array([1.0, 2.0]))}),
    }
    data = pair_species(streams, ("a", "y"), ("b", "x")).dataset
    assert PAIRS_PER_SWEEP_POINT_ATTR not in data.attrs


# ---------------------------------------------------------------------------
# Against the unswept path, sweep point by sweep point
# ---------------------------------------------------------------------------


def _record(n: int, step_s: float, rng: np.random.Generator) -> np.ndarray:
    """Methane over n readings step_s apart: 1900 ppb, three plumes, 0.7 ppb noise."""
    t = np.arange(n) * step_s
    plumes = sum(
        a * np.exp(-0.5 * ((t - c) / w) ** 2)
        for a, c, w in [(40, 600, 20), (15, 1500, 8), (80, 2100, 60)]
    )
    return 1900.0 + plumes + rng.normal(0.0, 0.7, n)


@pytest.fixture(scope="module")
def states() -> dict[str, xr.Dataset]:
    """Baseline states of a 1 s analyzer (methane and ethane) and a 2 s one, half a
    second out of phase, at 2 and 10 min and q 0.01 and 0.05.

    At 2 min and q 0.01 the 2 s analyzer's windows hold 60 readings, under the
    100 the count rule asks, so it is blank there and that sweep point has no
    pairs. Over one stretch the 2 s analyzer keeps only every eighth reading,
    so its 10 min windows there hold about 37, enough at q 0.05 (20) and not
    at 0.01, and its 2 min windows about 7, enough for neither; the sweep
    points therefore differ in which rows are pairs. A 30 s dropout in the
    1 s analyzer makes holes of its own.
    """
    rng = np.random.default_rng(7)
    config = BaselineConfig(windows=("2min", "10min"), quantiles=(0.01, 0.05))
    fast = _record(2400, 1.0, rng)
    fast[1000:1030] = np.nan
    ethane = 2.0 + 0.05 * (fast - 1900.0) + rng.normal(0.0, 0.05, fast.size)
    slow = _record(1200, 2.0, rng)
    sparse = np.arange(400, 800)
    slow[sparse[sparse % 8 != 0]] = np.nan
    aeris = stream_on(bounds(0, 1, 2400), {"ch4a": (("time",), fast), "c2h6": (("time",), ethane)})
    picarro = stream_on(bounds(0.5, 2, 1200), {"ch4": (("time",), slow)})
    return {
        "aeris": baseline_state(aeris, instrument="aeris", baseline=config),
        "picarro": baseline_state(picarro, instrument="picarro", baseline=config),
    }


def _at(state: xr.Dataset, index: tuple[int, ...]) -> xr.Dataset:
    return state.isel(dict(zip(SWEEP, index, strict=True)))


CASES = [
    pytest.param(
        ("aeris", "enhancement_c2h6"), ("picarro", "enhancement_ch4"), None, id="two-clocks"
    ),
    pytest.param(
        ("aeris", "enhancement_c2h6"), ("picarro", "enhancement_ch4"), "10s", id="a-period"
    ),
    pytest.param(
        ("aeris", "enhancement_c2h6"), ("aeris", "enhancement_ch4a"), None, id="one-instrument"
    ),
]


@pytest.mark.parametrize(("y", "x", "target"), CASES)
def test_every_sweep_point_is_what_pairing_it_alone_gives(
    states: dict[str, xr.Dataset],
    y: tuple[str, str],
    x: tuple[str, str],
    target: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bit for bit: values, counts, coverage, borrowed share and sigma at each point.

    The unswept path is pairing as Phase 4 built and verified it. The two can
    choose different clocks only on a tie in width, which none of these cases
    has (1 s against 2 s; one instrument; an explicit period).
    """
    caplog.set_level(logging.ERROR)
    swept = pair_species(states, y, x, target=target)
    data = swept.dataset
    sizes = tuple(int(states["aeris"].sizes[dim]) for dim in SWEEP)
    per_point = data.attrs[PAIRS_PER_SWEEP_POINT_ATTR]
    singles = []
    for column, index in enumerate(np.ndindex(*sizes)):
        alone = {name: _at(state, index) for name, state in states.items()}
        at = data.isel(dict(zip(SWEEP, index, strict=True)))
        pairs_here = np.isfinite(at[swept.y_name].values)
        # A row that is no pair here holds neither member here.
        assert np.array_equal(np.isfinite(at[swept.x_name].values), pairs_here)
        if per_point[column] == 0:
            # Nothing to pair at this point, alone or swept.
            with pytest.raises(TsaraAlignError, match="no usable pairs"):
                pair_species(alone, y, x, target=target)
            assert not pairs_here.any()
            continue
        single = pair_species(alone, y, x, target=target)
        singles.append(single)
        assert single.clock == swept.clock
        assert per_point[column] == single.n_pairs == int(pairs_here.sum())
        here = at.isel(time=np.flatnonzero(pairs_here))
        np.testing.assert_array_equal(here["time_bnds"].values, single.dataset["time_bnds"].values)
        for name in single.dataset.data_vars:
            np.testing.assert_array_equal(
                here[name].values, single.dataset[name].values, err_msg=f"{name} at {index}"
            )
    # Each member records the most any one point draws on, as the binner does.
    for name, attr in (
        (swept.y_name, READINGS_ATTR),
        (swept.y_name, SHARED_READINGS_ATTR),
        (swept.x_name, READINGS_ATTR),
        (swept.x_name, SHARED_READINGS_ATTR),
    ):
        assert data[name].attrs[attr] == max(s.dataset[name].attrs[attr] for s in singles)
    assert swept.y_readings == max(s.y_readings for s in singles)
    assert swept.x_readings == max(s.x_readings for s in singles)
    # A row is kept exactly when it is a pair at one sweep point or more.
    assert swept.n_pairs == int(
        np.isfinite(data[swept.y_name].values).reshape(swept.n_pairs, -1).any(axis=1).sum()
    )


def test_the_case_above_has_a_blank_sweep_point_and_points_that_differ(
    states: dict[str, xr.Dataset], caplog: pytest.LogCaptureFixture
) -> None:
    """Guard the fixture: without both, the test above would prove less than it says."""
    caplog.set_level(logging.ERROR)
    per_point = pair_species(
        states, ("aeris", "enhancement_c2h6"), ("picarro", "enhancement_ch4")
    ).dataset.attrs[PAIRS_PER_SWEEP_POINT_ATTR]
    assert per_point[0] == 0
    assert len(set(per_point[1:])) > 1


# ---------------------------------------------------------------------------
# Carried variables
# ---------------------------------------------------------------------------


def test_carried_readings_are_the_join_of_the_readings_on_the_surviving_cells(
    states: dict[str, xr.Dataset], caplog: pytest.LogCaptureFixture
) -> None:
    """What the ratio stage weighs a fit by (METHODS §4.3): the readings' own sigma."""
    caplog.set_level(logging.ERROR)
    paired = pair_species(
        states,
        ("aeris", "enhancement_c2h6"),
        ("picarro", "enhancement_ch4"),
        carry=[("aeris", "c2h6"), ("picarro", "ch4")],
    )
    assert paired.carried == ("c2h6", "ch4")
    data = paired.dataset
    # The same join over the clock's cells, then the surviving rows: a join's
    # last digit can depend on which other cells it fills, so the comparison is
    # with the join pairing ran, not with a join over the surviving cells alone.
    clock = stream_cells(states["picarro"], "picarro")
    direct = bin_streams_onto_cells(states, clock, [("aeris", "c2h6"), ("picarro", "ch4")])
    rows = np.searchsorted(
        clock.start_ns, data["time_bnds"].values[:, 0].astype("datetime64[ns]").astype(np.int64)
    )
    direct = direct.isel(time=rows)
    for name in ("c2h6", "ch4", sigma_rand_name("c2h6"), sigma_rand_name("ch4")):
        np.testing.assert_array_equal(data[name].values, direct[name].values)


def test_a_carried_variable_never_decides_which_rows_survive(
    states: dict[str, xr.Dataset], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR)
    blank = dict(states)
    blank["aeris"] = states["aeris"].assign(c2h6=states["aeris"]["c2h6"] * np.nan)
    y, x = ("aeris", "enhancement_c2h6"), ("picarro", "enhancement_ch4")
    without = pair_species(blank, y, x)
    with_blank = pair_species(blank, y, x, carry=[("aeris", "c2h6")])
    assert with_blank.n_pairs == without.n_pairs
    assert np.isnan(with_blank.dataset["c2h6"].values).all()


def test_a_carried_name_that_collides_is_suffixed_like_any_column(
    states: dict[str, xr.Dataset], caplog: pytest.LogCaptureFixture
) -> None:
    """Two instruments' 'sigma' companions aside, two carried 'ch4' would collide."""
    caplog.set_level(logging.ERROR)
    renamed = dict(states)
    renamed["aeris"] = states["aeris"].rename(
        {"ch4a": "ch4", sigma_rand_name("ch4a"): sigma_rand_name("ch4")}
    )
    paired = pair_species(
        renamed,
        ("aeris", "enhancement_c2h6"),
        ("picarro", "enhancement_ch4"),
        carry=[("aeris", "ch4"), ("picarro", "ch4")],
    )
    assert paired.carried == ("ch4_aeris", "ch4_picarro")
    assert {"ch4_aeris", "ch4_picarro"} <= set(paired.dataset.data_vars)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_members_with_different_sweeps_are_refused(states: dict[str, xr.Dataset]) -> None:
    with pytest.raises(TsaraAlignError, match="same ones"):
        pair_species(states, ("aeris", "enhancement_c2h6"), ("picarro", "ch4"))


def test_a_member_cannot_also_be_carried(states: dict[str, xr.Dataset]) -> None:
    y, x = ("aeris", "enhancement_c2h6"), ("picarro", "enhancement_ch4")
    with pytest.raises(TsaraAlignError, match="member of the pair"):
        pair_species(states, y, x, carry=[x])


def test_a_variable_carried_twice_is_refused(states: dict[str, xr.Dataset]) -> None:
    y, x = ("aeris", "enhancement_c2h6"), ("picarro", "enhancement_ch4")
    with pytest.raises(TsaraAlignError, match="twice"):
        pair_species(states, y, x, carry=[("aeris", "c2h6"), ("aeris", "c2h6")])
