"""Tests for the ratio stage's fits over the events' members (METHODS §5.2-5.4).

The lines themselves were checked in Phase 7a (tests/ratios/test_estimators.py).
What is new here is which points each fit gets, what it stores beside the
line, and the slope's sigma widened, so the evidence is, in order of strength:

* a dozen cells whose members, points and slopes can be worked out on paper;
* a reimplementation written from METHODS §5.2-5.4, one member at a time with
  a Python loop and the estimators called on that member alone, scored bit for
  bit against the vectorized stage on a whole generated chain (baseline,
  events, catalog, pairing), where midpoint membership is also held to the
  event index;
* Monte Carlo of the widening on readings whose errors persist (METHODS §4.5
  Table 7's case), drawn from the textbook AR(1) rather than the generator;
* the refusals.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.align import pair_species
from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.core.naming import borrowed_name, sigma_rand_name
from tsara.core.propagation import n_effective
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import event_catalog, event_states
from tsara.ratios import (
    FIT_COLUMNS,
    PairColumns,
    TsaraRatioError,
    member_fits,
    ols,
    york,
)

COLUMNS = PairColumns(
    numerator="enhancement_c2h6",
    reference="enhancement_ch4",
    numerator_reading="c2h6",
    reference_reading="ch4",
)


def make_pairs(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    sigma_x: float | np.ndarray | None = 0.7,
    sigma_y: float | np.ndarray | None = 0.05,
    windows: tuple[float, ...] = (600.0,),
    quantiles: tuple[float, ...] = (0.05,),
    tau: tuple[str | None, str | None] = (None, None),
    borrowed: tuple[float, float] = (0.0, 0.0),
    width_s: int = 1,
) -> xr.Dataset:
    """A paired product as pairing makes one: abutting cells of ``width_s`` (1 s
    by default), enhancements swept, readings carried.

    ``dx`` and ``dy`` are (cells,) arrays, the same at every sweep point, or
    (cells, windows, quantiles) arrays. ``tau`` is the (reference, numerator)
    decorrelation time declared on each enhancement, ``borrowed`` the
    (reference, numerator) borrowed share of every value.
    """
    n = dx.shape[0]
    shape = (n, len(windows), len(quantiles))
    dx3 = dx if dx.ndim == 3 else np.broadcast_to(dx[:, None, None], shape).copy()
    dy3 = dy if dy.ndim == 3 else np.broadcast_to(dy[:, None, None], shape).copy()
    start = np.arange(n, dtype=np.int64) * width_s * SECOND
    swept = ("time", "baseline_window", "baseline_quantile")
    data: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
    for name, values, declared, share in (
        (COLUMNS.reference, dx3, tau[0], borrowed[0]),
        (COLUMNS.numerator, dy3, tau[1], borrowed[1]),
    ):
        attrs: dict[str, object] = {"units": "ppb", "tsara_instrument": "van"}
        if declared is not None:
            attrs["decorrelation_timescale"] = declared
        data[name] = (swept, values, attrs)
        data[borrowed_name(name)] = (swept, np.where(np.isfinite(values), share, np.nan), {})
    for name, sigma in ((COLUMNS.reference_reading, sigma_x), (COLUMNS.numerator_reading, sigma_y)):
        data[name] = (("time",), np.zeros(n), {"units": "ppb", "tsara_instrument": "van"})
        if sigma is not None:
            data[sigma_rand_name(name)] = (
                ("time",),
                np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n,)).copy(),
                {"uncertainty_provenance": "declared"},
            )
    pairs = xr.Dataset(
        data,
        coords={
            "time": (start + width_s * SECOND // 2).astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([start, start + width_s * SECOND], 1).astype("datetime64[ns]"),
            ),
            "baseline_window": np.asarray(windows),
            "baseline_quantile": np.asarray(quantiles),
        },
    )
    pairs["time"].attrs["bounds"] = "time_bnds"
    return pairs


def event_row(
    event_id: str,
    candidate_id: str,
    span_s: tuple[float, float],
    *,
    window: float = 600.0,
    quantile: float = 0.05,
    entry: float = 3.0,
    exit_: float = 1.0,
    peak_z: float = 8.0,
    species: str = "ch4",
    trigger_event_id: str | None = None,
) -> dict[str, object]:
    """One catalog row with the columns the fits read, its interval in seconds."""
    start, end = (pd.Timestamp(0) + pd.Timedelta(seconds=edge) for edge in span_s)
    return {
        "event_id": event_id,
        "candidate_id": candidate_id,
        "instrument": "van",
        "species": species,
        "baseline_window": window,
        "baseline_quantile": quantile,
        "enter_multiple": entry,
        "exit_multiple": exit_,
        "start_time": start,
        "peak_time": start,
        "end_time": end,
        "peak_z": peak_z,
        "trigger_event_id": trigger_event_id,
    }


def fit(pairs: xr.Dataset, catalog: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    arguments: dict[str, object] = {
        "reference": "van.ch4",
        "numerator": "van.c2h6",
        "period": "native",
        **kwargs,
    }
    return member_fits(pairs, catalog, COLUMNS, **arguments)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# On paper
# ---------------------------------------------------------------------------


@pytest.fixture()
def paper() -> tuple[xr.Dataset, pd.DataFrame]:
    """Twelve 1 s cells, y = 1 + 0.5 x exactly; one candidate with three members.

    At quantile 0.05: exit 1 spans cells 2-7 (seen at entries 3 and 4, one
    interval), exit 2 spans cells 3-6. At quantile 0.10, exit 1 spans 2-8, and
    there the enhancements are 10 higher (a lower baseline): the slope is the
    same, the intercept moves.
    """
    x = np.array([0.0, 1, 4, 9, 16, 25, 30, 25, 16, 9, 4, 1])
    dx = np.stack([x, x + 10.0], axis=1)[:, None, :]
    dy = 1.0 + 0.5 * dx
    pairs = make_pairs(dx, dy, quantiles=(0.05, 0.10))
    catalog = pd.DataFrame(
        [
            event_row("e4", "c0", (2, 8), entry=4.0),
            event_row("e3", "c0", (2, 8), entry=3.0),
            event_row("f3", "c0", (3, 7), exit_=2.0),
            event_row("g3", "c0", (2, 9), quantile=0.10),
        ]
    )
    return pairs, catalog


def test_each_member_is_fitted_once_over_the_cells_whose_midpoint_it_holds(
    paper: tuple[xr.Dataset, pd.DataFrame],
) -> None:
    table = fit(*paper)
    assert list(table.columns) == list(FIT_COLUMNS)
    # Three members, each named by its lowest entry, each estimator once,
    # member-major in candidate, quantile, exit order.
    assert table["event_id"].tolist() == ["e3", "e3", "f3", "f3", "g3", "g3"]
    assert table["method"].tolist() == ["ols", "york"] * 3
    # Cells 2-7 hold midpoints 2.5 ... 7.5: six points; 3-6, four; 2-8, seven.
    assert table["n"].tolist() == [6, 6, 4, 4, 7, 7]
    np.testing.assert_allclose(table["slope"], 0.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(table["intercept"], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], atol=1e-10)
    assert (table["status"] == "OK").all()
    assert table["dof"].tolist() == [4, 4, 2, 2, 5, 5]


def test_the_signal_is_the_swing_over_the_noise(paper: tuple[xr.Dataset, pd.DataFrame]) -> None:
    """Cells 2-7 of x: 4, 9, 16, 25, 30, 25; their variance on n - 1 is 103.1, sigma 0.7."""
    table = fit(*paper)
    variance = np.var([4.0, 9, 16, 25, 30, 25], ddof=1)
    expected = math.sqrt(variance / 0.49 - 1.0)
    assert table["signal"].iloc[0] == pytest.approx(expected, rel=1e-12)
    # Least squares' rows carry the same number for the same pairs.
    assert table["signal"].iloc[0] == table["signal"].iloc[1]


def test_least_squares_rows_carry_yorks_mswd(paper: tuple[xr.Dataset, pd.DataFrame]) -> None:
    table = fit(*paper)
    np.testing.assert_array_equal(table["mswd"].iloc[0::2], table["mswd"].iloc[1::2])
    assert table["birge_factor"].iloc[0::2].isna().all()


def test_only_the_references_events_are_read(paper: tuple[xr.Dataset, pd.DataFrame]) -> None:
    pairs, catalog = paper
    other = pd.DataFrame([event_row("z3", "cz", (0, 12), species="c2h6")])
    table = fit(pairs, pd.concat([catalog, other], ignore_index=True))
    assert "z3" not in set(table["event_id"])


def test_a_member_holding_no_pair_cell_has_no_line() -> None:
    pairs = make_pairs(np.arange(10.0), np.arange(10.0))
    catalog = pd.DataFrame([event_row("e3", "c0", (3.6, 4.4))])
    table = fit(pairs, catalog)
    assert table["n"].tolist() == [0, 0]
    assert (table["status"] == "TOO_FEW_POINTS").all()


def test_no_events_gives_an_empty_table_of_the_declared_types() -> None:
    pairs = make_pairs(np.arange(10.0), np.arange(10.0))
    catalog = pd.DataFrame([event_row("e3", "c0", (0, 5), species="c2h6")])
    table = fit(pairs, catalog)
    assert table.empty
    assert list(table.columns) == list(FIT_COLUMNS)
    assert table["n"].dtype == np.int64
    assert table["slope"].dtype == np.float64


def test_a_triggered_reference_stands_as_tall_as_its_triggers_event() -> None:
    """A canister's methane taking its events from an analyzer has no z of its own."""
    pairs = make_pairs(np.arange(10.0), 0.5 * np.arange(10.0))
    catalog = pd.DataFrame(
        [
            event_row("t3", "c0", (0, 8), peak_z=6.5, species="ch4_dense"),
            event_row("e3", "c0", (0, 8), peak_z=np.nan, trigger_event_id="t3"),
        ]
    )
    assert fit(pairs, catalog)["peak_z"].tolist() == [6.5, 6.5]


def test_without_a_declared_sigma_york_cannot_weigh_and_least_squares_still_fits() -> None:
    pairs = make_pairs(np.arange(10.0), 0.5 * np.arange(10.0), sigma_x=None)
    catalog = pd.DataFrame([event_row("e3", "c0", (0, 10))])
    with _quiet():
        table = fit(pairs, catalog)
    assert table["status"].tolist() == ["OK", "MISSING_SIGMA"]
    assert table["reference_sigma_provenance"].tolist() == ["none", "none"]
    assert table["numerator_sigma_provenance"].tolist() == ["declared", "declared"]
    assert table["signal"].isna().all()


class _quiet:
    """Silence York's one warning about the groups it could not weigh."""

    def __enter__(self) -> None:
        logging.disable(logging.WARNING)

    def __exit__(self, *args: object) -> None:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize("borrowed", [(0.1, 0.25), (0.25, 0.1)])
def test_the_largest_borrowed_share_of_either_member_is_recorded(
    borrowed: tuple[float, float],
) -> None:
    pairs = make_pairs(np.arange(10.0), 0.5 * np.arange(10.0), borrowed=borrowed)
    catalog = pd.DataFrame([event_row("e3", "c0", (0, 10))])
    assert fit(pairs, catalog)["borrowed_max"].tolist() == [0.25, 0.25]


def test_a_midpoint_on_the_start_is_inside_and_on_the_end_outside() -> None:
    """start <= midpoint < end, as a cell holds its midpoint (METHODS §5.2, §6.8).

    On another clock an event's edges can fall on a pair cell's midpoint: here
    2.5 s and 7.5 s, the midpoints of cells 2 and 7, so cells 2-6 are inside.
    """
    pairs = make_pairs(np.arange(12.0) ** 2, 0.5 * np.arange(12.0) ** 2)
    catalog = pd.DataFrame([event_row("e3", "c0", (2.5, 7.5))])
    assert fit(pairs, catalog)["n"].tolist() == [5, 5]


# ---------------------------------------------------------------------------
# The slope's sigma widened (§5.4)
# ---------------------------------------------------------------------------


def _ramp_with_noise(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 50, n) + rng.normal(0, 0.7, n)
    return x, 0.05 * x + rng.normal(0, 0.05, n)


def test_the_persistence_factor_uses_the_longer_declared_time() -> None:
    x, y = _ramp_with_noise(40, 1)
    catalog = pd.DataFrame([event_row("e3", "c0", (0, 40))])
    table = fit(make_pairs(x, y, tau=("2s", "5s")), catalog)
    expected = math.sqrt(40 / float(n_effective(40, 1.0, 5.0)[0]))
    np.testing.assert_allclose(table["persistence_factor"], expected, rtol=1e-12)
    assert table["tau_s"].tolist() == [5.0, 5.0]
    # One declared is that one.
    assert fit(make_pairs(x, y, tau=("2s", None)), catalog)["tau_s"].tolist() == [2.0, 2.0]


def test_the_persistence_factor_reads_the_pair_clocks_spacing() -> None:
    """On 2 s cells, five seconds of persistence span fewer cells than on 1 s."""
    x, y = _ramp_with_noise(40, 5)
    catalog = pd.DataFrame([event_row("e3", "c0", (0, 80))])
    table = fit(make_pairs(x, y, tau=("5s", None), width_s=2), catalog)
    expected = math.sqrt(40 / float(n_effective(40, 2.0, 5.0)[0]))
    np.testing.assert_allclose(table["persistence_factor"], expected, rtol=1e-12)


def test_with_no_declared_time_the_factor_is_one_and_the_time_blank() -> None:
    x, y = _ramp_with_noise(40, 2)
    table = fit(make_pairs(x, y), pd.DataFrame([event_row("e3", "c0", (0, 40))]))
    assert table["persistence_factor"].tolist() == [1.0, 1.0]
    assert table["tau_s"].isna().all()


def test_york_is_widened_by_the_birge_factor_and_least_squares_is_not() -> None:
    """Declared sigmas a third of the scatter give MSWD near 9: York's sigma triples."""
    x, y = _ramp_with_noise(200, 3)
    table = fit(
        make_pairs(x, y, sigma_x=0.7 / 3, sigma_y=0.05 / 3, tau=("3s", None)),
        pd.DataFrame([event_row("e3", "c0", (0, 200))]),
    )
    ols_row, york_row = table.iloc[0], table.iloc[1]
    assert york_row["mswd"] > 4
    assert york_row["birge_factor"] == pytest.approx(math.sqrt(york_row["mswd"]), rel=1e-12)
    assert york_row["slope_sigma_widened"] == pytest.approx(
        york_row["slope_sigma"] * york_row["persistence_factor"] * york_row["birge_factor"],
        rel=1e-12,
    )
    assert ols_row["slope_sigma_widened"] == pytest.approx(
        ols_row["slope_sigma"] * ols_row["persistence_factor"], rel=1e-12
    )


def test_an_mswd_below_one_never_narrows() -> None:
    x, y = _ramp_with_noise(200, 4)
    table = fit(
        make_pairs(x, y, sigma_x=2.1, sigma_y=0.15), pd.DataFrame([event_row("e3", "c0", (0, 200))])
    )
    assert table.iloc[1]["mswd"] < 1
    assert table.iloc[1]["birge_factor"] == 1.0


def test_widening_by_the_declared_time_makes_yorks_sigma_honest_on_persistent_errors() -> None:
    """Monte Carlo, METHODS §4.5 Table 7's case, drawn from the textbook AR(1).

    A 60-reading plume, true dmethane 60 exp(-t^2 / (2 * 10^2)) ppb, dethane
    0.05 of it; noise 3 ppb and 0.3 ppb with lag-one correlation 0.7 on both
    (tau = -1 s / ln 0.7 = 2.8 s), declared exactly; 800 plumes, one member
    each, in one record. Table 7 measured the spread of York's slopes 2.26x its
    sigma as it comes and 0.97x widened; the stage must reproduce both.
    """
    rng = np.random.default_rng(20261117)
    n_events, length, gap, rho = 800, 60, 20, 0.7
    t = np.arange(length) - length / 2
    truth = 60.0 * np.exp(-(t**2) / (2 * 10.0**2))

    def ar1(scale: float) -> np.ndarray:
        noise = np.empty((n_events, length))
        noise[:, 0] = rng.normal(0, scale, n_events)
        innovations = rng.normal(0, scale * math.sqrt(1 - rho**2), (n_events, length))
        for k in range(1, length):
            noise[:, k] = rho * noise[:, k - 1] + innovations[:, k]
        return noise

    block = length + gap
    dx = np.full(n_events * block, np.nan)
    dy = np.full(n_events * block, np.nan)
    inside = (np.arange(n_events)[:, None] * block + np.arange(length)).ravel()
    dx[inside] = (truth + ar1(3.0)).ravel()
    dy[inside] = (0.05 * truth + ar1(0.3)).ravel()
    tau = f"{-1.0 / math.log(rho):.6f}s"
    pairs = make_pairs(dx, dy, sigma_x=3.0, sigma_y=0.3, tau=(tau, tau))
    catalog = pd.DataFrame(
        [event_row(f"e{k}", f"c{k}", (k * block, k * block + length)) for k in range(n_events)]
    )
    table = fit(pairs, catalog, methods=("york",))
    assert (table["status"] == "OK").all()
    spread = 1.4826 * np.median(np.abs(table["slope"] - np.median(table["slope"])))
    as_it_comes = spread / np.median(table["slope_sigma"])
    widened = spread / np.median(table["slope_sigma_widened"])
    assert 1.9 < as_it_comes < 2.6
    assert 0.85 < widened < 1.1
    inside_2sigma = np.abs(table["slope"] - 0.05) <= 2 * table["slope_sigma_widened"]
    assert 0.92 < inside_2sigma.mean() < 0.99


# ---------------------------------------------------------------------------
# Against a reimplementation, on a whole generated chain
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chain() -> tuple[xr.Dataset, pd.DataFrame, dict[str, xr.Dataset], xr.Dataset]:
    """One 1 s analyzer, an hour: methane and ethane at 0.05 in 15 plumes, declared sigmas.

    Rolled at 2 and 10 min and q 0.01 and 0.05; events at entry 3 and 5 and
    exit 1 and 2; methane's enhancements paired with ethane's, readings
    carried.
    """
    rng = np.random.default_rng(11)
    n = 3600
    t = np.arange(n, dtype=float)
    truth = np.zeros(n)
    for centre in rng.uniform(200, n - 200, 15):
        truth += rng.lognormal(np.log(40), 0.6) * np.exp(
            -0.5 * ((t - centre) / rng.uniform(5, 40)) ** 2
        )
    ch4 = 1950 + truth + rng.normal(0, 0.7, n)
    c2h6 = 2.0 + 0.05 * truth + rng.normal(0, 0.05, n)
    start = np.arange(n, dtype=np.int64) * SECOND
    data = {}
    for name, values, sigma in (("ch4", ch4, 0.7), ("c2h6", c2h6, 0.05)):
        data[name] = ("time", values, {"units": "ppb", "role": "gas", "cell_methods": "time: mean"})
        data[sigma_rand_name(name)] = (
            "time",
            np.full(n, sigma),
            {
                "units": "ppb",
                "uncertainty_component": "random",
                "uncertainty_provenance": "declared",
            },
        )
    stream = xr.Dataset(
        data,
        coords={
            "time": (start + SECOND // 2).astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([start, start + SECOND], 1).astype("datetime64[ns]"),
            ),
        },
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    baseline = BaselineConfig(windows=("2min", "10min"), quantiles=(0.01, 0.05))
    events = EventsConfig(enter_multiple=(3.0, 5.0), exit_multiple=(1.0, 2.0))
    with _quiet():
        states = {"van": baseline_state(stream, instrument="van", baseline=baseline)}
        found = event_states(states, events)
        catalog = event_catalog(found, states)
        paired = pair_species(
            states,
            ("van", "enhancement_c2h6"),
            ("van", "enhancement_ch4"),
            carry=[("van", "c2h6"), ("van", "ch4")],
        )
    return paired.dataset, catalog, found, stream


def test_every_member_is_what_fitting_it_alone_gives(
    chain: tuple[xr.Dataset, pd.DataFrame, dict[str, xr.Dataset], xr.Dataset],
) -> None:
    """Written from METHODS §5.2-5.4 one member at a time, the estimators called alone."""
    pairs, catalog, _, _ = chain
    table = fit(pairs, catalog)
    mid = pairs["time"].values.astype("datetime64[ns]").astype(np.int64)
    rows = catalog[catalog["species"] == "ch4"]
    members = (
        rows.sort_values(["candidate_id", "baseline_quantile", "exit_multiple", "enter_multiple"])
        .groupby(["candidate_id", "baseline_quantile", "exit_multiple"], sort=False)
        .head(1)
    )
    assert len(table) == 2 * len(members) > 100
    sx_all = pairs[sigma_rand_name("ch4")].values
    sy_all = pairs[sigma_rand_name("c2h6")].values
    starts = members["start_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    stops = members["end_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for position, member in enumerate(members.itertuples()):
        start, stop = starts[position], stops[position]
        cells = [i for i in range(mid.size) if start <= mid[i] < stop]
        at = {
            "baseline_window": member.baseline_window,
            "baseline_quantile": member.baseline_quantile,
        }
        x = pairs["enhancement_ch4"].sel(at).values[cells]
        y = pairs["enhancement_c2h6"].sel(at).values[cells]
        sx, sy = sx_all[cells], sy_all[cells]
        with _quiet():
            expected = {"ols": ols(x, y, sx, sy), "york": york(x, y, sx, sy)}
        for offset, method in enumerate(("ols", "york")):
            row = table.iloc[2 * position + offset]
            assert row["event_id"] == member.event_id
            line = expected[method]
            for field in ("slope", "intercept", "slope_sigma", "r_squared"):
                np.testing.assert_array_equal(row[field], getattr(line, field)[0])
            assert row["n"] == line.n[0]
        # The numbers beside the line, from their definitions.
        used = np.isfinite(x) & np.isfinite(y) & np.isfinite(sx)
        york_row = table.iloc[2 * position + 1]
        if used.sum() >= 2:
            signal = math.sqrt(max(0.0, np.var(x[used], ddof=1) / np.mean(sx[used] ** 2) - 1))
            assert york_row["signal"] == pytest.approx(signal, rel=1e-12)
        assert york_row["mswd"] == pytest.approx(expected["york"].mswd[0], rel=0, nan_ok=True)


def test_on_the_reference_s_own_cells_midpoint_membership_is_the_event_index(
    chain: tuple[xr.Dataset, pd.DataFrame, dict[str, xr.Dataset], xr.Dataset],
) -> None:
    """METHODS §5.2's promise, at every sweep point of the chain."""
    _, catalog, found, stream = chain
    mid = stream["time"].values.astype("datetime64[ns]").astype(np.int64)
    rows = catalog[catalog["species"] == "ch4"]
    points = rows.groupby(
        ["baseline_window", "baseline_quantile", "enter_multiple", "exit_multiple"]
    )
    assert points.ngroups == 16
    for (window, quantile, entry, exit_), events in points:
        index = (
            found["van"]["event_ch4"]
            .sel(
                baseline_window=window,
                baseline_quantile=quantile,
                enter_multiple=entry,
                exit_multiple=exit_,
            )
            .values
        )
        member = np.full(mid.size, -1)
        for number, start, end in zip(
            events["event_number"], events["start_time"], events["end_time"], strict=True
        ):
            member[(mid >= pd.Timestamp(start).value) & (mid < pd.Timestamp(end).value)] = number
        np.testing.assert_array_equal(member, index)


def test_york_reads_the_generated_ratio(
    chain: tuple[xr.Dataset, pd.DataFrame, dict[str, xr.Dataset], xr.Dataset],
) -> None:
    """A sanity check, not the evidence: tall members of one source read 0.05."""
    pairs, catalog, _, _ = chain
    table = fit(pairs, catalog, methods=("york",))
    tall = table[(table["status"] == "OK") & (table["peak_z"] >= 5)]
    assert len(tall) > 30
    assert np.median(tall["slope"]) == pytest.approx(0.05, rel=0.02)
    assert 0.8 < np.median(tall["mswd"]) < 1.2


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple() -> tuple[xr.Dataset, pd.DataFrame]:
    return make_pairs(np.arange(10.0), 0.5 * np.arange(10.0)), pd.DataFrame(
        [event_row("e3", "c0", (0, 10))]
    )


def test_a_missing_column_is_named(simple: tuple[xr.Dataset, pd.DataFrame]) -> None:
    pairs, catalog = simple
    with pytest.raises(TsaraRatioError, match="no column 'c2h6'"):
        fit(pairs.drop_vars("c2h6"), catalog)


def test_columns_from_another_instrument_are_refused(
    simple: tuple[xr.Dataset, pd.DataFrame],
) -> None:
    pairs, catalog = simple
    with pytest.raises(TsaraRatioError, match="swapped"):
        fit(pairs, catalog, reference="lab.ch4")


def test_an_enhancement_without_the_sweep_is_refused(
    simple: tuple[xr.Dataset, pd.DataFrame],
) -> None:
    pairs, catalog = simple
    flat = pairs.isel(baseline_quantile=0)
    with pytest.raises(TsaraRatioError, match="one version per sweep point"):
        fit(flat, catalog)


def test_unlinked_candidates_are_refused(simple: tuple[xr.Dataset, pd.DataFrame]) -> None:
    pairs, catalog = simple
    with pytest.raises(TsaraRatioError, match="link them"):
        fit(pairs, catalog.assign(candidate_id=None))


def test_entry_copies_that_disagree_are_refused(simple: tuple[xr.Dataset, pd.DataFrame]) -> None:
    pairs, _ = simple
    catalog = pd.DataFrame(
        [event_row("e3", "c0", (0, 10)), event_row("e4", "c0", (1, 10), entry=4.0)]
    )
    with pytest.raises(TsaraRatioError, match="different intervals"):
        fit(pairs, catalog)


def test_a_member_at_a_sweep_point_the_pairing_lacks_is_refused(
    simple: tuple[xr.Dataset, pd.DataFrame],
) -> None:
    pairs, _ = simple
    catalog = pd.DataFrame([event_row("e3", "c0", (0, 10), quantile=0.10)])
    with pytest.raises(TsaraRatioError, match=r"baseline_quantile \[0.1\]"):
        fit(pairs, catalog)


def test_cells_out_of_time_order_are_refused(simple: tuple[xr.Dataset, pd.DataFrame]) -> None:
    pairs, catalog = simple
    with pytest.raises(TsaraRatioError, match="time order"):
        fit(pairs.isel(time=np.arange(9, -1, -1)), catalog)


def test_an_unregistered_estimator_is_refused(simple: tuple[xr.Dataset, pd.DataFrame]) -> None:
    pairs, catalog = simple
    with pytest.raises(TsaraRatioError, match="odr"):
        fit(pairs, catalog, methods=("odr",))
