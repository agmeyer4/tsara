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
from scipy.stats import chi2

from tsara.align import pair_species
from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.core.naming import borrowed_name, sigma_rand_name
from tsara.core.propagation import n_effective
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import event_catalog, event_states
from tsara.ratios import (
    FIT_COLUMNS,
    PLATEAU_WIDTH,
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
#: The same columns with both gases' clean spreads carried (METHODS §5.4).
CLEAN_COLUMNS = PairColumns(
    numerator="enhancement_c2h6",
    reference="enhancement_ch4",
    numerator_reading="c2h6",
    reference_reading="ch4",
    numerator_clean_spread="clean_spread_c2h6",
    reference_clean_spread="clean_spread_ch4",
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
    clean_spread: tuple[float | None, float | None] = (None, None),
) -> xr.Dataset:
    """A paired product as pairing makes one: abutting cells of ``width_s`` (1 s
    by default), enhancements swept, readings carried.

    ``dx`` and ``dy`` are (cells,) arrays, the same at every sweep point, or
    (cells, windows, quantiles) arrays. ``tau`` is the (reference, numerator)
    decorrelation time declared on each enhancement, ``borrowed`` the
    (reference, numerator) borrowed share of every value, ``clean_spread``
    the (reference, numerator) clean spread carried beside them (columns
    :data:`CLEAN_COLUMNS` names), none when None.
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
    for column, spread in (
        (CLEAN_COLUMNS.reference_clean_spread, clean_spread[0]),
        (CLEAN_COLUMNS.numerator_clean_spread, clean_spread[1]),
    ):
        if spread is not None and column is not None:
            data[column] = (swept, np.full(shape, spread), {"units": "ppb"})
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


def test_every_row_carries_its_candidates_tallest_height() -> None:
    """The height gate reads the candidate's tallest member (METHODS §5.3)."""
    pairs = make_pairs(np.arange(20.0), 0.5 * np.arange(20.0), quantiles=(0.05, 0.10))
    catalog = pd.DataFrame(
        [
            event_row("a3", "c0", (0, 8), peak_z=7.5),
            event_row("b3", "c0", (0, 8), quantile=0.10, peak_z=4.2),
            event_row("c3", "c1", (10, 18), peak_z=3.9),
        ]
    )
    table = fit(pairs, catalog)
    assert table["peak_z"].tolist() == [7.5, 7.5, 4.2, 4.2, 3.9, 3.9]
    assert table["candidate_peak_z"].tolist() == [7.5, 7.5, 7.5, 7.5, 3.9, 3.9]


def test_the_plateau_is_york_over_twice_the_member_centred_on_it() -> None:
    """A member over cells 8-11 of a 20-cell ramp at 0.05, with a neighbour at 0 after it.

    The plateau window is 4 s either side of the member's centre, 10 s: cells
    6-13, eight points. The member's own four points lie on one line (MSWD 0);
    the plateau's do not, since cells 12 and 13 hold the neighbour. A window
    anchored on the member's start (cells 4-11) would miss it.
    """
    assert PLATEAU_WIDTH == 2
    dx = np.arange(20.0) + 10.0
    dy = 0.05 * dx
    dy[[12, 13]] = 0.0
    table = fit(make_pairs(dx, dy), pd.DataFrame([event_row("e3", "c0", (8, 12))]))
    york_row = table.iloc[1]
    assert york_row["n"] == 4
    assert york_row["mswd"] == pytest.approx(0.0, abs=1e-12)
    assert york_row["plateau_dof"] == 6
    # Well past the 99 % gate for six degrees of freedom, which the member passes.
    assert york_row["plateau_mswd"] > chi2.ppf(0.99, 6) / 6
    assert york_row["plateau_slope"] != pytest.approx(0.05, rel=0.1)
    # Least squares' row carries the same plateau numbers.
    assert table.iloc[0]["plateau_mswd"] == york_row["plateau_mswd"]


def test_on_a_single_line_the_plateau_agrees_with_the_member() -> None:
    dx = np.arange(20.0) + 10.0
    table = fit(make_pairs(dx, 0.05 * dx), pd.DataFrame([event_row("e3", "c0", (8, 12))]))
    assert table.iloc[1]["plateau_slope"] == pytest.approx(0.05, rel=1e-12)
    assert table.iloc[1]["plateau_mswd"] == pytest.approx(0.0, abs=1e-12)


def test_without_york_the_plateau_is_blank() -> None:
    dx = np.arange(20.0) + 10.0
    table = fit(
        make_pairs(dx, 0.05 * dx), pd.DataFrame([event_row("e3", "c0", (8, 12))]), methods=("ols",)
    )
    assert table["plateau_slope"].isna().all() and table["plateau_mswd"].isna().all()
    assert table["plateau_dof"].tolist() == [-2]


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
def test_the_largest_borrowed_share_of_either_variable_is_recorded(
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


def reference_factors(
    x: np.ndarray, weight: np.ndarray, seconds: np.ndarray, tau_s: float
) -> tuple[float, float]:
    """The persistence factor and the expected MSWD, by a plain double loop (METHODS §5.4).

    P^2 = sum_ij v_i v_j rho_ij / sum_i v_i^2, v_i = sqrt(w_i) (x_i - xbar_w);
    E = (n - sum_ij s_i s_j rho_ij / sum_i w_i - P^2) / (n - 2), s_i = sqrt(w_i);
    rho_ij = exp(-|t_i - t_j| / tau). Points with a non-finite x are left out.
    """
    keep = np.isfinite(x)
    x, weight, seconds = x[keep], weight[keep], seconds[keep]
    n = x.size
    mean = float(np.sum(weight * x) / np.sum(weight))
    v = [math.sqrt(weight[i]) * (x[i] - mean) for i in range(n)]
    s = [math.sqrt(weight[i]) for i in range(n)]
    vrv = srs = 0.0
    for i in range(n):
        for j in range(n):
            rho = math.exp(-abs(seconds[i] - seconds[j]) / tau_s)
            vrv += v[i] * v[j] * rho
            srs += s[i] * s[j] * rho
    persistence = math.sqrt(vrv / sum(value * value for value in v))
    return persistence, (n - srs / float(np.sum(weight)) - persistence**2) / (n - 2)


def test_the_factors_use_the_longer_declared_time() -> None:
    """Equal sigmas: York's weights are equal too, so both rows share both factors."""
    x, y = _ramp_with_noise(40, 1)
    catalog = pd.DataFrame([event_row("e3", "c0", (0, 40))])
    table = fit(make_pairs(x, y, tau=("2s", "5s")), catalog)
    persistence, expected = reference_factors(x, np.ones(40), np.arange(40) + 0.5, 5.0)
    np.testing.assert_allclose(table["persistence_factor"], persistence, rtol=1e-12)
    np.testing.assert_allclose(table["mswd_expected"], expected, rtol=1e-12)
    assert table["tau_s"].tolist() == [5.0, 5.0]
    # One declared is that one.
    assert fit(make_pairs(x, y, tau=("2s", None)), catalog)["tau_s"].tolist() == [2.0, 2.0]


def test_the_factors_read_the_pair_cells_own_times() -> None:
    """On 2 s cells, and across a hole: the correlation of two points is that of their times.

    Cells 10-14 hold no reference value, so the points either side of the hole
    are 12 s apart, not 2: the median spacing would not see it.
    """
    x, y = _ramp_with_noise(40, 5)
    x[10:15] = np.nan
    catalog = pd.DataFrame([event_row("e3", "c0", (0, 80))])
    table = fit(make_pairs(x, y, tau=("5s", None), width_s=2), catalog)
    persistence, expected = reference_factors(x, np.ones(40), 2.0 * np.arange(40) + 1.0, 5.0)
    np.testing.assert_allclose(table["persistence_factor"], persistence, rtol=1e-12)
    np.testing.assert_allclose(table["mswd_expected"], expected, rtol=1e-12)


def test_york_weighs_each_point_by_its_own_sigma_at_its_slope() -> None:
    """Unequal sigmas: York's factors are its weights', least squares' the equal weights'."""
    x, y = _ramp_with_noise(60, 6)
    rng = np.random.default_rng(60)
    sigma_x, sigma_y = rng.uniform(0.3, 2.0, 60), rng.uniform(0.02, 0.2, 60)
    table = fit(
        make_pairs(x, y, sigma_x=sigma_x, sigma_y=sigma_y, tau=("3s", "3s")),
        pd.DataFrame([event_row("e3", "c0", (0, 60))]),
    )
    ols_row, york_row = table.iloc[0], table.iloc[1]
    seconds = np.arange(60) + 0.5
    weight = 1.0 / (sigma_y**2 + york_row["slope"] ** 2 * sigma_x**2)
    persistence, expected = reference_factors(x, weight, seconds, 3.0)
    assert york_row["persistence_factor"] == pytest.approx(persistence, rel=1e-12)
    assert york_row["mswd_expected"] == pytest.approx(expected, rel=1e-12)
    persistence, expected = reference_factors(x, np.ones(60), seconds, 3.0)
    assert ols_row["persistence_factor"] == pytest.approx(persistence, rel=1e-12)
    assert ols_row["mswd_expected"] == pytest.approx(expected, rel=1e-12)


def test_with_no_declared_time_the_factors_are_one_and_the_time_blank() -> None:
    x, y = _ramp_with_noise(40, 2)
    table = fit(make_pairs(x, y), pd.DataFrame([event_row("e3", "c0", (0, 40))]))
    assert table["persistence_factor"].tolist() == [1.0, 1.0]
    assert table["mswd_expected"].tolist() == [1.0, 1.0]
    assert table["tau_s"].isna().all()
    ols_row, york_row = table.iloc[0], table.iloc[1]
    assert ols_row["slope_sigma_widened"] == ols_row["slope_sigma"]
    assert york_row["slope_sigma_widened"] == pytest.approx(
        york_row["slope_sigma"] * math.sqrt(max(1.0, york_row["mswd"])), rel=1e-12
    )


def test_york_is_widened_by_the_birge_factor_and_least_squares_by_its_scatter() -> None:
    """Declared sigmas a third of the scatter give an MSWD near 9: York's sigma triples.

    York: sigma x P x sqrt(max(1, MSWD / E)). Least squares' sigma comes from
    the scatter, which persistence shrinks by E: sigma x P / sqrt(E).
    """
    x, y = _ramp_with_noise(200, 3)
    table = fit(
        make_pairs(x, y, sigma_x=0.7 / 3, sigma_y=0.05 / 3, tau=("3s", None)),
        pd.DataFrame([event_row("e3", "c0", (0, 200))]),
    )
    ols_row, york_row = table.iloc[0], table.iloc[1]
    assert york_row["mswd"] > 4
    assert 0 < york_row["mswd_expected"] < 1
    assert york_row["birge_factor"] == pytest.approx(
        math.sqrt(york_row["mswd"] / york_row["mswd_expected"]), rel=1e-12
    )
    assert york_row["slope_sigma_widened"] == pytest.approx(
        york_row["slope_sigma"] * york_row["persistence_factor"] * york_row["birge_factor"],
        rel=1e-12,
    )
    assert math.isnan(ols_row["birge_factor"])
    assert ols_row["slope_sigma_widened"] == pytest.approx(
        ols_row["slope_sigma"]
        * ols_row["persistence_factor"]
        / math.sqrt(ols_row["mswd_expected"]),
        rel=1e-12,
    )


@pytest.mark.parametrize("tau", [None, "3s"])
def test_an_mswd_below_its_expectation_never_narrows(tau: str | None) -> None:
    x, y = _ramp_with_noise(200, 4)
    table = fit(
        make_pairs(x, y, sigma_x=2.1, sigma_y=0.15, tau=(tau, None)),
        pd.DataFrame([event_row("e3", "c0", (0, 200))]),
    )
    york_row = table.iloc[1]
    assert york_row["mswd"] < york_row["mswd_expected"]
    assert york_row["birge_factor"] == 1.0


def test_two_points_leave_no_degree_of_freedom_to_expect_an_mswd_on() -> None:
    x, y = _ramp_with_noise(10, 8)
    table = fit(make_pairs(x, y, tau=("3s", None)), pd.DataFrame([event_row("e3", "c0", (0, 2))]))
    assert table["n"].tolist() == [2, 2]
    assert table["mswd_expected"].isna().all()
    assert table["slope_sigma_widened"].isna().all()


def test_without_a_declared_sigma_york_has_no_factors_to_widen_by() -> None:
    pairs = make_pairs(np.arange(10.0), 0.5 * np.arange(10.0), sigma_x=None, tau=("3s", None))
    with _quiet():
        table = fit(pairs, pd.DataFrame([event_row("e3", "c0", (0, 10))]))
    york_row = table.iloc[1]
    assert york_row["status"] == "MISSING_SIGMA"
    assert math.isnan(york_row["persistence_factor"])
    assert math.isnan(york_row["mswd_expected"])
    assert math.isnan(york_row["birge_factor"])
    assert math.isnan(york_row["slope_sigma_widened"])
    assert np.isfinite(table.iloc[0]["slope_sigma_widened"])  # least squares weighs nothing


@pytest.mark.parametrize("n_members", [1, 2])
def test_noise_lingering_longer_than_a_member_moves_its_intercept_not_its_slope(
    n_members: int,
) -> None:
    """A decorrelation time far beyond the member: the factors tend to their limits.

    Noise constant over a member lifts all its points alike, which the
    intercept absorbs: the persistence factor falls as tau^(-1/2) and the
    scatter the line leaves (E) as 1/tau. Every lag within a member is summed
    (one member: the lags run out; two: no pair a lag apart is in one member).
    """
    x, y = _ramp_with_noise(30, 7)
    cells = np.concatenate([x, np.full(5, np.nan), x])[: 30 + (35 if n_members == 2 else 0)]
    values = np.concatenate([y, np.full(5, np.nan), y])[: cells.size]
    catalog = pd.DataFrame(
        [event_row(f"e{k}", f"c{k}", (35 * k, 35 * k + 30)) for k in range(n_members)]
    )
    table = fit(make_pairs(cells, values, tau=("100000s", None)), catalog)
    persistence, expected = reference_factors(x, np.ones(30), np.arange(30) + 0.5, 100000.0)
    np.testing.assert_allclose(table["persistence_factor"], persistence, rtol=1e-9)
    np.testing.assert_allclose(table["mswd_expected"], expected, rtol=1e-6)
    assert (table["persistence_factor"] < 0.05).all()
    assert (table["mswd_expected"] < 1e-4).all()


def _persistent_plumes(
    rho: float, width: float, length: int, scale: float, n_events: int, seed: int
) -> pd.DataFrame:
    """Fit ``n_events`` copies of one plume whose noise persists (Monte Carlo, §5.4).

    True dmethane 60 exp(-t^2 / (2 width^2)) ppb over ``length`` 1 s cells,
    dethane 0.05 of it; noise 3 and 0.3 ppb with lag-one correlation ``rho``
    on both (tau = -1 s / ln rho), drawn from the textbook AR(1) and declared
    as ``scale`` times the truth; one member each, 20 s apart, in one record.
    """
    rng = np.random.default_rng(seed)
    gap = 20
    t = np.arange(length) - length / 2
    truth = 60.0 * np.exp(-(t**2) / (2 * width**2))

    def ar1(sd: float) -> np.ndarray:
        noise = np.empty((n_events, length))
        noise[:, 0] = rng.normal(0, sd, n_events)
        innovations = rng.normal(0, sd * math.sqrt(1 - rho**2), (n_events, length))
        for k in range(1, length):
            noise[:, k] = rho * noise[:, k - 1] + innovations[:, k]
        return noise

    block = length + gap
    dx = np.full(n_events * block, np.nan)
    dy = np.full(n_events * block, np.nan)
    inside = (np.arange(n_events)[:, None] * block + np.arange(length)).ravel()
    dx[inside] = (truth + ar1(3.0)).ravel()
    dy[inside] = (0.05 * truth + ar1(0.3)).ravel()
    tau = f"{-1.0 / math.log(rho):.9f}s"
    pairs = make_pairs(dx, dy, sigma_x=3.0 * scale, sigma_y=0.3 * scale, tau=(tau, tau))
    catalog = pd.DataFrame(
        [event_row(f"e{k}", f"c{k}", (k * block, k * block + length)) for k in range(n_events)]
    )
    table = fit(pairs, catalog)
    assert (table["status"] == "OK").all()
    return table


def _spread_over_sigma(table: pd.DataFrame, method: str, column: str) -> float:
    rows = table[table["method"] == method]
    return float(np.std(rows["slope"], ddof=1) / np.median(rows[column]))


def test_widening_by_the_declared_time_makes_yorks_sigma_honest_on_persistent_errors() -> None:
    """Monte Carlo, METHODS §4.5 Table 7's case: rho 0.7 (tau 2.8 s), a 10 s plume, 60 cells.

    York's slopes spread 2.2x its sigma as it comes; widened, the sigma is
    honest, and least squares' too (its slope is diluted, its sigma is not).
    """
    table = _persistent_plumes(0.7, 10.0, 60, 1.0, 800, 20261117)
    assert 1.9 < _spread_over_sigma(table, "york", "slope_sigma") < 2.6
    assert 0.9 < _spread_over_sigma(table, "york", "slope_sigma_widened") < 1.1
    assert 0.9 < _spread_over_sigma(table, "ols", "slope_sigma_widened") < 1.1
    york_rows = table[table["method"] == "york"]
    inside = np.abs(york_rows["slope"] - 0.05) <= 2 * york_rows["slope_sigma_widened"]
    assert 0.93 < inside.mean() < 0.98


def test_a_slope_is_widened_by_its_own_factor_not_the_means() -> None:
    """rho 0.9 (tau 9.5 s) on a 5 s plume: the mean's sqrt(N / N_eff) would widen 1.75x too far.

    Measured through the stage before the repair: the slopes spread 0.57 of
    York's sigma widened by the mean's factor; by the slope's own, 0.97.
    """
    table = _persistent_plumes(0.9, 5.0, 40, 1.0, 800, 20261118)
    assert 0.9 < _spread_over_sigma(table, "york", "slope_sigma_widened") < 1.1
    york_rows = table[table["method"] == "york"]
    mean_factor = np.sqrt(40 / n_effective(40, 1.0, -1.0 / math.log(0.9)))
    assert 1.6 < float(mean_factor[0]) / np.median(york_rows["persistence_factor"]) < 1.9


def test_birge_reads_the_mswd_against_what_persistence_makes_it_expect() -> None:
    """rho 0.9, the noise declared at half: a raw-MSWD Birge factor under-widens.

    Persistence makes a single source's MSWD read low (E about 0.6 here), so
    an understated sigma shows only partly in the MSWD: widened on the raw
    MSWD, York's sigma promises a third less spread than the slopes have; on
    MSWD / E, within the scatter of an MSWD on few effective points.
    """
    table = _persistent_plumes(0.9, 10.0, 60, 0.5, 800, 20261119)
    york_rows = table[table["method"] == "york"]
    assert 0.5 < np.median(york_rows["mswd_expected"]) < 0.75
    assert 0.9 < _spread_over_sigma(table, "york", "slope_sigma_widened") < 1.2
    on_raw = (
        york_rows["slope_sigma"]
        * york_rows["persistence_factor"]
        * np.sqrt(np.fmax(1.0, york_rows["mswd"]))
    )
    assert np.std(york_rows["slope"], ddof=1) / np.median(on_raw) > 1.25


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


# ---------------------------------------------------------------------------
# The baseline's error (METHODS §5.4)
# ---------------------------------------------------------------------------


def _wandering_record(
    seed: int, n: int = 7_200, *, varying: bool = False
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Two hours of 1 s cells: four plumes of one source on backgrounds that wander, and
    an event around each plume, at two windows; with ``varying``, methane's declared
    sigma is 0.9 ppb every third second and 0.5 ppb
    otherwise, so a member's median wobble is not its mean."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    plume = np.zeros(n)
    spans = []
    for k, centre in enumerate((900, 2400, 4200, 6000)):
        width = 30 + 15 * k
        plume += (30 + 10 * k) * np.exp(-0.5 * ((t - centre) / width) ** 2)
        spans.append((centre - 3 * width, centre + 3 * width))
    walk_x = np.cumsum(rng.normal(0, 0.15, n))
    walk_y = np.cumsum(rng.normal(0, 0.006, n))
    dx = plume + walk_x + rng.normal(0, 0.7, n)
    dy = 0.05 * plume + walk_y + rng.normal(0, 0.05, n)
    sigma_x = np.where(np.arange(n) % 3 == 0, 0.9, 0.5) if varying else 0.7
    pairs = make_pairs(dx, dy, sigma_x=sigma_x, windows=(120.0, 600.0), clean_spread=(1.1, 0.07))
    rows = []
    for window in (120.0, 600.0):
        for k, (start, end) in enumerate(spans):
            rows.append(
                event_row(
                    f"e{window:.0f}_{k}",
                    f"van.ch4/w{window:.0f}s/c{k}",
                    (start, end),
                    window=window,
                )
            )
            rows.append(
                event_row(
                    f"f{window:.0f}_{k}",
                    f"van.ch4/w{window:.0f}s/c{k}",
                    (start + 20, end - 20),
                    window=window,
                    exit_=2.0,
                )
            )
    return pairs, pd.DataFrame(rows)


def _slow_factors_by_hand(
    pairs: xr.Dataset, catalog: pd.DataFrame, row: pd.Series, wandering: object
) -> tuple[float, float]:
    """The baseline factor and expected MSWD of one fit, by a double loop from METHODS §5.4."""
    from tsara.ratios.baseline_error import LAGS_S

    mid = pairs.time.values.astype("datetime64[ns]").astype(np.int64)
    inside = (mid >= pd.Timestamp(row.start_time).value) & (mid < pd.Timestamp(row.end_time).value)
    at = dict(baseline_window=row.baseline_window, baseline_quantile=row.baseline_quantile)
    x = pairs["enhancement_ch4"].sel(at).values[inside]
    t = mid[inside] / 1e9
    b = row.slope
    sx = pairs["sigma_rand_ch4"].values[inside]
    sy = pairs["sigma_rand_c2h6"].values[inside]
    w = 1.0 / (sy**2 + b * b * sx**2)
    weight = w if row.method == "york" else np.ones(x.size)
    # York's points are scaled by their own residual noise; least squares' by its mean.
    c = np.sqrt(w) if row.method == "york" else np.full(x.size, 1 / np.sqrt(np.mean(1 / w)))
    s = np.sqrt(weight)
    v = s * (x - np.sum(weight * x) / np.sum(weight))
    i = list(pairs.baseline_window.values).index(row.baseline_window)
    tables = [getattr(wandering, name)[i, 0] for name in ("reference", "numerator", "cross")]
    wobble_x = np.median(np.fmax(0.0, 1.1**2 - sx**2))
    wobble_y = np.median(np.fmax(0.0, 0.07**2 - sy**2))
    amplitude = wobble_y + b * b * wobble_x
    # S whole, as a matrix: METHODS' formula at every pair's |dt|, read in log lag.
    dt = np.abs(t[:, None] - t[None, :])
    logs = np.log(np.clip(dt, 1.0, 7200.0))
    g = [np.where(dt == 0, 0.0, np.interp(logs, np.log(LAGS_S), tab)) for tab in tables]
    S = amplitude * np.fmax(0.0, 1 - dt / row.baseline_window) - np.fmax(
        0.0, g[1] + b * b * g[0] - 2 * b * g[2]
    )
    St = (c[:, None] * S) * c[None, :]
    v_sv = max(0.0, v @ St @ v)
    total2 = (v @ v + v_sv) / (v @ v)
    left = max(0.0, np.trace(St) - (s @ St @ s) / np.sum(weight) - v_sv / (v @ v))
    return float(np.sqrt(total2)), float((x.size - 1 - 1 + left) / (x.size - 2))


@pytest.mark.parametrize("varying", [False, True], ids=["one sigma", "sigma varying"])
def test_the_baseline_factor_and_expected_mswd_are_their_definitions(varying: bool) -> None:
    from tsara.ratios.baseline_error import wander

    pairs, catalog = _wandering_record(11, varying=varying)
    got = member_fits(
        pairs, catalog, CLEAN_COLUMNS, reference="van.ch4", numerator="van.c2h6", period="native"
    )
    mid = pairs.time.values.astype("datetime64[ns]").astype(np.int64)
    deltas = tuple(
        pairs[name].transpose("time", "baseline_window", "baseline_quantile").values
        for name in ("enhancement_c2h6", "enhancement_ch4")
    )
    sigmas = (pairs["sigma_rand_c2h6"].values, pairs["sigma_rand_ch4"].values)
    wandering = wander(pairs, catalog, deltas, sigmas, mid, ("van.c2h6", "van.ch4"))  # type: ignore[arg-type]
    assert wandering.described.all()
    assert set(got.baseline_error) == {"wobble and wander"}
    for k, row in enumerate(got.itertuples()):
        baseline, expected = _slow_factors_by_hand(pairs, catalog, row, wandering)  # type: ignore[arg-type]
        assert got.baseline_factor.iloc[k] == pytest.approx(baseline, rel=1e-9)
        assert got.mswd_expected.iloc[k] == pytest.approx(expected, rel=1e-9)
        mswd = float(got.mswd.iloc[k])
        birge = (
            math.sqrt(max(1.0, mswd / expected))
            if got.method.iloc[k] == "york"
            else 1 / math.sqrt(expected)
        )
        assert got.slope_sigma_widened.iloc[k] == pytest.approx(
            float(got.slope_sigma.iloc[k]) * baseline * birge, rel=1e-9
        )
    if varying:
        return
    # The wobble is the clean spread beyond the readings' noise.
    assert got.wobble_reference.to_numpy() == pytest.approx(math.sqrt(1.1**2 - 0.7**2))
    assert got.wobble_numerator.to_numpy() == pytest.approx(math.sqrt(0.07**2 - 0.05**2))
    assert (got.baseline_factor > 1).all()


def test_the_baseline_factor_never_narrows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A measured semivariogram need not be a valid one (§5.4). Stand in the most invalid:
    S(dt) = +|dt|, minus a valid semivariogram, so that v'S~v is negative for every member;
    the baseline's part is floored, and every fit keeps the noise's own factors."""
    import tsara.ratios.fits as fits_module

    pairs, catalog = _wandering_record(12)
    no_clean = make_pairs(
        pairs["enhancement_ch4"].values[:, 0, 0],
        pairs["enhancement_c2h6"].values[:, 0, 0],
        windows=(120.0, 600.0),
    )
    monkeypatch.setattr(fits_module, "slow_covariance", lambda dt, *args: np.abs(dt))
    got = fit(no_clean, catalog)
    assert set(got.baseline_error) == {"wander"}
    assert got.baseline_factor.to_numpy() == pytest.approx(1.0, abs=1e-12)
    # No persistence and the baseline's part floored: E is the noise's own, 1.
    assert got.mswd_expected.to_numpy() == pytest.approx(1.0, abs=1e-12)


def test_each_fit_says_which_parts_of_the_baseline_error_it_holds() -> None:
    pairs, catalog = _wandering_record(13)
    no_clean = make_pairs(
        pairs["enhancement_ch4"].values[:, 0, 0],
        pairs["enhancement_c2h6"].values[:, 0, 0],
        windows=(120.0, 600.0),
    )
    assert set(fit(no_clean, catalog).baseline_error) == {"wander"}
    assert fit(no_clean, catalog).wobble_reference.isna().all()
    # A record too short to read any lag of, outside its events, with clean spreads: wobble only.
    short = make_pairs(
        np.linspace(0, 10, 40), np.linspace(0, 0.5, 40), windows=(600.0,), clean_spread=(1.1, 0.07)
    )
    short_catalog = pd.DataFrame([event_row("e0", "c0", (0, 40))])
    got = member_fits(
        short,
        short_catalog,
        CLEAN_COLUMNS,
        reference="van.ch4",
        numerator="van.c2h6",
        period="native",
    )
    assert set(got.baseline_error) == {"wobble"}
    # Neither: no clean spread and no lag.
    bare = make_pairs(np.linspace(0, 10, 40), np.linspace(0, 0.5, 40), windows=(600.0,))
    assert set(fit(bare, short_catalog).baseline_error) == {"none"}
    assert (fit(bare, short_catalog).baseline_factor == 1.0).all()


def test_enhancements_known_to_carry_no_baseline_error_are_not_widened_for_one() -> None:
    """On a true background (a test's, a notebook's) the wander outside the events is the
    air's own small plumes; with baseline_error=False the fit widens for nothing of it."""
    pairs, catalog = _wandering_record(14)
    got = member_fits(
        pairs,
        catalog,
        CLEAN_COLUMNS,
        reference="van.ch4",
        numerator="van.c2h6",
        period="native",
        baseline_error=False,
    )
    assert set(got.baseline_error) == {"none"}
    assert got.wobble_reference.isna().all() and got.wobble_numerator.isna().all()
    assert (got.baseline_factor == 1.0).all() and (got.mswd_expected == 1.0).all()
    york_rows = got[got.method == "york"]
    birge = np.sqrt(np.fmax(1.0, york_rows.mswd))
    np.testing.assert_allclose(
        york_rows.slope_sigma_widened, york_rows.slope_sigma * birge, rtol=1e-12
    )
