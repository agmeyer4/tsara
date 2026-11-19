"""Tests for the sweep summary (tsara.ratios.sweep, METHODS §5.6).

The evidence: fit tables small enough to count by hand at two sweep points
and two gate settings, the Birge ratio against its formula, a named
population, and agreement with candidate_ratios on how many candidates a
window gives at each setting.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tsara.config.analysis import GatesConfig
from tsara.ratios import FIT_COLUMNS, GATE_COLUMNS, SWEEP_DIMS, candidate_ratios, sweep_summary


def _fit(candidate: str, slope: float, sigma: float = 0.01, **values: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(FIT_COLUMNS, np.nan)
    row.update(
        event_id=f"{candidate}/{slope:g}",
        candidate_id=candidate,
        reference="van.ch4",
        numerator="van.c2h6",
        period="native",
        method="york",
        baseline_window=600.0,
        baseline_quantile=0.05,
        exit_multiple=1.0,
        peak_time=pd.Timestamp("2026-01-01"),
        peak_z=10.0,
        candidate_peak_z=10.0,
        n=50,
        status="OK",
        slope=slope,
        slope_sigma_widened=sigma,
        mswd=1.0,
        mswd_expected=1.0,
        dof=48,
        signal=10.0,
        plateau_mswd=1.0,
        plateau_dof=98,
    )
    row.update(values)
    return row


def _fits() -> pd.DataFrame:
    """Three candidates at exit 1, two at exit 2; c3 short of height 6 but not of 5; and
    c4's fit at exit 1, which has no line."""
    return pd.DataFrame(
        [
            _fit("c4", np.nan, np.nan, status="MISSING_SIGMA"),
            _fit("c1", 0.050, 0.01),
            _fit("c2", 0.060, 0.01),
            _fit("c3", 0.040, 0.02, candidate_peak_z=5.5),
            _fit("c1", 0.052, 0.01, exit_multiple=2.0),
            _fit("c2", 0.200, 0.01, exit_multiple=2.0, mswd=3.0),
        ]
    )


def test_the_counts_at_each_sweep_point_and_gate_setting() -> None:
    got = sweep_summary(_fits(), GatesConfig(min_peak_z=(5.0, 6.0)))
    assert set(got.dims) == {*SWEEP_DIMS, *GATE_COLUMNS}
    at = {
        "numerator": "van.c2h6",
        "estimator": "york",
        "period": "native",
        "baseline_window": 600.0,
        "baseline_quantile": 0.05,
        "min_points": 8,
        "min_signal": 2.0,
        "mswd_probability": 0.99,
        "plateau_probability": 1.0,
    }
    one = got.sel({**at, "exit_multiple": 1.0})
    assert list(one.n_fits.values) == [4, 4]
    assert list(one.n_pass.values) == [3, 2]  # height 6 drops c3
    assert list(one.n_pass_height.values) == [3, 2]
    # A fit with no line passes no gate, not even one its numbers would meet.
    assert list(one.n_pass_points.values) == [3, 3]
    # The population's summaries are over the fits that pass: c1 and c2 at height 6.
    assert list(one.median_ratio.values) == pytest.approx([0.05, 0.055])
    two = got.sel({**at, "exit_multiple": 2.0})
    assert list(two.n_pass.values) == [1, 1]  # c2's MSWD 3 fails the one-source gate
    assert list(two.n_pass_one_source.values) == [1, 1]
    assert list(two.n_pass_points.values) == [2, 2]


def test_the_population_summaries_and_the_birge_ratio() -> None:
    got = sweep_summary(_fits(), GatesConfig())
    point = got.sel(exit_multiple=1.0).squeeze()
    slopes, sigma = np.array([0.05, 0.06, 0.04]), np.array([0.01, 0.01, 0.02])
    weight = 1 / sigma**2
    mean = np.sum(weight * slopes) / np.sum(weight)
    assert int(point.n_population) == 3
    assert float(point.median_ratio) == pytest.approx(0.05)
    assert float(point.spread_ratio) == pytest.approx(1.4826 * 0.01)
    assert float(point.median_mswd) == 1.0
    assert float(point.birge_ratio) == pytest.approx(
        math.sqrt(np.sum(weight * (slopes - mean) ** 2) / 2)
    )
    # One passing fit at exit 2: no scatter to judge a budget by.
    two = got.sel(exit_multiple=2.0).squeeze()
    assert int(two.n_population) == 1 and np.isnan(float(two.birge_ratio))


def test_a_named_population_changes_the_summaries_not_the_counts() -> None:
    every = sweep_summary(_fits(), GatesConfig()).sel(exit_multiple=1.0).squeeze()
    named = sweep_summary(_fits(), GatesConfig(), population=iter(["c1", "c2"]))
    point = named.sel(exit_multiple=1.0).squeeze()
    assert int(point.n_fits) == int(every.n_fits) and int(point.n_pass) == int(every.n_pass)
    assert int(point.n_population) == 2
    assert float(point.median_ratio) == pytest.approx(0.055)
    assert named.attrs["population"] == "2 named candidates"
    assert every.attrs["population"] == "every candidate"


def test_a_sweep_point_with_no_fit_counts_zero_and_has_no_summary() -> None:
    fits = pd.concat([_fits(), pd.DataFrame([_fit("c9", 0.05, method="ols", exit_multiple=2.0)])])
    got = sweep_summary(fits, GatesConfig())
    empty = got.sel(estimator="ols", exit_multiple=1.0).squeeze()
    assert int(empty.n_fits) == 0 and int(empty.n_pass) == 0
    assert np.isnan(float(empty.median_ratio))


def test_a_fit_without_a_sigma_has_no_birge_ratio_to_join() -> None:
    fits = pd.DataFrame([_fit("c1", 0.05), _fit("c2", 0.06, sigma=np.nan)])
    point = sweep_summary(fits, GatesConfig()).squeeze()
    assert int(point.n_population) == 2 and np.isnan(float(point.birge_ratio))


def test_n_candidates_is_how_many_ratios_candidate_ratios_gives() -> None:
    fits = _fits()
    gates = GatesConfig(min_peak_z=(5.0, 6.0))
    got = sweep_summary(fits, gates)
    ratios = candidate_ratios(
        fits,
        pd.DataFrame({"candidate_id": ["c1", "c2", "c3", "c4"], "parent_candidate_id": [None] * 4}),
        gates,
    )
    for z, expected in (
        ratios.groupby("min_peak_z").n_members.apply(lambda n: int((n > 0).sum())).items()
    ):
        assert int(got.n_candidates.sel(min_peak_z=z).squeeze()) == expected
    assert "baseline_quantile" not in got.n_candidates.dims


def test_no_fits_give_an_empty_summary() -> None:
    got = sweep_summary(pd.DataFrame(columns=list(FIT_COLUMNS)), GatesConfig(), population=["c1"])
    assert not got.data_vars and got.attrs["population"] == "1 named candidates"


def test_the_median_mswd_is_the_median_and_a_zero_sigma_leaves_no_budget() -> None:
    fits = pd.DataFrame(
        [_fit("c1", 0.05, mswd=1.0), _fit("c2", 0.06, mswd=1.1), _fit("c3", 0.04, mswd=1.4)]
    )
    assert float(sweep_summary(fits, GatesConfig()).median_mswd.squeeze()) == pytest.approx(1.1)
    zero = pd.DataFrame([_fit("c1", 0.05), _fit("c2", 0.06, sigma=0.0)])
    assert np.isnan(float(sweep_summary(zero, GatesConfig()).birge_ratio.squeeze()))
