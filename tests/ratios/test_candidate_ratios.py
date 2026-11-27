"""Tests for a candidate's ratio and its fingerprint (tsara.ratios.candidate_ratio, METHODS §5.5).

The evidence, in order of strength:

* fit tables small enough to work out on paper: which member is the median,
  each term of the uncertainty, each gate, the blank row of a candidate no
  member passes, the parent copied from the candidate table;
* a reference written one candidate at a time from §5.5's paragraph, against
  which the table (built for every candidate at once) is compared on random
  fit tables with ties, candidates no member passes and a triggered
  variable's members, which have no z;
* the fingerprint as a pivot of that table;
* the whole chain (baseline, events, candidates, the pipeline's fits) handed
  to it, every candidate and parent present;
* the refusals.
"""

from __future__ import annotations

import itertools
import math
import statistics

import numpy as np
import pandas as pd
import pytest

from tsara.config.analysis import GatesConfig
from tsara.ratios import (
    FIT_COLUMNS,
    GATE_COLUMNS,
    RATIO_COLUMNS,
    RATIO_DTYPES,
    TsaraRatioError,
    candidate_ratios,
    fingerprints,
)
from tsara.ratios.gates import passes

Z95 = 1.959963984540054


def _fit(candidate: str, slope: float, **values: object) -> dict[str, object]:
    """One fit row with defaults that pass every default gate."""
    row: dict[str, object] = dict.fromkeys(FIT_COLUMNS, np.nan)
    row.update(
        event_id=f"{candidate}/e{slope:g}",
        candidate_id=candidate,
        reference="van.ch4",
        numerator="van.c2h6",
        period="native",
        method="york",
        baseline_window=600.0,
        baseline_quantile=0.05,
        exit_multiple=1.0,
        peak_time=pd.Timestamp("2026-01-01 00:10"),
        peak_z=10.0,
        candidate_peak_z=10.0,
        n=50,
        status="OK",
        slope=slope,
        intercept=0.1,
        slope_sigma_widened=0.002,
        mswd=1.0,
        mswd_expected=1.0,
        dof=48,
        signal=10.0,
        plateau_mswd=1.0,
        plateau_dof=98,
        reference_systematic_relative=0.0,
        numerator_systematic_relative=0.0,
    )
    row.update(values)
    return row


def _candidates(*ids: str, parent: str | None = None) -> pd.DataFrame:
    return pd.DataFrame({"candidate_id": list(ids), "parent_candidate_id": [parent] * len(ids)})


def _ratios(rows: list[dict[str, object]], gates: GatesConfig | None = None) -> pd.DataFrame:
    fits = pd.DataFrame(rows)
    ids = list(dict.fromkeys(fits["candidate_id"]))
    return candidate_ratios(fits, _candidates(*ids, parent="w3600s/c0"), gates or GatesConfig())


def test_the_ratio_is_the_median_member_and_sigma_method_the_members_spread() -> None:
    got = _ratios(
        [_fit("c1", 0.050), _fit("c1", 0.040, slope_sigma_widened=0.003), _fit("c1", 0.062)]
    )
    row = got.iloc[0]
    assert row.n_members == 3
    assert row.ratio == 0.050 and row.sigma_fit == 0.002
    assert row.sigma_method == pytest.approx(np.std([0.050, 0.040, 0.062], ddof=1))
    assert row.sigma_combined == pytest.approx(math.hypot(0.002, row.sigma_method))
    assert (row.lower, row.upper) == pytest.approx(
        (0.05 - Z95 * row.sigma_combined, 0.05 + Z95 * row.sigma_combined)
    )
    assert row.event_id == "c1/e0.05"


def test_with_an_even_count_the_median_member_is_the_lower_of_the_two_nearest() -> None:
    got = _ratios([_fit("c1", 0.040), _fit("c1", 0.060), _fit("c1", 0.030), _fit("c1", 0.070)])
    assert got.ratio.iloc[0] == 0.040


def test_one_member_has_no_spread_and_its_interval_is_the_fits_alone() -> None:
    got = _ratios([_fit("c1", 0.05)])
    row = got.iloc[0]
    assert row.n_members == 1 and np.isnan(row.sigma_method)
    assert row.sigma_combined == pytest.approx(0.002)


def test_the_systematic_scales_the_ratio_by_both_axes_relative_terms() -> None:
    got = _ratios(
        [_fit("c1", -0.05, reference_systematic_relative=0.03, numerator_systematic_relative=0.04)]
    )
    row = got.iloc[0]
    assert row.sigma_sys_known
    assert row.sigma_sys == pytest.approx(0.05 * 0.05)  # |b| sqrt(0.03^2 + 0.04^2)
    assert row.sigma_combined == pytest.approx(math.hypot(0.002, 0.0025))


def test_an_unknown_systematic_is_left_out_of_the_combination_and_said_so() -> None:
    got = _ratios([_fit("c1", 0.05, numerator_systematic_relative=np.nan)])
    row = got.iloc[0]
    assert not row.sigma_sys_known and np.isnan(row.sigma_sys)
    assert row.sigma_combined == pytest.approx(0.002)


def test_a_member_with_no_widened_sigma_gives_no_interval() -> None:
    got = _ratios([_fit("c1", 0.05, slope_sigma_widened=np.nan)])
    assert np.isnan(got.sigma_combined.iloc[0]) and np.isnan(got.lower.iloc[0])


@pytest.mark.parametrize(
    ("failing", "why"),
    [
        ({"candidate_peak_z": 4.9}, "height"),
        ({"n": 7}, "points"),
        ({"signal": 1.9}, "signal"),
        ({"mswd": 2.0, "dof": 48}, "one source"),
        # 460 s of a 600 s window: the plume crowds its window (0.77 > 3/4).
        (
            {
                "start_time": pd.Timestamp("2026-01-01 00:06:10"),
                "end_time": pd.Timestamp("2026-01-01 00:13:50"),
            },
            "crowding",
        ),
        ({"status": "MISSING_SIGMA"}, "no line"),
    ],
)
def test_each_gate_drops_a_member_that_fails_it(failing: dict[str, object], why: str) -> None:
    got = _ratios([_fit("c1", 0.050), _fit("c1", 0.040, **failing), _fit("c1", 0.060)])
    assert got.n_members.iloc[0] == 2, why
    assert got.ratio.iloc[0] == 0.050


def test_the_one_source_gate_reads_the_mswd_against_what_one_source_is_expected_to_read() -> None:
    # MSWD 1.6 on 48 dof fails the 99 % gate (1.47) against 1, and passes against E = 1.2.
    passes = _ratios([_fit("c1", 0.05, mswd=1.6, mswd_expected=1.2)])
    fails = _ratios([_fit("c1", 0.05, mswd=1.6, mswd_expected=1.0)])
    assert passes.n_members.iloc[0] == 1 and fails.n_members.iloc[0] == 0


def test_the_plateau_is_off_by_default_and_gates_when_swept_on() -> None:
    rows = [_fit("c1", 0.05, plateau_mswd=3.0), _fit("c1", 0.04)]
    got = _ratios(rows, GatesConfig(plateau_probability=(0.99, 1.0)))
    assert list(got.plateau_probability) == [0.99, 1.0]
    assert list(got.n_members) == [1, 2]


def test_every_gate_setting_is_one_row_the_last_gate_varying_fastest() -> None:
    gates = GatesConfig(min_peak_z=(3.0, 5.0), min_signal=(0.0, 2.0))
    got = _ratios([_fit("c1", 0.05)], gates)
    assert list(zip(got.min_peak_z, got.min_signal, strict=True)) == [
        (3.0, 0.0),
        (3.0, 2.0),
        (5.0, 0.0),
        (5.0, 2.0),
    ]


def test_a_candidate_no_member_passes_still_has_its_row() -> None:
    got = _ratios([_fit("c1", 0.05), _fit("c2", 0.05, candidate_peak_z=2.0, peak_z=2.0)])
    blank = got[got.candidate_id == "c2"].iloc[0]
    assert blank.n_members == 0 and np.isnan(blank.ratio) and pd.isna(blank.event_id)
    assert blank.candidate_peak_z == 2.0


def test_each_numerator_estimator_and_clock_is_its_own_ratio() -> None:
    rows = [
        _fit("c1", 0.05),
        _fit("c1", 0.30, numerator="van.co"),
        _fit("c1", 0.049, method="ols"),
        _fit("c1", 0.051, period="10s"),
    ]
    got = _ratios(rows)
    assert len(got) == 4 and (got.n_members == 1).all()


def test_each_row_names_its_candidates_parent() -> None:
    got = _ratios([_fit("van.ch4/w600s/c1", 0.05)])
    assert got.parent_candidate_id.iloc[0] == "w3600s/c0"


def test_the_table_has_its_declared_columns_and_types() -> None:
    got = _ratios([_fit("c1", 0.05)])
    assert list(got.columns) == list(RATIO_COLUMNS)
    assert {c: str(t) for c, t in got.dtypes.items()} == {
        c: str(pd.Series(dtype=t).dtype) for c, t in RATIO_DTYPES.items()
    }
    empty = candidate_ratios(pd.DataFrame(columns=list(FIT_COLUMNS)), _candidates(), GatesConfig())
    assert empty.empty and list(empty.columns) == list(RATIO_COLUMNS)


def test_refusals() -> None:
    fits = pd.DataFrame([_fit("c1", 0.05)])
    with pytest.raises(TsaraRatioError, match="candidate table does not hold"):
        candidate_ratios(fits, _candidates("c9"), GatesConfig())
    with pytest.raises(TsaraRatioError, match="confidence_level"):
        candidate_ratios(fits, _candidates("c1"), GatesConfig(), confidence_level=1.0)


def test_a_fingerprint_is_each_candidates_ratios_across_numerators() -> None:
    rows = [
        _fit("c1", 0.05),
        _fit("c1", 0.30, numerator="van.co"),
        _fit("c2", 0.00),
        _fit("c2", 0.10, numerator="van.co"),
        _fit("c1", 0.049, method="ols"),
    ]
    ratios = _ratios(rows)
    got = fingerprints(ratios, method="york")
    assert list(got.columns) == ["van.c2h6", "van.co"]
    assert list(got.index) == ["c1", "c2"]
    np.testing.assert_array_equal(got.to_numpy(), [[0.05, 0.30], [0.00, 0.10]])
    sigma = fingerprints(ratios, value="sigma_combined", method="york")
    assert sigma.loc["c1", "van.co"] == pytest.approx(0.002)
    with pytest.raises(TsaraRatioError, match="estimators, pair clocks or gate settings"):
        fingerprints(ratios)
    with pytest.raises(TsaraRatioError, match="no column 'colour'"):
        fingerprints(ratios, colour="red")


def test_gate_columns_are_the_gates_of_the_configuration() -> None:
    assert set(GATE_COLUMNS) == set(GatesConfig.model_fields)


def test_a_gate_at_its_threshold_lets_the_member_through() -> None:
    got = _ratios([_fit("c1", 0.05, n=8, signal=2.0, candidate_peak_z=5.0)])
    assert got.n_members.iloc[0] == 1
    # 450 s of a 600 s window is exactly 3/4.
    at_three_quarters = _fit(
        "c1",
        0.05,
        start_time=pd.Timestamp("2026-01-01 00:06:15"),
        end_time=pd.Timestamp("2026-01-01 00:13:45"),
    )
    assert _ratios([at_three_quarters]).n_members.iloc[0] == 1


def test_the_crowding_gate_reads_the_members_own_window() -> None:
    """The same 460 s member crowds a 600 s window and fits a 3600 s one; with no
    interval to read (a fit made without one) it is not judged, as a fit with no MSWD
    is not judged by the one-source gate."""
    span = {
        "start_time": pd.Timestamp("2026-01-01 00:06:10"),
        "end_time": pd.Timestamp("2026-01-01 00:13:50"),
    }
    assert _ratios([_fit("c1", 0.05, **span)]).n_members.iloc[0] == 0
    assert _ratios([_fit("c1", 0.05, baseline_window=3600.0, **span)]).n_members.iloc[0] == 1
    off = _ratios([_fit("c1", 0.05, **span)], GatesConfig(max_window_fill=(float("inf"),)))
    assert off.n_members.iloc[0] == 1
    assert _ratios([_fit("c1", 0.05)]).n_members.iloc[0] == 1


def test_a_fit_with_no_mswd_to_read_is_reported_ungated() -> None:
    """§5.3: least squares where York could not weigh (or was not run) has no MSWD and
    no plateau; the one-source and plateau gates cannot judge it, and its row says so."""
    row = _fit("c1", 0.05, method="ols", mswd=np.nan, mswd_expected=1.0, plateau_mswd=np.nan)
    got = _ratios([row], GatesConfig(plateau_probability=(0.99,)))
    assert got.n_members.iloc[0] == 1 and np.isnan(got.mswd.iloc[0])


def _reference(
    fits: pd.DataFrame, candidates: pd.DataFrame, gates: GatesConfig, level: float
) -> pd.DataFrame:
    """§5.5 one ratio at a time: the slow form the table must agree with."""
    key = ["candidate_id", "reference", "numerator", "method", "period"]
    parents = dict(zip(candidates.candidate_id, candidates.parent_candidate_id, strict=True))
    z = statistics.NormalDist().inv_cdf(0.5 + level / 2)
    rows = []
    for values in itertools.product(*(getattr(gates, name) for name in GATE_COLUMNS)):
        setting = dict(zip(GATE_COLUMNS, values, strict=True))
        ok = passes(fits, setting)
        for ratio in dict.fromkeys(fits[key].itertuples(index=False, name=None)):
            mine = np.all(fits[key].to_numpy() == np.array(ratio, dtype=object), axis=1)
            members = fits[mine]
            zs = members.peak_z.to_numpy()
            tallest = 0 if np.all(np.isnan(zs)) else int(np.nanargmax(zs))
            row: dict[str, object] = dict(zip(key, ratio, strict=True)) | setting
            row |= {
                "parent_candidate_id": parents[ratio[0]],
                "baseline_window": members.baseline_window.iloc[0],
                "candidate_peak_z": members.candidate_peak_z.max(),
                "peak_time": members.peak_time.iloc[tallest],
                "confidence_level": level,
            }
            kept = fits[mine & ok]
            row["n_members"] = len(kept)
            if kept.empty:
                rows.append(row | {"n": 0, "dof": 0, "sigma_sys_known": False})
                continue
            slopes = list(kept.slope)
            m = kept.iloc[
                sorted(range(len(slopes)), key=slopes.__getitem__)[(len(slopes) - 1) // 2]
            ]
            spread = statistics.stdev(slopes) if len(slopes) > 1 else math.nan
            known = math.isfinite(m.reference_systematic_relative) and math.isfinite(
                m.numerator_systematic_relative
            )
            sys = (
                abs(m.slope)
                * math.hypot(m.reference_systematic_relative, m.numerator_systematic_relative)
                if known
                else math.nan
            )
            terms = [t for t in (m.slope_sigma_widened, spread, sys) if math.isfinite(t)]
            combined = math.sqrt(sum(t * t for t in terms))
            if not math.isfinite(m.slope_sigma_widened):
                combined = math.nan
            rows.append(
                row
                | {
                    "event_id": m.event_id,
                    "ratio": m.slope,
                    "intercept": m.intercept,
                    "n": m.n,
                    "mswd": m.mswd,
                    "dof": m.dof,
                    "sigma_fit": m.slope_sigma_widened,
                    "sigma_method": spread,
                    "sigma_sys": sys,
                    "sigma_sys_known": known,
                    "sigma_combined": combined,
                    "lower": m.slope - z * combined,
                    "upper": m.slope + z * combined,
                }
            )
    return pd.DataFrame(rows, columns=list(RATIO_COLUMNS)).astype(RATIO_DTYPES)


def _random_fits(seed: int) -> pd.DataFrame:
    """Members of a few candidates, several numerators, estimators and clocks, rows shuffled.

    Slopes are drawn from a coarse set, so members tie; the gates' numbers
    straddle their thresholds, so some members pass and some candidates have
    none; one candidate is a triggered variable's, with no z; a few fits have
    no line, no widened sigma or no systematic term.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(6):
        no_z = c == 5
        for numerator, method, period in itertools.product(
            ("van.c2h6", "van.co"), ("york", "ols"), ("native", "5s")
        ):
            for q, x in itertools.product((0.01, 0.05), (1.0, 2.0)):
                if rng.random() < 0.2:
                    continue
                peak = pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=int(rng.integers(600)))
                # 100, 400 or 500 s of the 600 s window: the last crowds it.
                duration = float(rng.choice([100.0, 400.0, 500.0]))
                rows.append(
                    _fit(
                        f"c{c}",
                        float(rng.choice([0.03, 0.04, 0.05, 0.05, 0.06]) + rng.choice([0, 1e-3])),
                        numerator=numerator,
                        method=method,
                        period=period,
                        baseline_quantile=q,
                        exit_multiple=x,
                        event_id=f"c{c}/{numerator}/{method}/{period}/q{q}/x{x}",
                        peak_z=np.nan if no_z else float(rng.choice([4.0, 6.0, 9.0, 9.0])),
                        candidate_peak_z=np.nan if no_z else 9.0,
                        peak_time=peak,
                        start_time=peak - pd.Timedelta(seconds=duration / 2),
                        end_time=peak + pd.Timedelta(seconds=duration / 2),
                        n=int(rng.integers(5, 40)),
                        mswd=float(rng.uniform(0.3, 3.0)),
                        status="OK" if rng.random() > 0.05 else "MISSING_SIGMA",
                        slope_sigma_widened=np.nan
                        if rng.random() < 0.05
                        else float(rng.uniform(1e-3, 1e-2)),
                        reference_systematic_relative=np.nan if rng.random() < 0.3 else 0.01,
                        numerator_systematic_relative=0.02,
                    )
                )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize("seed", range(8))
def test_the_table_is_the_reference_one_ratio_at_a_time(seed: int) -> None:
    fits = _random_fits(seed)
    candidates = pd.DataFrame(
        {"candidate_id": [f"c{c}" for c in range(6)], "parent_candidate_id": [None, "p0"] * 3}
    )
    gates = GatesConfig(
        min_peak_z=(0.0, 5.0),
        min_points=(8, 20),
        mswd_probability=(0.9, 0.99),
        max_window_fill=(0.75, float("inf")),
    )
    got = candidate_ratios(fits, candidates, gates, confidence_level=0.9)
    pd.testing.assert_frame_equal(
        got, _reference(fits, candidates, gates, 0.9), check_exact=False, rtol=1e-12
    )
    # The cases the reference is there for occur.
    assert (got.n_members == 0).any() and (got.n_members > 1).any()
    assert got.candidate_peak_z.isna().any() and got.sigma_sys_known.any()
    # The crowding gate drops members: fewer pass at 3/4 than with it off.
    crowded = got.max_window_fill == 0.75
    assert got.n_members[crowded].sum() < got.n_members[~crowded].sum()
