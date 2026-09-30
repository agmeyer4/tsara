"""Tests for the baseline state (tsara.baseline.state) and the methods it runs.

Evidence, per METHODS §11.1: pencil-checkable fixtures (a flat background,
a step, a masked stretch), the engine's own result as the reference for the
state's columns, the generator's true background as ground truth, and the
three-call `from_field` flow end to end through the real binner.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

from tsara.align import bin_streams_onto_cells
from tsara.baseline import (
    TsaraBaselineError,
    baseline_state,
    baseline_states,
    rolling_quantile,
    window_cells,
)
from tsara.config.analysis import BaselineConfig
from tsara.core.naming import BASELINE_STAGE
from tsara.core.support import CellBounds, stream_cells
from tsara.core.timebase import SECOND_NS as SECOND


def cells(start_s: float, width_s: float, n: int, step_s: float | None = None) -> CellBounds:
    step = width_s if step_s is None else step_s
    start = (np.arange(n, dtype=np.int64) * int(step * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def make_stream(
    bounds: CellBounds,
    variables: dict[str, np.ndarray],
    *,
    attrs: dict[str, dict[str, object]] | None = None,
    cell_methods: str = "time: point",
) -> xr.Dataset:
    """A stream with CF cells; every variable a gas unless its attrs say otherwise."""
    given = attrs or {}
    dataset = xr.Dataset(
        data_vars={
            name: (
                "time",
                values,
                {"units": "ppb", "role": "gas", "field": name, **given.get(name, {})},
            )
            for name, values in variables.items()
        },
        coords={
            "time": bounds.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
        attrs={"tsara_stage": "synthetic", "tsara_support_label": "mid"},
    )
    dataset["time"].attrs["bounds"] = "time_bnds"
    for name in variables:
        if not name.startswith("sigma_"):
            dataset[name].attrs.setdefault("cell_methods", cell_methods)
    return dataset


def config(**overrides: object) -> BaselineConfig:
    base: dict[str, object] = {"windows": ("2min", "10min"), "quantiles": (0.05, 0.5)}
    base.update(overrides)
    return BaselineConfig.model_validate(base)


@pytest.fixture()
def drive() -> xr.Dataset:
    """Twenty minutes of 1 s cells every 2 s: a flat background, noise, one plume, a masked gap."""
    rng = np.random.default_rng(3)
    n = 600
    bounds = cells(0.0, 1.0, n, step_s=2.0)
    t = bounds.midpoint_ns / SECOND
    values = 1900.0 + rng.normal(0, 0.7, n) + 80 * np.exp(-0.5 * ((t - 600) / 20) ** 2)
    values[300:320] = np.nan
    return make_stream(
        bounds,
        {
            "ch4": values,
            "sigma_rand_ch4": np.full(n, 0.7),
            "sigma_sys_ch4": 0.01 * np.abs(values),
            "co2": 420.0 + rng.normal(0, 0.1, n),
            "wind_dir": rng.uniform(0, 360, n),
        },
        attrs={
            "wind_dir": {"role": "met", "circular": 1, "units": "degrees"},
            "sigma_rand_ch4": {"uncertainty_component": "random"},
            "sigma_sys_ch4": {"uncertainty_component": "systematic"},
            "ch4": {
                "uncertainty_provenance_random": "declared",
                "uncertainty_provenance_systematic": "declared",
            },
        },
    )


# ---------------------------------------------------------------------------
# The shape decided (§6.2, §6.6)
# ---------------------------------------------------------------------------


def test_the_state_lives_on_the_streams_own_cells_with_the_sweep_as_dimensions(
    drive: xr.Dataset,
) -> None:
    state = baseline_state(drive, instrument="van", baseline=config())
    assert state.attrs["tsara_stage"] == BASELINE_STAGE
    assert state.attrs["tsara_instrument"] == "van"
    assert state.attrs["tsara_support_label"] == "mid"  # the stream's own attrs carried
    assert np.array_equal(state["time"].values, drive["time"].values)
    assert np.array_equal(state["time_bnds"].values, drive["time_bnds"].values)
    assert state["time"].attrs["bounds"] == "time_bnds"
    assert state["baseline_window"].values.tolist() == [120.0, 600.0]
    assert state["baseline_window"].attrs["units"] == "s"
    assert state["baseline_quantile"].values.tolist() == [0.05, 0.5]
    assert state.attrs["tsara_baseline_windows"] == "2min, 10min"
    assert state["baseline_ch4"].dims == ("time", "baseline_window", "baseline_quantile")
    assert state["n_readings_window_ch4"].dims == ("time", "baseline_window")
    assert sorted(map(str, state.data_vars)) == sorted(
        [
            "ch4",
            "sigma_rand_ch4",
            "sigma_sys_ch4",
            "baseline_ch4",
            "enhancement_ch4",
            "n_readings_window_ch4",
            "coverage_window_ch4",
            "sigma_rand_baseline_ch4",
            "sigma_sys_baseline_ch4",
            "sigma_rand_enhancement_ch4",
            "sigma_sys_enhancement_ch4",
            "co2",
            "baseline_co2",
            "enhancement_co2",
            "n_readings_window_co2",
            "coverage_window_co2",
            "sigma_rand_baseline_co2",
        ]
    )


def test_the_default_selection_is_gas_and_not_circular_or_a_companion(drive: xr.Dataset) -> None:
    state = baseline_state(drive, instrument="van", baseline=config())
    assert "baseline_wind_dir" not in state.data_vars
    assert "baseline_sigma_rand_ch4" not in state.data_vars
    only = baseline_state(drive, instrument="van", baseline=config(), variables=["co2"])
    assert "baseline_co2" in only.data_vars and "baseline_ch4" not in only.data_vars


def test_the_baseline_carries_no_cell_method_and_the_enhancement_inherits_the_readings(
    drive: xr.Dataset,
) -> None:
    state = baseline_state(drive, instrument="van", baseline=config())
    assert "cell_methods" not in state["baseline_ch4"].attrs
    assert state["enhancement_ch4"].attrs["cell_methods"] == "time: point"
    attrs = state["baseline_ch4"].attrs
    assert attrs["tsara_baseline_method"] == "rolling_quantile"
    assert attrs["tsara_baseline_membership"] == "overlap"
    assert attrs["tsara_baseline_min_readings"].tolist() == [20, 2]
    assert attrs["units"] == "ppb" and attrs["field"] == "ch4"
    assert "cell_methods" not in state["n_readings_window_ch4"].attrs


# ---------------------------------------------------------------------------
# The baseline is the engine's quantile, blanked by the rules (§6.3, §6.4)
# ---------------------------------------------------------------------------


def test_the_baseline_is_the_engine_result_blanked_by_the_count_rule(drive: xr.Dataset) -> None:
    state = baseline_state(drive, instrument="van", baseline=config())
    readings = stream_cells(drive, "van")
    for w, window_s in enumerate((120.0, 600.0)):
        rolled = rolling_quantile(
            readings,
            drive["ch4"].values,
            window_cells(readings, int(window_s * SECOND)),
            [0.05, 0.5],
        )
        assert np.array_equal(state["n_readings_window_ch4"].values[:, w], rolled.n_readings)
        assert np.allclose(state["coverage_window_ch4"].values[:, w], rolled.coverage)
        for q, minimum in enumerate((20, 2)):
            want = np.where(rolled.n_readings < minimum, np.nan, rolled.values[:, q])
            assert np.array_equal(state["baseline_ch4"].values[:, w, q], want, equal_nan=True)
            want_sigma = np.where(rolled.n_readings < minimum, np.nan, rolled.sigma[:, q])
            assert np.array_equal(
                state["sigma_rand_baseline_ch4"].values[:, w, q], want_sigma, equal_nan=True
            )


def test_the_enhancement_is_the_reading_minus_the_baseline_and_is_never_clipped(
    drive: xr.Dataset,
) -> None:
    state = baseline_state(drive, instrument="van", baseline=config())
    baseline = state["baseline_ch4"].values
    want = drive["ch4"].values[:, None, None] - baseline
    assert np.array_equal(state["enhancement_ch4"].values, want, equal_nan=True)
    median = state["enhancement_ch4"].values[:, 1, 1]
    assert np.nanmin(median) < 0  # noise below the median is negative, and stays so
    assert np.isnan(state["enhancement_ch4"].values[300:320]).all()  # a masked reading has none


def test_the_edge_blanks_by_the_count_rule_and_the_record_says_so(drive: xr.Dataset) -> None:
    """The first 2 min window holds ~30 readings: enough for the median, too few for q = 0.01."""
    state = baseline_state(drive, instrument="van", baseline=config(quantiles=(0.01, 0.5)))
    counts = state["n_readings_window_ch4"].values[:, 0]
    assert 25 <= counts[0] <= 32
    assert np.isnan(state["baseline_ch4"].values[0, 0, 0])  # needs 100
    assert np.isfinite(state["baseline_ch4"].values[0, 0, 1])  # needs 2
    # The sampling sigma is blank wherever the baseline is, and nowhere else.
    blank = np.isnan(state["baseline_ch4"].values)
    assert np.array_equal(np.isnan(state["sigma_rand_baseline_ch4"].values), blank)
    fraction = state["baseline_ch4"].attrs["tsara_baseline_blank_fraction"].reshape(2, 2)
    assert fraction[0, 0] == pytest.approx(np.mean(counts < 100))
    assert fraction[0, 1] == 0.0
    assert state["baseline_ch4"].attrs["tsara_baseline_too_wide_fraction"].tolist() == [0.0, 0.0]


def test_a_sweep_point_blank_everywhere_is_warned_once_with_the_count_it_needed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Minute means under a 2 min window: at most three readings, never twenty."""
    minute = make_stream(
        cells(0.0, 60.0, 60), {"ch4": 1900 + np.arange(60.0)}, cell_methods="time: mean"
    )
    with caplog.at_level(logging.WARNING, logger="tsara.baseline.state"):
        state = baseline_state(minute, instrument="ground", baseline=config())
    warnings = [r for r in caplog.records if "blank at every reading" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    # The count rule, with what the windows held beside what the quantile needs.
    assert "ch4: 2min/0.05 (windows held at most 3 readings; 20 needed)" in message
    assert "10min/0.05 (windows held at most 11 readings; 20 needed)" in message
    assert "10min/0.5" not in message and "2min/0.5" not in message
    assert np.isnan(state["baseline_ch4"].values[:, 0, 0]).all()
    assert np.isfinite(state["baseline_ch4"].values[:, 1, 1]).all()


def blank_warning(caplog: pytest.LogCaptureFixture) -> str:
    """The one warning about sweep points blank everywhere, as the reader sees it."""
    found = [r.getMessage() for r in caplog.records if "blank at every reading" in r.getMessage()]
    assert len(found) == 1, found
    return found[0]


def test_the_warning_names_the_width_rule_when_the_count_was_met(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """60 s cells every 30 s under a 30 s window: two or three readings, each twice too wide.

    The count of two is met, so naming the count as the reason -- which the
    first version of this warning did -- would send the reader to the wrong fix.
    """
    stream = make_stream(cells(0.0, 60.0, 40, step_s=30.0), {"ch4": 1900 + np.arange(40.0)})
    with caplog.at_level(logging.WARNING, logger="tsara.baseline.state"):
        baseline_state(
            stream,
            instrument="a",
            baseline=config(windows=("30s", "10min"), quantiles=(0.5,), min_readings=2),
        )
    message = blank_warning(caplog)
    assert "30s/0.5 (every window held a reading at least 2x as wide as itself)" in message
    assert "needed" not in message


def test_the_warning_says_count_or_width_when_windows_failed_one_or_the_other(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Running 60 s means every 5 s, then 1 s readings every 10 s, under a 30 s window.

    The first stretch's windows hold plenty of readings, each too wide; the
    second's hold three narrow ones, fewer than four. Every window fails, not
    all for the same rule.
    """
    running = cells(0.0, 60.0, 50, step_s=5.0)
    sparse = cells(400.0, 1.0, 30, step_s=10.0)
    both = CellBounds(
        start_ns=np.concatenate([running.start_ns, sparse.start_ns]),
        stop_ns=np.concatenate([running.stop_ns, sparse.stop_ns]),
    )
    stream = make_stream(both, {"ch4": 1900 + np.arange(80.0)})
    with caplog.at_level(logging.WARNING, logger="tsara.baseline.state"):
        baseline_state(
            stream,
            instrument="a",
            baseline=config(windows=("30s",), quantiles=(0.5,), min_readings=4),
        )
    message = blank_warning(caplog)
    assert "30s/0.5 (too few readings, or a reading too wide, in every window)" in message


def test_the_warning_states_a_shared_pattern_once_and_names_at_most_eight(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ten variables on one instrument's minute cells: one pattern, eight names, 'and 2 more'."""
    minute = cells(0.0, 60.0, 60)
    stream = make_stream(minute, {f"voc{k}": 1900 + np.arange(60.0) for k in range(10)})
    with caplog.at_level(logging.WARNING, logger="tsara.baseline.state"):
        baseline_state(stream, instrument="can", baseline=config())
    message = blank_warning(caplog)
    assert "voc0, voc1, voc2, voc3, voc4, voc5, voc6, voc7 and 2 more:" in message
    assert "voc8" not in message
    assert message.count("2min/0.05") == 1  # the pattern once, not once per variable


def test_the_warning_separates_variables_whose_patterns_differ(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A variable masked but for every fifth minute is blank at a point the other is not."""
    values = 1900 + np.arange(60.0)
    thin = values.copy()
    thin[np.arange(60) % 5 != 0] = np.nan
    stream = make_stream(cells(0.0, 60.0, 60), {"full": values, "thin": thin})
    with caplog.at_level(logging.WARNING, logger="tsara.baseline.state"):
        baseline_state(stream, instrument="a", baseline=config())
    message = blank_warning(caplog)
    assert " | " in message
    full, thin_part = message.split(" -- ")[1].split(" | ")
    assert full.startswith("full:") and thin_part.startswith("thin:")
    # Every fifth minute leaves a 2 min window one reading: blank for the median too.
    assert "2min/0.5 (windows held at most 1 reading; 2 needed)" in thin_part
    assert "2min/0.5" not in full


def test_the_warning_counts_patterns_past_eight_rather_than_listing_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Asked of the formatter directly: nine distinct patterns are rare enough in a stream
    that building one would test the fixture more than the rule."""
    from tsara.baseline.state import _BlankPoint, _warn_blank_everywhere

    blank = [_BlankPoint(f"v{k}", f"{k + 1}min/0.05", "count", 20, 3) for k in range(9)]
    with caplog.at_level(logging.WARNING, logger="tsara.baseline.state"):
        _warn_blank_everywhere("a", blank)
    message = blank_warning(caplog)
    assert "and 1 more pattern(s)" in message
    assert "v8:" not in message


def test_an_adopted_baseline_blank_everywhere_is_warned_as_adopted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The donor cannot make a 0.5th percentile from a 2 min window of 121 readings (it needs
    200), so the canister's adopted baseline is blank there, and says why."""
    dense, sparse = sparse_and_dense()
    settings = config(
        quantiles=(0.005, 0.5),
        methods={"can.benzene_can": {"method": "from_field", "instrument": "ptr"}},
    )
    donor = baseline_state(dense, instrument="ptr", baseline=settings)
    joined = bin_streams_onto_cells(
        {"ptr": donor}, stream_cells(sparse, "can"), [("ptr", "baseline_benzene")]
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tsara.baseline.state"):
        baseline_state(
            sparse, instrument="can", baseline=settings, provided={"benzene_can": joined}
        )
    message = blank_warning(caplog)
    assert "benzene_can: 2min/0.005 (the adopted baseline is blank there)" in message


def test_a_reading_too_wide_for_the_window_blanks_it_with_the_reason_recorded() -> None:
    """60 s cells stepping every 30 s under a 30 s window: enough readings, each twice too wide."""
    stream = make_stream(cells(0.0, 60.0, 40, step_s=30.0), {"ch4": 1900 + np.arange(40.0)})
    state = baseline_state(
        stream,
        instrument="a",
        baseline=config(windows=("30s", "10min"), quantiles=(0.5,), min_readings=2),
    )
    assert (state["n_readings_window_ch4"].values[:, 0] >= 2).all()
    assert np.isnan(state["baseline_ch4"].values[:, 0, 0]).all()
    assert state["baseline_ch4"].attrs["tsara_baseline_too_wide_fraction"].tolist() == [1.0, 0.0]
    assert np.isfinite(state["baseline_ch4"].values[:, 1, 0]).all()


# ---------------------------------------------------------------------------
# Uncertainties (§6.7)
# ---------------------------------------------------------------------------


def test_the_enhancements_sigmas_follow_the_same_instrument_rules(drive: xr.Dataset) -> None:
    state = baseline_state(drive, instrument="van", baseline=config())
    reading_random = drive["sigma_rand_ch4"].values[:, None, None]
    reading_systematic = drive["sigma_sys_ch4"].values[:, None, None]
    baseline_random = state["sigma_rand_baseline_ch4"].values
    enhancement = state["enhancement_ch4"].values
    want_random = np.where(
        np.isfinite(enhancement), np.sqrt(reading_random**2 + baseline_random**2), np.nan
    )
    assert np.allclose(state["sigma_rand_enhancement_ch4"].values, want_random, equal_nan=True)
    assert state["sigma_rand_enhancement_ch4"].attrs["tsara_sigma_rule"] == "quadrature"
    want_systematic = (
        reading_systematic * np.abs(enhancement) / np.abs(drive["ch4"].values)[:, None, None]
    )
    assert np.allclose(state["sigma_sys_enhancement_ch4"].values, want_systematic, equal_nan=True)
    assert (
        state["sigma_sys_enhancement_ch4"].attrs["tsara_sigma_rule"].startswith("same instrument")
    )
    # The systematic figure of an enhancement is far below the reading's: a
    # 1 % gain error on 1900 ppb is 19 ppb on the reading and 0.8 ppb on an
    # 80 ppb enhancement.
    finite = np.isfinite(enhancement)
    assert np.nanmax(state["sigma_sys_enhancement_ch4"].values[finite]) < 2.0
    assert state["sigma_rand_baseline_ch4"].attrs["uncertainty_provenance"] == "empirical"
    assert "floor" in state["sigma_rand_baseline_ch4"].attrs["tsara_sigma_assumption"]
    assert state["sigma_sys_baseline_ch4"].attrs["uncertainty_provenance"] == "declared"


def test_the_baselines_systematic_sigma_is_the_readings_at_the_quantile(drive: xr.Dataset) -> None:
    state = baseline_state(drive, instrument="van", baseline=config())
    readings = stream_cells(drive, "van")
    rolled = rolling_quantile(
        readings,
        drive["ch4"].values,
        window_cells(readings, 600 * SECOND),
        [0.05, 0.5],
        carry=drive["sigma_sys_ch4"].values,
    )
    assert rolled.carried is not None
    got = state["sigma_sys_baseline_ch4"].values[:, 1, :]
    want = np.where(rolled.n_readings[:, None] < np.array([20, 2]), np.nan, rolled.carried)
    assert np.array_equal(got, want, equal_nan=True)


def test_a_reading_of_exactly_zero_has_no_systematic_enhancement_sigma() -> None:
    values = np.full(120, 5.0)
    values[60] = 0.0
    stream = make_stream(
        cells(0.0, 1.0, 120),
        {"ch4": values, "sigma_sys_ch4": np.full(120, 0.05)},
        attrs={"sigma_sys_ch4": {"uncertainty_component": "systematic"}},
    )
    state = baseline_state(stream, instrument="a", baseline=config(quantiles=(0.5,)))
    assert np.isnan(state["sigma_sys_enhancement_ch4"].values[60]).all()
    assert np.isfinite(state["sigma_sys_enhancement_ch4"].values[59]).all()


def test_without_a_declared_sigma_no_enhancement_sigma_is_invented() -> None:
    stream = make_stream(cells(0.0, 1.0, 300), {"ch4": 1900 + np.zeros(300)})
    state = baseline_state(stream, instrument="a", baseline=config())
    assert "sigma_rand_enhancement_ch4" not in state.data_vars
    assert "sigma_sys_enhancement_ch4" not in state.data_vars
    assert "sigma_sys_baseline_ch4" not in state.data_vars
    assert "sigma_rand_baseline_ch4" in state.data_vars  # the sampling figure needs no declaration
    # ... and the enhancement says so, rather than leaving the absence to be guessed (§6.7).
    attrs = state["enhancement_ch4"].attrs
    assert attrs["uncertainty_provenance_random"] == "unknown"
    assert attrs["uncertainty_provenance_systematic"] == "unknown"


def test_a_component_declared_zero_is_carried_as_zero_not_unknown() -> None:
    """A budget that omits a component states it is negligible; that is not ignorance (§2.4)."""
    stream = make_stream(
        cells(0.0, 1.0, 300),
        {"ch4": 1900 + np.zeros(300), "sigma_rand_ch4": np.full(300, 0.7)},
        attrs={
            "sigma_rand_ch4": {"uncertainty_component": "random"},
            "ch4": {
                "uncertainty_provenance_random": "declared",
                "uncertainty_provenance_systematic": "zero",
            },
        },
    )
    state = baseline_state(stream, instrument="a", baseline=config())
    attrs = state["enhancement_ch4"].attrs
    assert attrs["uncertainty_provenance_systematic"] == "zero"
    # The random component has its column, which carries its own provenance.
    assert "uncertainty_provenance_random" not in attrs
    assert state["sigma_rand_enhancement_ch4"].attrs["uncertainty_provenance"] == "declared"


# ---------------------------------------------------------------------------
# Ground truth: the generator's true background
# ---------------------------------------------------------------------------


def test_the_median_tracks_the_background_and_the_low_quantile_sits_a_noise_offset_below() -> None:
    """Flat air with 0.7 ppb noise: the 10 min median is the background to 0.1 ppb; the 5th
    percentile sits 1.645 sigma below it, the quantile offset Phase 6 corrects thresholds for.

    Flat on purpose: a background that slopes within the window moves its low
    quantile to the window's low end, which is the window selection of §6.1
    at work rather than an error, and would hide the offset being measured.
    """
    rng = np.random.default_rng(11)
    n = 3600
    bounds = cells(0.0, 1.0, n)
    truth = np.full(n, 1900.0)
    stream = make_stream(bounds, {"ch4": truth + rng.normal(0, 0.7, n)})
    state = baseline_state(stream, instrument="a", baseline=config(windows=("10min",)))
    inside = slice(300, 3300)
    median = state["baseline_ch4"].values[inside, 0, 1] - truth[inside]
    low = state["baseline_ch4"].values[inside, 0, 0] - truth[inside]
    assert abs(np.mean(median)) < 0.1
    assert np.mean(low) == pytest.approx(-1.645 * 0.7, abs=0.15)


# ---------------------------------------------------------------------------
# Methods: constant and from_field, the latter as the three calls (§6.5)
# ---------------------------------------------------------------------------


def test_a_constant_baseline_makes_the_enhancement_the_concentration(drive: xr.Dataset) -> None:
    state = baseline_state(
        drive,
        instrument="van",
        baseline=config(methods={"van.ch4": {"method": "constant", "value": 0.0}}),
        variables=["ch4"],
    )
    assert (state["baseline_ch4"].values == 0.0).all()
    assert np.array_equal(
        state["enhancement_ch4"].values,
        np.broadcast_to(drive["ch4"].values[:, None, None], (600, 2, 2)),
        equal_nan=True,
    )
    assert state["baseline_ch4"].attrs["tsara_baseline_method"] == "constant"
    assert state["baseline_ch4"].attrs["tsara_baseline_value"] == 0.0
    assert "n_readings_window_ch4" not in state.data_vars
    assert "sigma_rand_baseline_ch4" not in state.data_vars
    # Nothing cancels: the enhancement carries the reading's whole systematic sigma.
    want = np.broadcast_to(drive["sigma_sys_ch4"].values[:, None, None], (600, 2, 2))
    got = state["sigma_sys_enhancement_ch4"].values
    assert np.allclose(got[np.isfinite(got)], want[np.isfinite(got)])
    assert state["sigma_sys_enhancement_ch4"].attrs["tsara_sigma_rule"] == "reading"
    assert np.allclose(
        state["sigma_rand_enhancement_ch4"].values[np.isfinite(got)],
        np.broadcast_to(drive["sigma_rand_ch4"].values[:, None, None], (600, 2, 2))[
            np.isfinite(got)
        ],
    )


def sparse_and_dense() -> tuple[xr.Dataset, xr.Dataset]:
    """A dense 1 s analyzer and a canister-like sampler of the same field, one hour."""
    rng = np.random.default_rng(5)
    dense_cells = cells(0.0, 1.0, 3600)
    dense = make_stream(
        dense_cells,
        {"benzene": 0.5 + rng.normal(0, 0.05, 3600), "sigma_sys_benzene": np.full(3600, 0.02)},
        attrs={"sigma_sys_benzene": {"uncertainty_component": "systematic"}},
    )
    sparse_cells = cells(100.0, 15.0, 6, step_s=530.0)
    sparse = make_stream(
        sparse_cells,
        {"benzene_can": 0.55 + rng.normal(0, 0.05, 6), "sigma_sys_benzene_can": np.full(6, 0.03)},
        attrs={
            "benzene_can": {"field": "benzene"},
            "sigma_sys_benzene_can": {"uncertainty_component": "systematic"},
        },
    )
    return dense, sparse


def test_from_field_adopts_the_donors_baseline_joined_onto_these_cells() -> None:
    dense, sparse = sparse_and_dense()
    settings = config(methods={"can.benzene_can": {"method": "from_field", "instrument": "ptr"}})
    # 1. Roll the donor.  2. Join its swept baseline onto the adopter's cells.  3. Roll the adopter.
    donor = baseline_state(dense, instrument="ptr", baseline=settings)
    joined = bin_streams_onto_cells(
        {"ptr": donor}, stream_cells(sparse, "can"), [("ptr", "baseline_benzene")]
    )
    state = baseline_state(
        sparse, instrument="can", baseline=settings, provided={"benzene_can": joined}
    )
    base = state["baseline_benzene_can"]
    assert np.array_equal(base.values, joined["baseline_benzene"].values, equal_nan=True)
    assert base.attrs["tsara_baseline_method"] == "from_field"
    assert base.attrs["tsara_baseline_from"] == "ptr"
    assert base.attrs["tsara_support_transform"] == "averaged"  # the join's record travels
    assert base.attrs["tsara_instrument"] == "ptr"
    assert "tsara_baseline_min_readings" not in base.attrs  # the windows are the donor's
    assert base.attrs["tsara_baseline_blank_fraction"].tolist() == [0.0] * 4
    assert "n_readings_window_benzene_can" not in state.data_vars
    # The donor's sigmas came through the join, and the enhancement's systematic
    # sigma adds the two instruments' terms in quadrature (§6.7).
    assert np.array_equal(
        state["sigma_sys_baseline_benzene_can"].values,
        joined["sigma_sys_baseline_benzene"].values,
        equal_nan=True,
    )
    want = np.sqrt(0.03**2 + joined["sigma_sys_baseline_benzene"].values ** 2)
    got = state["sigma_sys_enhancement_benzene_can"].values
    assert np.allclose(got, want, equal_nan=True)
    assert (
        state["sigma_sys_enhancement_benzene_can"].attrs["tsara_sigma_rule"]
        == "quadrature with the donor's"
    )
    enhancement = sparse["benzene_can"].values[:, None, None] - base.values
    assert np.array_equal(state["enhancement_benzene_can"].values, enhancement, equal_nan=True)


def test_from_field_without_a_provided_product_names_the_three_calls() -> None:
    dense, sparse = sparse_and_dense()
    settings = config(methods={"can.benzene_can": {"method": "from_field", "instrument": "ptr"}})
    with pytest.raises(TsaraBaselineError, match="no product was provided.*bin_streams_onto_cells"):
        baseline_state(sparse, instrument="can", baseline=settings)


def test_from_field_refuses_a_product_that_is_not_the_join_described() -> None:
    dense, sparse = sparse_and_dense()
    settings = config(methods={"can.benzene_can": {"method": "from_field", "instrument": "ptr"}})
    donor = baseline_state(dense, instrument="ptr", baseline=settings)
    good = bin_streams_onto_cells(
        {"ptr": donor}, stream_cells(sparse, "can"), [("ptr", "baseline_benzene")]
    )

    def attempt(product: xr.Dataset) -> None:
        baseline_state(
            sparse, instrument="can", baseline=settings, provided={"benzene_can": product}
        )

    with pytest.raises(TsaraBaselineError, match="tsara_stage 'baseline', not 'binned'"):
        attempt(donor)
    with pytest.raises(TsaraBaselineError, match="does not sit on this stream's cells"):
        attempt(
            bin_streams_onto_cells(
                {"ptr": donor}, cells(100.0, 15.0, 6, step_s=531.0), [("ptr", "baseline_benzene")]
            )
        )
    other_sweep = baseline_state(
        dense, instrument="ptr", baseline=config(windows=("5min", "10min"))
    )
    with pytest.raises(TsaraBaselineError, match="rolled at baseline_window = \\[300.0, 600.0\\]"):
        attempt(
            bin_streams_onto_cells(
                {"ptr": other_sweep}, stream_cells(sparse, "can"), [("ptr", "baseline_benzene")]
            )
        )
    with pytest.raises(TsaraBaselineError, match="holds 0 baseline column"):
        attempt(good.rename({"baseline_benzene": "other_benzene"}))
    two = good.assign(baseline_again=good["baseline_benzene"])
    with pytest.raises(TsaraBaselineError, match="holds 2 baseline column"):
        attempt(two)
    no_sweep = good.drop_vars("baseline_quantile")
    with pytest.raises(TsaraBaselineError, match="carries no 'baseline_quantile'"):
        attempt(no_sweep)


# ---------------------------------------------------------------------------
# Refusals and the collection form
# ---------------------------------------------------------------------------


def test_naming_what_cannot_be_rolled_is_refused_by_name(drive: xr.Dataset) -> None:
    with pytest.raises(TsaraBaselineError, match="no variable 'sf6'"):
        baseline_state(drive, instrument="van", baseline=config(), variables=["sf6"])
    with pytest.raises(TsaraBaselineError, match="circular"):
        baseline_state(drive, instrument="van", baseline=config(), variables=["wind_dir"])
    with pytest.raises(TsaraBaselineError, match="No variables named"):
        baseline_state(drive, instrument="van", baseline=config(), variables=[])
    flat = drive.assign(matrix=(("time", "nv"), np.zeros((drive.sizes["time"], 2))))
    with pytest.raises(TsaraBaselineError, match="dimensions"):
        baseline_state(flat, instrument="van", baseline=config(), variables=["matrix"])
    met = drive[["wind_dir"]].copy()
    met.coords["time_bnds"] = drive["time_bnds"]
    with pytest.raises(TsaraBaselineError, match="no non-circular role='gas' variable"):
        baseline_state(met, instrument="met", baseline=config())


def test_baseline_states_rolls_every_stream_with_a_gas_and_skips_the_rest(
    drive: xr.Dataset, caplog: pytest.LogCaptureFixture
) -> None:
    met = drive[["wind_dir"]].copy()
    met.coords["time_bnds"] = drive["time_bnds"]
    with caplog.at_level(logging.INFO, logger="tsara.baseline.state"):
        states = baseline_states({"van": drive, "met": met}, config())
    assert sorted(states) == ["van"]
    assert "Stream 'met' has no gas variable to roll" in caplog.text


def test_an_enhancement_of_a_reading_without_a_cell_method_declares_none() -> None:
    """The enhancement inherits the reading's cell method, and inherits its absence too."""
    stream = make_stream(cells(0.0, 1.0, 300), {"ch4": 1900 + np.zeros(300)})
    del stream["ch4"].attrs["cell_methods"]
    state = baseline_state(stream, instrument="a", baseline=config())
    assert "cell_methods" not in state["enhancement_ch4"].attrs


def test_a_window_holding_exactly_the_required_count_is_kept() -> None:
    """The rule is fewer-than, so exactly 1/q readings, or exactly min_readings, is enough.

    1 s cells under a 5 s window centred on each cell hold exactly five
    readings away from the edges; with min_readings = 5 every interior
    window is valid and only the two edge windows on each side are blank.
    """
    stream = make_stream(cells(0.0, 1.0, 100), {"ch4": 1900 + np.arange(100.0)})
    state = baseline_state(
        stream, instrument="a", baseline=config(windows=("5s",), quantiles=(0.5,), min_readings=5)
    )
    counts = state["n_readings_window_ch4"].values[:, 0]
    assert (counts[2:-2] == 5).all()
    assert np.isfinite(state["baseline_ch4"].values[2:-2, 0, 0]).all()
    assert np.isnan(state["baseline_ch4"].values[:2, 0, 0]).all()


def test_from_field_refuses_a_baseline_of_the_field_from_another_instrument() -> None:
    dense, sparse = sparse_and_dense()
    settings = config(methods={"can.benzene_can": {"method": "from_field", "instrument": "ptr"}})
    donor = baseline_state(dense, instrument="ptr", baseline=settings)
    joined = bin_streams_onto_cells(
        {"ptr": donor}, stream_cells(sparse, "can"), [("ptr", "baseline_benzene")]
    )
    joined["baseline_benzene"].attrs["tsara_instrument"] = "another_ptr"
    with pytest.raises(TsaraBaselineError, match="holds 0 baseline column.*from 'ptr'"):
        baseline_state(
            sparse, instrument="can", baseline=settings, provided={"benzene_can": joined}
        )
