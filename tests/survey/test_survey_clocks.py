"""Tests for the clock survey (``tsara.survey.clocks``), on clocks whose errors are known.

Named ``test_survey_clocks`` because ``tests/core/test_clocks.py`` tests the
table's format, and test files need distinct basenames.

Every stream here is built from one sequence of plumes: an instrument reads the
plume of true second t and records it at t plus its clock error, so the lag the
survey should find, and the row it should write, are known by construction.
The last test runs the whole loop through the generator and real ingestion:
export with clock errors, survey, write the table, ingest with it, and land
every reading on its true time. ``docs/METHODS.md`` §12.4–12.5.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.config.loader import load_manifest
from tsara.config.survey import SurveyConfig
from tsara.config.synthetic import SyntheticConfig
from tsara.core.clocks import TsaraClockError, read_clock_corrections
from tsara.ingest import ingest_campaign
from tsara.survey import FAULT_KINDS, ClockSurvey, TsaraSurveyError, survey_clocks
from tsara.synthetic import generate
from tsara.synthetic.export import ClockError, export_raw

SECOND = 1_000_000_000
T0 = pd.Timestamp("2026-01-01T00:00:00")
N = 3600  # one hour of 1 s readings: six 10-minute segments

#: Ten-minute segments keep these tests quick; everything else is the default
#: but the search, the median and the pair count, scaled to them.
SETTINGS: dict[str, Any] = {
    "segment": "10min",
    "search": "30s",
    "min_pairs": 200,
    "running_median": "61s",
}


def _plumes(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = np.zeros(n)
    for centre in rng.choice(n, size=n // 30, replace=False):
        signal += rng.lognormal(3, 0.5) * np.exp(-0.5 * ((np.arange(n) - centre) / 3.0) ** 2)
    return signal


TRUTH = _plumes(N + 100)


def _stream(
    lag: Callable[[np.ndarray], np.ndarray] | float = 0.0,
    *,
    seed: int = 1,
    variables: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    attrs: dict[str, object] | None = None,
    first: int = 0,
    n: int = N,
) -> xr.Dataset:
    """Readings at true seconds ``first..first+n``, recorded at t + lag(t).

    ``variables`` maps a name to a function of the true seconds; by default one
    ``ch4`` reading the shared plumes with a little noise.
    """
    true_s = np.arange(first, first + n, dtype=float)
    lags = lag(true_s) if callable(lag) else np.full(n, float(lag))
    rng = np.random.default_rng(seed)
    makers = variables or {"ch4": lambda t: TRUTH[t.astype(int)]}
    data = {name: make(true_s) + rng.normal(0, 0.3, n) for name, make in makers.items()}
    recorded = T0 + pd.to_timedelta(true_s + lags, unit="s")
    order = np.argsort(np.asarray(recorded))
    stream = xr.Dataset(
        {name: ("time", values[order]) for name, values in data.items()},
        coords={"time": np.asarray(recorded)[order].astype("datetime64[ns]")},
    )
    stream.attrs.update(attrs or {})
    return stream


def _config(tracers: dict[str, list[dict[str, Any]]], **settings: Any) -> SurveyConfig:
    return SurveyConfig.model_validate(
        {"reference_clock": "ref", "tracers": tracers, **SETTINGS, **settings}
    )


ONE = {"inst": [{"partner": "ref", "pairs": [["ch4", "ch4"]]}]}


def _ns(seconds: float) -> int:
    return int(T0.value + round(seconds * SECOND))


def _faults(survey: ClockSurvey, kind: str) -> list[dict[str, Any]]:
    """The faults of one kind, as plain records."""
    found = survey.faults[survey.faults.kind == kind]
    return [{str(k): v for k, v in row.items()} for row in found.to_dict("records")]


# ---------------------------------------------------------------------------
# One clock, one row
# ---------------------------------------------------------------------------


def test_a_constant_clock_error_gives_one_row_over_the_whole_record() -> None:
    """Recorded 5 s late, so the lag is +5 s and the shift -5 s. The record's
    last 5 s spill into a seventh segment too short to measure, which the row
    reaches over rather than leaving uncorrected."""
    survey = survey_clocks({"ref": _stream(), "inst": _stream(5)}, _config(ONE))
    (row,) = survey.rows
    assert (row.instrument, row.reference_clock, row.provenance) == ("inst", "ref", "measured")
    assert row.shift_ns == -5 * SECOND and row.shift_end_ns is None
    assert (row.start_ns, row.end_ns) == (_ns(0), _ns(4200))
    assert row.reason == "clock survey: 6 segments, tier 1, lags +5 to +5 s"
    statuses = survey.segments[survey.segments.instrument == "inst"].status.tolist()
    assert statuses == ["decided"] * 6 + ["too short"]
    assert survey.faults.empty
    assert set(survey.measurements.columns) >= {"lag_s", "r", "n_pairs", "pairs_disagree"}


def test_a_step_at_a_logging_gap_puts_the_rows_boundary_at_the_gap() -> None:
    """3 s late, then 11 s from true second 1800: on the recorded clock a hole
    from 1802 to 1811 s. The rows meet at the reading after the hole, which is
    exactly where the step is, not at the segment edge."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: np.where(t < 1800, 3.0, 11.0))}, _config(ONE)
    )
    first, second = survey.rows
    assert (first.shift_ns, second.shift_ns) == (-3 * SECOND, -11 * SECOND)
    assert first.end_ns == second.start_ns == _ns(1811)
    (step,) = _faults(survey, "clock step")
    assert step["detail"] == "lag +3 s to +11 s"


def test_a_step_with_no_gap_stays_at_the_segment_edge() -> None:
    """A clock jumping back (11 s late, then 3 s) leaves no hole to find."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: np.where(t < 1800, 11.0, 3.0))}, _config(ONE)
    )
    first, second = survey.rows
    assert first.end_ns == second.start_ns == _ns(1800)


def test_a_step_across_a_logging_stop_says_so() -> None:
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: np.where(t < 1800, 3.0, 90.0))},
        _config(ONE, search="120s"),
    )
    (step,) = _faults(survey, "clock step")
    assert step["detail"] == "lag +3 s to +90 s, across a logging stop of 88 s"


def test_a_slow_drift_is_one_row_with_a_shift_end() -> None:
    """1 s late at the start, gaining 1 s every 10 minutes: under the 2 s
    tolerance from one segment to the next, 6 s across the hour."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: 1.0 + t / 600.0)}, _config(ONE)
    )
    (row,) = survey.rows
    assert row.shift_end_ns is not None
    assert row.shift_ns / SECOND == pytest.approx(-1.0, abs=1.0)
    # The row reaches over the record's short last segment, to 01:10. Found
    # building notebook 08: the line must be evaluated at the row's own end, or
    # the same change is spread over a longer stretch and the rate comes out
    # low (1 s per 700 s here instead of per 600).
    assert row.end_ns == _ns(4200)
    rate = (row.shift_ns - row.shift_end_ns) / (row.end_ns - row.start_ns)
    assert rate == pytest.approx(1 / 600, rel=0.05)
    (drift,) = _faults(survey, "clock drift")
    assert "s per hour" in drift["detail"]


def test_a_drift_faster_than_the_tolerance_per_segment_is_named() -> None:
    """3 s more every segment: each segment its own row, and five steps of one
    sign in a row are named a fast drift."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: 3.0 * (t // 600))}, _config(ONE)
    )
    assert len(survey.rows) == 6
    assert len(_faults(survey, "clock step")) == 5
    (fast,) = _faults(survey, "fast drift")
    assert fast["detail"].startswith("5 clock steps of one sign in a row")


# ---------------------------------------------------------------------------
# What breaks a row, and what does not
# ---------------------------------------------------------------------------


def test_one_segment_against_two_agreeing_neighbours_is_an_outlier() -> None:
    survey = survey_clocks(
        {
            "ref": _stream(),
            "inst": _stream(lambda t: np.where((t >= 1200) & (t < 1800), 12.0, 4.0)),
        },
        _config(ONE),
    )
    (row,) = survey.rows
    assert row.shift_ns == -4 * SECOND
    segments = survey.segments[survey.segments.instrument == "inst"]
    assert segments.status.tolist()[2] == "outlier"
    assert len(_faults(survey, "outlier")) == 1
    assert not _faults(survey, "clock step")


def test_a_segment_with_readings_but_no_decision_ends_the_row() -> None:
    """The tracer is missing for ten minutes while the instrument logs another
    variable: its readings there stay uncorrected, and the survey says so."""
    hole = {
        "ch4": lambda t: np.where((t >= 1800) & (t < 2400), np.nan, TRUTH[t.astype(int)]),
        "co2": lambda t: np.full(len(t), 420.0),
    }
    survey = survey_clocks({"ref": _stream(), "inst": _stream(5, variables=hole)}, _config(ONE))
    first, second = survey.rows
    assert first.shift_ns == second.shift_ns == -5 * SECOND
    assert first.end_ns == _ns(1800) and second.start_ns == _ns(2400)
    (undecided,) = _faults(survey, "undecided")
    assert (undecided["start"], undecided["end"]) == (
        "2026-01-01T00:30:00Z",
        "2026-01-01T00:40:00Z",
    )
    assert undecided["detail"] == "1 no tier decided"


def test_a_chain_composes_and_an_undecided_partner_stops_it() -> None:
    """ref <- a <- b. a is 4 s late; b 9 s late, so 5 s later than a. In a's
    fourth segment its tracer against ref is missing, so b's lag there, though
    measured against a, cannot reach the reference clock."""
    a_vars = {
        "ch4": lambda t: np.where((t >= 1800) & (t < 2400), np.nan, TRUTH[t.astype(int)]),
        "co": lambda t: TRUTH[t.astype(int)],
    }
    config = _config(
        {
            "a": [{"partner": "ref", "pairs": [["ch4", "ch4"]]}],
            "b": [{"partner": "a", "pairs": [["ch4", "co"]]}],
        }
    )
    survey = survey_clocks(
        {"ref": _stream(), "a": _stream(4, variables=a_vars), "b": _stream(9, seed=3)}, config
    )
    b = survey.segments[survey.segments.instrument == "b"]
    assert b.lag_s.iloc[0] == 5 and b.lag_to_reference_s.iloc[0] == 9
    assert b.status.tolist()[3] == "partner undecided"
    assert {row.shift_ns for row in survey.rows if row.instrument == "b"} == {-9 * SECOND}


def test_tiers_are_compared_on_the_reference_clock() -> None:
    """b's first tier, against a, says 5 s; composed through a's 4 s that is
    9 s, which its second tier, against ref directly, also says: no
    disagreement. Against a variable read 10 s later still, it disagrees."""
    tiers = [
        {"partner": "a", "pairs": [["ch4", "ch4"]]},
        {"partner": "ref", "pairs": [["ch4", "ch4"]]},
    ]
    agree = survey_clocks(
        {"ref": _stream(), "a": _stream(4), "b": _stream(9, seed=3)},
        _config({"a": ONE["inst"], "b": tiers}),
    )
    assert not _faults(agree, "tiers disagree")

    late = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "c2h6": lambda t: TRUTH[np.clip(t - 10, 0, None).astype(int)],
    }
    tiers[1]["pairs"] = [["c2h6", "ch4"]]
    disagree = survey_clocks(
        {"ref": _stream(), "a": _stream(4), "b": _stream(9, seed=3, variables=late)},
        _config({"a": ONE["inst"], "b": tiers}),
    )
    found = _faults(disagree, "tiers disagree")
    assert len(found) == 6
    assert found[0]["detail"] == "tier 1 +9 s, tier 2 +19 s to the reference clock"


def test_pairs_that_disagree_leave_the_segment_undecided() -> None:
    late = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "co": lambda t: TRUTH[np.clip(t - 6, 0, None).astype(int)],
    }
    config = _config({"inst": [{"partner": "ref", "pairs": [["ch4", "ch4"], ["co", "ch4"]]}]})
    survey = survey_clocks({"ref": _stream(), "inst": _stream(4, variables=late)}, config)
    assert survey.rows == ()
    statuses = set(survey.segments[survey.segments.instrument == "inst"].status)
    assert "pairs disagree" in statuses
    assert survey.measurements.pairs_disagree.any()


def test_a_lag_beyond_the_search_is_not_decided_and_writes_no_table(tmp_path: Path) -> None:
    survey = survey_clocks({"ref": _stream(), "inst": _stream(32)}, _config(ONE))
    assert survey.rows == ()
    assert "at search edge" in set(survey.segments.status)
    with pytest.raises(TsaraClockError, match="No clock corrections to write"):
        survey.write_table(tmp_path / "t.csv")


# ---------------------------------------------------------------------------
# The recorded clock, the after-check, refusals and the table
# ---------------------------------------------------------------------------


def test_row_edges_are_moved_back_to_the_clock_the_files_recorded() -> None:
    """A stream's times are cell midpoints after time_shift. With a time_shift
    of -4 s and start labels on 2 s cells, the recorded clock is 3 s behind the
    stream's, so each row edge is 3 s later on it."""
    attrs = {
        "tsara_time_shift": "-4s",
        "tsara_support_label": "start",
        "tsara_nominal_cell_width_s": 2.0,
    }
    survey = survey_clocks({"ref": _stream(), "inst": _stream(5, attrs=attrs)}, _config(ONE))
    (row,) = survey.rows
    assert (row.start_ns, row.end_ns) == (_ns(3), _ns(4203))
    assert row.shift_ns == -5 * SECOND


def test_a_stream_already_corrected_is_an_after_check() -> None:
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(0, attrs={"tsara_clock_corrections": "..."})},
        _config(ONE),
    )
    assert survey.rows == ()
    (check,) = _faults(survey, "after-check")
    assert "no rows derived" in check["detail"]
    lags = survey.segments.lag_to_reference_s.dropna()
    assert len(lags) == 6 and (lags == 0).all()


def test_missing_streams_are_refused() -> None:
    with pytest.raises(TsaraSurveyError, match=r"needs stream\(s\) \['ref'\]"):
        survey_clocks({"inst": _stream(5)}, _config(ONE))


def test_streams_without_readings_are_refused() -> None:
    empty = _stream(n=0)
    with pytest.raises(TsaraSurveyError, match="hold no readings"):
        survey_clocks({"ref": empty, "inst": empty}, _config(ONE))


def test_every_fault_named_is_a_known_kind() -> None:
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: 3.0 * (t // 600))}, _config(ONE)
    )
    assert set(survey.faults.kind) <= set(FAULT_KINDS)


def test_the_written_table_reads_back_as_the_rows(tmp_path: Path) -> None:
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: np.where(t < 1800, 3.0, 11.0))}, _config(ONE)
    )
    path = survey.write_table(tmp_path / "clocks.csv")
    back = read_clock_corrections([path])
    assert [(r.start_ns, r.end_ns, r.shift_ns, r.shift_end_ns, r.reason) for r in back] == [
        (r.start_ns, r.end_ns, r.shift_ns, r.shift_end_ns, r.reason) for r in survey.rows
    ]


# ---------------------------------------------------------------------------
# The whole loop, through the generator and real ingestion
# ---------------------------------------------------------------------------


def test_the_survey_recovers_exported_clock_errors_and_its_table_puts_every_reading_right(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Three methane analyzers see the same plumes. One steps from 7 s to 19 s
    fast; one, measured against it, is 4 s slow. The survey must find both
    through the chain, place the step where it was, and its own table must put
    every reading of every instrument on its true time; surveyed again, every
    lag is zero."""
    config = SyntheticConfig.model_validate(
        {
            "name": "clocks",
            "start": "2026-01-01T00:00:00Z",
            "duration": "150min",
            "seed": 11,
            "platform": {"kind": "stationary", "latitude": 40.0, "longitude": -111.0},
            "atmosphere": {
                "fields": {
                    "ch4": {"units": "ppb", "background": {"kind": "parametric", "offset": 1950.0}}
                },
                "sources": {
                    "street": {
                        "rate_per_hour": 40.0,
                        "reference_species": "ch4",
                        "shape": {"kind": "gaussian", "sigma": "4s"},
                        "amplitude": {"kind": "lognormal", "median": 80.0, "sigma_log": 0.6},
                    }
                },
            },
            "instruments": {
                "ref": {
                    "native_rate": "1s",
                    "measures": {"ch4": {"uncertainty": {"random": {"absolute": 1.0}}}},
                },
                "fast": {
                    "native_rate": "1s",
                    "measures": {"ch4": {"uncertainty": {"random": {"absolute": 2.0}}}},
                },
                "slow": {
                    "native_rate": "2s",
                    "measures": {"ch4": {"uncertainty": {"random": {"absolute": 1.5}}}},
                },
            },
        }
    )
    generated = generate(config)
    errors = {
        "fast": [
            ClockError("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "7s"),
            ClockError("2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z", "19s"),
        ],
        "slow": [ClockError("2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z", "-4s")],
    }
    manifest = load_manifest(
        export_raw(generated, tmp_path / "export", clock_errors=errors, reference_clock="ref")
    )
    survey_config = SurveyConfig.model_validate(
        {
            "reference_clock": "ref",
            "tracers": {
                "fast": [{"partner": "ref", "pairs": [["ch4", "ch4"]]}],
                "slow": [
                    {"partner": "fast", "pairs": [["ch4", "ch4"]]},
                    {"partner": "ref", "pairs": [["ch4", "ch4"]]},
                ],
            },
        }
    )
    survey = survey_clocks(
        ingest_campaign(manifest.model_copy(update={"clock_corrections": ()})), survey_config
    )
    truth = {(r.instrument, r.shift_ns) for r in read_clock_corrections(manifest.clock_corrections)}
    assert {(r.instrument, r.shift_ns) for r in survey.rows} == truth
    fast = [r for r in survey.rows if r.instrument == "fast"]
    assert fast[0].end_ns == fast[1].start_ns == _ns(3600 + 19)
    assert not _faults(survey, "tiers disagree")

    table = survey.write_table(tmp_path / "survey.csv")
    with caplog.at_level(logging.WARNING, logger="tsara.ingest"):
        corrected = ingest_campaign(manifest.model_copy(update={"clock_corrections": (table,)}))
    assert "dropped" not in caplog.text
    for name in generated.streams:
        np.testing.assert_array_equal(
            corrected[name]["time"].values, generated.streams[name]["time"].values
        )

    again = survey_clocks(corrected, survey_config)
    assert again.rows == ()
    assert (again.segments.lag_to_reference_s.dropna() == 0).all()


# ---------------------------------------------------------------------------
# Added after mutation testing: each pins a rule no test above could see
# ---------------------------------------------------------------------------


def test_a_weak_correlation_decides_nothing() -> None:
    """A partner variable sharing no plumes still has a best lag; below the
    threshold it decides nothing, and no row is written from it."""
    rng = np.random.default_rng(9)
    unrelated = {"ch4": lambda t: rng.normal(0, 1, len(t))}
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(5, variables=unrelated)}, _config(ONE)
    )
    assert survey.rows == ()
    measured = survey.measurements.dropna(subset=["lag_s"])
    assert len(measured) == 6 and (measured.r < 0.6).all()
    assert set(survey.segments.status) == {"no tier decided", "too short"}


def test_the_first_tier_that_decides_sets_the_lag() -> None:
    """Both tiers decide, 9 s and 19 s to the reference clock: the row takes
    tier 1's, and tier 2's only checks it."""
    late = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "c2h6": lambda t: TRUTH[np.clip(t - 10, 0, None).astype(int)],
    }
    tiers = [
        {"partner": "ref", "pairs": [["ch4", "ch4"]]},
        {"partner": "ref", "pairs": [["c2h6", "ch4"]]},
    ]
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(9, variables=late)}, _config({"inst": tiers})
    )
    assert {row.shift_ns for row in survey.rows} == {-9 * SECOND}
    assert set(survey.segments.tier.unique()) <= {0, 1}


def test_segments_are_whole_multiples_of_the_segment_from_the_epoch() -> None:
    """A record starting at 00:03:07 is still cut at 00:00, 00:10, ... so that
    every instrument's segment k is the same stretch of time."""
    survey = survey_clocks(
        {"ref": _stream(first=187, n=3000), "inst": _stream(5, first=187, n=3000)}, _config(ONE)
    )
    starts = survey.segments[survey.segments.instrument == "inst"].start.tolist()
    assert starts[:2] == ["2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z"]


def test_a_tier_may_ask_for_fewer_paired_seconds() -> None:
    """A partner logging every 5 s holds 120 readings in ten minutes: never
    enough for the survey's 200, decided when its tier asks for 60."""
    sparse = _stream()
    sparse = sparse.isel(time=slice(0, None, 5))
    default = survey_clocks({"ref": sparse, "inst": _stream(5)}, _config(ONE))
    assert default.rows == ()
    lenient = {"inst": [{"partner": "ref", "pairs": [["ch4", "ch4"]], "min_pairs": 60}]}
    survey = survey_clocks({"ref": sparse, "inst": _stream(5)}, _config(lenient))
    assert {row.shift_ns for row in survey.rows} == {-5 * SECOND}
