"""Tests for the clock survey (``tsara.survey.clocks``), on clocks whose errors are known.

Named ``test_survey_clocks`` because ``tests/core/test_clocks.py`` tests the
table's format, and test files need distinct basenames.

Every stream here is built from one sequence of plumes: an instrument reads the
plume of true second t and records it at t minus the shift that corrects its
clock, so the shift the survey should find, and the row it should write (the
same number), are known by construction.
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
    shift: Callable[[np.ndarray], np.ndarray] | float = 0.0,
    *,
    seed: int = 1,
    variables: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    attrs: dict[str, object] | None = None,
    first: int = 0,
    n: int = N,
    dropouts: tuple[tuple[int, int], ...] = (),
) -> xr.Dataset:
    """Readings at true seconds ``first..first+n``, recorded at t - shift(t).

    So ``recorded + shift`` is the truth: a clock running 5 s fast is ``-5``.

    ``variables`` maps a name to a function of the true seconds; by default one
    ``ch4`` reading the shared plumes with a little noise. Each of
    ``dropouts`` leaves out every reading of the true seconds ``[start, stop)``.
    """
    true_s = np.arange(first, first + n, dtype=float)
    for start, stop in dropouts:
        true_s = true_s[(true_s < start) | (true_s >= stop)]
    count = len(true_s)
    shifts = shift(true_s) if callable(shift) else np.full(count, float(shift))
    rng = np.random.default_rng(seed)
    makers = variables or {"ch4": lambda t: TRUTH[t.astype(int)]}
    data = {name: make(true_s) + rng.normal(0, 0.3, count) for name, make in makers.items()}
    recorded = T0 + pd.to_timedelta(true_s - shifts, unit="s")
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
    """Recorded 5 s late, so the shift is -5 s. The record's last 5 s spill into
    a seventh segment too short to measure, which the row reaches over rather
    than leaving uncorrected."""
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-5)}, _config(ONE))
    (row,) = survey.rows
    assert (row.instrument, row.reference_clock, row.provenance) == ("inst", "ref", "measured")
    assert row.shift_ns == -5 * SECOND and row.shift_end_ns is None
    assert (row.start_ns, row.end_ns) == (_ns(0), _ns(4200))
    assert row.reason == "clock survey: 6 segments, tier 1, shifts -5 to -5 s"
    statuses = survey.segments[survey.segments.instrument == "inst"].status.tolist()
    assert statuses == ["decided"] * 6 + ["too short"]
    assert survey.segments[survey.segments.status == "decided"].refined.all()
    assert survey.faults.empty
    assert set(survey.measurements.columns) >= {
        "shift_onto_partner_s",
        "r",
        "n_pairs",
        "pairs_disagree",
    }
    assert "shift_onto_reference_s" in survey.segments.columns


def test_a_step_at_a_logging_gap_puts_the_rows_boundary_at_the_gap() -> None:
    """3 s late, then 11 s from true second 1800: on the recorded clock a hole
    from 1802 to 1811 s. The rows meet at the reading after the hole, which is
    exactly where the step is, not at the segment edge."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: np.where(t < 1800, -3.0, -11.0))},
        _config(ONE),
    )
    first, second = survey.rows
    assert (first.shift_ns, second.shift_ns) == (-3 * SECOND, -11 * SECOND)
    assert first.end_ns == second.start_ns == _ns(1811)
    (step,) = _faults(survey, "clock step")
    assert step["detail"] == "shift -3 s to -11 s"


def test_a_step_with_no_gap_stays_at_the_segment_edge() -> None:
    """A clock jumping back (11 s late, then 3 s) leaves no hole to find."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: np.where(t < 1800, -11.0, -3.0))},
        _config(ONE),
    )
    first, second = survey.rows
    assert first.end_ns == second.start_ns == _ns(1800)


def test_a_step_across_a_logging_stop_says_so() -> None:
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: np.where(t < 1800, -3.0, -90.0))},
        _config(ONE, search="120s"),
    )
    (step,) = _faults(survey, "clock step")
    assert step["detail"] == "shift -3 s to -90 s, across a logging stop of 88 s"


def test_a_slow_drift_is_one_row_with_a_shift_end() -> None:
    """1 s late at the start, losing 1 s more every 10 minutes: under the 2 s
    tolerance from one segment to the next, 6 s across the hour. Measured to a
    tenth of a second: in whole seconds the segment shifts are a staircase, and
    the line through them misses the rate by 5.7 % here (1.7 % at 0.1 s)."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: -(1.0 + t / 600.0))},
        _config(ONE, resolution="0.1s"),
    )
    (row,) = survey.rows
    assert row.shift_end_ns is not None
    assert row.shift_ns / SECOND == pytest.approx(-1.0, abs=1.0)
    # The row reaches over the record's short last segment, to 01:10. Found
    # building notebook 08: the line must be evaluated at the row's own end, or
    # the same change is spread over a longer stretch and the rate comes out
    # low (1 s per 700 s here instead of per 600).
    assert row.end_ns == _ns(4200)
    rate = (row.shift_end_ns - row.shift_ns) / (row.end_ns - row.start_ns)
    assert rate == pytest.approx(-1 / 600, rel=0.03)
    (drift,) = _faults(survey, "clock drift")
    # The rate is the shift's own change: a clock running further ahead falls.
    hours = (row.end_ns - row.start_ns) / SECOND / 3600
    per_hour = (row.shift_end_ns - row.shift_ns) / SECOND / hours
    assert per_hour == pytest.approx(-6.0, rel=0.03)
    assert drift["detail"].endswith(f", {per_hour:+.2f} s per hour")


def test_a_drift_faster_than_the_tolerance_per_segment_is_named() -> None:
    """3 s more late every segment: each segment its own row, and five steps of
    one sign in a row are named a fast drift."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: -3.0 * (t // 600))}, _config(ONE)
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
            "inst": _stream(lambda t: np.where((t >= 1200) & (t < 1800), -12.0, -4.0)),
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
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-5, variables=hole)}, _config(ONE))
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
    """ref <- a <- b. a is 4 s late (-4 s); b 9 s late (-9 s), so -5 s onto a.
    In a's fourth segment its tracer against ref is missing, so b's shift
    there, though measured against a, cannot reach the reference clock."""
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
        {"ref": _stream(), "a": _stream(-4, variables=a_vars), "b": _stream(-9, seed=3)}, config
    )
    b = survey.segments[survey.segments.instrument == "b"]
    assert b.shift_onto_partner_s.iloc[0] == -5 and b.shift_onto_reference_s.iloc[0] == -9
    assert b.status.tolist()[3] == "partner undecided"
    assert {row.shift_ns for row in survey.rows if row.instrument == "b"} == {-9 * SECOND}


def test_tiers_are_compared_on_the_reference_clock() -> None:
    """b's first tier, against a, says -5 s; composed through a's -4 s that is
    -9 s, which its second tier, against ref directly, also says: no
    disagreement. Against a variable read 10 s later still, it disagrees."""
    tiers = [
        {"partner": "a", "pairs": [["ch4", "ch4"]]},
        {"partner": "ref", "pairs": [["ch4", "ch4"]]},
    ]
    agree = survey_clocks(
        {"ref": _stream(), "a": _stream(-4), "b": _stream(-9, seed=3)},
        _config({"a": ONE["inst"], "b": tiers}),
    )
    assert not _faults(agree, "tiers disagree")

    late = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "c2h6": lambda t: TRUTH[np.clip(t - 10, 0, None).astype(int)],
    }
    tiers[1]["pairs"] = [["c2h6", "ch4"]]
    disagree = survey_clocks(
        {"ref": _stream(), "a": _stream(-4), "b": _stream(-9, seed=3, variables=late)},
        _config({"a": ONE["inst"], "b": tiers}),
    )
    found = _faults(disagree, "tiers disagree")
    assert len(found) == 6
    assert found[0]["detail"] == "tier 1 -9 s, tier 2 -19 s onto the reference clock"


def test_pairs_that_disagree_leave_the_segment_undecided() -> None:
    late = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "co": lambda t: TRUTH[np.clip(t - 6, 0, None).astype(int)],
    }
    config = _config({"inst": [{"partner": "ref", "pairs": [["ch4", "ch4"], ["co", "ch4"]]}]})
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-4, variables=late)}, config)
    assert survey.rows == ()
    statuses = set(survey.segments[survey.segments.instrument == "inst"].status)
    assert "pairs disagree" in statuses
    assert survey.measurements.pairs_disagree.any()


def test_a_shift_beyond_the_search_is_not_decided_and_writes_no_table(tmp_path: Path) -> None:
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-32)}, _config(ONE))
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
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-5, attrs=attrs)}, _config(ONE))
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
    shifts = survey.segments.shift_onto_reference_s.dropna()
    assert len(shifts) == 6 and (shifts == 0).all()


def test_missing_streams_are_refused() -> None:
    with pytest.raises(TsaraSurveyError, match=r"needs stream\(s\) \['ref'\]"):
        survey_clocks({"inst": _stream(-5)}, _config(ONE))


def test_streams_without_readings_are_refused() -> None:
    empty = _stream(n=0)
    with pytest.raises(TsaraSurveyError, match="hold no readings"):
        survey_clocks({"ref": empty, "inst": empty}, _config(ONE))


def test_every_fault_named_is_a_known_kind() -> None:
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: -3.0 * (t // 600))}, _config(ONE)
    )
    assert set(survey.faults.kind) <= set(FAULT_KINDS)


def test_the_written_table_reads_back_as_the_rows(tmp_path: Path) -> None:
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: np.where(t < 1800, -3.0, -11.0))},
        _config(ONE),
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
    fast (shifts -7 s and -19 s); one, measured against it, is 4 s slow (+4 s).
    The survey must find both through the chain, place the step where it was,
    and its own table must put every reading of every instrument on its true
    time; surveyed again, every shift is zero."""
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
            ClockError("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "-7s"),
            ClockError("2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z", "-19s"),
        ],
        "slow": [ClockError("2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z", "4s")],
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
    assert (again.segments.shift_onto_reference_s.dropna() == 0).all()


# ---------------------------------------------------------------------------
# Added after mutation testing: each pins a rule no test above could see
# ---------------------------------------------------------------------------


def test_a_weak_correlation_decides_nothing() -> None:
    """A partner variable sharing no plumes still has a best shift; below the
    threshold it decides nothing, and no row is written from it."""
    rng = np.random.default_rng(9)
    unrelated = {"ch4": lambda t: rng.normal(0, 1, len(t))}
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(-5, variables=unrelated)}, _config(ONE)
    )
    assert survey.rows == ()
    measured = survey.measurements.dropna(subset=["shift_onto_partner_s"])
    assert len(measured) == 6 and (measured.r < 0.6).all()
    assert set(survey.segments.status) == {"no tier decided", "too short"}
    # With no row at all, the whole record is named, its sliver of a last segment too.
    (undecided,) = _faults(survey, "undecided")
    assert (undecided["start"], undecided["end"]) == (
        "2026-01-01T00:00:00Z",
        "2026-01-01T01:10:00Z",
    )


def test_the_first_tier_that_decides_sets_the_shift() -> None:
    """Both tiers decide, -9 s and -19 s onto the reference clock: the row takes
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
        {"ref": _stream(), "inst": _stream(-9, variables=late)}, _config({"inst": tiers})
    )
    assert {row.shift_ns for row in survey.rows} == {-9 * SECOND}
    assert set(survey.segments.tier.dropna().unique()) == {1}


def test_segments_are_whole_multiples_of_the_segment_from_the_epoch() -> None:
    """A record starting at 00:03:07 is still cut at 00:00, 00:10, ... so that
    every instrument's segment k is the same stretch of time."""
    survey = survey_clocks(
        {"ref": _stream(first=187, n=3000), "inst": _stream(-5, first=187, n=3000)}, _config(ONE)
    )
    starts = survey.segments[survey.segments.instrument == "inst"].start.tolist()
    assert starts[:2] == ["2026-01-01T00:00:00Z", "2026-01-01T00:10:00Z"]


def test_a_tier_may_ask_for_fewer_paired_seconds() -> None:
    """A partner logging every 5 s holds 120 readings in ten minutes: never
    enough for the survey's 200, decided when its tier asks for 60."""
    sparse = _stream()
    sparse = sparse.isel(time=slice(0, None, 5))
    default = survey_clocks({"ref": sparse, "inst": _stream(-5)}, _config(ONE))
    assert default.rows == ()
    lenient = {"inst": [{"partner": "ref", "pairs": [["ch4", "ch4"]], "min_pairs": 60}]}
    survey = survey_clocks({"ref": sparse, "inst": _stream(-5)}, _config(lenient))
    assert {row.shift_ns for row in survey.rows} == {-5 * SECOND}


# ---------------------------------------------------------------------------
# The shift below a second
# ---------------------------------------------------------------------------


def test_at_one_second_a_clock_7_7_s_late_takes_minus_8_not_minus_7() -> None:
    """Whole seconds put a -7.7 s shift at -7; the readings' own cells put it at
    the nearest second, so the shift written leaves 0.3 s, not 0.7."""
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-7.7)}, _config(ONE))
    (row,) = survey.rows
    assert row.shift_ns == -8 * SECOND


def test_fractional_clock_errors_come_back_exactly_at_the_resolution(tmp_path: Path) -> None:
    """The loop of the test above, with clock errors that are not whole
    seconds: 'fast' -7.3 s then -19.6 s, 'slow' +4.25 s, on the 0.05 s lattice.
    Every reading is the mean of the air over its cell and nothing is noisy, so
    the survey must give the shifts back exactly, through the chain, and its
    table must put every reading on its true time."""
    mean = {"method": "mean"}
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
                "ref": {"native_rate": "1s", "support": mean, "measures": {"ch4": {}}},
                "fast": {"native_rate": "1s", "support": mean, "measures": {"ch4": {}}},
                "slow": {"native_rate": "2s", "support": mean, "measures": {"ch4": {}}},
            },
        }
    )
    generated = generate(config)
    errors = {
        "fast": [
            ClockError("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "-7.3s"),
            ClockError("2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z", "-19.6s"),
        ],
        "slow": [ClockError("2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z", "4.25s")],
    }
    manifest = load_manifest(
        export_raw(generated, tmp_path / "export", clock_errors=errors, reference_clock="ref")
    )
    survey_config = SurveyConfig.model_validate(
        {
            "reference_clock": "ref",
            "resolution": "0.05s",
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
    assert not _faults(survey, "tiers disagree")

    table = survey.write_table(tmp_path / "survey.csv")
    corrected = ingest_campaign(manifest.model_copy(update={"clock_corrections": (table,)}))
    for name in generated.streams:
        np.testing.assert_array_equal(
            corrected[name]["time"].values, generated.streams[name]["time"].values
        )


def test_each_measurement_and_segment_says_whether_the_noise_was_removed() -> None:
    """With the readings' random sigma declared, every measured shift says the
    noise was removed, and so does every decided segment; a segment no tier
    measured says neither (NA). Without a sigma, nothing says it was."""

    def declared(stream: xr.Dataset) -> xr.Dataset:
        stream["sigma_rand_ch4"] = ("time", np.full(stream.sizes["time"], 0.3))
        stream["ch4"].attrs["uncertainty_provenance_random"] = "declared"
        return stream

    stated = survey_clocks(
        {"ref": declared(_stream()), "inst": declared(_stream(-5.0))}, _config(ONE)
    )
    measured = stated.measurements[np.isfinite(stated.measurements.shift_onto_partner_s)]
    assert len(measured) and measured.noise_removed.all()
    segments = stated.segments.set_index("status")
    assert segments.noise_removed.loc["decided"].all()
    assert segments.noise_removed.loc[["too short"]].isna().all()
    # A segment no tier decided names no tier (it read 0, like a real tier).
    assert segments.tier.loc[["too short"]].isna().all()

    plain = survey_clocks({"ref": _stream(), "inst": _stream(-5.0)}, _config(ONE))
    assert not plain.measurements.noise_removed.any()


def test_a_shift_at_the_edge_of_the_cells_window_is_not_decided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the shift on the readings' own cells lands at either end of its
    window, the true one may lie beyond: the segment is not decided."""
    import tsara.survey.clocks as clocks

    monkeypatch.setattr(clocks, "refined_shift", lambda *a, **k: (-7.0, True, False, True))
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-5.0)}, _config(ONE))
    assert survey.rows == ()
    assert set(survey.segments.status) <= {"at search edge", "too short"}


# ---------------------------------------------------------------------------
# Walkthrough stage A2: rules the code-only pass found wrong
# ---------------------------------------------------------------------------


def test_a_backup_tier_decides_where_the_first_tiers_partner_is_undecided() -> None:
    """ref <- mid <- end, with end's backup straight against ref. mid is tied to
    ref by its CO, missing from 00:30 to 00:40; end's first tier still decides
    against mid's methane there, but cannot reach the reference clock. Its
    backup measured the shift directly, so the segment is decided by tier 2 and
    end keeps one row (found in the walkthrough: it was left undecided)."""
    both = {"ch4": lambda t: TRUTH[t.astype(int)], "co": lambda t: TRUTH[t.astype(int)]}
    mid_vars = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "co": lambda t: np.where((t >= 1800) & (t < 2400), np.nan, TRUTH[t.astype(int)]),
    }
    config = _config(
        {
            "mid": [{"partner": "ref", "pairs": [["co", "co"]]}],
            "end": [
                {"partner": "mid", "pairs": [["ch4", "ch4"]]},
                {"partner": "ref", "pairs": [["ch4", "ch4"]]},
            ],
        }
    )
    survey = survey_clocks(
        {
            "ref": _stream(variables=both),
            "mid": _stream(-3, variables=mid_vars),
            "end": _stream(4, seed=5),
        },
        config,
    )
    end = survey.segments[survey.segments.instrument == "end"].set_index("start")
    assert end.loc["2026-01-01T00:30:00Z", "tier"] == 2
    assert end.loc["2026-01-01T00:30:00Z", "status"] == "decided"
    assert [row.shift_ns for row in survey.rows if row.instrument == "end"] == [4 * SECOND]
    assert not [f for f in _faults(survey, "undecided") if f["instrument"] == "end"]


def test_a_step_inside_an_undecided_segment_leaves_nothing_undecided() -> None:
    """The tracer is missing from 00:40 to 00:50 and the clock steps from 5 s
    to 15 s late at 00:45, leaving its 10 s hole. The rows meet at the hole and
    correct every reading of that segment, so it is not named undecided (found
    in the walkthrough: it was, counted from segments, not the rows' spans)."""
    hole = {
        "ch4": lambda t: np.where((t >= 2400) & (t < 3000), np.nan, TRUTH[t.astype(int)]),
        "co2": lambda t: np.full(len(t), 420.0),
    }
    step = _stream(lambda t: np.where(t < 2700, -5.0, -15.0), variables=hole)
    survey = survey_clocks({"ref": _stream(), "inst": step}, _config(ONE))
    first, second = survey.rows
    assert first.end_ns == second.start_ns == _ns(2700 + 15)
    assert len(_faults(survey, "clock step")) == 1
    assert not _faults(survey, "undecided")


def test_an_outliers_neighbours_are_adjacent_not_across_an_empty_stretch() -> None:
    """Readings 5 s late, then none for ten minutes, ten minutes 12 s late, none
    again, and 5 s late to the end. The 12 s stretch sits between two logging
    stops, where the clock may really have been elsewhere: it is its own row
    with a step either side, not an outlier inside one row at 5 s."""
    pieces = [
        _stream(-5, first=0, n=1195),
        _stream(-12, first=1788, n=600),
        _stream(-5, first=2995, n=600),
    ]
    inst = xr.concat(pieces, dim="time")
    survey = survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE))
    assert [row.shift_ns for row in survey.rows] == [-5 * SECOND, -12 * SECOND, -5 * SECOND]
    assert not _faults(survey, "outlier")
    assert len(_faults(survey, "clock step")) == 2


def test_an_outlier_is_still_set_aside_past_a_segment_too_short_to_measure() -> None:
    """Adjacency passes over segments too short to measure: a 12 s segment
    between two 5 s ones, the second reached past a sliver, is still set
    aside inside one row."""
    pieces = [
        _stream(-5, first=0, n=1800),
        _stream(-12, first=1800, n=600),
        _stream(-5, first=2410, n=20),
        _stream(-5, first=3000, n=600),
    ]
    inst = xr.concat(pieces, dim="time")
    survey = survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE))
    assert [f["start"] for f in _faults(survey, "outlier")] == ["2026-01-01T00:30:00Z"]
    assert [row.shift_ns for row in survey.rows] == [-5 * SECOND]


HOLE = {
    "ch4": lambda t: np.where((t >= 1800) & (t < 2400), np.nan, TRUTH[t.astype(int)]),
    "co2": lambda t: np.full(len(t), 420.0),
}


def test_up_to_max_bridged_of_undecided_segments_are_bridged() -> None:
    """The ten minutes without a tracer, between shifts that agree, readings
    unbroken: bridged into one row, named, and nothing left undecided."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(-5, variables=HOLE)}, _config(ONE, max_bridged="10min")
    )
    (row,) = survey.rows
    assert row.shift_ns == -5 * SECOND
    assert row.reason.endswith(", 1 undecided bridged")
    (bridge,) = _faults(survey, "bridged")
    assert (bridge["start"], bridge["end"]) == ("2026-01-01T00:30:00Z", "2026-01-01T00:40:00Z")
    assert not _faults(survey, "undecided")
    assert set(survey.faults.kind) <= set(FAULT_KINDS)


@pytest.mark.parametrize(
    ("max_bridged", "stop", "shift"),
    [
        ("5min", None, -5.0),  # longer than allowed
        ("10min", (2000, 2090), -5.0),  # a logging stop of 90 s inside it
    ],
)
def test_an_undecided_stretch_is_not_bridged_beyond_its_rule(
    max_bridged: str, stop: tuple[int, int] | None, shift: float
) -> None:
    inst = _stream(shift, variables=HOLE)
    if stop is not None:
        recorded_s = (inst["time"].values - np.datetime64(T0.value, "ns")) / np.timedelta64(1, "s")
        true = recorded_s + shift
        inst = inst.isel(time=~((true >= stop[0]) & (true < stop[1])))
    survey = survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE, max_bridged=max_bridged))
    assert len(survey.rows) == 2
    assert not _faults(survey, "bridged")
    assert len(_faults(survey, "undecided")) == 1


def test_a_step_across_an_undecided_stretch_is_a_step_not_a_bridge() -> None:
    clock = lambda t: np.where(t < 2100, -5.0, -15.0)  # noqa: E731
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(clock, variables=HOLE)},
        _config(ONE, max_bridged="10min"),
    )
    assert [row.shift_ns for row in survey.rows] == [-5 * SECOND, -15 * SECOND]
    assert len(_faults(survey, "clock step")) == 1
    assert not _faults(survey, "bridged")


def test_a_backup_tier_whose_partner_is_undecided_cannot_decide_either() -> None:
    """end's first tier, against ref, has no ethane from 00:30 to 00:40; its
    backup, against mid, decides there, but mid is undecided in that segment.
    Neither reaches the reference clock: the segment is 'partner undecided'."""
    both = {"ch4": lambda t: TRUTH[t.astype(int)], "co": lambda t: TRUTH[t.astype(int)]}

    def gap(t: np.ndarray) -> np.ndarray:
        return np.asarray((t >= 1800) & (t < 2400))

    mid_vars = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "co": lambda t: np.where(gap(t), np.nan, TRUTH[t.astype(int)]),
    }
    end_vars = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "c2h6": lambda t: np.where(gap(t), np.nan, TRUTH[t.astype(int)]),
    }
    config = _config(
        {
            "mid": [{"partner": "ref", "pairs": [["co", "co"]]}],
            "end": [
                {"partner": "ref", "pairs": [["c2h6", "ch4"]]},
                {"partner": "mid", "pairs": [["ch4", "ch4"]]},
            ],
        }
    )
    survey = survey_clocks(
        {
            "ref": _stream(variables=both),
            "mid": _stream(-3, variables=mid_vars),
            "end": _stream(4, seed=5, variables=end_vars),
        },
        config,
    )
    end = survey.segments[survey.segments.instrument == "end"].set_index("start")
    assert end.loc["2026-01-01T00:30:00Z", "status"] == "partner undecided"
    assert pd.isna(end.loc["2026-01-01T00:30:00Z", "tier"])


def test_a_segment_agreeing_with_one_neighbour_is_no_outlier() -> None:
    """Shifts of -5, -5, -7.6, -6, -6, -6 s: the third differs by more than the
    tolerance from the -5 s before it but not from the -6 s after, so it is a
    step to a new row, not a segment set aside."""
    clock = lambda t: np.select([t < 1200, t < 1800], [-5.0, -7.6], -6.0)  # noqa: E731
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(clock)}, _config(ONE, resolution="0.1s")
    )
    assert not _faults(survey, "outlier")
    assert _faults(survey, "clock step")


# ---------------------------------------------------------------------------
# Walkthrough 2: cells that do not line up
# ---------------------------------------------------------------------------


def _with_cells(stream: xr.Dataset, width_s: float) -> xr.Dataset:
    """Give a stream explicit cells of one width, centred on its times."""
    half = np.timedelta64(round(width_s * SECOND / 2), "ns")
    times = stream["time"].values
    return stream.assign(time_bnds=(("time", "nv"), np.stack([times - half, times + half], axis=1)))


def test_cells_that_do_not_line_up_keep_whole_seconds_and_are_named() -> None:
    """2 s cells against 1 s cells kept every other second, the shape of a slow
    analyzer written into a merge file's 1 s rows: the narrower series' cells
    never cover a wider one, so no shift below a second can be measured on the
    cells. Each segment keeps its whole-second peak, the row is written from
    them, and the run of such segments is named rather than left undecided."""
    every_other = slice(None, None, 2)
    ref = _with_cells(_stream().isel(time=every_other), 1.0)
    inst = _with_cells(_stream(-5).isel(time=every_other), 2.0)
    survey = survey_clocks({"ref": ref, "inst": inst}, _config(ONE, resolution="0.1s"))
    (row,) = survey.rows
    assert row.shift_ns == -5 * SECOND
    decided = survey.segments[survey.segments.status == "decided"]
    assert len(decided) == 6 and not decided.refined.any()
    assert not survey.measurements.refined.any()
    (fault,) = _faults(survey, "whole seconds")
    assert fault["detail"].startswith("6 segments measured on whole seconds")
    assert (fault["start"], fault["end"]) == ("2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z")


#: The tracer missing for twenty minutes, true 00:20 to 00:40, while the
#: instrument logs another variable: two undecided segments around a step.
LONG_HOLE = {
    "ch4": lambda t: np.where((t >= 1200) & (t < 2400), np.nan, TRUTH[t.astype(int)]),
    "co2": lambda t: np.full(len(t), 420.0),
}


def _clock_steps_at_1800(before: float, after: float) -> Callable[[np.ndarray], np.ndarray]:
    def clock(t: np.ndarray) -> np.ndarray:
        return np.where(t < 1800, before, after)

    return clock


def test_a_clock_set_back_inside_an_undecided_stretch_is_not_placed_at_a_dropout() -> None:
    """The clock goes back 10 s (shift -15 to -5) at true 00:30, inside a
    stretch the survey cannot decide, and the readings drop out for 5 s at
    00:23:20. A clock set back leaves no hole, so nothing in the readings says
    where it stepped: the rows meet at the segment edges and the stretch is
    named undecided. Found in the second walkthrough: the longest gap of any
    kind placed the step, the rows met at the dropout, 395 readings took a
    shift 10 s wrong, and the stretch was not named."""
    inst = _stream(_clock_steps_at_1800(-15.0, -5.0), variables=LONG_HOLE, dropouts=((1400, 1405),))
    survey = survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE))
    first, second = survey.rows
    assert (first.shift_ns, second.shift_ns) == (-15 * SECOND, -5 * SECOND)
    assert (first.end_ns, second.start_ns) == (_ns(1200), _ns(2400))
    (undecided,) = _faults(survey, "undecided")
    assert (undecided["start"], undecided["end"]) == (
        "2026-01-01T00:20:00Z",
        "2026-01-01T00:40:00Z",
    )


def test_a_clock_jumping_forward_is_placed_at_its_own_hole_not_a_longer_dropout() -> None:
    """The clock jumps 10 s forward (shift -5 to -15) at true 00:30, leaving a
    hole of 11 s in the recorded times, and the readings drop out for 30 s at
    00:23:20. The hole's excess over the 1 s spacing is the jump; the
    dropout's is not, and it is no logging stop: the rows meet at the hole, so
    every reading of the undecided stretch is corrected right (found in the
    second walkthrough: they met at the dropout, 370 readings 10 s wrong)."""
    inst = _stream(_clock_steps_at_1800(-5.0, -15.0), variables=LONG_HOLE, dropouts=((1400, 1430),))
    survey = survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE))
    first, second = survey.rows
    assert first.end_ns == second.start_ns == _ns(1800 + 15)
    assert not _faults(survey, "undecided")


def test_a_clock_set_back_at_a_logging_stop_is_placed_at_the_stop() -> None:
    """The clock goes back 10 s at true 00:30 across a logging stop of 90 s:
    clocks step where loggers stop (METHODS §12.2), so the stop places it."""
    inst = _stream(_clock_steps_at_1800(-15.0, -5.0), variables=LONG_HOLE, dropouts=((1790, 1880),))
    survey = survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE))
    first, second = survey.rows
    assert first.end_ns == second.start_ns == _ns(1880 + 5)
    (step,) = _faults(survey, "clock step")
    assert step["detail"].endswith("across a logging stop of 81 s")
    assert not _faults(survey, "undecided")


def _meet_at_the_edges(survey: ClockSurvey) -> None:
    """Assert the two rows meet at the segment edges around the undecided stretch, named."""
    first, second = survey.rows
    assert (first.end_ns, second.start_ns) == (_ns(1200), _ns(2400))
    (undecided,) = _faults(survey, "undecided")
    assert (undecided["start"], undecided["end"]) == (
        "2026-01-01T00:20:00Z",
        "2026-01-01T00:40:00Z",
    )


def test_two_logging_stops_leave_a_clock_set_back_unplaced() -> None:
    """Two logging stops of 90 s in the undecided stretch, the clock set back
    at the second: either could be where it stepped, so neither places it."""
    clock = _clock_steps_at_1800(-15.0, -5.0)
    inst = _stream(clock, variables=LONG_HOLE, dropouts=((1400, 1490), (1790, 1880)))
    _meet_at_the_edges(survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE)))


def test_a_forward_jumps_hole_places_it_before_a_logging_stop_does() -> None:
    """The clock jumps 10 s forward at 00:30, and the readings stop for 90 s at
    00:23:20: the hole matching the jump is the step's, the stop is not."""
    clock = _clock_steps_at_1800(-5.0, -15.0)
    inst = _stream(clock, variables=LONG_HOLE, dropouts=((1400, 1490),))
    first, second = survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE)).rows
    assert first.end_ns == second.start_ns == _ns(1800 + 15)


def test_a_dropped_reading_is_no_hole_however_well_it_matches() -> None:
    """A 3 s forward jump whose hole falls inside a 10 s dropout, so no gap
    matches it, and one reading dropped at 00:23:20: a 2 s gap on 1 s readings,
    whose excess of 1 s is within the 2 s tolerance of the jump. A gap must
    first be longer than twice the spacing, so the dropped reading places
    nothing and the rows meet at the segment edges."""
    clock = _clock_steps_at_1800(-5.0, -8.0)
    inst = _stream(clock, variables=LONG_HOLE, dropouts=((1400, 1401), (1795, 1805)))
    _meet_at_the_edges(survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE)))


def test_two_gaps_the_size_of_the_jump_leave_it_unplaced_whatever_the_stops() -> None:
    """A 10 s forward jump at 00:30 leaves its hole, a 10 s dropout at 00:21:40
    leaves another of the same size, and the readings stop for 90 s at
    00:25:00. Two gaps fit the jump, so neither places it, and a logging stop
    is no tiebreak: it is a weaker sign than a gap the size of the jump."""
    clock = _clock_steps_at_1800(-5.0, -15.0)
    dropouts = ((1300, 1310), (1500, 1590))
    inst = _stream(clock, variables=LONG_HOLE, dropouts=dropouts)
    _meet_at_the_edges(survey_clocks({"ref": _stream(), "inst": inst}, _config(ONE)))


def test_a_hole_is_measured_beyond_the_readings_own_spacing() -> None:
    """A record of one reading every 5 s, like the van's weather station:
    a 10 s forward jump leaves a gap of 15 s, whose excess over the spacing
    is the jump. Compared without the spacing it would be 5 s more than the
    jump, beyond the tolerance, and the step would go to the segment edges."""
    clock = _clock_steps_at_1800(-5.0, -15.0)
    inst = _stream(clock, variables=LONG_HOLE).isel(time=slice(None, None, 5))
    tiers = {"inst": [{"partner": "ref", "pairs": [["ch4", "ch4"]], "min_pairs": 60}]}
    first, second = survey_clocks({"ref": _stream(), "inst": inst}, _config(tiers)).rows
    assert first.end_ns == second.start_ns == _ns(1800 + 15)
