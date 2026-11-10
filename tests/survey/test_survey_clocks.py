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
WIDTH = "tsara_nominal_cell_width_s"
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


def test_a_drift_changing_less_than_the_tolerance_across_its_row_is_still_a_drift() -> None:
    """Found in the second walkthrough: a drift was written only when its line
    changed by more than the tolerance across the row, the number that also
    decides a step, so a clock losing 1 s an hour over the hour was corrected
    by a constant, wrong by half a second at each end. Now the change has to
    exceed the resolution and the segments' own scatter."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(lambda t: -(1.0 + t / 3600.0))},
        _config(ONE, resolution="0.1s"),
    )
    (row,) = survey.rows
    assert row.shift_end_ns is not None
    hours = (row.end_ns - row.start_ns) / SECOND / 3600
    assert (row.shift_end_ns - row.shift_ns) / SECOND / hours == pytest.approx(-1.0, abs=0.1)


def _scattered(ratio: float, n: int = 9) -> list[float]:
    """Segment shifts with scatter whose line changes by ``ratio`` standard errors.

    The scatter is held orthogonal to the line, so it is exactly the residual
    the drift test sees: about 0.2 s, alternating, sample standard deviation
    (n - 2 degrees of freedom) ``s``; one standard error of the change across n
    10-minute segments is ``s * n / sqrt(sum((k - mean)^2))``.
    """
    k = np.arange(n, dtype=float)
    basis = np.stack([np.ones(n), k], axis=1)
    alternating = 0.2 * (-1.0) ** k
    scatter = alternating - basis @ np.linalg.lstsq(basis, alternating, rcond=None)[0]
    s = np.sqrt(np.sum(scatter**2) / (n - 2))
    standard_error = s * n / np.sqrt(np.sum((k - k.mean()) ** 2))
    return list(scatter - ratio * standard_error * (k + 0.5) / n)


@pytest.mark.parametrize(
    ("shifts", "resolution_s", "drifts"),
    [
        # A steady clock measured with scatter whose line changes by 2.8 standard
        # errors (more than the resolution): not a drift. A drift at two
        # standard errors, or with the scatter taken over n degrees of freedom.
        (_scattered(2.8), 0.1, False),
        # 3.5 standard errors: a drift. Not one at four.
        (_scattered(3.5), 0.1, True),
        # A whole-second staircase whose line changes by 0.6 s: under a second.
        ([-1] * 8 + [-2], 1.0, False),
        # 0.6 s of change without scatter, measured to 0.1 s: the scatter's
        # floor is the resolution's rounding, so it is a drift.
        ([-0.6 * (k + 0.5) / 9 for k in range(9)], 0.1, True),
        # 36 segments, 0.8 s of change without scatter, at 1 s: three standard
        # errors on the rounding floor are 0.5 s, the resolution decides.
        ([-0.8 * (k + 0.5) / 36 for k in range(36)], 1.0, False),
        # Three segments, 1.5 s of change without scatter, at 1 s: more than the
        # resolution, not three standard errors on the rounding floor (1.8 s).
        ([-0.25, -0.75, -1.25], 1.0, False),
        # A staircase changing by 1.5 s at 1 s: a drift.
        ([-1] * 5 + [-2] * 4, 1.0, True),
    ],
)
def test_a_row_drifts_when_its_change_exceeds_the_resolution_and_its_scatter(
    shifts: list[float], resolution_s: float, drifts: bool
) -> None:
    """10-minute segments, the row spanning them exactly."""
    import tsara.survey.clocks as clocks

    edges = np.arange(len(shifts) + 1, dtype=np.int64) * 600 * SECOND
    shift_ns, shift_end_ns = clocks._row_shift(
        list(range(len(shifts))),
        np.array(shifts, dtype=float),
        edges,
        0,
        int(edges[-1]),
        resolution_s,
    )
    assert (shift_end_ns is not None) == drifts
    if not drifts:
        assert shift_ns == round(float(np.median(shifts)) * SECOND)


def test_a_run_holding_a_segment_on_whole_seconds_is_judged_at_a_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment left at its whole-second peak was measured to a second, so its
    run's drift is judged at a second, whatever resolution was asked for; one
    such segment is enough."""
    import tsara.survey.clocks as clocks
    from tsara.survey.shifts import refined_shift as refined

    seen: list[float] = []
    original = clocks._row_shift

    def spy(*args: Any) -> tuple[int, int | None]:
        seen.append(args[-1])
        return original(*args)

    def third_unrefined(*args: Any, **kwargs: Any) -> tuple[float, bool, bool, bool]:
        if args[1] == T0.value + 1200 * SECOND:  # the 00:20 segment only
            return kwargs["coarse_s"], False, False, False
        return refined(*args, **kwargs)

    monkeypatch.setattr(clocks, "_row_shift", spy)
    stream = {"ref": _stream(), "inst": _stream(-5.0)}
    survey_clocks(stream, _config(ONE, resolution="0.1s"))
    assert seen == [0.1]
    seen.clear()
    monkeypatch.setattr(clocks, "refined_shift", third_unrefined)
    survey_clocks(stream, _config(ONE, resolution="0.1s"))
    assert seen == [1.0]


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
    assert undecided["detail"] == "1 too few pairs"


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


# ---------------------------------------------------------------------------
# Pairs that move in opposite directions, and a tier's own threshold
# ---------------------------------------------------------------------------

#: Ozone where a plume of NO titrates it: a dip wherever the shared plumes rise.
DIP = {"o3": lambda t: 40.0 - 0.5 * TRUTH[t.astype(int)]}

#: Noise loud enough that a tracer reads r 0.38 to 0.70 across the six
#: segments (measured): a pair of gases emitted together, not one gas twice.
WEAK = np.random.default_rng(7).normal(0, 15, N + 100)


def _status(survey: ClockSurvey, instrument: str = "inst") -> list[str]:
    return list(survey.segments[survey.segments.instrument == instrument].status)


def test_a_pair_declared_negative_finds_a_dip_at_its_shift() -> None:
    """Read 7 s late (-7 s). Measured on the dip turned upside down, every
    segment decides, and r reads as high as any matching pair's."""
    config = _config({"inst": [{"partner": "ref", "pairs": [["o3", "ch4", "negative"]]}]})
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-7, variables=DIP)}, config)
    (row,) = survey.rows
    assert row.shift_ns == -7 * SECOND
    decided = survey.measurements[survey.measurements.decided]
    assert len(decided) == 6 and (decided.r > 0.98).all()


def test_the_same_dip_declared_positive_decides_nothing() -> None:
    """Undeclared, the dip correlates at -1 where it matches, and nowhere well:
    r stays below 0.3 (measured) at shifts scattered over the search, and no
    segment is decided at any of them."""
    config = _config({"inst": [{"partner": "ref", "pairs": [["o3", "ch4"]]}]})
    survey = survey_clocks({"ref": _stream(), "inst": _stream(-7, variables=DIP)}, config)
    assert survey.rows == ()
    assert not survey.measurements.decided.any()
    assert survey.measurements.r.max() < 0.6
    assert set(_status(survey)) <= {"weak correlation", "at search edge", "too short"}


def test_a_tier_pools_a_positive_and_a_negative_pair() -> None:
    """A rising pair and a dipping one at one shift pool to it; a dip read 6 s
    later than the rest makes the tier's pairs disagree."""
    tiers = {"inst": [{"partner": "ref", "pairs": [["co", "ch4"], ["o3", "ch4", "negative"]]}]}
    both = {"co": lambda t: TRUTH[t.astype(int)], **DIP}
    agree = survey_clocks({"ref": _stream(), "inst": _stream(-4, variables=both)}, _config(tiers))
    assert [row.shift_ns for row in agree.rows] == [-4 * SECOND]
    late = {
        "co": lambda t: TRUTH[t.astype(int)],
        "o3": lambda t: 40.0 - 0.5 * TRUTH[np.clip(t - 6, 0, None).astype(int)],
    }
    split = survey_clocks({"ref": _stream(), "inst": _stream(-4, variables=late)}, _config(tiers))
    assert split.rows == ()
    assert "pairs disagree" in _status(split)


def test_a_tier_may_set_its_own_threshold() -> None:
    """A weak tracer reaches the survey's 0.6 in one segment of six; with its
    tier's own threshold of 0.35 every segment decides, and one row results."""
    weak = {"w": lambda t: TRUTH[t.astype(int)] + WEAK[t.astype(int)]}
    streams = {"ref": _stream(), "inst": _stream(-4, variables=weak)}
    tier: dict[str, Any] = {"partner": "ref", "pairs": [["w", "ch4"]]}
    survey_wide = survey_clocks(streams, _config({"inst": [tier]}))
    assert _status(survey_wide).count("decided") == 1
    own = survey_clocks(streams, _config({"inst": [{**tier, "min_r": 0.35}]}))
    assert _status(own).count("decided") == 6
    assert [row.shift_ns for row in own.rows] == [-4 * SECOND]


def test_a_tiers_own_threshold_decides_which_pairs_may_disagree() -> None:
    """Two strong pairs at -4 s and a weak one read 10 s later still. A pair
    vetoes the tier only when it is decided on its own, at the tier's
    threshold: at 0.6 the weak pair gets there in one segment (r 0.74), at 0.35
    in five (all but the one where it reads 0.32)."""
    three = {
        "ch4": lambda t: TRUTH[t.astype(int)],
        "co": lambda t: TRUTH[t.astype(int)],
        "w": lambda t: TRUTH[np.clip(t - 10, 0, None).astype(int)] + WEAK[t.astype(int)],
    }
    streams = {"ref": _stream(), "inst": _stream(-4, variables=three)}
    pairs = [["ch4", "ch4"], ["co", "ch4"], ["w", "ch4"]]
    tier: dict[str, Any] = {"partner": "ref", "pairs": pairs}
    survey_wide = survey_clocks(streams, _config({"inst": [tier]}))
    assert _status(survey_wide).count("pairs disagree") == 1
    own = survey_clocks(streams, _config({"inst": [{**tier, "min_r": 0.35}]}))
    assert _status(own).count("pairs disagree") == 5


def test_a_chain_through_a_dip_lands_every_reading_on_its_true_time() -> None:
    """ref <- caps <- roze, as on the 2024 SLC-SOS van before its MIRO arrived:
    caps 4 s early (+4 s) on NOx, roze tied to caps only by its ozone dip, 7 s
    late and then 19 s from true second 1800. Every roze reading, moved by the
    row holding its recorded time, lands exactly on its true second."""

    def roze_shift(t: np.ndarray) -> np.ndarray:
        return np.where(t < 1800, -7.0, -19.0)

    nox = {"nox": lambda t: TRUTH[t.astype(int)]}
    streams = {
        "ref": _stream(variables=nox),
        "caps": _stream(4.0, seed=2, variables=nox),
        "roze": _stream(roze_shift, seed=3, variables=DIP),
    }
    config = _config(
        {
            "caps": [{"partner": "ref", "pairs": [["nox", "nox"]]}],
            "roze": [{"partner": "caps", "pairs": [["o3", "nox", "negative"]]}],
        }
    )
    survey = survey_clocks(streams, config)
    rows = [row for row in survey.rows if row.instrument == "roze"]
    assert [row.shift_ns for row in rows] == [-7 * SECOND, -19 * SECOND]
    true_s = np.arange(N, dtype=float)
    recorded = true_s - roze_shift(true_s)
    recorded_ns = T0.value + np.round(recorded * SECOND).astype(np.int64)
    corrected = np.full(N, np.nan)
    for row in rows:
        inside = (recorded_ns >= row.start_ns) & (recorded_ns < row.end_ns)
        corrected[inside] = recorded[inside] + row.shift_ns / SECOND
    assert np.array_equal(corrected, true_s)


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


def test_a_tier_pairing_exactly_its_minimum_is_a_weak_correlation() -> None:
    """The boundary of the two undecided statuses: a tier that pairs exactly
    the seconds it needs, and does not correlate, is a weak correlation, not
    too few pairs. Middle segments of a 1 s record pair all 600 seconds of a
    10-minute segment at any shift."""
    rng = np.random.default_rng(9)
    unrelated = {"ch4": lambda t: rng.normal(0, 1, len(t))}
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(0, variables=unrelated)}, _config(ONE, min_pairs=600)
    )
    middle = survey.segments[survey.segments.instrument == "inst"].iloc[1:5]
    assert (middle.status == "weak correlation").all()


def test_a_stream_already_corrected_is_an_after_check() -> None:
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(0, attrs={"tsara_clock_corrections": "..."})},
        _config(ONE),
    )
    assert survey.rows == ()
    (check,) = _faults(survey, "after-check")
    # The residuals are the after-check's point, so its fault states them.
    assert check["detail"].endswith(
        "no rows derived; 6 decided segments, residuals +0 to +0 s, 0 beyond ±2 s"
    )
    shifts = survey.segments.shift_onto_reference_s.dropna()
    assert len(shifts) == 6 and (shifts == 0).all()


def test_an_after_check_names_residuals_beyond_the_tolerance() -> None:
    """A table that left the stream 5 s out: the after-check says so in its fault,
    without a reviewer opening the segments."""
    corrected = _stream(-5, attrs={"tsara_clock_corrections": "..."})
    survey = survey_clocks({"ref": _stream(), "inst": corrected}, _config(ONE))
    (check,) = _faults(survey, "after-check")
    assert check["detail"].endswith("6 decided segments, residuals -5 to -5 s, 6 beyond ±2 s")


def test_an_after_check_with_nothing_decided_says_so() -> None:
    flat = {"ch4": lambda t: np.zeros(len(t))}
    survey = survey_clocks(
        {
            "ref": _stream(),
            "inst": _stream(0, variables=flat, attrs={"tsara_clock_corrections": "..."}),
        },
        _config(ONE),
    )
    (check,) = _faults(survey, "after-check")
    assert check["detail"].endswith("no rows derived; no segment decided")


def test_a_stream_ingestion_left_uncorrected_is_surveyed_not_after_checked() -> None:
    """An instrument a table has no row for records only its uncorrected share:
    a survey measures it and derives its rows, rather than taking it for an
    after-check (METHODS §12.3)."""
    survey = survey_clocks(
        {"ref": _stream(), "inst": _stream(-5, attrs={"tsara_clock_uncorrected_share": 1.0})},
        _config(ONE),
    )
    assert [row.shift_ns for row in survey.rows] == [-5 * SECOND]
    assert not _faults(survey, "after-check")


def test_a_chain_composes_through_the_partners_rows_across_its_outlier() -> None:
    """ref <- odd <- x. odd is 4 s late but one segment reads 12 s, set aside
    against its neighbours inside one -4 s row; x is 3 s later than odd
    throughout. Found in the second walkthrough: composed through odd's
    segment measurements, x's segment there had no partner shift and was left
    uncorrected, although odd's row corrected odd's readings. Composed through
    odd's rows, x takes -3 + -4 = -7 s there too: one row, nothing undecided,
    and x lands where odd's corrected readings land."""

    def odd(t: np.ndarray) -> np.ndarray:
        return np.where((t >= 1800) & (t < 2400), -12.0, -4.0)

    config = _config(
        {
            "odd": [{"partner": "ref", "pairs": [["ch4", "ch4"]]}],
            "x": [{"partner": "odd", "pairs": [["ch4", "ch4"]]}],
        }
    )
    streams = {
        "ref": _stream(),
        "odd": _stream(odd, seed=2),
        "x": _stream(lambda t: odd(t) - 3.0, seed=3),
    }
    survey = survey_clocks(streams, config)
    assert [f["start"] for f in _faults(survey, "outlier")] == ["2026-01-01T00:30:00Z"]
    x_rows = [row for row in survey.rows if row.instrument == "x"]
    assert [row.shift_ns for row in x_rows] == [-7 * SECOND]
    assert not _faults(survey, "undecided")
    x = survey.segments[survey.segments.instrument == "x"].set_index("start")
    assert x.loc["2026-01-01T00:30:00Z", "status"] == "decided"
    assert x.loc["2026-01-01T00:30:00Z", "shift_onto_reference_s"] == -7.0


def test_a_partner_drift_is_composed_where_the_air_lands_on_its_clock() -> None:
    """ref <- a <- b. a drifts (its shift falls 6 s an hour); b is 30 s later
    than a. b's shift onto ref in a segment is -30 s plus a's row read where
    b's air lands on a's clock, the segment's midpoint moved by -30 s, not at
    the midpoint itself. a's files label each reading's start, so its rows are
    keyed half a second before its stream's times, and are read there."""

    def a(t: np.ndarray) -> np.ndarray:
        return -(1.0 + 6.0 * t / 3600)

    config = _config(
        {
            "a": [{"partner": "ref", "pairs": [["ch4", "ch4"]]}],
            "b": [{"partner": "a", "pairs": [["ch4", "ch4"]]}],
        },
        resolution="0.1s",
        search="45s",
    )
    survey = survey_clocks(
        {
            "ref": _stream(),
            "a": _stream(a, seed=2, attrs={"tsara_support_label": "start", WIDTH: 1.0}),
            "b": _stream(lambda t: a(t) - 30.0, seed=3),
        },
        config,
    )
    (a_row,) = [row for row in survey.rows if row.instrument == "a"]
    assert a_row.shift_end_ns is not None
    b = survey.segments[(survey.segments.instrument == "b") & (survey.segments.status == "decided")]
    for start, onto_a, onto_ref in zip(
        b.start, b.shift_onto_partner_s, b.shift_onto_reference_s, strict=True
    ):
        landing = (
            pd.Timestamp(start).tz_localize(None)
            + pd.Timedelta("5min")
            + pd.Timedelta(seconds=onto_a)
        )
        recorded = landing.value - SECOND // 2
        fraction = (recorded - a_row.start_ns) / (a_row.end_ns - a_row.start_ns)
        a_there = (a_row.shift_ns + (a_row.shift_end_ns - a_row.shift_ns) * fraction) / SECOND
        assert onto_ref == pytest.approx(onto_a + a_there, abs=1e-6)


def test_a_chain_composes_through_a_partner_measured_as_an_after_check() -> None:
    """ref <- a <- b, a already corrected (an after-check, no rows of its own):
    b composes through a's residuals, here 0, so b's -5 s onto a is its shift
    onto ref, and b gets its row."""
    survey = survey_clocks(
        {
            "ref": _stream(),
            "a": _stream(0, seed=2, attrs={"tsara_clock_corrections": "..."}),
            "b": _stream(-5, seed=3),
        },
        _config(
            {
                "a": [{"partner": "ref", "pairs": [["ch4", "ch4"]]}],
                "b": [{"partner": "a", "pairs": [["ch4", "ch4"]]}],
            }
        ),
    )
    assert [(row.instrument, row.shift_ns) for row in survey.rows] == [("b", -5 * SECOND)]


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
    assert set(survey.segments.status) == {"weak correlation", "too short"}
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
    assert end["tier"].isna()["2026-01-01T00:30:00Z"]


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
