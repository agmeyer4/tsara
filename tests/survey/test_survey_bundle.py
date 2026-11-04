"""Tests for saving and reloading a clock survey (``tsara.survey.bundle``).

A table a person reviews must travel with the evidence for each of its rows,
and come back exactly: the three tables column for column and type for type,
the settings, and every row as ingestion would read it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.config.survey import SurveyConfig
from tsara.core.bundle import TsaraBundleError
from tsara.survey import ClockSurvey, load_clock_survey, save_clock_survey, survey_clocks

T0 = pd.Timestamp("2026-01-01T00:00:00")
N = 2400


def _stream(lag_s: float, seed: int) -> xr.Dataset:
    rng = np.random.default_rng(0)
    plumes = np.zeros(N + 100)
    for centre in rng.choice(N, size=N // 30, replace=False):
        plumes += rng.lognormal(3, 0.5) * np.exp(-0.5 * ((np.arange(N + 100) - centre) / 3.0) ** 2)
    noise = np.random.default_rng(seed).normal(0, 0.3, N)
    times = T0 + pd.to_timedelta(np.arange(N) + lag_s, unit="s")
    return xr.Dataset({"ch4": ("time", plumes[:N] + noise)}, coords={"time": np.asarray(times)})


CONFIG = SurveyConfig.model_validate(
    {
        "reference_clock": "ref",
        "tracers": {"inst": [{"partner": "ref", "pairs": [["ch4", "ch4"]]}]},
        "segment": "10min",
        "search": "30s",
        "min_pairs": 200,
        "running_median": "61s",
    }
)


@pytest.fixture(scope="module")
def survey() -> ClockSurvey:
    """A survey with rows: the instrument 6 s late."""
    return survey_clocks({"ref": _stream(0, 1), "inst": _stream(6, 2)}, CONFIG)


@pytest.fixture(scope="module")
def rowless() -> ClockSurvey:
    """A survey with no rows: the lag lies beyond the search."""
    return survey_clocks({"ref": _stream(0, 1), "inst": _stream(45, 2)}, CONFIG)


def _assert_same(back: ClockSurvey, survey: ClockSurvey) -> None:
    for name in ("measurements", "segments", "faults"):
        pd.testing.assert_frame_equal(getattr(back, name), getattr(survey, name))
    assert back.config == survey.config
    fields = (
        "instrument",
        "start_ns",
        "end_ns",
        "shift_ns",
        "shift_end_ns",
        "reference_clock",
        "reason",
    )
    assert [[getattr(r, f) for f in fields] for r in back.rows] == [
        [getattr(r, f) for f in fields] for r in survey.rows
    ]


def test_a_survey_with_rows_reloads_exactly(tmp_path: Path, survey: ClockSurvey) -> None:
    assert survey.rows
    target = save_clock_survey(survey, tmp_path / "bundle")
    assert target == tmp_path / "bundle" / "survey"
    assert sorted(p.name for p in target.iterdir()) == [
        "clock_corrections.csv",
        "faults.parquet",
        "measurements.parquet",
        "segments.parquet",
        "survey.yaml",
    ]
    back = load_clock_survey(tmp_path / "bundle")
    _assert_same(back, survey)
    # A row read back names the file it was read from, as any row read from a file does.
    assert {(r.source, r.provenance) for r in back.rows} == {("clock_corrections.csv", "measured")}
    _assert_same(load_clock_survey(target), survey)


def test_a_survey_without_rows_writes_no_table_and_removes_an_old_one(
    tmp_path: Path, survey: ClockSurvey, rowless: ClockSurvey
) -> None:
    assert not rowless.rows
    save_clock_survey(survey, tmp_path)
    target = save_clock_survey(rowless, tmp_path)
    assert not (target / "clock_corrections.csv").exists()
    back = load_clock_survey(tmp_path)
    assert back.rows == ()
    _assert_same(back, rowless)


def test_empty_tables_keep_their_types(tmp_path: Path, survey: ClockSurvey) -> None:
    assert survey.faults.empty
    save_clock_survey(survey, tmp_path)
    back = load_clock_survey(tmp_path)
    assert back.faults.empty
    assert dict(back.faults.dtypes.astype(str)) == dict(survey.faults.dtypes.astype(str))


def test_a_file_where_the_bundle_should_be_is_refused(tmp_path: Path, survey: ClockSurvey) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")
    with pytest.raises(TsaraBundleError, match="is not a directory"):
        save_clock_survey(survey, tmp_path / "file")


def test_a_table_without_its_columns_is_refused_before_anything_is_written(
    tmp_path: Path, survey: ClockSurvey
) -> None:
    edited = dataclasses.replace(survey, segments=survey.segments.drop(columns=["status"]))
    with pytest.raises(TsaraBundleError, match="segments table has columns"):
        save_clock_survey(edited, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_a_missing_survey_or_part_is_refused(tmp_path: Path, survey: ClockSurvey) -> None:
    with pytest.raises(TsaraBundleError, match="No clock survey"):
        load_clock_survey(tmp_path)
    target = save_clock_survey(survey, tmp_path)
    (target / "segments.parquet").unlink()
    with pytest.raises(TsaraBundleError, match=r"lacks its segments \(segments.parquet\)"):
        load_clock_survey(tmp_path)


def test_a_table_is_saved_with_its_declared_types(tmp_path: Path, survey: ClockSurvey) -> None:
    """A table whose types drifted (a tier read back as float) is written as declared."""
    drifted = dataclasses.replace(survey, segments=survey.segments.astype({"tier": "float64"}))
    save_clock_survey(drifted, tmp_path)
    assert str(load_clock_survey(tmp_path).segments.tier.dtype) == "Int64"
