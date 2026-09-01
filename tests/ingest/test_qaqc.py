"""Tests for QA/QC masking.

Two properties are asserted throughout because everything downstream depends
on them: rejected samples become ``NaN`` rather than disappearing, and each
rule reports what *it* rejected rather than contributing to one opaque total.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsara.config.manifest import FlagRule, QAQCRule, RangeRule
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.qaqc import apply_qaqc, masked_fraction

PATH = Path("f.dat")


def _series(values: list[float], freq: str = "1s") -> pd.Series:
    index = pd.date_range("2026-01-01", periods=len(values), freq=freq, name="time")
    return pd.Series(values, index=index, dtype="float64")


def _frame(series: pd.Series, **columns: list[object]) -> pd.DataFrame:
    return pd.DataFrame({"value": series.to_numpy(), **columns}, index=series.index)


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------


def test_range_masks_below_minimum() -> None:
    values = _series([1500.0, 1900.0, 2000.0])
    masked, reports = apply_qaqc(
        values, [RangeRule(min=1700.0)], _frame(values), variable="ch4", path=PATH
    )
    assert masked.isna().tolist() == [True, False, False]
    assert reports[0].n_masked == 1


def test_range_masks_above_maximum() -> None:
    values = _series([1900.0, 99999.0])
    masked, _ = apply_qaqc(
        values, [RangeRule(max=10000.0)], _frame(values), variable="ch4", path=PATH
    )
    assert masked.isna().tolist() == [False, True]


def test_range_bounds_are_inclusive_at_the_limit() -> None:
    """'Mask values strictly below min' means the bound itself survives."""
    values = _series([1700.0])
    masked, _ = apply_qaqc(
        values, [RangeRule(min=1700.0)], _frame(values), variable="ch4", path=PATH
    )
    assert masked.notna().all()


def test_masking_preserves_row_count() -> None:
    """Rows are the instrument's clock; a rule must not shorten the record."""
    values = _series([1.0, 2.0, 3.0])
    masked, _ = apply_qaqc(values, [RangeRule(min=2.5)], _frame(values), variable="x", path=PATH)
    assert len(masked) == 3
    assert masked.index.equals(values.index)


def test_input_series_is_not_mutated() -> None:
    values = _series([1.0, 2.0])
    apply_qaqc(values, [RangeRule(min=1.5)], _frame(values), variable="x", path=PATH)
    assert values.notna().all()


# ---------------------------------------------------------------------------
# Flag
# ---------------------------------------------------------------------------


def test_flag_good_values_keeps_only_those() -> None:
    values = _series([1.0, 2.0, 3.0])
    frame = _frame(values, QC=[0, 1, 0])
    rule = FlagRule(flag_column="QC", good_values=[0])
    masked, _ = apply_qaqc(values, [rule], frame, variable="x", path=PATH)
    assert masked.isna().tolist() == [False, True, False]


def test_flag_bad_values_masks_only_those() -> None:
    values = _series([1.0, 2.0, 3.0])
    frame = _frame(values, QC=[0, 1, 2])
    rule = FlagRule(flag_column="QC", bad_values=[1])
    masked, _ = apply_qaqc(values, [rule], frame, variable="x", path=PATH)
    assert masked.isna().tolist() == [False, True, False]


def test_unknown_flag_is_not_evidence_of_a_good_sample() -> None:
    """With good_values declared, anything unlisted is rejected."""
    values = _series([1.0, 2.0])
    frame = _frame(values, QC=[0, 7])
    rule = FlagRule(flag_column="QC", good_values=[0])
    masked, _ = apply_qaqc(values, [rule], frame, variable="x", path=PATH)
    assert masked.isna().tolist() == [False, True]


def test_flag_reads_a_different_column_than_the_masked_one() -> None:
    """The status word is never the converted quantity."""
    values = _series([1.0, 2.0])
    frame = _frame(values, ALARM_STATUS=[0, 3])
    rule = FlagRule(flag_column="ALARM_STATUS", bad_values=[3])
    masked, _ = apply_qaqc(values, [rule], frame, variable="ch4", path=PATH)
    assert masked.isna().tolist() == [False, True]


def test_missing_flag_column_is_an_error() -> None:
    values = _series([1.0])
    rule = FlagRule(flag_column="NOPE", good_values=[0])
    with pytest.raises(TsaraIngestError, match="names column 'NOPE'"):
        apply_qaqc(values, [rule], _frame(values), variable="ch4", path=PATH)


def test_string_flag_values() -> None:
    values = _series([1.0, 2.0])
    frame = _frame(values, STATUS=["ok", "bad"])
    rule = FlagRule(flag_column="STATUS", good_values=["ok"])
    masked, _ = apply_qaqc(values, [rule], frame, variable="x", path=PATH)
    assert masked.isna().tolist() == [False, True]


# ---------------------------------------------------------------------------
# Order independence
# ---------------------------------------------------------------------------


def test_rules_tolerate_an_unsorted_index() -> None:
    """Every remaining rule is pointwise, so record order cannot change them.

    This used to be true only of `range` and `flag`, with the rolling `spike`
    test needing a monotonic index and raising without one. With that rule
    removed (see the qaqc module docstring), order independence is a property
    of QA/QC as a whole -- sorting is purely the orchestration stage's job.
    """
    values = _series([1.0, 2.0, 3.0]).iloc[np.array([0, 2, 1])]
    masked, _ = apply_qaqc(values, [RangeRule(min=2.5)], _frame(values), variable="x", path=PATH)
    assert int(masked.notna().sum()) == 1


# ---------------------------------------------------------------------------
# Composition and reporting
# ---------------------------------------------------------------------------


def test_rules_compose_by_union() -> None:
    values = _series([1500.0, 1900.0, 99999.0])
    rules = [RangeRule(min=1700.0), RangeRule(max=10000.0)]
    masked, reports = apply_qaqc(values, rules, _frame(values), variable="ch4", path=PATH)
    assert masked.isna().tolist() == [True, False, True]
    assert [r.n_masked for r in reports] == [1, 1]


def test_overlapping_rules_do_not_double_count() -> None:
    """Two rules rejecting one sample must not each claim it."""
    values = _series([1.0, 2.0])
    rules = [RangeRule(min=1.5), RangeRule(min=1.5)]
    _, reports = apply_qaqc(values, rules, _frame(values), variable="x", path=PATH)
    assert [r.n_masked for r in reports] == [1, 0]


def test_reports_are_one_per_rule_in_declared_order() -> None:
    values = _series([1.0, 2.0, 3.0])
    frame = _frame(values, QC=[0, 0, 1])
    rules: list[QAQCRule] = [RangeRule(min=1.5), FlagRule(flag_column="QC", good_values=[0])]
    _, reports = apply_qaqc(values, rules, frame, variable="x", path=PATH)
    assert [r.kind for r in reports] == ["range", "flag"]


def test_report_fraction() -> None:
    values = _series([1.0, 2.0, 3.0, 4.0])
    _, reports = apply_qaqc(values, [RangeRule(min=3.5)], _frame(values), variable="x", path=PATH)
    assert reports[0].fraction == pytest.approx(0.75)


def test_report_fraction_of_an_empty_record() -> None:
    values = _series([])
    _, reports = apply_qaqc(values, [RangeRule(min=1.0)], _frame(values), variable="x", path=PATH)
    assert reports[0].fraction == 0.0


def test_no_rules_is_a_no_op() -> None:
    values = _series([1.0, 2.0])
    masked, reports = apply_qaqc(values, [], _frame(values), variable="x", path=PATH)
    assert masked.equals(values)
    assert reports == ()


def test_mostly_masked_records_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Usually a units or polarity mistake, and worth catching before a full run."""
    values = _series([1.0, 2.0, 3.0, 4.0])
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.qaqc"):
        apply_qaqc(values, [RangeRule(min=3.5)], _frame(values), variable="ch4", path=PATH)
    assert "left only 1 of 4 samples" in caplog.text
    assert "range" in caplog.text


def test_lightly_masked_records_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    values = _series([1.0, 2.0, 3.0, 4.0])
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.qaqc"):
        apply_qaqc(values, [RangeRule(min=1.5)], _frame(values), variable="ch4", path=PATH)
    assert "left only" not in caplog.text


def test_empty_record_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    values = _series([])
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.qaqc"):
        apply_qaqc(values, [RangeRule(min=1.0)], _frame(values), variable="x", path=PATH)
    assert "left only" not in caplog.text


def test_masked_fraction() -> None:
    assert masked_fraction(_series([1.0, np.nan, 3.0, np.nan])) == pytest.approx(0.5)
    assert masked_fraction(_series([])) == 0.0
