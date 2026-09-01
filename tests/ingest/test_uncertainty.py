"""Tests for uncertainty resolution.

The property under test throughout is METHODS §2.3: no uncertainty of
unstated origin ever enters the pipeline. Concretely that means every
component comes back with a provenance label, ``zero`` and ``unknown`` stay
distinguishable, and nothing is silently invented for a variable whose
manifest says nothing about it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tsara.config.manifest import UncertaintySpec, UnitConversion
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.uncertainty import ResolvedUncertainty, resolve_uncertainty

PATH = Path("f.dat")


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=len(values), freq="1s", name="time")
    return pd.Series(values, index=index, dtype="float64")


def _frame(series: pd.Series, **columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"value": series.to_numpy(), **columns}, index=series.index)


def _spec(**kwargs: Any) -> UncertaintySpec:
    return UncertaintySpec.model_validate(kwargs)


def _resolve(
    values: pd.Series, spec: UncertaintySpec | None, frame: pd.DataFrame, **kw: Any
) -> ResolvedUncertainty:
    kw.setdefault("conversion", None)
    return resolve_uncertainty(values, spec, frame, variable="ch4", path=PATH, **kw)


# ---------------------------------------------------------------------------
# Declared
# ---------------------------------------------------------------------------


def test_declared_absolute_only() -> None:
    values = _series([1900.0, 2000.0])
    spec = _spec(random={"mode": "declared", "absolute": 0.7})
    resolved = _resolve(values, spec, _frame(values))

    assert resolved.random is not None
    assert resolved.random.tolist() == pytest.approx([0.7, 0.7])
    assert resolved.random_source == "declared"


def test_declared_relative_scales_with_the_reading() -> None:
    values = _series([100.0, 200.0])
    spec = _spec(random={"mode": "declared", "relative": 0.01})
    resolved = _resolve(values, spec, _frame(values))
    assert resolved.random is not None
    assert resolved.random.tolist() == pytest.approx([1.0, 2.0])


def test_declared_terms_combine_in_quadrature() -> None:
    values = _series([100.0])
    spec = _spec(random={"mode": "declared", "absolute": 3.0, "relative": 0.04})
    resolved = _resolve(values, spec, _frame(values))
    assert resolved.random is not None
    assert resolved.random[0] == pytest.approx(5.0)  # sqrt(3^2 + 4^2)


def test_declared_uses_magnitude_so_sigma_is_never_negative() -> None:
    values = _series([-100.0])
    spec = _spec(random={"mode": "declared", "relative": 0.01})
    resolved = _resolve(values, spec, _frame(values))
    assert resolved.random is not None
    assert resolved.random[0] == pytest.approx(1.0)


def test_declared_absolute_is_not_converted_twice() -> None:
    """`absolute` is declared in canonical units and values are already canonical."""
    values = _series([1900.0])
    spec = _spec(random={"mode": "declared", "absolute": 0.7})
    conversion = UnitConversion(from_unit="ppm", to_unit="ppb", scale=1000.0)
    resolved = _resolve(values, spec, _frame(values), conversion=conversion)
    assert resolved.random is not None
    assert resolved.random[0] == pytest.approx(0.7)


def test_masked_value_yields_masked_sigma() -> None:
    values = _series([1900.0, np.nan])
    spec = _spec(random={"mode": "declared", "absolute": 0.7, "relative": 0.01})
    resolved = _resolve(values, spec, _frame(values))
    assert resolved.random is not None
    assert np.isnan(resolved.random[1])


# ---------------------------------------------------------------------------
# Reported
# ---------------------------------------------------------------------------


def test_reported_reads_the_column() -> None:
    values = _series([1900.0, 1901.0])
    frame = _frame(values, CH4_1SIGMA=[0.5, 0.6])
    spec = _spec(random={"mode": "reported", "column": "CH4_1SIGMA"})
    resolved = _resolve(values, spec, frame)

    assert resolved.random is not None
    assert resolved.random.tolist() == pytest.approx([0.5, 0.6])
    assert resolved.random_source == "reported"


def test_reported_column_is_scaled_but_not_offset() -> None:
    """METHODS §2.2: a spread has no origin, so the offset never applies."""
    values = _series([300.0])
    frame = _frame(values, SIG=[0.5])
    spec = _spec(random={"mode": "reported", "column": "SIG"})
    conversion = UnitConversion(from_unit="degC", to_unit="K", offset=273.15)
    resolved = _resolve(values, spec, frame, conversion=conversion)
    assert resolved.random is not None
    assert resolved.random[0] == pytest.approx(0.5)


def test_reported_column_takes_the_scale() -> None:
    values = _series([1.9])
    frame = _frame(values, SIG=[0.0007])
    spec = _spec(random={"mode": "reported", "column": "SIG"})
    conversion = UnitConversion(from_unit="ppm", to_unit="ppb", scale=1000.0)
    resolved = _resolve(values, spec, frame, conversion=conversion)
    assert resolved.random is not None
    assert resolved.random[0] == pytest.approx(0.7)


def test_missing_reported_column_is_an_error() -> None:
    values = _series([1900.0])
    spec = _spec(random={"mode": "reported", "column": "ABSENT"})
    with pytest.raises(TsaraIngestError, match="reported column 'ABSENT'"):
        _resolve(values, spec, _frame(values))


def test_negative_reported_sigma_is_masked_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Almost always an undeclared missing-value sentinel, not a real spread."""
    values = _series([1900.0, 1901.0])
    frame = _frame(values, SIG=[0.5, -9999.0])
    spec = _spec(random={"mode": "reported", "column": "SIG"})
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.uncertainty"):
        resolved = _resolve(values, spec, frame)

    assert resolved.random is not None
    assert np.isnan(resolved.random[1])
    assert "negative value" in caplog.text


def test_reported_sigma_is_masked_where_the_value_is() -> None:
    """A sigma without a measurement could re-enter a weighted fit."""
    values = _series([1900.0, np.nan])
    frame = _frame(values, SIG=[0.5, 0.6])
    spec = _spec(random={"mode": "reported", "column": "SIG"})
    resolved = _resolve(values, spec, frame)
    assert resolved.random is not None
    assert np.isnan(resolved.random[1])


def test_non_numeric_reported_values_become_nan() -> None:
    values = _series([1900.0, 1901.0])
    frame = pd.DataFrame({"SIG": ["0.5", "bad"]}, index=values.index)
    spec = _spec(random={"mode": "reported", "column": "SIG"})
    resolved = _resolve(values, spec, frame)
    assert resolved.random is not None
    assert np.isnan(resolved.random[1])


# ---------------------------------------------------------------------------
# Provenance: the part that must never lie
# ---------------------------------------------------------------------------


def test_no_budget_means_empirical_random_and_unknown_systematic() -> None:
    """diff_mad differences the signal, so it is blind to systematic error."""
    values = _series([1900.0])
    resolved = _resolve(values, None, _frame(values))

    assert resolved.random is None
    assert resolved.systematic is None
    assert resolved.random_source == "empirical"
    assert resolved.systematic_source == "unknown"
    assert resolved.source == "empirical"


def test_omitted_systematic_with_a_budget_is_zero_not_unknown() -> None:
    """A declared budget that omits systematic is a statement, not a silence."""
    values = _series([1900.0])
    spec = _spec(random={"mode": "declared", "absolute": 0.7})
    resolved = _resolve(values, spec, _frame(values))

    assert resolved.systematic_source == "zero"
    assert resolved.random_source == "declared"


def test_omitted_random_with_a_budget_falls_back_to_empirical() -> None:
    values = _series([1900.0])
    spec = _spec(systematic={"mode": "declared", "relative": 0.01})
    resolved = _resolve(values, spec, _frame(values))

    assert resolved.random is None
    assert resolved.random_source == "empirical"
    assert resolved.systematic_source == "declared"


def test_both_components_declared() -> None:
    values = _series([1000.0])
    spec = _spec(
        random={"mode": "declared", "absolute": 0.7},
        systematic={"mode": "declared", "relative": 0.01},
    )
    resolved = _resolve(values, spec, _frame(values))

    assert resolved.random is not None
    assert resolved.systematic is not None
    assert resolved.systematic[0] == pytest.approx(10.0)
    assert resolved.source == "declared"


def test_mixed_modes_report_mixed() -> None:
    """The shipped example manifest does exactly this."""
    values = _series([1000.0])
    frame = _frame(values, SIG=[0.5])
    spec = _spec(
        random={"mode": "reported", "column": "SIG"},
        systematic={"mode": "declared", "relative": 0.01},
    )
    resolved = _resolve(values, spec, frame)

    assert resolved.random_source == "reported"
    assert resolved.systematic_source == "declared"
    assert resolved.source == "mixed"


def test_both_reported_reports_reported() -> None:
    values = _series([1000.0])
    frame = _frame(values, A=[0.5], B=[1.0])
    spec = _spec(
        random={"mode": "reported", "column": "A"},
        systematic={"mode": "reported", "column": "B"},
    )
    assert _resolve(values, spec, frame).source == "reported"


def test_components_are_kept_separate() -> None:
    """METHODS §2.1: they average down differently, so they never merge here."""
    values = _series([1000.0])
    spec = _spec(
        random={"mode": "declared", "absolute": 3.0},
        systematic={"mode": "declared", "absolute": 4.0},
    )
    resolved = _resolve(values, spec, _frame(values))

    assert resolved.random is not None
    assert resolved.systematic is not None
    assert resolved.random[0] == pytest.approx(3.0)
    assert resolved.systematic[0] == pytest.approx(4.0)
    # Emphatically not 5.0 anywhere in the returned object.


def test_decorrelation_timescale_is_passed_through() -> None:
    values = _series([1900.0])
    spec = _spec(random={"mode": "declared", "absolute": 0.7}, decorrelation_timescale="5min")
    assert _resolve(values, spec, _frame(values)).decorrelation_timescale == "5min"


def test_decorrelation_timescale_defaults_to_none() -> None:
    values = _series([1900.0])
    spec = _spec(random={"mode": "declared", "absolute": 0.7})
    assert _resolve(values, spec, _frame(values)).decorrelation_timescale is None


def test_negative_sentinel_is_masked_even_under_a_unit_conversion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The guard has to run before the conversion, not after.

    `convert_spread` takes an absolute value — correctly, since a negative
    `scale` is a legitimate sign-convention flip. A `sigma < 0` test applied
    afterwards therefore finds nothing, so the sentinel guard worked only for
    variables with no conversion: precisely the wrong way round, since it
    failed where the manifest was doing more work. A -9999 under a ppm->ppb
    conversion silently became a 9,999,000 ppb "1-sigma", which drives a
    random component's inverse-variance weight to nothing and, for a
    systematic component combined as a weighted mean of sigmas, dominates the
    whole bin.
    """
    values = _series([1.9, 1.95, 2.0])
    frame = _frame(values, SIG=[0.002, -9999.0, 0.003])
    spec = _spec(random={"mode": "reported", "column": "SIG"})
    conversion = UnitConversion(from_unit="ppm", to_unit="ppb", scale=1000.0)

    with caplog.at_level(logging.WARNING, logger="tsara.ingest.uncertainty"):
        resolved = _resolve(values, spec, frame, conversion=conversion)

    assert resolved.random is not None
    assert bool(np.isnan(resolved.random[1]))
    # The surviving points still take the scale, so the fix did not disable it.
    assert resolved.random[0] == pytest.approx(2.0)
    assert resolved.random[2] == pytest.approx(3.0)
    assert "negative value" in caplog.text


def test_negative_sentinel_masking_survives_a_negative_scale() -> None:
    """A sign-convention flip must still yield a positive spread.

    Masking negatives *before* conversion could have broken this: the check
    is about the reported sentinel, not about the sign the conversion
    produces, and a legitimate negative scale still has to come out positive.
    """
    values = _series([1.0, 2.0])
    frame = _frame(values, SIG=[0.5, -9999.0])
    spec = _spec(random={"mode": "reported", "column": "SIG"})
    conversion = UnitConversion(from_unit="a", to_unit="b", scale=-2.0)
    resolved = _resolve(values, spec, frame, conversion=conversion)

    assert resolved.random is not None
    assert resolved.random[0] == pytest.approx(1.0)
    assert bool(np.isnan(resolved.random[1]))
