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
from tsara.ingest.uncertainty import (
    ResolvedUncertainty,
    resolve_uncertainty,
)

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
    assert resolved.random_provenance == "declared"


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
    assert resolved.random_provenance == "reported"


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
    assert resolved.random_provenance == "empirical"
    assert resolved.systematic_provenance == "unknown"
    assert resolved.provenance == "empirical"


def test_omitted_systematic_with_a_budget_is_zero_not_unknown() -> None:
    """A declared budget that omits systematic is a statement, not a silence."""
    values = _series([1900.0])
    spec = _spec(random={"mode": "declared", "absolute": 0.7})
    resolved = _resolve(values, spec, _frame(values))

    assert resolved.systematic_provenance == "zero"
    assert resolved.random_provenance == "declared"


def test_omitted_random_with_a_budget_falls_back_to_empirical() -> None:
    values = _series([1900.0])
    spec = _spec(systematic={"mode": "declared", "relative": 0.01})
    resolved = _resolve(values, spec, _frame(values))

    assert resolved.random is None
    assert resolved.random_provenance == "empirical"
    assert resolved.systematic_provenance == "declared"


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
    assert resolved.provenance == "declared"


def test_mixed_modes_report_mixed() -> None:
    """The shipped example manifest does exactly this."""
    values = _series([1000.0])
    frame = _frame(values, SIG=[0.5])
    spec = _spec(
        random={"mode": "reported", "column": "SIG"},
        systematic={"mode": "declared", "relative": 0.01},
    )
    resolved = _resolve(values, spec, frame)

    assert resolved.random_provenance == "reported"
    assert resolved.systematic_provenance == "declared"
    assert resolved.provenance == "mixed"


def test_both_reported_reports_reported() -> None:
    values = _series([1000.0])
    frame = _frame(values, A=[0.5], B=[1.0])
    spec = _spec(
        random={"mode": "reported", "column": "A"},
        systematic={"mode": "reported", "column": "B"},
    )
    assert _resolve(values, spec, frame).provenance == "reported"


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


# ---------------------------------------------------------------------------
# A declared sigma quoted at a different interval (Phase 3.5)
#
# Ingestion records the declaration and changes nothing. Moving a sigma onto
# another support needs a decorrelation timescale and an averaging model, so
# it belongs to the stage that wants a sigma at a particular support
# (METHODS 3.4, 10.8). These tests exist to keep that boundary where it is:
# the one that would fail loudest if a correction crept back in is
# `test_the_declared_figures_are_stored_untouched`.
# ---------------------------------------------------------------------------


def _widths(n: int = 60, seconds: float = 60.0) -> np.ndarray:
    return np.full(n, int(seconds * 1_000_000_000), dtype=np.int64)


def _quoted(at_width: str | None = "1s", **spec: Any) -> UncertaintySpec:
    random: dict[str, Any] = {"mode": "declared", "absolute": 1.0}
    if at_width is not None:
        random["at_width"] = at_width
    return UncertaintySpec.model_validate({"random": random, **spec})


def _resolve_quoted(
    spec: UncertaintySpec, widths: np.ndarray | None, n: int = 3
) -> ResolvedUncertainty:
    values = pd.Series([100.0] * n, index=pd.date_range("2026-01-01", periods=n, freq="60s"))
    return resolve_uncertainty(
        values,
        spec,
        pd.DataFrame(index=values.index),
        conversion=None,
        variable="ch4",
        path=PATH,
        cell_width_ns=widths,
    )


def test_the_declared_figures_are_stored_untouched() -> None:
    """The whole point of the boundary: a 1 s figure on 60 s cells is 60 s
    cells' worth of mismatch, and ingestion still stores 1.0."""
    resolved = _resolve_quoted(_quoted(decorrelation_timescale="1ns"), _widths(3))
    assert resolved.random is not None
    assert np.all(resolved.random == 1.0), "no sigma is rescaled at ingestion"


def test_the_quoted_interval_is_recorded_verbatim() -> None:
    resolved = _resolve_quoted(_quoted("1s"), _widths(3))
    assert resolved.at_width == "1s"


def test_the_ratio_is_how_many_quoted_intervals_fit_in_a_cell() -> None:
    """N of METHODS 3.4 -- the count a later stage corrects to N_eff."""
    resolved = _resolve_quoted(_quoted("1s"), _widths(3, seconds=60.0))
    assert resolved.at_width_ratio == pytest.approx(60.0)


def test_the_ratio_takes_the_median_of_varying_widths() -> None:
    """Widths are per row, and a duty-cycled sampler's genuinely vary. The
    median is not the mean here: a single long cell must not drag the summary
    of the other two with it."""
    widths = np.array([60, 60, 600], dtype=np.int64) * 1_000_000_000
    resolved = _resolve_quoted(_quoted("1s"), widths)
    assert resolved.at_width_ratio == pytest.approx(60.0), "the mean would be 240"


def test_a_finer_cell_gives_a_ratio_below_one_rather_than_a_refusal() -> None:
    """Cells finer than the quoted interval are a real configuration, not an
    error: the figure simply describes a longer interval than a cell. The
    number says so and nothing is inflated to compensate."""
    resolved = _resolve_quoted(_quoted("60s"), _widths(3, seconds=1.0))
    assert resolved.at_width_ratio == pytest.approx(1.0 / 60.0)
    assert resolved.random is not None
    assert np.all(resolved.random == 1.0)


def test_a_stream_without_cells_records_the_declaration_anyway(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Nothing to compare against is not nothing to say."""
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.uncertainty"):
        resolved = _resolve_quoted(_quoted("1s"), None)
    assert resolved.at_width == "1s"
    assert resolved.at_width_ratio is None
    assert caplog.text == ""


def test_a_mismatch_is_warned_about_and_says_where_it_gets_resolved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.uncertainty"):
        _resolve_quoted(_quoted("1s"), _widths(3))
    assert "stored exactly as declared" in caplog.text
    assert "60" in caplog.text, "the size of the mismatch is in the message"


def test_a_figure_quoted_at_its_own_cell_is_not_worth_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.uncertainty"):
        resolved = _resolve_quoted(_quoted("60s"), _widths(3))
    assert resolved.at_width_ratio == pytest.approx(1.0)
    assert caplog.text == ""


def test_an_undeclared_interval_records_nothing() -> None:
    resolved = _resolve_quoted(_quoted(at_width=None), _widths(3))
    assert resolved.at_width is None
    assert resolved.at_width_ratio is None


def test_a_systematic_interval_is_recorded_and_can_never_be_acted_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A systematic error does not average down, so no interval can change it
    (METHODS 3.3). Recording it anyway keeps the product as informative as
    the manifest that produced it."""
    spec = UncertaintySpec.model_validate(
        {
            "random": {"mode": "declared", "absolute": 1.0},
            "systematic": {"mode": "declared", "relative": 0.01, "at_width": "1s"},
        }
    )
    with caplog.at_level(logging.WARNING, logger="tsara.ingest.uncertainty"):
        resolved = _resolve_quoted(spec, _widths(3))
    assert resolved.systematic_at_width == "1s"
    assert resolved.at_width is None, "the random component declared none"
    assert "does not average down" in caplog.text
    assert resolved.systematic is not None
    assert resolved.systematic[0] == pytest.approx(1.0)


def test_a_reported_component_carries_no_interval() -> None:
    """A per-point sigma column describes the rows it sits beside, so there
    is no second interval to reconcile."""
    spec = UncertaintySpec.model_validate({"random": {"mode": "reported", "column": "sigma"}})
    values = pd.Series([100.0, 100.0], index=pd.date_range("2026-01-01", periods=2, freq="60s"))
    resolved = resolve_uncertainty(
        values,
        spec,
        pd.DataFrame({"sigma": [0.5, 0.5]}, index=values.index),
        conversion=None,
        variable="ch4",
        path=PATH,
        cell_width_ns=_widths(2),
    )
    assert resolved.at_width is None
    assert resolved.at_width_ratio is None
