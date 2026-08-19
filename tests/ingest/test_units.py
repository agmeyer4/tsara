"""Tests for unit conversion.

The only interesting behaviour here is the asymmetry between converting a
value and converting a spread: the offset applies to one and not the other.
Getting that wrong is silent and catastrophic in the degC->K case, where
every uncertainty would gain 273.15.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsara.config.manifest import UnitConversion
from tsara.ingest.units import canonical_units, convert_spread, convert_values


def _ppm_to_ppb() -> UnitConversion:
    return UnitConversion(from_unit="ppm", to_unit="ppb", scale=1000.0)


def _celsius_to_kelvin() -> UnitConversion:
    return UnitConversion(from_unit="degC", to_unit="K", offset=273.15)


def test_scale_is_applied() -> None:
    result = convert_values(np.array([1.9, 2.0]), _ppm_to_ppb())
    assert result.tolist() == [1900.0, 2000.0]


def test_offset_is_applied() -> None:
    assert convert_values(np.array([0.0]), _celsius_to_kelvin())[0] == pytest.approx(273.15)


def test_scale_then_offset_ordering() -> None:
    """canonical = native * scale + offset, in that order."""
    conversion = UnitConversion(from_unit="a", to_unit="b", scale=2.0, offset=1.0)
    assert convert_values(np.array([3.0]), conversion)[0] == 7.0


def test_no_conversion_returns_values_unchanged() -> None:
    values = np.array([1.0, 2.0])
    assert convert_values(values, None) is values


def test_nan_survives_conversion() -> None:
    """Masked samples must stay masked."""
    assert np.isnan(convert_values(np.array([np.nan]), _ppm_to_ppb())[0])


def test_spread_takes_the_scale() -> None:
    assert convert_spread(np.array([0.001]), _ppm_to_ppb())[0] == pytest.approx(1.0)


def test_spread_ignores_the_offset() -> None:
    """A spread is a difference on the axis, so the origin cancels."""
    assert convert_spread(np.array([0.5]), _celsius_to_kelvin())[0] == 0.5


def test_spread_stays_positive_under_a_negative_scale() -> None:
    """A sign-flip convention must not produce a negative 1-sigma."""
    conversion = UnitConversion(from_unit="a", to_unit="b", scale=-2.0)
    assert convert_spread(np.array([3.0]), conversion)[0] == 6.0


def test_spread_without_conversion_is_unchanged() -> None:
    spread = np.array([1.0])
    assert convert_spread(spread, None) is spread


def test_canonical_units_reports_the_stored_units() -> None:
    assert canonical_units("ppm", _ppm_to_ppb()) == "ppb"
    assert canonical_units("ppb", None) == "ppb"
