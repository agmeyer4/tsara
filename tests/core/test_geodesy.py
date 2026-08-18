"""Tests for the shared geodesy primitives.

These cover the metre<->degree mapping, coordinate bounding, and track
interpolation — the parts of position handling that real ingested GPS
(Phase 4) needs just as much as the synthetic track generator does. Tests
for *manufacturing* a track live in ``tests/synthetic/test_platform.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tsara.core.geodesy import (
    METERS_PER_DEGREE,
    METERS_PER_DEGREE_LAT,
    MIN_LONGITUDE_SCALE_M,
    clamp_latitude,
    meters_per_degree_longitude,
    positions_at,
    wrap_longitude,
)

# ---------------------------------------------------------------------------
# The metre <-> degree mapping
# ---------------------------------------------------------------------------

#: WGS-84 semi-major axis, for deriving the equatorial degree independently.
WGS84_SEMI_MAJOR_M = 6_378_137.0


def test_constant_is_the_wgs84_equatorial_degree_of_longitude() -> None:
    """The constant must be what its docstring claims it is.

    It was previously documented as the mean *meridional* degree (111132.9 m),
    which it never was. Pinning it against the independent derivation keeps
    the label and the value from drifting apart again.
    """
    expected = 2.0 * math.pi * WGS84_SEMI_MAJOR_M / 360.0
    assert METERS_PER_DEGREE == pytest.approx(expected, abs=1.0)


def test_legacy_alias_still_resolves() -> None:
    assert METERS_PER_DEGREE_LAT == METERS_PER_DEGREE


def test_longitude_scale_shrinks_with_latitude() -> None:
    """One degree of longitude is a shorter distance away from the equator."""
    assert meters_per_degree_longitude(0.0) == pytest.approx(METERS_PER_DEGREE)
    assert meters_per_degree_longitude(60.0) == pytest.approx(METERS_PER_DEGREE * 0.5, rel=1e-6)
    assert meters_per_degree_longitude(60.0) < meters_per_degree_longitude(30.0)


def test_longitude_scale_is_floored_at_the_pole() -> None:
    """cos(latitude) collapses at the pole; the floor keeps arithmetic finite."""
    assert meters_per_degree_longitude(90.0) == MIN_LONGITUDE_SCALE_M


# ---------------------------------------------------------------------------
# Coordinate bounding
# ---------------------------------------------------------------------------


def test_wrap_longitude_brings_antimeridian_crossings_back_in_range() -> None:
    """An unbounded metre offset can run past 180; 180.08 is not a coordinate."""
    wrapped = wrap_longitude(np.array([179.9, 180.08, -180.5, 0.0, 360.0]))
    assert np.all(wrapped >= -180.0)
    assert np.all(wrapped < 180.0)
    assert wrapped[0] == pytest.approx(179.9)
    assert wrapped[1] == pytest.approx(-179.92)
    assert wrapped[2] == pytest.approx(179.5)
    assert wrapped[3] == pytest.approx(0.0)


def test_wrap_longitude_leaves_valid_values_untouched() -> None:
    values = np.array([-179.0, -111.0, 0.0, 111.0, 179.0])
    assert wrap_longitude(values) == pytest.approx(values)


def test_clamp_latitude_bounds_without_wrapping() -> None:
    """Latitude does not wrap: crossing a pole is unrepresentable here."""
    clamped = clamp_latitude(np.array([-95.0, -90.0, 40.0, 90.0, 91.5]))
    assert clamped[0] == pytest.approx(-90.0)
    assert clamped[2] == pytest.approx(40.0)
    assert clamped[4] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def test_positions_at_interpolates_between_samples() -> None:
    track_times = pd.DatetimeIndex(["2026-01-01 00:00:00", "2026-01-01 00:00:10"])
    lat = np.array([40.0, 41.0])
    lon = np.array([-111.0, -110.0])

    got_lat, got_lon = positions_at(
        pd.DatetimeIndex(["2026-01-01 00:00:05"]), track_times, lat, lon
    )
    assert got_lat[0] == pytest.approx(40.5)
    assert got_lon[0] == pytest.approx(-110.5)


def test_positions_outside_the_track_clamp_to_its_endpoints() -> None:
    track_times = pd.DatetimeIndex(["2026-01-01 00:00:00", "2026-01-01 00:00:10"])
    lat = np.array([40.0, 41.0])
    lon = np.array([-111.0, -110.0])

    got_lat, _ = positions_at(pd.DatetimeIndex(["2025-01-01", "2027-01-01"]), track_times, lat, lon)
    assert got_lat[0] == pytest.approx(40.0)
    assert got_lat[1] == pytest.approx(41.0)


def test_positions_at_handles_timezone_aware_targets() -> None:
    """Conversion goes through timebase, so tz-aware input must not shift."""
    track_times = pd.DatetimeIndex(["2026-01-01 00:00:00", "2026-01-01 00:00:10"])
    lat = np.array([40.0, 41.0])
    lon = np.array([-111.0, -110.0])

    naive, _ = positions_at(pd.DatetimeIndex(["2026-01-01 00:00:05"]), track_times, lat, lon)
    aware, _ = positions_at(
        pd.DatetimeIndex(["2026-01-01 00:00:05"]).tz_localize("UTC"), track_times, lat, lon
    )
    assert naive[0] == pytest.approx(aware[0])
