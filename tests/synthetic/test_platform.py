"""Tests for fixed-site coordinates and synthetic mobile tracks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tsara.synthetic.config import MobileTrack
from tsara.synthetic.platform import (
    METERS_PER_DEGREE_LAT,
    build_track,
    positions_at,
)


def _times(n: int, freq: str = "1s") -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq=freq)


def _track(**overrides: object) -> MobileTrack:
    fields: dict[str, object] = {
        "kind": "mobile",
        "start_latitude": 40.0,
        "start_longitude": -111.0,
    }
    fields.update(overrides)
    return MobileTrack(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------


def test_circuit_starts_at_the_configured_coordinates() -> None:
    lat, lon = build_track(
        _track(pattern="circuit", radius_m=500.0), _times(10), np.random.default_rng(0)
    )
    assert lat[0] == pytest.approx(40.0)
    assert lon[0] == pytest.approx(-111.0)


def test_circuit_returns_to_its_start_after_one_lap() -> None:
    """Recurrence is what makes circuit tracks spatially clusterable."""
    radius_m = 500.0
    speed = 10.0
    period_s = 2.0 * math.pi * radius_m / speed
    track = _track(pattern="circuit", radius_m=radius_m, speed_m_s=speed)
    times = _times(int(round(period_s)) + 1)

    lat, lon = build_track(track, times, np.random.default_rng(0))
    # Compared in metres: the lap period (314.159 s) is not an integer number
    # of 1 s samples, so the closing error is the sub-sample remainder times
    # the speed — a couple of metres, not a defect.
    north_m = (lat[-1] - lat[0]) * METERS_PER_DEGREE_LAT
    east_m = (lon[-1] - lon[0]) * METERS_PER_DEGREE_LAT * math.cos(math.radians(40.0))
    assert math.hypot(east_m, north_m) < 2.0 * speed


def test_circuit_stays_within_its_radius() -> None:
    radius_m = 800.0
    track = _track(pattern="circuit", radius_m=radius_m)
    lat, lon = build_track(track, _times(1000), np.random.default_rng(0))

    north_m = (lat - 40.0) * METERS_PER_DEGREE_LAT
    east_m = (lon + 111.0) * METERS_PER_DEGREE_LAT * math.cos(math.radians(40.0))
    # Distance from the circle centre, which sits one radius north of start.
    distance = np.hypot(east_m, north_m + radius_m)
    assert distance == pytest.approx(radius_m, rel=1e-6)


# ---------------------------------------------------------------------------
# Random walk
# ---------------------------------------------------------------------------


def test_random_walk_moves_at_the_configured_speed() -> None:
    """Speed is constant; only heading diffuses."""
    speed = 12.0
    track = _track(pattern="random_walk", speed_m_s=speed)
    lat, lon = build_track(track, _times(500), np.random.default_rng(0))

    north_m = (lat - 40.0) * METERS_PER_DEGREE_LAT
    east_m = (lon + 111.0) * METERS_PER_DEGREE_LAT * math.cos(math.radians(40.0))
    steps = np.hypot(np.diff(east_m), np.diff(north_m))
    assert steps == pytest.approx(np.full(steps.size, speed), rel=1e-6)


def test_zero_volatility_drives_in_a_straight_line() -> None:
    track = _track(pattern="random_walk", heading_volatility=0.0)
    lat, lon = build_track(track, _times(200), np.random.default_rng(0))

    north_m = (lat - 40.0) * METERS_PER_DEGREE_LAT
    east_m = (lon + 111.0) * METERS_PER_DEGREE_LAT * math.cos(math.radians(40.0))
    # A straight line has constant heading between consecutive steps.
    headings = np.arctan2(np.diff(east_m), np.diff(north_m))
    assert np.std(headings) == pytest.approx(0.0, abs=1e-9)


def test_higher_volatility_wanders_less_far() -> None:
    """A tightly wandering drive covers less ground than a straight one."""

    def displacement(volatility: float) -> float:
        track = _track(pattern="random_walk", heading_volatility=volatility)
        lat, lon = build_track(track, _times(600), np.random.default_rng(3))
        north_m = (lat[-1] - 40.0) * METERS_PER_DEGREE_LAT
        east_m = (lon[-1] + 111.0) * METERS_PER_DEGREE_LAT * math.cos(math.radians(40.0))
        return float(math.hypot(east_m, north_m))

    assert displacement(0.5) < displacement(0.0)


def test_random_walk_starts_at_the_configured_coordinates() -> None:
    """The first step has zero elapsed time, so no displacement yet."""
    lat, lon = build_track(_track(pattern="random_walk"), _times(10), np.random.default_rng(0))
    assert lat[0] == pytest.approx(40.0)
    assert lon[0] == pytest.approx(-111.0)


def test_track_near_the_pole_does_not_blow_up() -> None:
    """cos(latitude) collapses at the pole; the longitude scale is floored."""
    lat, lon = build_track(
        _track(start_latitude=89.9999, pattern="circuit", radius_m=100.0),
        _times(50),
        np.random.default_rng(0),
    )
    assert np.all(np.isfinite(lat))
    assert np.all(np.isfinite(lon))


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
