"""Tests for manufacturing synthetic mobile tracks.

The shared position primitives these are expressed in (metre<->degree
mapping, coordinate bounding, interpolation) are tested in
``tests/core/test_geodesy.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tsara.core.geodesy import METERS_PER_DEGREE
from tsara.synthetic.config import MobileTrack
from tsara.synthetic.platform import build_track


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


def _offsets_m(
    lat: np.ndarray, lon: np.ndarray, start_lat: float = 40.0, start_lon: float = -111.0
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a track back to east/north metre offsets from its start."""
    north_m = (lat - start_lat) * METERS_PER_DEGREE
    east_m = (lon - start_lon) * METERS_PER_DEGREE * math.cos(math.radians(start_lat))
    return east_m, north_m


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
    east_m, north_m = _offsets_m(lat, lon)
    assert math.hypot(east_m[-1] - east_m[0], north_m[-1] - north_m[0]) < 2.0 * speed


def test_circuit_stays_within_its_radius() -> None:
    radius_m = 800.0
    track = _track(pattern="circuit", radius_m=radius_m)
    lat, lon = build_track(track, _times(1000), np.random.default_rng(0))

    east_m, north_m = _offsets_m(lat, lon)
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

    east_m, north_m = _offsets_m(lat, lon)
    steps = np.hypot(np.diff(east_m), np.diff(north_m))
    assert steps == pytest.approx(np.full(steps.size, speed), rel=1e-6)


def test_zero_volatility_drives_in_a_straight_line() -> None:
    track = _track(pattern="random_walk", heading_volatility=0.0)
    lat, lon = build_track(track, _times(200), np.random.default_rng(0))

    east_m, north_m = _offsets_m(lat, lon)
    # A straight line has constant heading between consecutive steps.
    headings = np.arctan2(np.diff(east_m), np.diff(north_m))
    assert np.std(headings) == pytest.approx(0.0, abs=1e-9)


def test_higher_volatility_wanders_less_far() -> None:
    """A tightly wandering drive covers less ground than a straight one."""

    def displacement(volatility: float) -> float:
        track = _track(pattern="random_walk", heading_volatility=volatility)
        lat, lon = build_track(track, _times(600), np.random.default_rng(3))
        east_m, north_m = _offsets_m(lat, lon)
        return float(math.hypot(east_m[-1], north_m[-1]))

    assert displacement(0.5) < displacement(0.0)


def test_wander_is_independent_of_the_gps_sampling_rate() -> None:
    r"""The ``sqrt(dt)`` increment scaling is what makes this rate-independent.

    ``heading_volatility`` is a Wiener diffusion coefficient in rad/sqrt(s),
    not radians per second. The practical consequence — and the reason the
    scaling has to be :math:`\sqrt{\Delta t}` — is that sampling the *same*
    drive more often must not change the drive. Under a linear ``dt``
    scaling the accumulated heading variance would be proportional to the
    sampling interval, so a 4x faster GPS would produce a visibly different
    track from identical configuration.

    Measured as mean net displacement across an ensemble, which is a smooth
    scalar function of how far the heading wandered, and unlike a raw heading
    is not confounded by the uniformly random initial bearing.
    """

    def mean_displacement(freq: str, n_samples: int) -> float:
        distances = []
        for seed in range(200):
            track = _track(pattern="random_walk", heading_volatility=0.05, speed_m_s=10.0)
            lat, lon = build_track(track, _times(n_samples, freq), np.random.default_rng(seed))
            east_m, north_m = _offsets_m(lat, lon)
            distances.append(math.hypot(east_m[-1], north_m[-1]))
        return float(np.mean(distances))

    # The same 400 s span, sampled three ways.
    coarse = mean_displacement("1s", 401)
    fine = mean_displacement("500ms", 801)
    finest = mean_displacement("250ms", 1601)

    assert fine == pytest.approx(coarse, rel=0.05)
    assert finest == pytest.approx(coarse, rel=0.05)


def test_random_walk_starts_at_the_configured_coordinates() -> None:
    """The first step has zero elapsed time, so no displacement yet."""
    lat, lon = build_track(_track(pattern="random_walk"), _times(10), np.random.default_rng(0))
    assert lat[0] == pytest.approx(40.0)
    assert lon[0] == pytest.approx(-111.0)


# ---------------------------------------------------------------------------
# Coordinate validity
# ---------------------------------------------------------------------------


def test_track_near_the_pole_does_not_blow_up() -> None:
    """cos(latitude) collapses at the pole; the longitude scale is floored."""
    lat, lon = build_track(
        _track(start_latitude=89.9999, pattern="circuit", radius_m=100.0),
        _times(50),
        np.random.default_rng(0),
    )
    assert np.all(np.isfinite(lat))
    assert np.all(np.isfinite(lon))


def test_track_crossing_the_antimeridian_stays_a_valid_coordinate() -> None:
    """Unbounded metre offsets used to emit longitudes past 180 degrees."""
    lat, lon = build_track(
        _track(
            start_latitude=0.0,
            start_longitude=179.9,
            pattern="circuit",
            speed_m_s=50.0,
            radius_m=20_000.0,
        ),
        _times(3000),
        np.random.default_rng(0),
    )
    assert np.all(lon >= -180.0)
    assert np.all(lon < 180.0)
    assert np.all(np.abs(lat) <= 90.0)
    # The crossing really did happen: the track spans both sides of the seam.
    assert lon.max() > 179.0
    assert lon.min() < -179.0


def test_long_northward_drive_cannot_exceed_the_pole() -> None:
    lat, _ = build_track(
        _track(start_latitude=89.0, pattern="random_walk", heading_volatility=0.0, speed_m_s=90.0),
        _times(5000),
        np.random.default_rng(1),
    )
    assert np.all(np.abs(lat) <= 90.0)
