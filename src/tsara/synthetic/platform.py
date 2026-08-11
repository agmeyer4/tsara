"""Platform coordinates: fixed sites and synthetic mobile tracks.

Position enters TSARA in two structurally different ways (CLAUDE.md §1), and
this module manufactures both so the distinction is exercised rather than
assumed:

* **Stationary** — a single lat/lon applied globally, attached to every
  stream as a scalar coordinate.
* **Mobile** — a GPS *timeseries*, emitted as its own stream at its own
  rate. Keeping GPS on a separate clock from the gas analyzers is deliberate:
  it is the canonical case for METHODS.md §1.2's asymmetric interpolation
  rule (position is a smooth auxiliary field and may be interpolated onto gas
  timestamps; the gases themselves may not be). A generator that put GPS on
  the gas clock would leave that rule untestable.

Geodesy
-------
Tracks are integrated in a local flat-Earth (equirectangular) approximation:
one degree of latitude is treated as a constant 111,320 m and one degree of
longitude as that scaled by ``cos(latitude)``. Over the tens-of-kilometres
extent of a survey drive the error is metres — far below GPS noise and utterly
below the scale at which plume clustering operates — while a full geodesic
integration would add a dependency and obscure the code. The approximation
degrades near the poles, which is where the alternative would matter.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from tsara.synthetic.config import MobileTrack
from tsara.synthetic.timebase import epoch_s as _epoch_s

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

logger = logging.getLogger(__name__)

#: Metres per degree of latitude (WGS-84 mean meridional degree).
METERS_PER_DEGREE_LAT = 111_320.0


def build_track(
    platform: MobileTrack,
    times: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate latitude/longitude arrays for a mobile platform.

    Two patterns, each chosen for what it makes testable:

    * ``random_walk`` — heading diffuses as a Wiener process while speed
      stays constant, giving an unstructured survey drive that rarely
      revisits a location. The realistic default.
    * ``circuit`` — a closed circle traversed at constant speed, which
      *does* return to the same coordinates repeatedly. That recurrence is
      what makes spatially clusterable data: encountering one source on
      successive laps is exactly the situation Phase 8's DBSCAN source
      complexes must merge into a single source.

    Parameters
    ----------
    platform : MobileTrack
        Track configuration.
    times : pandas.DatetimeIndex
        GPS sample times.
    rng : numpy.random.Generator
        Source of randomness (used by ``random_walk`` only).

    Returns
    -------
    tuple of numpy.ndarray
        ``(latitude, longitude)`` in decimal degrees, each of length
        ``len(times)``.
    """
    n = len(times)
    if n == 0:  # pragma: no cover - guarded by the caller's span validation
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    epoch_s = _epoch_s(times)
    elapsed_s = epoch_s - epoch_s[0]

    if platform.pattern == "circuit":
        east_m, north_m = _circuit_offsets(platform, elapsed_s)
    else:
        east_m, north_m = _random_walk_offsets(platform, elapsed_s, rng)

    # Convert metre offsets to degrees about the start point. The longitude
    # scale uses the *start* latitude throughout rather than each point's own
    # latitude: it keeps the mapping invertible and single-valued, and over a
    # survey-scale track the cos(lat) change is negligible.
    lat = platform.start_latitude + north_m / METERS_PER_DEGREE_LAT
    lon_scale = METERS_PER_DEGREE_LAT * math.cos(math.radians(platform.start_latitude))
    # Guard the degenerate polar case where the longitude scale collapses.
    lon_scale = max(lon_scale, 1.0)
    lon = platform.start_longitude + east_m / lon_scale

    return (
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
    )


def _circuit_offsets(
    platform: MobileTrack, elapsed_s: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return east/north metre offsets for a constant-speed circular circuit.

    The circle is positioned so the platform starts exactly at the configured
    start coordinates (angle 0), which keeps the configuration's meaning
    obvious: ``start_latitude``/``start_longitude`` is where the drive begins,
    not an abstract circle centre.

    Parameters
    ----------
    platform : MobileTrack
        Track configuration (uses ``speed_m_s`` and ``radius_m``).
    elapsed_s : numpy.ndarray
        Seconds since the track start.

    Returns
    -------
    tuple of numpy.ndarray
        ``(east_m, north_m)`` offsets from the start point.
    """
    angular_speed = platform.speed_m_s / platform.radius_m  # rad/s
    theta = angular_speed * elapsed_s
    east_m = platform.radius_m * np.sin(theta)
    north_m = platform.radius_m * (np.cos(theta) - 1.0)
    return np.asarray(east_m, dtype=np.float64), np.asarray(north_m, dtype=np.float64)


def _random_walk_offsets(
    platform: MobileTrack,
    elapsed_s: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return east/north metre offsets for a constant-speed heading random walk.

    Speed is held constant and only the *heading* diffuses, which is what a
    vehicle actually does — a position-space random walk would produce
    physically impossible instantaneous reversals and a track that never gets
    anywhere. Heading increments scale as ``sqrt(dt)`` so the wander is
    independent of the GPS sampling rate.

    Parameters
    ----------
    platform : MobileTrack
        Track configuration.
    elapsed_s : numpy.ndarray
        Seconds since the track start.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    tuple of numpy.ndarray
        ``(east_m, north_m)`` offsets from the start point.
    """
    dt_s = np.diff(elapsed_s, prepend=elapsed_s[0])
    dt_s = np.clip(dt_s, 0.0, None)

    heading = float(rng.uniform(0.0, 2.0 * math.pi)) + np.cumsum(
        rng.normal(0.0, platform.heading_volatility * np.sqrt(dt_s))
    )

    step_m = platform.speed_m_s * dt_s
    east_m = np.cumsum(step_m * np.sin(heading))
    north_m = np.cumsum(step_m * np.cos(heading))
    return np.asarray(east_m, dtype=np.float64), np.asarray(north_m, dtype=np.float64)


def positions_at(
    times: pd.DatetimeIndex,
    track_times: pd.DatetimeIndex,
    latitude: npt.NDArray[np.float64],
    longitude: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interpolate track coordinates onto arbitrary times.

    Used to stamp each ground-truth event with the platform position at its
    peak. Linear interpolation of position is legitimate here for the same
    reason METHODS.md §1.2 permits it for auxiliary fields generally —
    position is a smooth, continuously-varying quantity, unlike a
    concentration inside a plume.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Target times.
    track_times : pandas.DatetimeIndex
        Times at which the track is defined.
    latitude, longitude : numpy.ndarray
        Track coordinates.

    Returns
    -------
    tuple of numpy.ndarray
        Interpolated ``(latitude, longitude)`` at ``times``. Targets outside
        the track's span clamp to its endpoints.
    """
    target = _epoch_s(times)
    source = _epoch_s(track_times)
    return (
        np.interp(target, source, latitude),
        np.interp(target, source, longitude),
    )
