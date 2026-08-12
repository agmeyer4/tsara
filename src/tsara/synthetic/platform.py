"""Manufacturing synthetic mobile tracks.

This module *invents* a drive; :mod:`tsara.core.geodesy` holds the shared
primitives it is expressed in (the metre<->degree mapping, coordinate
bounding, and track interpolation), because those are needed just as much by
real ingested GPS as by a fabricated track.

Only the mobile case needs manufacturing at all. A stationary site is a
single configured lat/lon that the generator attaches to every stream as a
scalar coordinate — there is nothing to synthesize. Keeping GPS on a
*separate clock* from the gas analyzers is deliberate: it is the canonical
case for METHODS.md §1.2's asymmetric interpolation rule (position may be
interpolated onto gas timestamps; gases may not be interpolated onto
anything). A generator that put GPS on the gas clock would leave that rule
untestable.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.geodesy import (
    METERS_PER_DEGREE,
    clamp_latitude,
    meters_per_degree_longitude,
    wrap_longitude,
)
from tsara.core.timebase import epoch_s as _epoch_s
from tsara.synthetic.config import MobileTrack

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

logger = logging.getLogger(__name__)


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
        ``len(times)``, guaranteed to be valid coordinates.
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
    lat = platform.start_latitude + north_m / METERS_PER_DEGREE
    lon = platform.start_longitude + east_m / meters_per_degree_longitude(platform.start_latitude)

    # Offsets are integrated without bound, so a track starting near the
    # antimeridian can run past 180 degrees and a long northward drive past
    # 90. Both are non-coordinates, and they would propagate into the ground
    # truth (and from there into Phase 8 clustering) unnoticed.
    return clamp_latitude(lat), wrap_longitude(lon)


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
    r"""Return east/north metre offsets for a constant-speed heading random walk.

    Speed is held constant and only the *heading* diffuses, which is what a
    vehicle actually does — a position-space random walk would produce
    physically impossible instantaneous reversals and a track that never gets
    anywhere.

    Heading is a Wiener process: increments are drawn as
    :math:`\mathcal{N}(0, \sigma\sqrt{\Delta t})`, so the heading's spread
    after elapsed time :math:`T` grows as :math:`\sigma\sqrt{T}` — *not*
    :math:`\sigma T`. The ``heading_volatility`` parameter :math:`\sigma`
    therefore has units of rad·s^(-1/2), not radians per second. The
    :math:`\sqrt{\Delta t}` scaling is also what makes the wander independent
    of the GPS sampling rate: doubling the rate halves each increment's
    variance and doubles the number of increments, leaving the total
    unchanged.

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
