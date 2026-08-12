"""How TSARA represents position: a local equirectangular approximation.

Position enters the package in two structurally different ways (CLAUDE.md
§1) — a stationary site has one static lat/lon, while a mobile platform
carries a GPS *timeseries* on its own clock — but both reduce to the same
two primitives, which is why they live here rather than in any one stage:

* converting between metre offsets and degrees, for anything that needs a
  distance (Phase 8 clusters plumes in a scaled space-time metric);
* interpolating a position track onto arbitrary times
  (:func:`positions_at`), which Phase 4 needs to place real GPS onto gas
  timestamps and which the synthetic generator needs to geolocate events.

The flat-Earth approximation
----------------------------
Coordinates are treated as a local equirectangular plane: a degree of
latitude is a constant number of metres, and a degree of longitude is that
scaled by ``cos(latitude)``. Over the tens-of-kilometres extent of a survey
drive the error is metres — far below GPS noise, and far below the scale at
which plume clustering operates — while a full geodesic treatment would add
a dependency and obscure the code. The approximation degrades toward the
poles, which is precisely where the alternative would matter; see
:func:`meters_per_degree_longitude` for the supported domain.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.timebase import epoch_s as _epoch_s

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

#: Metres per degree, used for both axes under the flat-Earth approximation.
#:
#: The value is the WGS-84 **equatorial degree of longitude**,
#: :math:`2\pi a / 360` with :math:`a = 6378137` m, which evaluates to
#: 111319.5 m. It is deliberately *also* used for latitude even though the
#: true mean meridional degree is 111132.9 m — a 0.17 % difference, two
#: orders of magnitude below GPS noise over a survey-scale track, and the
#: standard simplification for equirectangular work. Using one constant for
#: both axes keeps the metre<->degree mapping invertible and single-valued,
#: which matters more here than the third significant figure.
METERS_PER_DEGREE = 111_320.0

#: Backwards-compatible alias. The old name claimed to be a *meridional*
#: degree, which the value never was; prefer :data:`METERS_PER_DEGREE`.
METERS_PER_DEGREE_LAT = METERS_PER_DEGREE

#: Below this longitude scale (metres per degree) the equirectangular mapping
#: is treated as unsupported rather than silently rescaled — see
#: :func:`meters_per_degree_longitude`.
MIN_LONGITUDE_SCALE_M = 1.0


def meters_per_degree_longitude(latitude_deg: float) -> float:
    """Return metres per degree of longitude at a given latitude.

    Parameters
    ----------
    latitude_deg : float
        Latitude in decimal degrees.

    Returns
    -------
    float
        Metres per degree of longitude, floored at
        :data:`MIN_LONGITUDE_SCALE_M`.

    Notes
    -----
    The floor is a guard against division by a vanishing scale within about
    0.0005° of a pole. It does not make polar work *correct* — nothing about
    an equirectangular plane is correct there — it only keeps the arithmetic
    finite. Polar platforms are outside this module's supported domain.
    """
    scale = METERS_PER_DEGREE * math.cos(math.radians(latitude_deg))
    return max(scale, MIN_LONGITUDE_SCALE_M)


def wrap_longitude(longitude: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Wrap longitudes into ``[-180, 180)``.

    A track integrated as an unbounded metre offset will happily run past the
    antimeridian and report 180.08°, which is not a coordinate. Wrapping
    restores a valid value.

    Note what wrapping does *not* fix: a track spanning the antimeridian is
    still discontinuous in longitude, so any downstream stage clustering on
    raw lat/lon (Phase 8) will see a seam there. That is a property of the
    coordinate system, not of this function, and it affects real data
    identically — which is the argument for wrapping rather than leaving
    out-of-range values that merely hide the seam behind an invalid number.

    Parameters
    ----------
    longitude : numpy.ndarray
        Longitudes in decimal degrees, possibly out of range.

    Returns
    -------
    numpy.ndarray
        Longitudes in ``[-180, 180)``.
    """
    return np.asarray((longitude + 180.0) % 360.0 - 180.0, dtype=np.float64)


def clamp_latitude(latitude: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Clamp latitudes into ``[-90, 90]``.

    Unlike longitude, latitude does not wrap: passing over a pole reverses
    the direction of travel and shifts longitude by 180°, which the
    equirectangular model cannot represent at all. Clamping keeps the value a
    valid coordinate and confines the error to the polar region that this
    module already declares unsupported.

    Parameters
    ----------
    latitude : numpy.ndarray
        Latitudes in decimal degrees, possibly out of range.

    Returns
    -------
    numpy.ndarray
        Latitudes in ``[-90, 90]``.
    """
    return np.asarray(np.clip(latitude, -90.0, 90.0), dtype=np.float64)


def positions_at(
    times: pd.DatetimeIndex,
    track_times: pd.DatetimeIndex,
    latitude: npt.NDArray[np.float64],
    longitude: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interpolate track coordinates onto arbitrary times.

    Linear interpolation of *position* is legitimate for the reason
    METHODS.md §1.2 permits it for auxiliary fields generally: position is a
    smooth, continuously-varying quantity, unlike a concentration inside a
    plume. This is the asymmetry at the heart of the synchronize-late design
    — GPS may be moved onto a gas clock; a gas may never be moved onto
    anything.

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

    Notes
    -----
    No maximum-gap guard is applied here: this returns a value for every
    target time, however far it sits from a real fix. Phase 4 adds the
    ``AlignmentConfig.max_interp_gap`` guard on top for real data, where
    interpolating across a long GPS outage would fabricate a position. The
    synthetic generator does not need it, since it controls the track.
    """
    target = _epoch_s(times)
    source = _epoch_s(track_times)
    return (
        np.interp(target, source, latitude),
        np.interp(target, source, longitude),
    )
