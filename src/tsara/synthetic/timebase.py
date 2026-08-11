"""Converting timestamps to numeric epochs, once and correctly.

Every module here needs timestamps as numbers (to evaluate a shape, integrate
a track, or correlate noise), and the naive spelling
``index.to_numpy().astype("datetime64[ns]")`` silently mishandles timezone-
aware input — NumPy has no timezone concept, so it warns and discards the
offset, which would shift a UTC+2 record by two hours without failing.

TSARA is UTC internally (see ``TimeParsing.timezone`` in the manifest schema),
so the correct normalization is: convert aware timestamps to UTC, then drop
the tzinfo. Doing it in one place means no caller can get it wrong, and
tz-aware and tz-naive configs produce identical numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

#: Nanoseconds per second, the scale between the two helpers below.
NS_PER_S = 1e9


def to_utc_naive(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return ``times`` as tz-naive UTC.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Timestamps, timezone-aware or naive. Naive input is assumed to
        already be UTC and is returned unchanged.

    Returns
    -------
    pandas.DatetimeIndex
        Timezone-naive index on the UTC timeline.
    """
    if times.tz is not None:
        return times.tz_convert("UTC").tz_localize(None)
    return times


def epoch_ns(times: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
    """Return integer nanoseconds since the Unix epoch.

    Integer nanoseconds are used (rather than float seconds) wherever exact
    comparison matters — notably ``searchsorted`` against event window
    bounds, where float rounding could drop a boundary sample.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Timestamps, timezone-aware or naive.

    Returns
    -------
    numpy.ndarray
        int64 nanoseconds since 1970-01-01T00:00:00Z.
    """
    return np.asarray(
        to_utc_naive(times).to_numpy().astype("datetime64[ns]").astype(np.int64),
        dtype=np.int64,
    )


def epoch_s(times: pd.DatetimeIndex) -> npt.NDArray[np.float64]:
    """Return float seconds since the Unix epoch.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Timestamps, timezone-aware or naive.

    Returns
    -------
    numpy.ndarray
        float64 seconds since 1970-01-01T00:00:00Z.
    """
    return np.asarray(epoch_ns(times) / NS_PER_S, dtype=np.float64)


def timestamp_epoch_s(stamp: pd.Timestamp) -> float:
    """Return one timestamp as float seconds since the Unix epoch.

    Parameters
    ----------
    stamp : pandas.Timestamp
        Timestamp, timezone-aware or naive.

    Returns
    -------
    float
        Seconds since 1970-01-01T00:00:00Z.
    """
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return float(stamp.value) / NS_PER_S


def timestamp_epoch_ns(stamp: pd.Timestamp) -> int:
    """Return one timestamp as integer nanoseconds since the Unix epoch.

    Parameters
    ----------
    stamp : pandas.Timestamp
        Timestamp, timezone-aware or naive.

    Returns
    -------
    int
        Nanoseconds since 1970-01-01T00:00:00Z.
    """
    if stamp.tz is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return int(stamp.value)
