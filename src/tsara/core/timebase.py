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

#: One second, in the integer nanoseconds TSARA does interval arithmetic in.
#:
#: The same fact as :data:`NS_PER_S` in the other of this module's two units,
#: and a separate name because the two do different jobs: ``NS_PER_S`` is a
#: scale that a float division applies, while this is a *duration* that cell
#: boundaries are built from and compared against. Integer nanoseconds are what
#: :mod:`tsara.core.support` works in throughout, because float seconds lose
#: nanosecond resolution above about 10^8 s of epoch.
SECOND_NS = 1_000_000_000

#: Seconds in an hour, and in a day.
#:
#: Here rather than beside the code that uses them because the same two numbers
#: had been written out in four places across two subpackages, and a unit
#: conversion is exactly the kind of fact that should have one home — this
#: module is where TSARA converts between time units. Both appear wherever a
#: configured *rate* meets a real time span: the synthetic generator quotes
#: drift and random-walk amplitudes per day, and plume rates per hour.
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86_400.0


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


def to_utc_naive_stamp(stamp: pd.Timestamp) -> pd.Timestamp:
    """Return one timestamp as tz-naive UTC.

    The scalar counterpart of :func:`to_utc_naive`. It exists because
    timestamps enter TSARA from two directions — as clocks (indexes) and as
    event boundaries (scalars) — and the two must land on the same
    representation or they cannot be compared at all: pandas raises
    ``TypeError`` on any comparison between an aware and a naive timestamp.
    Normalizing both through this module is what lets a ground-truth event
    window be used to slice a stream.

    Parameters
    ----------
    stamp : pandas.Timestamp
        Timestamp, timezone-aware or naive. Naive input is assumed to already
        be UTC and is returned unchanged.

    Returns
    -------
    pandas.Timestamp
        Timezone-naive timestamp on the UTC timeline.
    """
    if stamp.tz is not None:
        return stamp.tz_convert("UTC").tz_localize(None)
    return stamp


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


def timestamp_epoch_ns(stamp: pd.Timestamp) -> int:
    """Return one timestamp as integer nanoseconds since the Unix epoch.

    This is the single scalar conversion; :func:`timestamp_epoch_s` is
    derived from it, so the timezone rule this module exists to centralize
    is written down exactly once.

    Parameters
    ----------
    stamp : pandas.Timestamp
        Timestamp, timezone-aware or naive.

    Returns
    -------
    int
        Nanoseconds since 1970-01-01T00:00:00Z.
    """
    return int(to_utc_naive_stamp(stamp).value)


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
    return timestamp_epoch_ns(stamp) / NS_PER_S
