"""Turning what a file wrote into TSARA's time axis.

Every reader faces the same two problems, and they are the two easiest
places in ingestion to be subtly, silently wrong:

1. **Where is time, and how is it spelled?** One column of epoch seconds, or
   formatted text, or several columns that must be rejoined first.
2. **How does it reach UTC?** Exactly once, whether the file carried an
   explicit offset, declared a zone only in its documentation, or stored
   epoch seconds that were UTC all along.

Both live here rather than in any one reader, so there is one implementation
to get right instead of one per format. The failure they prevent is not a
crash: a record silently shifted by the UTC offset, or an index left at
microsecond resolution that changes dtype the first time it round-trips
through netCDF, both look completely healthy until something much later
compares two timestamps and disagrees.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from tsara.ingest.base import TIME_INDEX_NAME, TsaraIngestError

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from tsara.config.manifest import TimeParsing

logger = logging.getLogger(__name__)

__all__ = ["build_time_index", "to_utc_naive_ns"]

#: Sentinel accepted by ``TimeParsing.format`` meaning "epoch seconds".
_UNIX = "unix"

#: Sentinel accepted by ``TimeParsing.format`` meaning "ISO 8601". Mapped to
#: pandas' own ``'ISO8601'`` token, which parses the whole family of ISO
#: spellings (with or without ``T``, offset, or fractional seconds) rather
#: than locking to one of them.
_ISO = "iso8601"


def build_time_index(frame: pd.DataFrame, spec: TimeParsing, path: Path) -> pd.DatetimeIndex:
    """Construct the tz-naive UTC nanosecond time index for a parsed frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Parsed file contents.
    spec : TimeParsing
        Manifest description of where time lives and how it is written.
    path : pathlib.Path
        Source file, for error messages.

    Returns
    -------
    pandas.DatetimeIndex
        Timestamps aligned to ``frame``'s rows, possibly containing ``NaT``
        for rows that did not parse (the caller decides what to do with them).

    Raises
    ------
    TsaraIngestError
        If a declared time column is not present in the file.
    """
    missing = [name for name in spec.columns if name not in frame.columns]
    if missing:
        raise TsaraIngestError(
            f"'{path}' has no column(s) {missing} declared in loader.time. "
            f"Columns found: {list(frame.columns)}."
        )

    if spec.format == _UNIX:
        # Epoch seconds are UTC by definition, so `timezone` cannot apply.
        # Saying otherwise is a misunderstanding worth surfacing, but not
        # worth refusing a run over.
        if spec.timezone.upper() != "UTC":
            logger.warning(
                "loader.time.timezone=%r is ignored for format='unix': epoch "
                "seconds are UTC by definition (%s).",
                spec.timezone,
                path,
            )
        # errors='coerce' turns unparseable entries into NaT rather than
        # aborting the file; the caller reports and drops them.
        seconds = pd.to_numeric(frame[spec.columns[0]], errors="coerce")
        epoch = pd.DatetimeIndex(pd.to_datetime(seconds, unit="s", errors="coerce"))
        # Routed through the same normalizer as the text path, and not
        # returned directly: pandas infers the *unit* from `unit='s'` and
        # hands back a datetime64[s] index, which violates the RawTable
        # nanosecond contract. Passing 'UTC' makes the timezone step a no-op
        # (epoch seconds are already UTC) while still pinning resolution.
        return to_utc_naive_ns(epoch, "UTC", path)

    raw = _joined_time_strings(frame, spec)
    parsed = pd.to_datetime(raw, format=_pandas_format(spec.format), errors="coerce")
    return to_utc_naive_ns(pd.DatetimeIndex(parsed), spec.timezone, path)


def _joined_time_strings(frame: pd.DataFrame, spec: TimeParsing) -> pd.Series:
    """Return one string per row to be parsed as a timestamp.

    For the single-column case this is the column itself, converted to
    string. For several columns it is their values joined by ``spec.join`` —
    the ``DATE`` + ``TIME`` split that gas analyzers commonly write.

    Parameters
    ----------
    frame : pandas.DataFrame
        Parsed file contents.
    spec : TimeParsing
        Manifest time description.

    Returns
    -------
    pandas.Series
        String representation of each row's timestamp.
    """
    columns = [frame[name].astype("string") for name in spec.columns]
    joined = columns[0]
    for column in columns[1:]:
        joined = joined.str.cat(column, sep=spec.join)
    # `.str.cat` propagates NA, so a row missing either half becomes NA and
    # then NaT — exactly the intended treatment for an incomplete timestamp.
    return joined


def _pandas_format(declared: str | None) -> str | None:
    """Translate a manifest format string into what pandas expects.

    Parameters
    ----------
    declared : str or None
        The manifest's ``format``: a strftime pattern, the ``'iso8601'``
        sentinel, or ``None`` for inference.

    Returns
    -------
    str or None
        Format token to hand to :func:`pandas.to_datetime`.
    """
    if declared is None:
        return None
    if declared.lower() == _ISO:
        return "ISO8601"
    return declared


def to_utc_naive_ns(times: pd.DatetimeIndex, timezone: str, path: Path) -> pd.DatetimeIndex:
    """Normalize parsed timestamps to tz-naive UTC at nanosecond resolution.

    Shared by every reader, because "get to UTC exactly once, correctly" is
    the one piece of time handling that must not be reimplemented per format.

    Two distinct cases, which is why this cannot be a single pandas call:

    * The parsed values are **already tz-aware** — the file carried explicit
      offsets (common in ISO 8601). The declared ``timezone`` is then
      redundant and must not be applied a second time; the offsets win.
    * The parsed values are **naive** — the file wrote local wall-clock time
      and said so only in its documentation. The declared ``timezone`` is
      the missing information, so it is attached and converted.

    Resolution is pinned to nanoseconds at the end because netCDF stores ns:
    an index left at µs or s would change dtype across a save/load round trip
    and break later exact comparisons against event boundaries. pandas is a
    live source of such indexes — ``to_datetime(unit="s")`` returns
    ``datetime64[s]`` — so the pin is load-bearing, not defensive.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Parsed timestamps, aware or naive.
    timezone : str
        IANA zone to attach to naive timestamps.
    path : pathlib.Path
        Source file, for error messages.

    Returns
    -------
    pandas.DatetimeIndex
        Tz-naive UTC, ``datetime64[ns]``, named ``time``.

    Raises
    ------
    TsaraIngestError
        If localization fails — which for a real zone means the file contains
        a wall-clock time that is ambiguous or nonexistent under daylight
        saving. That is a genuine data problem, not something to paper over.
    """
    import pandas as pd

    if times.tz is not None:
        # The file won, which is correct -- an explicit offset is a statement
        # by the instrument, a manifest zone is a statement about it. But a
        # declared zone that had no effect should not pass in silence: the
        # author believed they were supplying the missing information, and if
        # their belief is wrong somewhere else in the manifest, this is the
        # cheapest place to notice. The `unix` branch above warns for the
        # same reason; leaving this one quiet made the two inconsistent.
        if timezone.upper() != "UTC":
            logger.warning(
                "loader.time.timezone=%r is ignored for '%s': the file's "
                "timestamps carry explicit UTC offsets, which take precedence.",
                timezone,
                path,
            )
        result = times.tz_convert("UTC").tz_localize(None)
    elif timezone.upper() == "UTC":
        result = times
    else:
        try:
            result = times.tz_localize(timezone).tz_convert("UTC").tz_localize(None)
        except Exception as exc:
            raise TsaraIngestError(
                f"Could not interpret timestamps in '{path}' as timezone "
                f"'{timezone}': {exc}. Daylight-saving transitions make some "
                "local wall-clock times ambiguous or nonexistent; recording "
                "in UTC avoids this entirely."
            ) from exc
    return pd.DatetimeIndex(result.astype("datetime64[ns]"), name=TIME_INDEX_NAME)
