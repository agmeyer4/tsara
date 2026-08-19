"""Reader for delimited text files (CSV/TSV/whitespace-padded logger output).

"CSV" is a loose label here, and deliberately so. The files this reader
actually has to handle in a field campaign arrive as ``.csv``, ``.dat`` and
``.txt`` in roughly equal measure, separated by commas, tabs, or runs of
spaces, and disagree about nearly everything: whether there is a header at
all, whether time is one column or two, whether missing data is ``NA``,
``-9999``, or an empty field. All of that variation is *data* — it belongs in
the manifest — so this module's job is to implement the manifest faithfully
and to fail loudly and specifically when a file does not match what the
manifest claims.

The two hard parts are both about time
---------------------------------------
1. **Building the timestamp** from one column of epoch seconds, or one
   column of formatted text, or several columns that must be rejoined first
   (see :class:`~tsara.config.manifest.TimeParsing`).
2. **Getting it to UTC exactly once.** Timestamps enter TSARA here, and
   ``docs/METHODS.md`` requires everything downstream to be tz-naive UTC at
   nanosecond resolution. This module is the only place in ingestion that
   performs that conversion, so there is one implementation to get right
   rather than one per format.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from tsara.config.manifest import CSVLoader, TimeParsing
from tsara.ingest.base import TIME_INDEX_NAME, RawTable, TsaraIngestError, to_utc_naive_ns
from tsara.ingest.registry import register_reader

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from tsara.config.manifest import LoaderConfig

logger = logging.getLogger(__name__)

__all__ = ["read_csv"]

#: Sentinel accepted by ``TimeParsing.format`` meaning "epoch seconds".
_UNIX = "unix"

#: Sentinel accepted by ``TimeParsing.format`` meaning "ISO 8601". Mapped to
#: pandas' own ``'ISO8601'`` token, which parses the whole family of ISO
#: spellings (with or without ``T``, offset, or fractional seconds) rather
#: than locking to one of them.
_ISO = "iso8601"

#: Separator that pandas' fast C parser understands natively. Any *other*
#: multi-character separator is a general regex and needs the Python engine.
_WHITESPACE_RUN = r"\s+"


@register_reader("csv")
def read_csv(path: Path, loader: LoaderConfig, /) -> RawTable:
    """Read one delimited-text file into a :class:`RawTable`.

    Parameters
    ----------
    path : pathlib.Path
        File to read.
    loader : LoaderConfig
        Must be a :class:`~tsara.config.manifest.CSVLoader`; the union type
        is required by the :class:`~tsara.ingest.base.Reader` protocol and is
        narrowed immediately.

    Returns
    -------
    RawTable
        Raw columns under their file names, indexed by tz-naive UTC time.

    Raises
    ------
    TsaraIngestError
        If the loader is the wrong type, the file cannot be parsed, a
        declared time column is absent, or no row yields a valid timestamp.
    """
    if not isinstance(loader, CSVLoader):
        raise TsaraIngestError(
            f"The 'csv' reader received a {type(loader).__name__}. This means a "
            "reader was registered under the wrong format name."
        )

    frame = _read_frame(path, loader)
    times = _build_time_index(frame, loader.time, path)

    # Rows whose timestamp did not parse cannot be placed on a time axis at
    # all. Dropping them is the only coherent option, but doing it silently
    # would hide a systematically wrong `format:` (which shows up as *most*
    # rows failing), so the count is logged and a total failure is an error.
    # Materialized as a NumPy array rather than left as an Index: it is used
    # three ways below (negate, reduce, mask) and only the array supports all.
    valid = np.asarray(times.notna())
    n_bad = int((~valid).sum())
    if n_bad:
        if not bool(valid.any()):
            raise TsaraIngestError(
                f"No row in '{path}' produced a valid timestamp from columns "
                f"{list(loader.time.columns)} with format={loader.time.format!r}. "
                "The format or the column names do not match this file."
            )
        logger.warning(
            "Dropped %d of %d rows from %s: timestamp did not parse.",
            n_bad,
            len(times),
            path,
        )
        frame = frame.loc[valid]
        times = times[valid]

    frame = frame.set_axis(pd.DatetimeIndex(times, name=TIME_INDEX_NAME), axis=0)
    return RawTable(frame=frame, path=path, attrs={})


def _read_frame(path: Path, loader: CSVLoader) -> pd.DataFrame:
    """Parse the file into a DataFrame with a default integer index.

    Separated from time handling so that each can be tested — and fail —
    independently: "the file did not parse" and "the file parsed but its
    timestamps did not" are different problems with different fixes.

    Parameters
    ----------
    path : pathlib.Path
        File to read.
    loader : CSVLoader
        Parsing configuration.

    Returns
    -------
    pandas.DataFrame
        Raw table, columns named as the file (or the manifest) names them.
    """
    kwargs: dict[str, Any] = {
        "sep": loader.delimiter,
        "header": loader.header_row,
        "comment": loader.comment,
        "skip_blank_lines": True,
    }
    # `na_values` extends pandas' default set rather than replacing it, so a
    # manifest declaring '-9999' does not stop '' and 'NaN' being missing.
    if loader.na_values:
        kwargs["na_values"] = list(loader.na_values)

    # The C parser handles ',' , '\t' and the '\s+' whitespace-run idiom; any
    # other multi-character separator is a regex only the Python engine
    # implements. Choosing explicitly avoids pandas' silent engine fallback,
    # which emits a warning and makes behaviour depend on the pandas version.
    if len(loader.delimiter) > 1 and loader.delimiter != _WHITESPACE_RUN:
        kwargs["engine"] = "python"

    if loader.column_names is not None:
        kwargs["header"] = None
        kwargs["names"] = _positional_names(path, loader)

    # `**kwargs` erases the return type, so restore it rather than letting
    # Any leak into every caller.
    frame = cast("pd.DataFrame", pd.read_csv(path, **kwargs))
    if frame.empty:
        raise TsaraIngestError(f"'{path}' contains no data rows.")
    return frame


def _positional_names(path: Path, loader: CSVLoader) -> list[str]:
    """Build the full positional name list for a headerless file.

    Why this is not simply ``list(loader.column_names)``: a headerless file
    may be far wider than the part anyone wants. An Aeris Spectralite log is
    522 columns, nearly all spectral bins; requiring a manifest to name all
    522 in order to read the three that matter would be unusable, and giving
    pandas a short ``names`` list is worse than unusable — it silently
    promotes the surplus leading columns into a MultiIndex instead of
    failing.

    So ``column_names`` names a *prefix*, and any remaining columns get
    generated positional names. They are still addressable (a manifest can
    reference ``column_11``) but nobody has to enumerate them.

    Parameters
    ----------
    path : pathlib.Path
        File to inspect.
    loader : CSVLoader
        Loader whose ``column_names`` supplies the prefix. The schema
        guarantees it is not ``None`` when this is called.

    Returns
    -------
    list of str
        Names for every column present in the file.
    """
    declared = list(loader.column_names or ())

    probe_kwargs: dict[str, Any] = {
        "sep": loader.delimiter,
        "header": None,
        "comment": loader.comment,
        "skip_blank_lines": True,
        "nrows": 1,
    }
    if len(loader.delimiter) > 1 and loader.delimiter != _WHITESPACE_RUN:
        probe_kwargs["engine"] = "python"
    # Reuse pandas for the width probe rather than splitting a line by hand:
    # it already knows about quoting, embedded separators and ragged endings.
    width = int(pd.read_csv(path, **probe_kwargs).shape[1])

    if width < len(declared):
        raise TsaraIngestError(
            f"'{path}' has {width} columns but column_names declares "
            f"{len(declared)}. The manifest describes a wider file than this one."
        )
    # Generated names use the 'column_' prefix (not 'col_') to stay clearly
    # distinct from anything an instrument would plausibly emit itself.
    return declared + [f"column_{i}" for i in range(len(declared), width)]


def _build_time_index(frame: pd.DataFrame, spec: TimeParsing, path: Path) -> pd.DatetimeIndex:
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
