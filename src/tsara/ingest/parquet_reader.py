"""Reader for Apache Parquet files.

Why parquet belongs in Phase 3 rather than "later"
--------------------------------------------------
Field campaigns rarely analyze the bytes their instruments wrote. They
analyze a *processed* stage — time-corrected, standardized, instrument-
aligned — and that stage is commonly stored as parquet even when every
instrument logged plain text. The archive this package targets is exactly
that shape: its entire instrument-aligned stage is parquet and contains no
text files at all, so without this reader that campaign is unreadable.

What makes parquet different from the text formats
--------------------------------------------------
**It stores the index.** A parquet written by pandas round-trips the
``DatetimeIndex`` as part of the file, so in the normal case there is no
timestamp to parse, no format string to get wrong, and no timezone to infer
— the answer is already there. That is why
:class:`~tsara.config.manifest.ParquetLoader` makes ``time`` optional where
:class:`~tsara.config.manifest.CSVLoader` requires it.

Two things still need doing, and both are the reason this is a real reader
rather than a one-line call:

* **Resolution is not guaranteed.** Files in the target archive carry
  ``datetime64[us, UTC]`` and ``datetime64[ns, UTC]`` indexes side by side,
  sometimes for the same instrument. Both are perfectly valid parquet and
  neither satisfies the ``RawTable`` nanosecond contract on its own.
* **Timezone is not guaranteed either.** These indexes are tz-*aware*,
  unlike anything the text readers produce, so they must be converted and
  flattened rather than passed through.

Both are handled by the same :mod:`tsara.ingest.timeparse` normalizer every
other reader uses, which is the point of having one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.config.manifest import ParquetLoader
from tsara.ingest.base import TIME_INDEX_NAME, RawTable, TsaraIngestError
from tsara.ingest.registry import register_reader
from tsara.ingest.timeparse import build_time_index, to_utc_naive_ns

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from tsara.config.manifest import LoaderConfig

logger = logging.getLogger(__name__)

__all__ = ["read_parquet"]


@register_reader("parquet")
def read_parquet(path: Path, loader: LoaderConfig, /) -> RawTable:
    """Read one parquet file into a :class:`RawTable`.

    Parameters
    ----------
    path : pathlib.Path
        File to read.
    loader : LoaderConfig
        Must be a :class:`~tsara.config.manifest.ParquetLoader`.

    Returns
    -------
    RawTable
        Columns under their stored names, indexed by tz-naive UTC time.

    Raises
    ------
    TsaraIngestError
        If the loader is the wrong type, the file cannot be read, it holds no
        rows, or no usable time axis can be found.
    """
    if not isinstance(loader, ParquetLoader):
        raise TsaraIngestError(
            f"The 'parquet' reader received a {type(loader).__name__}. This means "
            "a reader was registered under the wrong format name."
        )

    frame = pd.read_parquet(path)
    if frame.empty:
        raise TsaraIngestError(f"'{path}' contains no data rows.")

    if loader.time is None:
        times = _index_from_file(frame, path)
    else:
        # A declared `time:` block means the time axis lives in a column, so
        # whatever index the file stored is not it. Promote that index to a
        # column first — it may well be the very column being named, and
        # discarding it would make the manifest unable to reach it. A default
        # RangeIndex carries no information and is left alone, since
        # promoting it would only add a junk 'index' column.
        if not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index(drop=False)
        times = build_time_index(frame, loader.time, path)

    # Same policy as every other reader: a row without a timestamp cannot be
    # placed on a time axis, so it is dropped — but counted, never silently.
    valid = np.asarray(times.notna())
    n_bad = int((~valid).sum())
    if n_bad:
        if not bool(valid.any()):
            raise TsaraIngestError(
                f"No row in '{path}' produced a valid timestamp. The declared "
                "time columns or format do not match this file."
            )
        logger.warning(
            "Dropped %d of %d rows from %s: timestamp did not parse.", n_bad, len(times), path
        )
        frame = frame.loc[valid]
        times = times[valid]

    frame = frame.set_axis(pd.DatetimeIndex(times, name=TIME_INDEX_NAME), axis=0)
    return RawTable(frame=frame, path=path, attrs={})


def _index_from_file(frame: pd.DataFrame, path: Path) -> pd.DatetimeIndex:
    """Use the ``DatetimeIndex`` the parquet file already stores.

    The default path, and the one that makes parquet pleasant: the file was
    written from a dataframe that already had its time axis, so nothing needs
    parsing. It still needs *normalizing* — the stored index is typically
    tz-aware and may be at microsecond resolution — which is why this is not
    simply ``frame.index``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Contents of the parquet file.
    path : pathlib.Path
        Source file, for error messages.

    Returns
    -------
    pandas.DatetimeIndex
        Tz-naive UTC nanosecond timestamps.

    Raises
    ------
    TsaraIngestError
        If the file's index is not a ``DatetimeIndex``. The message points at
        the fix, since the manifest can supply a ``time:`` block naming a
        column instead.
    """
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TsaraIngestError(
            f"'{path}' stores a {type(index).__name__} index, not a DatetimeIndex, "
            "and the manifest declares no 'time:' block for this loader. Either "
            "add one naming the column that holds time, or re-export the file "
            "with its time axis as the index. "
            f"Columns present: {list(frame.columns)[:10]}."
        )
    return to_utc_naive_ns(index, "UTC", path)
