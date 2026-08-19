"""The contract every file reader honours: :class:`RawTable`.

Where the seam is, and why it is here
--------------------------------------
Ingestion has two halves with very different shapes. Reading a file is
*format-specific* and irreducibly fiddly — delimiters, header quirks, epoch
conventions, ICARTT's scale factors. Everything after that — masking, unit
conversion, resolving the uncertainty budget, assembling an
:class:`xarray.Dataset` — is *format-independent*: it depends only on the
manifest, never on whether the numbers arrived as CSV or ICARTT.

TSARA puts the seam between those halves at exactly one place, and this
module defines it. A reader's entire job is::

    (path, loader config)  ->  RawTable

after which no downstream code knows or cares what format the data was in.
That is what makes "add a parquet reader" a one-file change in Phase 3+
rather than a change rippling through QA/QC and stream assembly.

The deliberate choice: readers return **pandas**, not xarray
-------------------------------------------------------------
A :class:`pandas.DataFrame` is the right shape for a *file*: a table of
columns named the way the instrument named them, indexed by time. An
:class:`xarray.Dataset` is the right shape for a *stream*: canonical
variable names, units, uncertainty components, platform coordinates, and
provenance attrs. Those are different objects, and the conversion between
them is the same work for every format — so it happens once, in
:mod:`tsara.ingest.streams`, not once per reader.

The contract, stated precisely
------------------------------
A reader returns a :class:`RawTable` whose ``frame``:

1. is indexed by a **tz-naive UTC** :class:`pandas.DatetimeIndex` at
   nanosecond resolution, named ``time``;
2. keeps every column under the **name the raw file uses**. Canonical
   renaming (``CH4_dry`` → ``ch4``) is a manifest concern handled downstream;
   a reader that renamed columns would have to be taught the manifest, which
   is precisely the coupling this seam exists to prevent;
3. has **not** been masked, converted, sorted, or de-duplicated. Those are
   cross-file, campaign-level operations, and doing them per file would give
   a rolling QA/QC window a different answer depending on how the archive
   happened to be split into files.

Points 1 and 2 are enforced at runtime by :func:`check_raw_table`, which the
registry applies to every reader's output — including readers registered by
code outside TSARA.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from tsara.core.exceptions import TsaraError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from tsara.config.manifest import LoaderConfig

__all__ = [
    "TIME_INDEX_NAME",
    "RawTable",
    "Reader",
    "TsaraIngestError",
    "check_raw_table",
]

#: Name TSARA gives the time index everywhere, from raw table to final stream.
#: Centralized so a reader and the stream assembler cannot disagree about it.
TIME_INDEX_NAME = "time"


class TsaraIngestError(TsaraError):
    """Raised when raw data cannot be found, read, or parsed.

    Distinct from :class:`~tsara.core.exceptions.TsaraConfigError`: the
    configuration may be perfectly valid while the *archive* is missing,
    truncated, malformed, or simply not shaped the way the manifest claims.
    Keeping the two apart matters operationally — a config error means "fix
    your YAML", an ingest error means "look at your data".
    """


@dataclass(frozen=True)
class RawTable:
    """One raw file, parsed into a time-indexed table of its own columns.

    Frozen because a raw table is a *fact about a file*: every downstream
    stage derives new objects from it rather than editing it in place, which
    keeps provenance traceable when several stages have run.

    Attributes
    ----------
    frame : pandas.DataFrame
        The file's data, columns named as the file names them, indexed by a
        tz-naive UTC nanosecond :class:`pandas.DatetimeIndex` called ``time``.
        Unmasked, unconverted, and in file order.
    path : pathlib.Path
        The file this came from. Carried rather than discarded so that any
        later error — a bad value, an impossible timestamp — can name the
        file that produced it, which is the first thing anyone asks when
        ingesting a few hundred files.
    attrs : Mapping[str, object]
        Provenance the *file itself* declared, for formats that carry it
        (ICARTT headers name the PI, the instrument, the mission; a CSV
        declares nothing and yields an empty mapping). Kept separate from
        the path-template metadata harvested by the crawler, because these
        two answer different questions: "what did the data say about
        itself?" versus "what did its location in the archive say about it?".
        When they disagree, that disagreement is worth seeing.
    """

    frame: pd.DataFrame
    path: Path
    attrs: Mapping[str, object] = field(default_factory=dict)


class Reader(Protocol):
    """Callable signature every registered file reader must satisfy.

    Declared as a :class:`~typing.Protocol` rather than a base class so that
    a reader is a plain function: there is no state to carry, and requiring
    a subclass would be ceremony without benefit.

    Note the parameter type is the whole ``LoaderConfig`` union, not one
    member of it. Python's typing rules make a function accepting only
    ``CSVLoader`` an *invalid* implementation of a protocol that promises to
    accept any loader, so each reader takes the union and narrows it with an
    explicit ``isinstance`` check. That check is not busywork: it is what
    turns "this reader was registered under the wrong format name" into a
    clear error instead of an ``AttributeError`` on a missing field.
    """

    def __call__(self, path: Path, loader: LoaderConfig, /) -> RawTable:
        """Read one file into a :class:`RawTable`."""
        ...  # pragma: no cover - a Protocol body is never executed


def check_raw_table(table: RawTable, *, reader_name: str) -> RawTable:
    """Verify a reader honoured the :class:`RawTable` contract.

    Applied by the registry to every reader's output, so that a
    contract violation is reported at its source — naming the reader and the
    file — rather than surfacing hundreds of lines later as an unrelated
    failure in stream assembly. The failure mode this actually prevents is
    the tz-aware index: it compares fine, sorts fine, and survives every
    intermediate stage, then raises deep inside netCDF encoding or (worse)
    silently mismatches a tz-naive event boundary. Phase 2 hit that exact
    class of bug twice, which is why the check is mandatory rather than
    advisory.

    Parameters
    ----------
    table : RawTable
        Candidate returned by a reader.
    reader_name : str
        Registered name of the reader, used in error messages.

    Returns
    -------
    RawTable
        The same object, unchanged, so this can wrap a call site directly.

    Raises
    ------
    TsaraIngestError
        If the index is not a tz-naive nanosecond ``DatetimeIndex`` named
        ``time``, or the frame carries duplicate column names.
    """
    import pandas as pd

    frame = table.frame
    index = frame.index

    if not isinstance(index, pd.DatetimeIndex):
        raise TsaraIngestError(
            f"Reader '{reader_name}' returned {table.path} with a "
            f"{type(index).__name__} index; RawTable requires a DatetimeIndex."
        )
    if index.tz is not None:
        raise TsaraIngestError(
            f"Reader '{reader_name}' returned {table.path} with a timezone-aware "
            f"index ({index.tz}). TSARA is tz-naive UTC internally; convert with "
            "tsara.core.timebase before returning."
        )
    # Resolution is pinned, not merely checked: a µs index round-trips through
    # netCDF as ns and would change dtype on save/load, which quietly breaks
    # any later exact index comparison.
    if index.dtype != "datetime64[ns]":
        raise TsaraIngestError(
            f"Reader '{reader_name}' returned {table.path} with index dtype "
            f"{index.dtype}; RawTable requires datetime64[ns]."
        )
    if index.name != TIME_INDEX_NAME:
        raise TsaraIngestError(
            f"Reader '{reader_name}' returned {table.path} with index name "
            f"{index.name!r}; RawTable requires it to be {TIME_INDEX_NAME!r}."
        )
    if frame.columns.has_duplicates:
        duplicated = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise TsaraIngestError(
            f"Reader '{reader_name}' returned {table.path} with duplicate column "
            f"names {duplicated}; column selection would be ambiguous."
        )
    return table
