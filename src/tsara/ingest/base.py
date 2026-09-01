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
from tsara.core.naming import TIME_COORD

if TYPE_CHECKING:  # pragma: no cover
    import logging

    import pandas as pd

    from tsara.config.manifest import LoaderConfig

__all__ = [
    "RawTable",
    "Reader",
    "TIME_INDEX_NAME",
    "TsaraIngestError",
    "check_dropped_rows",
    "check_raw_table",
    "float_precision_kwarg",
]

#: Name TSARA gives the time index everywhere, from raw table to final stream.
#:
#: Bound to :data:`tsara.core.naming.TIME_COORD` rather than spelled again,
#: because the stream assembler names the finished axis from *that* constant
#: (:mod:`tsara.ingest.streams`). Two literals that merely happen to match
#: would let a rename split the reader contract from the stream it feeds --
#: the exact coupling :mod:`tsara.core.naming` exists to remove. The separate
#: name is kept because a reader is checked against a raw *index*, not against
#: a finished coordinate, and the contract reads better for saying so.
TIME_INDEX_NAME = TIME_COORD


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
        ``time``, contains ``NaT``, or the frame carries duplicate column
        names.
    """
    import numpy as np
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
    # A NaT is the same class of failure as the tz-aware index above: it
    # survives sorting, slicing and binning without complaint, then places
    # rows at an undefined point on the timeline. Every reader TSARA ships
    # drops unparseable timestamps itself, so this clause exists for readers
    # registered from outside -- which is exactly the population the contract
    # claims to police and the only one that could otherwise violate it.
    if index.hasnans:
        raise TsaraIngestError(
            f"Reader '{reader_name}' returned {table.path} with "
            # Counted through numpy: pandas-stubs types `.isna()` as
            # Index[bool], which has no `.sum()` as far as mypy is concerned.
            f"{int(np.count_nonzero(np.asarray(index.isna())))} NaT "
            "timestamp(s) in the index. Rows "
            "whose timestamp could not be parsed must be dropped by the "
            "reader, not carried forward."
        )
    if frame.columns.has_duplicates:
        duplicated = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise TsaraIngestError(
            f"Reader '{reader_name}' returned {table.path} with duplicate column "
            f"names {duplicated}; column selection would be ambiguous."
        )
    return table


def check_dropped_rows(
    *,
    n_dropped: int,
    n_total: int,
    path: Path,
    reason: str,
    max_fraction: float,
    logger: logging.Logger,
) -> None:
    """Report discarded rows, and refuse the file if too many were lost.

    Every reader discards rows it cannot place on a time axis, and every
    reader already refused a file where *no* row survived. That left a gap
    exactly where it hurts most: a file that yields 2 rows out of 10,235 is
    a misparse, not a thin dataset, but it produced only a warning — and
    three stages later it is indistinguishable from "this instrument barely
    ran that day". Warnings scroll past in a run over a thousand files;
    a raised error does not.

    This generalizes the old all-or-nothing rule to a threshold the manifest
    sets (``LoaderConfig.max_dropped_fraction``), and is shared by all three
    readers so the policy cannot drift between formats.

    Parameters
    ----------
    n_dropped : int
        Rows discarded.
    n_total : int
        Rows considered, before discarding.
    path : pathlib.Path
        File being read, for the message.
    reason : str
        Short phrase completing "Dropped N of M rows from PATH: ..." — e.g.
        ``"timestamp did not parse"``.
    max_fraction : float
        Largest tolerable ``n_dropped / n_total``. Exceeding it raises.
    logger : logging.Logger
        The *calling reader's* logger, so the message is attributed to the
        module that read the file rather than to this one.

    Raises
    ------
    TsaraIngestError
        If more than ``max_fraction`` of the rows were discarded.
    """
    if n_dropped <= 0:
        return
    fraction = n_dropped / n_total if n_total else 1.0
    if fraction > max_fraction:
        raise TsaraIngestError(
            f"Dropped {n_dropped} of {n_total} rows ({fraction:.1%}) from '{path}': "
            f"{reason}. That exceeds max_dropped_fraction={max_fraction:.1%}, so this "
            "is being treated as a misparse rather than a successful read. Check the "
            "loader configuration against the file, or raise max_dropped_fraction if "
            "the loss is genuinely expected."
        )
    logger.warning("Dropped %d of %d rows from %s: %s.", n_dropped, n_total, path, reason)


def float_precision_kwarg(loader: object) -> dict[str, str]:
    """Return the ``pandas.read_csv`` keyword for a loader's float precision.

    Kept in one place because two readers must interpret the same manifest
    field identically, and because the mapping is not obvious: pandas spells
    the exact mode ``"round_trip"``, while the manifest spells it ``"exact"``
    -- the manifest names the guarantee, not the library's implementation of
    it. Passing nothing at all for the fast mode preserves pandas' default
    rather than pinning a spelling that could change.

    Parameters
    ----------
    loader : object
        Any loader config. One without a ``float_precision`` field (parquet,
        which parses no text) yields no keyword.

    Returns
    -------
    dict
        Keyword arguments to splat into ``pandas.read_csv``.
    """
    if getattr(loader, "float_precision", "fast") == "exact":
        return {"float_precision": "round_trip"}
    return {}
