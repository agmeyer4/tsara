"""Running a whole campaign's ingestion: manifest in, streams out.

Everything else in :mod:`tsara.ingest` does one job on one thing. This module
is the one that turns a validated :class:`~tsara.config.manifest.Manifest`
into the object later phases actually start from, by driving the pieces in
the one order that is correct:

    crawl → read each file → concatenate → sort → de-duplicate → assemble

Why concatenate before assembling, rather than per file
-------------------------------------------------------
Several per-variable decisions are *campaign-level* quantities, and any of
them evaluated per file would give a different answer at every file
boundary — an archive split into hourly files would not agree with the same
data in daily files. De-duplication is inherently cross-file; so is the
empirical noise estimate a later phase computes from this record; so is any
rolling statistic. Rather than sort out which rules happen to be pointwise
today, all of an instrument's files become one table first, and every
per-variable decision is made once against the whole record.

Sorting is not a formality
--------------------------
Files crawled across several directory layouts arrive in path order, not
time order, and an instrument's own timestamps cannot be assumed sorted
either: logger clock corrections, buffered writes and merge steps in an
upstream processing chain all produce records that step backwards
occasionally. Downstream, everything from rolling baselines to event
intervals assumes a monotonic axis — :func:`~tsara.ingest.streams.build_stream`
refuses one that is not — so sorting happens here, once, where the whole
record is in hand.

What happens when a file will not read
--------------------------------------
It is logged and skipped, and the run continues; an instrument that loses
*every* file is an error. The alternative — aborting the campaign on the
first bad file — is the wrong trade for an archive of a few thousand files
on a cluster, where a single truncated file should not cost a twenty-minute
run. Skips are never silent: each is logged at ERROR level with its reason,
and the total is reported per instrument.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.naming import LOD_COUNT_KEY, SupportLabel
from tsara.core.support import nominal_cadence_ns
from tsara.core.timebase import epoch_ns
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.crawler import crawl
from tsara.ingest.registry import read_file
from tsara.ingest.streams import build_stream
from tsara.ingest.support import (
    LABEL_HINT_KEY,
    ResolvedSupport,
    resolve_support,
    shift_and_centre,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import numpy.typing as npt
    import xarray as xr

    from tsara.config.manifest import InstrumentConfig, Manifest

logger = logging.getLogger(__name__)

__all__ = ["StreamCollection", "ingest_campaign"]


@dataclass(frozen=True)
class StreamCollection(Mapping[str, "xr.Dataset"]):
    """A campaign's native-rate streams, one per instrument.

    The ingestion counterpart of
    :class:`~tsara.synthetic.generator.SyntheticDataset`, and deliberately
    the same shape: a mapping of instrument name to
    :class:`xarray.Dataset`, so that later phases accept either without
    knowing which they were given.

    Why it inherits :class:`collections.abc.Mapping`
    ------------------------------------------------
    Because the paragraph above has to be *true*. Defining ``__getitem__``
    and ``__len__`` by hand without ``__iter__`` left the class half a
    mapping: Python's legacy iteration protocol then falls back to calling
    ``__getitem__(0)``, so the first thing a user writes — ``for name in
    streams`` or ``sorted(streams)`` — failed with ``KeyError: 0``, an error
    naming neither the real problem nor this class. Meanwhile
    ``SyntheticDataset.streams`` is a plain dict and iterates fine, so the
    two objects that later phases are supposed to accept interchangeably
    behaved differently in the most basic loop. Inheriting the ABC supplies
    ``__iter__``-driven ``keys``/``items``/``values``/``get`` from the three
    methods below, which is less code than the wart was.

    Attributes
    ----------
    streams : dict of str to xarray.Dataset
        One dataset per instrument, on that instrument's own timestamps.
        Nothing here has been resampled (``docs/METHODS.md`` §1.1).
    manifest : Manifest
        The configuration that produced these streams, kept so a collection
        can be saved, reloaded and audited without a separate file.
    """

    streams: dict[str, xr.Dataset]
    manifest: Manifest

    def __getitem__(self, instrument: str) -> xr.Dataset:
        """Return one instrument's stream."""
        return self.streams[instrument]

    def __contains__(self, instrument: object) -> bool:
        """Return whether an instrument was ingested."""
        return instrument in self.streams

    def __iter__(self) -> Iterator[str]:
        """Iterate over instrument names, in manifest order."""
        return iter(self.streams)

    def __len__(self) -> int:
        """Return the number of ingested instruments."""
        return len(self.streams)


@dataclass
class _Ingested:
    """One instrument's concatenated table plus the files behind it."""

    frame: pd.DataFrame
    #: What TSARA concluded about this instrument's cells, and on what basis.
    #: Resolved here rather than in stream assembly because the per-file
    #: boundaries this is derived from only exist before concatenation.
    support: ResolvedSupport
    files: list[Path] = field(default_factory=list)
    #: What the files said about themselves, reconciled across all of them.
    #: See :func:`_merge_file_attrs`.
    file_attrs: dict[str, object] = field(default_factory=dict)


#: How many differing values of one attr key to list before summarizing.
#: Chosen so a handful of revisions or processing levels stays fully
#: readable while a per-file key (an ICARTT data date) is condensed.
_MAX_DISTINCT_ATTR_VALUES = 8


def _merge_file_attrs(per_file: list[Mapping[str, object]]) -> dict[str, object]:
    """Reconcile what each file declared about itself into one mapping.

    A reader returns :attr:`~tsara.ingest.base.RawTable.attrs` per file, and
    an instrument is usually many files. Most keys are campaign constants —
    the PI, the mission, the LOD flag values — and simply agree. The
    interesting case is when they do not, and the rule here is to *say so*
    rather than to pick: a silent choice between two PIs, or two different
    LOD flags, would put a false statement in a saved product that claims to
    be self-describing (CLAUDE.md §5).

    Counts (:data:`LOD_COUNT_KEY`) are summed instead, since a tally over
    files is exactly the tally over the concatenated record.
    """
    merged: dict[str, object] = {}
    totals: dict[str, int] = {}
    values: dict[str, list[object]] = {}

    for attrs in per_file:
        for key, value in attrs.items():
            if key == LOD_COUNT_KEY and isinstance(value, Mapping):
                for column, count in value.items():
                    totals[str(column)] = totals.get(str(column), 0) + int(count)
                continue
            seen = values.setdefault(key, [])
            if value not in seen:
                seen.append(value)

    for key, distinct in values.items():
        if len(distinct) == 1:
            merged[key] = distinct[0]
        else:
            # Joined, not dropped: that four files in a campaign carry
            # different revision strings is a fact worth reading in
            # `ncdump -h`, and it is the kind of thing nobody thinks to
            # check until an analysis disagrees with a colleague's.
            #
            # Summarized past a threshold, because some keys differ in every
            # file by design -- an ICARTT data date does -- and a thousand-file
            # instrument would otherwise write a thousand-item attr that is
            # unreadable in `ncdump -h` and useless as provenance. The first
            # and last of the sorted values plus a count says the same thing
            # in a line, and still makes the disagreement visible.
            ordered_values = sorted(str(item) for item in distinct)
            if len(ordered_values) > _MAX_DISTINCT_ATTR_VALUES:
                merged[key] = (
                    f"{ordered_values[0]} ... {ordered_values[-1]} "
                    f"({len(ordered_values)} distinct values)"
                )
            else:
                merged[key] = "; ".join(ordered_values)
    if totals:
        merged[LOD_COUNT_KEY] = totals
    return merged


def ingest_campaign(
    manifest: Manifest, *, instruments: Sequence[str] | None = None
) -> StreamCollection:
    """Ingest every instrument a manifest describes.

    Parameters
    ----------
    manifest : Manifest
        Validated manifest with an absolute ``base_path`` (which
        :func:`tsara.config.loader.load_manifest` guarantees).
    instruments : Sequence of str, optional
        Restrict ingestion to these instrument names. Useful in a notebook
        for iterating on one instrument, and on a cluster for splitting a
        campaign across jobs. ``None`` (default) ingests all of them.

    Returns
    -------
    StreamCollection
        One stream per requested instrument.

    Raises
    ------
    TsaraIngestError
        If a requested instrument is not in the manifest, an instrument
        matches no files, or every one of its files fails to read.
    """
    selected = _select(manifest, instruments)

    streams: dict[str, xr.Dataset] = {}
    for name in selected:
        instrument = manifest.instruments[name]
        logger.info("Ingesting instrument '%s'.", name)
        ingested = _ingest_instrument(manifest, name, instrument)
        streams[name] = build_stream(
            ingested.frame,
            instrument,
            name=name,
            platform=manifest.platform,
            campaign=manifest.name,
            files=ingested.files,
            file_attrs=ingested.file_attrs,
            support=ingested.support,
            time_shift=instrument.time_shift,
        )
        logger.info(
            "Instrument '%s': %d samples from %d file(s).",
            name,
            len(ingested.frame),
            len(ingested.files),
        )

    return StreamCollection(streams=streams, manifest=manifest)


def _select(manifest: Manifest, instruments: Sequence[str] | None) -> list[str]:
    """Resolve the requested instrument names against the manifest."""
    if instruments is None:
        return list(manifest.instruments)
    unknown = [name for name in instruments if name not in manifest.instruments]
    if unknown:
        raise TsaraIngestError(
            f"Manifest '{manifest.name}' has no instrument(s) {unknown}. "
            f"Available: {list(manifest.instruments)}."
        )
    return list(instruments)


def _ingest_instrument(manifest: Manifest, name: str, instrument: InstrumentConfig) -> _Ingested:
    """Crawl, read, concatenate, sort and de-duplicate one instrument."""
    matches = crawl(manifest.base_path, instrument.loader, instrument.metadata)
    logger.debug("Instrument '%s': %d file(s) matched.", name, len(matches))

    frames: list[pd.DataFrame] = []
    files: list[Path] = []
    file_attrs: list[Mapping[str, object]] = []
    # Cadence is measured per FILE, and that is load-bearing rather than
    # incidental: one instrument's files can legitimately disagree about it.
    # Measured in the target archive, some met records run at 1 s in one file
    # and 5 s in another, and a single instrument-wide cadence would give one
    # of them cells of the wrong width. Here is the only place the per-file
    # boundaries still exist.
    cadences: list[int | None] = []
    hints: list[str | None] = []
    failures = 0
    for match in matches:
        try:
            table = read_file(match.path, instrument.loader)
        except TsaraIngestError as exc:
            # Logged, not raised: one unreadable file must not cost a whole
            # campaign's run. The count is reported below so this can never
            # pass unnoticed.
            failures += 1
            logger.error("Skipping '%s' for instrument '%s': %s", match.path, name, exc)
            continue
        frames.append(table.frame)
        files.append(match.path)
        cadences.append(nominal_cadence_ns(epoch_ns(pd.DatetimeIndex(table.frame.index))))
        hint = table.attrs.get(LABEL_HINT_KEY)
        hints.append(str(hint) if hint is not None else None)
        # Kept, not discarded: this is what the file said about *itself*
        # (PI, mission, revision, LOD flags), as distinct from what the
        # manifest says about it. Dropping it here used to make the reader's
        # careful header harvest unreachable by every later stage.
        file_attrs.append(table.attrs)

    if not frames:
        raise TsaraIngestError(
            f"Instrument '{name}' matched {len(matches)} file(s) but none could "
            "be read. See the logged errors above for the reason on each."
        )
    if failures:
        logger.warning(
            "Instrument '%s': skipped %d of %d file(s) that failed to read.",
            name,
            failures,
            len(matches),
        )

    combined = frames[0] if len(frames) == 1 else pd.concat(frames)
    # Resolved before ordering, because the per-row width array below is
    # built in concatenation order; sorting first would misalign every file's
    # cadence with the rows it belongs to.
    combined, support = resolve_support(
        combined,
        instrument.loader.support,
        widths_ns=_per_row_widths(frames, cadences),
        label_hint=_agreed_hint(hints),
        path=files[0] if len(files) == 1 else Path(f"<{len(files)} files>"),
    )
    # Correct the clock and centre the axis before ordering, not after:
    # centring can reorder rows when widths vary per file, so the sort has to
    # see the axis the stream will actually carry.
    combined = shift_and_centre(combined, shift_ns=_shift_ns(instrument.time_shift, name=name))
    # Counted last, on the axis the rows are actually de-duplicated on. See
    # `_n_within_file` for why counting it any earlier was wrong.
    n_within = _n_within_file(combined.index, [len(frame) for frame in frames])
    return _Ingested(
        frame=_order(combined, name, n_within=n_within),
        files=files,
        file_attrs=_merge_file_attrs(file_attrs),
        support=support,
    )


def _n_within_file(index: pd.Index, sizes: Sequence[int]) -> int:
    """Count the duplicate timestamps one file explains, on the FINAL axis.

    :func:`_order` reports its dropped rows split two ways -- duplicated
    inside a single file, or duplicated because two files cover the same
    period -- because the two call for opposite fixes. That split is only
    meaningful if both halves are counted on the *same* time axis, and they
    were not: within-file duplicates were counted on each file's raw index
    before concatenation, while the total is counted after centring.

    Centring is not a translation when widths vary per row. A row whose cell
    is wide moves further than its neighbour, so two rows sharing a raw
    timestamp but declaring different stops land on *different* midpoints and
    stop being duplicates. The raw count could then exceed the final total
    and the reported overlap came out **negative** -- pointing a user at the
    manifest's path templates for rows that no two files ever shared. That is
    the same wrong accusation the Phase-3 walkthrough removed from this
    message once already, arrived at from the other direction.

    Counting per file on the final axis makes the split exact rather than
    approximate. For one instant held by ``c_i`` rows in each of ``k`` files,
    the total counts ``sum(c_i) - 1`` and this counts ``sum(c_i - 1)``, so the
    remainder is ``k - 1``: the number of *extra files* holding that instant,
    which is precisely what "overlap between files" means, and which cannot
    be negative.

    Parameters
    ----------
    index : pandas.Index
        The concatenated record's timestamps, still in concatenation order --
        which is what makes the per-file slices below correct. Neither
        :func:`~tsara.ingest.support.resolve_support` nor
        :func:`~tsara.ingest.support.shift_and_centre` reorders rows; the sort
        happens afterwards, inside :func:`_order`.
    sizes : Sequence of int
        Row count of each file, in the same order.

    Returns
    -------
    int
        How many rows duplicate an earlier row *of their own file*.
    """
    edges = np.cumsum([0, *sizes])
    return sum(
        int(index[start:stop].duplicated(keep="first").sum())
        for start, stop in zip(edges[:-1], edges[1:], strict=True)
    )


def _shift_ns(time_shift: str | None, *, name: str) -> int:
    """Return an instrument's declared clock correction in nanoseconds."""
    if time_shift is None:
        return 0
    shift = int(pd.Timedelta(time_shift).value)
    if shift:
        # Logged because a silent clock change is the one correction nobody
        # can spot afterwards: the numbers stay plausible and only their
        # relationship to another instrument moves.
        logger.info("Instrument '%s': applying declared time_shift %s.", name, time_shift)
    return shift


def _per_row_widths(
    frames: Sequence[pd.DataFrame], cadences: Sequence[int | None]
) -> npt.NDArray[np.int64] | None:
    """Expand each file's measured cadence to one width per row.

    A file too short to have a cadence borrows the median of the files that
    do, which is the least-surprising stand-in and is only ever reached by a
    file of one or two rows. When *no* file was long enough, there is nothing
    to borrow and None says so rather than inventing a number.
    """
    known = [cadence for cadence in cadences if cadence is not None]
    if not known:
        return None
    fallback = int(np.median(known))
    per_file = [fallback if cadence is None else cadence for cadence in cadences]
    return np.repeat(
        np.asarray(per_file, dtype=np.int64),
        np.asarray([len(frame) for frame in frames], dtype=np.int64),
    )


#: The label vocabulary, as a lookup that both validates and types.
#:
#: A hint arrives from ``RawTable.attrs``, which is untyped by design, so it
#: has to be checked before it can be trusted as a label. A dict does that and
#: gives the checker a typed result, where a membership test would give
#: neither.
_LABELS: dict[str, SupportLabel] = {
    "start": "start",
    "mid": "mid",
    "end": "end",
    "unknown": "unknown",
}


def _agreed_hint(hints: Sequence[str | None]) -> SupportLabel | None:
    """Return the label hint only when every file agrees on it.

    A disagreement means the instrument's files do not share a convention,
    and picking a winner would put half of them half a cell out. Returning
    None instead lets the resolver fall back to a centred cell, which is
    wrong by at most half that on every file rather than fully wrong on some.
    """
    distinct = {hint for hint in hints if hint is not None}
    if len(distinct) != 1 or any(hint is None for hint in hints):
        return None
    return _LABELS.get(distinct.pop())


def _order(frame: pd.DataFrame, name: str, *, n_within: int = 0) -> pd.DataFrame:
    """Sort by time and drop duplicate timestamps.

    A stable sort is used so that rows sharing a timestamp keep the order
    their files were crawled in, which is path order and therefore
    reproducible. The first row of each duplicated timestamp is kept.

    Keeping the first is a *policy*, not a truth: overlapping files may
    genuinely disagree, and averaging them would silently invent a value
    while erroring would reject archives that legitimately overlap. Keeping
    one real measurement and saying how many were dropped is the option that
    neither fabricates nor hides. If a campaign ever needs "last wins" or a
    per-instrument choice, this is the single place it would be configured.

    The warning distinguishes the two causes rather than guessing between
    them, because they call for opposite responses and the guess was wrong
    on the archive this was built for. Measured on the 43-file PTR-MS set:
    all 7,242 dropped rows were duplicated *within* a single file and none
    came from overlap between files, yet the message asked "Overlapping
    files?" — pointing at the crawler and the revision policy, both
    innocent. Within-file duplicates mean the instrument wrote two records
    under one timestamp (there, a nominally 1 Hz logger with 1 s
    resolution), so the fix is a resolution or averaging decision; overlap
    between files means the archive really does hold the same period twice,
    and the fix is in the manifest's path templates.
    """
    ordered = frame.sort_index(kind="stable")

    duplicated = ordered.index.duplicated(keep="first")
    n_duplicate = int(duplicated.sum())
    if n_duplicate:
        # `n_within` is counted per file on this same axis, which is what
        # makes the split exact rather than heuristic: a timestamp repeated
        # inside one file is still duplicated in the combined table, so it is
        # the part of the total that overlap cannot explain, and the
        # remainder is the part it can. See `_n_within_file` for why the
        # axis, not just the grouping, is the load-bearing part.
        n_across = n_duplicate - n_within
        logger.warning(
            "Instrument '%s': dropped %d row(s) sharing a timestamp with an "
            "earlier row (kept the first of each): %d duplicated within a "
            "single file%s, %d from overlap between files%s.",
            name,
            n_duplicate,
            n_within,
            " (the instrument logged two records under one timestamp)" if n_within else "",
            n_across,
            " (check the manifest's path templates)" if n_across else "",
        )
        ordered = ordered.loc[~duplicated]
    return ordered
