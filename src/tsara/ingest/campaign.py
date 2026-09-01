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
from typing import TYPE_CHECKING

import pandas as pd

from tsara.core.naming import LOD_COUNT_KEY
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.crawler import crawl
from tsara.ingest.registry import read_file
from tsara.ingest.streams import build_stream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterator, Sequence
    from pathlib import Path

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
    sources: list[Path] = field(default_factory=list)
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
            sources=ingested.sources,
            file_attrs=ingested.file_attrs,
        )
        logger.info(
            "Instrument '%s': %d samples from %d file(s).",
            name,
            len(ingested.frame),
            len(ingested.sources),
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
    sources: list[Path] = []
    file_attrs: list[Mapping[str, object]] = []
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
        sources.append(match.path)
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

    # Counted here, where the per-file boundaries still exist. After
    # concatenation the two causes are indistinguishable.
    n_within = sum(int(f.index.duplicated(keep="first").sum()) for f in frames)

    combined = frames[0] if len(frames) == 1 else pd.concat(frames)
    return _Ingested(
        frame=_order(combined, name, n_within=n_within),
        sources=sources,
        file_attrs=_merge_file_attrs(file_attrs),
    )


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
        # `n_within` is counted per file before concatenation, which is
        # what makes the split exact rather than heuristic: a timestamp
        # repeated inside one file is still duplicated in the combined
        # table, so it is the part of the total that overlap cannot
        # explain, and the remainder is the part it can.
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
