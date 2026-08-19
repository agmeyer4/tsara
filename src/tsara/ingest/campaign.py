"""Running a whole campaign's ingestion: manifest in, streams out.

Everything else in :mod:`tsara.ingest` does one job on one thing. This module
is the one that turns a validated :class:`~tsara.config.manifest.Manifest`
into the object later phases actually start from, by driving the pieces in
the one order that is correct:

    crawl → read each file → concatenate → sort → de-duplicate → assemble

Why concatenate before assembling, rather than per file
-------------------------------------------------------
QA/QC windows and uncertainty are *campaign-level* quantities. A rolling
spike test evaluated per file gives a different answer at every file
boundary, and an archive split into hourly files would produce different
masking than the same data in daily files. So all of an instrument's files
become one table first, and every per-variable decision is made once against
the whole record.

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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from tsara.ingest.base import TsaraIngestError
from tsara.ingest.crawler import crawl
from tsara.ingest.registry import read_file
from tsara.ingest.streams import build_stream

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from pathlib import Path

    import xarray as xr

    from tsara.config.manifest import InstrumentConfig, Manifest

logger = logging.getLogger(__name__)

__all__ = ["StreamCollection", "ingest_campaign"]


@dataclass(frozen=True)
class StreamCollection:
    """A campaign's native-rate streams, one per instrument.

    The ingestion counterpart of
    :class:`~tsara.synthetic.generator.SyntheticDataset`, and deliberately
    the same shape: a mapping of instrument name to
    :class:`xarray.Dataset`, so that later phases accept either without
    knowing which they were given.

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

    def __len__(self) -> int:
        """Return the number of ingested instruments."""
        return len(self.streams)


@dataclass
class _Ingested:
    """One instrument's concatenated table plus the files behind it."""

    frame: pd.DataFrame
    sources: list[Path] = field(default_factory=list)


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
    return _Ingested(frame=_order(combined, name), sources=sources)


def _order(frame: pd.DataFrame, name: str) -> pd.DataFrame:
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
    """
    ordered = frame.sort_index(kind="stable")

    duplicated = ordered.index.duplicated(keep="first")
    n_duplicate = int(duplicated.sum())
    if n_duplicate:
        logger.warning(
            "Instrument '%s': dropped %d row(s) sharing a timestamp with an "
            "earlier row (kept the first of each). Overlapping files?",
            name,
            n_duplicate,
        )
        ordered = ordered.loc[~duplicated]
    return ordered
