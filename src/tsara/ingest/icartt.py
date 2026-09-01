"""Reader and filename utilities for the ICARTT FFI-1001 format.

Why TSARA ships its own parser
------------------------------
The ``icartt`` package on PyPI is GPL-3.0, which is incompatible with TSARA's
MIT license, and it has been unmaintained since 2022 (decided 2026-07-08).
Owning the parser also buys something the dependency could not: tolerance for
the spec-noncompliant files that real campaign archives are full of. That is
not a hypothetical concern — see "What real files actually do" below.

The format, briefly
-------------------
An FFI-1001 file is a plain-text header followed by comma-separated data. The
header's first line declares its own length (``NLHEAD``), so data always
begins at line ``NLHEAD + 1``, and the header is self-describing::

    line 1        NLHEAD, FFI              (FFI is 1001 for this format)
    line 2        principal investigator
    line 3        organization
    line 4        data source description
    line 5        mission name
    line 6        volume number, total volumes
    line 7        data date, revision date  (YYYY, MM, DD, YYYY, MM, DD)
    line 8        DX                        (interval; 0 = non-uniform)
    line 9        independent variable      (name, units, description)
    line 10       NV                        (number of dependent variables)
    line 11       VSCAL                     (NV scale factors)
    line 12       VMISS                     (NV missing-value sentinels)
    lines 13..    NV dependent-variable definitions
    then          NSCOML + that many special comment lines
    then          NNCOML + that many normal comment lines

The **last normal comment line is the data column header** — a quirk of the
format worth knowing, since it means the column names are found at the end of
a comment block rather than anywhere obvious.

What real files actually do
---------------------------
Measured across the 1122 ICARTT files in the campaign archive this package
targets. All are FFI-1001, but:

* **The independent variable is usually seconds past midnight** of the data
  date — under at least a dozen different spellings of both name and units
  (``Time_Start``/``starttime_UTC``/``TIMESTAMP_UTC``/``IgorTime``... crossed
  with ``seconds``/``seconds_past_midnight``/``seconds_since_midnight``...).
  Notably ``IgorTime`` is *also* seconds past midnight, despite the name
  suggesting Igor Pro's 1904 epoch. **So this parser keys off the values, not
  the labels**: numeric means seconds past midnight; anything else is parsed
  as datetimes. That is robust to every spelling seen and to the next one.
* **43 files contradict their own header**, declaring
  ``number_of_seconds_from_0000_UTC`` and then writing
  ``2024-08-01 19:26:00`` in that column. They are the PTR-MS VOC files —
  35 species each, the most valuable data in the archive for source
  fingerprinting — so refusing them is not an option.
* **30 files are not valid UTF-8**, so decoding is tolerant.
* **``NV`` ranges from 1 to 59**, which is the "dozens of species" scale the
  package exists to handle.
* **Missing-value sentinels vary per file and per variable** (``-9999``,
  ``-99999``, ``-9999999``, ``-9.999e50``...), which is exactly why they are
  read from the header rather than assumed.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from tsara.config.manifest import ICARTTLoader
from tsara.ingest.base import (
    RawTable,
    TsaraIngestError,
    check_dropped_rows,
    float_precision_kwarg,
)
from tsara.ingest.registry import register_reader
from tsara.ingest.timeparse import to_utc_naive_ns

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from tsara.config.manifest import LoaderConfig

logger = logging.getLogger(__name__)

__all__ = [
    "IcarttFilename",
    "IcarttHeader",
    "IcarttVariable",
    "parse_icartt_filename",
    "parse_icartt_header",
    "read_icartt",
    "select_latest_revisions",
]

#: Keys ICARTT files conventionally put in their special-comment block as
#: ``KEY: value``. Scraped into header metadata so downstream stages can see
#: limit-of-detection flags and provenance without re-reading the file.
_METADATA_LINE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*:\s*(.*)$")

#: Revision token in an ICARTT filename: ``R`` followed by a single letter
#: (preliminary/field data: RA, RB, RC...) or digits (final data: R0, R1...).
_REVISION = re.compile(r"^R(?:[A-Z]|\d+)$")

#: The ``YYYYMMDD`` field every ICARTT filename carries.
_FILE_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


@dataclass(frozen=True)
class IcarttVariable:
    """One variable declared in an ICARTT header.

    Attributes
    ----------
    name : str
        Short name, matching the data column header.
    units : str
        Units string as declared.
    description : str
        Free-text description (may be empty).
    scale : float
        Multiplicative scale factor from the ``VSCAL`` line. Applied *after*
        missing values are masked, never before — scaling a ``-9999``
        sentinel would turn it into an unrecognizable number.
    missing : float
        Sentinel from the ``VMISS`` line marking absent data.
    """

    name: str
    units: str
    description: str
    scale: float = 1.0
    missing: float = float("nan")


@dataclass(frozen=True)
class IcarttHeader:
    """Everything an FFI-1001 header declares about its file.

    Exposed as a public dataclass rather than kept private because the header
    is genuinely useful on its own: it names the PI, the platform and the
    instrument, and a crawler can read one cheaply to decide whether the file
    is wanted before parsing megabytes of data.

    Attributes
    ----------
    n_header_lines : int
        ``NLHEAD``; data begins at this 0-based line index.
    file_format_index : int
        ``FFI``; 1001 for this format.
    pi_name, organization, data_source, mission : str
        Provenance lines 2-5.
    volume : tuple of int
        ``(volume number, total volumes)``.
    data_date : datetime.date
        UTC date the data was collected — the origin for seconds-past-midnight
        timestamps.
    revision_date : datetime.date
        Date this revision was produced.
    interval : float
        ``DX``; 0 means non-uniform sampling.
    independent_variable : IcarttVariable
        The time axis definition.
    variables : tuple of IcarttVariable
        The ``NV`` dependent variables, in column order.
    special_comments, normal_comments : tuple of str
        The two comment blocks, verbatim.
    column_names : tuple of str
        Data column names: the independent variable followed by the
        dependent ones, taken from the last normal comment line.
    metadata : dict
        ``KEY: value`` pairs scraped from the whole header (e.g.
        ``ULOD_FLAG``, ``LLOD_FLAG``, ``PLATFORM``, ``REVISION``). Scraped
        rather than read from the comment blocks because those are located
        by ``12 + NV`` arithmetic that a malformed variable count silently
        invalidates; see :func:`_metadata_scan_region`.
    """

    n_header_lines: int
    file_format_index: int
    pi_name: str
    organization: str
    data_source: str
    mission: str
    volume: tuple[int, int]
    data_date: date
    revision_date: date
    interval: float
    independent_variable: IcarttVariable
    variables: tuple[IcarttVariable, ...]
    special_comments: tuple[str, ...]
    normal_comments: tuple[str, ...]
    column_names: tuple[str, ...]
    metadata: dict[str, str]


@dataclass(frozen=True)
class IcarttFilename:
    """The parts of a standard ICARTT filename.

    ICARTT names files ``dataID_locationID_YYYYMMDD[_R#][_comments].ict``.
    TSARA parses this by **locating the 8-digit date token**, not by counting
    underscores, because ``dataID`` and ``locationID`` are not reliably one
    token each: 147 files in the target archive look like
    ``SLCSOS-ROZE-O3_UWyo_Sprinter_20240802_RA_L1.ict``, where the leading
    identifier spans three underscore-separated parts.

    Attributes
    ----------
    identifier : str
        Everything before the date — dataID and locationID together. Kept
        joined because the split between them is not recoverable in general
        and nothing downstream needs it separately.
    file_date : datetime.date
        The ``YYYYMMDD`` field.
    revision : str or None
        The revision token (``R0``, ``RA``, ...) if present.
    comment : str
        Everything after the revision, joined by underscores. This is *not*
        decoration: it distinguishes genuinely different files
        (``L1``/``L2`` processing levels, ``Drive01_LakeBreeze`` vs
        ``Stationary01``), so it is part of a file's identity.
    """

    identifier: str
    file_date: date
    revision: str | None
    comment: str

    @property
    def dedup_key(self) -> tuple[str, date, str]:
        """Identity under which revisions of *the same data* are compared.

        Includes ``comment`` deliberately. Keying on identifier and date
        alone — the obvious reading of "one file per instrument per day" —
        collapses distinct drives and processing levels into one another and
        silently discards real data (198 files rather than 67 in the target
        archive).
        """
        return (self.identifier, self.file_date, self.comment)

    @property
    def revision_rank(self) -> tuple[int, int, str]:
        """Sort key ordering revisions from oldest to newest.

        The ICARTT convention is that **alphabetic revisions are preliminary
        field data and numeric revisions are final**, so any ``R#`` supersedes
        any ``R<letter>``. The archive states this in its own comment blocks
        ("R0: Final data... RA: Preliminary Data. For in-field use only."),
        and 41 groups there hold both kinds, so the ordering is load-bearing
        rather than theoretical. A missing revision sorts lowest of all.
        """
        if self.revision is None:
            return (0, 0, "")
        token = self.revision[1:]
        if token.isdigit():
            return (2, int(token), "")
        return (1, 0, token)


def parse_icartt_filename(path: Path | str) -> IcarttFilename | None:
    """Parse a standard ICARTT filename.

    Parameters
    ----------
    path : pathlib.Path or str
        File path or bare filename.

    Returns
    -------
    IcarttFilename or None
        Parsed parts, or ``None`` if the name carries no ``YYYYMMDD`` token
        and therefore does not follow the convention at all. Returning
        ``None`` rather than raising lets a crawler pass such files through
        untouched instead of refusing the whole archive.
    """
    stem = path.stem if hasattr(path, "stem") else str(path).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parts = str(stem).split("_")

    for index, part in enumerate(parts):
        match = _FILE_DATE.match(part)
        if not match:
            continue
        try:
            file_date = date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            # An 8-digit run that is not a real date (e.g. a serial number);
            # keep scanning rather than declaring the name unparseable.
            continue
        identifier = "_".join(parts[:index])
        rest = parts[index + 1 :]
        has_revision = bool(rest) and bool(_REVISION.match(rest[0]))
        revision = rest[0] if has_revision else None
        comment = "_".join(rest[1:] if has_revision else rest)
        return IcarttFilename(
            identifier=identifier, file_date=file_date, revision=revision, comment=comment
        )
    return None


def select_latest_revisions(paths: list[Path]) -> list[Path]:
    """Keep only the newest revision of each distinct ICARTT product.

    Implements ``ICARTTLoader.revision_policy='latest'``. Archives routinely
    hold several revisions of one day's data (``R0`` preliminary, ``R1``
    final), and ingesting them all double-counts the same air.

    The subtlety this function exists to get right: **two files that differ
    only after the revision token are different data, not revisions of each
    other.** Grouping is therefore on
    :attr:`IcarttFilename.dedup_key`, which retains the trailing comment.

    Files whose names do not follow the convention are always kept — an
    unparseable name is not evidence of duplication.

    Parameters
    ----------
    paths : list of pathlib.Path
        Candidate files.

    Returns
    -------
    list of pathlib.Path
        Selected files, sorted for deterministic ordering.
    """
    best: dict[tuple[str, date, str], tuple[tuple[int, int, str], Path]] = {}
    keep: list[Path] = []

    for path in paths:
        parsed = parse_icartt_filename(path)
        if parsed is None:
            keep.append(path)
            continue
        key = parsed.dedup_key
        rank = parsed.revision_rank
        current = best.get(key)
        # Ties (same key and rank) are broken by path so the choice is stable
        # across filesystem orderings rather than dependent on scan order.
        if current is None or (rank, str(path)) > (current[0], str(current[1])):
            best[key] = (rank, path)

    dropped = len(paths) - len(keep) - len(best)
    if dropped:
        logger.info("revision_policy='latest' superseded %d ICARTT file(s).", dropped)

    selected = sorted(keep + [path for _, path in best.values()])
    _warn_on_repeated_basenames(selected, n_unparsed=len(keep))
    return selected


def _warn_on_repeated_basenames(selected: list[Path], *, n_unparsed: int) -> None:
    """Warn when the selected set holds one filename in several directories.

    The blind spot this covers: revision selection can only compare files
    whose names carry a ``YYYYMMDD`` token, and files without one are kept
    unconditionally — correctly, since an unparseable name is not evidence
    of duplication. But "kept unconditionally" and "kept silently" are
    different things. 147 of the 1122 names in the target archive have no
    date token, and 39 of those basenames exist in two or three directories
    at once (``Miro_Data_0809.ict`` appears under a dated directory, under
    ``Calibrated Data/``, and again under ``Calibrated Data (Updated)/``).

    A recursive template then ingests all three copies of the same day. If
    they are genuinely different products the counts are right; if they are
    successive reprocessings of one day, the campaign silently triple-counts
    that air, and nothing downstream can tell which happened — the files
    look like ordinary distinct inputs by the time they are concatenated.
    Whether it is duplication is a question only the data owner can answer,
    so this reports rather than decides.

    Parameters
    ----------
    selected : list of pathlib.Path
        Files revision selection chose.
    n_unparsed : int
        How many of them had no parseable ICARTT filename, quoted in the
        message so the cause is visible alongside the symptom.
    """
    counts = Counter(path.name for path in selected)
    repeated = sorted(name for name, count in counts.items() if count > 1)
    if not repeated:
        return
    logger.warning(
        "%d basename(s) appear in more than one directory among the %d selected "
        "ICARTT file(s) (e.g. %s). %d selected name(s) carry no YYYYMMDD token and "
        "so cannot be de-duplicated by revision; if any of these are copies of the "
        "same data rather than distinct products, that data will be counted twice.",
        len(repeated),
        len(selected),
        ", ".join(repeated[:3]),
        n_unparsed,
    )


def _read_text(path: Path) -> list[str]:
    """Read a file as text, tolerating non-UTF-8 bytes.

    30 files in the target archive are not valid UTF-8 (stray bytes in PI
    names and comment blocks). Refusing them would lose real data over a
    non-scientific detail, so decoding falls back to latin-1, which cannot
    fail and leaves the numeric content — always ASCII — untouched.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.debug("%s is not valid UTF-8; decoding as latin-1.", path)
        text = raw.decode("latin-1")
    return text.splitlines()


def _split(line: str) -> list[str]:
    """Split an ICARTT header line on commas, trimming whitespace."""
    return [part.strip() for part in line.split(",")]


def _variable_from_line(line: str, scale: float, missing: float) -> IcarttVariable:
    """Build a variable definition from a ``name, units, description`` line."""
    parts = _split(line)
    return IcarttVariable(
        name=parts[0],
        units=parts[1] if len(parts) > 1 else "",
        # Descriptions legitimately contain commas, so rejoin the remainder
        # instead of taking only the third field.
        description=", ".join(parts[2:]) if len(parts) > 2 else "",
        scale=scale,
        missing=missing,
    )


def parse_icartt_header(lines: list[str], path: Path) -> IcarttHeader:
    """Parse an FFI-1001 header block.

    Parameters
    ----------
    lines : list of str
        All lines of the file.
    path : pathlib.Path
        Source file, for error messages.

    Returns
    -------
    IcarttHeader
        Parsed header.

    Raises
    ------
    TsaraIngestError
        If the file is empty, its first line is not ``NLHEAD, FFI``, the
        format is not 1001, or the header is shorter than it declares.
    """
    if not lines:
        raise TsaraIngestError(f"'{path}' is empty.")

    try:
        first = _split(lines[0])
        n_header, ffi = int(first[0]), int(first[1])
    except (ValueError, IndexError) as exc:
        raise TsaraIngestError(
            f"'{path}' does not start with an ICARTT 'NLHEAD, FFI' line (found {lines[0][:60]!r})."
        ) from exc

    if ffi != 1001:
        raise TsaraIngestError(
            f"'{path}' declares file format index {ffi}; TSARA implements FFI-1001 only."
        )
    if len(lines) < n_header:
        raise TsaraIngestError(
            f"'{path}' declares a {n_header}-line header but has only {len(lines)} lines."
        )

    try:
        dates = [int(value) for value in _split(lines[6])[:6]]
        data_date = date(dates[0], dates[1], dates[2])
        revision_date = date(dates[3], dates[4], dates[5])
        interval = float(_split(lines[7])[0])
        n_vars = int(_split(lines[9])[0])
        scales = [float(value) for value in _split(lines[10])[:n_vars]]
        missings = [float(value) for value in _split(lines[11])[:n_vars]]
    except (ValueError, IndexError) as exc:
        raise TsaraIngestError(f"Malformed ICARTT header in '{path}': {exc}") from exc

    # NLHEAD is a self-declaration, and a file can declare something
    # arithmetically impossible. The first twelve lines are fixed by the
    # format and each of the NV dependent variables needs a definition line
    # of its own, so any valid header is at least 12 + NV lines. A file
    # claiming fewer has a wrong NLHEAD, full stop — and the consequence is
    # not cosmetic: trusting it admits header text into the data block,
    # where stray tokens go on to corrupt type inference for the whole file.
    #
    # Two files in the target archive do exactly this (NLHEAD=36, NV=35, so
    # 47 lines are required), and both are structurally identical to 41
    # sibling files that correctly declare 70. Clamping up to 12 + NV does
    # not fully repair them -- their true header is longer still -- but it
    # removes eleven junk lines, and the residue is caught downstream by the
    # modal-width name check and the majority-vote time branch. Raising the
    # floor is a no-op for every other file in the archive.
    minimum_header = 12 + n_vars
    nlhead_understated = n_header < minimum_header
    if n_header < minimum_header:
        logger.warning(
            "%s declares NLHEAD=%d, but NV=%d needs at least %d header lines "
            "(12 fixed + %d variable definitions), so NLHEAD is provably wrong. "
            "Reading data from line %d instead; expect residual header text in "
            "the first rows.",
            path,
            n_header,
            n_vars,
            minimum_header,
            n_vars,
            minimum_header + 1,
        )
        n_header = minimum_header
        if len(lines) < n_header:
            raise TsaraIngestError(
                f"'{path}' needs at least a {n_header}-line header for its "
                f"{n_vars} declared variables but has only {len(lines)} lines."
            )

    independent = _variable_from_line(lines[8], scale=1.0, missing=float("nan"))
    variables = tuple(
        _variable_from_line(lines[12 + i], scale=scales[i], missing=missings[i])
        for i in range(n_vars)
    )

    # Walk the two comment blocks. Their combined length is what makes the
    # header exactly NLHEAD lines, so a mismatch means the file is malformed
    # in a way worth noticing but not worth refusing over.
    index = 12 + n_vars
    special, index = _read_comment_block(lines, index)
    normal, index = _read_comment_block(lines, index)
    if index != n_header:
        # Deliberately debug, not warning. Measured against the target
        # archive this mismatch fires on 44 of 1055 files and is right about
        # none of them: 43 are PTR-MS files carrying one extra, blank-named
        # definition line, which offsets the walk without making the file
        # unreadable. A warning that is a false positive every time it fires
        # trains readers to ignore warnings. The genuinely diagnosable case
        # -- NLHEAD too small to be possible -- is checked above, where the
        # arithmetic proves it rather than merely suggesting it.
        logger.debug(
            "%s: comment blocks end at line %d but NLHEAD is %d; trusting NLHEAD.",
            path,
            index,
            n_header,
        )

    # The data column header is the LAST normal comment line — a genuine
    # quirk of the format. Falling back to the line immediately before the
    # data keeps malformed-comment-count files readable.
    header_line = normal[-1] if normal else lines[n_header - 1]
    column_names = tuple(_split(header_line))

    # Scraped from the WHOLE header rather than from the two comment blocks
    # above, because the blocks are located by arithmetic (12 + NV) that a
    # malformed variable count silently invalidates. Measured: the 43 PTR-MS
    # files in the 2024 archive carry one extra, blank-named definition line,
    # which pushes the walk one line short; both blocks are then read from
    # the wrong place and the metadata comes back EMPTY -- on exactly the
    # files that declare the LOD flags with the highest below-detection
    # fractions in the archive. A key/value scrape has no such dependency:
    # a `KEY: value` line means the same thing wherever in the header it
    # sits, so scanning all of it is strictly more robust and cannot read
    # less. Bounded by n_header so no data row can be mistaken for metadata.
    metadata: dict[str, str] = {}
    for line in _metadata_scan_region(lines, n_header, nlhead_understated):
        match = _METADATA_LINE.match(line.strip())
        if match:
            metadata[match[1]] = match[2].strip()

    return IcarttHeader(
        n_header_lines=n_header,
        file_format_index=ffi,
        pi_name=lines[1].strip(),
        organization=lines[2].strip(),
        data_source=lines[3].strip(),
        mission=lines[4].strip(),
        volume=(int(_split(lines[5])[0]), int(_split(lines[5])[1])),
        data_date=data_date,
        revision_date=revision_date,
        interval=interval,
        independent_variable=independent,
        variables=variables,
        special_comments=special,
        normal_comments=normal,
        column_names=column_names,
        metadata=metadata,
    )


def _read_comment_block(lines: list[str], index: int) -> tuple[tuple[str, ...], int]:
    """Read a count-prefixed comment block, returning it and the next index."""
    try:
        count = int(_split(lines[index])[0])
    except (ValueError, IndexError):
        # A non-numeric count means the header deviates from the spec; treat
        # the block as empty and let NLHEAD govern where data starts.
        return (), index + 1
    start = index + 1
    return tuple(lines[start : start + count]), start + count


@register_reader("icartt")
def read_icartt(path: Path, loader: LoaderConfig, /) -> RawTable:
    """Read one ICARTT FFI-1001 file into a :class:`RawTable`.

    Parameters
    ----------
    path : pathlib.Path
        File to read.
    loader : LoaderConfig
        Must be an :class:`~tsara.config.manifest.ICARTTLoader`.

    Returns
    -------
    RawTable
        Columns under their ICARTT short names, scaled and missing-masked,
        indexed by tz-naive UTC time. ``attrs`` carries the header
        provenance.

    Raises
    ------
    TsaraIngestError
        If the loader is the wrong type, the header is malformed, the file
        holds no data rows, or no timestamp could be constructed.
    """
    if not isinstance(loader, ICARTTLoader):
        raise TsaraIngestError(
            f"The 'icartt' reader received a {type(loader).__name__}. This means "
            "a reader was registered under the wrong format name."
        )

    lines = _read_text(path)
    header = parse_icartt_header(lines, path)

    frame = _read_data(lines, header, path, loader)
    frame, lod_counts = _apply_scales_and_missing(frame, header)
    if lod_counts:
        # Logged, because a species that is mostly below detection is a fact
        # about the campaign worth knowing before a ratio is fitted to it,
        # not a quiet detail of the parse.
        logger.info(
            "%s: masked out-of-detection-range samples: %s.",
            path,
            ", ".join(f"{name}:{count}" for name, count in sorted(lod_counts.items())),
        )

    times = _build_time_index(frame, header, path, loader.max_dropped_fraction)
    frame = frame.set_axis(times, axis=0)
    return RawTable(frame=frame, path=path, attrs=_provenance(header, lod_counts))


def _modal_field_count(body: list[str]) -> int:
    """Most common comma-separated field count among the data lines.

    This is the one statement about a file's shape that comes from the data
    rather than from the header's claims about the data, which is exactly
    why it is worth computing: when the two name lists disagree, the rows
    themselves are the tie-breaker. The *mode* rather than the maximum or
    the first row's width, because real archives contain truncated lines
    (one file here loses 14 rows of 84,362 to a logger interrupted
    mid-number) and a header whose text has leaked into the data block.
    """
    widths = Counter(len(line.split(",")) for line in body if line.strip())
    # Counter.most_common breaks ties by first insertion, i.e. by first
    # appearance in the file — deterministic, which is all that is needed.
    return widths.most_common(1)[0][0] if widths else 0


def _mangle_duplicates(names: list[str]) -> list[str]:
    """Disambiguate repeated names the way ``pandas.read_csv`` does.

    A duplicate is not merely untidy here: :func:`~tsara.ingest.base.check_raw_table`
    rejects a frame with duplicate columns (selection would be ambiguous),
    and ``pandas.read_csv`` refuses a duplicated ``names=`` list outright
    with an untyped ``ValueError`` — which would escape a reader whose
    contract promises :class:`~tsara.ingest.base.TsaraIngestError`. Renaming
    the later occurrences keeps the data readable and the failure typed.
    """
    seen: Counter[str] = Counter()
    out: list[str] = []
    for name in names:
        count = seen[name]
        seen[name] += 1
        out.append(name if count == 0 else f"{name}.{count}")
    return out


def _choose_column_names(body: list[str], header: IcarttHeader, path: Path) -> list[str]:
    """Pick the column-name list that matches the data's actual width.

    An FFI-1001 file states its column names twice, and the two statements
    disagree often enough to need a rule. The **declared** names are the last
    normal comment line, which the format designates as the data column
    header; the **definitions** are the NV variable-definition lines.

    The old rule was "use the declared names, unless their count disagrees
    with NV, in which case NV is authoritative". Measured against the 1055
    ICARTT files in the target archive that is right 1054 times and wrong
    once — and the once is a hard failure rather than a degradation. One file
    declares NV=1 with its independent *and* its single dependent variable
    both named ``Time_UTC``; the count check sees 2 names against 7-field
    rows, discards the perfectly good 7-name column-header line, and hands
    pandas a duplicated ``names=`` list, which it refuses with an untyped
    ``ValueError`` — escaping a reader contracted to raise
    :class:`~tsara.ingest.base.TsaraIngestError`.

    So the arbiter becomes the data rather than NV: prefer whichever list is
    as wide as the rows actually are. Measurement is what makes this safe to
    change. On the 1011 files where *both* lists match the row width their
    contents are byte-identical, so preferring the declared names on a tie
    changes nothing anywhere in the archive while keeping the format's own
    designated statement of the column names authoritative. 43 files match on
    definitions alone, 1 on the declared header alone, and **none** match on
    neither.

    When neither list matches, the width disagreement is about the *rows*,
    not the names — the archive's uniformly-too-wide files, where every row
    carries surplus trailing fields that ``index_col=False`` correctly
    discards. That case must not be refused, so it falls through to the
    original NV-based rule and is left to the reader's existing protections:
    surplus fields are dropped by pandas, and wholesale row loss is caught by
    ``max_dropped_fraction``.

    Parameters
    ----------
    body : list of str
        Data lines below the header.
    header : IcarttHeader
        Parsed header supplying both candidate name lists.
    path : pathlib.Path
        Source file, for messages.

    Returns
    -------
    list of str
        Column names, with any duplicates mangled.
    """
    definitions = [header.independent_variable.name] + [v.name for v in header.variables]
    declared = list(header.column_names)
    modal = _modal_field_count(body)

    if len(declared) == modal:
        # Covers both the ordinary case and the tie; see the docstring for
        # why a tie is provably content-neutral on the target archive.
        return _mangle_duplicates(declared)

    if len(definitions) == modal:
        logger.warning(
            "%s: the column-header line lists %d names but the data rows have %d "
            "fields, which the %d variable definitions match; using the variable "
            "definitions.",
            path,
            len(declared),
            modal,
            len(definitions),
        )
        return _mangle_duplicates(definitions)

    # Neither list is as wide as the rows. The names are not the problem here,
    # so fall back to the original rule and let the reader's row-level
    # protections handle the width mismatch.
    expected = 1 + len(header.variables)
    if len(declared) == expected:
        return _mangle_duplicates(declared)
    logger.warning(
        "%s: column header lists %d names but the header declares %d variables, "
        "and the data rows have %d fields matching neither; using the variable "
        "definitions.",
        path,
        len(declared),
        expected,
        modal,
    )
    return _mangle_duplicates(definitions)


def _read_data(
    lines: list[str], header: IcarttHeader, path: Path, loader: ICARTTLoader
) -> pd.DataFrame:
    """Parse the data block below the header into a DataFrame."""
    body = lines[header.n_header_lines :]
    if not any(line.strip() for line in body):
        raise TsaraIngestError(f"'{path}' has a valid header but no data rows.")

    names = _choose_column_names(body, header, path)

    from io import StringIO

    # Counted before parsing so that rows pandas discards can be reported. A
    # ragged line is a logging glitch, not a format difference: one file in
    # the target archive truncates 14 lines out of 84,362 (0.017%) where the
    # logger was interrupted mid-number. Refusing the file would throw away a
    # whole day of measurements to avoid 14 bad rows, so bad lines are skipped
    # and *counted* — silent skipping would be the genuinely dangerous option.
    n_nonblank = sum(1 for line in body if line.strip())

    read_kwargs: dict[str, Any] = {
        "header": None,
        "names": names,
        "skipinitialspace": True,
        "skip_blank_lines": True,
        "on_bad_lines": "skip",
        # Values stay as written; scaling and missing-masking happen next, in
        # that order, so a sentinel is never scaled.
        "dtype": None,
        # Parse each column in one pass rather than in chunks. Chunked
        # inference makes a column's dtype depend on where a truncated row
        # happens to fall, which pandas reports as a DtypeWarning and which
        # would make the result depend on file size.
        "low_memory": False,
        # Load-bearing, not cosmetic. Without it, a file whose rows are ALL
        # wider than the declared names makes pandas decide the surplus
        # leading fields are an index: it builds a MultiIndex and every value
        # silently shifts left, so the first declared column receives the
        # third field. index_col=False forces the columns to align with the
        # names as declared.
        "index_col": False,
        **float_precision_kwarg(loader),
    }
    # `**kwargs` erases the return type, so restore it rather than letting
    # Any leak into every caller.
    frame = cast("pd.DataFrame", pd.read_csv(StringIO("\n".join(body)), **read_kwargs))

    n_skipped = n_nonblank - len(frame)
    if n_skipped > 0:
        check_dropped_rows(
            n_dropped=n_skipped,
            n_total=n_nonblank,
            path=path,
            reason="malformed data row (wrong field count)",
            max_fraction=loader.max_dropped_fraction,
            logger=logger,
        )
    return frame


#: How far past a provably-wrong NLHEAD to keep looking for header text.
#: Generous enough to clear any real comment block (the longest in the 2024
#: archive is 23 lines) and small enough that a file with no residue costs
#: one failed match per line rather than a scan of the whole record.
_RESIDUAL_HEADER_LIMIT = 200


def _metadata_scan_region(lines: list[str], n_header: int, nlhead_understated: bool) -> list[str]:
    """Return the lines to scrape ``KEY: value`` metadata from.

    Normally the header, and nothing else. The exception is a file whose
    ``NLHEAD`` is provably too small: clamping it up to ``12 + NV`` recovers
    a *lower bound* on the header, not the header, so the comment block —
    LOD flags included — can still sit past the bound, in lines the reader
    otherwise treats as data.

    Two files in the 2024 archive are exactly this shape, and between them
    they were the last 19,398 unmasked LOD sentinels once the block-walk
    scrape was fixed.

    The window is scanned to its end rather than stopped at the first
    non-metadata line, because the residue does not begin with metadata: it
    opens with the leftover variable definitions and comment counts that the
    clamp could not account for, so any early stop would halt before
    reaching the ``KEY: value`` block. Scanning into data rows is harmless —
    :data:`_METADATA_LINE` is anchored on an uppercase key followed by a
    colon, and an ICARTT data row begins with a numeric time field — so the
    cost of over-scanning is a few failed matches, while the cost of
    under-scanning is an unmasked sentinel in a concentration.
    """
    if not nlhead_understated:
        return lines[:n_header]
    return list(lines[: n_header + _RESIDUAL_HEADER_LIMIT])


def _lod_sentinels(header: IcarttHeader) -> dict[str, tuple[float, ...]]:
    """Map each dependent variable to the LOD sentinels that apply to it.

    ICARTT marks out-of-detection-range samples with sentinels that are
    *not* the ``VMISS`` missing value: ``LLOD_FLAG`` for below the lower
    limit of detection and ``ULOD_FLAG`` for above the upper one. They are
    declared in the special-comment block rather than on a fixed header
    line, so they need parsing rather than indexing.

    Three shapes occur in real archives, and all three are handled:

    * a single value applying to every variable (``LLOD_FLAG: -8888``);
    * one value per dependent variable (a 16-item comma-separated list);
    * a non-numeric placeholder (``N/A``, ``NaN``) meaning "not used".

    When the item count matches ``NV`` the mapping is per-variable;
    otherwise the union of the declared values applies to all of them.
    Taking the union is the conservative reading: these sentinels are
    chosen precisely to be impossible measurements, so masking one that a
    given variable never uses costs nothing, while failing to mask one that
    it does use puts a large negative number into a concentration.
    """
    declared: list[tuple[float, ...]] = []
    for key in ("LLOD_FLAG", "ULOD_FLAG"):
        raw = header.metadata.get(key)
        if raw is None:
            continue
        values: list[float] = []
        for token in raw.split(","):
            try:
                value = float(token.strip())
            except ValueError:
                # 'N/A' and 'NaN' both mean the flag is unused. NaN is
                # excluded deliberately as well as accidentally: it can
                # never compare equal, so it could not mask anything.
                continue
            if not np.isnan(value):
                values.append(value)
        if values:
            declared.append(tuple(values))

    if not declared:
        return {}

    names = [variable.name for variable in header.variables]
    per_variable: dict[str, list[float]] = {name: [] for name in names}
    for sentinels in declared:
        if len(sentinels) == len(names):
            for name, value in zip(names, sentinels, strict=True):
                per_variable[name].append(value)
        else:
            for name in names:
                per_variable[name].extend(sentinels)
    return {name: tuple(dict.fromkeys(values)) for name, values in per_variable.items() if values}


def _apply_scales_and_missing(
    frame: pd.DataFrame, header: IcarttHeader
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask declared sentinels, then apply scale factors.

    Order matters and is not interchangeable: ``VSCAL`` applies to real
    measurements only. Scaling first would turn a ``-9999`` sentinel into
    ``-9999 * scale``, which no longer equals the declared sentinel and
    would silently enter the data as a plausible-looking number.

    Both kinds of sentinel are masked here — ``VMISS`` for missing data and
    the ``LLOD_FLAG``/``ULOD_FLAG`` values for out-of-detection-range
    samples. Leaving the latter in place was measured against the 2024
    archive as the more dangerous option by a wide margin: 10.03% of every
    numeric value there is an LOD sentinel, and for the PTR-MS VOCs — the
    species this package exists to fingerprint sources with — it reaches
    67-70% of samples, far enough past half that the *median* of a real
    benzene record is the sentinel rather than a concentration. A rolling
    low-quantile baseline would therefore have called ``-88888 ppbv`` the
    background (``docs/METHODS.md`` §9.2.1).

    Masking is not the same as discarding the information: the per-variable
    counts returned here, and the flag values themselves, travel on into the
    stream so a later phase can substitute LOD/2 or fit a censored model.

    Returns
    -------
    frame : pandas.DataFrame
        The frame, sentinel-masked and scaled.
    lod_counts : dict of str to int
        Number of samples masked as out-of-detection-range, per variable.
        Only variables with at least one such sample appear.
    """
    lod_by_variable = _lod_sentinels(header)
    lod_counts: dict[str, int] = {}

    for variable in header.variables:
        if variable.name not in frame.columns:
            continue
        column = pd.to_numeric(frame[variable.name], errors="coerce")
        if not np.isnan(variable.missing):
            # Compared with a tolerance because sentinels are written in
            # several spellings of the same number (-9999, -9999.0, -9.999e50).
            column = column.mask(np.isclose(column, variable.missing, rtol=1e-9, atol=0.0))
        for sentinel in lod_by_variable.get(variable.name, ()):
            flagged = np.isclose(column, sentinel, rtol=1e-9, atol=0.0)
            n_flagged = int(flagged.sum())
            if n_flagged:
                lod_counts[variable.name] = lod_counts.get(variable.name, 0) + n_flagged
                column = column.mask(flagged)
        if variable.scale != 1.0:
            column = column * variable.scale
        frame[variable.name] = column
    return frame, lod_counts


def _build_time_index(
    frame: pd.DataFrame, header: IcarttHeader, path: Path, max_dropped_fraction: float
) -> pd.DatetimeIndex:
    """Build the time index from the independent variable column.

    Keys off the *values*, not the declared name or units, because the
    archive uses at least a dozen spellings for "seconds past midnight" and
    43 files declare that and then write datetime strings instead. Trying
    numeric first and falling back to datetime parsing is correct for every
    variant observed, and for spellings not yet seen.
    """
    name = header.independent_variable.name
    if name not in frame.columns:
        raise TsaraIngestError(
            f"'{path}' declares independent variable '{name}' but the data has "
            f"columns {list(frame.columns)[:8]}."
        )

    raw = frame[name]
    seconds = pd.to_numeric(raw, errors="coerce")
    n_numeric = int(seconds.notna().sum())
    n_rows = len(raw)
    parsed: pd.Series | None = None

    if n_numeric == n_rows:
        # Unambiguous: every value is a number. No datetime parse needed.
        use_seconds = True
    elif n_numeric == 0:
        # Unambiguous the other way: nothing is numeric.
        use_seconds = False
    else:
        # Genuinely mixed, so count both interpretations and take the
        # majority. The rule this replaces asked only whether *any* value
        # was numeric, which let a single stray token decide the file: two
        # PTR-MS VOC files in the target archive hold 10,201 and 7,817
        # datetime strings alongside exactly 2 numeric tokens leaked in from
        # a mis-declared header block, and the old test sent both down the
        # seconds-past-midnight branch. Every datetime row then failed to
        # convert and was dropped, reducing 10,235 rows to 2 -- 35 VOC
        # species, the archive's most valuable data, cut to nothing behind a
        # warning. Only mixed columns pay for the second parse; the two
        # unambiguous cases above short-circuit before reaching here.
        parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
        n_datetime = int(np.asarray(pd.notna(parsed)).sum())
        # Ties favour seconds, keeping the spec-compliant reading as the
        # default when the evidence does not actually distinguish them.
        use_seconds = n_numeric >= n_datetime
        logger.warning(
            "%s: independent variable '%s' is mixed — %d of %d values parse as "
            "numeric seconds and %d as timestamps. Reading it as %s (majority).",
            path,
            name,
            n_numeric,
            n_rows,
            n_datetime,
            "seconds past midnight" if use_seconds else "timestamps",
        )

    if use_seconds:
        # The spec-compliant path: seconds elapsed since midnight UTC on the
        # header's data date. Values may exceed 86400 for records crossing
        # midnight, which Timedelta handles without special-casing.
        base = pd.Timestamp(header.data_date)
        times = pd.DatetimeIndex(base + pd.to_timedelta(seconds, unit="s"))
    else:
        if parsed is None:
            parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
        times = pd.DatetimeIndex(parsed)
        if not bool(np.asarray(times.notna()).any()):
            raise TsaraIngestError(
                f"'{path}': independent variable '{name}' is neither numeric "
                "seconds nor parseable timestamps."
            )

    times = to_utc_naive_ns(times, "UTC", path)
    # As an array, not an Index: pandas-stubs types `.notna()` as Index[bool],
    # which supports neither `~` nor `.any()`.
    valid = np.asarray(times.notna())
    n_bad = int((~valid).sum())
    if n_bad:
        check_dropped_rows(
            n_dropped=n_bad,
            n_total=len(times),
            path=path,
            reason="timestamp did not parse",
            max_fraction=max_dropped_fraction,
            logger=logger,
        )
        frame.drop(frame.index[~valid], inplace=True)
        times = times[valid]
    return times


def _provenance(header: IcarttHeader, lod_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Extract the header fields worth carrying alongside the data.

    Limit-of-detection flags are included deliberately. ICARTT files mark
    out-of-detection-range samples with sentinels distinct from ``VMISS``,
    which is scientifically *not* the same as missing data — a below-LOD
    benzene is an upper bound, whereas a dropout is no information at all.
    :func:`_apply_scales_and_missing` masks those samples so they cannot
    reach a baseline as numbers, and the flags and per-variable counts are
    carried here so a later stage can still tell the two apart and
    substitute LOD/2 or fit a censored model without re-reading every file.
    """
    provenance: dict[str, Any] = {
        "icartt_pi": header.pi_name,
        "icartt_organization": header.organization,
        "icartt_data_source": header.data_source,
        "icartt_mission": header.mission,
        "icartt_data_date": header.data_date.isoformat(),
        "icartt_revision_date": header.revision_date.isoformat(),
        "icartt_interval": header.interval,
    }
    for key in ("REVISION", "PLATFORM", "LOCATION", "ULOD_FLAG", "LLOD_FLAG", "LLOD_VALUE"):
        if key in header.metadata:
            provenance[f"icartt_{key.lower()}"] = header.metadata[key]
    if lod_counts:
        # Under the raw column names, because that is the only vocabulary a
        # reader has: the manifest's canonical names are not visible here.
        # Stream assembly translates them when it knows the mapping.
        provenance["icartt_lod_masked"] = dict(lod_counts)
    return provenance
