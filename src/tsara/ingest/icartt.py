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
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tsara.config.manifest import ICARTTLoader
from tsara.ingest.base import RawTable, TsaraIngestError, to_utc_naive_ns
from tsara.ingest.registry import register_reader

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
#: ``KEY: value``. Harvested into header metadata so downstream stages can
#: see limit-of-detection flags and provenance without re-reading the file.
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
        ``KEY: value`` pairs harvested from the comment blocks (e.g.
        ``ULOD_FLAG``, ``LLOD_FLAG``, ``PLATFORM``, ``REVISION``).
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
    return sorted(keep + [path for _, path in best.values()])


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
        logger.warning(
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

    metadata: dict[str, str] = {}
    for line in (*special, *normal):
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

    frame = _read_data(lines, header, path)
    frame = _apply_scales_and_missing(frame, header)

    times = _build_time_index(frame, header, path)
    frame = frame.set_axis(times, axis=0)
    return RawTable(frame=frame, path=path, attrs=_provenance(header))


def _read_data(lines: list[str], header: IcarttHeader, path: Path) -> pd.DataFrame:
    """Parse the data block below the header into a DataFrame."""
    body = lines[header.n_header_lines :]
    if not any(line.strip() for line in body):
        raise TsaraIngestError(f"'{path}' has a valid header but no data rows.")

    names: list[str] = list(header.column_names)
    expected = 1 + len(header.variables)
    if len(names) != expected:
        # A column-header line that disagrees with NV is common enough in
        # real archives to handle rather than refuse: the authoritative names
        # are the variable definitions, which NV guarantees the count of.
        logger.warning(
            "%s: column header lists %d names but the header declares %d variables; "
            "using the variable definitions.",
            path,
            len(names),
            expected,
        )
        names = [header.independent_variable.name] + [v.name for v in header.variables]

    from io import StringIO

    # Counted before parsing so that rows pandas discards can be reported. A
    # ragged line is a logging glitch, not a format difference: one file in
    # the target archive truncates 14 lines out of 84,362 (0.017%) where the
    # logger was interrupted mid-number. Refusing the file would throw away a
    # whole day of measurements to avoid 14 bad rows, so bad lines are skipped
    # and *counted* — silent skipping would be the genuinely dangerous option.
    n_nonblank = sum(1 for line in body if line.strip())

    frame = pd.read_csv(
        StringIO("\n".join(body)),
        header=None,
        names=names,
        skipinitialspace=True,
        skip_blank_lines=True,
        on_bad_lines="skip",
        # Values stay as written; scaling and missing-masking happen next, in
        # that order, so a sentinel is never scaled.
        dtype=None,
        # Parse each column in one pass rather than in chunks. Chunked
        # inference makes a column's dtype depend on where a truncated row
        # happens to fall, which pandas reports as a DtypeWarning and which
        # would make the result depend on file size.
        low_memory=False,
        # Load-bearing, not cosmetic. Without it, a file whose rows are ALL
        # wider than the declared names makes pandas decide the surplus
        # leading fields are an index: it builds a MultiIndex and every value
        # silently shifts left, so the first declared column receives the
        # third field. index_col=False forces the columns to align with the
        # names as declared.
        index_col=False,
    )

    n_skipped = n_nonblank - len(frame)
    if n_skipped > 0:
        logger.warning(
            "Skipped %d of %d malformed data row(s) in %s (wrong field count).",
            n_skipped,
            n_nonblank,
            path,
        )
    return frame


def _apply_scales_and_missing(frame: pd.DataFrame, header: IcarttHeader) -> pd.DataFrame:
    """Mask declared missing-value sentinels, then apply scale factors.

    Order matters and is not interchangeable: ``VSCAL`` applies to real
    measurements only. Scaling first would turn a ``-9999`` sentinel into
    ``-9999 * scale``, which no longer equals the declared sentinel and
    would silently enter the data as a plausible-looking number.
    """
    for variable in header.variables:
        if variable.name not in frame.columns:
            continue
        column = pd.to_numeric(frame[variable.name], errors="coerce")
        if not np.isnan(variable.missing):
            # Compared with a tolerance because sentinels are written in
            # several spellings of the same number (-9999, -9999.0, -9.999e50).
            column = column.mask(np.isclose(column, variable.missing, rtol=1e-9, atol=0.0))
        if variable.scale != 1.0:
            column = column * variable.scale
        frame[variable.name] = column
    return frame


def _build_time_index(frame: pd.DataFrame, header: IcarttHeader, path: Path) -> pd.DatetimeIndex:
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
    if bool(seconds.notna().any()):
        # The spec-compliant path: seconds elapsed since midnight UTC on the
        # header's data date. Values may exceed 86400 for records crossing
        # midnight, which Timedelta handles without special-casing.
        base = pd.Timestamp(header.data_date)
        times = pd.DatetimeIndex(base + pd.to_timedelta(seconds, unit="s"))
    else:
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
        logger.warning(
            "Dropped %d of %d rows from %s: timestamp did not parse.", n_bad, len(times), path
        )
        frame.drop(frame.index[~valid], inplace=True)
        times = times[valid]
    return times


def _provenance(header: IcarttHeader) -> dict[str, Any]:
    """Extract the header fields worth carrying alongside the data.

    Limit-of-detection flags are included deliberately. ICARTT files mark
    below-detection samples with a distinct sentinel (``LLOD_FLAG``), which
    is scientifically *not* the same as missing data — it is an upper bound.
    TSARA does not yet act on that distinction, so the flags are carried
    forward here rather than discarded, leaving a later stage free to
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
    return provenance
