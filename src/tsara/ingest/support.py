"""Deciding what interval of air each ingested row describes.

The ingestion counterpart of :mod:`tsara.ingest.uncertainty`, and shaped the
same way: a manifest says what it can, the file says what it can, TSARA
measures the rest, and every answer is labelled with where it came from. No
rung of that ladder is allowed to masquerade as a stronger one.

Three facts, established independently
---------------------------------------
A cell needs a **width**, a **label** saying where the timestamp sits inside
it, and a **method** saying whether the value is an average over it or a
sample within it. They come from different places, so their provenance is
recorded separately -- a stationary analyzer whose file carries a stop column
and whose manifest declares ``method: mean`` is honestly reported / reported
/ declared, and one label per stream could not say that.

What is inferred, and what deliberately is not
-----------------------------------------------
**Width** is inferred from each file's own median sampling interval when the
manifest does not declare one. Measured across 303 files of the target
archive, 99.2% of sampling intervals are jitter around a single nominal
cadence and 0.6% are dropped rows, so one width per file describes the data;
a dropped row leaves a hole in the tiling rather than a wider cell. Per
*file* rather than per instrument, because some met records really do run at
1 s in one file and 5 s in another.

**Label** is inferred only from evidence a reader hands up -- for ICARTT,
the name of the independent variable, since the specification says it is a
start time and the archive mostly agrees. A name that says nothing yields
``unknown``, which is treated as centred: the choice that minimises the
worst-case misplacement rather than the one that sounds most standard.

**Method is never inferred.** The only evidence real files carry is English
prose in a header, and a regular expression over prose is not a basis for
deciding whether a number may be treated as an integral. Where a header does
mention averaging it is surfaced as provenance a human can read, not acted
on. Undeclared means ``point``, marked ``assumed``, which claims nothing.

**Boundary columns are never guessed.** A file naming ``Time_Stop`` almost
certainly means it, but "almost certainly" applied to the wrong column
silently produces wrong cells for a whole campaign -- exactly the failure
this phase exists to prevent. The manifest names them; candidates found in
the file are reported as evidence so a user can see what is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.naming import RAW_TIME_START_COLUMN, RAW_TIME_STOP_COLUMN
from tsara.core.support import CellBounds
from tsara.core.timebase import epoch_ns
from tsara.ingest.base import TsaraIngestError, check_dropped_rows

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt

    from tsara.config.manifest import SupportSpec
    from tsara.core.naming import SupportLabel, SupportMethod, SupportSource

logger = logging.getLogger(__name__)

__all__ = [
    "CANDIDATE_COLUMNS_KEY",
    "LABEL_HINT_KEY",
    "ResolvedSupport",
    "attach_declared_boundaries",
    "resolve_support",
    "shift_and_centre",
]

#: ``RawTable.attrs`` key by which a reader hands up a label it can justify.
#:
#: A hint, not a decision: the resolver still lets a manifest override it, and
#: records that the answer was inferred rather than declared. Kept as an attr
#: rather than a new field on ``RawTable`` so that adding evidence never
#: changes the reader contract.
LABEL_HINT_KEY = "tsara_support_label_hint"

#: ``RawTable.attrs`` key listing columns that *look* like cell boundaries.
#:
#: Evidence for a human, never acted on. Seeing "this file has a column called
#: iWAS_Stop_UTC" in a stream's provenance is what tells someone their
#: manifest could be naming it.
CANDIDATE_COLUMNS_KEY = "tsara_boundary_column_candidates"


@dataclass(frozen=True)
class ResolvedSupport:
    """What TSARA concluded about one stream's cells, and on what basis.

    Attributes
    ----------
    label : {'start', 'mid', 'end', 'unknown'}
        Where the file's timestamp sits in its cell.
    method : {'point', 'mean'}
        Whether values are averages over their cells or samples within them.
    width_ns : int or None
        Nominal cell width in nanoseconds, or None when widths vary per row
        and no single nominal value applies.
    label_source, width_source, method_source : str
        Where each of the three came from.
    n_widened : int
        How many cells the file declared with no duration at all, and which
        were widened to the record's cadence so they could carry weight.
    """

    label: SupportLabel
    method: SupportMethod
    width_ns: int | None
    label_source: SupportSource
    width_source: SupportSource
    method_source: SupportSource
    n_widened: int = 0


def attach_declared_boundaries(
    frame: pd.DataFrame,
    support: SupportSpec,
    *,
    parse: Callable[[str], pd.DatetimeIndex],
    path: Path,
    max_dropped_fraction: float,
    reader_logger: logging.Logger,
) -> pd.DataFrame:
    """Read per-row cell boundaries the manifest says the file carries.

    Called by each reader, which supplies ``parse`` because turning a column
    into timestamps is the one part of this that is format-specific: a CSV
    column holds text in the loader's declared format, while an ICARTT column
    holds seconds past the header's date. Everything else -- which columns,
    what to do with a row that will not parse, what the result is called --
    is the same for every format and lives here.

    Parameters
    ----------
    frame : pandas.DataFrame
        The file's rows, already time-indexed.
    support : SupportSpec
        The manifest's declaration. If it names no ``stop_column`` the frame
        is returned unchanged.
    parse : callable
        Column name to timestamps, supplied by the reader.
    path : pathlib.Path
        File being read, for messages.
    max_dropped_fraction : float
        How much row loss is a misparse rather than a bad day.
    reader_logger : logging.Logger
        The calling reader's logger, so warnings are attributed to the module
        that read the file.

    Returns
    -------
    pandas.DataFrame
        The frame, with the reserved boundary columns added when declared.

    Raises
    ------
    TsaraIngestError
        If a declared boundary column is not in the file.
    """
    if support.stop_column is None:
        return frame

    columns = {RAW_TIME_STOP_COLUMN: support.stop_column}
    if support.start_column is not None:
        columns[RAW_TIME_START_COLUMN] = support.start_column

    parsed: dict[str, pd.DatetimeIndex] = {}
    for reserved, declared in columns.items():
        if declared not in frame.columns:
            # Raised rather than shrugged off: a missing boundary column means
            # the manifest and the archive disagree about what this file is,
            # and silently falling back to an assumed cell would bury that.
            raise TsaraIngestError(
                f"'{path}' has no column '{declared}', declared as a cell "
                f"boundary. Columns present: {list(frame.columns)[:12]}."
            )
        parsed[reserved] = parse(declared)

    if RAW_TIME_START_COLUMN not in parsed:
        # No start column given, so the file's own time axis is the cell
        # start. That is the ICARTT convention and the common case.
        parsed[RAW_TIME_START_COLUMN] = pd.DatetimeIndex(frame.index)

    out = frame.copy()
    for reserved, values in parsed.items():
        out[reserved] = np.asarray(values, dtype="datetime64[ns]")

    valid = np.asarray(out[RAW_TIME_START_COLUMN].notna() & out[RAW_TIME_STOP_COLUMN].notna())
    n_bad = int((~valid).sum())
    if n_bad:
        check_dropped_rows(
            n_dropped=n_bad,
            n_total=len(out),
            path=path,
            reason="cell boundary did not parse",
            max_fraction=max_dropped_fraction,
            logger=reader_logger,
        )
        out = out.loc[valid]
    return out


def _floor_declared_cells(
    frame: pd.DataFrame,
    start_ns: npt.NDArray[np.int64],
    stop_ns: npt.NDArray[np.int64],
    *,
    widths_ns: npt.NDArray[np.int64] | None,
    path: Path,
) -> tuple[pd.DataFrame, npt.NDArray[np.int64], npt.NDArray[np.int64], int]:
    """Give a declared cell of zero duration the record's own cadence instead.

    Real files do this. One airborne spectrometer in the target archive
    declares stop equal to start on 644 of its 90,673 rows, 0.71 %, and a
    cell of zero duration has zero measure and therefore zero weight in every
    overlap. Left alone, those rows would sit in the stream looking exactly
    like data and never contribute to a single paired regression point, with
    nothing anywhere saying so — which is the failure this whole phase exists
    to prevent.

    Widening is the repair that keeps the measurement. The alternative,
    refusing the file, would cost a whole instrument to save 0.71 % of it.
    Only *zero* widths are repaired, and they are repaired to the median of
    the file's own other cells rather than to the spacing between cells. A
    cell narrower than that spacing is not a defect but the definition of a
    duty-cycled instrument, and a sampler that fills for 15 s every 10
    minutes must not have one degenerate row inflated to 10 minutes.
    The cells are re-centred on their original midpoints, so nothing moves
    except the width, and the count is carried out so the stream can record
    that a repair took place.

    Returns
    -------
    pandas.DataFrame
        The frame, with the reserved boundary columns updated if anything
        was widened.
    numpy.ndarray, numpy.ndarray
        The (possibly widened) starts and stops.
    int
        How many cells were widened.
    """
    if widths_ns is None:
        # Nothing to widen them to. A record with no measurable cadence
        # cannot supply a floor, and inventing one would be a fabrication.
        return frame, start_ns, stop_ns, 0
    # ONLY cells of exactly zero duration are touched, and the minimum for
    # every other row is its own existing width. Flooring at the cadence
    # instead would be a serious error rather than a conservative one: a
    # duty-cycled sampler is *supposed* to have cells narrower than its
    # spacing, so it would inflate a 15 s canister fill to the several
    # minutes between canisters and quietly claim the instrument had been
    # sampling the whole time.
    current = stop_ns - start_ns
    positive = current[current > 0]
    # The file's own other cells are the best available answer for how long a
    # cell of this instrument lasts -- far better than the spacing between
    # them, which for a duty-cycled sampler is a different quantity entirely.
    # The cadence is the fallback for the degenerate file whose cells are
    # *all* zero-width, where there is nothing else to go on.
    replacement = (
        np.full(current.shape, int(np.median(positive)), dtype=np.int64)
        if positive.size
        else np.asarray(widths_ns, dtype=np.int64)
    )
    minimum = np.where(current == 0, replacement, current)
    bounds, n_widened = CellBounds(start_ns=start_ns, stop_ns=stop_ns).floor_width(minimum)
    if n_widened == 0:
        return frame, start_ns, stop_ns, 0
    logger.warning(
        "%s: %d cell(s) were declared with zero duration and have been widened "
        "to the record's own cadence, centred where they were. A zero-width "
        "cell carries no weight in any overlap, so it would otherwise have "
        "been silently absent from every pairing.",
        path,
        n_widened,
    )
    out = frame.copy()
    out[RAW_TIME_START_COLUMN] = bounds.start_ns.astype("datetime64[ns]")
    out[RAW_TIME_STOP_COLUMN] = bounds.stop_ns.astype("datetime64[ns]")
    return out, bounds.start_ns, bounds.stop_ns, n_widened


def _warn_if_cells_would_overlap(
    declared_width_ns: int, widths_ns: npt.NDArray[np.int64] | None, *, path: Path
) -> None:
    """Say so when a declared cell is wider than the gap between samples.

    The schema cannot catch this: a manifest is validated before any file is
    read, so the sampling interval is unknown to it. By the time support is
    resolved both numbers are in hand, and the consequence of getting it
    wrong is not subtle. Measured on a 60 s record declared as 120 s cells:
    every adjacent pair overlaps, the record reports 182 % coverage of
    itself, and binning anything onto those cells counts each sample twice.

    A warning rather than a refusal, because a cell wider than the spacing is
    physically possible -- overlapping integrations exist, even if none appear
    anywhere in the target archive -- and refusing would block an archive
    TSARA has merely misjudged. The generator refuses the same configuration
    outright, but it is manufacturing the data and so cannot be wrong about
    it.
    """
    if widths_ns is None or widths_ns.size == 0:
        return
    cadence = int(np.median(widths_ns))
    if declared_width_ns <= cadence:
        return
    logger.warning(
        "%s: the manifest declares %s cells, wider than the %s measured "
        "between samples, so cells will overlap and every sample will be "
        "counted into more than one of them. Check the declared width, or the "
        "path templates if two products have been mixed into one instrument.",
        path,
        pd.Timedelta(declared_width_ns, unit="ns"),
        pd.Timedelta(cadence, unit="ns"),
    )


def _label_from_bounds(
    index_ns: npt.NDArray[np.int64],
    start_ns: npt.NDArray[np.int64],
    stop_ns: npt.NDArray[np.int64],
) -> SupportLabel:
    """Work out where a file puts its timestamp, given the cells it declared.

    Derived rather than asked for, because a file that states every boundary
    has already answered the question: the label is wherever its index
    actually falls. This is what makes an independent variable named
    ``Time_Mid`` need no special handling.
    """
    if np.array_equal(index_ns, start_ns):
        return "start"
    if np.array_equal(index_ns, stop_ns):
        return "end"
    if np.array_equal(index_ns, start_ns + (stop_ns - start_ns) // 2):
        return "mid"
    return "unknown"


def resolve_support(
    frame: pd.DataFrame,
    support: SupportSpec,
    *,
    widths_ns: npt.NDArray[np.int64] | None,
    label_hint: SupportLabel | None,
    path: Path,
) -> tuple[pd.DataFrame, ResolvedSupport]:
    """Give every row a cell, and say where each part of that came from.

    Parameters
    ----------
    frame : pandas.DataFrame
        One instrument's rows, concatenated across its files and still in
        file order. Boundary columns are present if the manifest declared
        them and the readers found them.
    support : SupportSpec
        The manifest's declaration.
    widths_ns : numpy.ndarray or None
        One width per row, measured from each contributing file's own
        cadence. None when no file was long enough to measure one.
    label_hint : str or None
        A label a reader could justify from the file, e.g. from the name of
        an ICARTT independent variable.
    path : pathlib.Path
        Something path-like naming the data, for messages.

    Returns
    -------
    pandas.DataFrame
        The frame, with reserved boundary columns for every row.
    ResolvedSupport
        The conclusion and its provenance.
    """
    method: SupportMethod = support.method if support.method is not None else "point"
    method_source: SupportSource = "declared" if support.method is not None else "assumed"

    index_ns = epoch_ns(pd.DatetimeIndex(frame.index))
    reported = RAW_TIME_STOP_COLUMN in frame.columns and RAW_TIME_START_COLUMN in frame.columns
    if reported:
        start_ns = epoch_ns(pd.DatetimeIndex(frame[RAW_TIME_START_COLUMN]))
        stop_ns = epoch_ns(pd.DatetimeIndex(frame[RAW_TIME_STOP_COLUMN]))
        # Derived BEFORE any repair below, because the label is a fact about
        # how the file writes its timestamps, not about what TSARA did to a
        # handful of degenerate cells. Flooring first would let 644 rows in a
        # 90,000-row record turn the whole stream's label into 'unknown'.
        label_from_file = _label_from_bounds(index_ns, start_ns, stop_ns)

        frame, start_ns, stop_ns, n_widened = _floor_declared_cells(
            frame, start_ns, stop_ns, widths_ns=widths_ns, path=path
        )
        widths = stop_ns - start_ns
        distinct = np.unique(widths)
        return frame, ResolvedSupport(
            label=label_from_file,
            method=method,
            # One nominal width only if the file really has one; a sampler
            # whose fills vary has no single number to report and saying so
            # is more useful than reporting a mean nobody can use.
            width_ns=int(distinct[0]) if distinct.size == 1 else None,
            label_source="reported",
            width_source="reported",
            method_source=method_source,
            n_widened=n_widened,
        )

    label: SupportLabel = "unknown"
    label_source: SupportSource = "assumed"
    if support.label is not None:
        label, label_source = support.label, "declared"
    elif label_hint is not None:
        label, label_source = label_hint, "inferred"

    if support.width != "cadence":
        declared_width = int(pd.Timedelta(support.width).value)
        _warn_if_cells_would_overlap(declared_width, widths_ns, path=path)
        widths = np.full(len(frame), declared_width, dtype=np.int64)
        width_source: SupportSource = "declared"
        nominal: int | None = declared_width
    elif widths_ns is not None:
        widths = np.asarray(widths_ns, dtype=np.int64)
        width_source = "inferred"
        found = np.unique(widths)
        nominal = int(found[0]) if found.size == 1 else None
    else:
        # Nothing to build a cell from: no declared width, and no file long
        # enough to measure a cadence. Inventing one would be a fabrication
        # rather than a weak reading, so the stream goes on without cells and
        # says so. `tsara.core.support.ensure_time_bounds` takes the same view.
        logger.warning(
            "Instrument data from '%s' has no declared cell width and too few "
            "samples to measure one, so its rows carry no cell boundaries.",
            path,
        )
        return frame, ResolvedSupport(
            label=label,
            method=method,
            width_ns=None,
            label_source=label_source,
            width_source="assumed",
            method_source=method_source,
        )

    bounds = CellBounds.from_label(index_ns, widths, label)
    out = frame.copy()
    out[RAW_TIME_START_COLUMN] = bounds.start_ns.astype("datetime64[ns]")
    out[RAW_TIME_STOP_COLUMN] = bounds.stop_ns.astype("datetime64[ns]")
    return out, ResolvedSupport(
        label=label,
        method=method,
        width_ns=nominal,
        label_source=label_source,
        width_source=width_source,
        method_source=method_source,
    )


def shift_and_centre(frame: pd.DataFrame, *, shift_ns: int) -> pd.DataFrame:
    """Correct the clock, then move the index onto each cell's midpoint.

    Two operations that have to happen together and in this order, because
    both act on the same three quantities and the second one reads what the
    first one wrote.

    **The shift moves a cell; it never resizes one.** It is applied to the
    index and to both boundaries alike, so a stream corrected by nine seconds
    describes exactly the same intervals of air, nine seconds earlier.

    **The index becomes the midpoint** because that is the one label that
    means the same thing on every stream. Left at whatever each file happened
    to use, ``time`` would mean a start on one instrument and an end on
    another, and every operation that is not bounds-aware -- a plot, a
    ``sel``, someone else's code -- would carry up to a full cell of silent
    bias. Centred, the worst case is half a cell and it is unbiased. Nothing
    is lost: the original label is recorded in the stream's attributes and
    the boundaries themselves are exact.

    Note the sort that follows this is not redundant. With per-row widths a
    later row can have an earlier midpoint than its neighbour -- a wide cell
    starting just before a narrow one -- so centring can genuinely reorder a
    stream, and only re-sorting afterwards keeps the axis monotonic.

    Parameters
    ----------
    frame : pandas.DataFrame
        One instrument's rows, carrying the reserved boundary columns if its
        cells could be determined.
    shift_ns : int
        Nanoseconds to add to every timestamp; zero applies nothing.

    Returns
    -------
    pandas.DataFrame
        The frame, shifted and re-indexed on cell midpoints.
    """
    out = frame if shift_ns == 0 else frame.copy()
    if shift_ns:
        offset = pd.Timedelta(shift_ns, unit="ns")
        out.index = pd.DatetimeIndex(out.index + offset, name=out.index.name)
        for column in (RAW_TIME_START_COLUMN, RAW_TIME_STOP_COLUMN):
            if column in out.columns:
                out[column] = out[column] + offset

    if RAW_TIME_START_COLUMN not in out.columns:
        # No cells were determined, so there is no midpoint to move to. The
        # stream keeps the file's own timestamps and says, through its
        # provenance attributes, that nothing is known about their support.
        return out

    start = epoch_ns(pd.DatetimeIndex(out[RAW_TIME_START_COLUMN]))
    stop = epoch_ns(pd.DatetimeIndex(out[RAW_TIME_STOP_COLUMN]))
    if out is frame:
        out = frame.copy()
    out.index = pd.DatetimeIndex(
        (start + (stop - start) // 2).astype("datetime64[ns]"), name=frame.index.name
    )
    return out
