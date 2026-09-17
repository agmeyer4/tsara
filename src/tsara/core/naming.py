"""The names TSARA gives things inside a stream.

Why this is a module rather than a convention
---------------------------------------------
Two subsystems independently produce streams: :mod:`tsara.synthetic`
manufactures them and :mod:`tsara.ingest` reads them from an archive. Every
later stage consumes both through one code path, which only works if they
agree — exactly — on what a species' random-error variable is called.

Before this module the agreement was two f-strings in two packages that
happened to match. That is the kind of coupling that survives review and
then breaks silently: rename one and nothing fails until a baseline stage
quietly finds no sigma and falls back to an empirical estimate, which is a
*plausible* answer rather than an error.

The composition that has to keep working
----------------------------------------
The generator's answer-key variables are its observable names with a
``truth_`` prefix, so ``truth_`` + :func:`sigma_rand_name` must equal the
generator's ground-truth sigma name. Building both from here is what makes
that true by construction instead of by coincidence.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "ALTITUDE_COORD",
    "BOUNDS_ATTR",
    "BOUNDS_DIM",
    "CELL_METHODS_ATTR",
    "COVERAGE_PREFIX",
    "DISPERSION_SUFFIX",
    "LATITUDE_COORD",
    "LOD_COUNT_KEY",
    "LONGITUDE_COORD",
    "N_SOURCE_PREFIX",
    "RAW_TIME_START_COLUMN",
    "RAW_TIME_STOP_COLUMN",
    "RESULTANT_LENGTH_SUFFIX",
    "SIGMA_RAND_PREFIX",
    "SIGMA_SYS_PREFIX",
    "SUPPORT_COVERAGE_ATTR",
    "SUPPORT_LABEL_ATTR",
    "SUPPORT_LABEL_SOURCE_ATTR",
    "SUPPORT_METHOD_SOURCE_ATTR",
    "SUPPORT_WIDENED_ATTR",
    "SUPPORT_WIDTH_ATTR",
    "SUPPORT_WIDTH_SOURCE_ATTR",
    "SupportLabel",
    "SupportMethod",
    "SupportSource",
    "TIME_BOUNDS_VAR",
    "TIME_COORD",
    "TIME_SHIFT_ATTR",
    "coverage_name",
    "is_companion_name",
    "is_sigma_name",
    "n_source_name",
    "sigma_rand_name",
    "sigma_sys_name",
]

#: The time dimension and coordinate, everywhere in TSARA.
TIME_COORD = "time"

#: CF cell boundaries: the variable holding each cell's start and stop, the
#: length-2 dimension it varies over, and the attribute on ``time`` that
#: points at it.
#:
#: These follow the Climate and Forecast conventions rather than a TSARA
#: invention, which is what makes a saved stream readable by ncview, CDO and
#: cf_xarray without a translation layer. ``nv`` is CF's own example name for
#: the vertex dimension; it carries no values and is only ever length 2,
#: since a time cell has a start vertex and a stop vertex.
TIME_BOUNDS_VAR = "time_bnds"
BOUNDS_DIM = "nv"
BOUNDS_ATTR = "bounds"

#: CF attribute naming how a value relates to its cell, e.g. ``time: mean``.
#: Written per data variable, since one stream can in principle mix them.
CELL_METHODS_ATTR = "cell_methods"

#: The vocabulary of temporal support, defined here rather than beside the
#: arithmetic in :mod:`tsara.core.support`, for two reasons.
#:
#: The first is the reason everything else in this module is here: three
#: subsystems have to spell these identically -- the manifest schema that
#: lets a user declare them, the generator that manufactures streams with
#: them, and the assembler that writes them into stream attributes. The
#: second is weight. :mod:`tsara.core.support` imports NumPy, and the config
#: layer imports neither NumPy nor pandas today; measured, importing it from
#: the schema would add roughly 120 ms to every config load and every CLI
#: ``--help``. This module imports nothing at all.

#: Where the timestamp sits inside its cell.
#:
#: ``unknown`` is a first-class value rather than an error: hundreds of files
#: in the target archive declare nothing at all, and refusing them would be
#: worse than admitting them with the assumption recorded. An unknown label
#: is treated as centred, which minimises the worst-case misplacement -- half
#: a cell rather than a whole one.
SupportLabel = Literal["start", "mid", "end", "unknown"]

#: Whether a value is an average over its cell or a sample inside it.
#:
#: A claim about arithmetic, not about physics. ``mean`` says an explicit
#: averaging operation over a *known* interval was performed, so the value
#: times the width is the integral over the cell. ``point`` says the value is
#: a sample, whatever instrumental smoothing lies beneath it -- which is the
#: honest description of a cavity ring-down analyzer, since it is neither an
#: instantaneous sampler nor a box-car mean.
SupportMethod = Literal["point", "mean"]

#: Where a piece of support information came from, best evidence first.
#:
#: Recorded **per field** (label, width, method) rather than once per stream,
#: because the three are established independently: a stationary Picarro with
#: a stop column and a manifest declaring ``method: mean`` is honestly
#: described as reported / reported / declared. This mirrors the uncertainty
#: system, which already records ``random`` and ``systematic`` provenance
#: separately and reports ``mixed`` when they disagree (METHODS.md 2.4).
#:
#: * ``reported``  -- per-row start/stop columns in the file itself.
#: * ``declared``  -- the manifest states it.
#: * ``inferred``  -- TSARA read it from the file (column names, cadence).
#: * ``assumed``   -- nothing said; a default was applied and labelled.
SupportSource = Literal["reported", "declared", "inferred", "assumed"]

#: Stream attributes describing temporal support and where it came from.
#:
#: The cells themselves live in CF ``time_bnds``; these record what CF has no
#: place for. ``SUPPORT_LABEL_ATTR`` keeps the file's ORIGINAL label position,
#: which matters because TSARA moves ``time`` to the cell midpoint -- without
#: it, a user comparing a stream against their raw file could not tell whether
#: a 30 s difference was a correction or a bug. The nominal width is recorded
#: separately from the bounds because per-row widths (a canister) and one
#: nominal width (a continuous logger) are different situations and a later
#: stage may need to know which it has.
#:
#: The three ``_SOURCE`` attrs are the provenance ladder, recorded per field
#: rather than once per stream, exactly as the uncertainty system records it
#: per component. See :data:`SupportSource`.
SUPPORT_LABEL_ATTR = "tsara_support_label"
SUPPORT_WIDTH_ATTR = "tsara_nominal_cell_width_s"
SUPPORT_COVERAGE_ATTR = "tsara_cell_coverage"

#: How many cells TSARA had to widen because the file declared them as having
#: no duration at all.
#:
#: Written only when it happened. A zero-width cell has zero measure and so
#: zero weight in every overlap, which would leave the row in the stream
#: looking like data while never contributing to anything -- so it is widened
#: to the record's own cadence, and the count says the repair took place.
SUPPORT_WIDENED_ATTR = "tsara_cells_widened"
SUPPORT_LABEL_SOURCE_ATTR = "tsara_support_label_source"
SUPPORT_WIDTH_SOURCE_ATTR = "tsara_support_width_source"
SUPPORT_METHOD_SOURCE_ATTR = "tsara_support_method_source"

#: Stream attribute recording a clock correction that was applied.
#:
#: Written whenever a non-zero ``InstrumentConfig.time_shift`` moved a
#: stream's timestamps, and absent otherwise. Recording it is the guard
#: against the failure that actually happens: an archive corrected once
#: upstream and again here, which no amount of internal consistency would
#: reveal because both corrections are individually right.
TIME_SHIFT_ATTR = "tsara_time_shift"

#: Reserved columns by which a reader hands per-row cell boundaries to the
#: rest of ingestion.
#:
#: A reader's contract is ``(path, loader config) -> RawTable``, with columns
#: under the names the raw file uses -- so there is no way to return bounds
#: except as columns. The underscore prefix keeps them from colliding with a
#: real instrument column, and naming them here rather than in the reader
#: means the reader and the stream assembler cannot drift apart. Only files
#: that genuinely declare per-row bounds carry them; everything else gets
#: cells built from a nominal cadence further downstream.
RAW_TIME_START_COLUMN = "_tsara_time_start"
RAW_TIME_STOP_COLUMN = "_tsara_time_stop"

#: Platform position coordinates. Scalar for a stationary site, indexed by
#: ``time`` for a mobile platform — the same names either way, so downstream
#: code reads position identically and only needs to care about the shape.
LATITUDE_COORD = "latitude"
LONGITUDE_COORD = "longitude"
ALTITUDE_COORD = "altitude"

#: Prefixes for the two uncertainty components carried alongside each
#: species. They stay separate through the whole pipeline because they
#: behave differently under averaging (``docs/METHODS.md`` §2.1).
SIGMA_RAND_PREFIX = "sigma_rand_"
SIGMA_SYS_PREFIX = "sigma_sys_"


def is_sigma_name(name: str) -> bool:
    """Return whether a variable name is one of the uncertainty companions.

    The prefixes are the seam the two producers agree on -- the synthetic
    generator and ingestion both build companion names this way, and neither
    records anything else that identifies them -- so asking about a name is
    the only test that works on a stream from either source, or on one
    reloaded from disk.

    Parameters
    ----------
    name : str
        Variable name to test.

    Returns
    -------
    bool
        True for ``sigma_rand_*`` / ``sigma_sys_*``.
    """
    return name.startswith((SIGMA_RAND_PREFIX, SIGMA_SYS_PREFIX))


def sigma_rand_name(variable: str) -> str:
    """Return the name of a variable's random-uncertainty companion.

    Parameters
    ----------
    variable : str
        Canonical variable name, e.g. ``'ch4'``.

    Returns
    -------
    str
        e.g. ``'sigma_rand_ch4'``.
    """
    return f"{SIGMA_RAND_PREFIX}{variable}"


def sigma_sys_name(variable: str) -> str:
    """Return the name of a variable's systematic-uncertainty companion.

    Parameters
    ----------
    variable : str
        Canonical variable name, e.g. ``'ch4'``.

    Returns
    -------
    str
        e.g. ``'sigma_sys_ch4'``.
    """
    return f"{SIGMA_SYS_PREFIX}{variable}"


#: Columns the joining operation adds beside every value it puts on new cells
#: (``docs/METHODS.md`` §11.2): how many source cells contributed, and how much
#: of the target cell they covered.
N_SOURCE_PREFIX = "n_source_"
COVERAGE_PREFIX = "coverage_"

#: What an angular variable gets instead of a sigma, since a direction has no
#: meaningful arithmetic spread (§11.5).
RESULTANT_LENGTH_SUFFIX = "_resultant_length"
DISPERSION_SUFFIX = "_dispersion"


def n_source_name(variable: str) -> str:
    """Return the name of a variable's contributing-cell count.

    Parameters
    ----------
    variable : str
        Column name as it appears in the joined product, e.g. ``'ch4'``.

    Returns
    -------
    str
        e.g. ``'n_source_ch4'``.
    """
    return f"{N_SOURCE_PREFIX}{variable}"


def coverage_name(variable: str) -> str:
    """Return the name of a variable's coverage fraction.

    Parameters
    ----------
    variable : str
        Column name as it appears in the joined product, e.g. ``'ch4'``.

    Returns
    -------
    str
        e.g. ``'coverage_ch4'``.
    """
    return f"{COVERAGE_PREFIX}{variable}"


def is_companion_name(name: str) -> bool:
    """Return whether a name describes another variable rather than being one.

    Four families of column exist only to qualify the column they are named
    after: the two uncertainty components, the contributing-cell count, the
    coverage fraction, and the two an angular variable carries instead of a
    sigma. None is a measurement in its own right, and each is produced
    automatically alongside its parent.

    The distinction is load-bearing rather than tidy. A stage that selects
    "every variable" and gets these too will bin a coverage fraction as though
    it were data, yielding ``coverage_coverage_ch4`` -- a column with no parent
    and no meaning -- and will grow a further layer on every pass. Asking about
    the *name* is the only test available, exactly as for the sigma companions:
    nothing else in a stream marks them, and a stream may have come from the
    generator, from ingestion, or from a file reloaded from disk.

    Parameters
    ----------
    name : str
        Variable name to test.

    Returns
    -------
    bool
        True for a companion of any of the four families.
    """
    return (
        is_sigma_name(name)
        or name.startswith((N_SOURCE_PREFIX, COVERAGE_PREFIX))
        or name.endswith((RESULTANT_LENGTH_SUFFIX, DISPERSION_SUFFIX))
    )


#: Attr key under which a reader reports per-raw-column counts of samples
#: masked as out-of-detection-range.
#:
#: Lives here for the same reason the sigma prefixes do: two modules have to
#: spell it identically or the information silently disappears. A reader
#: writes it into ``RawTable.attrs``, campaign orchestration sums it across
#: files, and stream assembly pops it back out to attach each count to the
#: variable it censors. Keeping it in :mod:`tsara.core.naming` also avoids an
#: import cycle, since ingestion's orchestration and assembly modules already
#: depend on each other in one direction.
LOD_COUNT_KEY = "icartt_lod_masked"
