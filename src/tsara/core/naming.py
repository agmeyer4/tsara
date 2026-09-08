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
    "LATITUDE_COORD",
    "LOD_COUNT_KEY",
    "LONGITUDE_COORD",
    "RAW_TIME_START_COLUMN",
    "RAW_TIME_STOP_COLUMN",
    "SIGMA_RAND_PREFIX",
    "SIGMA_SYS_PREFIX",
    "SupportLabel",
    "SupportMethod",
    "SupportSource",
    "TIME_BOUNDS_VAR",
    "TIME_COORD",
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
