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
