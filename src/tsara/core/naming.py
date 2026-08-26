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
    "LATITUDE_COORD",
    "LOD_COUNT_KEY",
    "LONGITUDE_COORD",
    "SIGMA_RAND_PREFIX",
    "SIGMA_SYS_PREFIX",
    "TIME_COORD",
    "sigma_rand_name",
    "sigma_sys_name",
]

#: The time dimension and coordinate, everywhere in TSARA.
TIME_COORD = "time"

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
