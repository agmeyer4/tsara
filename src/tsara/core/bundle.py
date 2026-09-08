"""The on-disk bundle convention, shared by every stage that saves one.

CLAUDE.md §5 fixes a directory layout — a "TSARA bundle" — that holds each
stage's products alongside the configuration that produced them, so that
intermediates are inspectable in a notebook, a long HPC run can resume after
a crash, and a whole analysis can be handed to a collaborator as one
directory.

Two stages already write bundles: :mod:`tsara.synthetic` saves manufactured
streams with their answer key, and :mod:`tsara.ingest` saves streams read
from an archive. They share the parts that a *reader* has to agree on — the
manifest filename, the streams subdirectory, and the format version — so
this module holds those, and each stage adds only the files that are its
own. Duplicating them would let the two drift into layouts that look
identical and are not, which is precisely the case a format version exists
to catch and could no longer catch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tsara.core.exceptions import TsaraError
from tsara.core.naming import TIME_BOUNDS_VAR, TIME_COORD

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "BUNDLE_MANIFEST",
    "BUNDLE_STAGE_KEY",
    "BUNDLE_STREAMS_DIR",
    "TIME_ENCODING",
    "TsaraBundleError",
    "pin_time_encoding",
]

#: Machine-readable description of what a bundle directory contains.
BUNDLE_MANIFEST = "bundle.json"

#: Subdirectory holding one netCDF file per instrument stream.
BUNDLE_STREAMS_DIR = "streams"

#: Bumped only when the layout changes incompatibly, so a future reader can
#: refuse (or migrate) an old bundle rather than misinterpreting it.
BUNDLE_FORMAT_VERSION = 1

#: Key in ``bundle.json`` naming the stage that wrote the bundle.
#:
#: The shared skeleton above is deliberately identical across stages, so the
#: skeleton alone cannot tell a synthetic bundle from an ingest one -- both
#: are a ``bundle.json`` at format version 1 beside a ``streams/`` directory.
#: This key is what makes them distinguishable, which means every loader must
#: agree on its spelling; that is the definition of something belonging here
#: rather than in either stage.
BUNDLE_STAGE_KEY = "stage"


class TsaraBundleError(TsaraError):
    """Raised when a TSARA bundle cannot be written or read.

    Distinct from a config error: the configuration may be perfectly valid
    while the *directory* is missing, incomplete, or written by an
    incompatible version.
    """


#: netCDF encoding pinned onto every time axis TSARA writes.
#:
#: Not cosmetic. Left to itself, xarray chooses a units string per variable
#: from that variable's own values, so a ``time`` coordinate and its
#: ``time_bnds`` companion get **different reference epochs** -- measured
#: here, "nanoseconds since 2024-07-01 00:00:00" against "nanoseconds since
#: 2024-06-30 23:59:30", because the first cell's start precedes the first
#: timestamp. xarray warns about exactly this and CF requires the two to
#: agree. Pinning one absolute epoch also keeps a saved file byte-comparable
#: across runs whose records begin at different instants.
#:
#: Nanoseconds and int64 rather than a coarser unit, for the reason the whole
#: package pins ns: a microsecond axis round-trips through netCDF as
#: nanoseconds and changes dtype on save/load, after which an exact
#: comparison against an event boundary silently stops matching.
TIME_ENCODING = {
    "units": "nanoseconds since 1970-01-01",
    "dtype": "int64",
    "calendar": "proleptic_gregorian",
}


def pin_time_encoding(dataset: xr.Dataset) -> xr.Dataset:
    """Pin the netCDF time encoding on a stream, in place.

    Applied by every bundle writer before ``to_netcdf``, so that the two
    stages that persist streams cannot drift into writing different time
    representations for the same instants.

    Parameters
    ----------
    dataset : xarray.Dataset
        Stream about to be written. A missing time coordinate is not an
        error: the function pins what is present, so a caller need not know
        whether a particular product has a time axis or cell bounds.

    Returns
    -------
    xarray.Dataset
        The same object, for chaining.
    """
    for name in (TIME_COORD, TIME_BOUNDS_VAR):
        if name in dataset.variables:
            dataset[name].encoding.update(TIME_ENCODING)
    return dataset
