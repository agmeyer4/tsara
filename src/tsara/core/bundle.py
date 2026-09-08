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
    "SUPPORTED_BUNDLE_VERSIONS",
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
#:
#: Version 2 (Phase 3.5) added CF cell boundaries to every stream: each value
#: now carries the time interval it describes rather than a bare instant.
BUNDLE_FORMAT_VERSION = 2

#: Versions this TSARA can still read, oldest first.
#:
#: A version-1 bundle is *migrated* rather than refused, because the change
#: that produced version 2 is additive and the older layout has an exact,
#: honest reading: cells of the record's own nominal cadence, centred on each
#: timestamp, every field of their support marked ``assumed``. Refusing would
#: strand any bundle written before the upgrade for no gain -- the whole point
#: of labelling provenance is that a weak reading can be admitted safely.
SUPPORTED_BUNDLE_VERSIONS = (1, 2)

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
        if name not in dataset.variables:
            continue
        variable = dataset[name]
        if variable.dtype.kind == "M" and str(variable.dtype) != "datetime64[ns]":
            # Not a nicety. Pinning NANOSECOND units onto an axis stored at a
            # coarser resolution makes xarray write the NaT sentinel for every
            # value, and the file then reads back as an axis of NaT with no
            # error anywhere -- measured, on a plain `pd.date_range`, which
            # pandas now returns in microseconds. Widening to nanoseconds is
            # exact, so it is done here rather than refused. TSARA's own
            # producers already pin ns (the readers enforce it, the generator
            # builds it), so this fires only for a dataset assembled outside
            # the package.
            dataset[name] = variable.astype("datetime64[ns]")
        dataset[name].encoding.update(TIME_ENCODING)
    return dataset
