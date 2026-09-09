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
    "BUNDLE_VERSION_WITH_CELLS",
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

#: First format version whose streams record their own cells.
#:
#: The line between two absences that look identical on disk and mean opposite
#: things. In a version-1 stream, no ``time_bnds`` means *the format had no way
#: to record one* -- so completing it with a weak, labelled reading adds
#: information. In a version-2 stream it means *ingestion could not know*: no
#: width was declared and no file was long enough to measure one, and it said
#: so rather than inventing a number. Migrating that second case would move a
#: stream UP the provenance ladder -- ``assumed`` to ``inferred`` -- for no
#: reason but having been written to disk and read back, which is precisely
#: what recording provenance per field exists to prevent.
BUNDLE_VERSION_WITH_CELLS = 2

#: Versions this TSARA can still read, oldest first.
#:
#: A version-1 bundle is *migrated* rather than refused, because the change
#: that produced version 2 is additive and the older layout has an exact,
#: honest reading: cells of the record's own nominal cadence, centred on each
#: timestamp, with the label and method marked ``assumed`` and the width
#: ``inferred`` -- the cadence really was measured from the record, and saying
#: ``assumed`` there would understate it in the one direction the ladder is
#: not allowed to move. Refusing would strand any bundle written before the
#: upgrade for no gain: the whole point of labelling provenance is that a weak
#: reading can be admitted safely.
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
#: Not cosmetic -- though not for the reason first recorded here, which was
#: wrong. A CF bounds variable never carries units of its own: it inherits its
#: parent's, and xarray implements that, writing ``time_bnds`` with no
#: ``units`` attribute at all and encoding it with whatever ``time`` was
#: given. The two therefore *cannot* disagree about an epoch, and the pair of
#: differing epochs this comment once claimed to have measured is not a state
#: the writer can reach.
#:
#: What xarray actually warns about, measured, is a datetime coordinate that
#: has a bounds variable and no pinned units: left alone it picks a units
#: string from the data -- "minutes since 2024-07-01 00:01:00" on a four-row
#: stream -- and applies it to the bounds as well, so a file's resolution
#: depends on when its record happens to start. Pinning one absolute epoch
#: silences the warning and keeps a saved file byte-comparable across runs
#: that begin at different instants.
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
            # Not a nicety, and this -- not any epoch mismatch -- is why
            # `time_bnds` is in this loop at all. Nanosecond units applied to
            # an axis stored at a coarser resolution make xarray write the NaT
            # sentinel for every value, and the file reads back as NaT with no
            # error anywhere. Measured on a microsecond bounds array beside a
            # nanosecond time axis: every cell returns `['NaT' 'NaT']`, and
            # since the bounds inherit the *parent's* pinned units they hit
            # this even though nothing pinned anything onto them. Widening is
            # exact, so it is done here rather than refused. TSARA's own
            # producers already build ns (the readers enforce it, the
            # generator and `attach_time_bounds` construct it), so this fires
            # only for a dataset assembled outside the package.
            dataset[name] = variable.astype("datetime64[ns]")
        dataset[name].encoding.update(TIME_ENCODING)
    return dataset
