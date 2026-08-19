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

from tsara.core.exceptions import TsaraError

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "BUNDLE_MANIFEST",
    "BUNDLE_STREAMS_DIR",
    "TsaraBundleError",
]

#: Machine-readable description of what a bundle directory contains.
BUNDLE_MANIFEST = "bundle.json"

#: Subdirectory holding one netCDF file per instrument stream.
BUNDLE_STREAMS_DIR = "streams"

#: Bumped only when the layout changes incompatibly, so a future reader can
#: refuse (or migrate) an old bundle rather than misinterpreting it.
BUNDLE_FORMAT_VERSION = 1


class TsaraBundleError(TsaraError):
    """Raised when a TSARA bundle cannot be written or read.

    Distinct from a config error: the configuration may be perfectly valid
    while the *directory* is missing, incomplete, or written by an
    incompatible version.
    """
