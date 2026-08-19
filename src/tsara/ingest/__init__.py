"""Reading a campaign's raw archive into native-rate streams.

This is the stage that turns a validated :class:`~tsara.config.manifest.Manifest`
— a description of what the data *is* — into the data itself: one
:class:`xarray.Dataset` per instrument, on that instrument's own timestamps,
QA/QC masked, unit converted, and carrying a resolved uncertainty budget.

Shape of the subpackage
-----------------------
``base``
    The :class:`~tsara.ingest.base.RawTable` contract that separates
    format-specific reading from format-independent everything-else.
``crawler``
    Finding an instrument's files from path templates, and harvesting
    metadata out of where they sit in the archive.
``registry``
    Name-based reader dispatch (``@register_reader("csv")``), the same
    pattern ``docs/METHODS.md`` fixes for noise and regression estimators.
``csv_reader``
    Delimited text: comma, tab, or whitespace-padded logger output.
``icartt``
    The NASA/NOAA FFI-1001 format, plus its filename revision conventions.
``parquet_reader``
    Apache Parquet, the usual storage for a campaign's processed stages.
``units``, ``qaqc``, ``uncertainty``
    Format-independent per-variable processing: convert to canonical units,
    mask bad samples, resolve the two-component uncertainty budget.
``timeparse``
    Shared across readers: where time lives in a file, and how it reaches
    UTC exactly once at nanosecond resolution.

Importing this package imports the built-in reader modules, which is what
registers them. Readers provided by other packages must be imported by
whoever provides them.

Nothing here resamples anything. Streams stay at native rate until Phase 4
pairs them, per the "synchronize late" decision in ``docs/METHODS.md`` §1.1.
"""

from __future__ import annotations

# `csv_reader` is imported for the side effect of registering itself. It is
# not re-exported: which formats ship with TSARA is an implementation detail,
# and the public surface is the registry functions below.
from tsara.ingest import csv_reader as _csv_reader  # noqa: F401
from tsara.ingest import icartt as _icartt  # noqa: F401
from tsara.ingest import parquet_reader as _parquet_reader  # noqa: F401
from tsara.ingest.base import RawTable, TsaraIngestError
from tsara.ingest.crawler import FileMatch, compile_template, crawl
from tsara.ingest.qaqc import MaskReport, apply_qaqc
from tsara.ingest.uncertainty import ResolvedUncertainty, resolve_uncertainty
from tsara.ingest.units import canonical_units, convert_spread, convert_values
from tsara.ingest.icartt import (
    IcarttFilename,
    IcarttHeader,
    IcarttVariable,
    parse_icartt_filename,
    parse_icartt_header,
    select_latest_revisions,
)
from tsara.ingest.registry import available_readers, get_reader, read_file, register_reader

__all__ = [
    "FileMatch",
    "IcarttFilename",
    "IcarttHeader",
    "IcarttVariable",
    "MaskReport",
    "RawTable",
    "ResolvedUncertainty",
    "TsaraIngestError",
    "apply_qaqc",
    "available_readers",
    "canonical_units",
    "compile_template",
    "convert_spread",
    "convert_values",
    "crawl",
    "get_reader",
    "parse_icartt_filename",
    "parse_icartt_header",
    "read_file",
    "register_reader",
    "resolve_uncertainty",
    "select_latest_revisions",
]
