"""Assembling one instrument's raw table into a native-rate stream.

This is where ingestion stops speaking pandas and starts speaking xarray,
and where the manifest's *description* of an instrument becomes a dataset
that later stages can consume without knowing anything about files.

The output shape, and why it matches the synthetic generator exactly
--------------------------------------------------------------------
:mod:`tsara.synthetic` manufactures streams and this module reads them from
an archive, and every later phase must consume both through one code path —
that is the whole point of shipping a generator (CLAUDE.md §5): synthetic
data with known truth is the only correctness arbiter available, so it has
to be substitutable for real data everywhere. Concretely a stream is an
:class:`xarray.Dataset` with:

* a ``time`` dimension carrying tz-naive UTC nanosecond timestamps;
* one variable per canonical name, in canonical units, QA/QC masked, each
  recording in a ``field`` attribute the physical quantity it measures;
* ``sigma_rand_<name>`` / ``sigma_sys_<name>`` companions wherever the
  manifest let ingestion compute them, named via :mod:`tsara.core.naming` so
  the two producers cannot drift apart;
* ``latitude``/``longitude`` coordinates for a stationary platform;
* self-describing attrs — package version, stage, instrument, platform kind
  — so a file found on disk explains itself (CLAUDE.md §5).

Nothing is resampled here. Streams stay on the instrument's own irregular
timestamps until Phase 4 pairs them, per the "synchronize late" decision in
``docs/METHODS.md`` §1.1.

Why mobile platforms get no coordinates yet
-------------------------------------------
A stationary site has one position, so attaching it is exact and free. A
mobile platform's position lives on the GPS instrument's clock, and putting
it onto a gas instrument's clock is *interpolation* — permitted for smooth
auxiliary fields but only under the ``max_interp_gap`` guard, which lives in
``AlignmentConfig``, a Phase-4 object that ingestion has no business
reading (METHODS §1.2). So ingestion loads GPS as an ordinary stream,
records the binding in attrs, and leaves the join to the stage that owns the
guard. That keeps the interpolation rule enforced in exactly one place.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from tsara import __version__
from tsara.config.manifest import MobilePlatform, StationaryPlatform
from tsara.core.circular import wrap_degrees
from tsara.core.naming import (
    ALTITUDE_COORD,
    LATITUDE_COORD,
    LOD_COUNT_KEY,
    LONGITUDE_COORD,
    RAW_TIME_START_COLUMN,
    RAW_TIME_STOP_COLUMN,
    TIME_COORD,
    TIME_SHIFT_ATTR,
    sigma_rand_name,
    sigma_sys_name,
)
from tsara.core.support import CellBounds, attach_time_bounds, support_attrs
from tsara.core.timebase import epoch_ns
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.qaqc import apply_qaqc, masked_fraction
from tsara.ingest.uncertainty import resolve_uncertainty
from tsara.ingest.units import canonical_units, convert_values

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from pathlib import Path

    from tsara.config.manifest import InstrumentConfig, PlatformConfig, VariableConfig
    from tsara.ingest.support import ResolvedSupport

logger = logging.getLogger(__name__)

__all__ = ["build_stream"]


def build_stream(
    frame: pd.DataFrame,
    instrument: InstrumentConfig,
    *,
    name: str,
    platform: PlatformConfig,
    campaign: str = "",
    sources: Sequence[Path] = (),
    file_attrs: Mapping[str, object] | None = None,
    support: ResolvedSupport | None = None,
    time_shift: str | None = None,
) -> xr.Dataset:
    """Turn one instrument's combined raw table into a native-rate stream.

    Parameters
    ----------
    frame : pandas.DataFrame
        The instrument's data, time-indexed and **already concatenated across
        files and sorted**. Concatenating first is not incidental: QA/QC
        windows and uncertainty are campaign-level quantities, and computing
        them per file would make the answer depend on how the archive
        happened to be split.
    instrument : InstrumentConfig
        The manifest entry describing this instrument.
    name : str
        Instrument (stream) name — its key in ``Manifest.instruments``.
    platform : PlatformConfig
        Campaign platform, stationary or mobile.
    campaign : str, optional
        ``Manifest.name``, recorded in attrs.
    sources : Sequence of pathlib.Path, optional
        Files that contributed, recorded as provenance.
    file_attrs : Mapping, optional
        What the source files declared about *themselves*, reconciled
        across them by
        :func:`~tsara.ingest.campaign._merge_file_attrs` — an ICARTT
        header's PI, mission, revision and limit-of-detection flags, for
        instance. Written into the stream's attrs so a saved product
        explains itself without its source archive (CLAUDE.md §5). Counts
        of samples masked as out-of-detection-range are routed to the
        variable they describe instead of the dataset.
    support : ResolvedSupport, optional
        What was concluded about this instrument's cells, resolved before
        concatenation by :mod:`tsara.ingest.support`. None leaves the stream
        without cell boundaries, which is what a record too short to have a
        measurable cadence produces.
    time_shift : str, optional
        The clock correction orchestration has *already applied* to
        ``frame``, recorded here so a saved product says it happened. Passed
        rather than applied, because this function receives a frame whose
        axis is final: shifting here would move timestamps out from under the
        ordering and de-duplication that already ran on them.

    Returns
    -------
    xarray.Dataset
        The stream.

    Raises
    ------
    TsaraIngestError
        If the index is not a ``DatetimeIndex``, it is not sorted, or a
        declared column is absent from the data.
    """
    index = _normalize_index(frame, name)
    bounds = _cell_bounds(frame)
    declared = dict(file_attrs or {})
    lod_counts = declared.pop(LOD_COUNT_KEY, None)
    lod_by_column = dict(lod_counts) if isinstance(lod_counts, Mapping) else {}

    data_vars: dict[str, Any] = {}
    for canonical, variable in instrument.variables.items():
        _add_variable(
            data_vars,
            frame,
            canonical,
            variable,
            field=instrument.field_of(canonical),
            instrument_name=name,
            sources=sources,
            lod_by_column=lod_by_column,
            cell_width_ns=None if bounds is None else bounds.width_ns,
        )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={TIME_COORD: index},
        attrs=_stream_attrs(
            name,
            instrument,
            platform,
            campaign=campaign,
            sources=sources,
            declared=declared,
            support=support,
            bounds=bounds,
            time_shift=time_shift,
        ),
    )
    if bounds is not None and support is not None:
        attach_time_bounds(dataset, bounds, support.method)
    _attach_platform_coords(dataset, platform)
    return dataset


def _cell_bounds(frame: pd.DataFrame) -> CellBounds | None:
    """Read the cell boundaries orchestration resolved, if it managed to.

    They arrive as reserved frame columns rather than as an argument because
    that is how they survived concatenation, sorting and de-duplication
    alongside the rows they describe (:mod:`tsara.ingest.base`). Here is
    where they stop being table columns and become the CF representation.
    """
    if RAW_TIME_START_COLUMN not in frame.columns:
        return None
    return CellBounds(
        start_ns=epoch_ns(pd.DatetimeIndex(frame[RAW_TIME_START_COLUMN])),
        stop_ns=epoch_ns(pd.DatetimeIndex(frame[RAW_TIME_STOP_COLUMN])),
    )


def _normalize_index(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    """Validate the frame's time axis and pin it to nanoseconds.

    The readers already produce a tz-naive nanosecond index, but this
    function accepts any frame, and the resolution pin is the kind of
    invariant that has to hold at every entry point rather than most of
    them: an index left at microseconds looks perfectly healthy, survives
    every stage, and then changes dtype the first time it round-trips
    through netCDF — after which an exact comparison against an event
    boundary silently stops matching.
    """
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TsaraIngestError(
            f"Stream '{name}' must be built from a DatetimeIndex-ed frame, "
            f"got {type(index).__name__}."
        )
    if index.tz is not None:
        raise TsaraIngestError(
            f"Stream '{name}' has a timezone-aware time index ({index.tz}). "
            "TSARA is tz-naive UTC internally."
        )
    if not index.is_monotonic_increasing:
        # A reachable state rather than a formality: archive records do
        # step backwards sometimes, and files concatenated in path order are
        # not in time order at all. Sorting is the orchestration stage's job;
        # catching it here keeps a rolling QA/QC window from failing later
        # with a message that names nothing.
        raise TsaraIngestError(
            f"Stream '{name}' has timestamps that are not monotonically "
            "increasing. Sort the concatenated table before building a stream."
        )
    return pd.DatetimeIndex(index.astype("datetime64[ns]"), name=TIME_COORD)


def _add_variable(
    data_vars: dict[str, Any],
    frame: pd.DataFrame,
    canonical: str,
    variable: VariableConfig,
    *,
    field: str,
    instrument_name: str,
    sources: Sequence[Path],
    lod_by_column: Mapping[str, int] = MappingProxyType({}),
    cell_width_ns: Any = None,
) -> None:
    """Convert, mask and resolve one variable, adding it and its sigmas.

    ``field`` arrives resolved (:meth:`InstrumentConfig.field_of`) rather than
    read off ``variable``, which cannot know its own name and so cannot apply
    the default.
    """
    if variable.column not in frame.columns:
        raise TsaraIngestError(
            f"Variable '{canonical}' of instrument '{instrument_name}' reads "
            f"column '{variable.column}', which is not in the data. "
            f"Columns present: {list(frame.columns)[:12]}."
        )

    label = _source_label(sources)

    # Order is fixed and load-bearing: convert first so QA/QC bounds and the
    # declared `absolute` noise floor are both interpreted in canonical
    # units (see tsara.ingest.qaqc and METHODS §2.2).
    raw = pd.to_numeric(frame[variable.column], errors="coerce")
    values = convert_values(np.asarray(raw, dtype="float64"), variable.convert)
    if variable.circular:
        # A direction is wrapped back onto one turn BEFORE QA/QC, for the same
        # reason conversion comes first: a range bound is a statement about
        # canonical values. The commonest conversion for a direction is an
        # offset -- magnetic to true bearing -- and without the wrap it pushes
        # every reading within that offset of north past 360, where the
        # natural rule `range: [0, 360]` masks a whole sector of the wind rose
        # rather than a random fraction of it. The wrap is exact and changes
        # no direction, so it is a declaration's consequence, not a model.
        values = wrap_degrees(values)
    converted = pd.Series(values, index=frame.index)
    masked, reports = apply_qaqc(converted, variable.qaqc, frame, variable=canonical, path=label)
    resolved = resolve_uncertainty(
        masked,
        variable.uncertainty,
        frame,
        conversion=variable.convert,
        variable=canonical,
        path=label,
        cell_width_ns=cell_width_ns,
    )

    units = canonical_units(variable.units, variable.convert)
    attrs: dict[str, Any] = {
        "units": units,
        "role": variable.role,
        # What is measured, as opposed to what this stream calls it. Written
        # even when it equals the name, so a reader never has to know the
        # default: every variable says, and a joined product keeps it, since
        # the joins copy a variable's attributes onto its column (METHODS §1.6).
        "field": field,
        # netCDF has no boolean type, so this is stored as 0/1 — the same
        # convention the synthetic generator uses.
        "circular": int(variable.circular),
        "raw_column": variable.column,
        "uncertainty_source": resolved.source,
        "uncertainty_source_random": resolved.random_source,
        "uncertainty_source_systematic": resolved.systematic_source,
        "masked_fraction": masked_fraction(masked),
    }
    if variable.description:
        attrs["description"] = variable.description
    if resolved.decorrelation_timescale is not None:
        attrs["decorrelation_timescale"] = resolved.decorrelation_timescale
    # The interval a declared figure was quoted at, and how many of those
    # intervals fit in a cell. Nothing here rescales a sigma -- that needs a
    # decorrelation timescale and belongs to the stage that wants a sigma at
    # a particular support (METHODS 10.8) -- so these two attrs are how the
    # product stays honest about a figure that describes a different interval
    # from the cells it sits on. A reader cannot re-derive either.
    if resolved.at_width is not None:
        attrs["uncertainty_at_width"] = resolved.at_width
    if resolved.at_width_ratio is not None:
        attrs["uncertainty_at_width_ratio"] = float(resolved.at_width_ratio)
    if resolved.systematic_at_width is not None:
        attrs["uncertainty_systematic_at_width"] = resolved.systematic_at_width
    # Recorded per variable rather than per stream because that is the
    # scope of the fact: a below-detection count belongs to the species it
    # censors. Keyed on the RAW column name, the only name a reader knows.
    n_below_lod = lod_by_column.get(variable.column)
    if n_below_lod:
        attrs["n_lod_masked"] = int(n_below_lod)
    if reports:
        # One compact string rather than an attr per rule: two rules of the
        # same kind would collide as attr names, and this stays readable in
        # `ncdump -h`.
        attrs["qaqc_masked"] = ", ".join(f"{r.kind}:{r.n_masked}" for r in reports)

    data_vars[canonical] = (TIME_COORD, np.asarray(masked, dtype="float64"), attrs)

    if resolved.random is not None:
        data_vars[sigma_rand_name(canonical)] = (
            TIME_COORD,
            resolved.random,
            {
                "units": units,
                "description": f"Random 1-sigma for {canonical}.",
                "uncertainty_component": "random",
                "uncertainty_source": resolved.random_source,
            },
        )
    if resolved.systematic is not None:
        data_vars[sigma_sys_name(canonical)] = (
            TIME_COORD,
            resolved.systematic,
            {
                "units": units,
                "description": f"Systematic 1-sigma for {canonical}.",
                "uncertainty_component": "systematic",
                "uncertainty_source": resolved.systematic_source,
            },
        )


def _source_label(sources: Sequence[Path]) -> Path:
    """Return something path-like to name in messages about a merged table.

    After concatenation there is no single file to blame, so messages name
    the first contributing file when there is exactly one and a count
    otherwise. Keeping the parameter path-typed means the QA/QC and
    uncertainty helpers need no separate "merged" code path.
    """
    from pathlib import Path as _Path

    if len(sources) == 1:
        return sources[0]
    if not sources:
        return _Path("<data>")
    return _Path(f"<{len(sources)} files starting {sources[0].name}>")


def _stream_attrs(
    name: str,
    instrument: InstrumentConfig,
    platform: PlatformConfig,
    *,
    campaign: str,
    sources: Sequence[Path],
    declared: Mapping[str, object] = MappingProxyType({}),
    support: ResolvedSupport | None = None,
    bounds: CellBounds | None = None,
    time_shift: str | None = None,
) -> dict[str, Any]:
    """Build the self-describing attrs every stream carries.

    Saved products self-describe (CLAUDE.md §5): package version, the stage
    that produced them, and enough provenance to retrace where the numbers
    came from. For a mobile platform this is also where the GPS binding is
    recorded, so Phase 4 can find the track without re-reading the manifest.
    """
    # The file's own claims go in first so that TSARA's statements about
    # the run always win a name collision: what the package did is not
    # negotiable, whereas a header field is whatever the producer wrote.
    attrs: dict[str, Any] = {str(key): value for key, value in declared.items()}
    attrs |= {
        "tsara_version": __version__,
        "tsara_stage": "ingest",
        "instrument": name,
        "platform_kind": platform.kind,
        "n_source_files": len(sources),
        "loader_format": instrument.loader.format,
    }
    if support is not None:
        attrs |= support_attrs(
            label=support.label,
            width_ns=support.width_ns,
            coverage=bounds.coverage_fraction if bounds is not None else float("nan"),
            label_source=support.label_source,
            width_source=support.width_source,
            method_source=support.method_source,
            n_widened=support.n_widened,
        )
    if time_shift is not None:
        attrs[TIME_SHIFT_ATTR] = time_shift
    if campaign:
        attrs["campaign"] = campaign
    if instrument.description:
        attrs["description"] = instrument.description
    for key, value in instrument.metadata.items():
        attrs[f"meta_{key}"] = value

    if isinstance(platform, MobilePlatform):
        # Recorded, not applied: attaching the track to this clock is
        # interpolation, and its guard belongs to Phase 4 (see module docs).
        attrs["platform_gps_instrument"] = platform.gps_instrument
        attrs["platform_lat_variable"] = platform.lat_variable
        attrs["platform_lon_variable"] = platform.lon_variable
        if platform.alt_variable is not None:
            attrs["platform_alt_variable"] = platform.alt_variable
    return attrs


def _attach_platform_coords(dataset: xr.Dataset, platform: PlatformConfig) -> None:
    """Attach position coordinates, in place, where they are exactly known.

    Stationary only. The asymmetry mirrors the manifest's platform union and
    the interpolation rule: one fixed position can be attached to any clock
    without approximating anything, while a moving one cannot.
    """
    if not isinstance(platform, StationaryPlatform):
        return
    dataset.coords[LATITUDE_COORD] = platform.latitude
    dataset.coords[LONGITUDE_COORD] = platform.longitude
    if platform.altitude_m is not None:
        dataset.coords[ALTITUDE_COORD] = platform.altitude_m
