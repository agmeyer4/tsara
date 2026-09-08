"""Writing a synthetic dataset back out as raw files plus a matching manifest.

Why this exists
---------------
Ingestion has no other honest way to be checked. The campaign this package
is built for has no plume-free clean stretches and no controlled releases,
so there is no measured record whose right answer is known independently
(CLAUDE.md §5). Injected synthetic truth is the only arbiter available — but
it can only arbitrate *ingestion* if the synthetic data is made to travel
the same road real data does: written to files, discovered by a crawler,
parsed by a reader, converted, masked, and reassembled.

This module closes that loop. It takes a
:class:`~tsara.synthetic.generator.SyntheticDataset` — whose true values,
true error components and true event times are all known — and writes the
archive a real campaign would have produced, together with the manifest that
describes it. Ingesting that archive and comparing against the generator's
answer key checks crawling, reading, assembly and uncertainty resolution
against expectations no part of :mod:`tsara.ingest` wrote.

What it covers, and what it does not
------------------------------------
The default export declares no unit conversion and no QA/QC rules, because
it writes each species in its own canonical units under its own name: there
is nothing to convert or mask, and a manifest that claimed otherwise would
be describing a file that does not exist. That honesty costs coverage. A
round trip in that shape was measured against five injected ingestion bugs
and noticed one of them; three of the four misses were conversion-related.

``raw_units`` closes most of that gap. Given a scale and offset per species,
the exporter writes that species in *non-canonical* units — the shape a real
archive actually has, instrument units on disk and canonical units after the
manifest — and declares the conversion that undoes it. Ingestion must then
land back on the generator's truth, which exercises unit conversion, the
QA/QC bounds that are defined in canonical units, and the deliberate
asymmetry whereby a declared ``absolute`` sigma is already canonical while a
reported sigma column is not.

One detail is worth stating because it is easy to get wrong: a conversion
must carry **both** a scale and an offset to be discriminating. With a zero
offset, ``value * scale + offset`` and ``(value + offset) * scale`` are the
same function, and an ordering bug in
:func:`~tsara.ingest.units.convert_values` passes unnoticed.

It is public rather than a test fixture because the same trick is useful to
users: point this at a generated dataset, ingest it, and you have a working
manifest to copy and a demonstration that your reading of the schema matches
TSARA's.

A mobile platform needs no special handling: the generator already emits
its GPS track as a separate stream on its own clock — the canonical
multi-rate case — so it is written like any other instrument and only needs
*declaring*, since the platform is not listed among the configured
instruments.

Why CSV, and only CSV
---------------------
A round trip proves something about the *reader* only when the writer is
trivially, unarguably correct. Delimited text is: pandas writes it, and the
mapping from value to text and back involves no convention TSARA invented.
An ICARTT writer would be a second piece of TSARA-authored code, and a round
trip through it would demonstrate that the writer and reader agree with each
other — not that either matches the FFI-1001 specification. The ICARTT
reader is instead checked against hand-written fixtures reproducing the
quirks real archives contain, which is the evidence that actually bears on
its correctness.

The one place exactness needs care
----------------------------------
Timestamps are written in ISO 8601 with full precision, because the
generator's jitter puts real information in the nanosecond digits and a
``%f``-style format truncates at microseconds. Floating-point values are
written at ``repr`` precision, which is round-trip exact. Note that pandas'
*default* CSV parser is not: about 41% of noisy synthetic values come back
one unit in the last place away. Set ``float_precision: exact`` on the
loader to recover them bitwise (``docs/METHODS.md`` §9.2.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import yaml

from tsara.config.manifest import Manifest
from tsara.core.naming import (
    LATITUDE_COORD,
    LONGITUDE_COORD,
    TIME_BOUNDS_VAR,
    TIME_COORD,
)
from tsara.synthetic.background import TsaraSyntheticError
from tsara.synthetic.config import MobileTrack, StationarySite

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import xarray as xr

    from tsara.synthetic.config import SyntheticConfig
    from tsara.synthetic.generator import SyntheticDataset

logger = logging.getLogger(__name__)

__all__ = [
    "EXPORT_MANIFEST",
    "EXPORT_RAW_DIR",
    "RawUnits",
    "START_COLUMN",
    "STOP_COLUMN",
    "SupportDeclaration",
    "export_raw",
]

#: How much an exported archive says about its own temporal support.
#:
#: The three rungs a real archive actually occupies, so the round trip can be
#: run against each. ``reported`` writes per-row boundary columns and names
#: them, the way a canister sampler publishes its fill times. ``declared``
#: states a uniform label, width and method in the manifest, the way a
#: producer describes a 1-minute product in prose that a human transcribes.
#: ``none`` says nothing at all, which is the case for hundreds of real files
#: and the one where ingestion must fall back to assumptions and label them.
SupportDeclaration = Literal["reported", "declared", "none"]

#: Column names an exported file uses for per-row cell boundaries.
START_COLUMN = f"{TIME_COORD}_start"
STOP_COLUMN = f"{TIME_COORD}_stop"

#: Manifest written beside the raw files it describes.
EXPORT_MANIFEST = "manifest.yaml"

#: Subdirectory holding the generated raw files.
EXPORT_RAW_DIR = "raw"


@dataclass(frozen=True)
class RawUnits:
    """Units a species is written in, other than its canonical ones.

    Describes the round trip's inverse: the file receives
    ``(canonical - offset) / scale`` and the manifest declares
    ``canonical = raw * scale + offset``, so ingestion has to undo exactly
    what the writer did. Expressing it this way round — the *canonical*
    conversion, with the exporter inverting it — means the manifest holds
    the same numbers a real one would, rather than their reciprocals.

    Attributes
    ----------
    from_unit : str
        Unit written to the file, e.g. ``"ppm"``.
    scale, offset : float
        The conversion back to canonical units. Give a non-zero ``offset``
        when the point is to test conversion *ordering*; with ``offset = 0``
        the two possible orderings are the same function.
    """

    from_unit: str
    scale: float = 1.0
    offset: float = 0.0


def export_raw(
    dataset: SyntheticDataset,
    path: str | Path,
    *,
    raw_units: Mapping[str, RawUnits] | None = None,
    qaqc_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    support_declaration: SupportDeclaration = "declared",
    time_shift: Mapping[str, str] | None = None,
) -> Path:
    """Write a synthetic dataset as raw CSV files with a matching manifest.

    Parameters
    ----------
    dataset : SyntheticDataset
        Generated data. Only its observable variables are written; the
        ``truth_``-prefixed answer key stays behind, which is what makes the
        result a fair test rather than a leak.
    path : str or pathlib.Path
        Directory to write into. Created if absent.
    raw_units : Mapping of str to RawUnits, optional
        Species to write in non-canonical units, keyed by canonical species
        name. The file receives ``(canonical - offset) / scale`` and the
        manifest declares the conversion that undoes it, so ingestion must
        recover the generator's truth. See the module docstring for why this
        materially strengthens the round trip, and why an offset of zero
        weakens it.
    qaqc_bounds : Mapping of str to tuple, optional
        ``(min, max)`` range rules to declare per species, in **canonical**
        units. Present so the round trip can check that QA/QC bounds are
        applied after unit conversion rather than before
        (``docs/METHODS.md`` §9.4); either bound may be ``None``.
    support_declaration : {'declared', 'reported', 'none'}, optional
        How much the exported archive says about its own cells. The three
        rungs a real archive occupies; see :data:`SupportDeclaration`. The
        default declares, because a harness whose default exercises the
        weakest path is weaker than it looks -- the lesson ``raw_units``
        exists to remember (``docs/METHODS.md`` §9.9).

        Note what ``none`` does *not* change: the file still carries whatever
        timestamps the instrument's label implies, so exporting a
        start-labelled instrument with ``none`` writes start times that
        nothing identifies as such. Ingestion must then centre them and be
        wrong by half a cell. That is not a flaw in the fixture, it is the
        measurement the negative control exists to make.
    time_shift : Mapping of str to str, optional
        Per-instrument clock offsets to simulate, keyed by instrument name.
        The value is what the *manifest* will declare, and the file receives
        timestamps moved the other way, so ingestion must apply the
        correction to recover the generator's truth. Same shape as
        ``raw_units``: write the archive wrong, declare the fix, check that
        the answer comes back right.

    Returns
    -------
    pathlib.Path
        Path of the written manifest, ready to hand to
        :func:`tsara.config.loader.load_manifest`.

    Raises
    ------
    TsaraSyntheticError
        If ``path`` exists and is not a directory, if ``raw_units`` or
        ``qaqc_bounds`` names a species the dataset does not contain, if
        ``time_shift`` names an unknown instrument, or if a species collides
        with a boundary column name this exporter needs to write.
    """
    root = Path(path)
    if root.exists() and not root.is_dir():
        # Typed rather than left to the OS: `mkdir` would raise a bare
        # NotADirectoryError naming the `raw` subdirectory, which points at
        # the wrong path and does not say TSARA. `save_streams` refuses the
        # same mistake the same way.
        raise TsaraSyntheticError(f"Export path '{root}' exists and is not a directory.")

    scales = dict(raw_units or {})
    bounds = dict(qaqc_bounds or {})
    shifts = dict(time_shift or {})
    _check_species_exist(dataset.config, scales, bounds)
    _check_instruments_exist(dataset, shifts)
    if support_declaration == "reported":
        _check_boundary_columns_are_free(dataset)

    raw_dir = root / EXPORT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    sigma_columns = _reported_sigma_columns(dataset.config)
    for name in dataset.streams:
        _write_csv(
            dataset.observable(name),
            raw_dir / f"{name}.csv",
            scales,
            sigma_columns,
            label=_label_of(dataset.config, name),
            write_boundaries=support_declaration == "reported",
            shift_ns=_shift_ns(shifts.get(name)),
        )

    manifest = _build_manifest(dataset.config, raw_dir, scales, bounds, support_declaration, shifts)
    manifest_path = root / EXPORT_MANIFEST
    manifest_path.write_text(
        yaml.safe_dump(manifest.model_dump(mode="json", exclude_none=False), sort_keys=False),
        encoding="utf-8",
    )

    logger.info(
        "Exported %d synthetic stream(s) as raw CSV to %s with manifest %s.",
        len(dataset.streams),
        raw_dir,
        manifest_path,
    )
    return manifest_path


def _check_species_exist(
    config: SyntheticConfig,
    scales: Mapping[str, RawUnits],
    bounds: Mapping[str, tuple[float | None, float | None]],
) -> None:
    """Refuse a request naming a species the campaign does not have.

    A misspelled species would otherwise be accepted in silence and simply
    do nothing, which in a *test harness* is the worst possible outcome: the
    round trip would go on passing while checking less than it claims to.
    """
    known = {
        species for instrument in config.instruments.values() for species in instrument.species
    }
    unknown = sorted((set(scales) | set(bounds)) - known)
    if unknown:
        raise TsaraSyntheticError(
            f"Cannot export species {unknown}: campaign '{config.name}' declares {sorted(known)}."
        )


def _label_of(config: SyntheticConfig, name: str) -> str:
    """Return the support label of one emitted stream.

    The GPS track is not in ``config.instruments`` -- it is manufactured from
    the platform -- so it is answered separately rather than by a lookup that
    would raise on it.
    """
    instrument = config.instruments.get(name)
    return "mid" if instrument is None else instrument.support.label


def _width_of(config: SyntheticConfig, name: str) -> str:
    """Return the cell width of one emitted stream, as a duration string."""
    instrument = config.instruments.get(name)
    if instrument is None:
        return str(getattr(config.platform, "gps_rate", "1s"))
    return instrument.support.width or instrument.native_rate


def _method_of(config: SyntheticConfig, name: str) -> str:
    """Return the support method of one emitted stream."""
    instrument = config.instruments.get(name)
    return "point" if instrument is None else instrument.support.method


def _shift_ns(shift: str | None) -> int:
    """Return the nanoseconds a written timestamp must be moved by.

    The file receives ``truth - shift`` so that ingestion, adding the
    manifest's declared ``time_shift``, lands back on truth. Inverting here
    rather than in the manifest keeps the manifest holding the numbers a real
    one would, exactly as :class:`RawUnits` does for units.
    """
    import pandas as pd

    return 0 if shift is None else int(pd.Timedelta(shift).value)


def _check_instruments_exist(dataset: SyntheticDataset, shifts: Mapping[str, str]) -> None:
    """Refuse a clock offset for a stream that does not exist.

    Same reasoning as :func:`_check_species_exist`: silently ignoring a
    misspelled name would leave the harness passing while testing less than
    it claims.
    """
    unknown = sorted(set(shifts) - set(dataset.streams))
    if unknown:
        raise TsaraSyntheticError(
            f"Cannot shift the clock of {unknown}: this dataset has {sorted(dataset.streams)}."
        )


def _check_boundary_columns_are_free(dataset: SyntheticDataset) -> None:
    """Refuse to write boundary columns over a species of the same name.

    Species names are validated as identifiers, so ``time_start`` is a legal
    one. Writing the cell boundary into it would overwrite a measurement with
    a timestamp, and the round trip would then compare the wrong column
    against the answer key.
    """
    taken = sorted(
        {
            species
            for instrument in dataset.config.instruments.values()
            for species in instrument.species
        }
        & {START_COLUMN, STOP_COLUMN}
    )
    if taken:
        raise TsaraSyntheticError(
            f"Cannot export per-row cell boundaries: species {taken} would be "
            f"overwritten by the columns '{START_COLUMN}'/'{STOP_COLUMN}'. "
            "Rename the species, or export with support_declaration='declared'."
        )


def _write_csv(
    stream: xr.Dataset,
    target: Path,
    scales: Mapping[str, RawUnits],
    sigma_columns: Mapping[str, str],
    *,
    label: str,
    write_boundaries: bool,
    shift_ns: int,
) -> None:
    """Write one observable stream as a CSV with an ISO 8601 time column.

    The time column carries the timestamp the instrument's *label* implies,
    not the cell midpoint TSARA stores internally. That is what makes the
    fixture faithful: a start-labelled product writes start times, and
    ingestion has to be told (or assume) which it is looking at. Writing
    midpoints instead would quietly hand ingestion the answer.
    """
    import pandas as pd

    cells = stream[TIME_BOUNDS_VAR].values.astype("datetime64[ns]").astype(np.int64)
    midpoints = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    stamps = {"start": cells[:, 0], "end": cells[:, 1]}.get(label, midpoints)

    # Bounds dropped before `to_dataframe`: a two-dimensional coordinate turns
    # the frame into a (time, nv) MultiIndex and every row would appear twice.
    # They are re-emitted below as ordinary columns, which is how a file can
    # actually carry them.
    frame = stream.drop_vars(TIME_BOUNDS_VAR, errors="ignore").to_dataframe()
    # `to_dataframe` brings coordinates along as columns. Platform position
    # attached to a gas stream is a coordinate, not a measurement, and would
    # otherwise appear as an undeclared column. Dropped by asking the dataset
    # what its coordinates ARE rather than by name: on the GPS stream itself
    # `latitude`/`longitude` are the measurements, and dropping those by name
    # would export an empty file.
    frame = frame.drop(
        columns=[str(c) for c in stream.coords if c != TIME_COORD and c in frame.columns]
    )
    # Write the inverse of the declared conversion, so that ingestion
    # applying `raw * scale + offset` lands back on the generator's truth.
    # A reported-sigma column is scaled but NOT offset, mirroring
    # `convert_spread`: an offset shifts a measurement, never its spread.
    for species, raw in scales.items():
        if species in frame.columns:
            frame[species] = (frame[species] - raw.offset) / raw.scale
        sigma_column = sigma_columns.get(species)
        if sigma_column is not None and sigma_column in frame.columns:
            frame[sigma_column] = frame[sigma_column] / raw.scale

    def _iso(values: np.ndarray) -> list[str]:
        # Full precision, not `%f`: the generator's timestamp jitter puts real
        # information in the nanosecond digits, and strftime truncates at µs.
        # The shift is applied here so it reaches the boundary columns too --
        # a clock offset moves a cell, it does not resize it.
        moved = pd.DatetimeIndex((values - shift_ns).astype("datetime64[ns]"))
        return [stamp.isoformat() for stamp in moved]

    if write_boundaries:
        frame.insert(0, STOP_COLUMN, _iso(cells[:, 1]))
        if label != "start":
            # Only needed when the time column is not already the cell start;
            # a start column duplicating it would say nothing.
            frame.insert(0, START_COLUMN, _iso(cells[:, 0]))
    frame.insert(0, TIME_COORD, _iso(stamps))
    frame.to_csv(target, index=False)


def _reported_sigma_columns(config: SyntheticConfig) -> dict[str, str]:
    """Map each species to the sigma column it publishes, where it has one.

    Read from the config rather than from the generated stream, because
    ``report_as`` is a *configuration* fact: the generator writes the column
    under that name but records nothing in the variable's attrs, so asking
    the dataset would silently answer "no column" for every species.
    """
    columns: dict[str, str] = {}
    for instrument in config.instruments.values():
        for name, species in instrument.species.items():
            uncertainty = species.uncertainty
            if uncertainty is None:
                continue
            for component in (uncertainty.random, uncertainty.systematic):
                if component is not None and component.report_as is not None:
                    columns[name] = component.report_as
    return columns


def _support_block(
    config: SyntheticConfig, name: str, declaration: SupportDeclaration
) -> dict[str, Any] | None:
    """Return the manifest's ``support:`` block for one stream, if any.

    The three rungs, built from what the generator actually did. ``reported``
    names the boundary columns the writer emitted; ``declared`` restates the
    label, width and method; ``none`` returns None, leaving ingestion to
    assume and to say that it assumed.
    """
    if declaration == "none":
        return None
    label = _label_of(config, name)
    method = _method_of(config, name)
    if declaration == "declared":
        return {"label": label, "width": _width_of(config, name), "method": method}
    # Per-row boundaries: no label or width, which the schema refuses to
    # accept alongside them, since the columns already fix every cell.
    block: dict[str, Any] = {"method": method, "stop_column": STOP_COLUMN}
    if label != "start":
        block["start_column"] = START_COLUMN
    return block


def _build_manifest(
    config: SyntheticConfig,
    raw_dir: Path,
    scales: Mapping[str, RawUnits],
    bounds: Mapping[str, tuple[float | None, float | None]],
    support_declaration: SupportDeclaration,
    shifts: Mapping[str, str],
) -> Manifest:
    """Build the manifest that describes the files just written."""
    instruments: dict[str, Any] = {}
    for name, instrument in config.instruments.items():
        variables: dict[str, Any] = {}
        for species_name, species in instrument.species.items():
            variable: dict[str, Any] = {
                # Columns are written under their canonical names, so column
                # renaming is not exercised here; it is covered by the
                # ingest unit tests.
                "column": species_name,
                "role": species.role,
                "units": species.units,
                "circular": species.circular,
            }
            raw = scales.get(species_name)
            if raw is not None:
                # `units` names what is IN THE FILE, and `convert` takes it
                # to canonical -- the same shape a real manifest has, and
                # the reason the exporter wrote the inverse.
                variable["units"] = raw.from_unit
                variable["convert"] = {
                    "from_unit": raw.from_unit,
                    "to_unit": species.units,
                    "scale": raw.scale,
                    "offset": raw.offset,
                }
            limits = bounds.get(species_name)
            if limits is not None:
                # Declared in CANONICAL units deliberately: QA/QC runs after
                # conversion (METHODS §9.4), so a bound that only holds on
                # the converted values is exactly the assertion worth making.
                rule: dict[str, Any] = {"kind": "range"}
                if limits[0] is not None:
                    rule["min"] = limits[0]
                if limits[1] is not None:
                    rule["max"] = limits[1]
                variable["qaqc"] = [rule]
            if species.uncertainty is not None:
                # The seam Phase 2 built for exactly this: the generator's
                # true budget expressed as the manifest declaration that
                # should reproduce it.
                variable["uncertainty"] = species.uncertainty.to_manifest_uncertainty().model_dump(
                    mode="json", exclude_none=True
                )
            variables[species_name] = variable
        loader: dict[str, Any] = {
            "format": "csv",
            "path_template": f"{name}.csv",
            "time": {"column": TIME_COORD, "format": "iso8601"},
        }
        support = _support_block(config, name, support_declaration)
        if support is not None:
            loader["support"] = support
        entry: dict[str, Any] = {
            "description": f"Synthetic instrument '{name}' at {instrument.native_rate}.",
            "loader": loader,
            "variables": variables,
        }
        if name in shifts:
            entry["time_shift"] = shifts[name]
        instruments[name] = entry

    if isinstance(config.platform, MobileTrack):
        # The generator already emits the track as its own stream at its own
        # rate — the canonical multi-rate case — so it has been written like
        # any other. What is missing is only its *declaration*: the platform
        # is not in `config.instruments`, so nothing above described it.
        gps_instrument = config.platform.gps_instrument
        gps_loader: dict[str, Any] = {
            "format": "csv",
            "path_template": f"{gps_instrument}.csv",
            "time": {"column": TIME_COORD, "format": "iso8601"},
        }
        gps_support = _support_block(config, gps_instrument, support_declaration)
        if gps_support is not None:
            gps_loader["support"] = gps_support
        instruments[gps_instrument] = {
            "description": f"Synthetic GPS track at {config.platform.gps_rate}.",
            "loader": gps_loader,
            "variables": {
                LATITUDE_COORD: {
                    "column": LATITUDE_COORD,
                    "role": "gps_lat",
                    "units": "degrees_north",
                },
                LONGITUDE_COORD: {
                    "column": LONGITUDE_COORD,
                    "role": "gps_lon",
                    "units": "degrees_east",
                },
            },
        }
        platform: dict[str, Any] = {
            "kind": "mobile",
            "gps_instrument": gps_instrument,
            "lat_variable": LATITUDE_COORD,
            "lon_variable": LONGITUDE_COORD,
        }
    else:
        site = config.platform
        assert isinstance(site, StationarySite)  # noqa: S101 - the only other union member
        platform = {
            "kind": "stationary",
            "latitude": site.latitude,
            "longitude": site.longitude,
        }
        if site.altitude_m is not None:
            platform["altitude_m"] = site.altitude_m

    return Manifest.model_validate(
        {
            "name": config.name,
            "description": f"Manifest for synthetic dataset '{config.name}'.",
            "base_path": str(raw_dir),
            "platform": platform,
            "instruments": instruments,
        }
    )
