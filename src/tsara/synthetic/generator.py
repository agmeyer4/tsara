"""Assembling a complete synthetic dataset from a :class:`SyntheticConfig`.

This is the orchestrator. It realizes the atmosphere once
(:mod:`tsara.synthetic.atmosphere`), builds the platform, and then lets each
instrument sample that atmosphere: build its clock and cells, take the true
signal over them, add its own error, and record what it could see of every
event. The output is a :class:`SyntheticDataset` — per-instrument
``xarray.Dataset`` streams at native rates, a
:class:`~tsara.synthetic.plumes.GroundTruth` catalog, and the
:class:`~tsara.synthetic.atmosphere.Atmosphere` itself — which is exactly the
shape ingestion produces from real files, so every later phase can be
developed and tested against it.

Ordering matters and is load-bearing
------------------------------------
The single random generator is consumed in a fixed order: plume events, then
each field's background, then the platform, then each instrument's clock and
each measurement's noise.

* **Events before everything else.** A plume is one physical release: the
  same leak must appear on the 1 Hz analyzer and the 10 Hz analyzer with
  consistent amplitudes and a consistent ratio. Drawing per instrument would
  silently destroy the cross-species covariance TSARA exists to measure.
* **The atmosphere before the platform and instruments,** so the air a seed
  produces does not depend on who measures it, and a saved bundle can rebuild
  it from its config without regenerating any stream.
* **A background with no stochastic term draws nothing,** so moving
  backgrounds out of the instruments left every noise realization of such a
  configuration exactly where it was: the streams are byte-identical to the
  ones the previous generator produced.

Emitted variables
-----------------
Each stream carries, per measured field, under the measurement's variable
name (the field's own name unless it declared another):

* ``<name>`` — the observable. **The only variable the analysis pipeline may
  consume.** Its ``field`` attribute names the field it measures.
* ``truth_background_<name>``, ``truth_enhancement_<name>`` — the exact
  decomposition of the true signal over this instrument's cells, so a
  baseline estimator can be scored directly against what it was trying to
  recover. The atmosphere holds the same truth as a function of time.
* ``truth_sigma_rand_<name>``, ``truth_sigma_sys_<name>`` — the true
  per-point error budget.
* any configured ``report_as`` column, under its exact configured name and
  deliberately unprefixed, since that is a raw-file column a manifest will
  reference (and may be biased relative to the truth).

Everything beginning ``truth_`` is metadata about the answer, not data;
filter it out with ``[v for v in ds.data_vars if not v.startswith("truth_")]``
to obtain the pipeline-visible view.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tsara import __version__
from tsara.core.geodesy import positions_at
from tsara.core.naming import (
    ALTITUDE_COORD,
    LATITUDE_COORD,
    LONGITUDE_COORD,
    TIME_COORD,
    sigma_rand_name,
    sigma_sys_name,
)
from tsara.core.support import CellBounds, attach_time_bounds, support_attrs
from tsara.core.timebase import NS_PER_S
from tsara.core.timebase import epoch_ns as _epoch_ns
from tsara.core.timebase import epoch_s as _epoch_s
from tsara.core.timebase import timestamp_epoch_s as _stamp_s
from tsara.core.timebase import to_utc_naive as _to_utc_naive
from tsara.core.timebase import to_utc_naive_stamp as _to_utc_naive_stamp
from tsara.synthetic.atmosphere import Atmosphere, CellGrid, realize_atmosphere
from tsara.synthetic.background import TsaraSyntheticError
from tsara.synthetic.config import (
    TRUTH_PREFIX,
    InstrumentSpec,
    MobileTrack,
    StationarySite,
    SyntheticConfig,
)
from tsara.synthetic.noise import apply_uncertainty, quantize
from tsara.synthetic.platform import build_track
from tsara.synthetic.plumes import GroundTruth, GroundTruthEvent

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd
    import xarray as xr

    from tsara.synthetic.profiling import RealDataProfile

logger = logging.getLogger(__name__)

#: Prefix marking variables that describe the answer rather than the data.
#: Defined in :mod:`tsara.synthetic.config` (the schema layer reserves it so
#: a ``report_as`` column can never shadow the answer key) and re-exported
#: here, where it is used.
__all__ = ["SyntheticDataset", "TRUTH_PREFIX", "generate"]


@dataclass(frozen=True, eq=False)
class SyntheticDataset:
    """A generated dataset: native-rate streams, the answer key, the air, its config.

    Attributes
    ----------
    streams : dict of str to xarray.Dataset
        One Dataset per instrument, on that instrument's own irregular native
        timestamps (METHODS.md §1.1). For a mobile platform the GPS stream
        appears here too, under ``platform.gps_instrument``.
    ground_truth : GroundTruth
        Every injected event, one row per (event, measured variable).
    config : SyntheticConfig
        The configuration that produced this dataset, carried alongside so a
        saved bundle is self-describing and reproducible.
    atmosphere : Atmosphere or None
        The true atmosphere every stream sampled, queryable at any time.
        Always present on a freshly generated dataset. A loaded bundle rebuilds
        it from its config, and has None only when a bootstrap background's
        real-data profile was not supplied to the loader.
    """

    streams: dict[str, xr.Dataset]
    ground_truth: GroundTruth
    config: SyntheticConfig
    atmosphere: Atmosphere | None = None

    def observable(self, instrument: str) -> xr.Dataset:
        """Return one stream with all ``truth_`` variables removed.

        The pipeline-visible view: what ingestion would have produced from
        real files, with no privileged information about the answer.

        Parameters
        ----------
        instrument : str
            Stream name.

        Returns
        -------
        xarray.Dataset
            The stream without truth variables.

        Raises
        ------
        KeyError
            If no such stream exists.
        """
        if instrument not in self.streams:
            raise KeyError(f"No stream named '{instrument}'; available: {sorted(self.streams)}.")
        stream = self.streams[instrument]
        # Dropping the answer key rather than selecting the observables:
        # selecting by name returns only the coordinates those variables
        # need, which quietly discards the cell boundaries. This view is
        # meant to be exactly what ingestion would have produced from real
        # files, and a real stream has cells.
        answer_key = [name for name in stream.data_vars if str(name).startswith(TRUTH_PREFIX)]
        return stream.drop_vars(answer_key)

    def save(self, path: str | Path) -> Path:
        """Write this dataset as a TSARA bundle directory.

        Delegates to :func:`tsara.synthetic.bundle.save_bundle`; see there
        for the on-disk layout.

        Parameters
        ----------
        path : str or pathlib.Path
            Bundle directory to create.

        Returns
        -------
        pathlib.Path
            The bundle directory.
        """
        from tsara.synthetic.bundle import save_bundle

        return save_bundle(self, path)

    @classmethod
    def load(
        cls, path: str | Path, profiles: Mapping[str, RealDataProfile] | None = None
    ) -> SyntheticDataset:
        """Read a TSARA bundle written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Bundle directory.
        profiles : mapping of str to RealDataProfile, optional
            Real-data profiles, needed only to rebuild the atmosphere of a
            campaign with a bootstrap background.

        Returns
        -------
        SyntheticDataset
            The round-tripped dataset.
        """
        from tsara.synthetic.bundle import load_bundle

        return load_bundle(path, profiles=profiles)


def generate(
    config: SyntheticConfig,
    profiles: Mapping[str, RealDataProfile] | None = None,
) -> SyntheticDataset:
    """Generate a complete synthetic dataset.

    Parameters
    ----------
    config : SyntheticConfig
        Full specification of the dataset to manufacture.
    profiles : mapping of str to RealDataProfile, optional
        Real-data profiles keyed by name, required only if any field uses a
        :class:`~tsara.synthetic.config.BootstrapBackground`. Passed at call
        time rather than embedded in the config so that real-data-derived
        arrays can never be serialized into a config file.

    Returns
    -------
    SyntheticDataset
        Streams, ground truth, the atmosphere they sampled, and the
        originating config.

    Raises
    ------
    TsaraSyntheticError
        If an instrument's configuration yields no samples, or a referenced
        profile was not supplied.
    """
    import pandas as pd

    rng = np.random.default_rng(config.seed)
    # Normalized here as well as inside `_build_times`, so that `start`/`end`
    # are already on TSARA's one time representation everywhere they are used
    # rather than only where a clock happens to be built.
    start = _to_utc_naive_stamp(pd.Timestamp(config.start))
    end = start + pd.Timedelta(config.duration)

    # 1. The air: every plume event, then every field's background, before
    #    anything that measures them. See the module docstring for why the
    #    order is load-bearing.
    atmosphere = realize_atmosphere(config, rng, profiles)
    events = atmosphere.events
    logger.info(
        "Scheduled %d plume events (%d top-level, %d nested) across %d sources.",
        len(events),
        sum(1 for e in events if e.parent_event_id is None),
        sum(1 for e in events if e.parent_event_id is not None),
        len(config.atmosphere.sources),
    )

    # 2. Platform. A mobile track becomes its own stream and a position
    #    lookup used to geolocate every event in the answer key.
    streams: dict[str, xr.Dataset] = {}
    track: tuple[pd.DatetimeIndex, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None
    track = None
    if isinstance(config.platform, MobileTrack):
        gps_times = _build_times(
            start, end, config.platform.gps_rate, None, None, rng, config.platform.gps_instrument
        )
        latitude, longitude = build_track(config.platform, gps_times, rng)
        track = (gps_times, latitude, longitude)
        streams[config.platform.gps_instrument] = _build_gps_stream(
            config, gps_times, latitude, longitude
        )

    # 3. Instruments, each sampling the one atmosphere.
    truth_rows: list[GroundTruthEvent] = []
    for instrument_name, instrument in config.instruments.items():
        times = _build_times(
            start,
            end,
            instrument.native_rate,
            instrument.timestamp_jitter,
            instrument.dropouts,
            rng,
            instrument_name,
        )
        stream, rows = _render_instrument(
            config=config,
            instrument_name=instrument_name,
            instrument=instrument,
            times=times,
            atmosphere=atmosphere,
            rng=rng,
            track=track,
        )
        streams[instrument_name] = stream
        truth_rows.extend(rows)

    ground_truth = GroundTruth(events=tuple(truth_rows))
    logger.info("Generated %d streams with %d ground-truth rows.", len(streams), len(ground_truth))
    return SyntheticDataset(
        streams=streams, ground_truth=ground_truth, config=config, atmosphere=atmosphere
    )


# ---------------------------------------------------------------------------
# Time axis construction
# ---------------------------------------------------------------------------


def _build_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
    native_rate: str,
    jitter: str | None,
    dropouts: object,
    rng: np.random.Generator,
    label: str,
) -> pd.DatetimeIndex:
    """Build one instrument's native timestamps, with jitter and dropouts.

    Real instruments do not deliver perfect grids, and an architecture that
    claims to handle irregular native timestamps (METHODS.md §1.1) needs data
    that actually is irregular. Two independent departures from a perfect
    grid are applied:

    * **jitter** — each timestamp is nudged by a uniform draw, bounded by the
      schema at under half the nominal interval so the clock can never run
      backwards.
    * **dropouts** — outages *delete* samples rather than NaN-filling them,
      because that is what a logger that stops writing produces, and because
      the resulting gaps are what rolling-window valid-fraction logic must
      cope with.

    Parameters
    ----------
    start, end : pandas.Timestamp
        Record bounds; ``end`` is exclusive.
    native_rate : str
        Nominal sampling interval.
    jitter : str or None
        Timestamp jitter amplitude.
    dropouts : DropoutSpec or None
        Outage configuration.
    rng : numpy.random.Generator
        Source of randomness.
    label : str
        Stream name, for error messages.

    Returns
    -------
    pandas.DatetimeIndex
        Native timestamps, strictly increasing.

    Raises
    ------
    TsaraSyntheticError
        If the configuration produces no samples at all.
    """
    import pandas as pd

    from tsara.synthetic.config import DropoutSpec

    # Normalize to tz-naive UTC immediately. TSARA is UTC internally, and a
    # tz-aware axis would (a) make tz-aware and tz-naive configs produce
    # different streams for the same instants, and (b) fail to encode to
    # netCDF at save time. Doing it here means every downstream stage, and
    # every persisted file, sees one consistent time representation.
    #
    # `as_unit("ns")` pins the resolution as well as the timezone. Without it
    # the unit is inherited from `start`: a config whose start came from a
    # `datetime.datetime` yields microseconds, while the jitter branch below
    # casts to nanoseconds explicitly — so one dataset could carry streams of
    # two different resolutions depending on which instruments declared
    # jitter, and netCDF (which stores ns) would silently change the dtype on
    # every save/load. Nanoseconds span 1677-2262, comfortably beyond any
    # atmospheric record.
    #
    # Never empty: `duration` is validated strictly positive, so the range
    # always contains at least the start instant even when `native_rate` is
    # coarser than the whole record. Only dropouts can empty it, which is
    # checked after they are applied.
    times = _to_utc_naive(
        pd.date_range(start=start, end=end, freq=native_rate, inclusive="left")
    ).as_unit("ns")

    if jitter is not None:
        jitter_ns = float(pd.Timedelta(jitter).value)
        offsets = rng.uniform(-jitter_ns, jitter_ns, size=len(times))
        times = pd.DatetimeIndex(
            (_epoch_ns(times) + offsets.astype(np.int64)).astype("datetime64[ns]")
        )

    if isinstance(dropouts, DropoutSpec):
        times = _apply_dropouts(times, dropouts, rng, label)

    if len(times) == 0:
        raise TsaraSyntheticError(
            f"Instrument '{label}': dropouts removed every sample; reduce rate_per_day or duration."
        )
    return times


def _apply_dropouts(
    times: pd.DatetimeIndex,
    dropouts: object,
    rng: np.random.Generator,
    label: str,
) -> pd.DatetimeIndex:
    """Delete samples falling inside randomly placed outages.

    Outage count is Poisson over the record; each outage's length is
    exponential with the configured mean, so occasional long dropouts occur
    naturally rather than every gap being identical.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        Candidate timestamps.
    dropouts : DropoutSpec
        Outage configuration.
    rng : numpy.random.Generator
        Source of randomness.
    label : str
        Stream name, for logging.

    Returns
    -------
    pandas.DatetimeIndex
        Surviving timestamps.
    """
    import pandas as pd

    from tsara.synthetic.config import DropoutSpec

    assert isinstance(dropouts, DropoutSpec)  # narrowed by the caller

    span_days = (times[-1] - times[0]).total_seconds() / 86_400.0
    n_outages = int(rng.poisson(dropouts.rate_per_day * span_days))
    if n_outages == 0:
        return times

    mean_duration_s = float(pd.Timedelta(dropouts.duration).total_seconds())
    epoch_s = _epoch_s(times)
    # Onsets may begin up to one mean duration *before* the record starts: an
    # instrument can already be down when logging begins, and restricting
    # onsets to the record would leave the first samples artificially
    # immune to dropouts.
    onsets = rng.uniform(epoch_s[0] - mean_duration_s, epoch_s[-1], size=n_outages)
    durations = rng.exponential(mean_duration_s, size=n_outages)

    drop = np.zeros(len(times), dtype=bool)
    for onset, duration in zip(onsets, durations):
        drop |= (epoch_s >= onset) & (epoch_s < onset + duration)

    logger.debug(
        "Instrument %r: %d dropouts removed %d of %d samples.",
        label,
        n_outages,
        int(drop.sum()),
        len(times),
    )
    return times[~drop]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_instrument(
    *,
    config: SyntheticConfig,
    instrument_name: str,
    instrument: InstrumentSpec,
    times: pd.DatetimeIndex,
    atmosphere: Atmosphere,
    rng: np.random.Generator,
    track: tuple[pd.DatetimeIndex, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None,
) -> tuple[xr.Dataset, list[GroundTruthEvent]]:
    """Sample every field one instrument measures, and collect its truth rows.

    Everything about the air comes from ``atmosphere``; everything added here
    belongs to the instrument: its cells, its noise, its rounding, and which
    events its cells could see.

    Parameters
    ----------
    config : SyntheticConfig
        Full run configuration.
    instrument_name : str
        Name of the instrument being rendered.
    instrument : InstrumentSpec
        Its configuration.
    times : pandas.DatetimeIndex
        Its native timestamps.
    atmosphere : Atmosphere
        The realized air every instrument samples.
    rng : numpy.random.Generator
        Source of randomness for this instrument's noise.
    track : tuple or None
        ``(gps_times, latitude, longitude)`` for a mobile platform.

    Returns
    -------
    xarray.Dataset
        The instrument's stream.
    list of GroundTruthEvent
        Truth rows for the variables this instrument writes.
    """
    import pandas as pd
    import xarray as xr

    # Cells first: everything below is evaluated per cell, and for the default
    # point/mid configuration a cell is centred on its own timestamp, so its
    # single evaluation instant is the timestamp itself and this whole path
    # reduces exactly to sampling the atmosphere at each sample.
    grid = _build_cells(times, instrument)
    bounds = grid.cells
    midpoints = pd.DatetimeIndex(bounds.midpoint_ns.astype("datetime64[ns]"), name=TIME_COORD)
    midpoint_s = bounds.midpoint_ns / NS_PER_S

    data_vars: dict[str, tuple[str, npt.NDArray[np.float64], dict[str, object]]] = {}
    truth_rows: list[GroundTruthEvent] = []

    for field_name, measurement in instrument.measures.items():
        field = config.atmosphere.fields[field_name]
        name = instrument.variable_name(field_name)
        # The true signal over this instrument's cells. The same call serves
        # `Atmosphere.mean_over`, which is why a noise-free stream equals the
        # atmosphere exactly rather than approximately.
        truth = atmosphere.over_cells(field_name, grid)
        background = truth.background
        enhancement = truth.enhancement

        # What this instrument could see of each event, as an answer-key row.
        # Only gases carry plumes, so any other field has no peaks at all.
        for peak in truth.peaks:
            event = peak.event
            start_time, end_time = event.species_window(field_name)
            peak_time = event.species_peak_time(field_name)
            truth_rows.append(
                GroundTruthEvent(
                    event_id=event.event_id,
                    parent_event_id=event.parent_event_id,
                    source_name=event.source_name,
                    species=name,
                    field=field_name,
                    instrument=instrument_name,
                    reference_species=event.reference_species,
                    start_time=start_time,
                    peak_time=peak_time,
                    end_time=end_time,
                    true_amplitude=float(event.amplitudes[field_name]),
                    sampled_peak_amplitude=peak.sampled_peak,
                    # The background as these cells describe it, at the peak.
                    true_baseline_at_peak=float(
                        np.interp(_stamp_s(peak_time), midpoint_s, background)
                    ),
                    true_ratio_to_reference=float(event.ratios[field_name]),
                    **_event_position(config, peak_time, track),
                )
            )

        truth_signal = background + enhancement
        # Noise is drawn at the CELL, never on the fine grid. That keeps a
        # declared sigma meaning "the spread of the numbers this instrument
        # publishes", which is what an instrument specification states and
        # what `to_manifest_uncertainty` promises ingestion it can reproduce.
        applied = apply_uncertainty(truth_signal, measurement.uncertainty, midpoints, rng)
        observable = applied.values

        if measurement.quantization is not None:
            observable = quantize(observable, measurement.quantization)
        if field.circular:
            # Wrap after everything else: noise on a value near 0 or 360 must
            # be able to cross the discontinuity, which is precisely the case
            # circular statistics exist to handle (METHODS.md §1.5).
            observable = np.mod(observable, 360.0)

        attrs: dict[str, object] = {
            "units": field.units,
            "role": field.role,
            # The quantity measured, which a renamed variable does not spell.
            # Ingestion writes the same attribute from a manifest, so the two
            # producers' streams stay interchangeable (METHODS.md §1.6).
            "field": field_name,
            "circular": int(field.circular),
        }
        attrs.update(applied.scalars)
        if measurement.quantization is not None:
            attrs["quantization"] = float(measurement.quantization)

        data_vars[name] = (TIME_COORD, observable, attrs)
        data_vars[f"{TRUTH_PREFIX}background_{name}"] = (
            TIME_COORD,
            background,
            {"units": field.units, "description": "True background (answer key)."},
        )
        data_vars[f"{TRUTH_PREFIX}enhancement_{name}"] = (
            TIME_COORD,
            enhancement,
            {"units": field.units, "description": "True plume enhancement (answer key)."},
        )
        if applied.sigma_rand is not None:
            data_vars[f"{TRUTH_PREFIX}{sigma_rand_name(name)}"] = (
                TIME_COORD,
                applied.sigma_rand,
                {"units": field.units, "description": "True random 1-sigma (answer key)."},
            )
        if applied.sigma_sys is not None:
            data_vars[f"{TRUTH_PREFIX}{sigma_sys_name(name)}"] = (
                TIME_COORD,
                applied.sigma_sys,
                {"units": field.units, "description": "True systematic 1-sigma (answer key)."},
            )
        for column, values in applied.reported.items():
            data_vars[column] = (
                TIME_COORD,
                values,
                {
                    "units": field.units,
                    "description": f"Instrument-reported 1-sigma for {name}.",
                },
            )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={TIME_COORD: midpoints},
        attrs=_stream_attrs(config, instrument_name, instrument, bounds),
    )
    attach_time_bounds(dataset, bounds, instrument.support.method)
    _attach_platform_coords(dataset, config, midpoints, track)
    return dataset, truth_rows


def _build_cells(times: pd.DatetimeIndex, instrument: InstrumentSpec) -> CellGrid:
    """Build one instrument's cells and the instants its true signal is averaged at.

    One instant per cell for ``point`` -- the cell midpoint, which for the
    default centred label is the original timestamp exactly -- and
    ``subsamples`` midpoint-rule instants for ``mean``
    (:meth:`~tsara.synthetic.atmosphere.CellGrid.build`). That exactness is
    what lets one code path serve both methods.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        The instrument's native timestamps.
    instrument : InstrumentSpec
        Its configuration, including the support to manufacture.

    Returns
    -------
    CellGrid
        One cell per timestamp, with its evaluation instants.
    """
    import pandas as pd

    support = instrument.support
    width_ns = int(pd.Timedelta(support.width or instrument.native_rate).value)
    bounds = CellBounds.from_label(_epoch_ns(times), width_ns, support.label)
    return CellGrid.build(bounds, 1 if support.method == "point" else support.subsamples)


def _event_position(
    config: SyntheticConfig,
    peak_time: pd.Timestamp,
    track: tuple[pd.DatetimeIndex, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None,
) -> dict[str, float | None]:
    """Return the platform coordinates to stamp on a ground-truth event.

    For a stationary site this is the fixed position; for a mobile platform
    it is the interpolated track position at the event's peak — which is what
    a real mobile catalog records, since a drive-by measurement localizes the
    *encounter*, not the source.

    Parameters
    ----------
    config : SyntheticConfig
        Run configuration.
    peak_time : pandas.Timestamp
        Event peak.
    track : tuple or None
        Mobile track, if any.

    Returns
    -------
    dict
        ``{"latitude": ..., "longitude": ...}``.

    Notes
    -----
    These keys are :class:`~tsara.synthetic.plumes.GroundTruthEvent` *field
    names* -- the result is splatted into its constructor -- not stream
    coordinates, so they are deliberately spelled out rather than taken from
    :mod:`tsara.core.naming`. Binding them to ``LATITUDE_COORD`` would tie a
    dataclass signature to a netCDF coordinate name and break the catalog if
    either were ever renamed independently.
    """
    import pandas as pd

    if isinstance(config.platform, StationarySite):
        return {
            "latitude": config.platform.latitude,
            "longitude": config.platform.longitude,
        }
    assert track is not None  # a MobileTrack always builds one
    track_times, latitude, longitude = track
    lat, lon = positions_at(pd.DatetimeIndex([peak_time]), track_times, latitude, longitude)
    return {"latitude": float(lat[0]), "longitude": float(lon[0])}


def _build_gps_stream(
    config: SyntheticConfig,
    times: pd.DatetimeIndex,
    latitude: npt.NDArray[np.float64],
    longitude: npt.NDArray[np.float64],
) -> xr.Dataset:
    """Package a mobile track as its own instrument stream.

    Parameters
    ----------
    config : SyntheticConfig
        Run configuration.
    times : pandas.DatetimeIndex
        GPS timestamps.
    latitude, longitude : numpy.ndarray
        Track coordinates.

    Returns
    -------
    xarray.Dataset
        The GPS stream, with ``gps_lat``/``gps_lon`` roles matching the
        manifest vocabulary.
    """
    import pandas as pd
    import xarray as xr

    # A track is a sequence of position fixes, so `point` is the honest
    # method; the cells exist so that a later stage can bin the track onto a
    # gas instrument's cells by overlap like any other stream.
    width_ns = int(pd.Timedelta(getattr(config.platform, "gps_rate", "1s")).value)
    bounds = CellBounds.from_label(_epoch_ns(times), width_ns, "mid")

    stream = xr.Dataset(
        data_vars={
            LATITUDE_COORD: (
                TIME_COORD,
                latitude,
                {"units": "degrees_north", "role": "gps_lat", "field": LATITUDE_COORD},
            ),
            LONGITUDE_COORD: (
                TIME_COORD,
                longitude,
                {"units": "degrees_east", "role": "gps_lon", "field": LONGITUDE_COORD},
            ),
        },
        coords={TIME_COORD: times},
        attrs={
            "tsara_version": __version__,
            "tsara_stage": "synthetic",
            "synthetic_config_name": config.name,
            "instrument": getattr(config.platform, "gps_instrument", "gps"),
            "platform_kind": config.platform.kind,
            **support_attrs(
                label="mid",
                width_ns=width_ns,
                coverage=bounds.coverage_fraction,
                label_source="declared",
                width_source="declared",
                method_source="declared",
            ),
        },
    )
    attach_time_bounds(stream, bounds, "point")
    return stream


def _stream_attrs(
    config: SyntheticConfig,
    instrument_name: str,
    instrument: InstrumentSpec,
    bounds: CellBounds,
) -> dict[str, object]:
    """Build the self-describing attrs every stream carries.

    Saved products self-describe (CLAUDE.md §5): package version, the config
    that produced them, and enough provenance to tell synthetic data apart
    from real data at a glance — the last of which matters most, because a
    synthetic file mistaken for a measurement is a scientific hazard.

    Parameters
    ----------
    config : SyntheticConfig
        Run configuration.
    instrument_name : str
        Stream name.
    instrument : InstrumentSpec
        Instrument configuration.
    bounds : CellBounds
        The stream's cells, for the coverage diagnostic.

    Returns
    -------
    dict
        Attribute mapping.
    """
    import pandas as pd

    support = instrument.support
    width_ns = int(pd.Timedelta(support.width or instrument.native_rate).value)
    return {
        "tsara_version": __version__,
        "tsara_stage": "synthetic",
        "synthetic_config_name": config.name,
        "synthetic_seed": config.seed,
        "instrument": instrument_name,
        "native_rate": instrument.native_rate,
        "platform_kind": config.platform.kind,
        # Everything about this stream's support was stated by the config that
        # manufactured it, so every field is `declared` -- the generator never
        # has to guess about data it invented.
        **support_attrs(
            label=support.label,
            width_ns=width_ns,
            coverage=bounds.coverage_fraction,
            label_source="declared",
            width_source="declared",
            method_source="declared",
        ),
        "description": (
            "SYNTHETIC DATA generated by tsara.synthetic — not a measurement. "
            "Variables prefixed 'truth_' are the answer key and must not be "
            "consumed by analysis code."
        ),
    }


def _attach_platform_coords(
    dataset: xr.Dataset,
    config: SyntheticConfig,
    times: pd.DatetimeIndex,
    track: tuple[pd.DatetimeIndex, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None,
) -> None:
    """Attach lat/lon coordinates to a rendered stream, in place.

    Stationary platforms get scalar coordinates (one position, globally);
    mobile platforms get the track interpolated onto this instrument's own
    clock. The asymmetry mirrors the manifest's platform union exactly.

    Parameters
    ----------
    dataset : xarray.Dataset
        Stream to modify.
    config : SyntheticConfig
        Run configuration.
    times : pandas.DatetimeIndex
        The stream's timestamps.
    track : tuple or None
        Mobile track, if any.
    """
    if isinstance(config.platform, StationarySite):
        dataset.coords[LATITUDE_COORD] = config.platform.latitude
        dataset.coords[LONGITUDE_COORD] = config.platform.longitude
        if config.platform.altitude_m is not None:
            dataset.coords[ALTITUDE_COORD] = config.platform.altitude_m
        return

    assert track is not None
    track_times, latitude, longitude = track
    lat, lon = positions_at(times, track_times, latitude, longitude)
    dataset.coords[LATITUDE_COORD] = (TIME_COORD, lat)
    dataset.coords[LONGITUDE_COORD] = (TIME_COORD, lon)
