"""Assembling a complete synthetic dataset from a :class:`SyntheticConfig`.

This is the orchestrator: it draws the physical events once, builds each
instrument's clock, renders every species, injects error, and stamps out the
answer key. The output is a :class:`SyntheticDataset` — per-instrument
``xarray.Dataset`` streams at native rates plus a
:class:`~tsara.synthetic.plumes.GroundTruth` catalog — which is exactly the
shape Phase 3 ingestion will produce from real files, so every later phase
can be developed and tested before real data is readable.

Ordering matters and is load-bearing
------------------------------------
Events are scheduled **before** any instrument is rendered. A plume is one
physical release: the same leak must appear on the 1 Hz analyzer and the
10 Hz analyzer with consistent amplitudes and a consistent ratio. Drawing
per-instrument (or per-species) would silently destroy the cross-species
covariance that TSARA exists to measure, and every regression test built on
such data would be measuring an artifact.

Emitted variables
-----------------
Each stream carries, per species:

* ``<species>`` — the observable. **The only variable the analysis pipeline
  may consume.**
* ``truth_background_<species>``, ``truth_enhancement_<species>`` — the exact
  decomposition, so a baseline estimator can be scored directly against what
  it was trying to recover.
* ``truth_sigma_rand_<species>``, ``truth_sigma_sys_<species>`` — the true
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
from tsara.core.timebase import epoch_ns as _epoch_ns
from tsara.core.timebase import epoch_s as _epoch_s
from tsara.core.timebase import timestamp_epoch_ns as _stamp_ns
from tsara.core.timebase import timestamp_epoch_s as _stamp_s
from tsara.core.timebase import to_utc_naive as _to_utc_naive
from tsara.synthetic.background import TsaraSyntheticError, render_background
from tsara.synthetic.config import (
    TRUTH_PREFIX,
    InstrumentSpec,
    MobileTrack,
    StationarySite,
    SyntheticConfig,
)
from tsara.synthetic.noise import apply_uncertainty, quantize
from tsara.synthetic.platform import build_track
from tsara.synthetic.plumes import (
    GroundTruth,
    GroundTruthEvent,
    RealizedEvent,
    schedule_events,
)

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
__all__ = ["TRUTH_PREFIX", "SyntheticDataset", "generate"]


@dataclass(frozen=True, eq=False)
class SyntheticDataset:
    """A generated dataset: native-rate streams, the answer key, and its config.

    Attributes
    ----------
    streams : dict of str to xarray.Dataset
        One Dataset per instrument, on that instrument's own irregular native
        timestamps (METHODS.md §1.1). For a mobile platform the GPS stream
        appears here too, under ``platform.gps_instrument``.
    ground_truth : GroundTruth
        Every injected event, one row per (event, species).
    config : SyntheticConfig
        The configuration that produced this dataset, carried alongside so a
        saved bundle is self-describing and reproducible.
    """

    streams: dict[str, xr.Dataset]
    ground_truth: GroundTruth
    config: SyntheticConfig

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
        keep = [str(name) for name in stream.data_vars if not str(name).startswith(TRUTH_PREFIX)]
        return stream[keep]

    def save(self, path: str | Path) -> Path:
        """Write this dataset as a TSARA bundle directory.

        Delegates to :func:`tsara.synthetic.bundle.save_synthetic`; see there
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
        from tsara.synthetic.bundle import save_synthetic

        return save_synthetic(self, path)

    @classmethod
    def load(cls, path: str | Path) -> SyntheticDataset:
        """Read a TSARA bundle written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Bundle directory.

        Returns
        -------
        SyntheticDataset
            The round-tripped dataset.
        """
        from tsara.synthetic.bundle import load_synthetic

        return load_synthetic(path)


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
        Real-data profiles keyed by name, required only if any species uses a
        :class:`~tsara.synthetic.config.BootstrapBackground`. Passed at call
        time rather than embedded in the config so that real-data-derived
        arrays can never be serialized into a config file.

    Returns
    -------
    SyntheticDataset
        Streams, ground truth, and the originating config.

    Raises
    ------
    TsaraSyntheticError
        If an instrument's configuration yields no samples, or a referenced
        profile was not supplied.
    """
    import pandas as pd

    rng = np.random.default_rng(config.seed)
    start = pd.Timestamp(config.start)
    end = start + pd.Timedelta(config.duration)

    # 1. Physical events first — shared across every instrument and species.
    events = schedule_events(config, rng)
    logger.info(
        "Scheduled %d plume events (%d top-level, %d nested) across %d sources.",
        len(events),
        sum(1 for e in events if e.parent_event_id is None),
        sum(1 for e in events if e.parent_event_id is not None),
        len(config.sources),
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

    # 3. Instruments.
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
            events=events,
            rng=rng,
            profiles=profiles,
            track=track,
        )
        streams[instrument_name] = stream
        truth_rows.extend(rows)

    ground_truth = GroundTruth(events=tuple(truth_rows))
    logger.info("Generated %d streams with %d ground-truth rows.", len(streams), len(ground_truth))
    return SyntheticDataset(streams=streams, ground_truth=ground_truth, config=config)


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
    # Never empty: `duration` is validated strictly positive, so the range
    # always contains at least the start instant even when `native_rate` is
    # coarser than the whole record. Only dropouts can empty it, which is
    # checked after they are applied.
    times = _to_utc_naive(pd.date_range(start=start, end=end, freq=native_rate, inclusive="left"))

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
    events: list[RealizedEvent],
    rng: np.random.Generator,
    profiles: Mapping[str, RealDataProfile] | None,
    track: tuple[pd.DatetimeIndex, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None,
) -> tuple[xr.Dataset, list[GroundTruthEvent]]:
    """Render every species on one instrument and collect its truth rows.

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
    events : list of RealizedEvent
        All scheduled events (filtered per species inside).
    rng : numpy.random.Generator
        Source of randomness.
    profiles : mapping of str to RealDataProfile or None
        Real-data profiles for bootstrap backgrounds.
    track : tuple or None
        ``(gps_times, latitude, longitude)`` for a mobile platform.

    Returns
    -------
    xarray.Dataset
        The instrument's stream.
    list of GroundTruthEvent
        Truth rows for the species this instrument measures.
    """
    import xarray as xr

    data_vars: dict[str, tuple[str, npt.NDArray[np.float64], dict[str, object]]] = {}
    truth_rows: list[GroundTruthEvent] = []

    for species_name, species in instrument.species.items():
        background = render_background(species.background, times, rng, profiles)

        # Plume injection: only gases receive enhancements. Each event's
        # contribution is rendered on its own support window, which keeps
        # this O(events x window) rather than O(events x record) and yields
        # the per-event sampled peak needed for the answer key.
        enhancement = np.zeros(len(times), dtype=np.float64)
        event_rows: list[GroundTruthEvent] = []
        if species.role == "gas":
            enhancement, event_rows = _inject_plumes(
                config=config,
                instrument_name=instrument_name,
                species_name=species_name,
                times=times,
                background=background,
                events=events,
                track=track,
            )
            truth_rows.extend(event_rows)

        truth_signal = background + enhancement
        applied = apply_uncertainty(truth_signal, species.uncertainty, times, rng)
        observable = applied.values

        if species.quantization is not None:
            observable = quantize(observable, species.quantization)
        if species.circular:
            # Wrap after everything else: noise on a value near 0 or 360 must
            # be able to cross the discontinuity, which is precisely the case
            # circular statistics exist to handle (METHODS.md §1.5).
            observable = np.mod(observable, 360.0)

        attrs: dict[str, object] = {
            "units": species.units,
            "role": species.role,
            "circular": int(species.circular),
        }
        attrs.update(applied.scalars)
        if species.quantization is not None:
            attrs["quantization"] = float(species.quantization)

        data_vars[species_name] = ("time", observable, attrs)
        data_vars[f"{TRUTH_PREFIX}background_{species_name}"] = (
            "time",
            background,
            {"units": species.units, "description": "True background (answer key)."},
        )
        data_vars[f"{TRUTH_PREFIX}enhancement_{species_name}"] = (
            "time",
            enhancement,
            {"units": species.units, "description": "True plume enhancement (answer key)."},
        )
        if applied.sigma_rand is not None:
            data_vars[f"{TRUTH_PREFIX}sigma_rand_{species_name}"] = (
                "time",
                applied.sigma_rand,
                {"units": species.units, "description": "True random 1-sigma (answer key)."},
            )
        if applied.sigma_sys is not None:
            data_vars[f"{TRUTH_PREFIX}sigma_sys_{species_name}"] = (
                "time",
                applied.sigma_sys,
                {"units": species.units, "description": "True systematic 1-sigma (answer key)."},
            )
        for column, values in applied.reported.items():
            data_vars[column] = (
                "time",
                values,
                {
                    "units": species.units,
                    "description": f"Instrument-reported 1-sigma for {species_name}.",
                },
            )

    dataset = xr.Dataset(
        data_vars={
            name: (dims, values, attrs) for name, (dims, values, attrs) in data_vars.items()
        },
        coords={"time": times},
        attrs=_stream_attrs(config, instrument_name, instrument),
    )
    _attach_platform_coords(dataset, config, times, track)
    return dataset, truth_rows


def _inject_plumes(
    *,
    config: SyntheticConfig,
    instrument_name: str,
    species_name: str,
    times: pd.DatetimeIndex,
    background: npt.NDArray[np.float64],
    events: list[RealizedEvent],
    track: tuple[pd.DatetimeIndex, npt.NDArray[np.float64], npt.NDArray[np.float64]] | None,
) -> tuple[npt.NDArray[np.float64], list[GroundTruthEvent]]:
    """Add every event's contribution for one species and build its truth rows.

    Parameters
    ----------
    config : SyntheticConfig
        Full run configuration (for platform coordinates).
    instrument_name : str
        Instrument measuring this species.
    species_name : str
        Species being rendered.
    times : pandas.DatetimeIndex
        Native timestamps.
    background : numpy.ndarray
        Already-rendered background, used to record the true baseline under
        each peak.
    events : list of RealizedEvent
        All scheduled events; those not emitting this species are skipped.
    track : tuple or None
        Mobile track, if any.

    Returns
    -------
    numpy.ndarray
        Total enhancement on ``times``.
    list of GroundTruthEvent
        One row per event that emits this species.
    """
    import pandas as pd

    enhancement = np.zeros(len(times), dtype=np.float64)
    rows: list[GroundTruthEvent] = []

    epoch_ns = _epoch_ns(times)
    epoch_s = _epoch_s(times)

    for event in events:
        amplitude = event.amplitudes.get(species_name)
        if amplitude is None:
            continue

        center = event.species_center(species_name)
        kernel = event.kernel
        start_time = center - pd.Timedelta(seconds=kernel.support_before_s)
        end_time = center + pd.Timedelta(seconds=kernel.support_after_s)
        peak_time = event.species_peak_time(species_name)

        # Restrict to the support window: searchsorted keeps this cheap even
        # with tens of thousands of events over a long record.
        lo = int(np.searchsorted(epoch_ns, _stamp_ns(start_time), side="left"))
        hi = int(np.searchsorted(epoch_ns, _stamp_ns(end_time), side="right"))

        sampled_peak = float("nan")
        if hi > lo:
            dt_s = epoch_s[lo:hi] - _stamp_s(center)
            contribution = amplitude * kernel.evaluate(dt_s)
            enhancement[lo:hi] += contribution
            sampled_peak = float(contribution.max())

        rows.append(
            GroundTruthEvent(
                event_id=event.event_id,
                parent_event_id=event.parent_event_id,
                source_name=event.source_name,
                species=species_name,
                instrument=instrument_name,
                reference_species=event.reference_species,
                start_time=start_time,
                peak_time=peak_time,
                end_time=end_time,
                true_amplitude=float(amplitude),
                sampled_peak_amplitude=sampled_peak,
                true_baseline_at_peak=float(np.interp(_stamp_s(peak_time), epoch_s, background)),
                true_ratio_to_reference=float(event.ratios[species_name]),
                **_event_position(config, peak_time, track),
            )
        )

    return enhancement, rows


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
    import xarray as xr

    return xr.Dataset(
        data_vars={
            "latitude": (
                "time",
                latitude,
                {"units": "degrees_north", "role": "gps_lat"},
            ),
            "longitude": (
                "time",
                longitude,
                {"units": "degrees_east", "role": "gps_lon"},
            ),
        },
        coords={"time": times},
        attrs={
            "tsara_version": __version__,
            "tsara_stage": "synthetic",
            "synthetic_config_name": config.name,
            "instrument": getattr(config.platform, "gps_instrument", "gps"),
            "platform_kind": config.platform.kind,
        },
    )


def _stream_attrs(
    config: SyntheticConfig, instrument_name: str, instrument: InstrumentSpec
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

    Returns
    -------
    dict
        Attribute mapping.
    """
    return {
        "tsara_version": __version__,
        "tsara_stage": "synthetic",
        "synthetic_config_name": config.name,
        "synthetic_seed": config.seed,
        "instrument": instrument_name,
        "native_rate": instrument.native_rate,
        "platform_kind": config.platform.kind,
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
        dataset.coords["latitude"] = config.platform.latitude
        dataset.coords["longitude"] = config.platform.longitude
        if config.platform.altitude_m is not None:
            dataset.coords["altitude"] = config.platform.altitude_m
        return

    assert track is not None
    track_times, latitude, longitude = track
    lat, lon = positions_at(times, track_times, latitude, longitude)
    dataset.coords["latitude"] = ("time", lat)
    dataset.coords["longitude"] = ("time", lon)
