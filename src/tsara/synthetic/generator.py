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
from typing import TYPE_CHECKING, Any

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
from tsara.core.timebase import timestamp_epoch_ns as _stamp_ns
from tsara.core.timebase import timestamp_epoch_s as _stamp_s
from tsara.core.timebase import to_utc_naive as _to_utc_naive
from tsara.core.timebase import to_utc_naive_stamp as _to_utc_naive_stamp
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
__all__ = ["SyntheticDataset", "TRUTH_PREFIX", "generate"]


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
        from tsara.synthetic.bundle import load_bundle

        return load_bundle(path)


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
    # Normalized here as well as inside `_build_times`, so that `start`/`end`
    # are already on TSARA's one time representation everywhere they are used
    # rather than only where a clock happens to be built.
    start = _to_utc_naive_stamp(pd.Timestamp(config.start))
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
    import pandas as pd
    import xarray as xr

    # Cells first: everything below is rendered per cell, and for the default
    # point/mid configuration a cell is centred on its own timestamp, so the
    # fine grid collapses to the timestamps themselves and this whole path
    # reduces exactly to evaluating truth at each sample.
    bounds, fine_ns = _build_cells(times, instrument)
    n_sub = int(fine_ns.shape[1])
    midpoints = pd.DatetimeIndex(bounds.midpoint_ns.astype("datetime64[ns]"), name=TIME_COORD)
    fine_times = pd.DatetimeIndex(fine_ns.reshape(-1).astype("datetime64[ns]"))

    data_vars: dict[str, tuple[str, npt.NDArray[np.float64], dict[str, object]]] = {}
    truth_rows: list[GroundTruthEvent] = []

    for species_name, species in instrument.species.items():
        # Rendered on the fine grid and averaged, not evaluated once per cell:
        # a mean instrument must actually average the background's wander,
        # otherwise `mean` would be a label rather than an operation.
        background = (
            render_background(species.background, fine_times, rng, profiles)
            .reshape(-1, n_sub)
            .mean(axis=1)
        )

        # Plume injection: only gases receive enhancements. Each event's
        # contribution is rendered on its own support window, which keeps
        # this O(events x window) rather than O(events x record) and yields
        # the per-event sampled peak needed for the answer key.
        enhancement = np.zeros(len(bounds), dtype=np.float64)
        event_rows: list[GroundTruthEvent] = []
        if species.role == "gas":
            enhancement, event_rows = _inject_plumes(
                config=config,
                instrument_name=instrument_name,
                species_name=species_name,
                bounds=bounds,
                fine_ns=fine_ns,
                background=background,
                events=events,
                track=track,
            )
            truth_rows.extend(event_rows)

        truth_signal = background + enhancement
        # Noise is drawn at the CELL, never on the fine grid. That keeps a
        # declared sigma meaning "the spread of the numbers this instrument
        # publishes", which is what an instrument specification states and
        # what `to_manifest_uncertainty` promises ingestion it can reproduce.
        applied = apply_uncertainty(truth_signal, species.uncertainty, midpoints, rng)
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

        data_vars[species_name] = (TIME_COORD, observable, attrs)
        data_vars[f"{TRUTH_PREFIX}background_{species_name}"] = (
            TIME_COORD,
            background,
            {"units": species.units, "description": "True background (answer key)."},
        )
        data_vars[f"{TRUTH_PREFIX}enhancement_{species_name}"] = (
            TIME_COORD,
            enhancement,
            {"units": species.units, "description": "True plume enhancement (answer key)."},
        )
        if applied.sigma_rand is not None:
            data_vars[f"{TRUTH_PREFIX}{sigma_rand_name(species_name)}"] = (
                TIME_COORD,
                applied.sigma_rand,
                {"units": species.units, "description": "True random 1-sigma (answer key)."},
            )
        if applied.sigma_sys is not None:
            data_vars[f"{TRUTH_PREFIX}{sigma_sys_name(species_name)}"] = (
                TIME_COORD,
                applied.sigma_sys,
                {"units": species.units, "description": "True systematic 1-sigma (answer key)."},
            )
        for column, values in applied.reported.items():
            data_vars[column] = (
                TIME_COORD,
                values,
                {
                    "units": species.units,
                    "description": f"Instrument-reported 1-sigma for {species_name}.",
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


def _build_cells(times: pd.DatetimeIndex, instrument: InstrumentSpec) -> tuple[CellBounds, Any]:
    """Build one instrument's cells and the fine grid used to average them.

    The fine grid is ``subsamples`` points per cell at the midpoints of equal
    sub-intervals -- the midpoint rule, whose error falls as the inverse
    square of the count. Integer arithmetic throughout, so that with one
    subsample the single point lands exactly on the cell midpoint and, for
    the default centred label, exactly on the original timestamp. That
    exactness is what lets one code path serve both methods without changing
    any existing output by a single bit.

    Parameters
    ----------
    times : pandas.DatetimeIndex
        The instrument's native timestamps.
    instrument : InstrumentSpec
        Its configuration, including the support to manufacture.

    Returns
    -------
    CellBounds
        One cell per timestamp.
    numpy.ndarray
        Fine-grid epoch nanoseconds, shape ``(n_cells, subsamples)``.
    """
    import pandas as pd

    support = instrument.support
    width_ns = int(pd.Timedelta(support.width or instrument.native_rate).value)
    bounds = CellBounds.from_label(_epoch_ns(times), width_ns, support.label)
    n_sub = 1 if support.method == "point" else support.subsamples
    offsets = ((2 * np.arange(n_sub, dtype=np.int64) + 1) * width_ns) // (2 * n_sub)
    return bounds, bounds.start_ns[:, None] + offsets[None, :]


def _inject_plumes(
    *,
    config: SyntheticConfig,
    instrument_name: str,
    species_name: str,
    bounds: CellBounds,
    fine_ns: Any,
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
    bounds : CellBounds
        The instrument's cells, one per emitted row.
    fine_ns : numpy.ndarray
        Fine-grid epoch nanoseconds, shape ``(n_cells, subsamples)``. Each
        event is evaluated on this grid and averaged per cell, which is what
        makes a narrow plume inside a wide cell come out diluted rather than
        at full height -- the physically right answer, and the one that makes
        an unresolvable event visible in the answer key.
    background : numpy.ndarray
        Already-averaged background per cell, used to record the true
        baseline under each peak.
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

    enhancement = np.zeros(len(bounds), dtype=np.float64)
    rows: list[GroundTruthEvent] = []

    n_sub = int(fine_ns.shape[1])
    epoch_s = fine_ns.reshape(-1) / NS_PER_S
    midpoint_s = bounds.midpoint_ns / NS_PER_S
    # Events are located against the CELL BOUNDARIES, never against the
    # flattened fine grid. The flat grid looks sortable and is not: timestamp
    # jitter is permitted up to just under half the sampling interval, so
    # full-width cells centred on jittered stamps overlap, and the grid then
    # descends. Measured on a 1 Hz stream with 0.4 s jitter, 151 of 300
    # adjacent cells overlap and the flat grid has 146 descending steps —
    # enough for a binary search over it to select the wrong cells, by up to
    # 0.8 % of a plume's peak in the harshest configuration the schema allows.
    #
    # Cell starts are sorted whatever the jitter, since they are a constant
    # shift of an increasing clock, so the running maximum of the stops makes
    # a valid lower bound. Same pattern as `bin_onto_cells`.
    cell_start = bounds.start_ns
    running_stop = np.maximum.accumulate(bounds.stop_ns)

    for event in events:
        amplitude = event.amplitudes.get(species_name)
        if amplitude is None:
            continue

        center = event.species_center(species_name)
        kernel = event.kernel
        start_time = center - pd.Timedelta(seconds=kernel.support_before_s)
        end_time = center + pd.Timedelta(seconds=kernel.support_after_s)
        peak_time = event.species_peak_time(species_name)

        # Every cell that OVERLAPS the support window, which is the unit the
        # instrument emits: a cell partly inside the window is partly affected
        # by the event.
        #
        # This is a wider selection than the timestamps-in-window rule it
        # replaces, and it costs nothing, because `PlumeKernel.evaluate`
        # returns exactly zero outside the support rather than a very small
        # number. So the extra cells contribute exact zeros and the default
        # point path is unchanged bit for bit -- verified against a hash taken
        # before any of this phase was written.
        lo = int(np.searchsorted(running_stop, _stamp_ns(start_time), side="right"))
        hi = int(np.searchsorted(cell_start, _stamp_ns(end_time), side="left"))

        sampled_peak = float("nan")
        if hi > lo:
            dt_s = epoch_s[lo * n_sub : hi * n_sub] - _stamp_s(center)
            contribution = amplitude * kernel.evaluate(dt_s)
            per_cell = contribution.reshape(-1, n_sub).mean(axis=1)
            enhancement[lo:hi] += per_cell
            # The peak as this instrument could actually see it. For a wide
            # cell that is the diluted peak, not the true amplitude, which is
            # exactly the quantity a later stage needs to decide whether an
            # event was resolvable at all.
            sampled_peak = float(per_cell.max())

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
                true_baseline_at_peak=float(np.interp(_stamp_s(peak_time), midpoint_s, background)),
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
                {"units": "degrees_north", "role": "gps_lat"},
            ),
            LONGITUDE_COORD: (
                TIME_COORD,
                longitude,
                {"units": "degrees_east", "role": "gps_lon"},
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
