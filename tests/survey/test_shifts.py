"""Tests for one segment's shift (``tsara.survey.shifts``), on signals checkable by hand.

The sign is what matters most: a shift is added to the instrument's recorded
times (``corrected = recorded + shift``), so an instrument that shows the air
*later* than its partner needs a *negative* one (``docs/METHODS.md`` §12.4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.survey.shifts import (
    SecondsSeries,
    SegmentShift,
    TsaraSurveyError,
    cell_series,
    pooled_shift,
    refined_shift,
    seconds_series,
)

T0 = int(pd.Timestamp("2026-01-01T00:00:00").value // 1_000_000_000)


def _stream(seconds: np.ndarray, values: np.ndarray, **attrs: object) -> xr.Dataset:
    times = pd.to_datetime(seconds, unit="s").as_unit("ns")
    stream = xr.Dataset({"ch4": ("time", values)}, coords={"time": np.asarray(times)})
    stream["ch4"].attrs.update(attrs)
    stream.attrs["instrument"] = "test"
    return stream


def _plumes(n: int, seed: int = 0) -> np.ndarray:
    """Narrow plumes at random seconds on a flat background: a sharp correlation peak."""
    rng = np.random.default_rng(seed)
    signal = np.zeros(n)
    for centre in rng.choice(n, size=n // 40, replace=False):
        t = np.arange(n) - centre
        signal += rng.lognormal(3, 0.5) * np.exp(-0.5 * (t / 3.0) ** 2)
    return signal + rng.normal(0, 0.2, n)


# ---------------------------------------------------------------------------
# A variable on whole seconds
# ---------------------------------------------------------------------------


def test_times_are_floored_and_the_first_reading_of_a_second_is_kept() -> None:
    """Two readings in second 0 (at .2 and .9 s): the first, 1.0, is kept. The
    rest of the record sits at 7.0, so the running median is 7.0 and each
    wiggle is its value less 7; second 2 holds no reading."""
    seconds = np.r_[T0 + 0.2, T0 + 0.9, T0 + 1.5, T0 + np.arange(3.0, 200.0)]
    values = np.r_[1.0, 9.0, 8.0, np.full(197, 7.0)]
    series = seconds_series(_stream(seconds, values), "ch4", running_median_s=61)
    assert series.first_s == T0
    assert series.values[[0, 1, 3]].tolist() == [-6.0, 1.0, 0.0]
    assert np.isnan(series.values[2])


def test_a_flat_series_has_no_wiggles() -> None:
    seconds = T0 + np.arange(600.0)
    series = seconds_series(_stream(seconds, np.full(600, 1950.0)), "ch4", running_median_s=61)
    assert np.allclose(series.values, 0.0)


def test_a_sparse_record_still_gets_a_running_median() -> None:
    """A quarter of the window's possible readings is enough: a 5 s record
    holds 12 in a 61 s window, so 3 are needed, where a fixed 30 would leave it
    all NaN."""
    seconds = T0 + np.arange(0.0, 600.0, 5.0)
    series = seconds_series(_stream(seconds, np.arange(120.0)), "ch4", running_median_s=61)
    assert np.isfinite(series.values[::5]).all()


def test_missing_values_are_not_readings() -> None:
    values = np.ones(200)
    values[1] = np.nan
    series = seconds_series(_stream(T0 + np.arange(200.0), values), "ch4", running_median_s=61)
    assert np.isnan(series.values[1]) and np.isfinite(series.values[[0, 2, 3]]).all()


def test_a_variable_with_no_reading_gives_an_empty_series() -> None:
    series = seconds_series(
        _stream(T0 + np.arange(3.0), np.full(3, np.nan)), "ch4", running_median_s=61
    )
    assert len(series.values) == 0


def test_an_absent_variable_is_refused() -> None:
    with pytest.raises(TsaraSurveyError, match="has no variable 'co'"):
        seconds_series(_stream(T0 + np.arange(3.0), np.ones(3)), "co", running_median_s=61)


def test_a_circular_variable_is_refused() -> None:
    with pytest.raises(TsaraSurveyError, match="circular"):
        seconds_series(
            _stream(T0 + np.arange(3.0), np.ones(3), circular=1), "ch4", running_median_s=61
        )


def test_a_window_is_nan_outside_the_record() -> None:
    series = SecondsSeries(first_s=10, values=np.array([1.0, 2.0, 3.0]))
    window = series.window(8, 14)
    assert np.isnan(window[[0, 1, 5]]).all()
    assert window[2:5].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(series.window(20, 22)).all()


# ---------------------------------------------------------------------------
# The shift of one segment
# ---------------------------------------------------------------------------


def _shift(pairs: list[tuple[SecondsSeries, SecondsSeries]], **settings: float) -> SegmentShift:
    """One 1000 s segment, search 30 s, 200 pairs, decided at 0.6, tolerance 2 s."""
    options = {"search_s": 30, "min_pairs": 200, "min_r": 0.6, "tolerance_s": 2.0} | settings
    return pooled_shift(pairs, T0 + 100, T0 + 1100, **options)  # type: ignore[arg-type]


def _pair(shift_s: int, n: int = 1200, seed: int = 0) -> tuple[SecondsSeries, SecondsSeries]:
    """The partner shows the plumes at t, the instrument at t - shift: the shift puts it right."""
    truth = _plumes(n + 200, seed)
    partner = SecondsSeries(first_s=T0, values=truth[100 : 100 + n])
    instrument = SecondsSeries(first_s=T0, values=truth[100 + shift_s : 100 + shift_s + n])
    return instrument, partner


@pytest.mark.parametrize("shift_s", [17, 0, -5, -29])
def test_the_shift_puts_the_instrument_on_its_partners_clock(shift_s: int) -> None:
    shift = _shift([_pair(shift_s)])
    assert shift.shift_s == shift_s
    assert shift.r == pytest.approx(1.0)
    assert not shift.at_edge and not shift.pairs_disagree
    assert shift.n_pairs == 1000


def test_an_instrument_showing_the_air_later_needs_a_negative_shift() -> None:
    """The sign, from the definition rather than through ``_pair``: the
    instrument's file has each plume 5 s after the partner's."""
    truth = _plumes(1400)
    partner = SecondsSeries(first_s=T0, values=truth[100:1300])
    late = SecondsSeries(first_s=T0 + 5, values=truth[100:1300])
    assert _shift([(late, partner)]).shift_s == -5


def test_a_best_shift_at_the_search_edge_is_flagged() -> None:
    """A true shift just beyond the search shows as its peak's shoulder at the edge."""
    shift = _shift([_pair(-32)])
    assert shift.shift_s == -30
    assert shift.at_edge


def test_a_true_shift_far_beyond_the_search_finds_no_peak() -> None:
    """Measured while writing these tests: 40 s against a 30 s search does not
    show at the edge but as a weak peak somewhere inside, which min_r refuses."""
    shift = _shift([_pair(-40)])
    assert not shift.at_edge
    assert shift.r < 0.6


def test_too_few_paired_seconds_measure_nothing() -> None:
    shift = _shift([_pair(-5)], min_pairs=1001)
    assert np.isnan(shift.shift_s) and np.isnan(shift.r)
    assert shift.n_pairs == 0


def test_a_flat_segment_measures_nothing() -> None:
    flat = SecondsSeries(first_s=T0, values=np.zeros(1200))
    _, partner = _pair(-5)
    shift = _shift([(flat, partner)])
    assert np.isnan(shift.shift_s)


def test_a_tier_pools_its_pairs_curves() -> None:
    """Two pairs agreeing on -6 s pool to -6 s with their paired seconds summed;
    a pair with no data in the segment does not move the pooled curve."""
    one = _pair(-6, seed=1)
    two = _pair(-6, seed=2)
    empty = (SecondsSeries(first_s=T0, values=np.full(1200, np.nan)), one[1])
    shift = _shift([one, two, empty])
    assert shift.shift_s == -6
    assert shift.n_pairs == 2000
    assert shift.r == pytest.approx(1.0)
    assert not shift.pairs_disagree


def test_pairs_that_disagree_are_flagged_rather_than_averaged_between() -> None:
    """Found writing these tests: pairs at -4 s and -10 s, each certain alone,
    pool to a curve peaking at -7 s with r 0.76, which neither supports."""
    shift = _shift([_pair(-4, seed=3), _pair(-10, seed=4)])
    assert shift.shift_s == -7
    assert shift.r > 0.6
    assert shift.pairs_disagree


def test_a_pair_too_weak_to_decide_alone_cannot_veto() -> None:
    """A strong pair at -5 s and a weak one (mostly noise) peaking elsewhere."""
    strong = _pair(-5, seed=5)
    rng = np.random.default_rng(6)
    noise = SecondsSeries(first_s=T0, values=rng.normal(0, 1, 1200))
    shift = _shift([strong, (noise, strong[1])])
    assert shift.shift_s == -5
    assert not shift.pairs_disagree


# ---------------------------------------------------------------------------
# The shift on the readings' own cells
# ---------------------------------------------------------------------------

SEGMENT_NS = 1800 * 1_000_000_000
T0_NS = T0 * 1_000_000_000


def _air(
    starts: np.ndarray, stops: np.ndarray, centres: np.ndarray, amps: np.ndarray, width: float
) -> np.ndarray:
    """The exact mean over each cell of Gaussian plumes on a slow sine (seconds)."""
    from scipy.special import erf

    s2 = width * np.sqrt(2.0)
    area = amps * width * np.sqrt(np.pi / 2)
    plumes = (
        area * (erf((stops[:, None] - centres) / s2) - erf((starts[:, None] - centres) / s2))
    ).sum(axis=1)
    w = 2 * np.pi / 1200.0
    background = 5 * (np.cos(w * starts) - np.cos(w * stops)) / w
    return np.asarray((plumes + background) / (stops - starts))


def _cells_stream(
    width: float,
    phase: float,
    shift: float,
    *,
    seed: int = 0,
    plume_width: float = 4.0,
    noise: float = 0.0,
    sigma: str | None = None,
) -> xr.Dataset:
    """Contiguous cells over half an hour plus margins, recorded on a clock ``shift`` corrects.

    Every reading is the exact mean of the air over its own (true) cell, plus
    white noise; its times are written as ``true - shift``, so that adding the
    shift puts them right. ``sigma`` names the provenance of a stated random
    sigma equal to that noise, or None for none.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-300, 2100, 53)
    amps = rng.lognormal(3, 0.5, 53)
    # Each analyzer its own noise: two drawing the same would share it, and agree exactly.
    # Seeded on minus the shift, as when METHODS §12.4's numbers were measured.
    noise_rng = np.random.default_rng(
        [seed, round(width * 1000), round(phase * 100), round(-shift * 100) + 1000]
    )
    starts = -300 + phase + width * np.arange(int(2400 / width))
    stops = starts + width
    values = _air(starts, stops, centres, amps, plume_width)
    values = values + noise_rng.normal(0, noise, len(starts))

    def to_ns(seconds: np.ndarray) -> np.ndarray:
        return np.asarray(T0_NS + np.round((seconds - shift) * 1e9)).astype("datetime64[ns]")

    stream = xr.Dataset(
        {
            "ch4": ("time", values),
            "time_bnds": (("time", "nv"), np.stack([to_ns(starts), to_ns(stops)], axis=1)),
        },
        coords={"time": to_ns((starts + stops) / 2)},
    )
    if sigma is not None:
        stream["sigma_rand_ch4"] = ("time", np.full(len(starts), noise))
        stream["ch4"].attrs["uncertainty_provenance_random"] = sigma
    return stream


def _refine(
    own: xr.Dataset, ref: xr.Dataset, coarse_s: float, resolution_s: float, min_pairs: int = 300
) -> tuple[float, bool, bool]:
    pair = (
        cell_series(own, "ch4", running_median_s=121),
        cell_series(ref, "ch4", running_median_s=121),
    )
    return refined_shift(
        [pair],
        T0_NS,
        T0_NS + SEGMENT_NS,
        coarse_s=coarse_s,
        resolution_ns=round(resolution_s * 1e9),
        min_pairs=min_pairs,
    )


#: Two 1 s analyzers whose cells line up in true time, two half a cell apart,
#: and a 0.993 s analyzer against a 2 s one (the Aeris and the Picarro).
GEOMETRIES = [(1.0, 0.0, 1.0, 0.0), (1.0, 0.0, 1.0, 0.5), (0.993, 0.37, 2.0, 0.0)]


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("shift", [-7.3, -7.65, 4.25])
def test_the_shift_on_cells_is_exact_on_exact_cell_means(
    geometry: tuple[float, float, float, float], shift: float
) -> None:
    """On readings that are each the exact mean of the air over their cell, the
    shift comes back exactly, at every phase of the two sets of cells: the
    whole-second peak, the shift rounded up to a whole second, is all it is
    given."""
    own_width, own_phase, ref_width, ref_phase = geometry
    own = _cells_stream(own_width, own_phase, shift)
    ref = _cells_stream(ref_width, ref_phase, 0.0)
    found, at_edge, removed = _refine(own, ref, coarse_s=np.ceil(shift), resolution_s=0.05)
    assert found == pytest.approx(shift, abs=1e-9)
    assert not at_edge
    assert not removed


def test_a_shift_between_steps_comes_back_to_the_nearest() -> None:
    own, ref = _cells_stream(1.0, 0.0, -7.27), _cells_stream(1.0, 0.5, 0.0)
    assert _refine(own, ref, coarse_s=-7.0, resolution_s=0.1)[0] == pytest.approx(-7.3)


def test_at_one_second_the_shift_is_the_nearest_second_not_the_peaks() -> None:
    """A clock 7.7 s late needs -7.7 s: the whole seconds put it at -7, the cells say -8."""
    own, ref = _cells_stream(1.0, 0.0, -7.7), _cells_stream(1.0, 0.0, 0.0)
    assert _refine(own, ref, coarse_s=-7.0, resolution_s=1.0)[0] == -8.0


def test_a_best_shift_at_either_end_of_the_window_is_flagged() -> None:
    """Looked for within two seconds of a peak 4 s from the truth, the best shift
    is the window's nearest end, and it says so."""
    own, ref = _cells_stream(1.0, 0.0, -7.3), _cells_stream(1.0, 0.5, 0.0)
    found, at_edge, _ = _refine(own, ref, coarse_s=-3.0, resolution_s=0.1)
    assert found == pytest.approx(-5.0)
    assert at_edge


def test_too_few_fully_covered_cells_give_no_shift() -> None:
    own, ref = _cells_stream(1.0, 0.0, -7.3), _cells_stream(1.0, 0.5, 0.0)
    found, at_edge, _ = _refine(own, ref, coarse_s=-7.0, resolution_s=0.1, min_pairs=5000)
    assert np.isnan(found) and not at_edge


def test_a_narrower_partner_is_joined_onto_the_wider_instruments_cells() -> None:
    """The 2 s instrument against a 0.993 s partner: the join runs the other
    way round from the pair's order, and the shift keeps its sign."""
    own, ref = _cells_stream(2.0, 0.0, -6.4), _cells_stream(0.993, 0.37, 0.0)
    assert _refine(own, ref, coarse_s=-6.0, resolution_s=0.1)[0] == pytest.approx(-6.4)


def test_cell_series_takes_stated_sigma_only() -> None:
    """A sigma counts only when declared or reported for every finite reading."""
    stream = _cells_stream(1.0, 0.0, 0.0, noise=1.0, sigma="declared")
    stream["ch4"].values[5] = np.nan
    series = cell_series(stream, "ch4", running_median_s=121)
    assert len(series.cells) == len(series.values) == len(stream.time) - 1
    assert series.sigma is not None and np.all(series.sigma == 1.0)
    for provenance in ("reported", "unknown", "zero"):
        stream["ch4"].attrs["uncertainty_provenance_random"] = provenance
        stated = cell_series(stream, "ch4", running_median_s=121).sigma is not None
        assert stated == (provenance == "reported")
    stream["ch4"].attrs["uncertainty_provenance_random"] = "declared"
    stream["sigma_rand_ch4"].values[7] = np.nan
    assert cell_series(stream, "ch4", running_median_s=121).sigma is None


def test_a_stream_without_cells_gets_its_cadence_centred_on_each_time() -> None:
    seconds = T0 + np.arange(0.0, 600.0, 2.0)
    series = cell_series(_stream(seconds, _plumes(300)), "ch4", running_median_s=61)
    assert (series.cells.width_ns == 2_000_000_000).all()
    assert series.cells.midpoint_ns[0] == T0_NS


def test_stated_noise_is_removed_and_the_shift_stops_chasing_it() -> None:
    """Monte Carlo. Where two 1 s analyzers' cells line up in true time, the
    join of one onto the other's cells averages two readings at every shift but
    the true one, so their noise falls there and r rises: the shift is pulled
    off. With the readings' sigma stated, that noise is taken out of the joined
    series' variance and the pull goes (METHODS.md §12.4)."""
    kept: list[float] = []
    removed: list[float] = []
    common: dict[str, Any] = {"plume_width": 12.0, "noise": 1.0}
    for seed in range(3):
        for shift in (-7.0, -7.2, -7.4, -7.6, -7.8):
            for sigma, errors in ((None, kept), ("declared", removed)):
                own = _cells_stream(1.0, 0.0, shift, seed=seed, sigma=sigma, **common)
                ref = _cells_stream(1.0, 0.0, 0.0, seed=seed, sigma=sigma, **common)
                found, _, said = _refine(own, ref, coarse_s=np.ceil(shift), resolution_s=0.05)
                assert said == (sigma is not None)
                errors.append(found - shift)
    rms_kept = float(np.sqrt(np.mean(np.square(kept))))
    rms_removed = float(np.sqrt(np.mean(np.square(removed))))
    # Measured: 0.371 s kept, 0.085 s removed (METHODS.md §12.4).
    assert rms_kept > 0.15
    assert rms_removed < rms_kept / 2


def test_a_tracer_without_a_finite_reading_has_no_cells() -> None:
    stream = _cells_stream(1.0, 0.0, 0.0)
    stream["ch4"].values[:] = np.nan
    series = cell_series(stream, "ch4", running_median_s=121)
    assert len(series.cells) == len(series.values) == 0


def test_a_stated_sigma_larger_than_the_joined_series_spread_leaves_no_shift() -> None:
    """Taking out more noise than the joined series varies by leaves nothing to
    correlate: no shift rather than a made-up one."""
    own = _cells_stream(1.0, 0.0, -7.3)
    ref = _cells_stream(1.0, 0.5, 0.0, noise=1.0, sigma="declared")
    ref["sigma_rand_ch4"].values[:] = 1e4
    found, _, removed = _refine(own, ref, coarse_s=-7.0, resolution_s=0.1)
    assert np.isnan(found) and removed


def _true_seconds(stream: xr.Dataset) -> np.ndarray:
    """Each reading's true midpoint in seconds, for streams built by _cells_stream."""
    return np.asarray(
        (stream["time"].values - stream["time"].values[0]) / np.timedelta64(1, "s")
        + float(stream["time_bnds"].values[0, 1] - stream["time_bnds"].values[0, 0]) / 2e9
        - 300.0
    )


def test_a_cell_only_partly_covered_does_not_count() -> None:
    """A 2 s analyzer against a 1 s one whose readings in the first half of each
    2 s cell are missing from 300 to 900 s: there the mean of what is left
    describes the cell's second half, half a second late. Counted, those cells
    would pull the shift; only cells covered whole count, so it stays exact."""
    own = _cells_stream(2.0, 0.0, -6.4)
    ref = _cells_stream(1.0, 0.0, 0.0)
    true = _true_seconds(ref)
    first_half = np.floor(true) % 2 == 0
    ref["ch4"].values[(true > 300) & (true < 900) & first_half] = np.nan
    found, at_edge, _ = _refine(own, ref, coarse_s=-6.0, resolution_s=0.05)
    assert found == pytest.approx(-6.4, abs=1e-9)
    assert not at_edge


def test_readings_too_sparse_for_a_median_cover_nothing() -> None:
    """From 300 to 900 s the joined analyzer keeps two readings in ten: each
    pair covers a cell whole, but 121 s hold too few readings for a median, so
    they have no wiggle and must not count; nor must the instrument's own
    readings without one (one in ten kept from 1000 to 1200 s)."""
    own = _cells_stream(1.0, 0.0, -7.3)
    ref = _cells_stream(1.0, 0.5, 0.0)
    true_ref, true_own = _true_seconds(ref), _true_seconds(own)
    index = np.arange(len(true_ref))
    ref["ch4"].values[(true_ref > 300) & (true_ref < 900) & (index % 10 > 1)] = np.nan
    index = np.arange(len(true_own))
    own["ch4"].values[(true_own > 1000) & (true_own < 1200) & (index % 10 != 0)] = np.nan
    for stream in (own, ref):
        kept = len(cell_series(stream, "ch4", running_median_s=121).cells)
        assert kept < np.isfinite(stream["ch4"].values).sum()
    found, at_edge, _ = _refine(own, ref, coarse_s=-7.0, resolution_s=0.05)
    assert found == pytest.approx(-7.3, abs=1e-9)
    assert not at_edge


def test_the_wider_partners_cells_are_those_that_see_the_instruments_segment() -> None:
    """The partner has the wider cells, so its cells are the ones compared:
    those seeing the instrument's segment once it is moved by the shift onto
    the partner's clock. The instrument records nothing outside its segment, so
    cells taken without moving them (-60 s here) would lose 30 to nothing
    joined."""
    own = _cells_stream(1.0, 0.0, -60.3)
    own = own.sel(
        time=(own.time >= np.datetime64(T0_NS, "ns"))
        & (own.time < np.datetime64(T0_NS + SEGMENT_NS, "ns"))
    )
    ref = _cells_stream(2.0, 0.0, 0.0)
    found, _, _ = _refine(own, ref, coarse_s=-60.0, resolution_s=0.1, min_pairs=890)
    assert found == pytest.approx(-60.3)


def test_the_shifts_are_multiples_of_the_resolution_whatever_the_peak() -> None:
    """A step of 0.3 s that does not divide the window: the shifts tried are
    multiples of 0.3 s, so a clock needing -7.2 s is found exactly."""
    own, ref = _cells_stream(1.0, 0.0, -7.2), _cells_stream(1.0, 0.5, 0.0)
    found = _refine(own, ref, coarse_s=-7.0, resolution_s=0.3)[0]
    assert found == pytest.approx(-7.2, abs=1e-9)


def test_cells_are_ordered_by_start_where_the_cadence_changes() -> None:
    """A 1 s record followed by a 5 s one: the first 5 s cell, centred 1.6 s
    after the last 1 s reading, starts before it. The cells come back ordered
    by start, each with its own reading's wiggle."""
    times = np.r_[np.arange(0.0, 200.0), np.arange(200.6, 400.0, 5.0)]
    widths = np.r_[np.full(200, 1.0), np.full(40, 5.0)]

    def to_ns(seconds: np.ndarray) -> np.ndarray:
        return np.asarray(T0_NS + np.round(seconds * 1e9)).astype("datetime64[ns]")

    stream = xr.Dataset(
        {
            "ch4": ("time", times.copy()),
            "time_bnds": (
                ("time", "nv"),
                np.stack([to_ns(times - widths / 2), to_ns(times + widths / 2)], axis=1),
            ),
        },
        coords={"time": to_ns(times)},
    )
    series = cell_series(stream, "ch4", running_median_s=121)
    assert (np.diff(series.cells.start_ns) >= 0).all()
    swapped = np.flatnonzero(series.cells.width_ns == 5_000_000_000)[0]
    assert series.cells.midpoint_ns[swapped] == T0_NS + 200_600_000_000
