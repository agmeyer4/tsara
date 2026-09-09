"""Tests for temporal-support arithmetic.

Everything here is integer nanoseconds, and several of these tests exist
because the obvious float or "distance to the next row" spelling is wrong in
a way that produces plausible numbers rather than an error.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.core.bundle import TIME_ENCODING, pin_time_encoding
from tsara.core.naming import (
    BOUNDS_ATTR,
    BOUNDS_DIM,
    CELL_METHODS_ATTR,
    SUPPORT_LABEL_SOURCE_ATTR,
    SUPPORT_WIDENED_ATTR,
    SUPPORT_WIDTH_ATTR,
    TIME_BOUNDS_VAR,
    TIME_COORD,
)
from tsara.core.support import (
    CellBounds,
    SupportLabel,
    SupportMethod,
    TsaraSupportError,
    attach_time_bounds,
    bin_onto_cells,
    cell_methods_value,
    check_bounds_intact,
    declared_bounds_name,
    ensure_time_bounds,
    nominal_cadence_ns,
    overlap_pairs,
    support_attrs,
)

SECOND = 1_000_000_000


def _times(start: int, step: int, n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64) * step + start


# ---------------------------------------------------------------------------
# cell_methods_value
# ---------------------------------------------------------------------------


def test_cell_methods_values_are_the_cf_spellings() -> None:
    assert cell_methods_value("mean") == "time: mean"
    assert cell_methods_value("point") == "time: point"


# ---------------------------------------------------------------------------
# nominal_cadence_ns
# ---------------------------------------------------------------------------


def test_cadence_of_a_regular_record() -> None:
    assert nominal_cadence_ns(_times(0, SECOND, 100)) == SECOND


def test_cadence_needs_two_timestamps() -> None:
    assert nominal_cadence_ns(np.array([5], dtype=np.int64)) is None
    assert nominal_cadence_ns(np.array([], dtype=np.int64)) is None


def test_cadence_of_repeated_timestamps_is_undefined() -> None:
    """All-duplicate timestamps give no positive interval to measure."""
    assert nominal_cadence_ns(np.full(10, 7, dtype=np.int64)) is None


def test_a_gap_does_not_move_the_cadence() -> None:
    """The whole point of a median: one huge hole must not widen every cell.

    Modelled on a real GPS record whose file spans a campaign but whose
    samples are 10 s apart.
    """
    times = np.concatenate(
        [_times(0, 10 * SECOND, 500), _times(23 * 86400 * SECOND, 10 * SECOND, 500)]
    )
    assert nominal_cadence_ns(times) == 10 * SECOND


def test_dropped_rows_do_not_move_the_cadence() -> None:
    times = np.delete(_times(0, SECOND, 200), [10, 11, 12, 50, 90])
    assert nominal_cadence_ns(times) == SECOND


def test_jitter_yields_the_central_interval() -> None:
    rng = np.random.default_rng(0)
    jitter = rng.integers(-SECOND // 20, SECOND // 20, size=400)
    times = np.cumsum(np.full(400, SECOND, dtype=np.int64) + jitter)
    cadence = nominal_cadence_ns(times)
    assert cadence is not None
    assert abs(cadence - SECOND) < SECOND // 10


def test_cadence_is_rounded_not_truncated() -> None:
    """An even count makes np.median interpolate; truncating would bias low."""
    times = np.array([0, 3, 3 + 4], dtype=np.int64)  # diffs 3 and 4, median 3.5
    assert nominal_cadence_ns(times) == 4


# ---------------------------------------------------------------------------
# CellBounds validation
# ---------------------------------------------------------------------------


def test_bounds_reject_mismatched_lengths() -> None:
    with pytest.raises(TsaraSupportError, match="matching lengths"):
        CellBounds(start_ns=np.zeros(3, dtype=np.int64), stop_ns=np.zeros(2, dtype=np.int64))


def test_bounds_reject_two_dimensional_input() -> None:
    with pytest.raises(TsaraSupportError, match="one-dimensional"):
        CellBounds(
            start_ns=np.zeros((2, 2), dtype=np.int64), stop_ns=np.zeros((2, 2), dtype=np.int64)
        )


def test_bounds_reject_a_nat_start() -> None:
    start = np.array([0, np.iinfo(np.int64).min], dtype=np.int64)
    with pytest.raises(TsaraSupportError, match="NaT"):
        CellBounds(start_ns=start, stop_ns=start + 10)


def test_bounds_reject_a_nat_stop() -> None:
    """The second half of the guard: a valid start with a missing stop."""
    with pytest.raises(TsaraSupportError, match="NaT"):
        CellBounds(
            start_ns=np.array([0, 10], dtype=np.int64),
            stop_ns=np.array([5, np.iinfo(np.int64).min], dtype=np.int64),
        )


def test_bounds_reject_a_stop_before_its_start() -> None:
    with pytest.raises(TsaraSupportError, match="stop earlier than their start"):
        CellBounds(
            start_ns=np.array([0, 100], dtype=np.int64),
            stop_ns=np.array([10, 50], dtype=np.int64),
        )


def test_empty_bounds_are_legal() -> None:
    empty = CellBounds(start_ns=np.array([], dtype=np.int64), stop_ns=np.array([], dtype=np.int64))
    assert len(empty) == 0
    assert empty.span_ns == 0
    assert np.isnan(empty.coverage_fraction)


# ---------------------------------------------------------------------------
# CellBounds geometry
# ---------------------------------------------------------------------------


def test_start_labelled_cells_run_forward_from_the_stamp() -> None:
    bounds = CellBounds.from_label(_times(0, SECOND, 3), SECOND, "start")
    assert list(bounds.start_ns) == [0, SECOND, 2 * SECOND]
    assert list(bounds.stop_ns) == [SECOND, 2 * SECOND, 3 * SECOND]


def test_end_labelled_cells_run_backward_from_the_stamp() -> None:
    bounds = CellBounds.from_label(np.array([10 * SECOND], dtype=np.int64), 4 * SECOND, "end")
    assert list(bounds.start_ns) == [6 * SECOND]
    assert list(bounds.stop_ns) == [10 * SECOND]


def test_mid_and_unknown_labels_both_centre_the_cell() -> None:
    times = np.array([10 * SECOND], dtype=np.int64)
    mid = CellBounds.from_label(times, 4 * SECOND, "mid")
    unknown = CellBounds.from_label(times, 4 * SECOND, "unknown")
    assert list(mid.start_ns) == list(unknown.start_ns) == [8 * SECOND]
    assert list(mid.stop_ns) == list(unknown.stop_ns) == [12 * SECOND]


def test_an_unknown_label_leaves_the_timestamp_where_it_was() -> None:
    """Why the 2026 archive sees no timestamp change: centred cells keep it."""
    times = _times(1_700_000_000 * SECOND, 2 * SECOND, 5)
    bounds = CellBounds.from_label(times, 2 * SECOND, "unknown")
    assert list(bounds.midpoint_ns) == list(times)


def test_a_start_label_moves_the_midpoint_by_half_a_cell() -> None:
    times = _times(0, 60 * SECOND, 3)
    bounds = CellBounds.from_label(times, 60 * SECOND, "start")
    assert list(bounds.midpoint_ns - times) == [30 * SECOND] * 3


def test_odd_widths_keep_every_cell_exactly_the_declared_width() -> None:
    bounds = CellBounds.from_label(_times(0, 7, 4), 7, "mid")
    assert list(bounds.width_ns) == [7, 7, 7, 7]


def test_zero_width_cells_are_refused_at_construction() -> None:
    with pytest.raises(TsaraSupportError, match="strictly positive"):
        CellBounds.from_label(_times(0, SECOND, 3), 0, "mid")


def test_the_midpoint_does_not_overflow_int64() -> None:
    """`(start + stop) // 2` overflows after ~2116, inside datetime64[ns] range.

    A latent wrong answer rather than an error, which is why the safe form is
    used and pinned by a test.
    """
    far = int(pd.Timestamp("2200-01-01").value)
    bounds = CellBounds.from_label(np.array([far], dtype=np.int64), 60 * SECOND, "start")
    assert int(bounds.midpoint_ns[0]) == far + 30 * SECOND
    with np.errstate(over="ignore"):
        naive = (bounds.start_ns + bounds.stop_ns) // 2
    assert int(naive[0]) != far + 30 * SECOND


def test_coverage_is_one_when_cells_tile_the_record() -> None:
    bounds = CellBounds.from_label(_times(0, SECOND, 100), SECOND, "start")
    assert bounds.coverage_fraction == pytest.approx(1.0, abs=0.02)


def test_coverage_reports_the_hole_rather_than_hiding_it() -> None:
    """Half the record missing must read as coverage near one half."""
    # 50 cells covering 0-50 s, then 50 covering 150-200 s: 100 s of cells
    # inside a 200 s span.
    times = np.concatenate([_times(0, SECOND, 50), _times(150 * SECOND, SECOND, 50)])
    bounds = CellBounds.from_label(times, SECOND, "start")
    assert bounds.coverage_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# floor_width
# ---------------------------------------------------------------------------


def test_floor_width_leaves_wide_enough_cells_untouched() -> None:
    bounds = CellBounds.from_label(_times(0, SECOND, 5), SECOND, "start")
    widened, n = bounds.floor_width(SECOND)
    assert n == 0
    assert widened is bounds


def test_floor_width_widens_declared_zero_width_cells() -> None:
    """The real airborne case: stop equals start, so every weight is zero."""
    start = _times(0, SECOND, 4)
    bounds = CellBounds(start_ns=start, stop_ns=start.copy())
    widened, n = bounds.floor_width(SECOND)
    assert n == 4
    assert list(widened.width_ns) == [SECOND] * 4
    assert list(widened.midpoint_ns) == list(start)


def test_floor_width_only_touches_the_narrow_cells() -> None:
    bounds = CellBounds(
        start_ns=np.array([0, 100, 200], dtype=np.int64),
        stop_ns=np.array([50, 100, 260], dtype=np.int64),
    )
    widened, n = bounds.floor_width(40)
    assert n == 1
    assert list(widened.width_ns) == [50, 40, 60]


def test_floor_width_refuses_a_nonpositive_minimum() -> None:
    bounds = CellBounds.from_label(_times(0, SECOND, 2), SECOND, "start")
    with pytest.raises(TsaraSupportError, match="strictly positive"):
        bounds.floor_width(0)


# ---------------------------------------------------------------------------
# bin_onto_cells
# ---------------------------------------------------------------------------


def test_binning_averages_a_fast_stream_onto_a_slow_cell() -> None:
    source = CellBounds.from_label(_times(0, SECOND, 60), SECOND, "start")
    values = np.arange(60, dtype=np.float64)
    target = CellBounds.from_label(np.array([0], dtype=np.int64), 60 * SECOND, "start")
    out = bin_onto_cells(source, values, target)
    assert out.values[0] == pytest.approx(values.mean())
    assert out.n_source[0] == 60
    assert out.coverage[0] == pytest.approx(1.0)


def test_binning_weights_a_straddling_cell_by_its_overlap() -> None:
    """A source cell half inside the target counts half as much."""
    source = CellBounds(
        start_ns=np.array([0, 10], dtype=np.int64),
        stop_ns=np.array([10, 30], dtype=np.int64),
    )
    target = CellBounds(
        start_ns=np.array([0], dtype=np.int64), stop_ns=np.array([20], dtype=np.int64)
    )
    out = bin_onto_cells(source, np.array([0.0, 10.0]), target)
    # 10 ns of value 0 and 10 ns of value 10.
    assert out.values[0] == pytest.approx(5.0)
    assert out.coverage[0] == pytest.approx(1.0)


def test_a_target_cell_with_no_source_data_stays_nan() -> None:
    """Never interpolated: a hole is a hole (METHODS 1.2)."""
    source = CellBounds.from_label(_times(0, SECOND, 5), SECOND, "start")
    target = CellBounds.from_label(np.array([1000 * SECOND], dtype=np.int64), SECOND, "start")
    out = bin_onto_cells(source, np.ones(5), target)
    assert np.isnan(out.values[0])
    assert out.n_source[0] == 0
    assert out.coverage[0] == 0.0


def test_masked_source_samples_reduce_coverage_rather_than_passing_as_data() -> None:
    source = CellBounds.from_label(_times(0, SECOND, 10), SECOND, "start")
    values = np.arange(10, dtype=np.float64)
    values[:5] = np.nan
    target = CellBounds.from_label(np.array([0], dtype=np.int64), 10 * SECOND, "start")
    out = bin_onto_cells(source, values, target)
    assert out.values[0] == pytest.approx(np.nanmean(values))
    assert out.n_source[0] == 5
    assert out.coverage[0] == pytest.approx(0.5)


def test_binning_handles_source_cells_whose_stops_are_unsorted() -> None:
    """Jittered fixed-width cells can overlap, so raw stops are not sorted."""
    source = CellBounds(
        start_ns=np.array([0, 5, 12], dtype=np.int64),
        stop_ns=np.array([40, 15, 22], dtype=np.int64),
    )
    target = CellBounds(
        start_ns=np.array([10], dtype=np.int64), stop_ns=np.array([20], dtype=np.int64)
    )
    out = bin_onto_cells(source, np.array([1.0, 1.0, 1.0]), target)
    assert out.n_source[0] == 3


def test_binning_refuses_unsorted_source_cells() -> None:
    source = CellBounds(
        start_ns=np.array([100, 0], dtype=np.int64),
        stop_ns=np.array([110, 10], dtype=np.int64),
    )
    target = CellBounds(
        start_ns=np.array([0], dtype=np.int64), stop_ns=np.array([200], dtype=np.int64)
    )
    with pytest.raises(TsaraSupportError, match="sorted by start time"):
        bin_onto_cells(source, np.array([1.0, 2.0]), target)


def test_binning_refuses_a_value_count_that_does_not_match() -> None:
    source = CellBounds.from_label(_times(0, SECOND, 3), SECOND, "start")
    target = CellBounds.from_label(np.array([0], dtype=np.int64), SECOND, "start")
    with pytest.raises(TsaraSupportError, match="one to one"):
        bin_onto_cells(source, np.ones(2), target)


def test_binning_with_no_target_cells_returns_empty_arrays() -> None:
    source = CellBounds.from_label(_times(0, SECOND, 3), SECOND, "start")
    empty = CellBounds(start_ns=np.array([], dtype=np.int64), stop_ns=np.array([], dtype=np.int64))
    out = bin_onto_cells(source, np.ones(3), empty)
    assert out.values.size == 0


def test_binning_with_no_source_cells_returns_all_nan() -> None:
    empty = CellBounds(start_ns=np.array([], dtype=np.int64), stop_ns=np.array([], dtype=np.int64))
    target = CellBounds.from_label(_times(0, SECOND, 3), SECOND, "start")
    out = bin_onto_cells(empty, np.array([], dtype=np.float64), target)
    assert np.all(np.isnan(out.values))
    assert list(out.n_source) == [0, 0, 0]


def test_a_zero_width_target_cell_reports_zero_coverage_without_dividing_by_zero() -> None:
    source = CellBounds.from_label(_times(0, SECOND, 5), SECOND, "start")
    target = CellBounds(
        start_ns=np.array([2 * SECOND], dtype=np.int64),
        stop_ns=np.array([2 * SECOND], dtype=np.int64),
    )
    out = bin_onto_cells(source, np.ones(5), target)
    assert out.coverage[0] == 0.0
    assert np.isnan(out.values[0])


def test_binning_a_canister_against_one_hertz_data() -> None:
    """The motivating case: a ~15 s fill against a 1 Hz partner."""
    source = CellBounds.from_label(_times(0, SECOND, 120), SECOND, "start")
    values = np.zeros(120)
    values[30:45] = 10.0
    canister = CellBounds(
        start_ns=np.array([30 * SECOND], dtype=np.int64),
        stop_ns=np.array([45 * SECOND], dtype=np.int64),
    )
    out = bin_onto_cells(source, values, canister)
    assert out.values[0] == pytest.approx(10.0)
    assert out.n_source[0] == 15


# ---------------------------------------------------------------------------
# attach_time_bounds
# ---------------------------------------------------------------------------


def _stream(n: int = 4) -> xr.Dataset:
    times = pd.date_range("2024-07-01", periods=n, freq="60s")
    return xr.Dataset(
        {"ch4": (TIME_COORD, np.arange(float(n))), "site_id": ((), 3)},
        coords={TIME_COORD: times},
    )


def test_attaching_bounds_writes_the_cf_representation() -> None:
    stream = _stream()
    bounds = CellBounds.from_label(
        stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64), 60 * SECOND, "start"
    )
    attach_time_bounds(stream, bounds, "mean")
    assert stream[TIME_COORD].attrs[BOUNDS_ATTR] == TIME_BOUNDS_VAR
    assert TIME_BOUNDS_VAR in stream.coords
    assert stream[TIME_BOUNDS_VAR].dims == (TIME_COORD, BOUNDS_DIM)
    assert stream["ch4"].attrs[CELL_METHODS_ATTR] == "time: mean"


def test_attaching_bounds_skips_variables_without_a_time_axis() -> None:
    stream = _stream()
    bounds = CellBounds.from_label(
        stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64), 60 * SECOND, "start"
    )
    attach_time_bounds(stream, bounds, "point")
    assert CELL_METHODS_ATTR not in stream["site_id"].attrs


def test_attaching_the_wrong_number_of_cells_is_refused() -> None:
    stream = _stream(4)
    bounds = CellBounds.from_label(_times(0, SECOND, 3), SECOND, "start")
    with pytest.raises(TsaraSupportError, match="bounds are per-timestamp"):
        attach_time_bounds(stream, bounds, "mean")


# ---------------------------------------------------------------------------
# Persistence: the encoding pin, and the operation that must never be used
# ---------------------------------------------------------------------------


def test_pinning_leaves_a_stream_without_a_time_axis_alone() -> None:
    pin_time_encoding(xr.Dataset({"x": ((), 1)}))


def test_bounds_survive_netcdf_exactly_and_without_a_cf_warning(tmp_path: Path) -> None:
    """Unpinned, xarray gives time and time_bnds different reference epochs."""
    stream = _stream()
    stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    bounds = CellBounds.from_label(stamps, 60 * SECOND, "start")
    attach_time_bounds(stream, bounds, "mean")
    pin_time_encoding(stream)

    target = tmp_path / "stream.nc"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stream.to_netcdf(target, engine="netcdf4")
    assert not [w for w in caught if "bounds" in str(w.message)]

    with xr.open_dataset(target, engine="netcdf4", decode_coords="all") as back:
        assert TIME_BOUNDS_VAR in back.coords
        # The timestamps themselves, not only the bounds. Checking only the
        # bounds let a pin that wrote every time value as NaT pass unnoticed.
        assert back[TIME_COORD].dtype == "datetime64[ns]"
        assert np.array_equal(
            back[TIME_COORD].values.astype("datetime64[ns]"),
            stream[TIME_COORD].values.astype("datetime64[ns]"),
        )
        assert np.array_equal(
            back[TIME_BOUNDS_VAR].values.astype("datetime64[ns]").astype(np.int64),
            np.stack([bounds.start_ns, bounds.stop_ns], axis=1),
        )
        assert back[TIME_COORD].encoding["units"] == TIME_ENCODING["units"]
        assert back[TIME_BOUNDS_VAR].encoding["units"] == TIME_ENCODING["units"]


def test_resample_destroys_bounds_which_is_why_tsara_never_uses_it() -> None:
    """Documents the trap the operational rule exists to avoid.

    Measured, and it fails in two different ways depending on how the bounds
    are stored, neither of which raises:

    * as a **coordinate**, which is how TSARA stores them, ``resample`` drops
      the variable but leaves ``time.attrs['bounds']`` naming it -- a
      dangling CF reference to something that no longer exists;
    * as a **data variable**, it averages the boundary timestamps into new
      datetimes, so the cells claim intervals no instrument ever measured.

    It also silently coarsens the time axis from nanoseconds to microseconds,
    which is the resolution pin the rest of the package works to maintain.
    """
    stream = _stream(4)
    stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    attach_time_bounds(stream, CellBounds.from_label(stamps, 60 * SECOND, "start"), "mean")

    resampled = stream.resample({TIME_COORD: "120s"}).mean()
    assert TIME_BOUNDS_VAR not in resampled.variables
    assert resampled[TIME_COORD].attrs[BOUNDS_ATTR] == TIME_BOUNDS_VAR  # dangling
    assert resampled[TIME_COORD].dtype != "datetime64[ns]"

    as_variable = stream.reset_coords(TIME_BOUNDS_VAR)
    averaged = as_variable.resample({TIME_COORD: "120s"}).mean()
    # Two 60 s cells at 0-60 and 60-120 aggregate into a cell that ought to
    # span 0-120. Averaging their boundaries claims 30-90 instead: the width
    # survives by coincidence while the cell moves off the data it describes.
    aggregated_start = averaged[TIME_BOUNDS_VAR].values[0, 0]
    assert aggregated_start != stream[TIME_BOUNDS_VAR].values[0, 0]
    assert aggregated_start == stream[TIME_COORD].values[0] + np.timedelta64(30 * SECOND, "ns")


def test_a_dangling_bounds_reference_is_caught() -> None:
    """The guard that turns the rule above into a check rather than a note."""
    stream = _stream(4)
    stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    attach_time_bounds(stream, CellBounds.from_label(stamps, 60 * SECOND, "start"), "mean")
    check_bounds_intact(stream)

    broken = stream.drop_vars(TIME_BOUNDS_VAR)
    with pytest.raises(TsaraSupportError, match="names 'time_bnds'"):
        check_bounds_intact(broken)


def test_bounds_of_the_wrong_length_are_caught() -> None:
    stream = _stream(4)
    # A bounds variable that exists but is not one cell per timestamp.
    stream.coords["other"] = ("other", np.arange(3))
    stream[TIME_BOUNDS_VAR] = (
        ("other", BOUNDS_DIM),
        np.zeros((3, 2), dtype="datetime64[ns]"),
    )
    stream[TIME_COORD].attrs[BOUNDS_ATTR] = TIME_BOUNDS_VAR
    with pytest.raises(TsaraSupportError, match="shape"):
        check_bounds_intact(stream)


def test_a_stream_without_bounds_passes_the_check() -> None:
    """Not every product carries cells; the check must not invent a rule."""
    check_bounds_intact(_stream(3))


def test_a_product_without_a_time_axis_passes_the_check() -> None:
    """Not every stage product is a timeseries; the check must not assume one."""
    check_bounds_intact(xr.Dataset({"slope": ((), 1.4)}))


def test_an_independent_cf_reader_finds_our_cells(tmp_path: Path) -> None:
    """Third-party validation that what we write is really CF.

    Asserting our own attributes back to ourselves proves only
    self-consistency. This reopens a saved stream through ``cf_xarray``, an
    implementation that knows nothing about TSARA, and checks that it can
    find the time axis and the cell boundaries by their CF roles rather than
    by the names we happened to choose.
    """
    pytest.importorskip("cf_xarray")
    import cf_xarray  # noqa: F401

    stream = _stream(4)
    stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    attach_time_bounds(stream, CellBounds.from_label(stamps, 60 * SECOND, "start"), "mean")
    pin_time_encoding(stream)
    target = tmp_path / "cf.nc"
    stream.to_netcdf(target, engine="netcdf4")

    with xr.open_dataset(target, engine="netcdf4", decode_coords="all") as back:
        # Reachable both by name and by CF role, the latter only because the
        # axis is declared: cf_xarray keys bounds under 'T' as well as 'time'.
        assert back.cf.bounds[TIME_COORD] == [TIME_BOUNDS_VAR]
        assert back.cf.bounds["T"] == [TIME_BOUNDS_VAR]
        assert back.cf.get_bounds(TIME_COORD).name == TIME_BOUNDS_VAR
        assert back.cf.axes["T"] == [TIME_COORD]


def test_coarse_cell_bounds_are_widened_before_they_are_pinned(tmp_path: Path) -> None:
    """A bounds array coarser than nanoseconds round-trips to NaT, silently.

    A CF bounds variable carries no units of its own -- it inherits its
    parent's, and xarray implements that -- so `time_bnds` is encoded with
    whatever `time` was pinned to whether or not anything pinned it directly.
    Nanosecond units applied to a microsecond array make xarray write the NaT
    sentinel for every value and raise nothing, so the cells come back as
    `['NaT' 'NaT']` and every later overlap silently finds nothing.

    `pin_time_encoding` widens first, which is exact. Removing `time_bnds`
    from its loop passed the whole suite, because nothing built bounds at any
    other resolution.
    """
    stream = _stream(4)
    stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    attach_time_bounds(stream, CellBounds.from_label(stamps, 60 * SECOND, "start"), "mean")
    expected = stream[TIME_BOUNDS_VAR].values.astype("datetime64[ns]")
    # The shape a dataset assembled outside TSARA can easily have.
    stream[TIME_BOUNDS_VAR] = stream[TIME_BOUNDS_VAR].astype("datetime64[us]")

    pin_time_encoding(stream)
    target = tmp_path / "coarse.nc"
    stream.to_netcdf(target, engine="netcdf4")
    with xr.open_dataset(target, engine="netcdf4", decode_coords="all") as back:
        found = back[TIME_BOUNDS_VAR].values.astype("datetime64[ns]")
    assert not np.isnat(found).any()
    assert np.array_equal(found, expected)


# ---------------------------------------------------------------------------
# Finding the bounds a stream declares
# ---------------------------------------------------------------------------


def _saved_and_reopened(tmp_path: Path, label: SupportLabel, method: SupportMethod) -> xr.Dataset:
    stream = _stream(4)
    stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    attach_time_bounds(stream, CellBounds.from_label(stamps, 60 * SECOND, label), method)
    pin_time_encoding(stream)
    target = tmp_path / f"{label}_{method}.nc"
    stream.to_netcdf(target, engine="netcdf4")
    with xr.open_dataset(target, engine="netcdf4", decode_coords="all") as opened:
        return opened.load()


def test_a_reopened_stream_still_declares_its_bounds(tmp_path: Path) -> None:
    """Regression: `decode_coords='all'` moves the attribute into `encoding`.

    Checking only `attrs` reported "no cells" for every stream TSARA itself
    had just loaded.
    """
    back = _saved_and_reopened(tmp_path, "start", "mean")
    assert back[TIME_COORD].attrs.get(BOUNDS_ATTR) is None
    assert declared_bounds_name(back) == TIME_BOUNDS_VAR


def test_reloading_does_not_replace_declared_cells_with_assumed_ones(
    tmp_path: Path,
) -> None:
    """The failure the bug above would have caused, stated as a test.

    A start-labelled mean stream is the only shape that can see it: for a
    centred point stream the assumed cells are identical to the real ones, so
    a round-trip check cannot tell that they were silently substituted.
    """
    back = _saved_and_reopened(tmp_path, "start", "mean")
    before = back[TIME_BOUNDS_VAR].values.copy()
    assert ensure_time_bounds(back) is False
    assert np.array_equal(back[TIME_BOUNDS_VAR].values, before)
    # Still start-labelled: each cell runs forward from its stamp. Assumed
    # cells would have been centred on it, so the two are distinguishable.
    stamps = back[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    bounds = back[TIME_BOUNDS_VAR].values.astype("datetime64[ns]").astype(np.int64)
    assert np.array_equal(bounds[:, 0], stamps)
    assert np.array_equal(bounds[:, 1] - stamps, np.full(stamps.size, 60 * SECOND))


def test_bounds_survive_two_round_trips(tmp_path: Path) -> None:
    """Once the attribute lives in `encoding`, a re-save must still write it."""
    back = _saved_and_reopened(tmp_path, "start", "mean")
    again = tmp_path / "again.nc"
    pin_time_encoding(back)
    back.to_netcdf(again, engine="netcdf4")
    with xr.open_dataset(again, engine="netcdf4", decode_coords="all") as reopened:
        assert declared_bounds_name(reopened) == TIME_BOUNDS_VAR
        assert np.array_equal(reopened[TIME_BOUNDS_VAR].values, back[TIME_BOUNDS_VAR].values)


def test_a_stream_that_declares_nothing_gets_assumed_cells() -> None:
    stream = _stream(5)
    assert ensure_time_bounds(stream) is True
    assert TIME_BOUNDS_VAR in stream.coords
    stamps = stream[TIME_COORD].values.astype("datetime64[ns]").astype(np.int64)
    bounds = stream[TIME_BOUNDS_VAR].values.astype("datetime64[ns]").astype(np.int64)
    # Centred, so no timestamp moves.
    assert np.array_equal(bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) // 2, stamps)
    assert stream["ch4"].attrs[CELL_METHODS_ATTR] == "time: point"


def test_a_product_without_a_time_axis_is_left_alone() -> None:
    assert ensure_time_bounds(xr.Dataset({"slope": ((), 1.4)})) is False


def test_a_single_sample_stream_gets_no_invented_cadence() -> None:
    """One timestamp implies no interval, and guessing one would be a
    fabrication rather than the weak-but-honest reading migration applies."""
    assert ensure_time_bounds(_stream(1)) is False


def test_a_dangling_declaration_is_left_for_the_intactness_check() -> None:
    """Papering over it with assumed cells would hide a real corruption."""
    stream = _stream(4)
    stream[TIME_COORD].attrs[BOUNDS_ATTR] = TIME_BOUNDS_VAR
    assert ensure_time_bounds(stream) is False
    with pytest.raises(TsaraSupportError, match="names 'time_bnds'"):
        check_bounds_intact(stream)


def test_support_attrs_omit_a_width_that_does_not_exist() -> None:
    """A canister record has per-row widths and no single nominal one."""
    attrs = support_attrs(
        label="start",
        width_ns=None,
        coverage=0.5,
        label_source="reported",
        width_source="reported",
        method_source="declared",
    )
    assert SUPPORT_WIDTH_ATTR not in attrs
    assert attrs[SUPPORT_LABEL_SOURCE_ATTR] == "reported"


def test_support_attrs_record_the_nominal_width_in_seconds() -> None:
    attrs = support_attrs(
        label="mid",
        width_ns=60 * SECOND,
        coverage=1.0,
        label_source="declared",
        width_source="inferred",
        method_source="assumed",
    )
    assert attrs[SUPPORT_WIDTH_ATTR] == pytest.approx(60.0)


def test_pinning_widens_a_coarser_time_axis_instead_of_destroying_it(
    tmp_path: Path,
) -> None:
    """Regression: nanosecond units on a microsecond axis wrote all-NaT.

    `pd.date_range` returns microseconds in current pandas, so this is the
    shape any dataset assembled outside TSARA's own producers arrives in. The
    failure was silent in both directions: nothing raised on write, and the
    file read back as a full axis of NaT.
    """
    times = pd.date_range("2024-07-01", periods=4, freq="60s").as_unit("us")
    stream = xr.Dataset({"ch4": (TIME_COORD, np.arange(4.0))}, coords={TIME_COORD: times})
    assert stream[TIME_COORD].dtype == "datetime64[us]"

    pin_time_encoding(stream)
    assert stream[TIME_COORD].dtype == "datetime64[ns]"

    target = tmp_path / "coarse.nc"
    stream.to_netcdf(target, engine="netcdf4")
    with xr.open_dataset(target, engine="netcdf4") as back:
        assert np.array_equal(
            back[TIME_COORD].values.astype("datetime64[ns]"),
            times.values.astype("datetime64[ns]"),
        )


def test_a_per_row_width_must_have_one_value_per_row() -> None:
    with pytest.raises(TsaraSupportError, match="one value per row"):
        CellBounds.from_label(_times(0, SECOND, 4), np.full(3, SECOND, dtype=np.int64), "mid")


def test_a_width_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(TsaraSupportError, match="scalar or one value per row"):
        CellBounds.from_label(_times(0, SECOND, 4), np.ones((2, 2), dtype=np.int64), "mid")


def test_per_row_widths_build_cells_of_differing_size() -> None:
    """The mixed-cadence case: one instrument's files can disagree."""
    widths = np.array([SECOND, SECOND, 5 * SECOND], dtype=np.int64)
    bounds = CellBounds.from_label(_times(0, 10 * SECOND, 3), widths, "start")
    assert list(bounds.width_ns) == [SECOND, SECOND, 5 * SECOND]


def test_floor_width_accepts_a_per_row_minimum() -> None:
    """One instrument's files can disagree about cadence, so the floor can too."""
    start = _times(0, 10 * SECOND, 3)
    bounds = CellBounds(start_ns=start, stop_ns=start.copy())
    minimum = np.array([SECOND, 5 * SECOND, 2 * SECOND], dtype=np.int64)
    widened, n = bounds.floor_width(minimum)
    assert n == 3
    assert list(widened.width_ns) == [SECOND, 5 * SECOND, 2 * SECOND]


def test_binning_separates_no_data_from_rejected_data() -> None:
    """Both leave a NaN, and a later stage diagnosing a dropped pair needs to
    tell a gap apart from a QA/QC decision."""
    source = CellBounds.from_label(_times(0, SECOND, 10), SECOND, "start")
    values = np.full(10, np.nan)
    covered = CellBounds.from_label(np.array([0], dtype=np.int64), 10 * SECOND, "start")
    empty = CellBounds.from_label(np.array([1000 * SECOND], dtype=np.int64), SECOND, "start")

    masked = bin_onto_cells(source, values, covered)
    assert np.isnan(masked.values[0])
    assert masked.n_source[0] == 0, "nothing contributed"
    assert masked.n_overlapping[0] == 10, "but ten cells were there and rejected"

    absent = bin_onto_cells(source, np.ones(10), empty)
    assert np.isnan(absent.values[0])
    assert absent.n_source[0] == 0
    assert absent.n_overlapping[0] == 0, "genuinely nothing there"


def test_support_attrs_record_a_repair_only_when_one_happened() -> None:
    plain = support_attrs(
        label="start",
        width_ns=SECOND,
        coverage=1.0,
        label_source="reported",
        width_source="reported",
        method_source="declared",
    )
    repaired = support_attrs(
        label="start",
        width_ns=SECOND,
        coverage=1.0,
        label_source="reported",
        width_source="reported",
        method_source="declared",
        n_widened=644,
    )
    assert SUPPORT_WIDENED_ATTR not in plain
    assert repaired[SUPPORT_WIDENED_ATTR] == 644.0


# ---------------------------------------------------------------------------
# overlap_pairs
# ---------------------------------------------------------------------------
#
# Extracted from `bin_onto_cells` when circular binning became the second
# caller. It has direct tests because it is now the single place that decides
# which measurements describe which interval -- if it is wrong, a cell's wind
# direction and its methane silently come from different stretches of air.


def test_overlap_pairs_finds_every_overlap_and_no_others() -> None:
    source = CellBounds(
        start_ns=_times(0, SECOND, 5),
        stop_ns=_times(0, SECOND, 5) + SECOND,
    )
    target = CellBounds(
        start_ns=np.array([2 * SECOND], dtype=np.int64),
        stop_ns=np.array([4 * SECOND], dtype=np.int64),
    )
    pairs = overlap_pairs(source, target)
    contributing = pairs.overlap_ns > 0
    assert sorted(pairs.source_index[contributing].tolist()) == [2, 3]
    assert pairs.overlap_ns[contributing].tolist() == [SECOND, SECOND]
    assert pairs.n_target == 1


def test_overlap_pairs_measures_a_partial_overlap_exactly() -> None:
    """A source cell straddling the target boundary contributes its share."""
    source = CellBounds(
        start_ns=np.array([0], dtype=np.int64),
        stop_ns=np.array([10 * SECOND], dtype=np.int64),
    )
    target = CellBounds(
        start_ns=np.array([7 * SECOND], dtype=np.int64),
        stop_ns=np.array([20 * SECOND], dtype=np.int64),
    )
    pairs = overlap_pairs(source, target)
    assert pairs.overlap_ns.tolist() == [3 * SECOND]


def test_overlap_pairs_is_empty_when_the_records_do_not_meet() -> None:
    source = CellBounds(start_ns=_times(0, SECOND, 3), stop_ns=_times(0, SECOND, 3) + SECOND)
    target = CellBounds(
        start_ns=np.array([100 * SECOND], dtype=np.int64),
        stop_ns=np.array([101 * SECOND], dtype=np.int64),
    )
    pairs = overlap_pairs(source, target)
    assert pairs.overlap_ns.size == 0
    assert pairs.n_target == 1


def test_overlap_pairs_handles_empty_inputs() -> None:
    empty = CellBounds(start_ns=np.empty(0, dtype=np.int64), stop_ns=np.empty(0, dtype=np.int64))
    filled = CellBounds(start_ns=_times(0, SECOND, 3), stop_ns=_times(0, SECOND, 3) + SECOND)
    assert overlap_pairs(empty, filled).overlap_ns.size == 0
    assert overlap_pairs(empty, filled).n_target == 3
    assert overlap_pairs(filled, empty).n_target == 0


def test_overlap_pairs_requires_sorted_source_cells() -> None:
    source = CellBounds(
        start_ns=np.array([5 * SECOND, 0], dtype=np.int64),
        stop_ns=np.array([6 * SECOND, SECOND], dtype=np.int64),
    )
    target = CellBounds(
        start_ns=np.array([0], dtype=np.int64), stop_ns=np.array([SECOND], dtype=np.int64)
    )
    with pytest.raises(TsaraSupportError, match="sorted by start time"):
        overlap_pairs(source, target)


def test_overlap_pairs_tolerates_slightly_overlapping_source_cells() -> None:
    """Fixed-width cells centred on jittered stamps overlap each other.

    Their stops are then not sorted even though their starts are, which is why
    the candidate search runs on a cumulative maximum. Documented as a known
    limit in METHODS §10; here it is pinned as behaviour.
    """
    start = np.array([0, 900_000_000, 2 * SECOND], dtype=np.int64)
    source = CellBounds(start_ns=start, stop_ns=start + SECOND)
    target = CellBounds(
        start_ns=np.array([0], dtype=np.int64),
        stop_ns=np.array([3 * SECOND], dtype=np.int64),
    )
    pairs = overlap_pairs(source, target)
    assert sorted(pairs.source_index.tolist()) == [0, 1, 2]
    assert int(pairs.overlap_ns.sum()) == 3 * SECOND


def test_the_two_binners_weight_by_the_same_overlaps() -> None:
    """The reason the search was extracted rather than copied.

    A scalar and an angle measured over the same interval must be averaged
    over the same interval. Comparing the contributing counts and coverage of
    the two paths is how that stays true.
    """
    from tsara.core.circular import bin_circular_onto_cells

    source = CellBounds(_times(0, SECOND, 30), _times(0, SECOND, 30) + SECOND)
    target = CellBounds(
        start_ns=np.array([2 * SECOND, 11 * SECOND], dtype=np.int64),
        stop_ns=np.array([9 * SECOND, 40 * SECOND], dtype=np.int64),
    )
    values = np.arange(30.0)
    values[5] = np.nan
    scalar = bin_onto_cells(source, values, target)
    angular = bin_circular_onto_cells(source, values, target)
    assert scalar.n_source.tolist() == angular.n_source.tolist()
    assert scalar.n_overlapping.tolist() == angular.n_overlapping.tolist()
    assert scalar.coverage == pytest.approx(angular.coverage)
