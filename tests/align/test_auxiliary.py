"""Tests for auxiliary-field interpolation.

The two guards carry most of the weight here, because both are refusals and a
refusal that quietly stops refusing is invisible. One test asserts a gas
cannot be interpolated at all; another asserts a gap longer than the limit
yields ``nan`` rather than a bridged value; a third asserts a record with no
declared role is refused rather than assumed smooth.

The arithmetic is checked against closed forms, since interpolation has them:
a straight line interpolates to itself exactly, and a direction interpolated
across the compass seam must not travel the long way round.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tsara.align import TsaraAlignError
from tsara.align.auxiliary import attach_positions, interpolate_onto_cells
from tsara.core.support import CellBounds

SECOND = 1_000_000_000


def cells(start_s: float, width_s: float, n: int) -> CellBounds:
    """Return ``n`` abutting cells of ``width_s`` starting at ``start_s``."""
    start = (np.arange(n, dtype=np.int64) * int(width_s * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def make_stream(
    bounds: CellBounds,
    variables: dict[str, np.ndarray],
    *,
    attrs: dict[str, dict[str, object]] | None = None,
    stream_attrs: dict[str, object] | None = None,
) -> xr.Dataset:
    """Build a minimal stream with CF cells and per-variable attrs."""
    variable_attrs = attrs or {}
    dataset = xr.Dataset(
        data_vars={
            name: ("time", values, dict(variable_attrs.get(name, {"units": "1", "role": "aux"})))
            for name, values in variables.items()
        },
        coords={
            "time": bounds.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
        attrs=dict(stream_attrs or {}),
    )
    dataset["time"].attrs["bounds"] = "time_bnds"
    return dataset


# ---------------------------------------------------------------------------
# The guard on WHAT
# ---------------------------------------------------------------------------


def test_a_gas_is_refused_outright() -> None:
    """The central prohibition of METHODS §1.2, as a test.

    A concentration inside a plume is not a smooth field. If this ever stops
    raising, TSARA has quietly acquired the ability to invent structure
    exactly where the science is.
    """
    attrs: dict[str, dict[str, object]] = {"ch4": {"units": "ppb", "role": "gas"}}
    streams = {"aeris": make_stream(cells(0.0, 1.0, 10), {"ch4": np.arange(10.0)}, attrs=attrs)}
    with pytest.raises(TsaraAlignError, match="not a smooth field"):
        interpolate_onto_cells(streams, "ch4", cells(0.0, 2.0, 5))


def test_a_variable_with_no_role_is_refused_rather_than_assumed_smooth() -> None:
    """Absence is not permission.

    Every stream TSARA produces carries roles, so a missing one means the
    dataset came from somewhere else and its smoothness is unknown.
    """
    attrs: dict[str, dict[str, object]] = {"mystery": {"units": "1"}}
    streams = {"x": make_stream(cells(0.0, 1.0, 10), {"mystery": np.arange(10.0)}, attrs=attrs)}
    with pytest.raises(TsaraAlignError, match="declares no role"):
        interpolate_onto_cells(streams, "mystery", cells(0.0, 2.0, 5))


@pytest.mark.parametrize("role", ["met", "gps_lat", "gps_lon", "gps_alt", "aux"])
def test_every_smooth_role_is_permitted(role: str) -> None:
    attrs: dict[str, dict[str, object]] = {"field": {"units": "1", "role": role}}
    streams = {"x": make_stream(cells(0.0, 1.0, 10), {"field": np.arange(10.0)}, attrs=attrs)}
    result = interpolate_onto_cells(streams, "field", cells(0.0, 2.0, 5))
    assert np.isfinite(result.values).all()


# ---------------------------------------------------------------------------
# The guard on HOW FAR
# ---------------------------------------------------------------------------


def test_a_gap_longer_than_the_limit_is_not_bridged() -> None:
    """The value is nan, and the refusal is counted rather than silent."""
    source = CellBounds(
        start_ns=np.array([0, 100 * SECOND], dtype=np.int64),
        stop_ns=np.array([SECOND, 101 * SECOND], dtype=np.int64),
    )
    streams = {"gps": make_stream(source, {"field": np.array([0.0, 100.0])})}
    result = interpolate_onto_cells(streams, "field", cells(0.0, 10.0, 10), max_interp_gap="10s")
    assert result.n_gap_masked > 0
    assert np.isnan(result.values[3])


def test_a_gap_inside_the_limit_is_bridged() -> None:
    source = CellBounds(
        start_ns=np.array([0, 8 * SECOND], dtype=np.int64),
        stop_ns=np.array([SECOND, 9 * SECOND], dtype=np.int64),
    )
    streams = {"gps": make_stream(source, {"field": np.array([0.0, 8.0])})}
    result = interpolate_onto_cells(streams, "field", cells(0.0, 2.0, 4), max_interp_gap="10s")
    assert np.isfinite(result.values).all()
    assert result.n_gap_masked == 0


def test_a_target_landing_on_a_sample_is_measured_not_interpolated() -> None:
    """The gap guard must not mask a value that was actually observed.

    A GPS fix every 60 s with a 10 s guard leaves most gas cells unpositioned,
    which is correct -- but the cells that coincide with a fix are measured,
    and refusing those would discard real data.
    """
    source = cells(0.5, 1.0, 3)  # midpoints at 1, 2, 3 s
    streams = {"gps": make_stream(source, {"field": np.array([10.0, 20.0, 30.0])})}
    target = CellBounds(
        start_ns=np.array([0, 100 * SECOND], dtype=np.int64),
        stop_ns=np.array([2 * SECOND, 102 * SECOND], dtype=np.int64),
    )  # midpoints at 1 s (on a sample) and 101 s (outside)
    result = interpolate_onto_cells(streams, "field", target, max_interp_gap="1ns")
    assert result.values[0] == pytest.approx(10.0)
    assert result.n_exact == 1
    assert np.isnan(result.values[1])


def test_nothing_is_extrapolated_past_the_ends_of_a_record() -> None:
    source = cells(10.0, 1.0, 5)
    streams = {"gps": make_stream(source, {"field": np.arange(5.0)})}
    target = cells(0.0, 1.0, 30)
    result = interpolate_onto_cells(streams, "field", target, max_interp_gap="1h")
    assert result.n_outside > 0
    assert np.isnan(result.values[0])
    assert np.isnan(result.values[-1])


def test_a_source_with_no_finite_values_yields_nothing() -> None:
    streams = {"gps": make_stream(cells(0.0, 1.0, 4), {"field": np.full(4, np.nan)})}
    result = interpolate_onto_cells(streams, "field", cells(0.0, 1.0, 4))
    assert np.all(np.isnan(result.values))
    assert result.n_outside == 4


@pytest.mark.parametrize("gap", ["0s", "-5s"])
def test_a_nonpositive_gap_is_refused(gap: str) -> None:
    streams = {"gps": make_stream(cells(0.0, 1.0, 4), {"field": np.arange(4.0)})}
    with pytest.raises(TsaraAlignError, match="must be positive"):
        interpolate_onto_cells(streams, "field", cells(0.0, 1.0, 4), max_interp_gap=gap)


def test_an_unreadable_gap_names_what_was_given() -> None:
    streams = {"gps": make_stream(cells(0.0, 1.0, 4), {"field": np.arange(4.0)})}
    with pytest.raises(TsaraAlignError, match="Could not read"):
        interpolate_onto_cells(streams, "field", cells(0.0, 1.0, 4), max_interp_gap="soon")


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------


def test_a_straight_line_interpolates_to_itself() -> None:
    """Exact, and the test that would catch a timestamp precision bug.

    Epoch nanoseconds do not fit a float64 mantissa: at 2024 epochs the
    spacing between representable values is about 378 ns, so converting
    timestamps directly to float would quantise them onto a coarse grid and
    bend a straight line. The reference offset is what keeps this exact.
    """
    epoch_2024 = np.datetime64("2024-07-18T18:00:00", "ns").astype(np.int64)
    start = epoch_2024 + np.arange(600, dtype=np.int64) * SECOND
    source = CellBounds(start_ns=start, stop_ns=start + SECOND)
    slope = 3.0
    values = slope * np.arange(600.0)
    streams = {"gps": make_stream(source, {"field": values})}
    target_start = epoch_2024 + np.arange(120, dtype=np.int64) * 5 * SECOND
    target = CellBounds(start_ns=target_start, stop_ns=target_start + 5 * SECOND)
    result = interpolate_onto_cells(streams, "field", target, max_interp_gap="10s")
    # Each target midpoint is 2.5 s into a 5 s cell, i.e. 5k + 2.5 seconds in.
    expected = slope * (np.arange(120.0) * 5.0 + 2.0)
    assert result.values[:-1] == pytest.approx(expected[:-1], rel=1e-12)


def test_a_direction_does_not_travel_the_long_way_round_the_compass() -> None:
    """Interpolating 359 to 1 linearly would sweep through 180.

    A circular field is interpolated as a unit vector, so the path across the
    seam is the short one -- two degrees, not three hundred and fifty-eight.
    """
    attrs: dict[str, dict[str, object]] = {
        "wind_dir": {"units": "degrees", "role": "met", "circular": 1}
    }
    source = cells(0.0, 2.0, 2)  # midpoints at 1 s and 3 s
    streams = {"met": make_stream(source, {"wind_dir": np.array([359.0, 1.0])}, attrs=attrs)}
    target = CellBounds(
        start_ns=np.array([int(1.5 * SECOND)], dtype=np.int64),
        stop_ns=np.array([int(2.5 * SECOND)], dtype=np.int64),
    )  # midpoint exactly halfway between the two samples
    result = interpolate_onto_cells(streams, "wind_dir", target, max_interp_gap="10s")
    assert result.values[0] == pytest.approx(0.0, abs=1e-9)


def test_a_non_circular_field_is_interpolated_linearly() -> None:
    """The contrast: the same numbers without the circular flag sweep through 180."""
    attrs: dict[str, dict[str, object]] = {"bearing": {"units": "degrees", "role": "aux"}}
    source = cells(0.0, 2.0, 2)
    streams = {"met": make_stream(source, {"bearing": np.array([359.0, 1.0])}, attrs=attrs)}
    target = CellBounds(
        start_ns=np.array([int(1.5 * SECOND)], dtype=np.int64),
        stop_ns=np.array([int(2.5 * SECOND)], dtype=np.int64),
    )
    result = interpolate_onto_cells(streams, "bearing", target, max_interp_gap="10s")
    assert result.values[0] == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# The mobile position join
# ---------------------------------------------------------------------------


def mobile_campaign() -> dict[str, xr.Dataset]:
    """A gas stream bound to a GPS stream, as ingestion leaves them."""
    gps_attrs: dict[str, dict[str, object]] = {
        "lat": {"units": "degrees_north", "role": "gps_lat"},
        "lon": {"units": "degrees_east", "role": "gps_lon"},
        "alt": {"units": "m", "role": "gps_alt"},
    }
    gps = make_stream(
        cells(0.0, 1.0, 60),
        {
            "lat": np.linspace(40.0, 40.1, 60),
            "lon": np.linspace(-111.9, -111.8, 60),
            "alt": np.linspace(1400.0, 1450.0, 60),
        },
        attrs=gps_attrs,
    )
    gas = make_stream(
        cells(0.0, 5.0, 12),
        {"ch4": np.arange(12.0)},
        attrs={"ch4": {"units": "ppb", "role": "gas"}},
        stream_attrs={
            "instrument": "aeris",
            "platform_kind": "mobile",
            "platform_gps_instrument": "gps",
            "platform_lat_variable": "lat",
            "platform_lon_variable": "lon",
            "platform_alt_variable": "alt",
        },
    )
    return {"gps": gps, "aeris": gas}


def test_a_mobile_track_reaches_the_gas_clock() -> None:
    """The join ingestion deliberately left undone.

    The coordinates are named exactly as a stationary platform's are, so
    downstream code reads position identically and only has to care about the
    shape.
    """
    streams = mobile_campaign()
    joined = attach_positions(streams["aeris"], streams)
    assert "latitude" in joined.coords
    assert "longitude" in joined.coords
    assert "altitude" in joined.coords
    assert joined.sizes["time"] == 12
    assert joined["latitude"].values[0] == pytest.approx(40.0 + 0.1 * (2.0 / 59.0), abs=1e-6)
    assert joined["latitude"].attrs["tsara_interpolated_from"] == "gps.lat"


def test_the_join_records_its_guard_and_what_it_refused() -> None:
    streams = mobile_campaign()
    joined = attach_positions(streams["aeris"], streams, max_interp_gap="30s")
    assert joined["latitude"].attrs["tsara_max_interp_gap"] == "30s"
    assert joined["latitude"].attrs["tsara_interp_gap_masked"] == 0


def test_a_sparse_track_leaves_unpositioned_cells_rather_than_guessing() -> None:
    streams = mobile_campaign()
    sparse = streams["gps"].isel(time=[0, 59])
    result = attach_positions(streams["aeris"], {**streams, "gps": sparse}, max_interp_gap="5s")
    assert np.isnan(result["latitude"].values).any()
    assert result["latitude"].attrs["tsara_interp_gap_masked"] > 0


def test_a_stream_that_already_has_positions_is_returned_unchanged() -> None:
    """A stationary site, or one joined earlier. A caller iterating over a
    campaign should not have to check which."""
    streams = mobile_campaign()
    joined = attach_positions(streams["aeris"], streams)
    again = attach_positions(joined, streams)
    assert again is joined


def test_a_stream_with_no_binding_is_returned_unchanged() -> None:
    streams = mobile_campaign()
    assert attach_positions(streams["gps"], streams) is streams["gps"]


def test_a_missing_gps_instrument_is_an_error_naming_it() -> None:
    streams = mobile_campaign()
    with pytest.raises(TsaraAlignError, match="names 'gps' as its GPS instrument"):
        attach_positions(streams["aeris"], {"aeris": streams["aeris"]})


def test_a_platform_without_altitude_gets_only_latitude_and_longitude() -> None:
    streams = mobile_campaign()
    gas = streams["aeris"].copy()
    del gas.attrs["platform_alt_variable"]
    joined = attach_positions(gas, streams)
    assert "latitude" in joined.coords
    assert "altitude" not in joined.coords


def test_a_track_crossing_the_antimeridian_is_refused_rather_than_read_wrong() -> None:
    """Linear interpolation would read a 179.9 to -179.9 step as a journey
    round the planet, and a silently wrong position cannot be recovered."""
    streams = mobile_campaign()
    gps = streams["gps"].copy()
    gps["lon"].values = np.linspace(179.0, -179.0, 60)
    with pytest.raises(TsaraAlignError, match="antimeridian"):
        attach_positions(streams["aeris"], {**streams, "gps": gps})


def test_a_sub_microsecond_gap_is_accepted() -> None:
    """`pd.Timedelta("1ns").total_seconds()` is 0.0, so reading seconds off a
    duration rejects any gap below a microsecond as non-positive.

    Positivity is tested by comparing Timedelta objects instead, which matches
    the config layer and TSARA's integer-nanosecond convention everywhere else.
    """
    streams = {"gps": make_stream(cells(0.0, 1.0, 4), {"field": np.arange(4.0)})}
    result = interpolate_onto_cells(streams, "field", cells(0.0, 1.0, 4), max_interp_gap="1ns")
    # Every target is exactly on a source midpoint, so none needs bridging.
    assert result.n_exact == 4
