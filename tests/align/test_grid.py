"""Tests for the uniform output grid and its persistence.

The arithmetic is the binner's and is tested there. What is tested here is the
grid's own two jobs: choosing cells, and refusing a period the data cannot
support. The refusal carries the most weight, because a grid finer than its
widest input repeats one measurement across several cells and every count
downstream then believes there were several.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.align import TsaraAlignError, build_output_grid, grid_cells, load_grid, save_grid
from tsara.config.analysis import OutputGridConfig
from tsara.core.bundle import BUNDLE_GRID_FILE, TsaraBundleError
from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND


def cells(start_s: float, width_s: float, n: int) -> CellBounds:
    """Return ``n`` abutting cells of ``width_s`` starting at ``start_s``."""
    start = (np.arange(n, dtype=np.int64) * int(width_s * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def make_stream(
    bounds: CellBounds,
    variables: dict[str, np.ndarray],
    *,
    attrs: dict[str, dict[str, object]] | None = None,
) -> xr.Dataset:
    """Build a minimal stream with CF cells."""
    variable_attrs = attrs or {}
    dataset = xr.Dataset(
        data_vars={
            name: ("time", values, dict(variable_attrs.get(name, {"units": "ppb", "role": "gas"})))
            for name, values in variables.items()
        },
        coords={
            "time": bounds.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
    )
    dataset["time"].attrs["bounds"] = "time_bnds"
    for name in variables:
        dataset[name].attrs.setdefault("cell_methods", "time: point")
    return dataset


@pytest.fixture()
def campaign() -> dict[str, xr.Dataset]:
    """A fast instrument, a slow one, and a very slow one."""
    return {
        "aeris": make_stream(cells(0.0, 1.0, 600), {"ch4": np.arange(600.0)}),
        "picarro": make_stream(cells(0.0, 10.0, 60), {"co2": np.arange(60.0)}),
        "canister": make_stream(cells(0.0, 60.0, 10), {"benzene": np.arange(10.0)}),
    }


# ---------------------------------------------------------------------------
# The period rule
# ---------------------------------------------------------------------------


def test_a_period_too_fine_for_a_selected_instrument_is_refused(
    campaign: dict[str, xr.Dataset],
) -> None:
    """A 60 s value on 1 s cells is the same number sixty times.

    That is resolution the instrument never had, and sixty points where there
    is one measurement. The message has to name which instrument forced it and
    a period that would work, or the user has no way forward.
    """
    with pytest.raises(TsaraAlignError) as raised:
        grid_cells(campaign, OutputGridConfig(freq="1s"))
    message = str(raised.value)
    assert "too fine for 'canister'" in message
    assert "cover 60 grid cells' worth of time" in message
    assert "longer than 30 s" in message


def test_the_period_that_the_refusal_suggests_does_work(campaign: dict[str, xr.Dataset]) -> None:
    """Half the widest reading is the line: 30 s is refused, 31 s is not.

    A 60 s reading on 31 s cells spans two rows, which is sharing rather than
    replication and is counted by the readings record instead.
    """
    with pytest.raises(TsaraAlignError):
        grid_cells(campaign, OutputGridConfig(freq="30s"))
    assert len(grid_cells(campaign, OutputGridConfig(freq="31s"))) > 0


def test_jitter_just_wider_than_the_period_is_not_refused() -> None:
    """The 2026 LANL Aeris measures its cells at 1.023 s against a nominal 1 s.

    The rule this replaced compared median widths and refused a one-second grid
    for it, although the binner accepts the same cells and the only effect is a
    little over two percent more rows than readings.
    """
    aeris = make_stream(cells(0.0, 1.023, 600), {"ch4": np.arange(600.0)})
    grid = build_output_grid({"aeris": aeris}, OutputGridConfig(freq="1s"))
    assert grid.sizes["time"] > 600


def test_a_two_second_record_is_refused_on_a_one_second_grid_at_any_phase() -> None:
    """The phase hole the binner's former rule had, closed for the grid too."""
    for offset in (0.0, 0.3, 0.5):
        picarro = make_stream(cells(offset, 2.0, 100), {"co2": np.arange(100.0)})
        with pytest.raises(TsaraAlignError, match="cover 2 grid cells' worth"):
            grid_cells({"picarro": picarro}, OutputGridConfig(freq="1s"))


def test_a_wide_instrument_entirely_outside_the_window_constrains_nothing(
    campaign: dict[str, xr.Dataset],
) -> None:
    """The rule is checked against the cells the grid actually has."""
    late = make_stream(cells(100_000.0, 60.0, 5), {"benzene": np.arange(5.0)})
    window = OutputGridConfig(
        freq="1s",
        start=pd.Timestamp("1970-01-01 00:00:00").to_pydatetime(),
        end=pd.Timestamp("1970-01-01 00:10:00").to_pydatetime(),
    )
    bounds = grid_cells({"aeris": campaign["aeris"], "canister": late}, window)
    assert len(bounds) == 600


def test_excluding_the_slow_instrument_permits_a_finer_grid(
    campaign: dict[str, xr.Dataset],
) -> None:
    """Which variables go in is a real lever, not a formality.

    This is the whole reason the rule is checked against the *selection*
    rather than against every stream the campaign happens to contain: a run
    that does not need the canisters should not be coarsened by them.
    """
    with pytest.raises(TsaraAlignError):
        grid_cells(campaign, OutputGridConfig(freq="10s"))
    fine = grid_cells(campaign, OutputGridConfig(freq="10s"), ["ch4", "co2"])
    assert len(fine) > 0
    finer = grid_cells(campaign, OutputGridConfig(freq="1s"), ["ch4"])
    assert len(finer) > len(fine)


def test_a_grid_may_not_be_gridded_again(campaign: dict[str, xr.Dataset]) -> None:
    """Re-gridding is the case METHODS §11.2.3 refuses, reached through the grid.

    A coarser grid built from a finer one looks like an obvious shortcut and is
    the one route by which a product's rows would be weighted as though they
    were measurements. Both entry points refuse it, because both resolve their
    selection through the same function.
    """
    fine = build_output_grid(campaign, OutputGridConfig(freq="60s"))
    assert fine.attrs["tsara_stage"] == "gridded"
    for call in (
        lambda: build_output_grid({"grid": fine}, OutputGridConfig(freq="300s")),
        lambda: grid_cells({"grid": fine}, OutputGridConfig(freq="300s")),
    ):
        with pytest.raises(TsaraAlignError) as refused:
            call()
        assert "gridded" in str(refused.value)
        assert "native streams" in str(refused.value)


def test_the_coarser_grid_the_refusal_asks_for_is_buildable(
    campaign: dict[str, xr.Dataset],
) -> None:
    """A sweep over grid period rebuilds from the streams, which is exact."""
    coarse = build_output_grid(campaign, OutputGridConfig(freq="300s"))
    assert coarse.sizes["time"] == 2
    assert coarse["ch4"].attrs["tsara_grid_readings"] == 600


def test_a_period_exactly_equal_to_the_widest_cell_is_allowed(
    campaign: dict[str, xr.Dataset],
) -> None:
    """The rule is 'at least', so equality is the intended working case."""
    assert len(grid_cells(campaign, OutputGridConfig(freq="60s"))) == 10


# ---------------------------------------------------------------------------
# Where the cells fall
# ---------------------------------------------------------------------------


def test_cells_abut_and_are_exactly_the_requested_period(
    campaign: dict[str, xr.Dataset],
) -> None:
    bounds = grid_cells(campaign, OutputGridConfig(freq="60s"))
    assert np.all(bounds.width_ns == 60 * SECOND)
    assert np.array_equal(bounds.start_ns[1:], bounds.stop_ns[:-1])


def test_the_default_start_is_anchored_to_the_epoch_not_to_the_data() -> None:
    """Two runs over overlapping periods must produce cells that line up.

    Anchoring to whenever the data happened to start would give two grids
    that never share a boundary, and their outputs could not be compared at
    all.
    """
    late = make_stream(cells(137.0, 1.0, 100), {"ch4": np.arange(100.0)})
    bounds = grid_cells({"a": late}, OutputGridConfig(freq="60s"))
    assert int(bounds.start_ns[0]) == 120 * SECOND
    assert int(bounds.start_ns[0]) % (60 * SECOND) == 0


def test_an_explicit_window_is_honoured_exactly(campaign: dict[str, xr.Dataset]) -> None:
    config = OutputGridConfig(
        freq="60s",
        start=pd.Timestamp("1970-01-01 00:01:00").to_pydatetime(),
        end=pd.Timestamp("1970-01-01 00:04:00").to_pydatetime(),
    )
    bounds = grid_cells(campaign, config)
    assert len(bounds) == 3
    assert int(bounds.start_ns[0]) == 60 * SECOND


def test_a_one_sided_window_that_misses_the_data_is_refused(
    campaign: dict[str, xr.Dataset],
) -> None:
    """The schema already forbids start >= end, so this is the reachable case.

    Give only a start, after the record ends, and the other edge comes from
    the data -- so the window closes before it opens. Same for an end before
    the record begins. A user slicing a window outside their own record is an
    ordinary mistake and deserves a sentence rather than an empty result.
    """
    after = OutputGridConfig(freq="60s", start=pd.Timestamp("1971-01-01").to_pydatetime())
    before = OutputGridConfig(freq="60s", end=pd.Timestamp("1969-01-01").to_pydatetime())
    with pytest.raises(TsaraAlignError, match="window is empty"):
        grid_cells(campaign, after)
    with pytest.raises(TsaraAlignError, match="window is empty"):
        grid_cells(campaign, before)


def test_no_variables_selected_is_refused() -> None:
    bare = xr.Dataset(
        coords={
            "time": cells(0.0, 1.0, 4).midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([cells(0.0, 1.0, 4).start_ns, cells(0.0, 1.0, 4).stop_ns], axis=1).astype(
                    "datetime64[ns]"
                ),
            ),
        }
    )
    bare["time"].attrs["bounds"] = "time_bnds"
    with pytest.raises(TsaraAlignError, match="nothing to build a grid for"):
        grid_cells({"a": bare}, OutputGridConfig(freq="60s"))


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------


def test_the_grid_holds_every_selected_variable_with_its_qualifiers(
    campaign: dict[str, xr.Dataset],
) -> None:
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"))
    for name in ("ch4", "co2", "benzene"):
        assert name in grid.data_vars
        assert f"n_readings_{name}" in grid.data_vars
        assert f"coverage_{name}" in grid.data_vars
    assert grid.sizes["time"] == 10
    assert grid["ch4"].values[0] == pytest.approx(np.arange(60.0).mean())


def test_the_grid_records_what_it_was_built_from(campaign: dict[str, xr.Dataset]) -> None:
    """A reader cannot tell from the columns whether a variable is absent
    because it was excluded or because it had no data."""
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"), ["ch4", "benzene"])
    assert grid.attrs["tsara_stage"] == "gridded"
    assert grid.attrs["tsara_grid_freq"] == "60s"
    assert grid.attrs["tsara_grid_widest_reading_cell_s"] == pytest.approx(60.0)
    assert grid.attrs["tsara_grid_variables"] == "aeris.ch4, canister.benzene"
    assert "co2" not in grid.data_vars


def test_each_column_records_the_readings_behind_it(campaign: dict[str, xr.Dataset]) -> None:
    """Six hundred 1 s readings behind ten rows; ten 60 s readings behind ten."""
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"), ["ch4", "benzene"])
    assert grid["ch4"].attrs["tsara_grid_readings"] == 600
    assert grid["benzene"].attrs["tsara_grid_readings"] == 10


def test_a_masked_reading_is_not_counted_behind_a_column() -> None:
    """Ten of 120 readings masked: 110 readings behind the column, not 120."""
    values = np.arange(120.0)
    values[::12] = np.nan
    fast = make_stream(cells(0.0, 1.0, 120), {"ch4": values})
    grid = build_output_grid({"fast": fast}, OutputGridConfig(freq="60s"))
    assert grid["ch4"].attrs["tsara_grid_readings"] == 110


def test_the_widest_cell_attribute_is_the_widest_single_cell() -> None:
    """Canister fills vary; the record is the widest one, not a typical one."""
    starts = np.array([0, 100, 200], dtype=np.int64) * SECOND
    widths = np.array([14, 15, 17], dtype=np.int64) * SECOND
    canister = make_stream(
        CellBounds(start_ns=starts, stop_ns=starts + widths), {"benzene": np.ones(3)}
    )
    grid = build_output_grid({"iwas": canister}, OutputGridConfig(freq="60s"))
    assert grid.attrs["tsara_grid_widest_reading_cell_s"] == pytest.approx(17.0)


def test_a_fill_straddling_two_rows_is_counted_once_and_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The 2024 drives: 68 of 261 canister fills landed in two 60 s rows.

    Here one 15 s fill sits inside a minute and one crosses the boundary at
    120 s: three rows hold benzene, and there are two readings behind them.
    """
    starts = np.array([20, 112], dtype=np.int64) * SECOND
    fills = CellBounds(start_ns=starts, stop_ns=starts + 15 * SECOND)
    canister = make_stream(fills, {"benzene": np.array([1.0, 2.0])})
    fast = make_stream(cells(0.0, 1.0, 180), {"ch4": np.arange(180.0)})
    with caplog.at_level("WARNING", logger="tsara.align.grid"):
        grid = build_output_grid({"fast": fast, "iwas": canister}, OutputGridConfig(freq="60s"))
    assert int((grid["n_readings_benzene"].values > 0).sum()) == 3
    assert grid["benzene"].attrs["tsara_grid_readings"] == 2
    assert "1 grid column(s) hold values in more rows than they have readings" in caplog.text
    assert "Worst: 'benzene', 3 rows from 2 readings" in caplog.text


def test_no_readings_warning_when_every_row_has_its_own_readings(
    caplog: pytest.LogCaptureFixture, campaign: dict[str, xr.Dataset]
) -> None:
    with caplog.at_level("WARNING", logger="tsara.align.grid"):
        build_output_grid(campaign, OutputGridConfig(freq="60s"))
    assert "more rows than they have readings" not in caplog.text


def test_a_readings_warning_lists_at_most_eight_columns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A canister carries dozens of VOCs on one sampling pattern; one warning, not dozens."""
    starts = np.array([50], dtype=np.int64) * SECOND
    fills = CellBounds(start_ns=starts, stop_ns=starts + 15 * SECOND)
    many = {f"voc{k:02d}": np.array([float(k)]) for k in range(10)}
    canister = make_stream(fills, many)
    fast = make_stream(cells(0.0, 1.0, 120), {"ch4": np.arange(120.0)})
    with caplog.at_level("WARNING", logger="tsara.align.grid"):
        build_output_grid({"fast": fast, "iwas": canister}, OutputGridConfig(freq="60s"))
    assert caplog.text.count("more rows than they have readings") == 1
    assert "10 grid column(s)" in caplog.text
    assert caplog.text.rstrip().endswith("...")


def test_a_cell_with_no_data_stays_empty(campaign: dict[str, xr.Dataset]) -> None:
    """Never interpolated, and the count says so rather than the value alone."""
    gappy = make_stream(cells(0.0, 1.0, 60), {"ch4": np.arange(60.0)})
    grid = build_output_grid({"a": gappy, "b": campaign["canister"]}, OutputGridConfig(freq="60s"))
    assert np.isfinite(grid["ch4"].values[0])
    assert np.isnan(grid["ch4"].values[1:]).all()
    assert grid["n_readings_ch4"].values[1:].tolist() == [0] * 9


def test_the_canister_column_is_mostly_empty_on_a_fine_grid() -> None:
    """The measured argument for why a pair-specific clock exists.

    A sparse instrument on a campaign grid is mostly absent, which is honest
    and is exactly why a two-species ratio does not use this product.
    """
    fast = make_stream(cells(0.0, 1.0, 3600), {"ch4": np.arange(3600.0)})
    sparse = make_stream(cells(0.0, 15.0, 6), {"benzene": np.arange(6.0)})
    grid = build_output_grid({"a": fast, "b": sparse}, OutputGridConfig(freq="60s"))
    filled = np.isfinite(grid["benzene"].values).mean()
    assert filled < 0.2
    assert np.isfinite(grid["ch4"].values).mean() == 1.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_grid_round_trips_through_a_bundle(
    tmp_path: Path, campaign: dict[str, xr.Dataset]
) -> None:
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"))
    written = save_grid(grid, tmp_path / "bundle")
    assert written.name == BUNDLE_GRID_FILE
    reloaded = load_grid(tmp_path / "bundle")
    assert reloaded["ch4"].values == pytest.approx(grid["ch4"].values)
    assert reloaded.attrs["tsara_grid_freq"] == "60s"
    assert set(reloaded.data_vars) == set(grid.data_vars)


def test_the_reloaded_grid_still_has_its_cells(
    tmp_path: Path, campaign: dict[str, xr.Dataset]
) -> None:
    """`decode_coords="all"` is what brings `time_bnds` back as a coordinate
    rather than as an ordinary variable, which is where TSARA keeps it."""
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"))
    save_grid(grid, tmp_path / "bundle")
    reloaded = load_grid(tmp_path / "bundle")
    assert "time_bnds" in reloaded.coords
    assert np.array_equal(reloaded["time_bnds"].values, grid["time_bnds"].values)


def test_a_grid_file_can_be_loaded_by_its_own_path(
    tmp_path: Path, campaign: dict[str, xr.Dataset]
) -> None:
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"))
    written = save_grid(grid, tmp_path / "bundle")
    assert load_grid(written).sizes["time"] == grid.sizes["time"]


def test_saving_over_an_existing_grid_replaces_it(
    tmp_path: Path, campaign: dict[str, xr.Dataset]
) -> None:
    save_grid(build_output_grid(campaign, OutputGridConfig(freq="60s")), tmp_path / "b")
    save_grid(build_output_grid(campaign, OutputGridConfig(freq="120s")), tmp_path / "b")
    assert load_grid(tmp_path / "b").attrs["tsara_grid_freq"] == "120s"


def test_saving_onto_a_file_is_refused(tmp_path: Path, campaign: dict[str, xr.Dataset]) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(TsaraBundleError, match="is not a directory"):
        save_grid(build_output_grid(campaign, OutputGridConfig(freq="60s")), blocker)


def test_loading_a_bundle_with_no_grid_says_so(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(TsaraBundleError, match="holds no grid"):
        load_grid(tmp_path / "empty")


def test_loading_another_stage_s_product_is_refused_rather_than_misread(
    tmp_path: Path, campaign: dict[str, xr.Dataset]
) -> None:
    """A stream and a grid are both netCDF files with a time axis.

    Reading one as the other would produce a plausible object with the wrong
    meaning, which is what the stage label exists to prevent.
    """
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"))
    grid.attrs["tsara_stage"] = "ingest"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    grid.to_netcdf(bundle / BUNDLE_GRID_FILE, engine="netcdf4")
    with pytest.raises(TsaraBundleError, match="was written by the 'ingest' stage"):
        load_grid(bundle)


def test_saving_a_grid_whose_bounds_were_destroyed_is_refused(
    tmp_path: Path, campaign: dict[str, xr.Dataset]
) -> None:
    """`resample` drops the bounds variable and leaves the CF attribute
    pointing at nothing. Writing that would produce a product claiming cells
    it does not have."""
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"))
    broken = grid.drop_vars("time_bnds")
    with pytest.raises(Exception, match="bounds"):
        save_grid(broken, tmp_path / "bundle")


def test_saving_another_stage_s_product_as_a_grid_is_refused(
    tmp_path: Path, campaign: dict[str, xr.Dataset]
) -> None:
    """What save_grid writes, load_grid must read.

    A paired product is written with `to_netcdf`. Before this was refused,
    save_grid wrote it as `grid.nc` without complaint and load_grid then
    refused the file, so the mistake surfaced only when someone loaded it.
    """
    from tsara.align import pair_species

    paired = pair_species(campaign, "ch4", "co2").dataset
    with pytest.raises(TsaraBundleError, match="tsara_stage is 'paired'"):
        save_grid(paired, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def sparse_grid() -> xr.Dataset:
    """A one-second grid that is mostly empty, like a campaign of separate drives."""
    early = make_stream(cells(0.0, 1.0, 600), {"ch4": np.linspace(1900.0, 2100.0, 600)})
    late = make_stream(cells(20_000.0, 1.0, 600), {"co2": np.linspace(420.0, 440.0, 600)})
    return build_output_grid({"a": early, "b": late}, OutputGridConfig(freq="1s"))


def test_compression_shrinks_a_sparse_grid_and_changes_no_value(tmp_path: Path) -> None:
    grid = sparse_grid()
    plain = save_grid(grid, tmp_path / "plain")
    small = save_grid(grid, tmp_path / "small", compression=4)
    assert small.stat().st_size < plain.stat().st_size / 3
    back = load_grid(tmp_path / "small")
    for name in grid.data_vars:
        assert np.array_equal(grid[name].values, back[name].values, equal_nan=True), name
    assert np.array_equal(grid["time_bnds"].values, back["time_bnds"].values)


def test_compression_keeps_the_pinned_time_encoding(tmp_path: Path) -> None:
    """Compression is merged into the time axis's encoding, never replaces it.

    Replacing it drops the pinned units, and xarray then chooses its own: in the
    walkthrough that wrote `time` as seconds since 00:00:00.5 and its bounds as
    seconds since 00:00:00, two epochs for one axis, with a CF warning. Values
    still round-trip on whole-second cells, which is why only the units show it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        written = save_grid(sparse_grid(), tmp_path / "bundle", compression=4)
    with xr.open_dataset(written, engine="netcdf4", decode_times=False) as raw:
        assert raw["time"].attrs["units"] == "nanoseconds since 1970-01-01"
        assert raw["time"].encoding.get("zlib") is True


def test_the_default_writes_uncompressed(tmp_path: Path) -> None:
    written = save_grid(sparse_grid(), tmp_path / "bundle")
    with xr.open_dataset(written, engine="netcdf4") as opened:
        assert not opened["ch4"].encoding.get("zlib", False)


@pytest.mark.parametrize("level", [0, 10, True, 4.0, "4"])
def test_a_compression_level_outside_one_to_nine_is_refused(
    tmp_path: Path, level: object, campaign: dict[str, xr.Dataset]
) -> None:
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"))
    with pytest.raises(TsaraBundleError, match="zlib level from 1 to 9"):
        save_grid(grid, tmp_path / "bundle", compression=level)  # type: ignore[arg-type]
