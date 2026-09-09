"""Tests for the uniform output grid and its persistence.

The arithmetic is the binner's and is tested there. What is tested here is the
grid's own two jobs: choosing cells, and refusing a period the data cannot
support. The refusal carries the most weight, because a grid finer than its
widest input repeats one measurement across several cells and every count
downstream then believes there were several.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.align import TsaraAlignError, build_output_grid, grid_cells, load_grid, save_grid
from tsara.config.analysis import OutputGridConfig
from tsara.core.bundle import BUNDLE_GRID_FILE, TsaraBundleError
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


def test_a_period_shorter_than_the_widest_cell_is_refused(
    campaign: dict[str, xr.Dataset],
) -> None:
    """A 60 s value on 1 s cells is the same number sixty times.

    That is resolution the instrument never had, and sixty points where there
    is one measurement. The message has to name which variable forced it and
    what period would work, or the user has no way forward.
    """
    with pytest.raises(TsaraAlignError, match="shorter than the widest selected cell"):
        grid_cells(campaign, OutputGridConfig(freq="1s"))


def test_the_refusal_names_the_offender_and_the_smallest_workable_period(
    campaign: dict[str, xr.Dataset],
) -> None:
    with pytest.raises(TsaraAlignError, match="'canister' has 60 s cells"):
        grid_cells(campaign, OutputGridConfig(freq="1s"))
    with pytest.raises(TsaraAlignError, match="at least 60 s"):
        grid_cells(campaign, OutputGridConfig(freq="1s"))


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
        assert f"n_source_{name}" in grid.data_vars
        assert f"coverage_{name}" in grid.data_vars
    assert grid.sizes["time"] == 10
    assert grid["ch4"].values[0] == pytest.approx(np.arange(60.0).mean())


def test_the_grid_records_what_it_was_built_from(campaign: dict[str, xr.Dataset]) -> None:
    """A reader cannot tell from the columns whether a variable is absent
    because it was excluded or because it had no data."""
    grid = build_output_grid(campaign, OutputGridConfig(freq="60s"), ["ch4", "benzene"])
    assert grid.attrs["tsara_stage"] == "gridded"
    assert grid.attrs["tsara_grid_freq"] == "60s"
    assert grid.attrs["tsara_grid_widest_source_cell_s"] == pytest.approx(60.0)
    assert grid.attrs["tsara_grid_variables"] == "aeris.ch4, canister.benzene"
    assert "co2" not in grid.data_vars


def test_a_cell_with_no_data_stays_empty(campaign: dict[str, xr.Dataset]) -> None:
    """Never interpolated, and the count says so rather than the value alone."""
    gappy = make_stream(cells(0.0, 1.0, 60), {"ch4": np.arange(60.0)})
    grid = build_output_grid({"a": gappy, "b": campaign["canister"]}, OutputGridConfig(freq="60s"))
    assert np.isfinite(grid["ch4"].values[0])
    assert np.isnan(grid["ch4"].values[1:]).all()
    assert grid["n_source_ch4"].values[1:].tolist() == [0] * 9


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
