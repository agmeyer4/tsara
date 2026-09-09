"""Saving and reloading a gridded product.

A grid is written as a single netCDF file inside a bundle directory, beside
whatever else that bundle holds. It does not touch ``bundle.json``: that
descriptor records which stage created the bundle and what streams it wrote,
and a grid is a different stage's product arriving later. Editing another
stage's record would make it say something its writer never said.

What makes the file readable on its own is that it carries its own
provenance: package version, stage, grid period, the widest source cell the
period was validated against, the variables selected, and the propagation
form used for correlated uncertainty. That is what CLAUDE.md §5 asks of every
saved output, and it is why no second descriptor is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

from tsara.core.bundle import BUNDLE_GRID_FILE, TsaraBundleError, pin_time_encoding
from tsara.core.support import check_bounds_intact

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

__all__ = ["load_grid", "save_grid"]

#: What a gridded product's ``tsara_stage`` says, and what loading checks for.
GRID_STAGE = "gridded"


def save_grid(grid: xr.Dataset, path: str | Path) -> Path:
    """Write a gridded product into a bundle directory.

    Parameters
    ----------
    grid : xarray.Dataset
        The product of :func:`~tsara.align.grid.build_output_grid`.
    path : str or pathlib.Path
        Bundle directory. Created if absent; an existing grid is replaced.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    TsaraBundleError
        If ``path`` exists and is not a directory, or the grid has lost its
        cell boundaries.
    """
    bundle = Path(path)
    if bundle.exists() and not bundle.is_dir():
        raise TsaraBundleError(f"Bundle path '{bundle}' exists and is not a directory.")
    bundle.mkdir(parents=True, exist_ok=True)
    # Checked rather than assumed: a grid whose bounds were destroyed upstream
    # -- by a `resample`, which drops the variable and leaves the CF `bounds`
    # attribute dangling -- would otherwise be written as a product claiming
    # cells it does not have.
    check_bounds_intact(grid)
    pin_time_encoding(grid)
    target = bundle / BUNDLE_GRID_FILE
    grid.to_netcdf(target, engine="netcdf4")
    logger.info(
        "Wrote grid to %s (%d cells, %d variables).",
        target,
        grid.sizes.get("time", 0),
        len(grid.data_vars),
    )
    return target


def load_grid(path: str | Path) -> xr.Dataset:
    """Read a gridded product written by :func:`save_grid`.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory, or the grid file itself.

    Returns
    -------
    xarray.Dataset
        The grid, with ``time_bnds`` restored as a coordinate.

    Raises
    ------
    TsaraBundleError
        If the file is missing, or holds a product from a different stage.
    """
    candidate = Path(path)
    target = candidate if candidate.is_file() else candidate / BUNDLE_GRID_FILE
    if not target.is_file():
        raise TsaraBundleError(f"'{target}' is missing; this bundle holds no grid.")
    # `decode_coords="all"` is what brings `time_bnds` back as a coordinate
    # rather than as an ordinary variable, which is where TSARA keeps it.
    with xr.open_dataset(target, engine="netcdf4", decode_coords="all") as opened:
        grid = opened.load()
    stage = grid.attrs.get("tsara_stage")
    if stage != GRID_STAGE:
        raise TsaraBundleError(
            f"'{target}' was written by the '{stage}' stage, not '{GRID_STAGE}'. "
            "Refusing rather than misreading it as a grid."
        )
    logger.info("Loaded grid from %s (%d cells).", target, grid.sizes.get("time", 0))
    return grid
