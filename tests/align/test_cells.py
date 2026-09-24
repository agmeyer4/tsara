"""Tests for the cell geometry every join asks about before averaging anything.

Each function here is exact in integer nanoseconds with no tolerance, so the
tests are pencil-checkable fixtures (METHODS §11.1): a reading over two cells,
a reading the search visits at zero overlap, a single nanosecond of overlap,
a tiling half a cell out of phase. The width ratio and the copy line are
exercised through the binner's refusals in `test_binning.py`, including the
27-row probe table, because the verdict is the binner's.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tsara.align.cells import (
    phase_offset_s,
    readings_behind,
    shared_readings,
    stream_cells,
    targets_overlap,
)
from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND


def cells(start_s: float, width_s: float, n: int) -> CellBounds:
    """Return ``n`` abutting cells of ``width_s`` starting at ``start_s``."""
    start = (np.arange(n, dtype=np.int64) * int(width_s * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def spaced(start_s: float, width_s: float, n: int, step_s: float) -> CellBounds:
    """Return ``n`` cells of ``width_s`` whose starts are ``step_s`` apart."""
    start = (np.arange(n, dtype=np.int64) * int(round(step_s * SECOND))) + int(
        round(start_s * SECOND)
    )
    return CellBounds(start_ns=start, stop_ns=start + int(round(width_s * SECOND)))


def bounds(start_s: list[float], stop_s: list[float]) -> CellBounds:
    """Return cells from explicit starts and stops in seconds."""
    return CellBounds(
        start_ns=(np.array(start_s) * SECOND).astype(np.int64),
        stop_ns=(np.array(stop_s) * SECOND).astype(np.int64),
    )


def make_stream(
    start_s: float,
    width_s: float,
    n: int,
    variables: dict[str, np.ndarray],
    *,
    attrs: dict[str, dict[str, object]] | None = None,
) -> xr.Dataset:
    """Build a minimal stream with CF cells."""
    bounds = cells(start_s, width_s, n)
    variable_attrs = attrs or {}
    dataset = xr.Dataset(
        data_vars={
            name: ("time", values, dict(variable_attrs.get(name, {"units": "ppb"})))
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


def test_shared_readings_counts_finite_readings_forming_two_cells_and_ignores_touching() -> None:
    """A reading over two cells is shared; one the search visits at zero overlap is not.

    Readings [0, 4) and [1, 1.5) against cells [1, 2) and [2, 3). The wide one
    forms both cells. The nested one forms [1, 2) only, but the overlap search
    still visits it for [2, 3) with an overlap of zero, which must count for
    nothing -- the same membership rule as ``readings_behind``. Masking the wide
    reading takes the count to zero: a masked reading formed nothing.
    """
    readings = CellBounds(
        start_ns=np.array([0, SECOND], dtype=np.int64),
        stop_ns=np.array([4 * SECOND, SECOND + SECOND // 2], dtype=np.int64),
    )
    target = cells(1.0, 1.0, 2)
    stream = make_stream(0.0, 1.0, 2, {"v": np.array([1.0, 2.0])})
    assert shared_readings(stream, "v", readings, target) == 1
    masked = make_stream(0.0, 1.0, 2, {"v": np.array([np.nan, 2.0])})
    assert shared_readings(masked, "v", readings, target) == 0


def test_readings_behind_counts_distinct_finite_contributors() -> None:
    """Five readings, one masked, spread over three 2 s cells: four readings."""
    stream = make_stream(0.0, 1.0, 5, {"ch4": np.array([1.0, np.nan, 3.0, 4.0, 5.0])})
    count = readings_behind(stream, "ch4", stream_cells(stream, "a"), cells(0.0, 2.0, 3))
    assert count == 4


def test_targets_overlap_is_exact_and_order_free() -> None:
    assert not targets_overlap(spaced(0, 60, 5, 60))
    assert targets_overlap(bounds([0, 1], [2, 3]))
    # A single nanosecond of overlap counts; abutting cells do not.
    one_ns = CellBounds(
        start_ns=np.array([0, SECOND - 1], dtype=np.int64),
        stop_ns=np.array([SECOND, 2 * SECOND], dtype=np.int64),
    )
    assert targets_overlap(one_ns)
    assert not targets_overlap(bounds([5, 0], [6, 5]))  # unsorted, abutting
    assert not targets_overlap(bounds([0], [1]))


def test_phase_offset_is_reported_only_for_equal_widths_out_of_phase() -> None:
    """Same width and offset: the offset. Anything else: None, and no tolerance anywhere."""
    grid = spaced(0, 60, 10, 60)
    assert phase_offset_s(spaced(30, 60, 9, 60), grid) == pytest.approx(30.0)
    assert phase_offset_s(spaced(10, 60, 9, 60), grid) == pytest.approx(10.0)
    assert phase_offset_s(spaced(0, 60, 10, 60), grid) is None  # in phase
    assert phase_offset_s(spaced(0, 1.023, 400, 1.023), spaced(0, 1, 409, 1)) is None  # jitter
    assert phase_offset_s(spaced(0, 1, 600, 1), grid) is None  # narrower readings
    empty = CellBounds(start_ns=np.array([], dtype=np.int64), stop_ns=np.array([], dtype=np.int64))
    assert phase_offset_s(empty, grid) is None
    assert phase_offset_s(grid, empty) is None
    # A degenerate target of zero width has no phase to be out of.
    assert phase_offset_s(bounds([0, 5], [0, 5]), bounds([0, 5], [0, 5])) is None
    # Mixed target widths: not a uniform tiling, so no phase question either.
    assert phase_offset_s(spaced(0, 60, 4, 60), bounds([0, 60, 120], [60, 120, 150])) is None
