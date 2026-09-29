"""Tests for windows as cells (tsara.baseline.windows)."""

from __future__ import annotations

import numpy as np
import pytest

from tsara.baseline import TsaraBaselineError, duration_ns, window_cells
from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND


def cells(start_s: float, width_s: float, n: int, step_s: float | None = None) -> CellBounds:
    """Return ``n`` cells of ``width_s`` whose starts are ``step_s`` (default: the width) apart."""
    step = width_s if step_s is None else step_s
    start = (np.arange(n, dtype=np.int64) * int(step * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def test_windows_are_centred_on_each_cell_midpoint_and_exactly_as_wide_as_asked() -> None:
    readings = cells(0.0, 1.0, 5, step_s=2.3)
    windows = window_cells(readings, 10 * SECOND)
    assert np.array_equal(windows.width_ns, np.full(5, 10 * SECOND))
    assert np.array_equal(windows.midpoint_ns, readings.midpoint_ns)
    assert np.array_equal(windows.start_ns, readings.midpoint_ns - 5 * SECOND)


def test_an_odd_window_is_centred_the_way_a_cell_is_centred_on_its_timestamp() -> None:
    """start = midpoint - window // 2, the rule CellBounds.from_label uses, so the odd
    nanosecond falls on the same side in both places and the width stays exact."""
    readings = cells(0.0, 1.0, 3)
    windows = window_cells(readings, 7)
    assert np.array_equal(windows.start_ns, readings.midpoint_ns - 3)
    assert np.array_equal(windows.width_ns, np.full(3, 7))


@pytest.mark.parametrize("window_ns", [0, -1, -SECOND])
def test_a_window_of_no_duration_is_refused(window_ns: int) -> None:
    with pytest.raises(TsaraBaselineError, match="positive duration"):
        window_cells(cells(0.0, 1.0, 3), window_ns)


def test_windows_can_be_centred_on_any_cells() -> None:
    """A single cell, or cells that are not a stream's: the function does not care."""
    one = CellBounds(start_ns=np.array([5 * SECOND]), stop_ns=np.array([20 * SECOND]))
    windows = window_cells(one, 60 * SECOND)
    assert len(windows) == 1
    assert int(windows.midpoint_ns[0]) == int(one.midpoint_ns[0])


@pytest.mark.parametrize(
    ("spec", "expected_s"),
    [("10min", 600), ("2min", 120), ("90s", 90), ("6h", 21600), ("1.5s", 1.5)],
)
def test_duration_ns_parses_the_config_spelling(spec: str, expected_s: float) -> None:
    assert duration_ns(spec) == int(expected_s * SECOND)


@pytest.mark.parametrize("spec", ["0s", "-5min", "soon", ""])
def test_a_non_positive_or_unparsable_duration_is_refused(spec: str) -> None:
    with pytest.raises(TsaraBaselineError, match="duration"):
        duration_ns(spec)
