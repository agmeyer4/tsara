"""Tests for circular statistics.

The evidence here follows METHODS §11.1. Several cases are checkable with a
pencil (four directions at the compass points cancel; 359 and 1 average to
0). One scores the vectorized cell binning against a slow reimplementation
written straight from the definition, which is the check that the clever
spelling did not change the answer. One is a closed form: a wrapped normal
distribution has a known resultant length, so the dispersion has a right
answer that owes nothing to this module.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsara.core.circular import (
    TsaraCircularError,
    bin_circular_onto_cells,
    circular_dispersion,
    circular_mean,
    wrap_degrees,
)
from tsara.core.support import CellBounds

SECOND = 1_000_000_000


def cells(start_s: float, width_s: float, n: int) -> CellBounds:
    """Return ``n`` abutting cells of ``width_s`` starting at ``start_s``."""
    start = (np.arange(n, dtype=np.int64) * int(width_s * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def slow_bin(
    readings: CellBounds, angles: np.ndarray, target: CellBounds
) -> list[tuple[float, float]]:
    """Bin angles onto cells with a Python loop, written from the definition.

    Deliberately naive: for every target cell, walk every reading, compute
    the overlap by hand, and accumulate. It is O(N*M) and unmistakably
    correct, which is the point -- the fast path uses a binary search, an
    index expansion and three ``bincount`` calls, none of which is obviously
    right by inspection.
    """
    out = []
    for t in range(len(target)):
        sin_sum = cos_sum = weight = 0.0
        for s in range(len(readings)):
            if not np.isfinite(angles[s]):
                continue
            lo = max(int(readings.start_ns[s]), int(target.start_ns[t]))
            hi = min(int(readings.stop_ns[s]), int(target.stop_ns[t]))
            overlap = max(hi - lo, 0)
            if overlap == 0:
                continue
            rad = np.radians(angles[s])
            sin_sum += overlap * np.sin(rad)
            cos_sum += overlap * np.cos(rad)
            weight += overlap
        if weight == 0:
            out.append((float("nan"), float("nan")))
            continue
        s_mean, c_mean = sin_sum / weight, cos_sum / weight
        r = float(np.hypot(s_mean, c_mean))
        angle = float(np.degrees(np.arctan2(s_mean, c_mean)) % 360.0)
        out.append((angle, r))
    return out


# ---------------------------------------------------------------------------
# wrap_degrees
# ---------------------------------------------------------------------------


def test_wrap_brings_angles_into_a_single_turn() -> None:
    assert wrap_degrees([-1.0, 0.0, 359.0, 360.0, 720.5]) == pytest.approx(
        [359.0, 0.0, 359.0, 0.0, 0.5]
    )


def test_wrap_passes_nan_through() -> None:
    assert np.isnan(wrap_degrees([np.nan])[0])


# ---------------------------------------------------------------------------
# The case the module exists for
# ---------------------------------------------------------------------------


def test_the_wraparound_case_that_defeats_the_arithmetic_mean() -> None:
    """359 and 1 are both nearly north, and their arithmetic mean is south."""
    result = circular_mean([359.0, 1.0])
    assert result.mean_deg == pytest.approx(0.0, abs=1e-9)
    assert np.mean([359.0, 1.0]) == pytest.approx(180.0)


def test_identical_directions_have_a_resultant_length_of_one() -> None:
    result = circular_mean([42.0, 42.0, 42.0])
    assert result.mean_deg == pytest.approx(42.0)
    assert result.resultant_length == pytest.approx(1.0)
    assert result.dispersion_deg == pytest.approx(0.0)


def test_the_four_compass_points_cancel_exactly() -> None:
    """North, east, south and west sum to nothing, and there is no mean.

    Checkable by hand, and the case where IEEE arithmetic would otherwise
    answer 'due north': ``atan2(0, 0)`` is 0.0, so a direction would be
    reported for a measurement that has none.
    """
    result = circular_mean([0.0, 90.0, 180.0, 270.0])
    assert np.isnan(result.mean_deg)
    assert result.resultant_length == 0.0
    assert np.isinf(result.dispersion_deg)
    assert result.n_readings == 4


def test_two_opposite_directions_also_cancel() -> None:
    """Two samples 180 degrees apart, which is entirely ordinary on a van."""
    result = circular_mean([30.0, 210.0])
    assert np.isnan(result.mean_deg)
    assert result.resultant_length == 0.0


def test_a_spread_of_directions_has_a_mean_between_them() -> None:
    result = circular_mean([350.0, 10.0])
    assert result.mean_deg == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < result.resultant_length < 1.0


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_weights_pull_the_mean_toward_the_heavier_sample() -> None:
    heavy_north = circular_mean([0.0, 90.0], weights=[3.0, 1.0])
    even = circular_mean([0.0, 90.0])
    assert even.mean_deg == pytest.approx(45.0)
    assert heavy_north.mean_deg < even.mean_deg


def test_weights_are_normalized_so_their_scale_does_not_matter() -> None:
    """Overlap in nanoseconds may be passed straight in."""
    small = circular_mean([10.0, 50.0], weights=[1.0, 3.0])
    large = circular_mean([10.0, 50.0], weights=[1e9, 3e9])
    assert small.mean_deg == pytest.approx(large.mean_deg)
    assert small.resultant_length == pytest.approx(large.resultant_length)


def test_zero_weight_samples_do_not_contribute() -> None:
    result = circular_mean([10.0, 200.0], weights=[1.0, 0.0])
    assert result.mean_deg == pytest.approx(10.0)
    assert result.n_readings == 1


def test_circular_mean_rejects_mismatched_weights() -> None:
    with pytest.raises(TsaraCircularError, match="correspond one to one"):
        circular_mean([1.0, 2.0], weights=[1.0])


def test_circular_mean_rejects_negative_weights() -> None:
    with pytest.raises(TsaraCircularError, match="backwards"):
        circular_mean([1.0, 2.0], weights=[1.0, -1.0])


def test_circular_mean_rejects_two_dimensional_input() -> None:
    with pytest.raises(TsaraCircularError, match="one-dimensional"):
        circular_mean(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# Masked and empty
# ---------------------------------------------------------------------------


def test_masked_angles_drop_out() -> None:
    result = circular_mean([10.0, np.nan, 10.0])
    assert result.mean_deg == pytest.approx(10.0)
    assert result.n_readings == 2


def test_nothing_to_average_is_nan_everywhere() -> None:
    """Distinct from cancellation: there R is 0, here it is nan.

    Both mean 'no direction', for opposite reasons -- one had no data, the
    other had data that cancelled -- and a later stage diagnosing an empty
    cell needs to tell them apart.
    """
    result = circular_mean([np.nan, np.nan])
    assert np.isnan(result.mean_deg)
    assert np.isnan(result.resultant_length)
    assert np.isnan(result.dispersion_deg)
    assert result.n_readings == 0


def test_all_zero_weights_is_nan_everywhere() -> None:
    result = circular_mean([10.0, 20.0], weights=[0.0, 0.0])
    assert np.isnan(result.resultant_length)
    assert result.n_readings == 0


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def test_dispersion_is_the_closed_form_for_a_wrapped_normal() -> None:
    """A distribution with a known answer this module had no part in setting.

    A wrapped normal with underlying standard deviation sigma has resultant
    length exp(-sigma^2 / 2), so feeding that R back must return sigma. The
    check runs over three decades of sigma, from a steady wind to a tumbling
    one.
    """
    for sigma_deg in (1.0, 10.0, 45.0, 120.0):
        sigma_rad = np.radians(sigma_deg)
        r = np.exp(-0.5 * sigma_rad**2)
        assert float(circular_dispersion(r)) == pytest.approx(sigma_deg, rel=1e-12)


def test_dispersion_recovers_a_sampled_wrapped_normal() -> None:
    """And the same identity survives actual sampling, not just algebra."""
    rng = np.random.default_rng(20260912)
    sigma_deg = 25.0
    draws = rng.normal(70.0, sigma_deg, size=200_000)
    result = circular_mean(draws)
    assert result.mean_deg == pytest.approx(70.0, abs=0.2)
    assert result.dispersion_deg == pytest.approx(sigma_deg, rel=0.02)


def test_dispersion_is_zero_when_every_sample_agrees() -> None:
    assert float(circular_dispersion(1.0)) == 0.0


def test_dispersion_is_infinite_when_the_vectors_cancel() -> None:
    """Unbounded on purpose: a uniform set of directions has no scale.

    This is the property that separates the exact form from the Yamartino
    approximation, which saturates near 105 degrees and so reports a
    plausible-looking number for a direction that does not exist.
    """
    assert np.isinf(float(circular_dispersion(0.0)))


def test_dispersion_clips_a_resultant_length_above_one() -> None:
    """Rounding can put R a few ULPs above 1, where the logarithm turns."""
    assert float(circular_dispersion(1.0 + 1e-15)) == 0.0


def test_dispersion_passes_nan_through() -> None:
    assert np.isnan(float(circular_dispersion(np.nan)))


def test_dispersion_is_monotone_in_the_resultant_length() -> None:
    values = circular_dispersion([1.0, 0.9, 0.5, 0.1, 0.01])
    assert np.all(np.diff(values) > 0)


# ---------------------------------------------------------------------------
# Binning onto cells
# ---------------------------------------------------------------------------


def test_binning_matches_a_slow_reimplementation() -> None:
    """The fast path against an O(N*M) loop written from the definition.

    The vectorized version uses a binary search, an index expansion and three
    bincount calls, and none of those is obviously right by inspection. The
    loop is.
    """
    rng = np.random.default_rng(20260913)
    readings = cells(0.0, 1.0, 240)
    angles = rng.uniform(0.0, 360.0, size=240)
    angles[::17] = np.nan
    target = cells(0.5, 7.0, 34)
    fast = bin_circular_onto_cells(readings, angles, target)
    slow = slow_bin(readings, angles, target)
    for i, (angle, r) in enumerate(slow):
        if np.isnan(r):
            assert np.isnan(fast.resultant_length[i])
            continue
        assert fast.resultant_length[i] == pytest.approx(r, rel=1e-12)
        assert fast.mean_deg[i] == pytest.approx(angle, rel=1e-9)


def test_binning_onto_the_same_cells_is_the_identity() -> None:
    """An invariant that needs no ground truth to check.

    Averaging a stream onto its own cells must return the stream. If it does
    not, the weighting is wrong in a way no comparison against another
    implementation of the same idea would reveal.
    """
    readings = cells(0.0, 1.0, 50)
    angles = np.linspace(0.0, 359.0, 50)
    result = bin_circular_onto_cells(readings, angles, readings)
    assert result.mean_deg == pytest.approx(angles, rel=1e-9)
    assert result.resultant_length == pytest.approx(np.ones(50))
    assert result.coverage == pytest.approx(np.ones(50))


def test_binning_wraps_across_the_seam() -> None:
    """Four samples straddling north average to north, not to south."""
    readings = cells(0.0, 1.0, 4)
    angles = np.array([358.0, 359.0, 1.0, 2.0])
    target = cells(0.0, 4.0, 1)
    result = bin_circular_onto_cells(readings, angles, target)
    assert result.mean_deg[0] == pytest.approx(0.0, abs=1e-9)
    assert result.n_readings[0] == 4


def test_a_partly_covered_cell_reports_its_coverage() -> None:
    """The canister case: 15 s of data inside a 60 s cell."""
    readings = cells(0.0, 15.0, 1)
    target = cells(0.0, 60.0, 1)
    result = bin_circular_onto_cells(readings, np.array([90.0]), target)
    assert result.mean_deg[0] == pytest.approx(90.0)
    assert result.coverage[0] == pytest.approx(0.25)
    assert result.n_readings[0] == 1


def test_an_empty_cell_is_nan_and_is_not_bridged() -> None:
    readings = cells(0.0, 1.0, 3)
    target = cells(100.0, 1.0, 2)
    result = bin_circular_onto_cells(readings, np.array([1.0, 2.0, 3.0]), target)
    assert np.all(np.isnan(result.mean_deg))
    assert np.all(result.n_readings == 0)
    assert np.all(result.coverage == 0.0)


def test_a_masked_sample_separates_no_data_from_rejected_data() -> None:
    readings = cells(0.0, 1.0, 2)
    target = cells(0.0, 2.0, 1)
    result = bin_circular_onto_cells(readings, np.array([np.nan, np.nan]), target)
    assert np.isnan(result.mean_deg[0])
    assert result.n_readings[0] == 0
    assert result.n_overlapping[0] == 2


def test_a_cancelling_cell_has_no_direction_but_keeps_its_count() -> None:
    readings = cells(0.0, 1.0, 2)
    target = cells(0.0, 2.0, 1)
    result = bin_circular_onto_cells(readings, np.array([0.0, 180.0]), target)
    assert np.isnan(result.mean_deg[0])
    assert result.resultant_length[0] == pytest.approx(0.0, abs=1e-15)
    assert result.n_readings[0] == 2


def test_binning_with_no_reading_cells() -> None:
    empty = CellBounds(start_ns=np.empty(0, dtype=np.int64), stop_ns=np.empty(0, dtype=np.int64))
    result = bin_circular_onto_cells(empty, np.empty(0), cells(0.0, 1.0, 3))
    assert np.all(np.isnan(result.mean_deg))
    assert result.mean_deg.size == 3


def test_binning_with_no_target_cells() -> None:
    empty = CellBounds(start_ns=np.empty(0, dtype=np.int64), stop_ns=np.empty(0, dtype=np.int64))
    result = bin_circular_onto_cells(cells(0.0, 1.0, 3), np.array([1.0, 2.0, 3.0]), empty)
    assert result.mean_deg.size == 0


def test_a_zero_width_target_cell_has_no_coverage_rather_than_a_division() -> None:
    readings = cells(0.0, 1.0, 3)
    target = CellBounds(
        start_ns=np.array([0], dtype=np.int64), stop_ns=np.array([0], dtype=np.int64)
    )
    result = bin_circular_onto_cells(readings, np.array([1.0, 2.0, 3.0]), target)
    assert result.coverage[0] == 0.0


def test_binning_rejects_mismatched_angles() -> None:
    with pytest.raises(TsaraCircularError, match="correspond one to one"):
        bin_circular_onto_cells(cells(0.0, 1.0, 3), np.array([1.0, 2.0]), cells(0.0, 1.0, 1))


# ---------------------------------------------------------------------------
# Two defects this module shipped with for about an hour
# ---------------------------------------------------------------------------
#
# Both were found by tests written before the code was believed, and both are
# the same shape: the mathematics is right and the floating point is not, so
# the wrong answer arrives looking entirely ordinary.


def test_a_tiny_negative_angle_does_not_wrap_to_a_full_turn() -> None:
    """`-1e-17 % 360` is 360.0 in float64, outside the documented [0, 360).

    Mathematically the modulo cannot reach a full turn; numerically it rounds
    up to one. It reached the mean direction too, so two readings either side
    of north averaged to 360.0 rather than 0.0 -- the same direction, spelled
    in a way that breaks any comparison, sort or bin boundary at zero.
    """
    # Shape-preserving like a ufunc, so a scalar comes back 0-dimensional.
    assert float(wrap_degrees(-1e-17)) == 0.0
    assert wrap_degrees([-1e-18, -1e-300])[0] == 0.0
    assert circular_mean([359.0, 1.0]).mean_deg < 1.0


def test_cancelling_directions_do_not_report_rounding_noise() -> None:
    """sin(180 degrees) is 1.22e-16, so mathematical cancellation is not numerical.

    A north/south pair leaves a residual vector of length 6.1e-17 pointing due
    east, and atan2 reports 90.000 degrees with complete confidence. The guard
    is the rounding floor of the sum that produced it, not a judgement about
    wind.
    """
    result = circular_mean([0.0, 180.0])
    assert np.isnan(result.mean_deg)
    assert result.resultant_length == 0.0
    assert np.isinf(result.dispersion_deg)


def test_a_cancelled_direction_is_reproducible_under_reordering() -> None:
    """The argument that settled the design.

    Measured before the guard existed, the four compass points summed in three
    different orders gave 129.60, 153.43 and 132.19 degrees. A number that
    changes when its inputs are reordered is not a measurement, so it must not
    be reported as one.
    """
    orders = [
        [0.0, 90.0, 180.0, 270.0],
        [270.0, 180.0, 90.0, 0.0],
        [90.0, 270.0, 0.0, 180.0],
    ]
    for order in orders:
        assert np.isnan(circular_mean(order).mean_deg)


def test_a_genuine_direction_survives_the_rounding_guard() -> None:
    """The guard must not eat real answers.

    A resultant length of 1e-6 is scientifically useless and numerically
    solid -- ten orders of magnitude above the rounding floor -- so it is
    reported, with the R that says how much to trust it.
    """
    # Two directions a hair under 180 degrees apart leave a small but real
    # resultant.
    result = circular_mean([0.0, 180.0 - 1e-4])
    assert not np.isnan(result.mean_deg)
    assert 0.0 < result.resultant_length < 1e-5


def test_the_binned_path_has_the_same_guard_as_the_scalar_one() -> None:
    """Two code paths, one rule; they were written separately."""
    readings = cells(0.0, 1.0, 4)
    target = cells(0.0, 4.0, 1)
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    binned = bin_circular_onto_cells(readings, angles, target)
    scalar = circular_mean(angles)
    assert np.isnan(binned.mean_deg[0]) and np.isnan(scalar.mean_deg)
    assert binned.resultant_length[0] == scalar.resultant_length == 0.0
    assert np.isinf(binned.dispersion_deg[0]) and np.isinf(scalar.dispersion_deg)
