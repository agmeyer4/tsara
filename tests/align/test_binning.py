"""Tests for the joining primitive.

This is the one operation everything with more than one clock goes through, so
the evidence follows METHODS §11.1. `test_pairing.py` covers the arithmetic
against a hand fixture, an independent reimplementation and two closed forms;
what is tested here is what the *general* form adds over the pairwise one:
many variables at once, angular variables, name collisions, streams already on
the target support, and the promise that a variable it has never seen a name
for is handled the same as one it has.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tsara.align import TsaraAlignError, bin_streams_onto_cells, resolve_variable
from tsara.align.binning import readings_behind, stream_cells
from tsara.core.naming import sigma_rand_name
from tsara.core.support import CellBounds

SECOND = 1_000_000_000


def cells(start_s: float, width_s: float, n: int) -> CellBounds:
    """Return ``n`` abutting cells of ``width_s`` starting at ``start_s``."""
    start = (np.arange(n, dtype=np.int64) * int(width_s * SECOND)) + int(start_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


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


# ---------------------------------------------------------------------------
# The general shape
# ---------------------------------------------------------------------------


def test_many_variables_from_many_streams_land_on_one_support() -> None:
    """The thing the pairwise form could not do.

    Three instruments at three rates, every variable on one set of cells,
    each carrying its own count and coverage.
    """
    streams = {
        "aeris": make_stream(0.0, 0.5, 120, {"ch4": np.arange(120.0), "c2h6": np.arange(120.0)}),
        "picarro": make_stream(0.0, 2.0, 30, {"co2": np.arange(30.0)}),
        "met": make_stream(0.0, 10.0, 6, {"temperature": np.arange(6.0)}),
    }
    joined = bin_streams_onto_cells(streams, cells(0.0, 10.0, 6))
    for name in ("ch4", "c2h6", "co2", "temperature"):
        assert name in joined.data_vars
        assert f"n_source_{name}" in joined.data_vars
        assert f"coverage_{name}" in joined.data_vars
    assert joined.sizes["time"] == 6
    assert joined["ch4"].values[0] == pytest.approx(np.arange(20.0).mean())


def test_the_default_selection_takes_every_non_sigma_variable() -> None:
    """Sigma companions are not variables in their own right.

    Each travels with the value it describes; selecting one directly would
    produce a column with no parent and no meaning.
    """
    stream = make_stream(
        0.0,
        1.0,
        8,
        {"ch4": np.arange(8.0), sigma_rand_name("ch4"): np.full(8, 2.0)},
    )
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 4.0, 2))
    assert "ch4" in joined.data_vars
    # Present because it travelled with ch4, not because it was selected.
    assert sigma_rand_name("ch4") in joined.data_vars
    assert joined[sigma_rand_name("ch4")].attrs["uncertainty_component"] == "random"


def test_an_explicit_selection_takes_only_what_it_names() -> None:
    streams = {"a": make_stream(0.0, 1.0, 8, {"ch4": np.arange(8.0), "co2": np.arange(8.0)})}
    joined = bin_streams_onto_cells(streams, cells(0.0, 4.0, 2), ["ch4"])
    assert "ch4" in joined.data_vars
    assert "co2" not in joined.data_vars


def test_a_variable_the_binner_has_never_heard_of_is_handled_the_same() -> None:
    """The property the design is for.

    Nothing here knows what a species is. A baseline, an enhancement or
    anything a later stage invents goes through the same path as a raw
    concentration, so the joining block does not need editing when a new kind
    of variable appears upstream.
    """
    invented = make_stream(
        0.0,
        1.0,
        12,
        {"baseline_ch4": np.arange(12.0), "enhancement_ch4": np.arange(12.0) * 2},
    )
    joined = bin_streams_onto_cells({"derived": invented}, cells(0.0, 4.0, 3))
    assert joined["baseline_ch4"].values == pytest.approx([1.5, 5.5, 9.5])
    assert joined["enhancement_ch4"].values == pytest.approx([3.0, 11.0, 19.0])
    assert joined["enhancement_ch4"].attrs["cell_methods"] == "time: mean"


# ---------------------------------------------------------------------------
# Already on the target support
# ---------------------------------------------------------------------------


def test_a_stream_already_on_the_target_passes_through_untouched() -> None:
    """Tested on the cells, not on the instrument name.

    Averaging a cell onto itself is the identity mathematically and not in
    floating point, so the check has to be exact equality rather than
    approximate.
    """
    values = np.linspace(1900.0, 2100.0, 40)
    stream = make_stream(0.0, 1.0, 40, {"ch4": values})
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 1.0, 40))
    assert np.array_equal(joined["ch4"].values, values)
    assert joined["ch4"].attrs["tsara_binned"] == 0
    assert joined["ch4"].attrs["cell_methods"] == "time: point"


def test_a_binned_stream_is_marked_as_binned() -> None:
    stream = make_stream(0.0, 1.0, 40, {"ch4": np.arange(40.0)})
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 4.0, 10))
    assert joined["ch4"].attrs["tsara_binned"] == 1
    assert joined["ch4"].attrs["cell_methods"] == "time: mean"


def test_one_stream_can_pass_through_while_another_is_binned() -> None:
    streams = {
        "slow": make_stream(0.0, 4.0, 10, {"co2": np.arange(10.0)}),
        "fast": make_stream(0.0, 1.0, 40, {"ch4": np.arange(40.0)}),
    }
    joined = bin_streams_onto_cells(streams, cells(0.0, 4.0, 10))
    assert joined["co2"].attrs["tsara_binned"] == 0
    assert joined["ch4"].attrs["tsara_binned"] == 1
    assert np.array_equal(joined["co2"].values, np.arange(10.0))


# ---------------------------------------------------------------------------
# Angular variables
# ---------------------------------------------------------------------------


def test_an_angular_variable_is_vector_averaged_and_carries_its_quality() -> None:
    """A direction declaring itself circular must not be averaged arithmetically.

    Four samples straddling north average to north, not to south, and the
    resultant length says how well determined that is.
    """
    attrs: dict[str, dict[str, object]] = {
        "wind_dir": {"units": "degrees", "circular": 1, "role": "met"}
    }
    met = make_stream(0.0, 1.0, 4, {"wind_dir": np.array([358.0, 359.0, 1.0, 2.0])}, attrs=attrs)
    joined = bin_streams_onto_cells({"met": met}, cells(0.0, 4.0, 1))
    assert joined["wind_dir"].values[0] == pytest.approx(0.0, abs=1e-9)
    assert np.mean([358.0, 359.0, 1.0, 2.0]) == pytest.approx(180.0)
    assert joined["wind_dir_resultant_length"].values[0] > 0.99
    assert joined["wind_dir_dispersion"].values[0] < 5.0


def test_a_tumbling_direction_reports_no_direction_and_says_why() -> None:
    attrs: dict[str, dict[str, object]] = {"wind_dir": {"units": "degrees", "circular": 1}}
    met = make_stream(0.0, 1.0, 4, {"wind_dir": np.array([0.0, 90.0, 180.0, 270.0])}, attrs=attrs)
    joined = bin_streams_onto_cells({"met": met}, cells(0.0, 4.0, 1))
    assert np.isnan(joined["wind_dir"].values[0])
    assert joined["wind_dir_resultant_length"].values[0] == 0.0
    assert np.isinf(joined["wind_dir_dispersion"].values[0])


def test_angular_quality_columns_carry_no_cell_method() -> None:
    """A resultant length is a property of the cell, not a statistic of the
    data inside it -- the same reason the sigma companions carry none."""
    attrs: dict[str, dict[str, object]] = {"wind_dir": {"units": "degrees", "circular": 1}}
    met = make_stream(0.0, 1.0, 8, {"wind_dir": np.linspace(0.0, 40.0, 8)}, attrs=attrs)
    joined = bin_streams_onto_cells({"met": met}, cells(0.0, 4.0, 2))
    assert "cell_methods" not in joined["wind_dir_resultant_length"].attrs
    assert "cell_methods" not in joined["wind_dir_dispersion"].attrs
    assert joined["n_source_wind_dir"].attrs["cell_methods"] == "time: sum"


def test_an_angular_variable_gets_no_sigma_column() -> None:
    """An angle's uncertainty is its resultant length, not a standard deviation
    of the numbers, which would be meaningless across the wrap."""
    attrs: dict[str, dict[str, object]] = {"wind_dir": {"units": "degrees", "circular": 1}}
    met = make_stream(
        0.0,
        1.0,
        8,
        {"wind_dir": np.linspace(0.0, 40.0, 8), sigma_rand_name("wind_dir"): np.full(8, 3.0)},
        attrs=attrs,
    )
    joined = bin_streams_onto_cells({"met": met}, cells(0.0, 4.0, 2))
    assert sigma_rand_name("wind_dir") not in joined.data_vars


# ---------------------------------------------------------------------------
# The pass-through path must be the general path, only faster
# ---------------------------------------------------------------------------
#
# A stream whose cells already are the target skips the averaging, because
# averaging a cell onto itself is the identity mathematically and not in
# floating point. What it must NOT skip is meaning: its companion columns have
# to be the ones the general path would produce, or a product's columns depend
# on whether a stream happened to share the target's cells -- which a sweep
# over grid period changes. Both tests below compare the two paths on the same
# rows: the full cell array takes the pass-through, and the same cells minus
# the last one are not identical arrays, so they take the general path.


def test_the_pass_through_path_counts_a_masked_value_as_nothing() -> None:
    """A masked value contributes no count and no coverage on either path."""
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0])
    stream = make_stream(0.0, 1.0, 6, {"ch4": values})
    native = bin_streams_onto_cells({"a": stream}, cells(0.0, 1.0, 6))
    general = bin_streams_onto_cells({"a": stream}, cells(0.0, 1.0, 5))
    assert native["ch4"].attrs["tsara_binned"] == 0
    assert general["ch4"].attrs["tsara_binned"] == 1
    assert native["n_source_ch4"].values.tolist() == [1, 0, 1, 0, 1, 1]
    assert native["n_source_ch4"].values[:5].tolist() == general["n_source_ch4"].values.tolist()
    assert native["coverage_ch4"].values[:5] == pytest.approx(general["coverage_ch4"].values)


def test_an_angle_on_its_own_cells_carries_the_same_columns_as_a_binned_one() -> None:
    """The 60 s ground wind on a 60 s grid aligned to it is the real case.

    Before, the pass-through path gave that wind no resultant length or
    dispersion and kept its sigma, while one grid period longer it had both and
    no sigma.
    """
    attrs: dict[str, dict[str, object]] = {"wind_dir": {"units": "degrees", "circular": 1}}
    readings = np.array([350.0, np.nan, 10.0, 90.0, 180.0, 270.0])
    met = make_stream(
        0.0,
        1.0,
        6,
        {"wind_dir": readings, sigma_rand_name("wind_dir"): np.full(6, 5.0)},
        attrs=attrs,
    )
    native = bin_streams_onto_cells({"met": met}, cells(0.0, 1.0, 6), ["wind_dir"])
    general = bin_streams_onto_cells({"met": met}, cells(0.0, 1.0, 5), ["wind_dir"])
    assert sorted(map(str, native.data_vars)) == sorted(map(str, general.data_vars))
    assert sigma_rand_name("wind_dir") not in native.data_vars
    # The dispersion is compared more loosely than the rest, for a float64
    # reason and not a logical one: the general path's hypot of one unit vector
    # can land one ULP below R = 1, and sqrt(-2 ln R) turns 1e-16 into 8.5e-7
    # degrees. The pass-through reports exactly 0, which is the true value.
    for column, tolerance in (
        ("n_source_wind_dir", 0.0),
        ("coverage_wind_dir", 1e-12),
        ("wind_dir_resultant_length", 1e-12),
        ("wind_dir_dispersion", 1e-5),
    ):
        np.testing.assert_allclose(
            native[column].values[:5], general[column].values, atol=tolerance, err_msg=column
        )
    assert np.isnan(native["wind_dir_resultant_length"].values[1])
    assert native["wind_dir"].values[:5] == pytest.approx(general["wind_dir"].values, nan_ok=True)


# ---------------------------------------------------------------------------
# Name collisions
# ---------------------------------------------------------------------------


def test_one_species_measured_twice_keeps_both_columns() -> None:
    """Comparing two analyzers is a real campaign, so neither may win silently."""
    streams = {
        "aeris": make_stream(0.0, 1.0, 20, {"ch4": np.full(20, 1900.0)}),
        "picarro": make_stream(0.0, 2.0, 10, {"ch4": np.full(10, 1910.0)}),
    }
    joined = bin_streams_onto_cells(streams, cells(0.0, 5.0, 4))
    assert "ch4" not in joined.data_vars
    assert joined["ch4_aeris"].values == pytest.approx(np.full(4, 1900.0))
    assert joined["ch4_picarro"].values == pytest.approx(np.full(4, 1910.0))
    assert joined["ch4_aeris"].attrs["tsara_source_instrument"] == "aeris"


def test_a_name_only_one_stream_claims_is_left_alone() -> None:
    streams = {
        "aeris": make_stream(0.0, 1.0, 20, {"ch4": np.full(20, 1900.0)}),
        "picarro": make_stream(0.0, 2.0, 10, {"co2": np.full(10, 420.0)}),
    }
    joined = bin_streams_onto_cells(streams, cells(0.0, 5.0, 4))
    assert set(joined.data_vars) >= {"ch4", "co2"}


# ---------------------------------------------------------------------------
# Nothing is invented
# ---------------------------------------------------------------------------


def test_a_cell_with_no_data_is_nan_rather_than_bridged() -> None:
    stream = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 2.0, 6))
    assert np.isfinite(joined["ch4"].values[:2]).all()
    assert np.isnan(joined["ch4"].values[2:]).all()
    assert joined["n_source_ch4"].values.tolist() == [2, 2, 0, 0, 0, 0]


def test_coverage_records_a_partly_filled_cell() -> None:
    stream = make_stream(0.0, 1.0, 3, {"ch4": np.arange(3.0)})
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 4.0, 1))
    assert joined["coverage_ch4"].values[0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_no_streams() -> None:
    with pytest.raises(TsaraAlignError, match="No streams to bin"):
        bin_streams_onto_cells({}, cells(0.0, 1.0, 2))


def test_no_target_cells() -> None:
    stream = make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})
    empty = CellBounds(start_ns=np.empty(0, dtype=np.int64), stop_ns=np.empty(0, dtype=np.int64))
    with pytest.raises(TsaraAlignError, match="No target cells"):
        bin_streams_onto_cells({"a": stream}, empty)


def test_a_selection_of_only_sigma_companions_is_refused() -> None:
    """They travel with their parent; alone they are a column with no meaning."""
    stream = xr.Dataset(
        {sigma_rand_name("ch4"): ("time", np.full(4, 1.0))},
        coords={
            "time": cells(0.0, 1.0, 4).midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([cells(0.0, 1.0, 4).start_ns, cells(0.0, 1.0, 4).stop_ns], axis=1).astype(
                    "datetime64[ns]"
                ),
            ),
        },
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    with pytest.raises(TsaraAlignError, match="No variables selected"):
        bin_streams_onto_cells({"a": stream}, cells(0.0, 2.0, 2))


def test_resolve_variable_is_the_shared_lookup() -> None:
    streams = {"a": make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})}
    assert resolve_variable(streams, "ch4") == ("a", "ch4")
    assert resolve_variable(streams, ("a", "ch4")) == ("a", "ch4")


# ---------------------------------------------------------------------------
# The product describes itself
# ---------------------------------------------------------------------------


def test_the_joined_product_carries_cells_and_its_own_provenance() -> None:
    streams = {"a": make_stream(0.0, 1.0, 8, {"ch4": np.arange(8.0)})}
    joined = bin_streams_onto_cells(streams, cells(0.0, 4.0, 2))
    assert joined.attrs["tsara_stage"] == "binned"
    assert joined.attrs["tsara_propagation_form"] == "ar1_neff"
    assert "time_bnds" in joined.coords
    assert joined["time"].attrs["bounds"] == "time_bnds"
    assert joined["ch4"].attrs["tsara_source_instrument"] == "a"


# ---------------------------------------------------------------------------
# The one direction this operation must not run in
# ---------------------------------------------------------------------------


def test_a_slow_stream_is_refused_onto_fast_cells() -> None:
    """Evaluating a 60 s mean on 1 s cells hands back rows nobody measured.

    The value in each of them is real; the *rows* are not, and they would
    enter a fit as independent measurements. Both callers of the primitive
    already prevent this by choosing their target, so the refusal here is
    about the primitive being public.
    """
    slow = make_stream(0.0, 60.0, 5, {"minute_ch4": np.arange(5.0)})
    with pytest.raises(TsaraAlignError, match="resolution the instrument never had"):
        bin_streams_onto_cells({"minute": slow}, cells(0.0, 1.0, 300))


def test_the_refusal_names_the_stream_the_widths_and_the_damage() -> None:
    slow = make_stream(0.0, 60.0, 2, {"minute_ch4": np.arange(2.0)})
    with pytest.raises(TsaraAlignError) as raised:
        bin_streams_onto_cells({"minute": slow}, cells(0.0, 1.0, 120))
    message = str(raised.value)
    assert "'minute'" in message
    assert "60 s cells" in message
    assert "1 s" in message
    # One 60 s reading would cover sixty 1 s cells' worth of time, and any
    # cells wider than 30 s would not be replicated.
    assert "cover 60 target cells' worth of time" in message
    assert "wider than 30 s" in message


def test_a_duty_cycled_sampler_is_refused_onto_a_fine_grid() -> None:
    """A canister fills for 15 s and then waits; the 15 s is still an average.

    Its cells are narrower than its spacing, so a width-versus-cadence rule
    would get this backwards. What matters is that one fill would fill many
    one-second rows.
    """
    start = (np.arange(4, dtype=np.int64) * 90 * SECOND) + 0
    canister = CellBounds(start_ns=start, stop_ns=start + 15 * SECOND)
    stream = xr.Dataset(
        {"can_ch4": ("time", np.arange(4.0), {"units": "ppb"})},
        coords={
            "time": canister.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([canister.start_ns, canister.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    with pytest.raises(TsaraAlignError, match="cover 15 target cells' worth"):
        bin_streams_onto_cells({"canister": stream}, cells(0.0, 1.0, 360))


def test_cells_that_differ_only_by_jitter_are_not_refused() -> None:
    """The guard is aimed at replication, not at a comparison of medians.

    Two real analyzers in the 2026 archive nominally sample at 1 s and measure
    0.993 s and 1.024 s. Binning the wider onto the narrower is a hair's worth
    of support mismatch, not an invention of rows, and a median-width rule
    would refuse it while this one does not.
    """
    wider = make_stream(0.0, 1.024, 40, {"ch4": np.arange(40.0)})
    joined = bin_streams_onto_cells({"a": wider}, cells(0.0, 0.993, 40))
    assert np.isfinite(joined["ch4"].values).any()


@pytest.mark.parametrize("offset_s", [0.0, 0.3, 0.5])
def test_a_reading_two_target_cells_wide_is_refused_at_any_phase(offset_s: float) -> None:
    """The phase hole in the rule this guard replaced (METHODS §11.2.1).

    That rule counted target cells lying wholly inside one source cell. A
    regular 2 s record exactly in phase with a 1 s grid wholly contains two and
    was refused; offset by 0.3 s or 0.5 s it wholly contains one, passed, and
    each reading fed two or three rows. Two cells' worth of time is two cells'
    worth whatever the phase.
    """
    source = make_stream(offset_s, 2.0, 50, {"ch4": np.arange(50.0)})
    with pytest.raises(TsaraAlignError, match="cover 2 target cells' worth"):
        bin_streams_onto_cells({"picarro": source}, cells(0.0, 1.0, 102))


def test_a_reading_just_under_two_target_cells_wide_is_allowed() -> None:
    """Below the line nothing is refused; the sharing is counted instead.

    A 1.9 s reading on 1 s cells does feed two rows, and a later count of the
    readings behind them says so. It does not state a value at a resolution
    two whole cells finer than the instrument's.
    """
    source = make_stream(0.3, 1.9, 50, {"ch4": np.arange(50.0)})
    joined = bin_streams_onto_cells({"a": source}, cells(0.0, 1.0, 96))
    assert np.isfinite(joined["ch4"].values).sum() > 50


def test_the_rule_is_measured_against_the_widest_target_cell_touched() -> None:
    """Non-uniform target cells: two cells' worth means of the widest one.

    A 3 s reading across a 1 s cell and a 2 s cell covers one of the wider
    cells and a half of the narrower, so it is not refused; the same reading
    across three 1 s cells is.
    """
    source = make_stream(0.0, 3.0, 1, {"ch4": np.array([1.0])})
    mixed = CellBounds(
        start_ns=np.array([0, SECOND], dtype=np.int64),
        stop_ns=np.array([SECOND, 3 * SECOND], dtype=np.int64),
    )
    assert np.isfinite(bin_streams_onto_cells({"a": source}, mixed)["ch4"].values).all()
    with pytest.raises(TsaraAlignError):
        bin_streams_onto_cells({"a": source}, cells(0.0, 1.0, 3))


def test_a_zero_width_target_cell_makes_nothing_look_replicated() -> None:
    source = make_stream(0.0, 10.0, 2, {"ch4": np.arange(2.0)})
    degenerate = CellBounds(
        start_ns=np.array([5 * SECOND, 5 * SECOND], dtype=np.int64),
        stop_ns=np.array([5 * SECOND, 15 * SECOND], dtype=np.int64),
    )
    joined = bin_streams_onto_cells({"a": source}, degenerate)
    assert np.isfinite(joined["ch4"].values[1])


def test_readings_behind_counts_distinct_finite_contributors() -> None:
    """Five readings, one masked, spread over three 2 s cells: four readings."""
    stream = make_stream(0.0, 1.0, 5, {"ch4": np.array([1.0, np.nan, 3.0, 4.0, 5.0])})
    count = readings_behind(stream, "ch4", stream_cells(stream, "a"), cells(0.0, 2.0, 3))
    assert count == 4


def test_a_grid_out_of_phase_with_an_equal_width_source_is_not_refused() -> None:
    """Half a period out of phase, every source cell straddles two targets.

    So no target cell lies wholly inside a source cell, nothing is replicated,
    and the operation is allowed. It is documented as lossy elsewhere and
    warned about by the grid builder; it is not this guard's business.
    """
    source = make_stream(0.0, 10.0, 6, {"ch4": np.arange(6.0)})
    offset = cells(5.0, 10.0, 5)
    joined = bin_streams_onto_cells({"a": source}, offset)
    assert np.all(joined["n_source_ch4"].values == 2)


def test_a_zero_width_target_cell_does_not_count_as_replication() -> None:
    """A degenerate cell sits inside everything and means nothing.

    Counting it would make any binning onto a grid containing one look like
    replication, which would refuse the whole call over a single bad row.
    """
    degenerate = CellBounds(
        start_ns=np.array([0, SECOND, 2 * SECOND], dtype=np.int64),
        stop_ns=np.array([0, SECOND, 2 * SECOND], dtype=np.int64),
    )
    stream = make_stream(0.0, 10.0, 1, {"ch4": np.array([5.0])})
    joined = bin_streams_onto_cells({"a": stream}, degenerate)
    assert np.all(np.isnan(joined["ch4"].values))


# ---------------------------------------------------------------------------
# Companion columns are not variables
# ---------------------------------------------------------------------------


def test_a_joined_product_can_be_joined_again() -> None:
    """Phase 5 will bin a baseline it computed from an already-binned product.

    The default selection must therefore leave this stage's own bookkeeping
    alone, or a second pass grows `coverage_coverage_ch4` and a third grows
    another layer.
    """
    streams = {"a": make_stream(0.0, 1.0, 120, {"ch4": np.arange(120.0)})}
    once = bin_streams_onto_cells(streams, cells(0.0, 10.0, 12))
    twice = bin_streams_onto_cells({"binned": once}, cells(0.0, 60.0, 2))
    assert "ch4" in twice.data_vars
    assert not [name for name in twice.data_vars if str(name).startswith("coverage_coverage")]
    assert not [name for name in twice.data_vars if str(name).startswith("n_source_n_source")]


def test_angular_quality_columns_are_not_rebinned_either() -> None:
    """A resultant length and a dispersion describe a cell, not the air in it."""
    attrs: dict[str, dict[str, object]] = {"wind_dir": {"units": "degrees", "circular": 1}}
    met = make_stream(0.0, 1.0, 60, {"wind_dir": np.linspace(0.0, 50.0, 60)}, attrs=attrs)
    once = bin_streams_onto_cells({"met": met}, cells(0.0, 10.0, 6))
    assert "wind_dir_resultant_length" in once.data_vars
    twice = bin_streams_onto_cells({"binned": once}, cells(0.0, 30.0, 2))
    assert "wind_dir_resultant_length_resultant_length" not in twice.data_vars
    assert "wind_dir_dispersion_dispersion" not in twice.data_vars


def test_cell_boundaries_carried_as_a_data_variable_are_not_selected() -> None:
    """`stream_cells` accepts that shape, so the default selection must too."""
    stream = make_stream(0.0, 1.0, 20, {"ch4": np.arange(20.0)}).reset_coords("time_bnds")
    assert "time_bnds" in stream.data_vars
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 5.0, 4))
    assert sorted(map(str, joined.data_vars)) == ["ch4", "coverage_ch4", "n_source_ch4"]


def test_a_variable_that_is_not_one_value_per_cell_is_named_not_broadcast() -> None:
    """Asking for the boundaries explicitly is a mistake worth a clear error."""
    stream = make_stream(0.0, 1.0, 20, {"ch4": np.arange(20.0)}).reset_coords("time_bnds")
    with pytest.raises(TsaraAlignError, match="only one value per cell can be binned"):
        bin_streams_onto_cells({"a": stream}, cells(0.0, 5.0, 4), [("a", "time_bnds")])
