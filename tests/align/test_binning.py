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
from tsara.align.binning import readings_behind, select_variables, stream_cells, targets_overlap
from tsara.core.naming import sigma_rand_name
from tsara.core.support import CellBounds
from tsara.core.timebase import SECOND_NS as SECOND


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
        assert f"n_readings_{name}" in joined.data_vars
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
    assert joined["n_readings_wind_dir"].attrs["cell_methods"] == "time: sum"


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
    assert native["n_readings_ch4"].values.tolist() == [1, 0, 1, 0, 1, 1]
    assert native["n_readings_ch4"].values[:5].tolist() == general["n_readings_ch4"].values.tolist()
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
        ("n_readings_wind_dir", 0.0),
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
    assert joined["ch4_aeris"].attrs["tsara_instrument"] == "aeris"


def test_a_suffixed_column_still_says_what_it_measures() -> None:
    """The instrument goes into the spelling; the quantity stays in `field`.

    Checked on both paths, one stream already on the target cells and one
    averaged onto them, because they assemble a column in different branches.
    """
    methane: dict[str, object] = {"units": "ppb", "field": "ch4"}
    streams = {
        "aeris": make_stream(0.0, 1.0, 20, {"ch4": np.full(20, 1900.0)}, attrs={"ch4": methane}),
        "picarro": make_stream(0.0, 5.0, 4, {"ch4": np.full(4, 1910.0)}, attrs={"ch4": methane}),
    }
    joined = bin_streams_onto_cells(streams, cells(0.0, 5.0, 4))
    assert joined["ch4_aeris"].attrs["tsara_binned"] == 1
    assert joined["ch4_picarro"].attrs["tsara_binned"] == 0
    assert joined["ch4_aeris"].attrs["field"] == "ch4"
    assert joined["ch4_picarro"].attrs["field"] == "ch4"


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
    assert joined["n_readings_ch4"].values.tolist() == [2, 2, 0, 0, 0, 0]


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
    # A propagation form belongs to a propagated sigma, not to the product: the
    # dataset attr recorded the form *requested* while each column records the
    # form used, and the two read differently on every product whose variables
    # declare no decorrelation timescale (METHODS §11.2).
    assert "tsara_propagation_form" not in joined.attrs
    assert "time_bnds" in joined.coords
    assert joined["time"].attrs["bounds"] == "time_bnds"
    assert joined["ch4"].attrs["tsara_instrument"] == "a"


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
    # One 60 s reading is sixty times as wide as a 1 s cell, and any cells
    # wider than 30 s would not be copies of it.
    assert "is 60 times as wide as a cell it fills" in message
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
    with pytest.raises(TsaraAlignError, match="is 15 times as wide as a cell"):
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

    That rule counted target cells lying wholly inside one reading. A
    regular 2 s record exactly in phase with a 1 s grid wholly contains two and
    was refused; offset by 0.3 s or 0.5 s it wholly contains one, passed, and
    each reading fed two or three rows. Two cells' worth of time is two cells'
    worth whatever the phase.
    """
    stream = make_stream(offset_s, 2.0, 50, {"ch4": np.arange(50.0)})
    with pytest.raises(TsaraAlignError, match="is 2 times as wide as a cell"):
        bin_streams_onto_cells({"picarro": stream}, cells(0.0, 1.0, 102))


def test_a_reading_just_under_two_target_cells_wide_is_allowed() -> None:
    """Below the line nothing is refused; the sharing is counted instead.

    A 1.9 s reading on 1 s cells does feed two rows, and a later count of the
    readings behind them says so. It does not state a value at a resolution
    two whole cells finer than the instrument's.
    """
    stream = make_stream(0.3, 1.9, 50, {"ch4": np.arange(50.0)})
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 1.0, 96))
    assert np.isfinite(joined["ch4"].values).sum() > 50


def test_the_rule_is_measured_per_pair_not_summed_over_a_reading() -> None:
    """Non-uniform target cells: each cell is judged against the reading on its own.

    A 3 s reading across a 1 s cell and a 2 s cell is three times as wide as
    the narrow one, so the join is refused -- the summed rule this replaced
    passed it, because one narrow cell never adds up to two cells' worth
    (METHODS §11.2.4). Across two 2 s cells it is narrowed by half and
    allowed, labelled.
    """
    stream = make_stream(0.0, 3.0, 1, {"ch4": np.array([1.0])})
    mixed = CellBounds(
        start_ns=np.array([0, SECOND], dtype=np.int64),
        stop_ns=np.array([SECOND, 3 * SECOND], dtype=np.int64),
    )
    with pytest.raises(TsaraAlignError, match="is 3 times as wide as a cell"):
        bin_streams_onto_cells({"a": stream}, mixed)
    halves = CellBounds(
        start_ns=np.array([0, 2 * SECOND], dtype=np.int64),
        stop_ns=np.array([2 * SECOND, 4 * SECOND], dtype=np.int64),
    )
    joined = bin_streams_onto_cells({"a": stream}, halves)
    assert joined["ch4"].attrs["tsara_support_transform"] == "narrowed"
    assert joined["ch4"].attrs["tsara_width_ratio_max"] == pytest.approx(1.5)


def test_a_zero_width_target_cell_makes_nothing_look_replicated() -> None:
    stream = make_stream(0.0, 10.0, 2, {"ch4": np.arange(2.0)})
    degenerate = CellBounds(
        start_ns=np.array([5 * SECOND, 5 * SECOND], dtype=np.int64),
        stop_ns=np.array([5 * SECOND, 15 * SECOND], dtype=np.int64),
    )
    joined = bin_streams_onto_cells({"a": stream}, degenerate)
    assert np.isfinite(joined["ch4"].values[1])


def test_readings_behind_counts_distinct_finite_contributors() -> None:
    """Five readings, one masked, spread over three 2 s cells: four readings."""
    stream = make_stream(0.0, 1.0, 5, {"ch4": np.array([1.0, np.nan, 3.0, 4.0, 5.0])})
    count = readings_behind(stream, "ch4", stream_cells(stream, "a"), cells(0.0, 2.0, 3))
    assert count == 4


def test_a_grid_out_of_phase_with_an_equal_width_stream_is_not_refused() -> None:
    """Half a period out of phase, every reading straddles two targets.

    So no target cell lies wholly inside a reading, nothing is replicated,
    and the operation is allowed. It is documented as lossy elsewhere and
    warned about by the grid builder; it is not this guard's business.
    """
    stream = make_stream(0.0, 10.0, 6, {"ch4": np.arange(6.0)})
    offset = cells(5.0, 10.0, 5)
    joined = bin_streams_onto_cells({"a": stream}, offset)
    assert np.all(joined["n_readings_ch4"].values == 2)


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


def test_a_joined_product_is_refused_rather_than_joined_again() -> None:
    """A product's rows are not readings, and the second pass cannot tell.

    It would take each row as a measurement covering its whole cell, so
    coverage, counts and weights would describe rows rather than air. Refused
    on the antimeridian precedent: the information is gone from the input, so
    nothing downstream could repair it (METHODS §11.2.3).
    """
    streams = {"a": make_stream(0.0, 1.0, 120, {"ch4": np.arange(120.0)})}
    once = bin_streams_onto_cells(streams, cells(0.0, 10.0, 12))
    assert once.attrs["tsara_stage"] == "binned"
    with pytest.raises(TsaraAlignError) as refused:
        bin_streams_onto_cells({"binned": once}, cells(0.0, 60.0, 2))
    # The message has to name which dataset and what it is, or a user holding a
    # dict of ten streams cannot tell which one to replace.
    assert "binned" in str(refused.value)
    assert "tsara_stage" in str(refused.value)
    assert "native streams" in str(refused.value)


def test_the_same_product_built_from_the_streams_is_what_the_refusal_asks_for() -> None:
    """The escape route must exist, or the refusal is just a wall."""
    streams = {"a": make_stream(0.0, 1.0, 120, {"ch4": np.arange(120.0)})}
    direct = bin_streams_onto_cells(streams, cells(0.0, 60.0, 2))
    assert np.isfinite(direct["ch4"].values).all()
    assert direct["n_readings_ch4"].values.tolist() == [60, 60]


def test_joining_leaves_the_input_stream_joinable() -> None:
    """The stage travels with the *product*, never back onto its inputs.

    Worth pinning because the check reads an attribute of the input: if a join
    ever stamped its stage on what it consumed, a campaign would become
    unusable after the first pairing, and every later call would fail with a
    refusal about data the user never joined.
    """
    stream = make_stream(0.0, 1.0, 120, {"ch4": np.arange(120.0)})
    stream.attrs["tsara_stage"] = "ingest"
    bin_streams_onto_cells({"a": stream}, cells(0.0, 10.0, 12))
    assert stream.attrs["tsara_stage"] == "ingest"
    again = bin_streams_onto_cells({"a": stream}, cells(0.0, 20.0, 6))
    assert again.sizes["time"] == 6


@pytest.mark.parametrize("stage", ["ingest", "synthetic"])
def test_a_stream_from_either_producer_is_joinable(stage: str) -> None:
    """Both stream producers, and a bundle reloaded from either, carry these."""
    stream = make_stream(0.0, 1.0, 60, {"ch4": np.arange(60.0)})
    stream.attrs["tsara_stage"] = stage
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 10.0, 6))
    assert joined.sizes["time"] == 6


def test_angular_quality_columns_are_not_selected_as_variables() -> None:
    """A resultant length and a dispersion describe a cell, not the air in it.

    Checked through the selection rather than through a second join, which is
    now refused: the rule is about which columns count as variables, and
    `select_variables` is public and answers that for the grid as well.
    """
    attrs: dict[str, dict[str, object]] = {"wind_dir": {"units": "degrees", "circular": 1}}
    met = make_stream(0.0, 1.0, 60, {"wind_dir": np.linspace(0.0, 50.0, 60)}, attrs=attrs)
    once = bin_streams_onto_cells({"met": met}, cells(0.0, 10.0, 6))
    assert "wind_dir_resultant_length" in once.data_vars
    # The product's stage is what refuses a re-join; strip it and the selection
    # rule alone is what is being measured here.
    stripped = once.copy()
    del stripped.attrs["tsara_stage"]
    assert select_variables({"binned": stripped}) == [("binned", "wind_dir")]


def test_cell_boundaries_carried_as_a_data_variable_are_not_selected() -> None:
    """`stream_cells` accepts that shape, so the default selection must too."""
    stream = make_stream(0.0, 1.0, 20, {"ch4": np.arange(20.0)}).reset_coords("time_bnds")
    assert "time_bnds" in stream.data_vars
    joined = bin_streams_onto_cells({"a": stream}, cells(0.0, 5.0, 4))
    assert sorted(map(str, joined.data_vars)) == [
        "borrowed_ch4",
        "ch4",
        "coverage_ch4",
        "n_readings_ch4",
    ]


def test_a_variable_that_is_not_one_value_per_cell_is_named_not_broadcast() -> None:
    """Asking for the boundaries explicitly is a mistake worth a clear error."""
    stream = make_stream(0.0, 1.0, 20, {"ch4": np.arange(20.0)}).reset_coords("time_bnds")
    with pytest.raises(TsaraAlignError, match="only one value per cell can be binned"):
        bin_streams_onto_cells({"a": stream}, cells(0.0, 5.0, 4), [("a", "time_bnds")])


# ---------------------------------------------------------------------------
# The overlap search runs once per instrument
# ---------------------------------------------------------------------------


def _count_overlap_searches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count every overlap search a join performs, wherever it is made.

    Patched in all three modules that can search -- the binner, the scalar
    arithmetic and the circular arithmetic -- because each imported the name
    directly, and a search hidden inside one of the two arithmetic modules
    would otherwise go uncounted while the binner's own count read one.
    """
    import tsara.align.binning as binning
    import tsara.core.circular as circular
    import tsara.core.support as support

    calls: list[int] = []
    real = support.overlap_pairs

    def counting(readings: CellBounds, target: CellBounds) -> object:
        calls.append(1)
        return real(readings, target)

    for module in (binning, circular, support):
        monkeypatch.setattr(module, "overlap_pairs", counting)
    return calls


def test_overlaps_are_searched_once_per_instrument(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spectral stream carries a thousand columns on one clock.

    The search depends only on the cells, so three variables -- one with a
    declared sigma to propagate, one a direction -- and the replication check
    share a single search rather than one each.
    """
    calls = _count_overlap_searches(monkeypatch)
    stream = make_stream(
        0.0,
        1.0,
        60,
        {"a": np.arange(60.0), "b": np.arange(60.0), "wind": np.linspace(0.0, 90.0, 60)},
        attrs={"wind": {"units": "degrees", "circular": 1}},
    )
    stream[sigma_rand_name("a")] = ("time", np.full(60, 0.5))
    joined = bin_streams_onto_cells({"x": stream}, cells(0.0, 10.0, 6))
    assert len(calls) == 1
    assert {"a", "b", "wind", sigma_rand_name("a")} <= set(joined.data_vars)


def test_a_stream_already_on_the_target_cells_searches_no_overlaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical cells pass through, and a reading that is its own target cannot replicate."""
    calls = _count_overlap_searches(monkeypatch)
    stream = make_stream(0.0, 1.0, 60, {"a": np.arange(60.0)})
    bin_streams_onto_cells({"x": stream}, cells(0.0, 1.0, 60))
    assert calls == []


# ---------------------------------------------------------------------------
# What a target cell may be built from (METHODS §11.2.4)
# ---------------------------------------------------------------------------


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


def stream_on(
    readings: CellBounds,
    variables: dict[str, np.ndarray],
    attrs: dict[str, dict[str, object]] | None = None,
) -> xr.Dataset:
    """Build a stream on arbitrary cells."""
    variable_attrs = attrs or {}
    dataset = xr.Dataset(
        {
            name: ("time", values, dict(variable_attrs.get(name, {"units": "ppb"})))
            for name, values in variables.items()
        },
        coords={
            "time": readings.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([readings.start_ns, readings.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
    )
    dataset["time"].attrs["bounds"] = "time_bnds"
    for name in variables:
        dataset[name].attrs.setdefault("cell_methods", "time: point")
    return dataset


#: METHODS §11.2.4's table, as executed: the verdict under the default policy,
#: then under ``finer_support="allow"`` the worst reading-to-cell ratio, the
#: column's borrowed share (both to three decimals, as measured on
#: 2026-09-18) and the label they imply.
PROBE = [
    (
        "60 s -> 1 s (copy)",
        spaced(0, 60, 10, 60),
        spaced(0, 1, 600, 1),
        "refuse",
        60.0,
        0.983,
        "copied",
    ),
    (
        "2 s -> 1 s, phase 0",
        spaced(0, 2, 50, 2),
        spaced(0, 1, 102, 1),
        "refuse",
        2.0,
        0.500,
        "copied",
    ),
    (
        "2 s -> 1 s, phase 0.3",
        spaced(0.3, 2, 50, 2),
        spaced(0, 1, 102, 1),
        "refuse",
        2.0,
        0.605,
        "copied",
    ),
    (
        "2 s -> 1 s, phase 0.5",
        spaced(0.5, 2, 50, 2),
        spaced(0, 1, 102, 1),
        "refuse",
        2.0,
        0.625,
        "copied",
    ),
    (
        "1.9 s -> 1 s",
        spaced(0.3, 1.9, 50, 1.9),
        spaced(0, 1, 96, 1),
        "allow",
        1.9,
        0.565,
        "narrowed",
    ),
    (
        "1.023 s -> 1 s (LANL jitter)",
        spaced(0, 1.023, 400, 1.023),
        spaced(0, 1, 409, 1),
        "allow",
        1.023,
        0.337,
        "narrowed",
    ),
    (
        "0.993 s -> 1 s (WYO jitter)",
        spaced(0, 0.993, 400, 0.993),
        spaced(0, 1, 397, 1),
        "allow",
        0.993,
        0.342,
        "shared",
    ),
    (
        "1 s -> 1 s half a cell out of phase",
        spaced(0.5, 1, 400, 1),
        spaced(0, 1, 401, 1),
        "allow",
        1.0,
        0.500,
        "shared",
    ),
    (
        "15 s fills every 530 s -> 8 s grid",
        spaced(0, 15, 20, 530),
        spaced(0, 8, 1400, 8),
        "allow",
        1.875,
        0.560,
        "narrowed",
    ),
    (
        "15 s fills every 530 s -> 7.5 s grid",
        spaced(0, 15, 20, 530),
        spaced(0, 7.5, 1500, 7.5),
        "refuse",
        2.0,
        0.572,
        "copied",
    ),
    (
        "15 s fills every 530 s -> 15 s grid",
        spaced(7, 15, 20, 530),
        spaced(0, 15, 750, 15),
        "allow",
        1.0,
        0.356,
        "shared",
    ),
    (
        "15 s fills every 530 s -> 60 s grid",
        spaced(50, 15, 20, 530),
        spaced(0, 60, 180, 60),
        "allow",
        0.25,
        0.089,
        "shared",
    ),
    ("60 s -> 30 s", spaced(0, 60, 10, 60), spaced(0, 30, 20, 30), "refuse", 2.0, 0.500, "copied"),
    (
        "60 s -> 30.5 s",
        spaced(0, 60, 10, 60),
        spaced(0, 30.5, 20, 30.5),
        "allow",
        1.967,
        0.558,
        "narrowed",
    ),
    (
        "60 s -> 45 s",
        spaced(0, 60, 10, 60),
        spaced(0, 45, 14, 45),
        "allow",
        1.333,
        0.4125,
        "narrowed",
    ),
    (
        "(4.8, 5) cell inside a 1 s reading",
        spaced(0, 1, 5, 1),
        bounds([0, 4.8], [1, 5]),
        "refuse",
        5.0,
        0.133,
        "copied",
    ),
    (
        "60 s means -> 15 s canister fills",
        spaced(0, 60, 100, 60),
        spaced(50, 15, 11, 530),
        "refuse",
        4.0,
        0.770,
        "copied",
    ),
    (
        "30 s -> sliding 60 s every 10 s",
        spaced(0, 30, 20, 30),
        spaced(0, 60, 55, 10),
        "allow",
        0.5,
        0.145,
        "shared",
    ),
    (
        "60 s -> sliding 300 s every 30 s",
        spaced(0, 60, 60, 60),
        spaced(0, 300, 110, 30),
        "allow",
        0.2,
        0.050,
        "shared",
    ),
    (
        "1 s -> sliding 60 s every 1 s",
        spaced(0, 1, 600, 1),
        spaced(0, 60, 540, 1),
        "allow",
        0.017,
        0.000,
        "averaged",
    ),
    (
        "three copies of one 60 s target",
        spaced(0, 1, 60, 1),
        bounds([0, 0, 0], [60, 60, 60]),
        "allow",
        0.017,
        0.000,
        "averaged",
    ),
    (
        "60 s -> 60 s in phase (passthrough)",
        spaced(0, 60, 10, 60),
        spaced(0, 60, 10, 60),
        "allow",
        1.0,
        0.000,
        "passthrough",
    ),
    (
        "60 s -> 60 s offset 30 s (blend)",
        spaced(0, 60, 10, 60),
        spaced(30, 60, 9, 60),
        "allow",
        1.0,
        0.500,
        "shared",
    ),
    (
        "60 s -> 60 s offset 10 s (blend)",
        spaced(0, 60, 10, 60),
        spaced(10, 60, 9, 60),
        "allow",
        1.0,
        0.278,
        "shared",
    ),
    (
        "10 s LGR -> 1 s grid (copy)",
        spaced(0, 10, 10, 10),
        spaced(0, 1, 100, 1),
        "refuse",
        10.0,
        0.900,
        "copied",
    ),
    (
        "1 s -> 60 s (average)",
        spaced(0, 1, 600, 1),
        spaced(0, 60, 10, 60),
        "allow",
        0.017,
        0.000,
        "averaged",
    ),
]


@pytest.mark.parametrize(
    ("readings", "target", "verdict", "ratio", "share", "label"),
    [row[1:] for row in PROBE],
    ids=[row[0] for row in PROBE],
)
def test_the_probe_table(
    readings: CellBounds,
    target: CellBounds,
    verdict: str,
    ratio: float,
    share: float,
    label: str,
) -> None:
    """Every case METHODS §11.2.4 tabulates, as verdict, worst ratio, share and label.

    The two narrowing holes of the summed rule are refused here and the
    sliding windows it wrongly refused are built; the numbers are the ones
    measured before the rule was chosen, so a change in any of them is a
    change in what the package does, not in what it says.
    """
    streams = {"x": stream_on(readings, {"v": np.ones(len(readings))})}
    if verdict == "refuse":
        with pytest.raises(TsaraAlignError, match="times as wide as a cell it fills"):
            bin_streams_onto_cells(streams, target)
    joined = bin_streams_onto_cells(streams, target, finer_support="allow")
    assert joined["v"].attrs["tsara_width_ratio_max"] == pytest.approx(ratio, abs=5e-4)
    assert joined["v"].attrs["tsara_borrowed_share"] == pytest.approx(share, abs=5e-4)
    assert joined["v"].attrs["tsara_support_transform"] == label


def test_copying_is_refused_by_default_and_allowed_loudly(caplog: pytest.LogCaptureFixture) -> None:
    """The interpolation rule stays the default; the escape is by name, labelled and warned."""
    minute = stream_on(spaced(0, 60, 10, 60), {"v": np.arange(10.0)})
    with pytest.raises(TsaraAlignError, match="finer_support='allow'"):
        bin_streams_onto_cells({"m": minute}, spaced(0, 1, 600, 1))
    with caplog.at_level("WARNING", logger="tsara.align.binning"):
        joined = bin_streams_onto_cells({"m": minute}, spaced(0, 1, 600, 1), finer_support="allow")
    assert int(np.isfinite(joined["v"].values).sum()) == 600
    assert joined["v"].attrs["tsara_support_transform"] == "copied"
    assert joined["v"].attrs["tsara_readings"] == 10
    assert "finer_support='allow': readings of 'm' up to 60 times as wide" in caplog.text
    assert "'v' (copied, readings up to 60x a cell; 600 rows from 10 readings" in caplog.text


def test_one_warning_names_the_narrowed_and_shared_columns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The LANL analyzer's shape: 1.023 s readings on a 1 s grid, beside a 0.5 s instrument.

    The jittered column is narrowed by 2.3 % and holds 409 rows from 400
    readings, so it is named with both numbers; the fast column sits wholly
    inside every cell and is not mentioned. One warning for the call.
    """
    pico = stream_on(spaced(0, 1.023, 400, 1.023), {"v": np.ones(400)})
    fast = stream_on(spaced(0, 0.5, 818, 0.5), {"v": np.ones(818)})
    with caplog.at_level("WARNING", logger="tsara.align.binning"):
        joined = bin_streams_onto_cells({"pico": pico, "fast": fast}, spaced(0, 1, 409, 1))
    assert caplog.text.count("rest on air") == 1
    assert "1 column(s) of this join" in caplog.text
    assert (
        "'v_pico' (narrowed, readings up to 1.02x a cell; 409 rows from 400 readings; "
        "borrowed share 0.34)" in caplog.text
    )
    assert "v_fast" not in caplog.text
    assert joined["v_fast"].attrs["tsara_support_transform"] == "averaged"
    assert joined["v_fast"].attrs["tsara_borrowed_share"] == 0.0


def test_a_blend_is_recorded_and_not_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """Equal cells half a cell out of phase: every reading straddles, none is wider, none repeated.

    Measured, no value of the borrowed share separates this 0.50 from ordinary
    jitter's 0.34, so it is recorded rather than thresholded (METHODS §11.2.4).
    """
    stream = stream_on(spaced(0.5, 1, 400, 1), {"v": np.ones(400)})
    with caplog.at_level("WARNING", logger="tsara.align.binning"):
        joined = bin_streams_onto_cells({"a": stream}, spaced(1, 1, 399, 1))
    assert "rest on air" not in caplog.text
    assert joined["v"].attrs["tsara_support_transform"] == "shared"
    assert joined["v"].attrs["tsara_borrowed_share"] == pytest.approx(0.5)
    assert np.allclose(joined["borrowed_v"].values, 0.5)


def test_overlapping_targets_share_readings_by_construction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sliding 60 s windows every 10 s over 30 s readings: built, recorded, not warned about.

    Every reading feeds six windows because that is what was asked; rows
    outnumbering readings is the question, not a defect. The summed rule this
    replaced refused it as a copy.
    """
    stream = stream_on(spaced(0, 30, 20, 30), {"v": np.arange(20.0)})
    windows = spaced(0, 60, 55, 10)
    assert targets_overlap(windows)
    with caplog.at_level("WARNING", logger="tsara.align.binning"):
        joined = bin_streams_onto_cells({"a": stream}, windows)
    assert "rest on air" not in caplog.text
    assert int(np.isfinite(joined["v"].values).sum()) == 55
    assert joined["v"].attrs["tsara_readings"] == 20
    assert joined["v"].attrs["tsara_support_transform"] == "shared"


def test_disjoint_targets_that_share_a_reading_are_warned_about(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 15 s fill across a minute boundary: one reading in two rows of a 60 s grid."""
    fills = stream_on(bounds([20, 112], [35, 127]), {"benzene": np.array([1.0, 2.0])})
    with caplog.at_level("WARNING", logger="tsara.align.binning"):
        joined = bin_streams_onto_cells({"iwas": fills}, spaced(0, 60, 3, 60))
    assert "'benzene' (3 rows from 2 readings; borrowed share" in caplog.text
    assert joined["benzene"].attrs["tsara_readings"] == 2
    assert joined["benzene"].attrs["tsara_support_transform"] == "shared"


def test_the_borrowed_companion_is_the_per_cell_share_and_is_not_a_variable() -> None:
    """A third qualifier beside the count and the coverage, excluded from selection like them."""
    stream = stream_on(spaced(0.3, 1.9, 50, 1.9), {"v": np.arange(50.0)})
    target = spaced(0, 1, 96, 1)
    joined = bin_streams_onto_cells({"a": stream}, target)
    from tsara.core.support import bin_onto_cells

    reference = bin_onto_cells(stream_cells(stream, "a"), np.arange(50.0), target)
    assert np.array_equal(joined["borrowed_v"].values, reference.borrowed, equal_nan=True)
    assert "cell_methods" not in joined["borrowed_v"].attrs
    assert joined["borrowed_v"].attrs["units"] == "1"
    # Not a variable: the default selection skips it, so it can never be binned again.
    carrier = make_stream(0.0, 1.0, 5, {"a": np.ones(5), "borrowed_a": np.zeros(5)})
    assert select_variables({"s": carrier}) == [("s", "a")]


def test_a_passed_through_column_borrows_nothing_and_says_so() -> None:
    """On its own cells a reading borrows nothing; a masked one has no share at all."""
    values = np.array([1.0, np.nan, 3.0])
    stream = make_stream(0.0, 1.0, 3, {"a": values})
    joined = bin_streams_onto_cells({"s": stream}, cells(0.0, 1.0, 3))
    assert joined["a"].attrs["tsara_support_transform"] == "passthrough"
    assert joined["a"].attrs["tsara_width_ratio_max"] == 1.0
    assert joined["a"].attrs["tsara_borrowed_share"] == 0.0
    assert joined["a"].attrs["tsara_readings"] == 2
    assert np.array_equal(joined["borrowed_a"].values, [0.0, np.nan, 0.0], equal_nan=True)


def test_a_direction_carries_the_same_record_as_a_scalar() -> None:
    """Same overlaps, same weights: the record does not depend on the kind of variable."""
    stream = stream_on(
        spaced(0, 1.023, 100, 1.023),
        {"v": np.ones(100), "wind": np.linspace(0.0, 90.0, 100)},
        attrs={"wind": {"units": "degrees", "circular": 1}},
    )
    joined = bin_streams_onto_cells({"a": stream}, spaced(0, 1, 103, 1))
    for attr in (
        "tsara_support_transform",
        "tsara_width_ratio_max",
        "tsara_borrowed_share",
        "tsara_readings",
    ):
        assert joined["wind"].attrs[attr] == joined["v"].attrs[attr]
    assert np.array_equal(
        joined["borrowed_wind"].values, joined["borrowed_v"].values, equal_nan=True
    )


def test_a_column_with_nothing_behind_it_records_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """All masked: no readings, no ratio, no share, no warning -- and no label but 'averaged'."""
    stream = make_stream(0.0, 1.0, 60, {"a": np.full(60, np.nan)})
    with caplog.at_level("WARNING", logger="tsara.align.binning"):
        joined = bin_streams_onto_cells({"s": stream}, cells(0.0, 10.0, 6))
    assert joined["a"].attrs["tsara_readings"] == 0
    assert np.isnan(joined["a"].attrs["tsara_width_ratio_max"])
    assert np.isnan(joined["a"].attrs["tsara_borrowed_share"])
    assert joined["a"].attrs["tsara_support_transform"] == "averaged"
    assert "rest on air" not in caplog.text


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
