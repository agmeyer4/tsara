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
