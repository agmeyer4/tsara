"""Tests for resolving which variables a call acts on.

The lookup and the default selection are public because the binner and the
output grid must agree on them; what is tested here is the answer they give
on their own. The refusal of a joined product, which the selection carries,
is exercised through the binner in `test_binning.py`, where the product to
refuse is made.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from tsara.align import bin_streams_onto_cells
from tsara.align.variables import resolve_variable, select_variables
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


def test_resolve_variable_is_the_shared_lookup() -> None:
    streams = {"a": make_stream(0.0, 1.0, 4, {"ch4": np.arange(4.0)})}
    assert resolve_variable(streams, "ch4") == ("a", "ch4")
    assert resolve_variable(streams, ("a", "ch4")) == ("a", "ch4")


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
