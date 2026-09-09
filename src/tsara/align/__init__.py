"""Putting measurements on a common support, without inventing any.

This is the first stage that **combines** measurements. Everything before it
applies declared, exact, one-to-one maps — a unit conversion, a per-point
sigma from a declared budget, a timestamp moved to its cell midpoint — and
each of those is recoverable from what the product records. Nothing here is:
a joined value is a weighted mean no instrument reported, and there is no way
back to the values that went into it.

Shape of the subpackage
-----------------------
``binning``
    The one joining operation: any set of variables onto any set of cells,
    with uncertainty, counts and coverage travelling automatically. Everything
    else here is a choice of *which cells*.
``pairing``
    Two species for a regression, on the cells of the wider-supported member
    (``docs/METHODS.md`` §1.3).
``auxiliary``
    The one exception to the interpolation rule: smooth fields evaluated
    between samples, under a gap guard, and the mobile position join that
    ingestion deliberately left undone (§1.2).
``grid``
    The uniform campaign-wide support for the continuous state and a receptor
    model's input matrix (§1.4) -- the same binner with a different target,
    plus the rule that the period must respect the data going into it.
``bundle``
    Saving and reloading that grid.

The arithmetic itself is not here. Overlap-weighted binning lives in
:mod:`tsara.core.support`, angular averaging in :mod:`tsara.core.circular`,
and uncertainty propagation in :mod:`tsara.core.propagation` — all three
because Phase 5 rolling and Phase 7 fitting need the same operations and must
get the same answers. This subpackage is the part that knows about
configuration, streams and provenance.
"""

from __future__ import annotations

from tsara.align.auxiliary import InterpolatedField, attach_positions, interpolate_onto_cells
from tsara.align.binning import (
    TsaraAlignError,
    bin_streams_onto_cells,
    resolve_variable,
    select_variables,
)
from tsara.align.bundle import load_grid, save_grid
from tsara.align.grid import build_output_grid, grid_cells
from tsara.align.pairing import PairedSpecies, pair_species

__all__ = [
    "InterpolatedField",
    "PairedSpecies",
    "TsaraAlignError",
    "attach_positions",
    "bin_streams_onto_cells",
    "build_output_grid",
    "grid_cells",
    "interpolate_onto_cells",
    "load_grid",
    "pair_species",
    "resolve_variable",
    "save_grid",
    "select_variables",
]
