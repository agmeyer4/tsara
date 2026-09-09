"""Putting two measurements on one clock, without inventing either.

This is the first stage that **combines** measurements. Everything before it
applies declared, exact, one-to-one maps — a unit conversion, a per-point
sigma from a declared budget, a timestamp moved to its cell midpoint — and
each of those is recoverable from what the product records. Nothing here is:
a paired value is a weighted mean no instrument reported, and there is no way
back to the values that went into it.

Shape of the subpackage
-----------------------
``pairing``
    The cross-species product of ``docs/METHODS.md`` §1.3: two species, one
    pairing clock, real pairs only.

Modules for auxiliary-field interpolation and the output grid land with their
stages.

The arithmetic itself is not here. Overlap-weighted binning lives in
:mod:`tsara.core.support`, angular averaging in :mod:`tsara.core.circular`,
and uncertainty propagation in :mod:`tsara.core.propagation` — all three
because Phase 5 rolling and Phase 7 fitting need the same operations and must
get the same answers. This subpackage is the part that knows about
configuration, streams and provenance.
"""

from __future__ import annotations

from tsara.align.pairing import PairedSpecies, TsaraAlignError, pair_species

__all__ = [
    "PairedSpecies",
    "TsaraAlignError",
    "pair_species",
]
