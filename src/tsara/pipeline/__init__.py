"""The pipeline: what strings the stages together, above them.

The stages (``tsara.ingest``, ``align``, ``baseline``, ``events``, ``ratios``
and the rest) never import each other: each hands the next an xarray Dataset
or a table whose vocabulary lives in ``tsara.core.naming``, which is what lets
a new stage be inserted anywhere (``CONTRIBUTING.md``). Where one step of the
science needs two stages in turn, the loop that calls them lives here, on the
layer above, and computes nothing either stage does not. The stateful
``Pipeline`` class and the command line (roadmap 7d, 9) will live here too.

Shape of the subpackage
-----------------------
``ratios``
    The ratio stage over a configuration: for every ratio set, every
    numerator it expands to and every pair clock, one pairing
    (``tsara.align``) and one call of the fits (``tsara.ratios``), their
    tables concatenated (``docs/METHODS.md`` §5.2).
"""

from __future__ import annotations

from tsara.pipeline.ratios import ratio_fits

__all__ = ["ratio_fits"]
