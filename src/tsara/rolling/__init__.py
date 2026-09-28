"""The continuous rolling state: baselines and enhancements per stream, at native rate.

For every reading of every instrument and every point of the sweep, this
stage computes the baseline, the enhancement, and the noise scale that
detection thresholds are quoted in, and keeps them on the reading's own cell
beside the reading (``docs/METHODS.md`` §6.2). Nothing here changes a
stream's support: a window is an interval the statistic looks at, and the
value it yields belongs to the cell the window is centred on.

Shape of the subpackage
-----------------------
``windows``
    Windows as cells, centred on each reading; the package's error class.
``quantile``
    The weighted rolling quantile: one definition (§6.3), a per-window
    reference form and the block form that evaluates a whole record, with
    the count and width qualifiers every window records (§6.4) and the
    order-statistic uncertainty of the quantile (§6.7).

The joining of a rolling state onto other cells is not here: a per-reading
product is joined like a stream by :mod:`tsara.align`, which hands its result
back as a Dataset (§11.2, *swept variables*).
"""

from __future__ import annotations

from tsara.rolling.quantile import (
    MAX_BLOCK_ELEMENTS,
    RollingQuantile,
    rolling_quantile,
    weighted_quantile,
)
from tsara.rolling.windows import TsaraRollingError, duration_ns, window_cells

__all__ = [
    "MAX_BLOCK_ELEMENTS",
    "RollingQuantile",
    "TsaraRollingError",
    "duration_ns",
    "rolling_quantile",
    "weighted_quantile",
    "window_cells",
]
