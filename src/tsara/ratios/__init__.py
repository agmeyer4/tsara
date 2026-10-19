"""Ratios: the slope of one gas's enhancement against another's, and how far to trust it.

An enhancement ratio is a straight line fitted through one gas's enhancement
against another's over a stretch of readings (``docs/METHODS.md`` §4). Both
axes carry measurement error, so the preferred line is York's, which weighs
every point by its declared sigmas on both axes and reports how well they
explain the scatter; ordinary least squares is kept beside it. This package
holds the line fits; it reads nothing from the other stages.

Shape of the subpackage
-----------------------
``estimators``
    Straight lines fitted through many groups of points at once, York's and
    least squares, registered by name: what each fit reports, why a group
    holds no line, and the package's error class.
"""

from __future__ import annotations

from tsara.ratios.estimators import (
    MIN_FIT_POINTS,
    FitStatus,
    LineFit,
    LineFitter,
    TsaraRatioError,
    available_line_fits,
    get_line_fit,
    ols,
    register_line_fit,
    york,
)

__all__ = [
    "FitStatus",
    "LineFit",
    "LineFitter",
    "MIN_FIT_POINTS",
    "TsaraRatioError",
    "available_line_fits",
    "get_line_fit",
    "ols",
    "register_line_fit",
    "york",
]
