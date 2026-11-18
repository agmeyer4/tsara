"""Ratios: the slope of one gas's enhancement against another's, and how far to trust it.

An enhancement ratio is a straight line fitted through one gas's enhancement
against another's over a stretch of readings (``docs/METHODS.md`` §4). Both
axes carry measurement error, so the preferred line is York's, which weighs
every point by its declared sigmas on both axes and reports how well they
explain the scatter; ordinary least squares is kept beside it. This package
holds the line fits and the stage's fits over the events' members; it reads
nothing from the other stages, which hand it their products.

Shape of the subpackage
-----------------------
``estimators``
    Straight lines fitted through many groups of points at once, York's and
    least squares, registered by name: what each fit reports, why a group
    holds no line, and the package's error class.
``baseline_error``
    The baseline's error across a member, as a covariance between its points:
    the background's wander, measured on the pair cells outside every event,
    and the baseline's own wobble, from the events stage's clean spread
    (§5.4).
``fits``
    The stage's fits: every numerator against its reference over each member
    of each of the reference's candidates, on a paired product, with every
    number the gates read and the slope's sigma widened for the readings'
    persistence and the baseline's error (§5.2-5.4).
``candidate_ratio``
    A candidate's ratio, read from its members that pass the gates at every
    gate setting: the median member, its fit's sigma, the members' spread and
    the declared systematic, combined; and the fingerprint, a candidate's
    ratios across numerators (§5.5).
"""

from __future__ import annotations

from tsara.ratios.candidate_ratio import (
    GATE_COLUMNS,
    RATIO_COLUMNS,
    RATIO_DTYPES,
    candidate_ratios,
    fingerprints,
)
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
from tsara.ratios.fits import FIT_COLUMNS, FIT_DTYPES, PLATEAU_WIDTH, PairColumns, member_fits

__all__ = [
    "FIT_COLUMNS",
    "FIT_DTYPES",
    "FitStatus",
    "GATE_COLUMNS",
    "LineFit",
    "LineFitter",
    "MIN_FIT_POINTS",
    "PLATEAU_WIDTH",
    "PairColumns",
    "RATIO_COLUMNS",
    "RATIO_DTYPES",
    "TsaraRatioError",
    "available_line_fits",
    "candidate_ratios",
    "fingerprints",
    "get_line_fit",
    "member_fits",
    "ols",
    "register_line_fit",
    "york",
]
