"""Ratios: the slope of one gas's enhancement against another's, and how far to trust it.

An enhancement ratio is a straight line fitted through one gas's enhancement
against another's over a stretch of readings (``docs/METHODS.md`` §4). Both
axes carry measurement error, so the preferred line is York's, which weighs
every point by its declared sigmas on both axes and reports how well they
explain the scatter; ordinary least squares is kept beside it. This package
holds the line fits, the stage's fits over the events' members, a candidate's
ratio read from them, the sweep summary, and their saving; it reads nothing
from the other stages, which hand it their products.

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
``gates``
    Which fits count at each setting of the gates' thresholds: height, points,
    signal, one source and plateau, each a list, every combination a setting
    (§5.3).
``candidate_ratio``
    A candidate's ratio, read from its members that pass the gates at every
    gate setting: the median member, its fit's sigma, the members' spread and
    the declared systematic, combined; and the fingerprint, a candidate's
    ratios across numerators (§5.5).
``sweep``
    The sweep summary: at every sweep point and gate setting, how many fits
    count and, over a named population, their median ratio, its spread, the
    median MSWD and the Birge ratio; a view built on demand (§5.6).
``bundle``
    Saving and reloading the stage's tables under a bundle's ``ratios/``: the
    fits, every candidate's ratio at every gate setting (built from the fits
    as they are written) and the ratios section of the configuration (§5.7).
"""

from __future__ import annotations

from tsara.ratios.bundle import RatioTables, load_ratios, save_ratios
from tsara.ratios.candidate_ratio import (
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
from tsara.ratios.gates import GATE_COLUMNS, GATE_NAMES, gate_passes, gate_settings, passes
from tsara.ratios.sweep import SWEEP_DIMS, sweep_summary

__all__ = [
    "FIT_COLUMNS",
    "FIT_DTYPES",
    "FitStatus",
    "GATE_COLUMNS",
    "GATE_NAMES",
    "LineFit",
    "LineFitter",
    "MIN_FIT_POINTS",
    "PLATEAU_WIDTH",
    "PairColumns",
    "RATIO_COLUMNS",
    "RATIO_DTYPES",
    "RatioTables",
    "SWEEP_DIMS",
    "TsaraRatioError",
    "available_line_fits",
    "candidate_ratios",
    "fingerprints",
    "gate_passes",
    "gate_settings",
    "get_line_fit",
    "load_ratios",
    "member_fits",
    "ols",
    "passes",
    "register_line_fit",
    "save_ratios",
    "sweep_summary",
    "york",
]
