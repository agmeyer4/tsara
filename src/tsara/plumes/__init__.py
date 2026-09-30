"""Plume events: intervals where an enhancement stands above plume-free air.

For every gas variable, at every point of the baseline sweep, this stage
describes plume-free air by its clean level and clean spread and marks an
event where the enhancement rises a set multiple of the spread above the
level (``docs/METHODS.md`` §6.8). It reads the baseline state as a Dataset
and never imports the baseline stage. Neither the level nor the spread is a
measurement uncertainty: the spread holds the background's wobble at the
window's scale, and a declared or reported sigma plays no part in any
threshold (§2.3).

Shape of the subpackage
-----------------------
``records``
    The stretches of a stream the air is described over, one at a time: a
    variable's finite readings split at long gaps and cut to a maximum
    length; where the dropouts are; the package's error class.
``clean``
    The clean level and spread per record and sweep point, with the
    clean-level estimators registered by name (the half-sample mode), the
    quantization floor and the count below which a record is blank.
``hysteresis``
    The detector: runs of the statistic z above the exit multiple that reach
    the entry multiple, never across a dropout, a record boundary or a blank
    reading, with short dips bridged; what each event records; and the
    closed-form rate at which chance alone makes events.
"""

from __future__ import annotations

from tsara.plumes.clean import (
    MAD_TO_SIGMA,
    CleanAir,
    CleanDescription,
    CleanLevelEstimator,
    available_clean_level_estimators,
    clean_air,
    describe_clean_air,
    get_clean_level_estimator,
    half_sample_mode,
    quantization_step,
    register_clean_level_estimator,
)
from tsara.plumes.hysteresis import Events, expected_chance_rate, find_events
from tsara.plumes.records import DROPOUT_SPACING_FACTOR, Records, TsaraPlumeError, find_records

__all__ = [
    "CleanAir",
    "CleanDescription",
    "CleanLevelEstimator",
    "DROPOUT_SPACING_FACTOR",
    "Events",
    "MAD_TO_SIGMA",
    "Records",
    "TsaraPlumeError",
    "available_clean_level_estimators",
    "clean_air",
    "describe_clean_air",
    "expected_chance_rate",
    "find_events",
    "find_records",
    "get_clean_level_estimator",
    "half_sample_mode",
    "quantization_step",
    "register_clean_level_estimator",
]
