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
``state``
    The product: every gas variable of a baseline state described, scored
    and searched at every sweep point, on the stream's own cells; a variable
    named in ``plumes.triggers`` takes its trigger's events instead.
``catalog``
    The events as rows: one long table keyed by ``event_id``, spelled like
    the generator's answer key where the meaning is the same, with the
    parent-child tree along the baseline's window.
"""

from __future__ import annotations

from tsara.plumes.catalog import CATALOG_COLUMNS, link_parents, plume_catalog
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
from tsara.plumes.hysteresis import Events, describe_events, expected_chance_rate, find_events
from tsara.plumes.records import DROPOUT_SPACING_FACTOR, Records, TsaraPlumeError, find_records
from tsara.plumes.state import (
    CHANCE_ASSUMPTION_ATTR,
    ENTER_DIM,
    EXIT_DIM,
    PLUMES_STAGE,
    TRIGGER_ATTR,
    plume_state,
    plume_states,
)

__all__ = [
    "CATALOG_COLUMNS",
    "CHANCE_ASSUMPTION_ATTR",
    "CleanAir",
    "CleanDescription",
    "CleanLevelEstimator",
    "DROPOUT_SPACING_FACTOR",
    "ENTER_DIM",
    "EXIT_DIM",
    "Events",
    "MAD_TO_SIGMA",
    "PLUMES_STAGE",
    "Records",
    "TRIGGER_ATTR",
    "TsaraPlumeError",
    "available_clean_level_estimators",
    "clean_air",
    "describe_clean_air",
    "describe_events",
    "expected_chance_rate",
    "find_events",
    "find_records",
    "get_clean_level_estimator",
    "half_sample_mode",
    "link_parents",
    "plume_catalog",
    "plume_state",
    "plume_states",
    "quantization_step",
    "register_clean_level_estimator",
]
