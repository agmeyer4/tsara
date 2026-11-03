"""Saving and reloading a clock survey: its table, the measurements behind it, its settings.

A clock survey is saved as a ``survey`` directory inside a bundle:

``clock_corrections.csv``
    The correction table the survey derived (:mod:`tsara.core.clocks`): the
    file a person reads, and names in a manifest's ``clock_corrections`` once
    content with it. Absent when the survey derived no rows.
``measurements.parquet``, ``segments.parquet``, ``faults.parquet``
    The three tables behind it (:class:`~tsara.survey.clocks.ClockSurvey`):
    every tier's lag on every segment, the lag each segment was given, and the
    faults named. Written with their declared columns and types, which Parquet
    keeps, so a reload is exact.
``survey.yaml``
    The settings the survey ran with: a survey YAML like any other, which
    :func:`~tsara.config.loader.load_survey` reads.

Why it ships in the phase that introduces it: every stage product gains
persistence where it is made (CLAUDE.md §5), and a table a person reviews
should travel with the evidence for each of its rows. ``bundle.json`` is not
touched, for the grid's reason (:mod:`tsara.core.bundle`).

What a reload preserves
-----------------------
The three tables exactly, the settings, and every row of the table, read back
by :func:`~tsara.core.clocks.read_clock_corrections` with the checks
ingestion would apply; each reloaded row names the file and line it was read
from rather than ``survey``, which is what a row read from a file names.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from tsara.config.loader import load_survey
from tsara.core.bundle import (
    BUNDLE_CLOCK_TABLE_FILE,
    BUNDLE_SURVEY_CONFIG,
    BUNDLE_SURVEY_DIR,
    BUNDLE_SURVEY_FAULTS_FILE,
    BUNDLE_SURVEY_MEASUREMENTS_FILE,
    BUNDLE_SURVEY_SEGMENTS_FILE,
    TsaraBundleError,
)
from tsara.core.clocks import read_clock_corrections, write_clock_corrections
from tsara.survey.clocks import FAULT_COLUMNS, MEASUREMENT_COLUMNS, SEGMENT_COLUMNS, ClockSurvey

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

__all__ = ["load_clock_survey", "save_clock_survey"]

#: The three tables, their files and their declared columns.
_TABLES: dict[str, tuple[str, Mapping[str, str]]] = {
    "measurements": (BUNDLE_SURVEY_MEASUREMENTS_FILE, MEASUREMENT_COLUMNS),
    "segments": (BUNDLE_SURVEY_SEGMENTS_FILE, SEGMENT_COLUMNS),
    "faults": (BUNDLE_SURVEY_FAULTS_FILE, FAULT_COLUMNS),
}


def save_clock_survey(survey: ClockSurvey, path: str | Path) -> Path:
    """Write a clock survey into a bundle's ``survey`` directory.

    Parameters
    ----------
    survey : ClockSurvey
        What :func:`~tsara.survey.clocks.survey_clocks` returned.
    path : str or pathlib.Path
        Bundle directory, created if absent. A table left by an earlier run is
        removed when this survey derived no rows, so the directory never holds
        a table its evidence does not explain.

    Returns
    -------
    pathlib.Path
        The ``survey`` directory written.

    Raises
    ------
    TsaraBundleError
        If ``path`` exists and is not a directory, or a table lacks its
        declared columns. Nothing is written then.
    """
    for name, (_, columns) in _TABLES.items():
        table: pd.DataFrame = getattr(survey, name)
        if list(table.columns) != list(columns):
            raise TsaraBundleError(
                f"The survey's {name} table has columns {list(table.columns)}; a saved "
                f"survey needs {list(columns)}. Save what survey_clocks returned."
            )
    bundle = Path(path)
    if bundle.exists() and not bundle.is_dir():
        raise TsaraBundleError(f"Bundle path '{bundle}' exists and is not a directory.")
    target = bundle / BUNDLE_SURVEY_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name, (filename, columns) in _TABLES.items():
        getattr(survey, name).astype(dict(columns)).to_parquet(target / filename, index=False)
    table_path = target / BUNDLE_CLOCK_TABLE_FILE
    if survey.rows:
        write_clock_corrections(survey.rows, table_path)
    else:
        table_path.unlink(missing_ok=True)
        logger.info("The clock survey derived no rows; no correction table written to %s.", target)
    (target / BUNDLE_SURVEY_CONFIG).write_text(
        yaml.safe_dump(survey.config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    logger.info("Wrote the clock survey, %d row(s), to %s.", len(survey.rows), target)
    return target


def load_clock_survey(path: str | Path) -> ClockSurvey:
    """Read a clock survey written by :func:`save_clock_survey`.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory, or its ``survey`` directory.

    Returns
    -------
    ClockSurvey
        The tables, the rows of the table (none if it was not written) and the
        settings.

    Raises
    ------
    TsaraBundleError
        If there is no survey there, or a part of it is missing.
    TsaraConfigError
        If the settings no longer validate.
    TsaraClockError
        If the table no longer reads as a correction table.
    """
    given = Path(path)
    target = given if given.name == BUNDLE_SURVEY_DIR else given / BUNDLE_SURVEY_DIR
    config_path = target / BUNDLE_SURVEY_CONFIG
    if not config_path.is_file():
        raise TsaraBundleError(f"No clock survey in '{given}': '{config_path}' does not exist.")
    tables = {}
    for name, (filename, _) in _TABLES.items():
        if not (target / filename).is_file():
            raise TsaraBundleError(f"The clock survey in '{target}' lacks its {name} ({filename}).")
        # Parquet keeps the types they were written with, the declared ones.
        tables[name] = pd.read_parquet(target / filename)
    config = load_survey(config_path)
    table_path = target / BUNDLE_CLOCK_TABLE_FILE
    rows = read_clock_corrections([table_path]) if table_path.is_file() else ()
    return ClockSurvey(
        measurements=tables["measurements"],
        segments=tables["segments"],
        rows=rows,
        faults=tables["faults"],
        config=config,
    )
