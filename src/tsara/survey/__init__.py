"""The survey: measuring what a manifest can declare but files do not state.

The survey runs before the pipeline. It reads streams ingested as they are,
measures something no file states, and writes it as a table a person reviews
and then names in the manifest; it never changes a stream, and nothing it
writes is applied until a manifest names it (``docs/METHODS.md`` §12). Its
first measurement is clocks: how far apart each instrument's files put the
same air, segment by segment, chained to one reference clock, written as the
rows of a correction table (:mod:`tsara.core.clocks`).

Shape of the subpackage
-----------------------
``lags``
    One lag: a variable on whole seconds with its running median removed, and
    the lag at which a tier of tracer pairs correlates best over one segment;
    the package's error class.
``clocks``
    The clock survey: every instrument's tiers on every segment, the chain to
    the reference clock, outliers, rows, clock steps and drifts, and the
    faults it names.
``bundle``
    Saving and reloading a clock survey: the table to review, the three tables
    behind it, and the settings.
"""

from __future__ import annotations

from tsara.survey.bundle import load_clock_survey, save_clock_survey
from tsara.survey.clocks import FAULT_KINDS, ClockSurvey, survey_clocks
from tsara.survey.lags import (
    SecondsSeries,
    SegmentLag,
    TsaraSurveyError,
    pooled_lag,
    seconds_series,
)

__all__ = [
    "ClockSurvey",
    "FAULT_KINDS",
    "SecondsSeries",
    "SegmentLag",
    "TsaraSurveyError",
    "load_clock_survey",
    "pooled_lag",
    "save_clock_survey",
    "seconds_series",
    "survey_clocks",
]
