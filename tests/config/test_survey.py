"""Tests for the survey settings (``tsara.config.survey``).

The chain is the part that must hold: every instrument's partners lead to the
reference clock, without loops, so that every lag composes to it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from tsara.config.loader import load_survey
from tsara.config.survey import SurveyConfig
from tsara.core.exceptions import TsaraConfigError

BASE: dict[str, Any] = {
    "reference_clock": "airmar",
    "tracers": {
        "picarro": [{"partner": "airmar", "pairs": [["h2o", "rh"]], "min_pairs": 100}],
        "aeris": [
            {"partner": "picarro", "pairs": [["ch4", "ch4"]]},
            {"partner": "airmar", "pairs": [["h2o", "rh"]]},
        ],
    },
}


def _spec(**changes: Any) -> dict[str, Any]:
    spec = copy.deepcopy(BASE)
    spec.update(changes)
    return spec


def test_a_minimal_config_takes_the_defaults() -> None:
    config = SurveyConfig.model_validate(BASE)
    assert (config.segment, config.search, config.min_pairs) == ("30min", "120s", 300)
    assert (config.min_r, config.tolerance, config.running_median) == (0.6, "2s", "121s")
    assert config.resolution == "1s"
    assert config.max_bridged == "0s"
    assert config.tracers["picarro"][0].pairs == (("h2o", "rh"),)
    assert config.tracers["picarro"][0].min_pairs == 100
    assert config.tracers["aeris"][0].min_pairs is None


def test_a_segment_holding_exactly_the_paired_seconds_a_tier_needs_is_allowed() -> None:
    SurveyConfig.model_validate(_spec(segment="5min", min_pairs=300))


def test_the_composition_order_puts_every_partner_first() -> None:
    spec = _spec(
        tracers={
            "caps": [{"partner": "miro", "pairs": [["nox", "nox"]]}],
            "miro": [{"partner": "picarro", "pairs": [["ch4", "ch4"]]}],
            "picarro": [{"partner": "airmar", "pairs": [["h2o", "rh"]]}],
        }
    )
    assert SurveyConfig.model_validate(spec).composition_order() == ("picarro", "miro", "caps")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (
            {
                "tracers": {
                    **BASE["tracers"],
                    "airmar": [{"partner": "picarro", "pairs": [["a", "b"]]}],
                }
            },
            "reference_clock 'airmar' has tiers",
        ),
        (
            {"tracers": {"aeris": [{"partner": "miro", "pairs": [["ch4", "ch4"]]}]}},
            "partner 'miro', which is neither",
        ),
        (
            {"tracers": {"aeris": [{"partner": "aeris", "pairs": [["ch4", "ch4"]]}]}},
            "'aeris' is its own partner",
        ),
        (
            {
                "tracers": {
                    "aeris": [{"partner": "miro", "pairs": [["ch4", "ch4"]]}],
                    "miro": [{"partner": "aeris", "pairs": [["ch4", "ch4"]]}],
                }
            },
            "form a loop",
        ),
        ({"search": "30min"}, "must be shorter than a segment"),
        ({"segment": "0s"}, "positive"),
        ({"tolerance": "-2s"}, "positive"),
        ({"min_r": 0.0}, "greater than 0"),
        ({"min_r": 1.5}, "less than or equal to 1"),
        ({"min_pairs": 0}, "greater than 0"),
        ({"tracers": {"aeris": []}}, "has no tiers"),
        ({"tracers": {"aeris": [{"partner": "airmar", "pairs": []}]}}, "at least 1 item"),
        ({"tracers": {"aeris": [{"partner": "airmar", "pairs": [["ch4"]]}]}}, "pairs"),
        (
            {
                "tracers": {
                    "aeris": [{"partner": "airmar", "pairs": [["ch4", "ch4"]], "min_pairs": 0}]
                }
            },
            "greater than 0",
        ),
        ({"tracers": {"aeris": [{"partner": "airmar", "pairs": [["ch 4", "ch4"]]}]}}, "identifier"),
        ({"tracers": {}}, "at least 1 item"),
        ({"segmnet": "10min"}, "Extra inputs are not permitted"),
        # Read as whole seconds, so a fraction would be cut off without a word:
        # "20.7s" ran as 20, and "500ms" as a search of no width at all.
        ({"search": "20.7s"}, "search .* whole number of seconds"),
        ({"search": "500ms"}, "search .* whole number of seconds"),
        ({"running_median": "121.5s"}, "running_median .* whole number of seconds"),
        ({"resolution": "2s"}, "at most 1s"),
        ({"resolution": "0s"}, "positive"),
        ({"max_bridged": "-1min"}, "must not be negative"),
        ({"max_bridged": "soon"}, "not a valid timedelta"),
        # Paired seconds cannot outnumber a segment's: such a tier never decides.
        ({"segment": "3min", "min_pairs": 200}, "'aeris' tier 1 needs 200 paired seconds"),
        (
            {
                "tracers": {
                    "aeris": [{"partner": "airmar", "pairs": [["ch4", "ch4"]], "min_pairs": 2000}]
                }
            },
            "'aeris' tier 1 needs 2000 paired seconds, more than a segment of 30min",
        ),
    ],
)
def test_impossible_settings_are_refused(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        SurveyConfig.model_validate(_spec(**changes))


def test_load_survey_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "survey.yaml"
    path.write_text(yaml.safe_dump(_spec(segment="10min")), encoding="utf-8")
    assert load_survey(path).segment == "10min"


def test_load_survey_names_the_file_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(_spec(min_r=2)), encoding="utf-8")
    with pytest.raises(TsaraConfigError, match="bad.yaml"):
        load_survey(path)
