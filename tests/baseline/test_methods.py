"""Tests for the baseline method registry (tsara.baseline.methods).

The methods themselves are exercised through the state in `test_state.py`,
where their results are visible as columns; here is the registry's own
contract, which mirrors the reader registry's.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import TypeAdapter

from tsara.baseline import (
    BaselineRequest,
    BaselineResult,
    TsaraBaselineError,
    available_baseline_methods,
    get_baseline_method,
    register_baseline_method,
)
from tsara.baseline.methods import _METHODS
from tsara.config.analysis import BaselineMethod


def test_the_three_configured_methods_are_registered() -> None:
    """The schema's discriminator and the registry name the same three methods."""
    assert available_baseline_methods() == ("constant", "from_field", "rolling_quantile")
    adapter: TypeAdapter[Any] = TypeAdapter(BaselineMethod)
    for name in available_baseline_methods():
        entry: dict[str, object] = {"method": name}
        if name == "from_field":
            entry["instrument"] = "ptr"
        if name == "constant":
            entry["value"] = 0.0
        assert adapter.validate_python(entry).method == name


def test_an_unregistered_name_is_refused_listing_what_exists() -> None:
    with pytest.raises(TsaraBaselineError, match="lowess.*Available.*rolling_quantile"):
        get_baseline_method("lowess")


def test_a_name_is_registered_once_unless_replacement_is_asked_for(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def method(request: BaselineRequest, /) -> BaselineResult:  # pragma: no cover
        raise NotImplementedError

    try:
        register_baseline_method("test_method")(method)
        with pytest.raises(ValueError, match="already registered as 'test_method'"):
            register_baseline_method("test_method")(method)
        with caplog.at_level(logging.WARNING, logger="tsara.baseline.methods"):
            register_baseline_method("test_method", replace=True)(method)
        assert "Replacing baseline method 'test_method'" in caplog.text
        assert get_baseline_method("test_method") is method
    finally:
        _METHODS.pop("test_method", None)


@pytest.mark.parametrize("name", ["", "   "])
def test_a_blank_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_baseline_method(name)
