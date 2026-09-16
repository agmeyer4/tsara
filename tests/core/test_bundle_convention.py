"""Tests for the shared bundle convention: the format-3 attribute respelling.

The loaders' end-to-end behaviour on an older bundle is tested beside each
loader (`tests/ingest/test_campaign.py`, `tests/synthetic/test_bundle.py`);
this file pins the helper both of them call.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tsara.core.bundle import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_VERSION_WITH_READINGS_AND_PROVENANCE,
    RETIRED_ATTR_NAMES,
    SUPPORTED_BUNDLE_VERSIONS,
    TsaraBundleError,
    rename_retired_attrs,
)


def _stream(attrs: dict[str, object], variable_attrs: dict[str, object]) -> xr.Dataset:
    return xr.Dataset(
        {"ch4": ("time", np.arange(3.0), dict(variable_attrs))},
        coords={"time": np.arange(3)},
        attrs=dict(attrs),
    )


def test_the_current_format_is_the_one_with_the_current_names() -> None:
    assert BUNDLE_FORMAT_VERSION == BUNDLE_VERSION_WITH_READINGS_AND_PROVENANCE
    assert SUPPORTED_BUNDLE_VERSIONS[-1] == BUNDLE_FORMAT_VERSION


def test_no_retired_name_still_says_source_and_no_current_name_does() -> None:
    """The point of the rename, stated as a check on the map itself."""
    assert all("source" in old for old in RETIRED_ATTR_NAMES)
    assert not any("source" in new for new in RETIRED_ATTR_NAMES.values())


def test_retired_names_are_respelled_on_the_dataset_and_its_variables() -> None:
    stream = _stream(
        {"tsara_support_label_source": "declared", "n_source_files": 2, "other": "kept"},
        {"uncertainty_source": "declared", "uncertainty_source_random": "reported"},
    )
    assert rename_retired_attrs(stream) is True
    assert stream.attrs == {
        "tsara_support_label_provenance": "declared",
        "n_files": 2,
        "other": "kept",
    }
    assert stream["ch4"].attrs == {
        "uncertainty_provenance": "declared",
        "uncertainty_provenance_random": "reported",
    }


def test_a_stream_with_current_names_is_left_alone() -> None:
    stream = _stream({"tsara_support_label_provenance": "declared"}, {"units": "ppb"})
    assert rename_retired_attrs(stream) is False
    assert stream.attrs == {"tsara_support_label_provenance": "declared"}
    assert stream["ch4"].attrs == {"units": "ppb"}


def test_both_spellings_of_one_attribute_are_refused() -> None:
    """No TSARA version writes both, and keeping either would drop the other."""
    stream = _stream({}, {"uncertainty_source": "declared", "uncertainty_provenance": "reported"})
    with pytest.raises(TsaraBundleError, match="both 'uncertainty_source'"):
        rename_retired_attrs(stream)
