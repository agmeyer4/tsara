"""Tests for the shared bundle convention: the format-3 respelling and the format-4 relabel.

The loaders' end-to-end behaviour on an older bundle is tested beside each
loader (`tests/ingest/test_campaign.py`, `tests/synthetic/test_bundle.py`,
and the grid's and baseline state's bundle tests); this file pins the
helpers they call.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tsara.core.bundle import (
    BUNDLE_FORMAT_VERSION,
    BUNDLE_VERSION_WITH_READINGS_AND_PROVENANCE,
    BUNDLE_VERSION_WITHOUT_PROMISED_ESTIMATES,
    RETIRED_ATTR_NAMES,
    SUPPORTED_BUNDLE_VERSIONS,
    TsaraBundleError,
    relabel_promised_estimates,
    rename_retired_attrs,
)


def _stream(attrs: dict[str, object], variable_attrs: dict[str, object]) -> xr.Dataset:
    return xr.Dataset(
        {"ch4": ("time", np.arange(3.0), dict(variable_attrs))},
        coords={"time": np.arange(3)},
        attrs=dict(attrs),
    )


def test_the_current_format_is_the_one_with_the_current_labels() -> None:
    assert BUNDLE_FORMAT_VERSION == BUNDLE_VERSION_WITHOUT_PROMISED_ESTIMATES
    assert BUNDLE_VERSION_WITH_READINGS_AND_PROVENANCE < BUNDLE_FORMAT_VERSION
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


# ---------------------------------------------------------------------------
# Format 4: the `empirical` nothing estimated
# ---------------------------------------------------------------------------


def _labelled(**variables: dict[str, object]) -> xr.Dataset:
    """A dataset of one-value variables carrying the given attrs."""
    return xr.Dataset(
        {name: ("time", np.arange(3.0), dict(attrs)) for name, attrs in variables.items()},
        coords={"time": np.arange(3)},
    )


def test_an_unkept_empirical_becomes_unknown_with_its_species_label() -> None:
    """No budget at all: format 3 wrote `empirical` twice, and both were promises."""
    stream = _labelled(
        ch4={
            "uncertainty_provenance": "empirical",
            "uncertainty_provenance_random": "empirical",
            "uncertainty_provenance_systematic": "unknown",
        }
    )
    assert relabel_promised_estimates(stream) is True
    assert stream["ch4"].attrs == {
        "uncertainty_provenance": "unknown",
        "uncertainty_provenance_random": "unknown",
        "uncertainty_provenance_systematic": "unknown",
    }


def test_a_mixed_species_stays_mixed() -> None:
    """A declared systematic with no random made `mixed`, which is still true."""
    stream = _labelled(
        ch4={
            "uncertainty_provenance": "mixed",
            "uncertainty_provenance_random": "empirical",
            "uncertainty_provenance_systematic": "declared",
        }
    )
    assert relabel_promised_estimates(stream) is True
    assert stream["ch4"].attrs["uncertainty_provenance_random"] == "unknown"
    assert stream["ch4"].attrs["uncertainty_provenance"] == "mixed"


def test_an_estimate_that_exists_keeps_its_label() -> None:
    """The baseline's order-statistic sigma is `empirical` and true (METHODS §6.7).

    Two shapes are left alone: the sigma companion itself, and a variable
    that has one beside it.
    """
    state = _labelled(
        baseline_ch4={"uncertainty_provenance_random": "empirical"},
        sigma_rand_baseline_ch4={
            "uncertainty_component": "random",
            "uncertainty_provenance": "empirical",
        },
    )
    assert relabel_promised_estimates(state) is False
    assert state["baseline_ch4"].attrs["uncertainty_provenance_random"] == "empirical"
    assert state["sigma_rand_baseline_ch4"].attrs["uncertainty_provenance"] == "empirical"


def test_a_dataset_with_honest_labels_is_left_alone() -> None:
    stream = _labelled(ch4={"uncertainty_provenance_random": "declared"}, t={"units": "K"})
    assert relabel_promised_estimates(stream) is False
    assert stream["ch4"].attrs == {"uncertainty_provenance_random": "declared"}
    assert stream["t"].attrs == {"units": "K"}
