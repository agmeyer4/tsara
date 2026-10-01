"""Tests for saving and reloading event states and the catalog (tsara.events.bundle)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import xarray as xr

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.core.bundle import (
    BUNDLE_ANALYSIS_CONFIG,
    BUNDLE_CATALOG_FILE,
    BUNDLE_EVENTS_DIR,
    TsaraBundleError,
    pin_time_encoding,
)
from tsara.core.support import CellBounds, declared_bounds_name
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import event_catalog, event_states, load_events, save_events

EVENTS = EventsConfig(enter_multiple=(3.0, 4.0), exit_multiple=(1.0,), triggers={"iwas": "van.ch4"})


def make_stream(
    midpoints_s: npt.NDArray[np.float64],
    values: dict[str, npt.NDArray[np.float64]],
    *,
    width_s: float = 1.0,
) -> xr.Dataset:
    """A stream of gas variables on cells of ``width_s`` centred on the midpoints."""
    mid = np.round(midpoints_s * SECOND).astype(np.int64)
    half = int(width_s * SECOND) // 2
    bounds = CellBounds(start_ns=mid - half, stop_ns=mid + half)
    stream = xr.Dataset(
        {name: ("time", v, {"units": "ppb", "role": "gas"}) for name, v in values.items()},
        coords={
            "time": bounds.midpoint_ns.astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([bounds.start_ns, bounds.stop_ns], axis=1).astype("datetime64[ns]"),
            ),
        },
        attrs={"tsara_stage": "synthetic"},
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    return stream


@pytest.fixture(scope="module")
def run() -> tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]:
    """A 1 s analyzer and a canister that takes its events: baseline and event
    states, and the catalog."""
    rng = np.random.default_rng(1)
    t = np.arange(7200.0)
    events = np.zeros(t.size)
    for centre in np.arange(300.0, 7200.0, 600.0):
        events += 40 * np.exp(-0.5 * ((t - centre) / 15) ** 2)
    van = make_stream(t, {"ch4": 1900 + rng.normal(0, 0.7, t.size) + events})
    fills = np.arange(300.0, 7200.0, 300.0)
    canister = make_stream(fills, {"benzene": np.ones(fills.size)}, width_s=15.0)
    config = BaselineConfig(windows=("2min", "10min"), quantiles=(0.05,))
    baselines = {
        "van": baseline_state(van, instrument="van", baseline=config),
        "iwas": baseline_state(canister, instrument="iwas", baseline=config),
    }
    states = event_states(baselines, EVENTS)
    return baselines, states, event_catalog(states, baselines)


def assert_identical(a: xr.Dataset, b: xr.Dataset) -> None:
    """Every variable, coordinate and attribute exactly.

    One documented exception, as for the baseline state: opening a file with
    ``decode_coords="all"`` moves the CF ``bounds`` attribute of ``time`` into
    its encoding, so that attribute is compared through the function that reads
    both places.
    """
    assert sorted(map(str, a.variables)) == sorted(map(str, b.variables))
    assert declared_bounds_name(a) == declared_bounds_name(b) == "time_bnds"
    for name in a.variables:
        assert a[name].dims == b[name].dims, name
        assert a[name].dtype == b[name].dtype, name
        assert np.array_equal(a[name].values, b[name].values, equal_nan=True), name
        attrs_a = {k: v for k, v in a[name].attrs.items() if not (name == "time" and k == "bounds")}
        attrs_b = {k: v for k, v in b[name].attrs.items() if not (name == "time" and k == "bounds")}
        assert attrs_a == attrs_b, name
    assert a.attrs == b.attrs


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


def test_the_round_trip_is_exact_for_states_catalog_and_config(
    tmp_path: Path, run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]
) -> None:
    _, states, catalog = run
    target = save_events(states, tmp_path / "bundle", catalog=catalog, events=EVENTS)
    assert target == tmp_path / "bundle" / BUNDLE_EVENTS_DIR
    assert sorted(p.name for p in target.iterdir()) == [
        BUNDLE_ANALYSIS_CONFIG,
        BUNDLE_CATALOG_FILE,
        "iwas.nc",
        "van.nc",
    ]
    back = load_events(tmp_path / "bundle")
    assert sorted(back.states) == ["iwas", "van"]
    for name, state in states.items():
        assert_identical(state, back.states[name])
        assert "time_bnds" in back.states[name].coords
    assert back.catalog is not None
    pd.testing.assert_frame_equal(back.catalog, catalog)
    assert back.events == EVENTS
    # The events directory itself is a valid path too.
    assert sorted(load_events(target).states) == ["iwas", "van"]


def test_a_reloaded_state_catalogs_as_the_saved_one_did(
    tmp_path: Path, run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]
) -> None:
    """The catalog is rebuilt from the states, so the two saved together agree."""
    baselines, states, catalog = run
    save_events(states, tmp_path, catalog=catalog)
    pd.testing.assert_frame_equal(event_catalog(load_events(tmp_path).states, baselines), catalog)


def test_only_the_events_section_is_saved(
    tmp_path: Path, run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]
) -> None:
    from tsara.config.loader import read_yaml

    _, states, _ = run
    target = save_events(states, tmp_path, events=EVENTS)
    assert list(read_yaml(target / BUNDLE_ANALYSIS_CONFIG)) == ["events"]


def test_compression_changes_the_size_and_nothing_else(
    tmp_path: Path, run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]
) -> None:
    _, states, _ = run
    plain = save_events(states, tmp_path / "plain")
    small = save_events(states, tmp_path / "small", compression=4)
    assert (small / "van.nc").stat().st_size < 0.5 * (plain / "van.nc").stat().st_size
    assert_identical(
        load_events(tmp_path / "plain").states["van"], load_events(tmp_path / "small").states["van"]
    )
    for bad in (0, 10, True):
        with pytest.raises(TsaraBundleError, match="zlib level"):
            save_events(states, tmp_path / "bad", compression=bad)


# ---------------------------------------------------------------------------
# The directory is the record of what ran
# ---------------------------------------------------------------------------


def test_saving_without_a_catalog_or_config_writes_neither_and_says_so(
    tmp_path: Path,
    run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, states, _ = run
    with caplog.at_level(logging.INFO, logger="tsara.events.bundle"):
        target = save_events(states, tmp_path)
    assert not (target / BUNDLE_CATALOG_FILE).exists()
    assert not (target / BUNDLE_ANALYSIS_CONFIG).exists()
    assert "without their events configuration" in caplog.text
    back = load_events(tmp_path)
    assert back.catalog is None and back.events is None


def test_what_an_earlier_run_left_is_removed_and_nothing_else(
    tmp_path: Path,
    run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A wider run with a catalog, then a narrower one without: the state file and
    the catalog it no longer backs are removed, a note left there is kept."""
    _, states, catalog = run
    save_events(states, tmp_path, catalog=catalog)
    (tmp_path / BUNDLE_EVENTS_DIR / "notes.txt").write_text("kept")
    with caplog.at_level(logging.INFO, logger="tsara.events.bundle"):
        save_events({"van": states["van"]}, tmp_path)
    names = sorted(p.name for p in (tmp_path / BUNDLE_EVENTS_DIR).iterdir())
    assert names == ["notes.txt", "van.nc"]
    assert "Removing stale event state file" in caplog.text
    assert "Removing a catalog left by an earlier run" in caplog.text


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_what_is_not_an_event_state_is_refused_at_save_and_at_load(
    tmp_path: Path, run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]
) -> None:
    baselines, states, _ = run
    with pytest.raises(TsaraBundleError, match="No event states"):
        save_events({}, tmp_path)
    with pytest.raises(TsaraBundleError, match="tsara_stage 'baseline'"):
        save_events({"van": baselines["van"]}, tmp_path)
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(TsaraBundleError, match="not a directory"):
        save_events(states, blocker)
    with pytest.raises(TsaraBundleError, match="not an existing directory"):
        load_events(tmp_path / "missing")
    (tmp_path / BUNDLE_EVENTS_DIR).mkdir()
    with pytest.raises(TsaraBundleError, match="holds no event state file"):
        load_events(tmp_path)
    stranger = baselines["van"].copy()
    pin_time_encoding(stranger)
    stranger.to_netcdf(tmp_path / BUNDLE_EVENTS_DIR / "van.nc")
    with pytest.raises(TsaraBundleError, match="written by the 'baseline' stage"):
        load_events(tmp_path)


def test_a_state_that_lost_its_bounds_is_refused_before_writing(
    tmp_path: Path, run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]
) -> None:
    _, states, _ = run
    with pytest.raises(Exception, match="bounds"):
        save_events({"van": states["van"].drop_vars("time_bnds")}, tmp_path)


def test_a_table_that_is_not_the_catalog_is_refused_at_save_and_at_load(
    tmp_path: Path, run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]
) -> None:
    _, states, catalog = run
    with pytest.raises(TsaraBundleError, match="columns are not the catalog's"):
        save_events(states, tmp_path, catalog=catalog.drop(columns="covered"))
    target = save_events(states, tmp_path, catalog=catalog)
    catalog[["event_id"]].to_parquet(target / BUNDLE_CATALOG_FILE)
    with pytest.raises(TsaraBundleError, match="columns are not the catalog's"):
        load_events(tmp_path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("baseline:\n  windows: [2min]\n  quantiles: [0.05]\n", "no 'events' section"),
        ("events:\n  enter_multiple: [3.0]\n  enter_multiple: [4.0]\n", "duplicate key"),
        ("events:\n  enter_multiple: [1.0]\n", "must exceed every exit_multiple"),
    ],
)
def test_a_configuration_that_does_not_read_is_refused_by_name(
    tmp_path: Path,
    run: tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame],
    text: str,
    message: str,
) -> None:
    _, states, _ = run
    target = save_events(states, tmp_path, events=EVENTS)
    (target / BUNDLE_ANALYSIS_CONFIG).write_text(text)
    with pytest.raises(
        TsaraBundleError, match=f"(?s)Could not read the events configuration.*{message}"
    ):
        load_events(tmp_path)
