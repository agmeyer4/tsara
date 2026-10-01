"""Fixtures shared by the plumes tests: the example campaign, rolled and searched once."""

from __future__ import annotations

from typing import TypeAlias

import pytest
import xarray as xr

from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig
from tsara.config.loader import load_synthetic
from tsara.events import event_state
from tsara.synthetic import SyntheticDataset, generate

#: The example campaign, its Picarro's baseline state and its event state. Test
#: files spell this alias again rather than importing it, as they do the other
#: fixture types.
ExampleChain: TypeAlias = tuple[SyntheticDataset, xr.Dataset, xr.Dataset]


@pytest.fixture(scope="session")
def example_chain() -> ExampleChain:
    """The scoping measurement's rule (METHODS §6.8).

    The example campaign generated; its 2 s Picarro's methane (true sigma
    0.6 ppb, one 6 h record, plume-dense) rolled at 2, 10 and 60 min with
    q = 0.05, its sigma columns set aside; events at entry 3, exit 1, nothing
    bridged. Generated once per session, since three tests read it.
    """
    campaign = generate(load_synthetic("examples/configs/synthetic_example.yaml"))
    stream = campaign.observable("picarro")
    stream = stream.drop_vars([v for v in stream.data_vars if str(v).startswith("sigma_")])
    state = baseline_state(
        stream,
        instrument="picarro",
        baseline=BaselineConfig(windows=("2min", "10min", "60min"), quantiles=(0.05,)),
        variables=["ch4"],
    )
    found = event_state(state, instrument="picarro", events=EventsConfig(max_internal_gap="1ns"))
    return campaign, state, found
