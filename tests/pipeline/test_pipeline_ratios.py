"""Tests for the ratio stage over a configuration (tsara.pipeline.ratio_fits, METHODS §5.2).

``ratio_fits`` computes nothing of its own: it pairs each numerator with its
reference on each pair clock and hands the pairing to the fits. So the
evidence is that its table is, frame for frame, what those two calls give by
hand in the configuration's order; that an ``'<instrument>'`` numerator
expands over the variables that instrument's baseline state holds an
enhancement of, the reference left out; and the refusals.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.align import pair_species
from tsara.baseline import baseline_state
from tsara.config.analysis import BaselineConfig, EventsConfig, RatiosConfig
from tsara.core.naming import enhanced_variables
from tsara.core.timebase import SECOND_NS as SECOND
from tsara.events import event_catalog, event_states
from tsara.pipeline import ratio_fits
from tsara.ratios import FIT_DTYPES, PairColumns, TsaraRatioError, member_fits

START = pd.Timestamp("2026-01-01").value
#: The fixture: baseline states, event states and the event catalog.
Campaign = tuple[dict[str, xr.Dataset], dict[str, xr.Dataset], pd.DataFrame]
DECLARED = {
    "uncertainty_provenance": "declared",
    "uncertainty_provenance_random": "declared",
    "uncertainty_provenance_systematic": "zero",
}


def _stream(values: dict[str, np.ndarray], sigmas: dict[str, float], width_s: int) -> xr.Dataset:
    """A stream of back-to-back cells ``width_s`` long holding the given gases."""
    n = next(iter(values.values())).size
    start = START + np.arange(n, dtype=np.int64) * width_s * SECOND
    data: dict[str, tuple[str, np.ndarray, dict[str, object]]] = {}
    for name, v in values.items():
        data[name] = (
            "time",
            v,
            {"units": "ppb", "role": "gas", "cell_methods": "time: mean", **DECLARED},
        )
        data["sigma_rand_" + name] = (
            "time",
            np.full(n, sigmas[name]),
            {
                "units": "ppb",
                "uncertainty_component": "random",
                "uncertainty_provenance": "declared",
            },
        )
    stream = xr.Dataset(
        data,
        coords={
            "time": (start + width_s * SECOND // 2).astype("datetime64[ns]"),
            "time_bnds": (
                ("time", "nv"),
                np.stack([start, start + width_s * SECOND], 1).astype("datetime64[ns]"),
            ),
        },
    )
    stream["time"].attrs["bounds"] = "time_bnds"
    return stream


@pytest.fixture(scope="module")
def campaign() -> Campaign:
    """Half an hour of one source's plumes: a 1 s analyzer of three gases, a 2 s one of methane."""
    rng = np.random.default_rng(7)
    n = 1800
    t = np.arange(n, dtype=float)
    plume = np.zeros(n)
    for centre, width, height in ((300, 20, 25), (700, 35, 40), (1100, 15, 18), (1500, 25, 30)):
        plume += height * np.exp(-0.5 * ((t - centre) / width) ** 2)
    van = _stream(
        {
            "ch4": 1950 + plume + rng.normal(0, 0.7, n),
            "c2h6": 2.0 + 0.05 * plume + rng.normal(0, 0.05, n),
            "co": 120 + 0.4 * plume + rng.normal(0, 2.0, n),
        },
        {"ch4": 0.7, "c2h6": 0.05, "co": 2.0},
        width_s=1,
    )
    half = plume.reshape(-1, 2).mean(axis=1)
    lab = _stream({"ch4": 1950 + half + rng.normal(0, 1.0, half.size)}, {"ch4": 1.0}, width_s=2)
    baseline = BaselineConfig(windows=("2min", "10min"), quantiles=(0.05,))
    logging.disable(logging.WARNING)
    try:
        states = {
            name: baseline_state(stream, instrument=name, baseline=baseline)
            for name, stream in (("van", van), ("lab", lab))
        }
        events = event_states(
            states, EventsConfig(enter_multiple=(3.0, 5.0), exit_multiple=(1.0, 2.0))
        )
        catalog = event_catalog(events, states)
    finally:
        logging.disable(logging.NOTSET)
    return states, events, catalog


def _by_hand(
    states: dict[str, xr.Dataset],
    events: dict[str, xr.Dataset],
    catalog: pd.DataFrame,
    reference: str,
    numerator: str,
    period: str,
    methods: tuple[str, ...] = ("ols", "york"),
) -> pd.DataFrame:
    """The pairing and the fits called directly, as METHODS §5.2 and §5.4 describe them:
    both readings and both clean spreads carried onto the pair cells."""
    r_inst, r_var = reference.split(".")
    n_inst, n_var = numerator.split(".")
    streams = dict(states)
    for inst, var in ((n_inst, n_var), (r_inst, r_var)):
        streams[inst] = streams[inst].assign(
            {f"clean_spread_{var}": events[inst][f"clean_spread_{var}"]}
        )
    paired = pair_species(
        streams,
        (n_inst, f"enhancement_{n_var}"),
        (r_inst, f"enhancement_{r_var}"),
        carry=[
            (n_inst, n_var),
            (r_inst, r_var),
            (n_inst, f"clean_spread_{n_var}"),
            (r_inst, f"clean_spread_{r_var}"),
        ],
        target=None if period == "native" else period,
    )
    return member_fits(
        paired.dataset,
        catalog,
        PairColumns(paired.y_name, paired.x_name, *paired.carried),
        reference=reference,
        numerator=numerator,
        period=period,
        methods=methods,
    )


def _config(**kwargs: object) -> RatiosConfig:
    return RatiosConfig.model_validate(kwargs)


def test_the_table_is_each_pairings_fits_in_the_configurations_order(
    campaign: Campaign,
) -> None:
    states, events, catalog = campaign
    got = ratio_fits(
        states,
        events,
        catalog,
        _config(sets=[{"reference": "van.ch4", "numerators": ["van"]}], periods=["native", "10s"]),
    )
    expected = pd.concat(
        [
            _by_hand(states, events, catalog, "van.ch4", numerator, period)
            for numerator in ("van.c2h6", "van.co")
            for period in ("native", "10s")
        ],
        ignore_index=True,
    )
    assert len(got) > 0
    pd.testing.assert_frame_equal(got, expected)


def test_an_instrument_expands_over_its_enhancements_without_the_reference(
    campaign: Campaign,
) -> None:
    states, events, catalog = campaign
    assert enhanced_variables(states["van"]) == ["ch4", "c2h6", "co"]
    got = ratio_fits(
        states, events, catalog, _config(sets=[{"reference": "van.c2h6", "numerators": ["van"]}])
    )
    assert list(dict.fromkeys(got.numerator)) == ["van.ch4", "van.co"]
    assert set(got.reference) == {"van.c2h6"}


def test_a_reference_on_another_instrument_is_paired_on_the_wider_cells(
    campaign: Campaign,
) -> None:
    states, events, catalog = campaign
    got = ratio_fits(
        states,
        events,
        catalog,
        _config(sets=[{"reference": "lab.ch4", "numerators": ["van.c2h6"]}]),
    )
    pd.testing.assert_frame_equal(
        got, _by_hand(states, events, catalog, "lab.ch4", "van.c2h6", "native")
    )
    # The 2 s analyzer's methane is the reference, so its candidates are the members'.
    assert len(got) > 0
    assert got.candidate_id.str.startswith("lab.ch4/").all()


def test_ratio_sets_follow_one_another_and_only_the_named_estimators_run(
    campaign: Campaign,
) -> None:
    states, events, catalog = campaign
    got = ratio_fits(
        states,
        events,
        catalog,
        _config(
            sets=[
                {"reference": "van.ch4", "numerators": ["van.co"]},
                {"reference": "lab.ch4", "numerators": ["van.c2h6"]},
            ],
            methods=["york"],
        ),
    )
    expected = pd.concat(
        [
            _by_hand(states, events, catalog, "van.ch4", "van.co", "native", ("york",)),
            _by_hand(states, events, catalog, "lab.ch4", "van.c2h6", "native", ("york",)),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(got, expected)
    assert set(got.method) == {"york"}


def test_no_ratio_set_gives_an_empty_table_of_the_declared_types(
    campaign: Campaign,
) -> None:
    states, events, catalog = campaign
    got = ratio_fits(states, events, catalog, _config())
    assert got.empty
    assert {name: str(dtype) for name, dtype in got.dtypes.items()} == {
        name: str(pd.Series(dtype=dtype).dtype) for name, dtype in FIT_DTYPES.items()
    }


@pytest.mark.parametrize(
    ("ratio_set", "match"),
    [
        ({"reference": "van.so2", "numerators": ["van"]}, "reference 'van.so2'"),
        ({"reference": "ship.ch4", "numerators": ["van"]}, "reference 'ship.ch4'"),
        ({"reference": "van.ch4", "numerators": ["ship"]}, "no instrument 'ship'"),
        ({"reference": "van.ch4", "numerators": ["van.so2"]}, "so2"),
    ],
)
def test_names_no_baseline_state_holds_are_refused(
    campaign: Campaign, ratio_set: dict[str, object], match: str
) -> None:
    states, events, catalog = campaign
    with pytest.raises(TsaraRatioError, match=match):
        ratio_fits(states, events, catalog, _config(sets=[ratio_set]))


def test_the_fixture_has_events_on_both_references(
    campaign: Campaign,
) -> None:
    """Guard: a test over a reference with no events would compare two empty tables."""
    _, _, catalog = campaign
    held = catalog.groupby(["instrument", "species"]).size()
    assert held.get(("van", "ch4"), 0) > 0
    assert held.get(("lab", "ch4"), 0) > 0


def test_each_gas_brings_its_clean_spread_so_the_fits_hold_the_baselines_wobble(
    campaign: Campaign,
) -> None:
    states, events, catalog = campaign
    got = ratio_fits(
        states,
        events,
        catalog,
        _config(sets=[{"reference": "van.ch4", "numerators": ["van.c2h6"]}]),
    )
    assert got.wobble_reference.notna().all() and got.wobble_numerator.notna().all()
    assert set(got.baseline_error) <= {"wobble and wander", "wobble"}


def test_a_variable_with_no_event_state_has_no_wobble(
    campaign: Campaign, caplog: pytest.LogCaptureFixture
) -> None:
    states, events, catalog = campaign
    with caplog.at_level(logging.WARNING, logger="tsara.pipeline.ratios"):
        got = ratio_fits(
            states,
            {"van": events["van"]},  # the 2 s analyzer's event state left out
            catalog,
            # Two numerators against the one reference with no clean spread: said once.
            _config(sets=[{"reference": "lab.ch4", "numerators": ["van"]}]),
        )
    assert set(got.numerator) == {"van.ch4", "van.c2h6", "van.co"}
    assert got.wobble_reference.isna().all() and got.wobble_numerator.notna().all()
    assert set(got.baseline_error) <= {"wander", "none"}
    said = [r.getMessage() for r in caplog.records if "clean spread" in r.getMessage()]
    assert len(said) == 1 and said[0].startswith("lab.ch4 has no clean spread")


def test_events_found_on_the_reference_alone_are_warned_once_per_gas(
    campaign: Campaign, caplog: pytest.LogCaptureFixture
) -> None:
    """The usual way to lose the wobble: events found on methane only, so ethane has no
    clean spread. Said once, though two pair clocks fit it twice."""
    states, events, catalog = campaign
    reference_only = {"van": events["van"].drop_vars("clean_spread_c2h6"), "lab": events["lab"]}
    with caplog.at_level(logging.WARNING, logger="tsara.pipeline.ratios"):
        got = ratio_fits(
            states,
            reference_only,
            catalog,
            _config(
                sets=[{"reference": "van.ch4", "numerators": ["van.c2h6"]}],
                periods=["native", "10s"],
            ),
        )
    assert got.wobble_numerator.isna().all() and got.wobble_reference.notna().any()
    said = [r.getMessage() for r in caplog.records if "clean spread" in r.getMessage()]
    assert len(said) == 1 and said[0].startswith("van.c2h6 has no clean spread")
    assert "Find events on every gas" in said[0]


def test_no_warning_when_every_gas_brings_its_clean_spread(
    campaign: Campaign, caplog: pytest.LogCaptureFixture
) -> None:
    states, events, catalog = campaign
    with caplog.at_level(logging.WARNING, logger="tsara.pipeline.ratios"):
        ratio_fits(
            states,
            events,
            catalog,
            _config(sets=[{"reference": "van.ch4", "numerators": ["van.c2h6"]}]),
        )
    assert not [r for r in caplog.records if "clean spread" in r.getMessage()]


def test_the_chain_ends_in_a_ratio_per_candidate_naming_its_parent(campaign: Campaign) -> None:
    from tsara.config.analysis import GatesConfig
    from tsara.events import event_candidates
    from tsara.ratios import candidate_ratios

    states, events, catalog = campaign
    fits = ratio_fits(
        states,
        events,
        catalog,
        _config(sets=[{"reference": "van.ch4", "numerators": ["van.c2h6"]}]),
    )
    candidates = event_candidates(catalog, events)
    ratios = candidate_ratios(fits, candidates, GatesConfig())
    reference = candidates[(candidates.instrument == "van") & (candidates.species == "ch4")]
    # Every one of the reference's candidates has a row per estimator, passing or not.
    assert set(ratios.candidate_id) == set(reference.candidate_id)
    assert len(ratios) == 2 * len(reference)
    parents = reference.set_index("candidate_id").parent_candidate_id
    pd.testing.assert_series_equal(
        ratios.set_index("candidate_id").parent_candidate_id.groupby(level=0).first(),
        parents.sort_index(),
        check_names=False,
    )
    assert (ratios.n_members > 0).any()
