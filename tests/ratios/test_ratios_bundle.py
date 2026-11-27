"""Tests for saving and reloading the ratio stage's tables (tsara.ratios.bundle, METHODS §5.7).

The evidence: a save and a load give back the fits, the ratio table and the
configuration exactly, column types included; the ratio table saved is the one
``candidate_ratios`` reads from those fits at the configuration's gates and
confidence level, each row naming the parent the candidate table gave; what
the configuration does not ask for is refused and nothing is written; and a
directory that is not what ``save_ratios`` writes is refused on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tsara.config.analysis import GatesConfig, RatiosConfig, RatioSetConfig
from tsara.core.bundle import (
    BUNDLE_ANALYSIS_CONFIG,
    BUNDLE_FITS_FILE,
    BUNDLE_RATIOS_DIR,
    BUNDLE_RATIOS_FILE,
    TsaraBundleError,
)
from tsara.ratios import (
    FIT_COLUMNS,
    FIT_DTYPES,
    candidate_ratios,
    load_ratios,
    save_ratios,
)

CONFIG = RatiosConfig(
    sets=(RatioSetConfig(reference="van.ch4", numerators=("van.c2h6", "ptr")),),
    methods=("york", "ols"),
    periods=("native", "5s"),
    gates=GatesConfig(min_peak_z=(4.0, 6.0), plateau_probability=(0.99, 1.0)),
    confidence_level=0.9,
)


def _fit(candidate: str, slope: float, **values: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(FIT_COLUMNS, np.nan)
    row.update(
        event_id=f"{candidate}/e{slope:g}",
        candidate_id=candidate,
        reference="van.ch4",
        numerator="van.c2h6",
        period="native",
        method="york",
        baseline_window=600.0,
        baseline_quantile=0.05,
        exit_multiple=1.0,
        start_time=pd.Timestamp("2026-01-01 00:09:30"),
        peak_time=pd.Timestamp("2026-01-01 00:10"),
        end_time=pd.Timestamp("2026-01-01 00:11"),
        peak_z=10.0,
        candidate_peak_z=10.0,
        n=50,
        n_dropped=0,
        iterations=6,
        dof=48,
        plateau_dof=98,
        status="OK",
        slope=slope,
        intercept=0.1,
        slope_sigma=0.001,
        slope_sigma_widened=0.002,
        mswd=1.0,
        mswd_expected=1.0,
        signal=10.0,
        plateau_mswd=1.0,
        baseline_error="entered",
        reference_sigma_provenance="declared",
        numerator_sigma_provenance="declared",
        reference_systematic_relative=0.01,
        numerator_systematic_relative=np.nan,
        reference_systematic_provenance="declared",
        numerator_systematic_provenance="unknown",
    )
    row.update(values)
    return row


def _fits() -> pd.DataFrame:
    """Three candidates: one with two members, one fitted twice (a PTR numerator, least
    squares on the 5 s clock), one whose only fit has no line."""
    return pd.DataFrame(
        [
            _fit("van.ch4/w600s/c0", 0.050),
            _fit("van.ch4/w600s/c0", 0.054, exit_multiple=2.0, candidate_peak_z=10.0),
            _fit(
                "van.ch4/w600s/c1", 0.30, numerator="ptr.benzene", peak_z=5.0, candidate_peak_z=5.0
            ),
            _fit(
                "van.ch4/w600s/c1",
                0.31,
                numerator="ptr.benzene",
                method="ols",
                period="5s",
                peak_z=5.0,
                candidate_peak_z=5.0,
                mswd=np.nan,
            ),
            _fit("van.ch4/w600s/c2", np.nan, status="MISSING_SIGMA"),
        ]
    ).astype(FIT_DTYPES)


def _candidates() -> pd.DataFrame:
    """The events stage's candidate table, its other columns left out (save reads two)."""
    return pd.DataFrame(
        {
            "candidate_id": [f"van.ch4/w600s/c{i}" for i in range(4)],
            "parent_candidate_id": ["van.ch4/w3600s/c0", None, "van.ch4/w3600s/c0", None],
        }
    )


def test_a_reload_gives_back_the_tables_and_the_configuration_exactly(tmp_path: Path) -> None:
    fits = _fits()
    target = save_ratios(fits, tmp_path, candidates=_candidates(), ratios=CONFIG)
    assert target == tmp_path / BUNDLE_RATIOS_DIR
    assert sorted(p.name for p in target.iterdir()) == sorted(
        [BUNDLE_FITS_FILE, BUNDLE_RATIOS_FILE, BUNDLE_ANALYSIS_CONFIG]
    )
    loaded = load_ratios(tmp_path)
    pd.testing.assert_frame_equal(loaded.fits, fits)
    expected = candidate_ratios(fits, _candidates(), CONFIG.gates, confidence_level=0.9)
    pd.testing.assert_frame_equal(loaded.candidate_ratios, expected)
    assert loaded.ratios == CONFIG
    # Read from the ratios directory itself too.
    pd.testing.assert_frame_equal(load_ratios(target).fits, fits)


def test_the_ratio_table_is_read_at_the_configurations_gates_and_level(tmp_path: Path) -> None:
    save_ratios(_fits(), tmp_path, candidates=_candidates(), ratios=CONFIG)
    table = load_ratios(tmp_path).candidate_ratios
    # Four gate settings (height 4/6 x plateau on/off) of three ratios of c0..c1 and c2's.
    assert len(table) == 4 * 4
    assert set(table.min_peak_z) == {4.0, 6.0} and set(table.plateau_probability) == {0.99, 1.0}
    assert set(table.confidence_level) == {0.9}
    # Each row names the parent the candidate table gave, blank where it gave none.
    parents = table.drop_duplicates("candidate_id").set_index("candidate_id").parent_candidate_id
    assert parents["van.ch4/w600s/c0"] == "van.ch4/w3600s/c0"
    assert pd.isna(parents["van.ch4/w600s/c1"])
    # c1's height 5 fails the 6 gate; c2's fit has no line and passes nothing.
    c1 = table[(table.candidate_id == "van.ch4/w600s/c1") & (table.method == "york")]
    assert c1.set_index("min_peak_z").n_members.to_dict() == {4.0: 1, 6.0: 0}
    assert (table[table.candidate_id == "van.ch4/w600s/c2"].n_members == 0).all()


def test_no_fits_save_and_reload_as_empty_tables_of_their_types(tmp_path: Path) -> None:
    empty = pd.DataFrame(columns=list(FIT_COLUMNS)).astype(FIT_DTYPES)
    save_ratios(empty, tmp_path, candidates=_candidates(), ratios=RatiosConfig())
    loaded = load_ratios(tmp_path)
    pd.testing.assert_frame_equal(loaded.fits, empty)
    assert loaded.candidate_ratios.empty
    assert loaded.ratios == RatiosConfig()


def test_a_second_save_replaces_the_first(tmp_path: Path) -> None:
    save_ratios(_fits(), tmp_path, candidates=_candidates(), ratios=CONFIG)
    fewer = _fits().iloc[:2].reset_index(drop=True)
    other = CONFIG.model_copy(update={"gates": GatesConfig()})
    save_ratios(fewer, tmp_path, candidates=_candidates(), ratios=other)
    loaded = load_ratios(tmp_path)
    pd.testing.assert_frame_equal(loaded.fits, fewer)
    assert loaded.ratios == other and len(loaded.candidate_ratios) == 1


def test_an_instrument_entry_asks_for_every_one_of_its_variables(tmp_path: Path) -> None:
    """``ptr`` in the set's numerators stands for every PTR gas: ptr.toluene is asked for."""
    fits = pd.DataFrame([_fit("van.ch4/w600s/c0", 0.2, numerator="ptr.toluene")]).astype(FIT_DTYPES)
    save_ratios(fits, tmp_path, candidates=_candidates(), ratios=CONFIG)
    assert load_ratios(tmp_path).fits.numerator.tolist() == ["ptr.toluene"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"reference": "picarro.ch4"}, "which no ratio set of the configuration names"),
        ({"numerator": "van.co"}, "that ratio set does not name"),
        ({"method": "odr"}, r"method \['odr'\]"),
        ({"period": "10s"}, r"period \['10s'\]"),
        ({"candidate_id": "van.ch4/w600s/c9"}, "candidate table does not hold"),
    ],
)
def test_what_the_configuration_or_the_candidates_do_not_hold_is_refused(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    fits = pd.DataFrame([_fit("van.ch4/w600s/c0", 0.05, **change)]).astype(FIT_DTYPES)
    with pytest.raises(TsaraBundleError, match=message):
        save_ratios(fits, tmp_path / "bundle", candidates=_candidates(), ratios=CONFIG)
    assert not (tmp_path / "bundle").exists()


def test_tables_that_are_not_the_stages_are_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    with pytest.raises(TsaraBundleError, match="columns are not the fit table's"):
        save_ratios(_fits().drop(columns="mswd"), bundle, candidates=_candidates(), ratios=CONFIG)
    with pytest.raises(TsaraBundleError, match=r"without \['parent_candidate_id'\]"):
        save_ratios(_fits(), bundle, candidates=_candidates()[["candidate_id"]], ratios=CONFIG)
    assert not bundle.exists()
    bundle.write_text("not a directory")
    with pytest.raises(TsaraBundleError, match="is not a directory"):
        save_ratios(_fits(), bundle, candidates=_candidates(), ratios=CONFIG)


def test_a_directory_save_ratios_did_not_write_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TsaraBundleError, match="holds no ratios"):
        load_ratios(tmp_path)
    target = save_ratios(_fits(), tmp_path, candidates=_candidates(), ratios=CONFIG)
    for name in (BUNDLE_FITS_FILE, BUNDLE_RATIOS_FILE, BUNDLE_ANALYSIS_CONFIG):
        kept = (target / name).read_bytes()
        (target / name).unlink()
        with pytest.raises(TsaraBundleError, match=f"has no '{name}'"):
            load_ratios(tmp_path)
        (target / name).write_bytes(kept)
    load_ratios(tmp_path)


@pytest.mark.parametrize(
    ("name", "whose"), [(BUNDLE_FITS_FILE, "fit table's"), (BUNDLE_RATIOS_FILE, "ratio table's")]
)
def test_a_table_without_its_columns_is_refused_on_load(
    tmp_path: Path, name: str, whose: str
) -> None:
    target = save_ratios(_fits(), tmp_path, candidates=_candidates(), ratios=CONFIG)
    table = pd.read_parquet(target / name)
    table.drop(columns=table.columns[-1]).to_parquet(target / name, index=False)
    with pytest.raises(TsaraBundleError, match=f"columns are not the {whose}"):
        load_ratios(tmp_path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("events:\n  enter_multiple: [3.0]\n", "no 'ratios' section"),
        ("ratios:\n  confidence_level: 2.0\n", "confidence_level"),
        ("ratios:\n  methods: [york]\n  methods: [ols]\n", "found duplicate key"),
    ],
)
def test_a_configuration_that_is_not_a_ratios_section_is_refused(
    tmp_path: Path, text: str, message: str
) -> None:
    target = save_ratios(_fits(), tmp_path, candidates=_candidates(), ratios=CONFIG)
    (target / BUNDLE_ANALYSIS_CONFIG).write_text(text, encoding="utf-8")
    with pytest.raises(TsaraBundleError, match=message):
        load_ratios(tmp_path)


def test_the_configuration_is_written_as_a_section_of_an_analysis_configuration(
    tmp_path: Path,
) -> None:
    target = save_ratios(_fits(), tmp_path, candidates=_candidates(), ratios=CONFIG)
    written = yaml.safe_load((target / BUNDLE_ANALYSIS_CONFIG).read_text(encoding="utf-8"))
    assert list(written) == ["ratios"]
    assert RatiosConfig.model_validate(written["ratios"]) == CONFIG
