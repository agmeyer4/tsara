"""End-to-end acceptance tests for Phase 4, against generated ground truth.

Everything else in the suite checks a piece. These check the whole path, and
they check it against numbers the generator *knows* rather than against
another implementation of the same idea:

* a known ratio between two species measured on different clocks must survive
  pairing exactly, which it can only do if both members were averaged over the
  same air;
* the uncertainty a binned value reports must match the scatter that value
  actually has around the truth, which is the second half of the
  effective-sample-size question (``docs/METHODS.md`` §11.8) and the only way
  to find out whether the AR(1) *model* describes the generator's error rather
  than merely being solved correctly.

These are slower than the unit tests on purpose. They are the ones that would
notice a correct-looking pipeline computing the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsara.align import (
    PairedSpecies,
    bin_streams_onto_cells,
    build_output_grid,
    pair_species,
)
from tsara.config.analysis import OutputGridConfig
from tsara.core.naming import sigma_rand_name
from tsara.core.propagation import PROPAGATION_FORMS, PropagationForm
from tsara.core.support import CellBounds
from tsara.synthetic import generate
from tsara.synthetic.config import (
    GaussianShape,
    InstrumentSpec,
    LognormalAmplitude,
    ParametricBackground,
    RatioSpec,
    SourceSpec,
    SpeciesSpec,
    StationarySite,
    SyntheticConfig,
    TrueComponent,
    TrueSupport,
    TrueUncertainty,
)
from tsara.synthetic.generator import SyntheticDataset

START = pd.Timestamp("2026-07-01T00:00:00Z").to_pydatetime()
SECOND = 1_000_000_000
#: Known ratio between the two species the acceptance campaign injects.
TRUE_RATIO = 0.25


def cells_over(start_ns: int, width_s: float, n: int) -> CellBounds:
    """Return ``n`` abutting cells of ``width_s`` from ``start_ns``."""
    start = start_ns + np.arange(n, dtype=np.int64) * int(width_s * SECOND)
    return CellBounds(start_ns=start, stop_ns=start + int(width_s * SECOND))


def two_clock_campaign(*, noisy: bool) -> SyntheticConfig:
    """One source, two species, two instruments at rates 60 apart.

    Both instruments deliver *means* over their cells. That is what makes the
    ratio recoverable exactly in the noise-free case: sixty one-second means
    tile a sixty-second cell, so averaging them reproduces that cell's mean
    of the same underlying signal. A point sampler would sample instants
    instead, and the comparison would carry a discretisation error that has
    nothing to do with the code under test.
    """
    uncertainty = (
        TrueUncertainty(random=TrueComponent(absolute=2.0, report_as="fast_err")) if noisy else None
    )
    return SyntheticConfig(
        name="acceptance",
        start=START,
        duration="2h",
        seed=4242,
        platform=StationarySite(kind="stationary", latitude=40.0, longitude=-111.0),
        instruments={
            "fast": InstrumentSpec(
                native_rate="1s",
                support=TrueSupport(method="mean"),
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0),
                        units="ppb",
                        uncertainty=uncertainty,
                    )
                },
            ),
            "slow": InstrumentSpec(
                native_rate="60s",
                support=TrueSupport(method="mean"),
                species={
                    "tracer": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=0.0),
                        units="ppb",
                    )
                },
            ),
        },
        sources={
            "pad": SourceSpec(
                rate_per_hour=30.0,
                shape=GaussianShape(kind="gaussian", sigma="90s"),
                reference_species="ch4",
                amplitude=LognormalAmplitude(kind="lognormal", median=200.0, sigma_log=0.3),
                ratios={"tracer": RatioSpec(mean=TRUE_RATIO)},
            )
        },
    )


# ---------------------------------------------------------------------------
# A known ratio must survive being put on one clock
# ---------------------------------------------------------------------------


def enhancements(
    dataset: SyntheticDataset, y_name: str, x_name: str, *, min_coverage: float = 1.0
) -> tuple[PairedSpecies, np.ndarray, np.ndarray]:
    """Return the paired enhancements of the two species, above a floor."""
    paired = pair_species(dataset.streams, y_name, x_name, min_coverage=min_coverage)
    y = paired.dataset[y_name].values
    x = paired.dataset[x_name].values
    if x_name == "ch4":
        x = x - 1900.0
    enhanced = np.isfinite(x) & np.isfinite(y) & (x > 1.0)
    return paired, y[enhanced], x[enhanced]


def test_a_known_ratio_survives_pairing_across_a_sixty_fold_rate_difference() -> None:
    """The acceptance criterion of the phase.

    One source drives two species. One is measured every second, the other
    once a minute. If pairing averages both members over the same air, the
    ratio of the paired enhancements is the ratio the source was given. If it
    averages them over *different* air -- an off-by-one cell, a label read as
    a midpoint when it was a start, an overlap weight applied to the wrong
    partner -- the ratio drifts, and nothing else in the suite would say so.

    The quantity asserted is the mass-weighted ratio, which is what a
    regression slope estimates: the total tracer over the total methane. The
    per-cell scatter around it is a separate and smaller thing, checked
    below.
    """
    dataset = generate(two_clock_campaign(noisy=False))
    paired, y, x = enhancements(dataset, "tracer", "ch4")
    assert paired.clock == "slow"
    assert paired.n_pairs > 100
    assert y.sum() / x.sum() == pytest.approx(TRUE_RATIO, rel=1e-6)


def test_the_per_cell_ratio_is_right_too_and_its_scatter_is_the_generator_s() -> None:
    """Cell by cell the ratio is right to a few parts per million.

    The residual is not the aligner's: it is the midpoint-rule quadrature the
    slow instrument uses to average truth over its own 60 s cells, sixty
    times coarser than the fast instrument's. It shows up as a *relative*
    error only where the enhancement is small, which is the signature of a
    fixed absolute error rather than a ratio bias.
    """
    dataset = generate(two_clock_campaign(noisy=False))
    _, y, x = enhancements(dataset, "tracer", "ch4")
    error = np.abs((y / x) / TRUE_RATIO - 1.0)
    assert np.median(error) < 1e-4
    # Largest where the enhancement is smallest, as an absolute error must be.
    biggest = np.argmax(error)
    assert x[biggest] < np.median(x)


def test_coverage_predicts_which_paired_values_are_wrong() -> None:
    """The number that qualifies a pair, scored against ground truth.

    Coverage is meant to say how much of a cell was actually measured. With a
    known ratio there is a right answer for every cell, so the claim can be
    tested rather than asserted: the cells coverage flags are exactly the
    cells that are wrong.

    Measured on this campaign, one cell of 120 is partly covered -- the fast
    instrument's record starts inside the slow instrument's first cell -- and
    it is the only one whose ratio is off, by 10 %. Everything else is right
    to parts per million.
    """
    dataset = generate(two_clock_campaign(noisy=False))
    paired = pair_species(dataset.streams, "tracer", "ch4")
    x = paired.dataset["ch4"].values - 1900.0
    enhanced = x > 1.0
    coverage = paired.dataset["coverage_ch4"].values[enhanced]
    error = np.abs((paired.dataset["tracer"].values[enhanced] / x[enhanced]) / TRUE_RATIO - 1.0)

    partial = coverage < 1.0 - 1e-12
    assert partial.sum() == 1
    assert error[partial].min() > 100 * error[~partial].max()
    assert np.corrcoef(1.0 - coverage, error)[0, 1] > 0.99


def test_a_grid_in_phase_with_its_source_reproduces_the_paired_answer() -> None:
    """Two products, one answer -- when the cells actually line up.

    A grid is a different set of target cells, not a different operation, so
    a ratio taken from gridded columns must agree with the paired one.
    """
    dataset = generate(two_clock_campaign(noisy=False))
    boundary = pd.Timestamp(dataset.streams["slow"]["time_bnds"].values[0, 0])
    config = OutputGridConfig(freq="60s", start=boundary.to_pydatetime())
    grid = build_output_grid(dataset.streams, config, ["ch4", "tracer"])
    x = grid["ch4"].values - 1900.0
    usable = (grid["coverage_ch4"].values >= 1.0 - 1e-12) & np.isfinite(x) & (x > 1.0)
    assert np.all(grid["n_source_tracer"].values[2:-2] == 1)
    assert grid["tracer"].values[usable].sum() / x[usable].sum() == pytest.approx(
        TRUE_RATIO, rel=1e-6
    )


def test_a_grid_out_of_phase_with_its_source_blends_two_cells_and_says_so() -> None:
    """A legitimate but easily-missed smoothing, with its own diagnostic.

    The slow instrument's cells are centred on its timestamps, so they sit
    half a cell off an epoch-anchored grid of the same period. Every grid
    value is then a weighted mean of two adjacent source cells -- honest, but
    smoothed, and a ratio across two columns treated differently that way
    carries a small bias. Measured here: 0.249940526 against a truth of 0.25,
    where the in-phase grid gives 0.250000034.

    Nothing is silently wrong: `n_source` reports the blend, and the grid
    warns when it detects the phase mismatch.
    """
    dataset = generate(two_clock_campaign(noisy=False))
    grid = build_output_grid(dataset.streams, OutputGridConfig(freq="60s"), ["ch4", "tracer"])
    assert np.all(grid["n_source_tracer"].values[2:-2] == 2)
    x = grid["ch4"].values - 1900.0
    usable = (grid["coverage_ch4"].values >= 1.0 - 1e-12) & np.isfinite(x) & (x > 1.0)
    blended = grid["tracer"].values[usable].sum() / x[usable].sum()
    assert blended == pytest.approx(TRUE_RATIO, rel=1e-3)
    assert abs(blended - TRUE_RATIO) > 1e-6


def test_the_out_of_phase_grid_warns(caplog: pytest.LogCaptureFixture) -> None:
    dataset = generate(two_clock_campaign(noisy=False))
    with caplog.at_level("WARNING", logger="tsara.align.grid"):
        build_output_grid(dataset.streams, OutputGridConfig(freq="60s"), ["ch4", "tracer"])
    assert "blend two of its cells" in caplog.text


def test_pairing_never_invents_a_pair_the_generator_did_not_produce() -> None:
    """Every surviving pair holds at least one real measurement of each species."""
    dataset = generate(two_clock_campaign(noisy=True))
    paired = pair_species(dataset.streams, "tracer", "ch4")
    assert np.all(paired.dataset["n_source_ch4"].values > 0)
    assert np.all(paired.dataset["n_source_tracer"].values > 0)
    assert np.isfinite(paired.dataset["ch4"].values).all()
    assert np.isfinite(paired.dataset["tracer"].values).all()


# ---------------------------------------------------------------------------
# Does the reported uncertainty match the scatter that actually occurs?
# ---------------------------------------------------------------------------


def error_study_config(tau: str | None) -> SyntheticConfig:
    """A long, quiet record whose only feature is a known error structure.

    No plumes and a flat background, so the residual between a binned
    observable and the binned truth *is* the error of the binned value and
    nothing else. Absolute-only sigma, so that error is identically
    distributed across cells and their scatter estimates it.
    """
    return SyntheticConfig(
        name="error_study",
        start=START,
        duration="6h",
        seed=20260909,
        platform=StationarySite(kind="stationary", latitude=40.0, longitude=-111.0),
        instruments={
            "analyzer": InstrumentSpec(
                native_rate="1s",
                species={
                    "ch4": SpeciesSpec(
                        background=ParametricBackground(kind="parametric", offset=1900.0),
                        units="ppb",
                        uncertainty=TrueUncertainty(
                            random=TrueComponent(absolute=5.0, report_as="ch4_err"),
                            decorrelation_timescale=tau,
                        ),
                    )
                },
            )
        },
        sources={},
    )


def binned_error(
    tau: str | None, width_s: float, form: PropagationForm
) -> tuple[float, float, float]:
    """Return (observed scatter, reported sigma, mean cells) for one setting.

    The observed scatter is the standard deviation, across cells, of the
    difference between the binned observable and the binned truth. The
    reported sigma is what the propagation says that difference should be.
    """
    dataset = generate(error_study_config(tau))
    stream = dataset.streams["analyzer"].copy()
    # The generator publishes a per-point sigma; name it the way ingestion
    # would so the binner picks it up as the random component.
    stream[sigma_rand_name("ch4")] = stream["ch4_err"]
    stream["ch4"].attrs["uncertainty_source_random"] = "reported"
    if tau is not None:
        stream["ch4"].attrs["decorrelation_timescale"] = tau
    truth = stream["truth_background_ch4"] + stream["truth_enhancement_ch4"]
    stream["truth"] = truth
    stream["truth"].attrs = {"units": "ppb"}

    first = int(stream["time_bnds"].values[0, 0].astype("int64"))
    n_cells = int((6 * 3600) / width_s) - 2
    target = cells_over(first, width_s, n_cells)
    joined = bin_streams_onto_cells(
        {"analyzer": stream}, target, ["ch4", "truth"], propagation_form=form
    )
    residual = joined["ch4"].values - joined["truth"].values
    good = np.isfinite(residual)
    return (
        float(np.std(residual[good], ddof=1)),
        float(np.nanmean(joined[sigma_rand_name("ch4")].values)),
        float(np.nanmean(joined["n_source_ch4"].values)),
    )


def test_uncorrelated_error_averages_down_exactly_as_reported() -> None:
    """No timescale declared, so the samples are independent by declaration.

    Sixty one-second samples of a 5 ppb error should leave a 60 s mean
    uncertain by 5/sqrt(60). This is the baseline the correlated case is
    measured against.
    """
    observed, reported, n = binned_error(None, 60.0, "ar1_neff")
    assert n == pytest.approx(60.0, rel=0.05)
    assert reported == pytest.approx(5.0 / np.sqrt(60.0), rel=0.05)
    assert observed == pytest.approx(reported, rel=0.15)


def test_correlated_error_does_not_average_down_as_far_and_the_report_knows() -> None:
    """The measurement that closes the second half of the N_eff question.

    Stage 1 established that the finite-N form solves the AR(1) model
    correctly. It could not establish that the AR(1) *model* describes the
    error the generator injects. This does: the same record, with a declared
    30 s timescale, scatters far more than the independent rule predicts, and
    the reported sigma follows the scatter rather than the naive rule.
    """
    observed, reported, _ = binned_error("30s", 60.0, "ar1_neff")
    naive = 5.0 / np.sqrt(60.0)
    assert observed > 2 * naive
    assert observed == pytest.approx(reported, rel=0.25)


def test_the_asymptotic_form_is_the_one_that_disagrees() -> None:
    """Same data, three forms, one answer to compare them against.

    The finite-N form tracks the observed scatter; the large-N approximation
    overstates it, which is the practical consequence of METHODS §3.4's
    comparison table and the reason it is not the default.
    """
    observed, exact, _ = binned_error("30s", 60.0, "ar1_neff")
    _, asymptotic, _ = binned_error("30s", 60.0, "ar1_asymptotic")
    assert asymptotic > exact
    assert abs(exact - observed) < abs(asymptotic - observed)


@pytest.mark.parametrize("form", PROPAGATION_FORMS)
def test_every_registered_form_runs_end_to_end(form: PropagationForm) -> None:
    """Including the pairwise double sum, which no other test drives through
    the binner. A registered name that cannot be selected is not registered."""
    observed, reported, _ = binned_error("30s", 30.0, form)
    assert np.isfinite(observed)
    assert np.isfinite(reported)
