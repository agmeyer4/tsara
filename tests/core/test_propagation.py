"""Tests for uncertainty propagation under averaging.

The load-bearing tests here are the ones that check the arithmetic against an
*independent* evaluation rather than against itself: the finite-sample
effective-size form is scored against a brute-force double sum built from the
correlation matrix, and the two components are checked to move in opposite
directions under averaging. The rest guard the edges where a wrong answer
would look plausible -- a masked sample, an empty cell, a missing timescale.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsara.core.propagation import (
    DOUBLE_SUM_MAX_POINTS,
    INDEPENDENT_FORM,
    PROPAGATION_FORMS,
    TsaraPropagationError,
    kish_sample_size,
    lag1_correlation,
    n_effective,
    propagate_random,
    propagate_random_binned,
    propagate_systematic,
    propagate_systematic_binned,
    sigma_at_support,
)
from tsara.core.propagation import (
    _variance_ratio_equal_weight as variance_ratio,
)


def brute_force_ratio(n: int, rho: float) -> float:
    """Return ``Var(mean)/sigma^2`` by building the correlation matrix.

    The independent check. It is the definition from METHODS §3.1 with no
    algebra applied, so agreement with it means the closed form is right
    rather than merely self-consistent.
    """
    index = np.arange(n)
    correlation = rho ** np.abs(index[:, None] - index[None, :])
    return float(correlation.sum() / n**2)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_registered_forms_match_the_type() -> None:
    """The tuple is derived from the Literal, so they cannot drift."""
    assert PROPAGATION_FORMS == ("ar1_neff", "ar1_asymptotic", "ar1_double_sum")


def test_independent_is_not_a_selectable_form() -> None:
    """Independence is what an undeclared timescale means, not a choice.

    Keeping it out of the registry is the same rule as the uncertainty
    provenance ladder: the order is automatic, only the algorithm backing a
    rung is a user choice.
    """
    assert INDEPENDENT_FORM not in PROPAGATION_FORMS


# ---------------------------------------------------------------------------
# lag1_correlation
# ---------------------------------------------------------------------------


def test_lag1_correlation_is_the_exponential() -> None:
    assert lag1_correlation(1.0, 20.0) == pytest.approx(np.exp(-0.05))
    # One timescale of spacing leaves 1/e.
    assert lag1_correlation(20.0, 20.0) == pytest.approx(1 / np.e)


@pytest.mark.parametrize(("spacing", "tau"), [(0.0, 20.0), (-1.0, 20.0)])
def test_lag1_correlation_rejects_nonpositive_spacing(spacing: float, tau: float) -> None:
    with pytest.raises(TsaraPropagationError, match="spacing must be positive"):
        lag1_correlation(spacing, tau)


@pytest.mark.parametrize("tau", [0.0, -5.0])
def test_lag1_correlation_rejects_nonpositive_tau(tau: float) -> None:
    """A zero timescale must be spelled as no timescale.

    Both mean 'uncorrelated', but only one of them says so in the provenance
    label, and a stored number whose meaning depends on which spelling was
    used is exactly what the provenance system exists to prevent.
    """
    with pytest.raises(TsaraPropagationError, match="Declare no"):
        lag1_correlation(1.0, tau)


# ---------------------------------------------------------------------------
# The finite-N variance ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 17, 60, 200])
@pytest.mark.parametrize("rho", [0.0, 0.1, 0.5, 0.9, 0.99, 0.999])
def test_finite_n_sum_matches_brute_force(n: int, rho: float) -> None:
    """The closed form is the double sum, evaluated cheaply.

    The direct summation exists because the algebraic closed form loses all
    precision as rho approaches 1: it subtracts two quantities of order 1e4
    that agree to four figures, through a denominator of 1e-6. This test is
    what proves the cheap spelling did not cost accuracy.
    """
    assert variance_ratio(n, rho) == pytest.approx(brute_force_ratio(n, rho), rel=1e-12)


def test_finite_n_sum_at_perfect_correlation() -> None:
    """rho = 1 exactly: averaging changes nothing, so the ratio is 1."""
    assert variance_ratio(10, 1.0) == 1.0


def test_finite_n_sum_of_one_sample() -> None:
    assert variance_ratio(1, 0.9) == 1.0


# ---------------------------------------------------------------------------
# n_effective
# ---------------------------------------------------------------------------


def test_n_effective_recovers_n_when_uncorrelated() -> None:
    """tau -> 0 is the independent case: every sample counts."""
    assert n_effective(60, 1.0, 1e-9)[0] == pytest.approx(60.0)


def test_n_effective_collapses_to_one_when_fully_correlated() -> None:
    """tau -> infinity: sixty samples carry one sample's information."""
    assert n_effective(60, 1.0, 1e12)[0] == pytest.approx(1.0, abs=1e-6)


def test_n_effective_is_bounded_by_one_and_n() -> None:
    for tau in (0.01, 1.0, 20.0, 1e6):
        value = n_effective(60, 1.0, tau)[0]
        assert 1.0 <= value <= 60.0


def test_n_effective_clips_counts_below_one() -> None:
    """An empty cell cannot hold less than one sample's worth of information."""
    assert n_effective(0, 1.0, 20.0)[0] == 1.0


def test_n_effective_is_vectorized_and_matches_scalar_calls() -> None:
    counts = np.array([1, 5, 5, 60])
    batch = n_effective(counts, 1.0, 20.0)
    one_at_a_time = [n_effective(int(c), 1.0, 20.0)[0] for c in counts]
    assert batch == pytest.approx(one_at_a_time)


def test_n_effective_always_returns_an_array() -> None:
    """Scalar in, array out, so callers need one code path."""
    assert isinstance(n_effective(10, 1.0, 20.0), np.ndarray)


def test_asymptotic_form_understates_effective_size_on_short_windows() -> None:
    """METHODS §3.4's large-N limit, and the size of its error.

    The worked case from the Phase-2 notes: a 200 s window at 1 Hz with a
    100 s timescale. The asymptotic form says the window is worth exactly one
    independent sample; the finite-N truth is 1.76, so it overstates the
    uncertainty by 33 %.
    """
    exact = n_effective(200, 1.0, 100.0)[0]
    asymptotic = n_effective(200, 1.0, 100.0, form="ar1_asymptotic")[0]
    assert exact == pytest.approx(1.7616, abs=1e-3)
    assert asymptotic == pytest.approx(1.0, abs=1e-9)
    assert np.sqrt(exact / asymptotic) == pytest.approx(1.327, abs=1e-3)


def test_asymptotic_form_agrees_when_the_window_is_many_timescales() -> None:
    """The limit it is the limit of: 600 samples, 2 s timescale."""
    exact = n_effective(600, 1.0, 2.0)[0]
    asymptotic = n_effective(600, 1.0, 2.0, form="ar1_asymptotic")[0]
    assert asymptotic == pytest.approx(exact, rel=0.01)


def test_n_effective_refuses_the_pairwise_form() -> None:
    with pytest.raises(TsaraPropagationError, match="needs the actual timestamps"):
        n_effective(10, 1.0, 20.0, form="ar1_double_sum")


def test_n_effective_rejects_an_unregistered_form() -> None:
    with pytest.raises(TsaraPropagationError, match="Unknown propagation form"):
        n_effective(10, 1.0, 20.0, form="magic")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# kish_sample_size
# ---------------------------------------------------------------------------


def test_kish_size_of_equal_weights_is_the_count() -> None:
    assert kish_sample_size(np.ones(7)) == pytest.approx(7.0)


def test_kish_size_of_one_dominant_weight_approaches_one() -> None:
    weights = np.array([1000.0, 1.0, 1.0])
    assert kish_sample_size(weights) == pytest.approx(1.004, abs=1e-3)


def test_kish_size_of_no_weight_is_zero() -> None:
    assert kish_sample_size(np.zeros(4)) == 0.0


def test_kish_size_rejects_negative_weights() -> None:
    with pytest.raises(TsaraPropagationError, match="non-negative"):
        kish_sample_size([1.0, -1.0])


def test_kish_size_rejects_two_dimensional_weights() -> None:
    with pytest.raises(TsaraPropagationError, match="one-dimensional"):
        kish_sample_size(np.ones((2, 2)))


# ---------------------------------------------------------------------------
# propagate_random
# ---------------------------------------------------------------------------


def test_independent_random_errors_average_down_as_root_n() -> None:
    sigma = np.full(60, 2.0)
    result = propagate_random(sigma, np.ones(60))
    assert result.sigma == pytest.approx(2.0 / np.sqrt(60))
    assert result.form == INDEPENDENT_FORM
    assert result.n_effective == pytest.approx(60.0)


def test_random_propagation_uses_the_callers_weights_not_inverse_variance() -> None:
    """The uncertainty must describe the number it is attached to.

    Inverse-variance weights would be the minimum-variance choice for
    *estimating a constant*, but the value being reported is an
    overlap-weighted time average, so its uncertainty is the one implied by
    those same overlap weights. Two very different sigmas with lopsided
    weights make the difference visible.
    """
    sigma = np.array([1.0, 10.0])
    weights = np.array([1.0, 3.0])
    result = propagate_random(sigma, weights)
    expected = np.sqrt((0.25 * 1.0) ** 2 + (0.75 * 10.0) ** 2)
    assert result.sigma == pytest.approx(expected)
    inverse_variance = np.sqrt(1.0 / (1.0 / 1.0**2 + 1.0 / 10.0**2))
    assert result.sigma != pytest.approx(inverse_variance)


def test_correlated_random_errors_average_down_less() -> None:
    sigma = np.full(60, 2.0)
    times = np.arange(60.0)
    independent = propagate_random(sigma, np.ones(60))
    correlated = propagate_random(sigma, np.ones(60), tau_s=20.0, times_s=times)
    assert correlated.sigma > independent.sigma
    assert correlated.n_effective < independent.n_effective


def test_neff_and_double_sum_agree_on_equal_spacing() -> None:
    """Where the effective-size form's assumptions hold, it is exact.

    Equal spacing and equal sigma is the case the closed form was derived
    for, so any disagreement here would be an implementation error rather
    than a modelling approximation.
    """
    sigma = np.full(60, 2.0)
    times = np.arange(60.0)
    fast = propagate_random(sigma, np.ones(60), tau_s=20.0, times_s=times, form="ar1_neff")
    exact = propagate_random(sigma, np.ones(60), tau_s=20.0, times_s=times, form="ar1_double_sum")
    assert fast.sigma == pytest.approx(exact.sigma, rel=1e-12)
    assert fast.n_effective == pytest.approx(exact.n_effective, rel=1e-12)


def test_double_sum_and_neff_disagree_on_unequal_sigmas() -> None:
    """And this is why the pairwise form is kept.

    The effective-size route reduces the weights to one number before it ever
    sees the times, so it cannot know that the large-sigma samples are the
    ones bunched together. The pairwise form does.
    """
    sigma = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    times = np.array([0.0, 1.0, 2.0, 100.0, 200.0, 300.0])
    fast = propagate_random(sigma, np.ones(6), tau_s=20.0, times_s=times, form="ar1_neff")
    exact = propagate_random(sigma, np.ones(6), tau_s=20.0, times_s=times, form="ar1_double_sum")
    assert exact.sigma != pytest.approx(fast.sigma, rel=1e-3)


def test_the_cheap_form_is_not_always_conservative() -> None:
    """Half the readings in a burst and half spread: `ar1_neff` understates σ.

    METHODS §3.4 claimed the cheap form errs "on the safe side, bounded by how
    badly a cell's readings clump". That is true of the ordinary clumping
    patterns and false of this one, which is why the claim was withdrawn and
    this case pinned.

    The mechanism is the median. `ar1_neff` knows only a count and the median
    gap; with fifteen readings inside 0.15 s and fifteen spread 100 s apart,
    the median gap is the *large* one, so it treats thirty readings as thirty
    nearly independent ones — while half of them are really a single effective
    reading. Measured against exact AR(1) draws it reports about a third of the
    true uncertainty; the pairwise form is right to within a per cent.
    """
    rng = np.random.default_rng(20260920)
    tau_s, sigma_value, n_trials = 20.0, 2.0, 4000
    times = np.concatenate([np.linspace(0.0, 0.15, 15), 100.0 * (1 + np.arange(15.0))])
    sigmas = np.full(times.size, sigma_value)
    weights = np.ones(times.size)

    # The truth: exact AR(1) errors at these instants, averaged, many times.
    lag = np.abs(times[:, None] - times[None, :])
    covariance = sigma_value**2 * np.exp(-lag / tau_s)
    draws = rng.multivariate_normal(
        np.zeros(times.size), covariance, size=n_trials, method="cholesky"
    )
    observed = float(draws.mean(axis=1).std(ddof=1))

    cheap = propagate_random(sigmas, weights, tau_s=tau_s, times_s=times, form="ar1_neff")
    exact = propagate_random(sigmas, weights, tau_s=tau_s, times_s=times, form="ar1_double_sum")

    # The pairwise form needs no tolerance argument: it evaluates the same
    # covariance the draws came from. 4 % covers the sampling error of the
    # standard deviation of 4000 means (about 1.1 %) with room to spare.
    assert exact.sigma == pytest.approx(observed, rel=0.04)
    # And the cheap one is wrong in the dangerous direction, by a lot.
    assert cheap.sigma < 0.5 * observed


def test_the_cheap_form_is_conservative_for_one_cadence_with_a_hole() -> None:
    """The shape every real stream has, where the claim does hold.

    Two blocks of fifteen a half-minute apart carry *more* information than a
    single run, because the blocks have had time to decorrelate; the cheap
    form cannot see that and overstates σ, which is the harmless direction.
    """
    rng = np.random.default_rng(20260921)
    tau_s, sigma_value, n_trials = 20.0, 2.0, 4000
    times = np.concatenate([np.arange(15.0), 30.0 + np.arange(15.0)])
    sigmas = np.full(times.size, sigma_value)
    weights = np.ones(times.size)

    lag = np.abs(times[:, None] - times[None, :])
    covariance = sigma_value**2 * np.exp(-lag / tau_s)
    draws = rng.multivariate_normal(
        np.zeros(times.size), covariance, size=n_trials, method="cholesky"
    )
    observed = float(draws.mean(axis=1).std(ddof=1))

    cheap = propagate_random(sigmas, weights, tau_s=tau_s, times_s=times, form="ar1_neff")
    exact = propagate_random(sigmas, weights, tau_s=tau_s, times_s=times, form="ar1_double_sum")
    assert exact.sigma == pytest.approx(observed, rel=0.04)
    assert cheap.sigma > observed


def test_double_sum_reproduces_independence_when_tau_is_tiny() -> None:
    sigma = np.full(20, 3.0)
    times = np.arange(20.0)
    exact = propagate_random(sigma, np.ones(20), tau_s=1e-9, times_s=times, form="ar1_double_sum")
    assert exact.sigma == pytest.approx(3.0 / np.sqrt(20))
    assert exact.n_effective == pytest.approx(20.0)


def test_double_sum_reproduces_full_correlation_when_tau_is_huge() -> None:
    sigma = np.full(20, 3.0)
    times = np.arange(20.0)
    exact = propagate_random(sigma, np.ones(20), tau_s=1e12, times_s=times, form="ar1_double_sum")
    assert exact.sigma == pytest.approx(3.0, rel=1e-6)
    assert exact.n_effective == pytest.approx(1.0, rel=1e-6)


def test_double_sum_needs_times() -> None:
    with pytest.raises(TsaraPropagationError, match="needs times_s"):
        propagate_random(np.ones(4), np.ones(4), tau_s=20.0, form="ar1_double_sum")


def test_double_sum_needs_one_time_per_sigma() -> None:
    with pytest.raises(TsaraPropagationError, match="correspond one to one"):
        propagate_random(
            np.ones(4), np.ones(4), tau_s=20.0, times_s=np.arange(3.0), form="ar1_double_sum"
        )


def test_double_sum_refuses_to_allocate_an_enormous_matrix() -> None:
    """Fail with a sentence, not a MemoryError.

    The limit is not arbitrary: the matrix is N by N in float64, so the point
    of the message is to name the linear-cost alternative rather than to let
    the allocation decide.
    """
    n = DOUBLE_SUM_MAX_POINTS + 1
    with pytest.raises(TsaraPropagationError, match="above the"):
        propagate_random(
            np.ones(n), np.ones(n), tau_s=20.0, times_s=np.arange(float(n)), form="ar1_double_sum"
        )


def test_masked_samples_drop_out_with_their_weights() -> None:
    """A NaN sigma is a QA/QC-masked sample by the time it reaches here.

    The remaining weights are renormalized, so the returned sigma describes
    the mean that was actually formed rather than the one that would have
    been formed had nothing been masked.
    """
    sigma = np.array([2.0, np.nan, 2.0, 2.0])
    result = propagate_random(sigma, np.ones(4))
    assert result.sigma == pytest.approx(2.0 / np.sqrt(3))
    assert result.n_effective == pytest.approx(3.0)


def test_zero_weight_samples_do_not_contribute() -> None:
    sigma = np.array([2.0, 100.0])
    result = propagate_random(sigma, np.array([1.0, 0.0]))
    assert result.sigma == pytest.approx(2.0)


def test_empty_overlap_gives_nan_not_zero() -> None:
    """The same answer binning gives the value, so the two never disagree.

    A zero here would be an uncertainty of zero on a value that does not
    exist, which is the most dangerous number a pipeline can emit.
    """
    result = propagate_random(np.array([2.0, 3.0]), np.zeros(2))
    assert np.isnan(result.sigma)
    assert result.n_effective == 0.0


def test_all_masked_gives_nan() -> None:
    result = propagate_random(np.array([np.nan, np.nan]), np.ones(2))
    assert np.isnan(result.sigma)
    assert result.n_effective == 0.0


def test_random_propagation_rejects_a_negative_sigma() -> None:
    with pytest.raises(TsaraPropagationError, match="sign error"):
        propagate_random(np.array([1.0, -1.0]), np.ones(2))


def test_random_propagation_rejects_a_negative_weight() -> None:
    with pytest.raises(TsaraPropagationError, match="subtract another"):
        propagate_random(np.ones(2), np.array([1.0, -1.0]))


def test_random_propagation_rejects_mismatched_lengths() -> None:
    with pytest.raises(TsaraPropagationError, match="correspond one to one"):
        propagate_random(np.ones(3), np.ones(2))


def test_random_propagation_rejects_a_nonpositive_declared_tau() -> None:
    with pytest.raises(TsaraPropagationError, match="pass None"):
        propagate_random(np.ones(3), np.ones(3), tau_s=0.0, times_s=np.arange(3.0))


def test_correlation_correction_refuses_to_invent_a_spacing() -> None:
    """The correction is most sensitive to the quantity it would be inventing.

    Declaring a timescale and withholding the timestamps asks TSARA to guess
    how far apart the samples are, which decides the whole correction.
    """
    with pytest.raises(TsaraPropagationError, match="how far apart"):
        propagate_random(np.ones(3), np.ones(3), tau_s=20.0)


def test_correlation_correction_needs_one_time_per_sigma() -> None:
    with pytest.raises(TsaraPropagationError, match="correspond one to one"):
        propagate_random(np.ones(3), np.ones(3), tau_s=20.0, times_s=np.arange(2.0))


def test_single_contributing_sample_needs_no_spacing() -> None:
    """One sample has no interval, and its effective size is one either way."""
    result = propagate_random(
        np.array([2.0, np.nan]), np.ones(2), tau_s=20.0, times_s=np.array([0.0, 1.0])
    )
    assert result.sigma == pytest.approx(2.0)
    assert result.n_effective == pytest.approx(1.0)


def test_simultaneous_samples_have_no_lag_to_correlate_over() -> None:
    with pytest.raises(TsaraPropagationError, match="share one timestamp"):
        propagate_random(np.ones(3), np.ones(3), tau_s=20.0, times_s=np.zeros(3))


def test_spacing_uses_the_median_so_one_gap_does_not_stretch_it() -> None:
    """A hole in the middle of a cell must not relax the whole correction.

    With a mean spacing the single 100 s gap would dominate and make the
    samples look nearly independent; the median keeps the correction set by
    the cadence the samples actually have.
    """
    times = np.array([0.0, 1.0, 2.0, 3.0, 103.0, 104.0, 105.0])
    sigma = np.full(7, 2.0)
    gapped = propagate_random(sigma, np.ones(7), tau_s=20.0, times_s=times)
    even = propagate_random(sigma, np.ones(7), tau_s=20.0, times_s=np.arange(7.0))
    assert gapped.n_effective == pytest.approx(even.n_effective)


def test_unsorted_times_give_the_same_spacing() -> None:
    """Spacing is a property of the set of samples, not of their order."""
    times = np.array([3.0, 0.0, 2.0, 1.0])
    sigma = np.full(4, 2.0)
    shuffled = propagate_random(sigma, np.ones(4), tau_s=20.0, times_s=times)
    ordered = propagate_random(sigma, np.ones(4), tau_s=20.0, times_s=np.arange(4.0))
    assert shuffled.n_effective == pytest.approx(ordered.n_effective)


def test_random_propagation_rejects_two_dimensional_sigmas() -> None:
    with pytest.raises(TsaraPropagationError, match="one-dimensional"):
        propagate_random(np.ones((2, 2)), np.ones(4))


# ---------------------------------------------------------------------------
# propagate_systematic
# ---------------------------------------------------------------------------


def test_systematic_error_does_not_average_down() -> None:
    """The headline consequence of METHODS §3.3, stated as a test.

    A hundred samples of a species whose calibration is uncertain by 2 % have
    a mean that is uncertain by 2 %, not by 0.2 %.
    """
    sigma = np.full(100, 0.02)
    assert propagate_systematic(sigma, np.ones(100)).sigma == pytest.approx(0.02)


def test_systematic_propagation_is_the_weighted_mean_of_the_sigmas() -> None:
    sigma = np.array([1.0, 3.0])
    weights = np.array([3.0, 1.0])
    assert propagate_systematic(sigma, weights).sigma == pytest.approx(0.75 * 1.0 + 0.25 * 3.0)


def test_the_two_components_move_in_opposite_directions() -> None:
    """The property that makes keeping them separate necessary."""
    sigma = np.full(50, 1.0)
    random = propagate_random(sigma, np.ones(50)).sigma
    systematic = propagate_systematic(sigma, np.ones(50)).sigma
    assert random < systematic
    assert systematic == pytest.approx(1.0)


def test_systematic_propagation_skips_masked_samples() -> None:
    sigma = np.array([1.0, np.nan, 3.0])
    assert propagate_systematic(sigma, np.ones(3)).sigma == pytest.approx(2.0)


def test_systematic_propagation_of_nothing_is_nan() -> None:
    assert np.isnan(propagate_systematic(np.array([1.0, 2.0]), np.zeros(2)).sigma)


def test_systematic_propagation_of_all_masked_is_nan() -> None:
    result = propagate_systematic(np.array([np.nan, np.nan]), np.ones(2))
    assert np.isnan(result.sigma)
    assert result.n_effective == 0.0


def test_systematic_propagation_rejects_a_negative_sigma() -> None:
    with pytest.raises(TsaraPropagationError, match="non-negative"):
        propagate_systematic(np.array([1.0, -2.0]), np.ones(2))


def test_systematic_propagation_reports_its_form() -> None:
    assert propagate_systematic(np.ones(3), np.ones(3)).form == "systematic"


# ---------------------------------------------------------------------------
# sigma_at_support
# ---------------------------------------------------------------------------


def test_same_width_returns_the_sigma_untouched() -> None:
    value, form = sigma_at_support(0.5, quoted_width_s=60.0, target_width_s=60.0)
    assert value == pytest.approx(0.5)
    assert form == "unchanged"


def test_no_timescale_leaves_the_sigma_alone_and_says_so() -> None:
    """The refusal ingestion makes, made again at the point of use.

    METHODS §10.8: with no timescale the naive root-N is not merely imprecise
    but confidently wrong, so the honest answer is the unscaled number plus a
    label saying it was not scaled.
    """
    value, form = sigma_at_support(0.5, quoted_width_s=1.0, target_width_s=60.0)
    assert value == pytest.approx(0.5)
    assert form == "unscaled"


def test_widening_shrinks_the_sigma_by_less_than_root_n() -> None:
    """The whole reason the naive shortcut is refused.

    A 1 s figure on 60 s cells with a 20 s decorrelation timescale is worth
    about 2.2 independent samples, not 60 -- so the sigma falls by 1.48, not
    by 7.75.
    """
    value, form = sigma_at_support(0.5, quoted_width_s=1.0, target_width_s=60.0, tau_s=20.0)
    naive = 0.5 / np.sqrt(60)
    assert form == "ar1_neff"
    assert value == pytest.approx(0.3375, abs=1e-4)
    assert value > naive * 2


def test_narrowing_grows_the_sigma_by_the_same_factor() -> None:
    """The algebra is symmetric, so the round trip returns the original."""
    wide, _ = sigma_at_support(0.5, quoted_width_s=1.0, target_width_s=60.0, tau_s=20.0)
    back, _ = sigma_at_support(wide, quoted_width_s=60.0, target_width_s=1.0, tau_s=20.0)
    assert back == pytest.approx(0.5)


def test_sigma_at_support_accepts_an_array() -> None:
    values, _ = sigma_at_support(
        np.array([0.5, 1.0]), quoted_width_s=1.0, target_width_s=4.0, tau_s=1e-9
    )
    assert values == pytest.approx(np.array([0.25, 0.5]))


def test_sigma_at_support_honours_the_asymptotic_form() -> None:
    value, form = sigma_at_support(
        1.0, quoted_width_s=1.0, target_width_s=200.0, tau_s=100.0, form="ar1_asymptotic"
    )
    assert form == "ar1_asymptotic"
    assert value == pytest.approx(1.0)


def test_a_fractional_count_is_rounded() -> None:
    """A canister's 14.7 s cell against a 1 s quote is 15 samples, not 14.7.

    The rounding is immaterial -- the correction goes as its square root,
    under a correlation model that is itself an approximation -- but it has
    to happen somewhere, and it happens here rather than in each caller.
    """
    rounded, _ = sigma_at_support(1.0, quoted_width_s=1.0, target_width_s=14.7, tau_s=1e-9)
    fifteen, _ = sigma_at_support(1.0, quoted_width_s=1.0, target_width_s=15.0, tau_s=1e-9)
    assert rounded == pytest.approx(fifteen)


@pytest.mark.parametrize(
    ("quoted", "target"), [(0.0, 60.0), (1.0, 0.0), (-1.0, 60.0), (1.0, -60.0)]
)
def test_sigma_at_support_rejects_nonpositive_widths(quoted: float, target: float) -> None:
    with pytest.raises(TsaraPropagationError, match="must be positive"):
        sigma_at_support(0.5, quoted_width_s=quoted, target_width_s=target)


# ---------------------------------------------------------------------------
# Monte Carlo: does the reported sigma match the scatter that actually occurs?
# ---------------------------------------------------------------------------
#
# Every test above this line checks that a formula was typed correctly. None of
# them can tell whether it is the *right* formula, because they compare algebra
# against algebra. These three compare against an experiment: draw many
# realizations of an error with known structure, average each one, and measure
# how much the averages actually scatter. If the propagated sigma is a
# different number from that scatter, the propagation is wrong no matter how
# cleanly it was derived.
#
# The comparison is statistical, so it needs a tolerance with a reason rather
# than a round number. The sample standard deviation of M draws has a relative
# standard error of about 1/sqrt(2M), so at M = 20000 the measurement itself is
# good to 0.5 % and a 3 % assertion is roughly six of its own standard errors.
# The seed is fixed, so a failure is a real disagreement and not a bad night.

MONTE_CARLO_DRAWS = 20_000
MONTE_CARLO_TOLERANCE = 0.03


def draw_ar1(n: int, rho: float, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``draws`` independent AR(1) series of length ``n``, unit variance.

    Written from the textbook recursion rather than borrowed from
    :mod:`tsara.synthetic.noise`, deliberately. Borrowing would test TSARA's
    propagation against TSARA's own generator, and if both carried the same
    misunderstanding of AR(1) they would agree with each other and be wrong
    together.
    """
    innovation = rng.standard_normal((draws, n))
    series = np.empty((draws, n))
    series[:, 0] = innovation[:, 0]
    scale = np.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        series[:, i] = rho * series[:, i - 1] + scale * innovation[:, i]
    return series


def test_monte_carlo_independent_errors() -> None:
    """White noise, sixty samples: the means really do scatter by sigma/sqrt(60)."""
    rng = np.random.default_rng(20260909)
    sigma_value, n = 2.0, 60
    draws = rng.standard_normal((MONTE_CARLO_DRAWS, n)) * sigma_value
    observed = float(draws.mean(axis=1).std(ddof=1))
    predicted = propagate_random(np.full(n, sigma_value), np.ones(n)).sigma
    assert observed == pytest.approx(predicted, rel=MONTE_CARLO_TOLERANCE)


def test_monte_carlo_correlated_errors() -> None:
    """AR(1) with a 20 s timescale at 1 Hz: the experiment picks the right form.

    This is the test that decides between the registered forms rather than
    merely describing them. The asymptotic form predicts a scatter 21 % larger
    than what actually occurs, so it is not within tolerance of the truth while
    the finite-N form is -- which is the empirical half of the argument in
    METHODS §3.4, the algebraic half being the brute-force comparison above.
    """
    rng = np.random.default_rng(20260910)
    sigma_value, n, tau = 2.0, 60, 20.0
    rho = lag1_correlation(1.0, tau)
    draws = draw_ar1(n, rho, MONTE_CARLO_DRAWS, rng) * sigma_value
    observed = float(draws.mean(axis=1).std(ddof=1))
    times = np.arange(float(n))
    exact = propagate_random(np.full(n, sigma_value), np.ones(n), tau_s=tau, times_s=times)
    asymptotic = propagate_random(
        np.full(n, sigma_value), np.ones(n), tau_s=tau, times_s=times, form="ar1_asymptotic"
    )
    assert observed == pytest.approx(exact.sigma, rel=MONTE_CARLO_TOLERANCE)
    assert observed != pytest.approx(asymptotic.sigma, rel=MONTE_CARLO_TOLERANCE)


def test_monte_carlo_systematic_errors_do_not_average_down() -> None:
    """The claim of METHODS §3.3, put to an experiment.

    A systematic error is one offset shared by every sample in a realization,
    which is what makes it a rank-1 draw in the generator. Averaging a hundred
    samples of it leaves the offset exactly where it was, so the means scatter
    by the full sigma -- not by a hundredth of it, and not by a tenth.
    """
    rng = np.random.default_rng(20260911)
    sigma_value, n = 0.02, 100
    offsets = rng.standard_normal(MONTE_CARLO_DRAWS) * sigma_value
    draws = np.repeat(offsets[:, None], n, axis=1)
    observed = float(draws.mean(axis=1).std(ddof=1))
    predicted = propagate_systematic(np.full(n, sigma_value), np.ones(n)).sigma
    assert observed == pytest.approx(predicted, rel=MONTE_CARLO_TOLERANCE)
    # And the number the naive rule would have given, for contrast.
    assert observed > 5 * (sigma_value / np.sqrt(n))


# ---------------------------------------------------------------------------
# A case small enough to check on paper
# ---------------------------------------------------------------------------


def test_worked_example_checkable_by_hand() -> None:
    """Two samples, weights 3 and 1, sigmas 4 and 2. No computer required.

    Normalized weights are 0.75 and 0.25.

        random      sqrt(0.75^2 * 4^2 + 0.25^2 * 2^2) = sqrt(9.25) = 3.0414
        systematic  0.75 * 4 + 0.25 * 2                              = 3.5

    Kept because a test whose expected value is itself computed by the library
    proves consistency and nothing else. This one can be checked by a reader
    with a pencil, which is a different and stronger kind of assurance.
    """
    sigmas, weights = np.array([4.0, 2.0]), np.array([3.0, 1.0])
    assert propagate_random(sigmas, weights).sigma == pytest.approx(np.sqrt(9.25))
    assert propagate_random(sigmas, weights).sigma == pytest.approx(3.041381, abs=1e-6)
    assert propagate_systematic(sigmas, weights).sigma == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# The vectorized form, bound to the scalar one it was derived from
# ---------------------------------------------------------------------------
#
# Two implementations of one piece of arithmetic is the risk this module was
# written to avoid, so the binned form is not tested on its own merits. It is
# tested by being made to agree with the scalar form, which is the one the
# Monte Carlo checks above measured against an experiment. That keeps the
# chain unbroken: observed scatter -> scalar form -> binned form.


def cell_slices(counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return long-form target indices and the slice boundaries they imply."""
    index = np.concatenate([np.full(n, c, dtype=np.int64) for c, n in enumerate(counts)])
    edges = np.cumsum([0, *counts])
    return index, edges


@pytest.mark.parametrize("tau", [None, 20.0])
def test_binned_random_agrees_with_the_scalar_form(tau: float | None) -> None:
    rng = np.random.default_rng(20260916)
    counts = [4, 7, 1, 12]
    index, edges = cell_slices(counts)
    sigma = rng.uniform(0.5, 3.0, size=index.size)
    weights = rng.uniform(0.1, 2.0, size=index.size)
    times = np.arange(float(index.size))
    binned = propagate_random_binned(sigma, weights, index, len(counts), spacing_s=1.0, tau_s=tau)
    for cell in range(len(counts)):
        lo, hi = edges[cell], edges[cell + 1]
        scalar = propagate_random(sigma[lo:hi], weights[lo:hi], tau_s=tau, times_s=times[lo:hi])
        assert binned.sigma[cell] == pytest.approx(scalar.sigma, rel=1e-12)
        assert binned.n_effective[cell] == pytest.approx(scalar.n_effective, rel=1e-12)


def test_binned_systematic_agrees_with_the_scalar_form() -> None:
    rng = np.random.default_rng(20260917)
    counts = [3, 5, 2]
    index, edges = cell_slices(counts)
    sigma = rng.uniform(0.5, 3.0, size=index.size)
    weights = rng.uniform(0.1, 2.0, size=index.size)
    binned = propagate_systematic_binned(sigma, weights, index, len(counts))
    for cell in range(len(counts)):
        lo, hi = edges[cell], edges[cell + 1]
        scalar = propagate_systematic(sigma[lo:hi], weights[lo:hi])
        assert binned.sigma[cell] == pytest.approx(scalar.sigma, rel=1e-12)


def test_binned_forms_leave_an_empty_cell_as_nan() -> None:
    index = np.array([0, 0, 2, 2], dtype=np.int64)
    sigma, weights = np.full(4, 1.0), np.ones(4)
    for result in (
        propagate_random_binned(sigma, weights, index, 3),
        propagate_systematic_binned(sigma, weights, index, 3),
    ):
        assert np.isnan(result.sigma[1])
        assert result.n_effective[1] == 0.0
        assert np.isfinite(result.sigma[0]) and np.isfinite(result.sigma[2])


def test_binned_random_with_nothing_contributing_anywhere() -> None:
    result = propagate_random_binned(np.full(3, np.nan), np.ones(3), np.zeros(3, dtype=np.int64), 2)
    assert np.all(np.isnan(result.sigma))
    assert result.form == INDEPENDENT_FORM


def test_binned_random_needs_a_spacing_when_a_timescale_is_declared() -> None:
    with pytest.raises(TsaraPropagationError, match="how far apart"):
        propagate_random_binned(np.ones(3), np.ones(3), np.zeros(3, dtype=np.int64), 1, tau_s=20.0)


def test_binned_random_rejects_a_nonpositive_timescale() -> None:
    with pytest.raises(TsaraPropagationError, match="pass None"):
        propagate_random_binned(
            np.ones(3), np.ones(3), np.zeros(3, dtype=np.int64), 1, tau_s=0.0, spacing_s=1.0
        )


def test_binned_forms_reject_mismatched_long_form_arrays() -> None:
    with pytest.raises(TsaraPropagationError, match="correspond one to one"):
        propagate_random_binned(np.ones(3), np.ones(2), np.zeros(3, dtype=np.int64), 1)


def test_binned_forms_reject_negative_weights_and_sigmas() -> None:
    index = np.zeros(2, dtype=np.int64)
    with pytest.raises(TsaraPropagationError, match="non-negative"):
        propagate_random_binned(np.ones(2), np.array([1.0, -1.0]), index, 1)
    with pytest.raises(TsaraPropagationError, match="non-negative"):
        propagate_random_binned(np.array([1.0, -1.0]), np.ones(2), index, 1)


# ---------------------------------------------------------------------------
# The pairwise form through the binned interface
# ---------------------------------------------------------------------------
#
# It is reachable there because a registered name that cannot be chosen where
# forms are chosen is not really registered. Under the hood it loops cells and
# calls the scalar form, so the reference implementation is literally what
# runs -- which is also how `propagate_random` came to have a caller.


def test_binned_double_sum_agrees_with_the_scalar_form_per_cell() -> None:
    rng = np.random.default_rng(20260918)
    counts = [5, 9, 4]
    index, edges = cell_slices(counts)
    sigma = rng.uniform(0.5, 3.0, size=index.size)
    weights = rng.uniform(0.1, 2.0, size=index.size)
    times = np.arange(float(index.size))
    binned = propagate_random_binned(
        sigma, weights, index, len(counts), times_s=times, tau_s=15.0, form="ar1_double_sum"
    )
    assert binned.form == "ar1_double_sum"
    for cell in range(len(counts)):
        lo, hi = edges[cell], edges[cell + 1]
        scalar = propagate_random(
            sigma[lo:hi], weights[lo:hi], tau_s=15.0, times_s=times[lo:hi], form="ar1_double_sum"
        )
        assert binned.sigma[cell] == pytest.approx(scalar.sigma, rel=1e-12)


def test_binned_double_sum_leaves_an_empty_cell_alone() -> None:
    index = np.array([0, 0, 2, 2], dtype=np.int64)
    result = propagate_random_binned(
        np.ones(4),
        np.ones(4),
        index,
        3,
        times_s=np.arange(4.0),
        tau_s=10.0,
        form="ar1_double_sum",
    )
    assert np.isnan(result.sigma[1])
    assert np.isfinite(result.sigma[0]) and np.isfinite(result.sigma[2])


def test_binned_double_sum_needs_times() -> None:
    with pytest.raises(TsaraPropagationError, match="needs times_s"):
        propagate_random_binned(
            np.ones(3),
            np.ones(3),
            np.zeros(3, dtype=np.int64),
            1,
            tau_s=10.0,
            form="ar1_double_sum",
        )


def test_binned_double_sum_needs_matching_long_form_arrays() -> None:
    with pytest.raises(TsaraPropagationError, match="correspond one to one"):
        propagate_random_binned(
            np.ones(3),
            np.ones(3),
            np.zeros(3, dtype=np.int64),
            1,
            times_s=np.arange(2.0),
            tau_s=10.0,
            form="ar1_double_sum",
        )
