r"""Propagating measurement uncertainty when values are combined.

Why this is a module and not four copies
-----------------------------------------
Three later stages all need the same question answered — *if I average these
values, what is the uncertainty of the answer?* — and they need it answered
identically:

* alignment bins a fast stream onto a slow stream's cells (``docs/METHODS.md``
  §1.3) and onto the output grid (§1.4);
* baselines roll a time window over cells (§5);
* regression weights each point by its uncertainty (§4.3).

Phase 3.5 removed the one place this arithmetic had started to grow — a copy
inside ingestion that moved a declared sigma onto its cells — for reasons
recorded in §10.8, and the decision came with an obligation: when the
arithmetic is finally needed it exists **once**. This module is that once.

The two components never mix
-----------------------------
TSARA carries a two-component budget for every variable (§2.1), and the two
behave in opposite ways under averaging:

* **random** errors are uncorrelated point to point, so averaging shrinks
  them — by :math:`\sqrt{N}` when the samples really are independent, and by
  less than that when they are not (§3.2, §3.4);
* **systematic** errors are correlated by definition, so averaging does not
  shrink them at all: the uncertainty of the mean is the weighted mean of the
  uncertainties (§3.3).

Collapsing them into one number before averaging would silently apply one
rule to both, which is why they stay separate through the entire pipeline and
why this module offers two functions rather than one with a flag.

Weights come from the operation, not from this module
------------------------------------------------------
A tempting mistake: §3.2 notes that inverse-variance weights are the
minimum-variance choice, so it looks as though a propagation routine should
use them. It must not. The *value* being reported was formed with
overlap weights, because an overlap-weighted mean is what "the average of
this stream over that interval" means. An uncertainty computed with different
weights would not describe the number it is attached to. So every function
here takes the caller's weights and uses exactly those.

Correlated random error, and the honest default
------------------------------------------------
A random component may declare a decorrelation timescale τ (§2.2), and where
it does, this module models the error autocorrelation as
:math:`\rho(\Delta t) = e^{-|\Delta t|/\tau}` and reduces the sample count
accordingly. Where it does not, samples are treated as independent — and that
is a *declaration*, not an assumption TSARA invents: a component declared
random is by definition uncorrelated point to point, and τ is the refinement
that says "less so than that". Every returned uncertainty carries the name of
the form used, so the difference is legible in the product rather than
implied by its absence.

Three AR(1) forms are registered because their disagreement is measurable
and was an open question when this module was written; see
``docs/METHODS.md`` §3.4 for which one is right where, and for the
ground-truth measurement that settled it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from tsara.core.exceptions import TsaraError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

__all__ = [
    "BinnedSigma",
    "PROPAGATION_FORMS",
    "PropagatedSigma",
    "PropagationForm",
    "TsaraPropagationError",
    "kish_sample_size",
    "lag1_correlation",
    "n_effective",
    "propagate_random",
    "propagate_random_binned",
    "propagate_systematic",
    "propagate_systematic_binned",
    "sigma_at_support",
]


class TsaraPropagationError(TsaraError):
    """Raised when an uncertainty cannot be propagated as asked.

    Separate from a config error and from a support error because the causes
    are different in kind: a negative sigma, weights that do not correspond to
    the values they weight, or a request for the exact double sum on more
    points than it can hold in memory.
    """


#: How a correlated random component is reduced under averaging.
#:
#: All three assume the same AR(1)-like error autocorrelation
#: ``rho(dt) = exp(-|dt|/tau)``; they differ only in how faithfully they
#: evaluate the resulting double sum, and therefore in cost.
#:
#: * ``ar1_neff`` -- the finite-*N* effective sample size, evaluated exactly
#:   for equally spaced samples and carried to unequal weights through their
#:   Kish sample size. Linear in *N*, and the default.
#: * ``ar1_asymptotic`` -- the large-*N* limit ``N (1 - rho1) / (1 + rho1)``
#:   written in ``docs/METHODS.md`` §3.4. Constant cost, and wrong by a factor
#:   approaching two when a window holds only a few correlation times.
#: * ``ar1_double_sum`` -- the double sum of §3.1 evaluated pair by pair on
#:   the real timestamps. Assumption-free given AR(1): it needs neither equal
#:   spacing nor equal sigma. Quadratic in *N*, so it exists as the reference
#:   the other two are scored against rather than as a production default.
PropagationForm = Literal["ar1_neff", "ar1_asymptotic", "ar1_double_sum"]

#: The registered forms, in the order they appear in ``docs/METHODS.md`` §3.4.
#:
#: Derived from the type rather than retyped, so a form cannot be added to one
#: and forgotten in the other.
PROPAGATION_FORMS: tuple[PropagationForm, ...] = get_args(PropagationForm)

#: Provenance recorded when a random component was reduced without knowing τ.
#:
#: Not one of :data:`PROPAGATION_FORMS` because it is not a choice a caller
#: makes: it is what happens when the manifest declared no decorrelation
#: timescale, and the samples are therefore independent by declaration.
INDEPENDENT_FORM = "independent"

#: Largest number of points the pairwise double sum will build a matrix for.
#:
#: ``ar1_double_sum`` allocates an ``N x N`` float64 array, so 20 000 points is
#: 3.2 GB and 5 000 is 200 MB. The limit exists so that an accidental call on
#: a whole campaign fails with a sentence naming the alternative rather than
#: with a MemoryError naming nothing.
DOUBLE_SUM_MAX_POINTS = 5_000

#: Below this the geometric terms of the finite-*N* sum stop contributing to a
#: float64 total, so summing further adds cost and no accuracy.
_GEOMETRIC_FLOOR = 1e-18


@dataclass(frozen=True)
class PropagatedSigma:
    """One propagated uncertainty, with the name of the form that produced it.

    The name travels with the number for the same reason
    :class:`~tsara.ingest.uncertainty.ResolvedUncertainty` carries a provenance
    label: a sigma reduced by ``sqrt(N)`` and a sigma not reduced at all look
    identical once written to a file, and mean opposite things.

    Attributes
    ----------
    sigma : float
        The propagated 1-sigma uncertainty of the weighted mean.
    form : str
        Which registered form produced it, or ``'independent'`` when no
        decorrelation timescale was declared and the samples were therefore
        treated as uncorrelated (which is what declaring them random means).
    n_effective : float
        The effective sample size the reduction used. Equal to the Kish
        sample size of the weights when the samples are independent, and
        smaller when they are correlated. Reported because it is the number a
        reader needs to judge the uncertainty, and it cannot be recovered
        from the sigma alone.
    """

    sigma: float
    form: str
    n_effective: float


@dataclass(frozen=True)
class BinnedSigma:
    """Propagated uncertainties for many cells at once.

    The vectorized counterpart of :class:`PropagatedSigma`. Binning a stream
    onto a day of one-second grid cells is 86 400 propagations, and calling
    the scalar form in a Python loop for each of them costs seconds per
    species per sweep point.

    Attributes
    ----------
    sigma : numpy.ndarray
        Propagated 1-sigma per target cell, ``nan`` where nothing contributed.
    form : str
        The registered form used, or ``'independent'`` / ``'systematic'``.
        One label for the whole call, because one call uses one form.
    n_effective : numpy.ndarray
        Effective sample size per cell.
    """

    sigma: npt.NDArray[np.float64]
    form: str
    n_effective: npt.NDArray[np.float64]


def kish_sample_size(weights: npt.ArrayLike) -> float:
    r"""Return the effective number of samples a set of weights represents.

    Kish's formula, :math:`(\sum w)^2 / \sum w^2`. It answers "how many
    equally weighted samples would carry as much information as these
    unequally weighted ones", and it is the bridge that lets the
    equal-weight effective sample size of :func:`n_effective` be applied to
    the overlap weights that binning actually produces.

    Equal weights give exactly *N*; one dominant weight gives nearly 1, which
    is the property that makes it the right *N* to feed a correlation
    correction — a cell whose average is really one sample should not be
    credited with the correlation structure of many.

    Parameters
    ----------
    weights : array_like
        Non-negative weights, not necessarily normalized.

    Returns
    -------
    float
        Effective sample size, in ``[0, len(weights)]``. Zero when every
        weight is zero.

    Raises
    ------
    TsaraPropagationError
        If any weight is negative.
    """
    w = _as_1d(weights, name="weights")
    if np.any(w < 0):
        raise TsaraPropagationError("Weights must be non-negative.")
    sum_squares = float(np.sum(w * w))
    if sum_squares <= 0:
        return 0.0
    return float(w.sum() ** 2 / sum_squares)


def lag1_correlation(spacing_s: float, tau_s: float) -> float:
    r"""Return the AR(1) lag-1 error correlation for a sampling interval.

    :math:`\rho_1 = e^{-\Delta t / \tau}` (``docs/METHODS.md`` §3.4).

    Parameters
    ----------
    spacing_s : float
        Interval between consecutive samples, in seconds. Must be positive.
    tau_s : float
        Decorrelation timescale, in seconds. Must be positive.

    Returns
    -------
    float
        Lag-1 correlation in ``(0, 1)``.

    Raises
    ------
    TsaraPropagationError
        If either argument is not strictly positive. Zero is rejected rather
        than special-cased: a zero spacing means two samples at the same
        instant, and a zero timescale is a white-noise declaration that
        should be spelled by omitting τ, not by setting it to zero.
    """
    if not spacing_s > 0:
        raise TsaraPropagationError(
            f"Sample spacing must be positive, got {spacing_s} s. Two samples at "
            "the same instant have no lag to correlate over."
        )
    if not tau_s > 0:
        raise TsaraPropagationError(
            f"Decorrelation timescale must be positive, got {tau_s} s. Declare no "
            "timescale rather than a zero one; an undeclared timescale already "
            "means uncorrelated."
        )
    return float(np.exp(-spacing_s / tau_s))


def n_effective(
    n: npt.ArrayLike,
    spacing_s: float,
    tau_s: float,
    *,
    form: PropagationForm = "ar1_neff",
) -> npt.NDArray[np.float64]:
    r"""Return the effective sample size of *n* correlated samples.

    The number that replaces *N* in :math:`\sigma/\sqrt{N}` when consecutive
    errors are correlated (``docs/METHODS.md`` §3.4). Averaging 60 samples
    whose errors decorrelate over 20 s is not worth 60 independent
    measurements, and using 60 would understate the uncertainty of every
    binned value, every baseline and every fit weight downstream.

    Parameters
    ----------
    n : array_like
        Sample count, or an array of counts. Values below 1 are clipped to 1;
        a cell holding one sample cannot hold less than one sample's worth of
        information.
    spacing_s : float
        Interval between consecutive samples, in seconds.
    tau_s : float
        Decorrelation timescale, in seconds.
    form : {'ar1_neff', 'ar1_asymptotic'}, optional
        Which registered form to evaluate. ``'ar1_double_sum'`` is not
        accepted here: it needs the actual timestamps and per-point sigmas,
        so it is reached through :func:`propagate_random` instead.

    Returns
    -------
    numpy.ndarray
        Effective sample sizes, each in ``[1, n]``, with the same shape as
        ``n``. Always an array, including for scalar input, so that callers
        need only one code path.

    Raises
    ------
    TsaraPropagationError
        If ``form`` is not a form this function can evaluate, or if the
        spacing or timescale is not positive.

    Notes
    -----
    Distinct values of ``n`` are evaluated once each and broadcast back,
    because a uniform grid produces long runs of identical counts and the
    finite-*N* sum is the expensive part.
    """
    if form == "ar1_double_sum":
        raise TsaraPropagationError(
            "n_effective cannot evaluate 'ar1_double_sum': the pairwise form needs "
            "the actual timestamps and per-point sigmas, not a count. Call "
            "propagate_random with times_s instead."
        )
    if form not in PROPAGATION_FORMS:
        raise TsaraPropagationError(
            f"Unknown propagation form {form!r}; registered forms are {list(PROPAGATION_FORMS)}."
        )
    # rho: how alike two consecutive errors are, exp(-spacing / tau).
    rho = lag1_correlation(spacing_s, tau_s)
    counts = np.atleast_1d(np.asarray(n, dtype=np.float64))
    counts = np.clip(counts, 1.0, None)
    if form == "ar1_asymptotic":
        # METHODS §3.4's large-N limit. Kept because it is the form the
        # document specified before the finite-N version existed, and because
        # the size of its error is worth being able to reproduce.
        effective = counts * (1.0 - rho) / (1.0 + rho)
        return np.asarray(np.clip(effective, 1.0, counts), dtype=np.float64)
    # ar1_neff: evaluate Var(mean) / sigma^2 exactly for N equally spaced readings.
    # N_eff is its reciprocal, because Var(mean) = sigma^2 / N_eff by definition.
    out = np.empty_like(counts)
    for value in np.unique(counts):
        ratio = _variance_ratio_equal_weight(int(round(float(value))), rho)
        out[counts == value] = 1.0 / ratio
    return np.asarray(np.clip(out, 1.0, counts), dtype=np.float64)


def propagate_random(
    sigmas: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    tau_s: float | None = None,
    times_s: npt.ArrayLike | None = None,
    form: PropagationForm = "ar1_neff",
) -> PropagatedSigma:
    r"""Propagate the random component through a weighted mean.

    With independent errors this is §3.2, :math:`\sum_i w_i^2 \sigma_i^2`
    for weights summing to one. With a declared decorrelation timescale the
    result is inflated by the ratio of the sample count to the effective
    sample count (§3.4), or evaluated pair by pair when
    ``form='ar1_double_sum'``.

    Parameters
    ----------
    sigmas : array_like
        Per-point random 1-sigma values. ``nan`` entries are excluded along
        with their weights, which is what a QA/QC-masked sample looks like by
        the time it reaches here.
    weights : array_like
        Non-negative weights, one per sigma, in whatever scale the caller
        used to form the value — overlap in nanoseconds, for instance. They
        are normalized internally, so the caller need not.
    tau_s : float, optional
        Decorrelation timescale in seconds. ``None`` means the manifest
        declared none, and the samples are treated as independent.
    times_s : array_like, optional
        Sample times in seconds, required only by ``'ar1_double_sum'``.
    form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form to use when ``tau_s`` is given. Ignored when it
        is not: with no timescale there is no correlation to model.

    Returns
    -------
    PropagatedSigma
        The propagated sigma, the form used, and the effective sample size.
        ``sigma`` is ``nan`` when nothing contributed — no finite sigma, or
        no weight — which is the same answer binning gives for the value
        itself, so a NaN uncertainty never accompanies a real number.

    Raises
    ------
    TsaraPropagationError
        If the arrays disagree in length, a sigma is negative, a weight is
        negative, or ``'ar1_double_sum'`` is requested without ``times_s``.

    Notes
    -----
    For unequal weights the effective sample size fed to the correlation
    correction is the Kish sample size of the weights
    (:func:`kish_sample_size`), which is exact for equal weights and an
    approximation otherwise. ``'ar1_double_sum'`` makes no such substitution
    and is the way to check what it costs.
    """
    # Validate: a negative sigma is a sign error, never a small uncertainty.
    sigma = _as_1d(sigmas, name="sigmas")
    if np.any(sigma[np.isfinite(sigma)] < 0):
        raise TsaraPropagationError(
            "Uncertainties must be non-negative; a negative sigma is a sign error, "
            "not a small uncertainty."
        )
    normalized = _normalized_weights(weights, expected=sigma.size)
    if normalized is None:
        return PropagatedSigma(sigma=float("nan"), form=INDEPENDENT_FORM, n_effective=0.0)
    # A masked sample contributes neither its value nor its uncertainty. Its
    # weight is dropped with it and the rest renormalized, so the returned
    # sigma describes the mean that was actually formed.
    finite = np.isfinite(sigma) & (normalized > 0)
    if not finite.any():
        return PropagatedSigma(sigma=float("nan"), form=INDEPENDENT_FORM, n_effective=0.0)
    sigma = sigma[finite]
    w = normalized[finite]
    w = w / w.sum()

    # With weights summing to one, the variance of the mean of independent errors
    # is sum(w^2 sigma^2) (METHODS §3.2).
    independent_variance = float(np.sum(w * w * sigma * sigma))
    # How many equally weighted readings these weights are worth (Kish).
    n_kish = kish_sample_size(w)
    if tau_s is None:
        # No timescale declared: the readings are independent by declaration.
        return PropagatedSigma(
            sigma=float(np.sqrt(independent_variance)),
            form=INDEPENDENT_FORM,
            n_effective=n_kish,
        )
    if not tau_s > 0:
        raise TsaraPropagationError(
            f"Decorrelation timescale must be positive, got {tau_s} s; pass None to "
            "declare the samples uncorrelated."
        )

    if form == "ar1_double_sum":
        # The reference form: every pair of readings correlated by the time between them.
        if times_s is None:
            raise TsaraPropagationError(
                "'ar1_double_sum' needs times_s: it correlates each pair by the time "
                "between them, which a count cannot supply."
            )
        times = _as_1d(times_s, name="times_s")
        if times.size != finite.size:
            raise TsaraPropagationError(
                f"Got {times.size} time(s) for {finite.size} sigma(s); they must "
                "correspond one to one."
            )
        variance = _double_sum_variance(sigma, w, times[finite], tau_s)
        # The effective sample size implied by the answer, so the three forms
        # report the same quantity and can be compared directly.
        implied = independent_variance * n_kish / variance if variance > 0 else float("nan")
        return PropagatedSigma(
            sigma=float(np.sqrt(variance)), form=form, n_effective=float(implied)
        )

    # The effective-sample-size forms: shrink the Kish count for correlation,
    # then inflate the independent variance by how much was lost.
    spacing = _mean_spacing(times_s, count=int(finite.sum()), mask=finite)
    effective = float(n_effective(n_kish, spacing, tau_s, form=form)[0])
    inflation = n_kish / effective if effective > 0 else float("nan")
    return PropagatedSigma(
        sigma=float(np.sqrt(independent_variance * inflation)),
        form=form,
        n_effective=effective,
    )


def propagate_systematic(
    sigmas: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> PropagatedSigma:
    r"""Propagate the systematic component through a weighted mean.

    :math:`\sigma_{\bar{x}} = \sum_i w_i \sigma_i` for weights summing to
    one (``docs/METHODS.md`` §3.3). **No reduction with sample count**, and
    that is not conservatism: it is the exact evaluation of the §3.1 double
    sum when every pair is perfectly correlated, which is what declaring a
    component systematic asserts.

    The practical consequence is worth stating plainly, because it is the
    single most common error in reported enhancement ratios: averaging a
    hundred samples of a species whose calibration is 2 % uncertain leaves the
    mean 2 % uncertain, not 0.2 %.

    Parameters
    ----------
    sigmas : array_like
        Per-point systematic 1-sigma values; ``nan`` entries are excluded
        with their weights.
    weights : array_like
        Non-negative weights, one per sigma, normalized internally.

    Returns
    -------
    PropagatedSigma
        The propagated sigma, with form ``'systematic'`` and the Kish sample
        size of the contributing weights — reported for symmetry with the
        random component and as a check on which samples contributed, never
        as a divisor.

    Raises
    ------
    TsaraPropagationError
        If the arrays disagree in length, or a sigma or weight is negative.
    """
    sigma = _as_1d(sigmas, name="sigmas")
    if np.any(sigma[np.isfinite(sigma)] < 0):
        raise TsaraPropagationError("Uncertainties must be non-negative.")
    normalized = _normalized_weights(weights, expected=sigma.size)
    if normalized is None:
        return PropagatedSigma(sigma=float("nan"), form="systematic", n_effective=0.0)
    finite = np.isfinite(sigma) & (normalized > 0)
    if not finite.any():
        return PropagatedSigma(sigma=float("nan"), form="systematic", n_effective=0.0)
    # Renormalize over the readings that hold a sigma, then take the weighted mean
    # of the sigmas: fully correlated errors do not average down (METHODS §3.3).
    w = normalized[finite]
    w = w / w.sum()
    return PropagatedSigma(
        sigma=float(np.sum(w * sigma[finite])),
        form="systematic",
        n_effective=kish_sample_size(w),
    )


def sigma_at_support(
    sigma: npt.ArrayLike,
    *,
    quoted_width_s: float,
    target_width_s: float,
    tau_s: float | None = None,
    form: PropagationForm = "ar1_neff",
) -> tuple[npt.NDArray[np.float64], str]:
    r"""Move a declared random sigma from one averaging interval to another.

    An instrument's precision is quoted at some interval — "0.5 ppb at 1 s" —
    and the data may be delivered on cells of a different width. Comparing the
    two needs the sigma restated at the width in hand, and restating it needs
    a model: how quickly the errors decorrelate. That is the whole argument
    for why this lives here and not in ingestion (``docs/METHODS.md`` §10.8).
    Ingestion records what was declared and performs no arithmetic on it; the
    stage that wants a sigma at a particular support does the arithmetic, at
    the point of use, once.

    The model is that the wider value is the mean of
    ``target_width_s / quoted_width_s`` values of the quoted width. That count
    is rounded to a whole number, since a fractional sample has no meaning in
    the effective-sample-size forms, and the rounding is immaterial: the
    correction goes as its square root, under a correlation model that is
    itself an approximation.

    Parameters
    ----------
    sigma : array_like
        The declared 1-sigma value(s), at ``quoted_width_s``.
    quoted_width_s : float
        The interval the figure was quoted at, in seconds.
    target_width_s : float
        The interval it is wanted at, in seconds. May be wider or narrower;
        the algebra is symmetric, and a narrower target means a *larger*
        sigma.
    tau_s : float, optional
        Decorrelation timescale, in seconds. ``None`` returns the sigma
        unchanged with provenance ``'unscaled'`` — the same refusal ingestion
        makes, for the same reason: with no timescale the naive
        :math:`\sqrt{N}` is not merely imprecise but confidently wrong, and a
        recorded non-answer beats a confident wrong one.
    form : {'ar1_neff', 'ar1_asymptotic'}, optional
        Which registered form supplies the effective sample size.

    Returns
    -------
    numpy.ndarray
        The sigma at the target width.
    str
        Provenance: the form used, ``'unchanged'`` when the two widths are
        equal, or ``'unscaled'`` when no timescale was declared.

    Raises
    ------
    TsaraPropagationError
        If either width is not strictly positive.
    """
    values = np.asarray(sigma, dtype=np.float64)
    if not quoted_width_s > 0 or not target_width_s > 0:
        raise TsaraPropagationError(
            f"Both widths must be positive, got quoted {quoted_width_s} s and "
            f"target {target_width_s} s."
        )
    if quoted_width_s == target_width_s:
        return values, "unchanged"  # already quoted at the width wanted
    if tau_s is None:
        return values, "unscaled"  # moving it needs a timescale; refuse rather than guess
    if target_width_s > quoted_width_s:
        # Wider target: the value there is the mean of `count` quoted-width values.
        count = max(int(round(target_width_s / quoted_width_s)), 1)
        effective = float(n_effective(count, quoted_width_s, tau_s, form=form)[0])
        return np.asarray(values / np.sqrt(effective), dtype=np.float64), form
    # Narrower target: invert the reduction that would take the fine sigma to
    # the coarse one. Same model, same correction, opposite direction.
    count = max(int(round(quoted_width_s / target_width_s)), 1)
    effective = float(n_effective(count, target_width_s, tau_s, form=form)[0])
    return np.asarray(values * np.sqrt(effective), dtype=np.float64), form


def propagate_random_binned(
    sigmas: npt.ArrayLike,
    weights: npt.ArrayLike,
    target_index: npt.ArrayLike,
    n_target: int,
    *,
    spacing_s: float | None = None,
    times_s: npt.ArrayLike | None = None,
    tau_s: float | None = None,
    form: PropagationForm = "ar1_neff",
) -> BinnedSigma:
    """Propagate the random component onto many cells at once.

    The same arithmetic as :func:`propagate_random`, evaluated with
    ``bincount`` instead of one call per cell; a test binds the two together
    so they cannot drift. The same arithmetic in a different *order*, though,
    not the same floating-point operations: this form sums the raw weights and
    normalizes at the end, where the scalar form normalizes first. Measured
    over 200 random cells the two agree to within about six float64 epsilons,
    which is why the test that binds them compares approximately.

    One difference is deliberate. The scalar form derives the sample spacing
    from the timestamps it is given, per call. This one takes a single
    ``spacing_s`` for the whole operation -- normally the input stream's
    nominal cadence -- because a per-cell median would have to be computed
    group by group, which is the Python loop this function exists to avoid.
    The two agree exactly on evenly sampled data, and where a cell contains a
    gap the effective sample size is still reduced through the Kish weights,
    which count the samples that are actually there.

    ``'ar1_double_sum'`` is the exception: it correlates every pair of samples
    by the time between them, which no per-cell summary can supply, so it is
    evaluated cell by cell through :func:`propagate_random`. That is slow and
    is meant to be -- it is the reference the cheap forms are scored against,
    not a production default -- but it is *selectable*, because a registered
    name that cannot be chosen where forms are chosen is not registered.

    Parameters
    ----------
    sigmas, weights, target_index : array_like
        Long form: one entry per (target cell, reading) pair. ``nan``
        sigmas and zero weights are excluded together.
    n_target : int
        Number of target cells, which sets the output length.
    spacing_s : float, optional
        Sample spacing in seconds, required when ``tau_s`` is given for one
        of the effective-sample-size forms.
    times_s : array_like, optional
        Long-form sample times in seconds, required by ``'ar1_double_sum'``.
    tau_s : float, optional
        Decorrelation timescale. ``None`` treats the samples as independent.
    form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form to use.

    Returns
    -------
    BinnedSigma
        Per-cell sigma, the form used, and per-cell effective sample size.

    Raises
    ------
    TsaraPropagationError
        If the long-form arrays disagree, a value is negative, or the form's
        required companion argument is missing.
    """
    if tau_s is not None and form == "ar1_double_sum":
        return _double_sum_binned(sigmas, weights, target_index, n_target, times_s, tau_s)
    total, total_sq, _, weighted_var = _binned_totals(sigmas, weights, target_index, n_target)
    sigma = np.full(n_target, np.nan, dtype=np.float64)
    effective = np.zeros(n_target, dtype=np.float64)
    filled = total > 0  # cells with at least one contributing sigma
    if not filled.any():
        return BinnedSigma(sigma=sigma, form=INDEPENDENT_FORM, n_effective=effective)
    # Per cell: sum(w^2 sigma^2) / (sum w)^2, the independent-error variance of the
    # overlap-weighted mean, and the Kish count (sum w)^2 / sum w^2.
    independent_variance = weighted_var[filled] / (total[filled] ** 2)
    kish = total[filled] ** 2 / total_sq[filled]
    effective[filled] = kish
    if tau_s is None:
        sigma[filled] = np.sqrt(independent_variance)
        return BinnedSigma(sigma=sigma, form=INDEPENDENT_FORM, n_effective=effective)
    if not tau_s > 0:
        raise TsaraPropagationError(
            f"Decorrelation timescale must be positive, got {tau_s} s; pass None to "
            "declare the samples uncorrelated."
        )
    if spacing_s is None:
        raise TsaraPropagationError(
            "A decorrelation timescale was declared but no spacing_s was given. "
            "Correcting for correlation needs to know how far apart the samples "
            "are; supply the input stream's cadence, or pass tau_s=None."
        )
    # Correlated errors: fewer independent readings than the Kish count, so the
    # independent variance is inflated by kish / N_eff.
    corrected = n_effective(kish, spacing_s, tau_s, form=form)
    effective[filled] = corrected
    sigma[filled] = np.sqrt(independent_variance * kish / corrected)
    return BinnedSigma(sigma=sigma, form=form, n_effective=effective)


def propagate_systematic_binned(
    sigmas: npt.ArrayLike,
    weights: npt.ArrayLike,
    target_index: npt.ArrayLike,
    n_target: int,
) -> BinnedSigma:
    """Propagate the systematic component onto many cells at once.

    The weighted mean of the sigmas per cell (§3.3), with no reduction by
    sample count. Vectorized counterpart of :func:`propagate_systematic`.

    Parameters
    ----------
    sigmas, weights, target_index : array_like
        Long form, one entry per (target cell, reading) pair.
    n_target : int
        Number of target cells.

    Returns
    -------
    BinnedSigma
        Per-cell sigma, the label ``'systematic'``, and the Kish sample size
        of the contributing weights -- reported for symmetry, never used as a
        divisor.
    """
    total, total_sq, weighted_sigma, _ = _binned_totals(sigmas, weights, target_index, n_target)
    sigma = np.full(n_target, np.nan, dtype=np.float64)
    effective = np.zeros(n_target, dtype=np.float64)
    filled = total > 0
    # Per cell: sum(w sigma) / sum w, the weighted mean of the sigmas, with no 1/sqrt(N).
    sigma[filled] = weighted_sigma[filled] / total[filled]
    effective[filled] = total[filled] ** 2 / total_sq[filled]
    return BinnedSigma(sigma=sigma, form="systematic", n_effective=effective)


def _as_1d(array: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    """Return ``array`` as a 1-D float64 array, or raise naming the field."""
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 1:
        raise TsaraPropagationError(f"{name} must be one-dimensional, got shape {values.shape}.")
    return values


def _variance_ratio_equal_weight(n: int, rho: float) -> float:
    r"""Return ``Var(mean) / sigma^2`` for *n* equally spaced, equal-sigma samples.

    Evaluates the §3.1 double sum in its equal-weight, equal-sigma,
    equally-spaced case, where :math:`\rho_{ij} = \rho^{|i-j|}` collapses the
    :math:`N^2` terms onto *N* distinct lags::

        Var/sigma^2 = (1 / N^2) * (N + 2 * sum_{k=1}^{N-1} (N - k) rho^k)

    The sum is evaluated **term by term** rather than through its closed
    form. The closed form,
    ``N (rho - rho^N)/(1 - rho) - rho(1 - N rho^(N-1) + (N-1) rho^N)/(1 - rho)^2``,
    is algebraically identical and numerically hopeless as rho approaches 1:
    at rho = 0.999 it subtracts two quantities of order 1e4 that agree to
    four figures, through a denominator of 1e-6. The direct sum costs one
    array of length N and is exact.

    The terms decay geometrically, so the sum is truncated once ``rho**k``
    falls below :data:`_GEOMETRIC_FLOOR`; past that point the contributions no
    longer change a float64 accumulator.
    """
    if n <= 1:
        return 1.0
    if rho <= 0.0:
        return 1.0 / n
    if rho >= 1.0:
        # Perfectly correlated: averaging changes nothing.
        return 1.0
    # How many lags can still matter, given geometric decay.
    reach = int(np.ceil(np.log(_GEOMETRIC_FLOOR) / np.log(rho)))
    k = np.arange(1, min(n, max(reach, 1) + 1), dtype=np.float64)
    tail = float(np.sum((n - k) * rho**k))
    return (n + 2.0 * tail) / (n * n)


def _normalized_weights(weights: npt.ArrayLike, *, expected: int) -> npt.NDArray[np.float64] | None:
    """Return weights scaled to sum to one, or ``None`` if they sum to zero.

    A zero total is a real case rather than a defect -- it is what an empty
    overlap looks like -- so it is answered with ``None`` and turned into a
    NaN uncertainty by the caller, not raised.
    """
    w = _as_1d(weights, name="weights")
    if w.size != expected:
        raise TsaraPropagationError(
            f"Got {w.size} weight(s) for {expected} value(s); they must correspond one to one."
        )
    if np.any(w < 0):
        raise TsaraPropagationError(
            "Weights must be non-negative; a negative weight would let one sample "
            "subtract another's uncertainty."
        )
    total = float(w.sum())
    if total <= 0:
        return None
    return w / total


def _double_sum_variance(
    sigmas: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    times_s: npt.NDArray[np.float64],
    tau_s: float,
) -> float:
    r"""Return the §3.1 double sum evaluated pair by pair.

    :math:`\mathrm{Var} = \sum_i \sum_j w_i w_j \rho_{ij} \sigma_i \sigma_j`
    with :math:`\rho_{ij} = e^{-|t_i - t_j|/\tau}`. Written as
    :math:`a^\mathsf{T} C\, a` with :math:`a_i = w_i \sigma_i`, which is the
    same arithmetic in one matrix product.

    Assumption-free given AR(1): unlike the effective-sample-size forms it
    needs neither equal spacing nor equal sigma, which is exactly why it is
    the reference the others are measured against.
    """
    if sigmas.size > DOUBLE_SUM_MAX_POINTS:
        raise TsaraPropagationError(
            f"'ar1_double_sum' builds an N x N correlation matrix and got "
            f"N = {sigmas.size}, above the {DOUBLE_SUM_MAX_POINTS} limit. Use "
            "'ar1_neff', which is linear in N and agrees with this form to "
            "within the margin recorded in docs/METHODS.md §3.4."
        )
    a = weights * sigmas
    lag = np.abs(times_s[:, None] - times_s[None, :])
    correlation = np.exp(-lag / tau_s)
    return float(a @ correlation @ a)


def _mean_spacing(
    times_s: npt.ArrayLike | None,
    *,
    count: int,
    mask: npt.NDArray[np.bool_],
) -> float:
    """Return the sampling interval the effective-sample-size forms should use.

    Taken from the contributing samples when timestamps are available, and
    otherwise from the caller's own claim that the samples are consecutive.
    The median of the differences is used rather than the mean, so one gap in
    the middle of a cell does not stretch the interval that the other samples
    are corrected with.

    Raises
    ------
    TsaraPropagationError
        If no timestamps were supplied. A correlation correction without a
        spacing would have to invent one, and inventing the very quantity the
        correction is most sensitive to is the failure this module exists to
        avoid.
    """
    if times_s is None:
        raise TsaraPropagationError(
            "A decorrelation timescale was declared but no times_s was given. "
            "Correcting for correlation needs to know how far apart the samples "
            "are; supply times_s, or pass tau_s=None to treat them as independent."
        )
    times = _as_1d(times_s, name="times_s")
    if times.size != mask.size:
        raise TsaraPropagationError(
            f"Got {times.size} time(s) for {mask.size} sigma(s); they must correspond one to one."
        )
    if count < 2:
        # One sample has no spacing; any positive value gives N_eff = 1.
        return 1.0
    deltas = np.diff(np.sort(times[mask]))
    positive = deltas[deltas > 0]
    if positive.size == 0:
        raise TsaraPropagationError(
            "All contributing samples share one timestamp, so they have no spacing "
            "to correlate over."
        )
    return float(np.median(positive))


def _double_sum_binned(
    sigmas: npt.ArrayLike,
    weights: npt.ArrayLike,
    target_index: npt.ArrayLike,
    n_target: int,
    times_s: npt.ArrayLike | None,
    tau_s: float,
) -> BinnedSigma:
    """Evaluate the pairwise form one cell at a time.

    A deliberate Python loop. The pairwise form correlates every pair of
    samples by the time between them, so there is no per-cell summary to
    vectorize over -- and this is the reference form, whose whole job is to
    be obviously right rather than fast. Each cell goes through
    :func:`propagate_random`, so the reference implementation is literally
    what runs.
    """
    if times_s is None:
        raise TsaraPropagationError(
            "'ar1_double_sum' needs times_s: it correlates each pair by the time "
            "between them, which a spacing cannot supply."
        )
    sigma = _as_1d(sigmas, name="sigmas")
    w = _as_1d(weights, name="weights")
    times = _as_1d(times_s, name="times_s")
    index = np.asarray(target_index, dtype=np.int64)
    if not (sigma.shape == w.shape == index.shape == times.shape):
        raise TsaraPropagationError(
            "Long-form arrays must correspond one to one for 'ar1_double_sum'; got "
            f"{sigma.size} sigma(s), {w.size} weight(s), {index.size} target "
            f"index/indices and {times.size} time(s)."
        )
    out_sigma = np.full(n_target, np.nan, dtype=np.float64)
    out_effective = np.zeros(n_target, dtype=np.float64)
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    starts = np.searchsorted(sorted_index, np.arange(n_target), side="left")
    stops = np.searchsorted(sorted_index, np.arange(n_target), side="right")
    for cell, (lo, hi) in enumerate(zip(starts, stops)):
        if hi <= lo:
            continue
        rows = order[lo:hi]
        result = propagate_random(
            sigma[rows], w[rows], tau_s=tau_s, times_s=times[rows], form="ar1_double_sum"
        )
        out_sigma[cell] = result.sigma
        out_effective[cell] = result.n_effective
    return BinnedSigma(sigma=out_sigma, form="ar1_double_sum", n_effective=out_effective)


def _binned_totals(
    sigmas: npt.ArrayLike,
    weights: npt.ArrayLike,
    target_index: npt.ArrayLike,
    n_target: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Return per-cell weight, weight-squared, weighted-sigma and weighted-variance sums.

    Shared by both components so that a masked sample is excluded from each
    of them identically. Everything is "long form": one entry per
    (target cell, reading) pair, aggregated with ``bincount``.
    """
    sigma = _as_1d(sigmas, name="sigmas")
    w = _as_1d(weights, name="weights")
    index = np.asarray(target_index, dtype=np.int64)
    if not (sigma.shape == w.shape == index.shape):
        raise TsaraPropagationError(
            f"Long-form arrays must correspond one to one; got {sigma.size} sigma(s), "
            f"{w.size} weight(s) and {index.size} target index/indices."
        )
    if np.any(w < 0):
        raise TsaraPropagationError("Weights must be non-negative.")
    if np.any(sigma[np.isfinite(sigma)] < 0):
        raise TsaraPropagationError("Uncertainties must be non-negative.")
    # A pair with no sigma, or no weight, is excluded from every total alike.
    keep = np.isfinite(sigma) & (w > 0)
    w = np.where(keep, w, 0.0)
    sigma = np.where(keep, sigma, 0.0)

    def total_of(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # `bincount` returns an integer array when it is handed no weights at
        # all, which happens for an empty long form; asking for float64 keeps
        # every caller's arithmetic in one dtype.
        return np.asarray(np.bincount(index, weights=values, minlength=n_target), dtype=np.float64)

    return (
        total_of(w),  # sum w            (normalizer)
        total_of(w * w),  # sum w^2          (for the Kish sample size)
        total_of(w * sigma),  # sum w sigma      (systematic numerator)
        total_of(w * w * sigma * sigma),  # sum w^2 sigma^2  (random numerator)
    )
