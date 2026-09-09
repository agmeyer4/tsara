"""Resolving each variable's two-component uncertainty budget.

``docs/METHODS.md`` §2 models every measurement as

    x_i = x_true_i + e_rand_i + e_sys_i

and insists the two error components stay **separate** all the way through
the pipeline, because they behave differently under averaging: random error
falls off with the number of effective samples, systematic error does not.
They are combined only at the point of use.

What this module decides, and what it deliberately does not
-----------------------------------------------------------
Ingestion knows the *manifest*. It does not know the analysis config. So it
resolves exactly the budgets the manifest can state — ``declared`` (a noise
floor plus a percent-of-reading term) and ``reported`` (a per-point sigma
column the instrument wrote) — and computes those pointwise, in canonical
units.

It does **not** compute the empirical fallback. The empirical estimator is a
*rolling* statistic whose name and window come from the analysis config
(``DetectionConfig.noise_estimator``, METHODS §2.5), which ingestion has no
business reading. What ingestion does instead is *label* the variable
``empirical``, so the obligation is recorded and cannot be forgotten. This
is the shape of METHODS §2.3's promise: there is no code path in which an
uncertainty of unstated origin enters a confidence interval.

For the same reason it does **not** move a declared sigma onto the stream's
cells. A manifest may say its figures were quoted at one second while the
product is a one-minute mean (``DeclaredUncertainty.at_width``), and the
arithmetic that reconciles the two needs a decorrelation timescale, an AR(1)
model of how the error forgets itself, and the assumption that averaging is
what produced the cell (METHODS §3.4, §10.8). A unit conversion needs none of
those: it is a declared scale and offset, exact and assumption-free, which is
why *that* is applied here and this is not. So the figures are stored exactly
as declared and the mismatch is written down — the interval they describe, and
how many of those intervals fit in a cell — for the stage that needs a sigma
at a particular support to resolve in one hop, from the declaration to the
support it actually wants.

The five provenance values, and why "zero" is not "unknown"
-----------------------------------------------------------
``declared``
    Computed here from ``absolute``/``relative``.
``reported``
    Read here from the instrument's own per-point sigma column.
``empirical``
    Deferred to the stage holding the analysis config.
``zero``
    The manifest supplied a budget and deliberately omitted this component.
    METHODS §2.2: "an omitted ``systematic`` is zero". That is a *statement*
    — the author considered systematic error and declared it negligible.
``unknown``
    No budget at all. The random component then falls back to the empirical
    estimator, but the systematic component cannot: ``diff_mad`` differences
    the signal, which cancels anything slowly varying, so it is structurally
    blind to systematic error (METHODS §2.5, "honest scope"). An undeclared
    systematic component is genuinely unknown, and saying so is different
    from claiming it is zero.

Keeping ``zero`` and ``unknown`` apart is the whole point of §2.3. Collapsing
them would let an undeclared calibration silently become a claim of perfect
calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from tsara.config.manifest import DeclaredUncertainty, ReportedUncertainty
from tsara.ingest.base import TsaraIngestError
from tsara.ingest.units import convert_spread

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    import numpy.typing as npt

    from tsara.config.manifest import ComponentUncertainty, UncertaintySpec, UnitConversion

logger = logging.getLogger(__name__)

__all__ = ["ResolvedUncertainty", "UncertaintySource", "resolve_uncertainty"]

#: Provenance of one uncertainty component. See the module docstring.
UncertaintySource = Literal["declared", "reported", "empirical", "zero", "unknown"]


@dataclass(frozen=True)
class ResolvedUncertainty:
    """One variable's resolved uncertainty budget.

    Attributes
    ----------
    random : numpy.ndarray or None
        Per-point 1-sigma random component in canonical units, or ``None``
        when the component was not computed here (``empirical``, ``zero``,
        ``unknown``). ``None`` is not "no uncertainty" — read
        :attr:`random_source` to learn what it means.
    systematic : numpy.ndarray or None
        Per-point 1-sigma systematic component, same convention.
    random_source, systematic_source : str
        Provenance of each component.
    decorrelation_timescale : str or None
        The random component's correlation timescale as declared, passed
        through untouched for the alignment and regression stages that
        consume it (METHODS §3.4).
    at_width : str or None
        The averaging interval a declared *random* sigma was quoted at, when
        the manifest said. The figures are stored as declared; this records
        what they describe, which is a fact a reader six months later needs
        and cannot re-derive from the numbers.
    at_width_ratio : float or None
        Median cell width divided by that quoted interval — the *N* of
        METHODS §3.4, and emphatically not N_eff. A 1 s figure on 60 s cells
        gives 60.0, so the size of the mismatch is legible at a glance and a
        later stage has the count it needs without re-reading bounds. None
        when the stream has no cells to compare against.
    systematic_at_width : str or None
        The same declaration on the *systematic* component. Recorded rather
        than acted on in a stronger sense than the random one: a systematic
        error is correlated across samples by definition, so it does not
        average down at all (METHODS §3.3) and no interval can change it.
        Recorded anyway, because a manifest that states one has said
        something about its instrument, and silently dropping it would make
        the product quieter than the manifest.
    """

    random: npt.NDArray[np.float64] | None
    systematic: npt.NDArray[np.float64] | None
    random_source: UncertaintySource
    systematic_source: UncertaintySource
    decorrelation_timescale: str | None = None
    at_width: str | None = None
    at_width_ratio: float | None = None
    systematic_at_width: str | None = None

    @property
    def source(self) -> str:
        """Species-level provenance label for the output's ``uncertainty_source``.

        METHODS §2.4 specifies one label per species. Real manifests mix
        modes freely — the shipped example pairs a *reported* random
        component with a *declared* systematic one — so genuinely mixed
        budgets report ``mixed``, and the per-component labels remain
        available for anyone who needs the detail.

        The wholly-undeclared case reports ``empirical`` rather than
        ``mixed``: its systematic component is unknown precisely *because*
        the fallback is empirical, so one word describes it honestly.
        """
        if self.random_source == "empirical" and self.systematic_source == "unknown":
            return "empirical"
        if self.random_source == self.systematic_source:
            return self.random_source
        return "mixed"


def resolve_uncertainty(
    values: pd.Series,
    spec: UncertaintySpec | None,
    frame: pd.DataFrame,
    *,
    conversion: UnitConversion | None,
    variable: str,
    path: Path,
    cell_width_ns: npt.NDArray[np.int64] | None = None,
) -> ResolvedUncertainty:
    """Resolve a variable's uncertainty budget into per-point sigmas.

    Parameters
    ----------
    values : pandas.Series
        Canonical-unit values, already QA/QC masked. Masked samples produce
        masked sigmas: an uncertainty without a measurement is meaningless.
    spec : UncertaintySpec or None
        The manifest's budget for this variable, if any.
    frame : pandas.DataFrame
        Raw table, needed to read a ``reported`` sigma column.
    conversion : UnitConversion or None
        The variable's unit conversion. Applied to a reported sigma column by
        scale only (see :func:`~tsara.ingest.units.convert_spread`).
    variable : str
        Canonical variable name, for messages.
    path : pathlib.Path
        Source file, for messages.
    cell_width_ns : numpy.ndarray or None
        Each row's cell width. Used **only** to record how a declared
        ``at_width`` compares with the cells the figures landed on; no sigma
        is rescaled here (see the module docstring). None when the stream has
        no cell boundaries.

    Returns
    -------
    ResolvedUncertainty
        Computed components and their provenance.

    Raises
    ------
    TsaraIngestError
        If a ``reported`` component names a column absent from the file.
    """
    if spec is None:
        # No budget at all: random falls back to the empirical estimator
        # later, systematic is genuinely unknown.
        return ResolvedUncertainty(
            random=None,
            systematic=None,
            random_source="empirical",
            systematic_source="unknown",
        )

    random, random_source = _resolve_component(
        spec.random,
        values,
        frame,
        conversion=conversion,
        variable=variable,
        path=path,
        absent_source="empirical",
    )
    systematic, systematic_source = _resolve_component(
        spec.systematic,
        values,
        frame,
        conversion=conversion,
        variable=variable,
        path=path,
        absent_source="zero",
    )

    at_width, at_width_ratio = None, None
    if isinstance(spec.random, DeclaredUncertainty) and spec.random.at_width is not None:
        at_width = spec.random.at_width
        at_width_ratio = _cells_per_quoted_interval(
            at_width, cell_width_ns, variable=variable, path=path
        )

    systematic_at_width = (
        spec.systematic.at_width if isinstance(spec.systematic, DeclaredUncertainty) else None
    )
    if systematic_at_width is not None:
        # Worth a warning of its own rather than the same one: for the random
        # component the interval is a real fact that a later stage will act
        # on, while here it can never do anything, and a manifest author who
        # wrote it probably expects otherwise.
        logger.warning(
            "%s: '%s' declares a systematic uncertainty at %s. A systematic "
            "error does not average down, so no averaging interval can change "
            "it (METHODS 3.3); the figure is used as given and the "
            "declaration is recorded.",
            path,
            variable,
            systematic_at_width,
        )
    return ResolvedUncertainty(
        random=random,
        systematic=systematic,
        random_source=random_source,
        systematic_source=systematic_source,
        decorrelation_timescale=spec.decorrelation_timescale,
        at_width=at_width,
        at_width_ratio=at_width_ratio,
        systematic_at_width=systematic_at_width,
    )


def _cells_per_quoted_interval(
    at_width: str,
    cell_width_ns: npt.NDArray[np.int64] | None,
    *,
    variable: str,
    path: Path,
) -> float | None:
    """How many quoted intervals fit in a cell, recorded and never applied.

    This is the *N* of METHODS §3.4 — the count a later stage divides by,
    after correcting it to N_eff with a decorrelation timescale. Returning
    the raw count rather than a category ("wider", "finer") avoids inventing
    a tolerance: measured cadences are jittered, so 60 s cells against a 1 s
    figure give 59.98 rather than 60, and a category boundary would have to
    guess how close is equal. The number says it exactly.

    The median is used because widths are per row and a duty-cycled sampler's
    cells genuinely vary (canister fills of 14-16 s, METHODS §10.5). One
    summary number belongs in an attribute; the per-row widths are in the
    bounds variable for anyone who needs them.
    """
    if cell_width_ns is None:
        # No cells to compare against — a record too short to have a
        # measurable cadence. The declaration is still recorded.
        return None
    quoted_ns = float(pd.Timedelta(at_width).value)
    ratio = float(np.median(np.asarray(cell_width_ns, dtype="float64"))) / quoted_ns
    if ratio != 1.0:
        logger.warning(
            "%s: '%s' declares its uncertainty at %s, but its cells are %.4g "
            "times that interval. The figures are stored exactly as declared. "
            "Moving them onto another support needs a decorrelation timescale "
            "and is done by the stage that needs it (METHODS 3.4, 10.8), not "
            "at ingestion.",
            path,
            variable,
            at_width,
            ratio,
        )
    return ratio


def _resolve_component(
    declared: ComponentUncertainty | None,
    values: pd.Series,
    frame: pd.DataFrame,
    *,
    conversion: UnitConversion | None,
    variable: str,
    path: Path,
    absent_source: UncertaintySource,
) -> tuple[npt.NDArray[np.float64] | None, UncertaintySource]:
    """Resolve one component (random or systematic) to sigmas plus provenance."""
    if declared is None:
        return None, absent_source
    if isinstance(declared, DeclaredUncertainty):
        return _declared_sigma(declared, values), "declared"
    return (
        _reported_sigma(
            declared, values, frame, conversion=conversion, variable=variable, path=path
        ),
        "reported",
    )


def _declared_sigma(component: DeclaredUncertainty, values: pd.Series) -> npt.NDArray[np.float64]:
    """Compute sigma = sqrt(absolute^2 + (relative * value)^2).

    The two-term form matches how instrument teams report precision: a noise
    floor plus a percent-of-reading term, combined in quadrature because the
    two contributions are independent.

    ``absolute`` is declared in *canonical* units (the schema says so), and
    ``values`` are canonical by the time this runs, so no conversion is
    applied here — doing so would scale the floor twice. The magnitude of
    the value is used, so a legitimately negative reading (a below-baseline
    difference, a sign-flipped convention) still yields a positive spread.
    A masked value yields a masked sigma automatically, since NaN propagates.
    """
    magnitude = np.abs(np.asarray(values, dtype="float64"))
    sigma: npt.NDArray[np.float64] = np.sqrt(
        component.absolute**2 + (component.relative * magnitude) ** 2
    )
    return sigma


def _reported_sigma(
    component: ReportedUncertainty,
    values: pd.Series,
    frame: pd.DataFrame,
    *,
    conversion: UnitConversion | None,
    variable: str,
    path: Path,
) -> npt.NDArray[np.float64]:
    """Read a per-point sigma column and put it in canonical units."""
    if component.column not in frame.columns:
        raise TsaraIngestError(
            f"Uncertainty for '{variable}' declares reported column "
            f"'{component.column}', which is not in '{path}'. "
            f"Columns present: {list(frame.columns)[:10]}."
        )

    reported = np.asarray(pd.to_numeric(frame[component.column], errors="coerce"), dtype="float64")

    # A negative spread is not a spread. In practice this is a missing-value
    # sentinel (-9999) in a column whose na_values were never declared, so
    # masking it is both the safe reading and the informative one.
    #
    # This MUST happen before the unit conversion, not after. `convert_spread`
    # takes an absolute value -- correctly, since a negative `scale` is a
    # legitimate sign-convention flip whose magnitude must survive -- so a
    # test applied afterwards sees nothing negative to find. The guard then
    # worked only for variables with no conversion, i.e. it failed precisely
    # where the manifest was doing more work: a -9999 under a ppm->ppb
    # conversion became a silent 9,999,000 ppb "1-sigma". For a random
    # component that merely drives the point's inverse-variance weight to
    # nothing; for a systematic component, combined as a weighted mean of
    # sigmas rather than in inverse variance, one such value dominates the
    # entire bin.
    negative = reported < 0
    n_negative = int(np.count_nonzero(negative))
    if n_negative:
        logger.warning(
            "Uncertainty column '%s' for '%s' has %d negative value(s) in %s; "
            "masking them. A negative 1-sigma is usually an undeclared "
            "missing-value sentinel.",
            component.column,
            variable,
            n_negative,
            path,
        )
        reported = np.where(negative, np.nan, reported)

    sigma = convert_spread(reported, conversion)

    # An uncertainty without a surviving measurement is meaningless, and
    # carrying one would let a masked sample re-enter a weighted fit.
    return np.where(np.asarray(values.isna()), np.nan, sigma)
