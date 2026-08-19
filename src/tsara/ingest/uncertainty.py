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

The four provenance values, and why "zero" is not "unknown"
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
    """

    random: npt.NDArray[np.float64] | None
    systematic: npt.NDArray[np.float64] | None
    random_source: UncertaintySource
    systematic_source: UncertaintySource
    decorrelation_timescale: str | None = None

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
    return ResolvedUncertainty(
        random=random,
        systematic=systematic,
        random_source=random_source,
        systematic_source=systematic_source,
        decorrelation_timescale=spec.decorrelation_timescale,
    )


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

    reported = pd.to_numeric(frame[component.column], errors="coerce")
    sigma = convert_spread(np.asarray(reported, dtype="float64"), conversion)

    # A negative spread is not a spread. In practice this is a missing-value
    # sentinel (-9999) in a column whose na_values were never declared, so
    # masking it is both the safe reading and the informative one.
    negative = sigma < 0
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
        sigma = np.where(negative, np.nan, sigma)

    # An uncertainty without a surviving measurement is meaningless, and
    # carrying one would let a masked sample re-enter a weighted fit.
    return np.where(np.asarray(values.isna()), np.nan, sigma)
