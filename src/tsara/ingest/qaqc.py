"""Masking bad samples: range limits and instrument flags.

Masking, not deleting
---------------------
Every rule here replaces a rejected sample with ``NaN`` rather than dropping
the row. The rows of an instrument's record are its clock, and a rolling
window that silently closes over a removed sample computes a different
answer than one that sees a gap. Downstream, "no valid measurement here" and
"no measurement attempted here" must remain distinguishable, and only a
preserved timestamp does that.

Where masking sits relative to unit conversion
----------------------------------------------
**Rules run on canonical values, after conversion.** This is a real decision
with a real consequence, so it is worth stating plainly:

* ``range`` bounds are physical statements ("methane below 1700 ppb is
  impossible here"), and a manifest author writes them in the units they
  think in — the canonical ones. The shipped example manifest does exactly
  this, converting ppm→ppb and then bounding in ppb. Running the rule before
  conversion would compare ppb bounds against ppm numbers and mask the
  entire record.
* ``flag`` reads a *different* column — an instrument status word — which is
  never a converted quantity. It is looked up in the raw frame directly.

Rules compose by union: a sample rejected by any rule is masked, and the
count each rule rejected is reported separately so that "the range rule
removed 40 % of this record" is visible rather than buried in a total.

Why there is no spike rule
--------------------------
There was one — a centered rolling median/MAD (Hampel) test for sub-second
electronic glitches, kept deliberately distinct from plume detection. It was
removed during the Phase-3 walkthrough (2026-08-26, owner decision) because
on this package's own subject matter it cannot work.

A Hampel filter rejects short excursions that are large relative to a local
robust scale. That is exactly the description of a **plume** in mobile
trace-gas data. Measured on real 2-second analyzer records, 27-29 % of clear
enhancement events are two samples wide or fewer, and even among events
exceeding 100 sigma over baseline, 18 % are that narrow. A filter tuned to
reject 1-2 sample excursions therefore cannot distinguish a glitch from the
signal the package exists to find, and the measurement bore that out: at a
61-second window the rule masked plume samples 2.6 times more often than
quiet-air samples.

Removing it is the honest resolution. A rule whose safe operating window is
narrow, undocumented, and bounded on one side by "silently tests almost
nothing" and on the other by "clips real plumes" is a trap, not a tool.
Genuine instrument glitches are better handled where the information to
identify them exists: an instrument status ``flag`` column, a physical
``range`` bound, or a later stage that already knows what a plume looks like.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.config.manifest import FlagRule, RangeRule
from tsara.ingest.base import TsaraIngestError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence
    from pathlib import Path

    from tsara.config.manifest import QAQCRule

logger = logging.getLogger(__name__)

__all__ = ["MaskReport", "apply_qaqc", "masked_fraction"]


@dataclass(frozen=True)
class MaskReport:
    """How many samples one QA/QC rule rejected.

    Kept per rule rather than as a single total because the interesting
    question is almost always *which* rule fired. A range rule masking
    everything usually means the bounds are in the wrong units; a flag rule
    masking everything usually means the flag polarity is inverted. One
    combined count cannot distinguish them.

    Attributes
    ----------
    kind : str
        The rule's ``kind`` discriminator (``range`` or ``flag``).
    n_masked : int
        Samples this rule newly rejected — counted against samples that were
        still valid when it ran, so overlapping rules do not double-count.
    n_total : int
        Samples in the record, for context.
    """

    kind: str
    n_masked: int
    n_total: int

    @property
    def fraction(self) -> float:
        """Share of the record this rule rejected, in [0, 1]."""
        return self.n_masked / self.n_total if self.n_total else 0.0


def apply_qaqc(
    values: pd.Series,
    rules: Sequence[QAQCRule],
    frame: pd.DataFrame,
    *,
    variable: str,
    path: Path,
) -> tuple[pd.Series, tuple[MaskReport, ...]]:
    """Mask samples rejected by an instrument's QA/QC rules.

    Parameters
    ----------
    values : pandas.Series
        Time-indexed values **in canonical units** (see module docstring).
    rules : Sequence of QAQCRule
        Rules from the variable's manifest entry, applied in declared order.
    frame : pandas.DataFrame
        The full raw table, needed because a ``flag`` rule reads a different
        column than the one being masked.
    variable : str
        Canonical variable name, for messages.
    path : pathlib.Path
        Source file, for messages.

    Returns
    -------
    tuple
        ``(masked_values, reports)`` — a new Series with rejected samples set
        to ``NaN``, and one :class:`MaskReport` per rule in declared order.

    Raises
    ------
    TsaraIngestError
        If a ``flag`` rule names a column the file does not contain.
    """
    masked = values.astype("float64").copy()
    reports: list[MaskReport] = []
    total = len(masked)

    for rule in rules:
        # Counted against what was still valid, so two rules rejecting the
        # same sample do not each claim it.
        already_bad = masked.isna()
        reject = _evaluate(rule, masked, frame, variable=variable, path=path)
        newly = int((reject & ~already_bad).sum())
        masked = masked.mask(reject)
        reports.append(MaskReport(kind=rule.kind, n_masked=newly, n_total=total))

        if newly:
            logger.debug(
                "QA/QC '%s' masked %d/%d samples of %s in %s",
                rule.kind,
                newly,
                total,
                variable,
                path,
            )

    _warn_if_mostly_masked(masked, reports, variable=variable, path=path)
    return masked, tuple(reports)


def _evaluate(
    rule: QAQCRule,
    values: pd.Series,
    frame: pd.DataFrame,
    *,
    variable: str,
    path: Path,
) -> pd.Series:
    """Return a boolean Series marking samples this rule rejects."""
    if isinstance(rule, RangeRule):
        return _evaluate_range(rule, values)
    return _evaluate_flag(rule, values, frame, variable=variable, path=path)


def _evaluate_range(rule: RangeRule, values: pd.Series) -> pd.Series:
    """Reject values outside the declared physical bounds."""
    reject = pd.Series(False, index=values.index)
    if rule.min is not None:
        reject |= values < rule.min
    if rule.max is not None:
        reject |= values > rule.max
    # A NaN comparison is False in pandas, so already-masked samples are not
    # re-flagged here; the caller's `already_bad` bookkeeping handles them.
    return reject


def _evaluate_flag(
    rule: FlagRule,
    values: pd.Series,
    frame: pd.DataFrame,
    *,
    variable: str,
    path: Path,
) -> pd.Series:
    """Reject samples according to a separate instrument status column."""
    if rule.flag_column not in frame.columns:
        raise TsaraIngestError(
            f"QA/QC flag rule for '{variable}' names column "
            f"'{rule.flag_column}', which is not in '{path}'. "
            f"Columns present: {list(frame.columns)[:10]}."
        )

    flags = frame[rule.flag_column]
    if rule.good_values is not None:
        # Anything not explicitly good is rejected — including a missing
        # flag, since an unknown status is not evidence of a good sample.
        return pd.Series(~flags.isin(rule.good_values).to_numpy(), index=values.index)
    return pd.Series(flags.isin(rule.bad_values or []).to_numpy(), index=values.index)


def _warn_if_mostly_masked(
    masked: pd.Series,
    reports: Sequence[MaskReport],
    *,
    variable: str,
    path: Path,
) -> None:
    """Warn when QA/QC removed most of a record.

    Losing most of a variable is occasionally correct — an instrument really
    was down — but far more often means a rule is misconfigured: bounds
    written in the pre-conversion units, or a flag polarity inverted. Either
    way it is not something to discover after a full pipeline run.
    """
    total = len(masked)
    if not total or not reports:
        return
    remaining = int(masked.notna().sum())
    if remaining <= total // 2:
        worst = max(reports, key=lambda report: report.n_masked)
        logger.warning(
            "QA/QC left only %d of %d samples of '%s' in %s (largest rejecter: "
            "'%s' at %.0f%%). Check the rule's units and polarity.",
            remaining,
            total,
            variable,
            path,
            worst.kind,
            100 * worst.fraction,
        )


def masked_fraction(values: pd.Series) -> float:
    """Return the share of a series that is ``NaN``.

    A small helper used by stream assembly to record per-variable data
    coverage in the output dataset, so a user can see how much of a record
    survived without recomputing it.
    """
    return float(np.mean(np.asarray(values.isna()))) if len(values) else 0.0
