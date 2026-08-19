"""Masking bad samples: range limits, instrument flags, and spikes.

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
* ``spike`` is unaffected either way: both the rolling median and the
  rolling MAD scale linearly, so the ratio the test thresholds on is
  invariant under any affine conversion. It is ordered with the others
  because there is no reason to make the order matter.
* ``flag`` reads a *different* column — an instrument status word — which is
  never a converted quantity. It is looked up in the raw frame directly.

Rules compose by union: a sample rejected by any rule is masked, and the
count each rule rejected is reported separately so that "the range rule
removed 40 % of this record" is visible rather than buried in a total.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.config.manifest import FlagRule, RangeRule, SpikeRule
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
        The rule's ``kind`` discriminator (``range``/``flag``/``spike``).
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
    if isinstance(rule, FlagRule):
        return _evaluate_flag(rule, values, frame, variable=variable, path=path)
    return _evaluate_spike(rule, values)


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


def _evaluate_spike(rule: SpikeRule, values: pd.Series) -> pd.Series:
    """Reject isolated glitches with a rolling median/MAD (Hampel) test.

    Two implementation notes that matter scientifically:

    * **The window is centered.** A spike is symmetric in time; a trailing
      window would compare each sample only with its past and would flag the
      *leading edge of a real plume* as readily as a glitch.
    * **A zero MAD masks nothing.** When enough of a window shares one value
      the MAD collapses to zero, and the threshold with it, which would
      reject every sample differing from the local median at all. This is
      not a corner case: on real analyzer records a substantial fraction of
      rolling windows have zero MAD, and acting on that zero scale rejects
      several percent of perfectly good data as "spikes". Refusing to act on
      a degenerate scale is the safe reading, and it mirrors the
      quantization guard the noise estimators use (``docs/METHODS.md`` §2.5).

      The cost is a documented blind spot: a glitch in *perfectly* constant
      data is not masked, because such data offers no scale to judge it
      against. Real measurements always carry local variation, so a real
      glitch always has one.

    The threshold is ``n_mad`` × the raw MAD, exactly as
    :class:`~tsara.config.manifest.SpikeRule` documents. Note that this is
    *not* scaled to sigma units: for Gaussian noise MAD ≈ 0.6745·σ, so the
    default ``n_mad=6`` corresponds to roughly 4σ.
    """
    if not values.index.is_monotonic_increasing:
        # Not defensive padding. Archive records really do step backwards
        # sometimes — logger clock corrections, buffered writes, merge steps
        # in an upstream processing chain — and a time-based rolling window
        # is the first thing in the pipeline to notice. pandas' own message
        # ("index values must be monotonic") names neither the variable nor
        # the file. Sorting is the orchestration stage's job — deliberately,
        # since it is a cross-file concern — so this reports rather than
        # silently reordering, which would hide a real clock problem.
        raise TsaraIngestError(
            "Spike rule needs a monotonically increasing time index, but this "
            "record's timestamps go backwards somewhere. Concatenate and sort "
            "the instrument's files before applying QA/QC."
        )

    window = pd.Timedelta(rule.window)
    # min_periods=1 keeps the record's first and last samples testable rather
    # than silently exempt; a short window there simply yields a wider MAD.
    rolling = values.rolling(window, center=True, min_periods=1)
    median = rolling.median()
    deviation = (values - median).abs()
    mad = deviation.rolling(window, center=True, min_periods=1).median()

    threshold = rule.n_mad * mad
    return (deviation > threshold) & (mad > 0)


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
