"""A candidate's ratio: read from its members that pass the gates, with every term of its error.

A candidate (``tsara.events``, ``docs/METHODS.md`` §6.8) is one thing in the
air at one baseline window, seen by its **members**, its distinct (quantile,
exit) intervals, each fitted once (``fits``). Its ratio is read from the
members that pass the gates (§5.3), one row per candidate, numerator,
estimator, pair clock and **gate setting**, every gate's thresholds swept as
the configuration lists them (``RatiosConfig.gates``), so that how many
candidates count and how their ratios move is seen across the gates as it is
across windows (§5.5):

* the **point estimate** is the **median member**: the passing member whose
  slope is the lower median of their slopes (the middle one; of two middle
  ones, the lower), so that the slope, its sigma, n and MSWD all describe one
  real fit (decided by measurement, §5.5);
* ``sigma_fit`` is that member's widened sigma, with the readings'
  persistence and the baseline's error in it (§5.4);
* ``sigma_method`` is the standard deviation of the passing members' slopes
  (n - 1), blank below two members: the spread the sweep's choices of
  quantile and exit leave in the ratio. The window and the pair clock are
  never pooled into it (a ratio belongs to its window, each clock is
  reported apart), nor is a gate setting, since a stricter gate only drops
  members whose spread this already is;
* ``sigma_sys`` is ``|b| sqrt(s_x^2 + s_y^2)``, ``s`` each axis's declared
  relative systematic term (``fits``' ``*_systematic_relative``): a gain error
  scales the ratio, an offset cancels in every enhancement and slope. Blank
  where either axis's term is not known (not declared, or reported per
  reading, which cannot be split into a scale and an offset);
* ``sigma_combined`` is the root sum of squares of the terms that are known,
  with ``sigma_method`` taken as zero for a single member and ``sigma_sys``
  left out where blank, so ``n_members`` and ``sigma_sys_known`` say what it
  holds; the interval is the ratio plus and minus the normal quantile of
  ``confidence_level`` times it. Quadrature treats the fit's error and the
  members' spread as independent, though members share readings; it is kept
  because it measured right (§5.5).

A candidate that no member passes at a gate setting still has its row, its
ratio blank and ``n_members`` 0, so that counting the candidates a gate keeps
needs no second table. Each row names its candidate's parent
(``parent_candidate_id``, the candidate at the nearest longer window, §6.8),
copied from the events stage's candidate table, so that a rule across windows
can be added later from saved tables (§5.7). Upper limits are results: a
numerator that never rises still gets its slope and interval.

A candidate's **fingerprint** is its row of ratios across numerators
(:func:`fingerprints`), built from this table rather than stored.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm

from tsara.ratios.estimators import TsaraRatioError
from tsara.ratios.gates import GATE_COLUMNS, gate_settings, passes

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import DTypeLike

    from tsara.config.analysis import GatesConfig

logger = logging.getLogger(__name__)

__all__ = ["RATIO_COLUMNS", "RATIO_DTYPES", "candidate_ratios", "fingerprints"]

#: The ratio table's columns, in order (METHODS §5.5).
RATIO_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "parent_candidate_id",
    "reference",
    "numerator",
    "method",
    "period",
    "baseline_window",
    *GATE_COLUMNS,
    "n_members",
    "event_id",
    "ratio",
    "intercept",
    "n",
    "mswd",
    "dof",
    "sigma_fit",
    "sigma_method",
    "sigma_sys",
    "sigma_sys_known",
    "sigma_combined",
    "confidence_level",
    "lower",
    "upper",
    "candidate_peak_z",
    "peak_time",
)

#: The ratio table's column types: declared so it survives Parquet exactly.
RATIO_DTYPES: dict[str, str] = {
    **dict.fromkeys(
        (
            "candidate_id",
            "parent_candidate_id",
            "reference",
            "numerator",
            "method",
            "period",
            "event_id",
        ),
        "str",
    ),
    **dict.fromkeys(("min_points", "n_members", "n", "dof"), "int64"),
    "sigma_sys_known": "bool",
    "peak_time": "datetime64[ns]",
    **dict.fromkeys(
        (
            "baseline_window",
            "min_peak_z",
            "min_signal",
            "mswd_probability",
            "plateau_probability",
            "max_window_fill",
            "ratio",
            "intercept",
            "mswd",
            "sigma_fit",
            "sigma_method",
            "sigma_sys",
            "sigma_combined",
            "confidence_level",
            "lower",
            "upper",
            "candidate_peak_z",
        ),
        "float64",
    ),
}

#: One ratio: a candidate's members, for one numerator, estimator and pair clock.
_KEY = ["candidate_id", "reference", "numerator", "method", "period"]


def candidate_ratios(
    fits: pd.DataFrame,
    candidates: pd.DataFrame,
    gates: GatesConfig,
    *,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Read every candidate's ratio from its passing members, at every gate setting.

    Parameters
    ----------
    fits : pandas.DataFrame
        The fit table (``tsara.ratios.member_fits`` or the pipeline's
        ``ratio_fits``).
    candidates : pandas.DataFrame
        The events stage's candidate table (``tsara.events.event_candidates``),
        for each candidate's parent.
    gates : GatesConfig
        Every gate's thresholds; each combination is one gate setting.
    confidence_level : float, optional
        For the interval (``RatiosConfig.confidence_level``).

    Returns
    -------
    pandas.DataFrame
        :data:`RATIO_COLUMNS`, one row per candidate, numerator, estimator,
        pair clock and gate setting, gate settings in the order of
        :data:`GATE_COLUMNS` with the last varying fastest.

    Raises
    ------
    TsaraRatioError
        If a fit names a candidate the candidate table does not hold, or the
        confidence level is not between 0 and 1.
    """
    if not 0.0 < confidence_level < 1.0:
        raise TsaraRatioError(f"confidence_level must lie between 0 and 1, not {confidence_level}.")
    parents = candidates.set_index("candidate_id")["parent_candidate_id"]
    strangers = sorted(set(fits["candidate_id"]) - set(parents.index))
    if strangers:
        raise TsaraRatioError(
            f"Fits name candidates the candidate table does not hold: {strangers[:5]}; hand "
            "the candidate table of the same event catalog."
        )
    if fits.empty:
        return pd.DataFrame(columns=list(RATIO_COLUMNS)).astype(RATIO_DTYPES)
    codes, heads = _heads(fits)
    table = pd.concat(
        [
            _at_setting(fits, codes, heads, setting, confidence_level)
            for setting in gate_settings(gates)
        ],
        ignore_index=True,
    )
    table["parent_candidate_id"] = table["candidate_id"].map(parents)
    return table.loc[:, list(RATIO_COLUMNS)].astype(RATIO_DTYPES)


def fingerprints(ratios: pd.DataFrame, *, value: str = "ratio", **select: object) -> pd.DataFrame:
    """Return each candidate's ratios across numerators, one row per candidate.

    Parameters
    ----------
    ratios : pandas.DataFrame
        The ratio table (:func:`candidate_ratios`).
    value : str, optional
        Which column fills the fingerprint (``'ratio'``, ``'sigma_combined'``,
        ``'lower'`` ...).
    **select
        Columns to hold at one value, e.g. ``method='york'``,
        ``period='native'``, ``min_peak_z=5.0``. What is left must be one
        estimator, one pair clock and one gate setting.

    Returns
    -------
    pandas.DataFrame
        Indexed by candidate, one column per numerator, in the table's order.

    Raises
    ------
    TsaraRatioError
        If a selection names no column, or more than one estimator, pair
        clock or gate setting is left.
    """
    rows = ratios
    for column, wanted in select.items():
        if column not in rows.columns:
            raise TsaraRatioError(f"The ratio table has no column '{column}' to select on.")
        rows = rows[rows[column] == wanted]
    left = rows[["method", "period", *GATE_COLUMNS]].drop_duplicates()
    if len(left) > 1:
        raise TsaraRatioError(
            f"{len(left)} estimators, pair clocks or gate settings are left; select one of each "
            f"(e.g. method='york', period='native', and each of {list(GATE_COLUMNS)})."
        )
    table = rows.pivot(index="candidate_id", columns="numerator", values=value)
    order = list(dict.fromkeys(rows["numerator"]))
    return table.loc[list(dict.fromkeys(rows["candidate_id"])), order]


def _heads(fits: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Give each fit its ratio's number, and read what does not depend on the gates.

    A ratio is one candidate, numerator, estimator and pair clock; they are
    numbered in the order of their first fit, as a groupby that does not sort
    gives them. Per ratio: its key, its window and the candidate's peak z
    from its first fit (every member holds the same, ``fits``' tallest
    member's z); and the peak time of its tallest member (the first of equal
    peaks, as argmax picks it).

    Returns
    -------
    codes : numpy.ndarray
        Each fit's ratio number.
    heads : pandas.DataFrame
        One row per ratio, in number order.
    """
    codes = fits.groupby(_KEY, sort=False).ngroup().to_numpy()
    first = np.unique(codes, return_index=True)[1]
    numbers = np.arange(first.size)
    # Each ratio's fits, tallest first. The sort is stable, so of equal peaks
    # (or a triggered variable's, which have no z at all) the first fit leads.
    tallest = np.lexsort((-fits["peak_z"].to_numpy(dtype=np.float64), codes))
    lead = tallest[np.searchsorted(codes[tallest], numbers)]
    heads = fits.iloc[first][[*_KEY, "baseline_window", "candidate_peak_z"]].reset_index(drop=True)
    heads["peak_time"] = fits["peak_time"].to_numpy()[lead]
    return codes, heads


def _at_setting(
    fits: pd.DataFrame,
    codes: np.ndarray,
    heads: pd.DataFrame,
    setting: dict[str, float],
    confidence_level: float,
) -> pd.DataFrame:
    """Return every ratio at one gate setting, from its members that pass it.

    One pass over every ratio at once: the passing members are sorted by
    ratio and then by slope, so each ratio's members sit together in slope
    order and its median member is at position (count - 1) // 2 of its run.
    The sort is stable, so equal slopes keep the order of the fits. A ratio
    with no passing member keeps its row, blank but for what ``heads`` holds.
    """
    n = len(heads)
    slope = fits["slope"].to_numpy(dtype=np.float64)
    passing = np.flatnonzero(passes(fits, setting))
    order = passing[np.lexsort((slope[passing], codes[passing]))]
    counts = np.bincount(codes[order], minlength=n)
    found = counts > 0
    member = order[(np.cumsum(counts) - counts + (counts - 1) // 2)[found]]

    def chosen(column: str, blank: object, dtype: DTypeLike = np.float64) -> np.ndarray:
        """Return the median member's value of a column, ``blank`` where no member passes."""
        values: np.ndarray = np.full(n, blank, dtype=dtype)
        values[found] = fits[column].to_numpy()[member]
        return values

    b = chosen("slope", np.nan)
    sigma_fit = chosen("slope_sigma_widened", np.nan)
    s_x = chosen("reference_systematic_relative", np.nan)
    s_y = chosen("numerator_systematic_relative", np.nan)
    # The members' spread (n - 1), blank below two members.
    in_run = codes[order]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(in_run, weights=slope[order], minlength=n) / counts
        squares = np.bincount(in_run, weights=(slope[order] - mean[in_run]) ** 2, minlength=n)
        sigma_method = np.where(counts > 1, np.sqrt(squares / (counts - 1)), np.nan)
    known = np.isfinite(s_x) & np.isfinite(s_y)
    sigma_sys = np.where(known, np.abs(b) * np.hypot(s_x, s_y), np.nan)
    # The terms that are known in quadrature: a single member's spread as
    # zero, an unknown systematic left out, and no interval without the fit's.
    combined = np.sqrt(np.nansum(np.stack([sigma_fit**2, sigma_method**2, sigma_sys**2]), axis=0))
    combined[~np.isfinite(sigma_fit)] = np.nan
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    return heads.assign(
        **setting,
        confidence_level=confidence_level,
        n_members=counts,
        event_id=chosen("event_id", None, object),
        ratio=b,
        intercept=chosen("intercept", np.nan),
        n=chosen("n", 0, np.int64),
        mswd=chosen("mswd", np.nan),
        dof=chosen("dof", 0, np.int64),
        sigma_fit=sigma_fit,
        sigma_method=sigma_method,
        sigma_sys=sigma_sys,
        sigma_sys_known=known,
        sigma_combined=combined,
        lower=b - z * combined,
        upper=b + z * combined,
    )
