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
    table = pd.concat(
        [_at_setting(fits, setting, confidence_level) for setting in gate_settings(gates)],
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


def _at_setting(
    fits: pd.DataFrame, setting: dict[str, float], confidence_level: float
) -> pd.DataFrame:
    """Return every candidate's ratio at one gate setting."""
    passing = pd.Series(passes(fits, setting), index=fits.index)
    z = float(norm.ppf(0.5 + confidence_level / 2.0))
    rows = []
    for key, members in fits.groupby(_KEY, sort=False):
        row: dict[str, object] = dict(zip(_KEY, key, strict=True))
        row.update(setting)
        row["baseline_window"] = float(members["baseline_window"].iloc[0])
        row["candidate_peak_z"] = float(members["candidate_peak_z"].max())
        row["peak_time"] = members["peak_time"].iloc[int(np.argmax(members["peak_z"].to_numpy()))]
        row["confidence_level"] = confidence_level
        kept = members[passing[members.index]]
        row["n_members"] = len(kept)
        row.update(_ratio(kept, z))
        rows.append(row)
    return pd.DataFrame(rows)


def _ratio(kept: pd.DataFrame, z: float) -> dict[str, object]:
    """Return the ratio and its terms from one candidate's passing members."""
    blank: dict[str, object] = {
        "event_id": None,
        "ratio": np.nan,
        "intercept": np.nan,
        "n": 0,
        "mswd": np.nan,
        "dof": 0,
        "sigma_fit": np.nan,
        "sigma_method": np.nan,
        "sigma_sys": np.nan,
        "sigma_sys_known": False,
        "sigma_combined": np.nan,
        "lower": np.nan,
        "upper": np.nan,
    }
    if kept.empty:
        return blank
    slopes = kept["slope"].to_numpy(dtype=np.float64)
    # The median member: the lower median, the member at sorted position
    # (n - 1) // 2. It is the one nearest the median slope, the lower on a tie,
    # in exact arithmetic; with an even count the two middle slopes are equally
    # near only there, and in floats one wins by 1e-17, so a tie is never read
    # off the distances.
    k = int(np.argsort(slopes, kind="stable")[(slopes.size - 1) // 2])
    member = kept.iloc[k]
    b = float(member["slope"])
    sigma_fit = float(member["slope_sigma_widened"])
    sigma_method = float(np.std(slopes, ddof=1)) if slopes.size > 1 else np.nan
    s_x = float(member["reference_systematic_relative"])
    s_y = float(member["numerator_systematic_relative"])
    known = bool(np.isfinite(s_x) and np.isfinite(s_y))
    sigma_sys = abs(b) * float(np.hypot(s_x, s_y)) if known else np.nan
    combined = float(np.sqrt(np.nansum([sigma_fit**2, sigma_method**2, sigma_sys**2])))
    if not np.isfinite(sigma_fit):
        combined = np.nan
    return {
        "event_id": member["event_id"],
        "ratio": b,
        "intercept": float(member["intercept"]),
        "n": int(member["n"]),
        "mswd": float(member["mswd"]),
        "dof": int(member["dof"]),
        "sigma_fit": sigma_fit,
        "sigma_method": sigma_method,
        "sigma_sys": sigma_sys,
        "sigma_sys_known": known,
        "sigma_combined": combined,
        "lower": b - z * combined,
        "upper": b + z * combined,
    }
