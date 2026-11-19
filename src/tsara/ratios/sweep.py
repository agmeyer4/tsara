"""The sweep summary: every sweep point at every gate setting, built on demand (METHODS §5.6).

The fit table holds one row per member: one candidate's interval at one
(baseline window, quantile, exit), per numerator, estimator and pair clock
(``fits``). The sweep summary counts and summarizes those rows at every
**sweep point** and every **gate setting** (``gates``), so that a reader sees
where in the sweep the fits count, how their ratios move, and where the error
budget closes, without refitting. It is a view, not a saved product: a saved
one would freeze one population and one set of gates, and it is a table of
summaries (one row per sweep point and setting), small beside the fits it is
read from.

Per numerator, estimator, pair clock, window, quantile, exit and gate setting:

* ``n_fits``, the members fitted there, and ``n_pass``, those passing every
  gate; ``n_pass_<gate>``, those passing each gate alone (``gates.GATE_NAMES``);
* over the passing fits of the **population** (the candidates the caller
  names as one, the parked nights' burps of one source, say; every candidate
  when none is named): ``n_population``, ``median_ratio``, ``spread_ratio``
  (1.4826 times the median absolute deviation), ``median_mswd``, and
  ``birge_ratio``, R_B = sqrt(sum w (b - b_bar)^2 / (K - 1)) with
  w = 1 / sigma^2 the widened sigma and b_bar the weighted mean, blank below
  two fits (§5.1). R_B near 1 says the fits' sigmas explain the scatter across
  candidates; well above, that they do not, or that the population is not one
  source. Over a whole drive it mixes sources and reads high by construction,
  which is why the caller names the population;

and per numerator, estimator, pair clock, window and gate setting,
``n_candidates``, the candidates with a passing member at any quantile and
exit: how many ratios the window gives (``candidate_ratio``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from tsara.ratios.gates import GATE_COLUMNS, GATE_NAMES, gate_passes, gate_settings

if TYPE_CHECKING:  # pragma: no cover
    from tsara.config.analysis import GatesConfig

logger = logging.getLogger(__name__)

__all__ = ["SWEEP_DIMS", "sweep_summary"]

#: A sweep point of the summary: which ratio, and where in the sweep (§5.6). The
#: fit table's ``method`` is ``estimator`` here, because ``method=`` is xarray's own
#: keyword in ``.sel`` and a dimension of that name could not be selected by it.
SWEEP_DIMS: tuple[str, ...] = (
    "numerator",
    "estimator",
    "period",
    "baseline_window",
    "baseline_quantile",
    "exit_multiple",
)

#: The counts, which are zero (never blank) where a sweep point holds no fit.
_COUNTS = ("n_fits", "n_pass", *(f"n_pass_{name}" for name in GATE_NAMES), "n_population")


def sweep_summary(
    fits: pd.DataFrame,
    gates: GatesConfig,
    *,
    population: Iterable[str] | None = None,
) -> xr.Dataset:
    """Summarize the fits at every sweep point and every gate setting.

    Parameters
    ----------
    fits : pandas.DataFrame
        The fit table (``tsara.ratios.member_fits``, ``tsara.pipeline.ratio_fits``).
    gates : GatesConfig
        Every gate's thresholds; each combination is a dimension's value.
    population : iterable of str, optional
        The candidates (``candidate_id``) to read the ratio's summaries and
        R_B over, as one population; every candidate when None.

    Returns
    -------
    xarray.Dataset
        Dimensions :data:`SWEEP_DIMS` and the gates' (``gates.GATE_COLUMNS``);
        ``n_candidates`` without the quantile and the exit. Empty when there
        are no fits.
    """
    named = None if population is None else set(population)
    if fits.empty:
        return xr.Dataset(attrs={"population": _described(named)})
    fits = fits.rename(columns={"method": "estimator"})
    in_population = (
        np.ones(len(fits), dtype=bool)
        if named is None
        else fits["candidate_id"].isin(named).to_numpy()
    )
    points, candidates = [], []
    for setting in gate_settings(gates):
        each = gate_passes(fits, setting)
        every = np.logical_and.reduce([each[name] for name in GATE_NAMES])
        frame = fits.loc[
            :, [*SWEEP_DIMS, "candidate_id", "slope", "slope_sigma_widened", "mswd"]
        ].assign(
            passes=every,
            in_population=every & in_population,
            **{f"n_pass_{name}": each[name] for name in GATE_NAMES},
        )
        counts = frame.groupby(list(SWEEP_DIMS)).agg(
            n_fits=("passes", "size"),
            n_pass=("passes", "sum"),
            **{f"n_pass_{name}": (f"n_pass_{name}", "sum") for name in GATE_NAMES},
        )
        kept = frame[frame["in_population"]]
        summaries = pd.DataFrame(
            [
                _summary(group) | dict(zip(SWEEP_DIMS, key, strict=True))
                for key, group in kept.groupby(list(SWEEP_DIMS))
            ],
            columns=[
                *SWEEP_DIMS,
                "n_population",
                "median_ratio",
                "spread_ratio",
                "median_mswd",
                "birge_ratio",
            ],
        ).set_index(list(SWEEP_DIMS))
        points.append(counts.join(summaries).assign(**setting))
        window = [*SWEEP_DIMS[:4]]
        found = frame.groupby([*window, "candidate_id"])["passes"].any().groupby(level=window).sum()
        candidates.append(found.rename("n_candidates").to_frame().assign(**setting))
    table = pd.concat(points).reset_index().set_index([*SWEEP_DIMS, *GATE_COLUMNS])
    summary = table.to_xarray()
    for name in _COUNTS:
        summary[name] = summary[name].fillna(0).astype(np.int64)
    window_table = pd.concat(candidates).reset_index().set_index([*SWEEP_DIMS[:4], *GATE_COLUMNS])
    summary["n_candidates"] = window_table["n_candidates"].to_xarray().fillna(0).astype(np.int64)
    summary.attrs["population"] = _described(named)
    return summary


def _summary(group: pd.DataFrame) -> dict[str, float]:
    """Return one sweep point's population: its count, median, spread, median MSWD and R_B."""
    slopes = group["slope"].to_numpy(dtype=np.float64)
    sigma = group["slope_sigma_widened"].to_numpy(dtype=np.float64)
    median = float(np.median(slopes))
    # Blank where there is no budget to close: one fit (0 / 0), or a fit with no
    # sigma or a zero one, whose weight is not a number.
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = 1.0 / sigma**2
        mean = float(np.sum(weight * slopes) / np.sum(weight))
        birge = float(np.sqrt(np.sum(weight * (slopes - mean) ** 2) / (slopes.size - 1)))
    return {
        "n_population": float(slopes.size),
        "median_ratio": median,
        "spread_ratio": float(1.4826 * np.median(np.abs(slopes - median))),
        "median_mswd": float(np.nanmedian(group["mswd"]))
        if group["mswd"].notna().any()
        else np.nan,
        "birge_ratio": birge,
    }


def _described(named: set[str] | None) -> str:
    """Return what the population was, for the summary's attrs."""
    return "every candidate" if named is None else f"{len(named)} named candidates"
