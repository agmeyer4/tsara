"""The gates: which fits count, at each setting of the thresholds (METHODS §5.3).

A gate never changes a fit; it decides whether the fit counts. Every fit
stores the numbers a gate reads (``fits``), and a gate is a threshold applied
afterwards. Each gate takes a list of thresholds (``GatesConfig``), and every
combination of the lists is one **gate setting**, so that a candidate's ratio
(``candidate_ratio``) and the sweep summary (``sweep``) are read at every
setting without refitting. Both read the gates here, so they cannot disagree
about which fits count.

The gates, each at its own threshold:

* **height**: the candidate's tallest member's peak z (``candidate_peak_z``)
  at least ``min_peak_z``;
* **points**: ``n`` at least ``min_points``;
* **signal**: the reference's swing over its noise at least ``min_signal``;
* **one source**: the MSWD over what a single source is expected to read
  (``mswd / mswd_expected``, §5.4) at most the chi-squared quantile at
  ``mswd_probability`` over the degrees of freedom;
* **plateau**: the plateau's MSWD at most the same quantile at
  ``plateau_probability`` over its degrees of freedom (off at 1, the default);
* **crowding**: the member's duration over its baseline window at most
  ``max_window_fill``: a plume that fills most of its window is read at a
  longer window, where it fits (3/4 by default, infinite lets everything
  through).

A probability of 1 lets everything through. A fit with no MSWD to read (least
squares where York could not weigh, or was not run) is not judged by the
one-source or plateau gate, and one with no interval to read by the crowding
gate: it is reported ungated there (§5.3). A fit with no line (``status`` not
``OK``) passes no setting.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import chi2

if TYPE_CHECKING:  # pragma: no cover
    from tsara.config.analysis import GatesConfig

__all__ = ["GATE_COLUMNS", "GATE_NAMES", "gate_passes", "gate_settings", "passes"]

#: The gate setting a row is read at, one column per gate's threshold (§5.3).
GATE_COLUMNS: tuple[str, ...] = (
    "min_peak_z",
    "min_points",
    "min_signal",
    "mswd_probability",
    "plateau_probability",
    "max_window_fill",
)

#: Each gate by name, in the order of :data:`GATE_COLUMNS`.
GATE_NAMES: tuple[str, ...] = ("height", "points", "signal", "one_source", "plateau", "crowding")


def gate_settings(gates: GatesConfig) -> list[dict[str, float]]:
    """Return every gate setting, the last gate's threshold varying fastest.

    Parameters
    ----------
    gates : GatesConfig

    Returns
    -------
    list of dict
        One ``{column: threshold}`` per setting, keyed by :data:`GATE_COLUMNS`.
    """
    lists = [getattr(gates, name) for name in GATE_COLUMNS]
    return [dict(zip(GATE_COLUMNS, setting, strict=True)) for setting in itertools.product(*lists)]


def gate_passes(fits: pd.DataFrame, setting: dict[str, float]) -> dict[str, np.ndarray]:
    """Return, for each gate of :data:`GATE_NAMES`, which fits pass it alone at one setting.

    The fit's own line is not a gate: a fit with no line fails every gate
    here, so a gate's count is of the fits it could judge and passed.
    """
    line = (fits["status"] == "OK").to_numpy()
    return {
        "height": line & (fits["candidate_peak_z"] >= setting["min_peak_z"]).to_numpy(),
        "points": line & (fits["n"] >= setting["min_points"]).to_numpy(),
        "signal": line & (fits["signal"] >= setting["min_signal"]).to_numpy(),
        "one_source": line
        & _below(fits["mswd"] / fits["mswd_expected"], fits["dof"], setting["mswd_probability"]),
        "plateau": line
        & _below(fits["plateau_mswd"], fits["plateau_dof"], setting["plateau_probability"]),
        "crowding": line & _fits_its_window(fits, setting["max_window_fill"]),
    }


def passes(fits: pd.DataFrame, setting: dict[str, float]) -> np.ndarray:
    """Return which fits pass every gate at one setting."""
    each = gate_passes(fits, setting)
    return np.asarray(np.logical_and.reduce([each[name] for name in GATE_NAMES]), dtype=bool)


def _below(mswd: pd.Series, dof: pd.Series, probability: float) -> np.ndarray:
    """Return whether each MSWD is unknown or at most its chi-squared quantile over dof."""
    m = mswd.to_numpy(dtype=np.float64)
    d = dof.to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        # At a probability of 1 the quantile is infinite: everything known passes.
        limit = chi2.ppf(probability, np.where(d > 0, d, np.nan)) / d
    return np.asarray(np.isnan(m) | (m <= limit), dtype=bool)


def _fits_its_window(fits: pd.DataFrame, most: float) -> np.ndarray:
    """Return whether each member's duration over its window is unknown or at most ``most``."""
    # Through pandas: a fit table built without intervals holds them as NaN, read as NaT.
    duration = pd.to_datetime(fits["end_time"]) - pd.to_datetime(fits["start_time"])
    seconds = duration.dt.total_seconds().to_numpy(dtype=np.float64)
    fill = seconds / fits["baseline_window"].to_numpy(dtype=np.float64)
    return np.asarray(np.isnan(fill) | (fill <= most), dtype=bool)
