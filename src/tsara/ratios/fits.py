"""A line through every member of every candidate: the ratio stage's fits.

What a fit is
-------------
A ratio is read across a candidate's **members**: its distinct (quantile,
exit) intervals at its window (``docs/METHODS.md`` §5.2). Entry copies are one
interval (§6.8, "Candidates"), so each member is fitted once, never once per
entry. A **fit** is one straight line through the pairs of one member: the
numerator's enhancement against the reference's, both put on one pair clock
by pairing (``tsara.align.pair_species``, §11.4), at the member's own sweep
point. Every candidate is fitted, gated or not; the gates are thresholds on
numbers every fit stores (§5.3), applied afterwards.

Which pairs a member holds
--------------------------
A pair cell belongs to a member when its midpoint lies in the member's
interval, start <= midpoint < end: the rule a triggered variable's readings
follow (§6.8), and on the reference's own cells exactly the event index
(tested). Members overlap (the exits nest), so a cell can belong to several;
the estimators fit many groups at once, each on its own rows, so a member
fitted beside the others gets exactly the line it gets alone (§4).

What a fit stores
-----------------
The line and its statistics as the estimator returns them (§4), and the
numbers the gates read (§5.3): the member's peak z (for a reference that
takes its events from a trigger, the trigger event's), the points used, the
**signal**, the reference's swing inside the member over its noise,
sqrt(max(0, var(dx) / mean(sigma_x^2) - 1)) with the variance on n - 1, and
York's MSWD with its n - 2 degrees of freedom. Least squares' rows carry
York's MSWD for the same pairs, since least squares has none of its own.

The height a gate reads is the candidate's tallest member's (measured in
7c1's fourth stage: a tall candidate's lower members are as good air as its
tallest, §5.3), so every row also carries ``candidate_peak_z``. And beside
each member's own line, York's line over the **plateau**: a window
:data:`PLATEAU_WIDTH` times the member's duration, centred on it, every pair
cell in it at the member's sweep point. Its MSWD is a gate's number (off by
default); measured, it rejects most of the mixtures the MSWD gate lets
through, and half of the good single-source members with them, so whether
to use it is the sweep's to show (§5.3).

The slope's sigma widened (§5.4)
--------------------------------
York's sigma times sqrt(N / N_eff), for readings whose errors persist (the
MSWD cannot see them), times sqrt(max(1, MSWD)), the Birge scale factor,
which widens an understated sigma and never narrows a generous one. N_eff is
``tsara.core.propagation.n_effective`` over the fit's points at the pair
clock's median spacing, with the longer of the two enhancements' declared
decorrelation times; with none declared the factor is 1 and ``tau_s`` is
blank. Least squares' sigma already comes from the scatter, so it takes the
persistence factor only. Every factor is stored beside the widened sigma.

Weights and support
-------------------
The weights are the readings' own random sigma, carried beside the
enhancements by pairing, not the enhancement's, which adds the baseline's
error that every reading of an event shares (§4.3). Each fit records the
largest borrowed share among its pairs (§11.2.4); neither it nor the
pairing's count of readings shared between pairs (§11.4.1) weighs a point.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.core.naming import (
    INSTRUMENT_ATTR,
    QUANTILE_DIM,
    TIME_COORD,
    WINDOW_DIM,
    borrowed_name,
    sigma_rand_name,
)
from tsara.core.propagation import n_effective
from tsara.core.support import stream_cells
from tsara.core.timebase import NS_PER_S
from tsara.ratios.estimators import FitStatus, LineFitter, TsaraRatioError, get_line_fit

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import xarray as xr

logger = logging.getLogger(__name__)

__all__ = ["FIT_COLUMNS", "PLATEAU_WIDTH", "PairColumns", "member_fits"]

#: The fit table's columns, in order: one row per member, estimator,
#: numerator and pair clock (METHODS §5.2-5.4). The member is named by the
#: ``event_id`` of its lowest entry and by its ``candidate_id``.
FIT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "candidate_id",
    "reference",
    "numerator",
    "period",
    "method",
    "baseline_window",
    "baseline_quantile",
    "exit_multiple",
    "start_time",
    "peak_time",
    "end_time",
    "peak_z",
    "candidate_peak_z",
    "n",
    "n_dropped",
    "status",
    "slope",
    "intercept",
    "slope_sigma",
    "intercept_sigma",
    "r_squared",
    "iterations",
    "mswd",
    "dof",
    "signal",
    "plateau_slope",
    "plateau_mswd",
    "plateau_dof",
    "tau_s",
    "persistence_factor",
    "birge_factor",
    "slope_sigma_widened",
    "borrowed_max",
    "reference_sigma_provenance",
    "numerator_sigma_provenance",
)

#: Declared so a table survives Parquet exactly, as the event catalog's do.
_FIT_DTYPES: dict[str, str] = {
    **dict.fromkeys(
        (
            "event_id",
            "candidate_id",
            "reference",
            "numerator",
            "period",
            "method",
            "status",
            "reference_sigma_provenance",
            "numerator_sigma_provenance",
        ),
        "str",
    ),
    **dict.fromkeys(("start_time", "peak_time", "end_time"), "datetime64[ns]"),
    **dict.fromkeys(("n", "n_dropped", "iterations", "dof", "plateau_dof"), "int64"),
    **dict.fromkeys(
        (
            "baseline_window",
            "baseline_quantile",
            "exit_multiple",
            "peak_z",
            "candidate_peak_z",
            "slope",
            "intercept",
            "slope_sigma",
            "intercept_sigma",
            "r_squared",
            "mswd",
            "signal",
            "plateau_slope",
            "plateau_mswd",
            "tau_s",
            "persistence_factor",
            "birge_factor",
            "slope_sigma_widened",
            "borrowed_max",
        ),
        "float64",
    ),
}

#: How many times its member's duration the plateau fit's window is, centred
#: on the member (§5.3). A measured constant, not a knob: at 2 the plateau's
#: MSWD gate rejected 120 of the 142 mixtures the member's own MSWD let
#: through and kept 37 % of the good single-source members; at 4, 12 %.
PLATEAU_WIDTH = 2

#: The attribute a variable declares its decorrelation time in (§3.4), as
#: ingestion writes it and the baseline stage copies it onto an enhancement.
_TAU_ATTR = "decorrelation_timescale"

#: What a provenance column says of a sigma that was never carried.
_NO_SIGMA = "none"


@dataclass(frozen=True)
class PairColumns:
    """The columns of a paired product a fit reads, as pairing named them.

    Attributes
    ----------
    numerator, reference : str
        The two enhancements (``PairedSpecies.y_name`` and ``.x_name``),
        each over ``(time, baseline_window, baseline_quantile)``.
    numerator_reading, reference_reading : str
        The two readings, carried beside them (``PairedSpecies.carried``);
        their random sigma, where declared, weighs the fit (METHODS §4.3).
    """

    numerator: str
    reference: str
    numerator_reading: str
    reference_reading: str


def member_fits(
    pairs: xr.Dataset,
    catalog: pd.DataFrame,
    columns: PairColumns,
    *,
    reference: str,
    numerator: str,
    period: str,
    methods: Sequence[str] = ("ols", "york"),
) -> pd.DataFrame:
    """Fit the numerator against the reference over every member of every candidate.

    Parameters
    ----------
    pairs : xarray.Dataset
        One pairing of the numerator's and the reference's enhancements, with
        both readings carried (``tsara.align.pair_species(..., carry=)``).
    catalog : pandas.DataFrame
        The event catalog, its candidates linked (``tsara.events``); only the
        reference's rows are read, and a triggered reference's trigger rows
        for their height.
    columns : PairColumns
        Which columns of ``pairs`` hold what.
    reference, numerator : str
        The two variables as ``'<instrument>.<variable>'``: the reference
        selects the catalog's rows, and both label the table.
    period : str
        The pair clock's name, ``'native'`` or a duration, for the table.
    methods : sequence of str, optional
        Registered estimators (``RatiosConfig.methods``).

    Returns
    -------
    pandas.DataFrame
        :data:`FIT_COLUMNS`, one row per member and estimator, members in
        the order of their candidate, quantile and exit, estimators in the
        order given. Empty, with those columns, when the reference has no
        events.

    Raises
    ------
    TsaraRatioError
        If a column is missing or does not carry the sweep, the reference's
        enhancement comes from another instrument, the catalog's candidates
        are not linked, an entry copy's interval differs from its member's,
        a member's sweep point is not in the pairing, the pair cells are not
        in time order, or an estimator is not registered.
    """
    instrument, _, variable = reference.partition(".")
    deltas = _deltas(pairs, columns, instrument)
    fitters = [(method, get_line_fit(method)) for method in methods]
    members = _members(catalog, instrument, variable)
    if members.empty:
        return _table([])
    # The pair cells, and their midpoints in time order: membership is a search.
    cells = stream_cells(pairs, "paired")
    midpoints = cells.midpoint_ns
    if np.any(np.diff(midpoints) < 0):
        raise TsaraRatioError(
            "The pair cells are not in time order, so membership cannot be searched."
        )
    rows = _rows(members, pairs, midpoints)
    # Each member's pairs at its own sweep point, and the readings' own sigma.
    sigma_x, provenance_x = _sigma(pairs, columns.reference_reading)
    sigma_y, provenance_y = _sigma(pairs, columns.numerator_reading)
    x, y, sx, sy = _gather(deltas, (sigma_x, sigma_y), rows)
    n_members = len(members)
    fits = {
        method: fitter(x, y, sigma_x=sx, sigma_y=sy, group=rows.group, n_groups=n_members)
        for method, fitter in fitters
    }
    york = fits.get("york")
    # York again over each member's plateau window, when York is run at all.
    plateau = _plateau(members, pairs, midpoints, deltas, (sigma_x, sigma_y), dict(fitters))
    signal = _signal(x, y, sx, rows.group, n_members)
    tau_s = _tau(pairs, columns)
    spacing_s = float(np.median(np.diff(midpoints))) / NS_PER_S if midpoints.size > 1 else np.nan
    borrowed = _borrowed(pairs, columns, rows, x, y, n_members)
    frames = []
    for position, (method, fit) in enumerate(fits.items()):
        persistence = _persistence(fit.n, spacing_s, tau_s)
        is_york = method == "york"
        birge = np.sqrt(np.fmax(1.0, fit.mswd)) if is_york else np.full(n_members, np.nan)
        widened = fit.slope_sigma * persistence * (birge if is_york else 1.0)
        mswd = york.mswd if york is not None else np.full(n_members, np.nan)
        dof = (york.n if york is not None else fit.n) - 2
        frames.append(
            pd.DataFrame(
                {
                    "event_id": members["event_id"].to_numpy(),
                    "candidate_id": members["candidate_id"].to_numpy(),
                    "reference": reference,
                    "numerator": numerator,
                    "period": period,
                    "method": method,
                    "baseline_window": members["baseline_window"].to_numpy(),
                    "baseline_quantile": members["baseline_quantile"].to_numpy(),
                    "exit_multiple": members["exit_multiple"].to_numpy(),
                    "start_time": members["start_time"].to_numpy(),
                    "peak_time": members["peak_time"].to_numpy(),
                    "end_time": members["end_time"].to_numpy(),
                    "peak_z": members["peak_z"].to_numpy(),
                    "candidate_peak_z": members.groupby("candidate_id", sort=False)["peak_z"]
                    .transform("max")
                    .to_numpy(),
                    "n": fit.n,
                    "n_dropped": fit.n_dropped,
                    "status": [FitStatus(code).name for code in fit.status],
                    "slope": fit.slope,
                    "intercept": fit.intercept,
                    "slope_sigma": fit.slope_sigma,
                    "intercept_sigma": fit.intercept_sigma,
                    "r_squared": fit.r_squared,
                    "iterations": fit.iterations,
                    "mswd": mswd,
                    "dof": dof,
                    "signal": signal,
                    "plateau_slope": plateau[0],
                    "plateau_mswd": plateau[1],
                    "plateau_dof": plateau[2],
                    "tau_s": tau_s,
                    "persistence_factor": persistence,
                    "birge_factor": birge,
                    "slope_sigma_widened": widened,
                    "borrowed_max": borrowed,
                    "reference_sigma_provenance": provenance_x,
                    "numerator_sigma_provenance": provenance_y,
                },
                # Numbered member-major, so that sorting by it puts each
                # member's estimators together, in the order given.
                index=np.arange(n_members) * len(fits) + position,
            )
        )
    return _table(frames).sort_index(kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class _Rows:
    """Every (member, pair cell) membership, as aligned arrays: one entry per fit point."""

    group: npt.NDArray[np.intp]
    cell: npt.NDArray[np.intp]
    window: npt.NDArray[np.intp]
    quantile: npt.NDArray[np.intp]


def _deltas(
    pairs: xr.Dataset, columns: PairColumns, instrument: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the numerator's and the reference's enhancements as (cell, window, quantile)."""
    for name in (
        columns.numerator,
        columns.reference,
        columns.numerator_reading,
        columns.reference_reading,
    ):
        if name not in pairs:
            raise TsaraRatioError(f"The paired product holds no column '{name}'.")
    source = pairs[columns.reference].attrs.get(INSTRUMENT_ATTR)
    if source != instrument:
        raise TsaraRatioError(
            f"Column '{columns.reference}' comes from instrument '{source}', and the reference "
            f"is on '{instrument}'; were the numerator and the reference swapped?"
        )
    arrays = []
    for name in (columns.numerator, columns.reference):
        dims = set(map(str, pairs[name].dims))
        if dims != {TIME_COORD, WINDOW_DIM, QUANTILE_DIM}:
            raise TsaraRatioError(
                f"Column '{name}' has dimensions {sorted(dims)}; a fit reads an enhancement over "
                f"'{TIME_COORD}', '{WINDOW_DIM}' and '{QUANTILE_DIM}', one version per sweep point."
            )
        arrays.append(
            np.asarray(
                pairs[name].transpose(TIME_COORD, WINDOW_DIM, QUANTILE_DIM).values, dtype=np.float64
            )
        )
    return arrays[0], arrays[1]


def _members(catalog: pd.DataFrame, instrument: str, variable: str) -> pd.DataFrame:
    """Return the reference's members: one row per (candidate, quantile, exit), its lowest entry.

    The lowest entry names the member, since every entry copy holds the same
    interval and the lowest holds it whatever the gate; that the copies do
    agree is checked rather than assumed (§6.8: 54 of 54).
    """
    rows = catalog[(catalog["instrument"] == instrument) & (catalog["species"] == variable)]
    if rows.empty:
        return rows
    if rows["candidate_id"].isna().any():
        raise TsaraRatioError(
            f"The catalog's events of '{instrument}.{variable}' name no candidate; link them "
            "first (tsara.events.link_candidates, which event_catalog calls)."
        )
    key = ["candidate_id", "baseline_quantile", "exit_multiple"]
    rows = rows.sort_values([*key, "enter_multiple"], kind="stable")
    spans = rows.groupby(key, sort=False)[["start_time", "end_time"]].nunique()
    if (spans > 1).to_numpy().any():
        raise TsaraRatioError(
            f"Entry copies of one member of '{instrument}.{variable}' hold different intervals, "
            "which the events stage promises they do not (METHODS §6.8); fitting one of them "
            "would choose among them."
        )
    members = rows.drop_duplicates(key, keep="first").reset_index(drop=True)
    # A reference that takes its events from a trigger has no z of its own:
    # its members stand as tall as the trigger's events they copy.
    triggered = members["peak_z"].isna() & members["trigger_event_id"].notna()
    if triggered.any():
        heights = catalog.set_index("event_id")["peak_z"]
        members.loc[triggered, "peak_z"] = (
            members.loc[triggered, "trigger_event_id"].map(heights).to_numpy()
        )
    return members


def _rows(members: pd.DataFrame, pairs: xr.Dataset, midpoints: npt.NDArray[np.int64]) -> _Rows:
    """Return every member's pair cells, start <= midpoint < end, at its sweep point."""
    start = members["start_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    stop = members["end_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    first = np.searchsorted(midpoints, start, side="left")
    last = np.searchsorted(midpoints, stop, side="left")
    counts = np.maximum(last - first, 0)
    group = np.repeat(np.arange(len(members), dtype=np.intp), counts)
    # The k-th point of a member is its first cell plus k.
    heads = np.cumsum(counts) - counts
    cell = (np.arange(int(counts.sum()), dtype=np.intp) - np.repeat(heads, counts)) + np.repeat(
        first, counts
    )
    return _Rows(
        group=group,
        cell=cell.astype(np.intp),
        window=np.repeat(_positions(members["baseline_window"], pairs, WINDOW_DIM), counts),
        quantile=np.repeat(_positions(members["baseline_quantile"], pairs, QUANTILE_DIM), counts),
    )


def _gather(
    deltas: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    sigmas: tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None],
    rows: _Rows,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]:
    """Return the reference's and numerator's values at each row's sweep point, and their sigmas."""
    x = deltas[1][rows.cell, rows.window, rows.quantile]
    y = deltas[0][rows.cell, rows.window, rows.quantile]
    sx = None if sigmas[0] is None else sigmas[0][rows.cell]
    sy = None if sigmas[1] is None else sigmas[1][rows.cell]
    return x, y, sx, sy


def _plateau(
    members: pd.DataFrame,
    pairs: xr.Dataset,
    midpoints: npt.NDArray[np.int64],
    deltas: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    sigmas: tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None],
    fitters: dict[str, LineFitter],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return York's slope, MSWD and dof over each member's plateau window (§5.3).

    The window is :data:`PLATEAU_WIDTH` times the member's duration, centred
    on it, and holds every pair cell whose midpoint lies in it, at the
    member's sweep point, inside the member's events or not. Blank when York
    is not among the estimators run: only York has an MSWD to gate on.
    """
    n_members = len(members)
    if "york" not in fitters:
        return np.full(n_members, np.nan), np.full(n_members, np.nan), np.full(n_members, -2)
    start = members["start_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    stop = members["end_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    centre, half = start + (stop - start) // 2, (stop - start) * PLATEAU_WIDTH // 2
    wide = members.assign(
        start_time=pd.to_datetime(centre - half), end_time=pd.to_datetime(centre + half)
    )
    rows = _rows(wide, pairs, midpoints)
    x, y, sx, sy = _gather(deltas, sigmas, rows)
    line = fitters["york"](x, y, sigma_x=sx, sigma_y=sy, group=rows.group, n_groups=n_members)
    return line.slope, line.mswd, line.n - 2


def _positions(values: pd.Series, pairs: xr.Dataset, dim: str) -> npt.NDArray[np.intp]:
    """Return where each member's sweep coordinate sits in the pairing's, refusing a stranger."""
    known = {float(value): index for index, value in enumerate(pairs[dim].values)}
    wanted = values.to_numpy(dtype=np.float64)
    missing = sorted({float(value) for value in wanted} - set(known))
    if missing:
        raise TsaraRatioError(
            f"Members sit at {dim} {missing}, which the pairing does not hold "
            f"({sorted(known)}); pair the baseline states the catalog was found on."
        )
    return np.asarray([known[float(value)] for value in wanted], dtype=np.intp)


def _sigma(pairs: xr.Dataset, reading: str) -> tuple[npt.NDArray[np.float64] | None, str]:
    """Return a carried reading's random sigma on the pair cells, and its provenance."""
    name = sigma_rand_name(reading)
    if name not in pairs:
        return None, _NO_SIGMA
    sigma = pairs[name]
    return (
        np.asarray(sigma.values, dtype=np.float64),
        str(sigma.attrs.get("uncertainty_provenance", "unknown")),
    )


def _signal(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    sigma_x: npt.NDArray[np.float64] | None,
    group: npt.NDArray[np.intp],
    n_groups: int,
) -> npt.NDArray[np.float64]:
    """Return each member's swing of x over its noise, sqrt(max(0, var(x)/mean(sigma^2) - 1)).

    Over the rows a fit could weigh (finite x, y and sigma_x), the variance on
    n - 1 so that the noise's share of it is unbiased. Blank below two rows or
    with no sigma; unbounded for an exact reference, whose noise is 0.
    """
    if sigma_x is None:
        return np.full(n_groups, np.nan)
    used = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma_x)
    g, xs = group[used], x[used]
    count = np.bincount(g, minlength=n_groups).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(g, weights=xs, minlength=n_groups) / count
        spread = np.bincount(g, weights=(xs - mean[g]) ** 2, minlength=n_groups) / (count - 1)
        noise = np.bincount(g, weights=sigma_x[used] ** 2, minlength=n_groups) / count
        ratio = spread / noise
    return np.where(count >= 2, np.sqrt(np.fmax(0.0, ratio - 1.0)), np.nan)


def _tau(pairs: xr.Dataset, columns: PairColumns) -> float:
    """Return the longer of the two enhancements' declared decorrelation times, in seconds.

    NaN when neither declares one: the persistence factor is then 1, and the
    blank says so.
    """
    declared = [
        pd.Timedelta(str(pairs[name].attrs[_TAU_ATTR])).total_seconds()
        for name in (columns.numerator, columns.reference)
        if _TAU_ATTR in pairs[name].attrs
    ]
    return max(declared) if declared else np.nan


def _borrowed(
    pairs: xr.Dataset,
    columns: PairColumns,
    rows: _Rows,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_groups: int,
) -> npt.NDArray[np.float64]:
    """Return each member's largest borrowed share over its pairs, of either member (§11.2.4)."""
    used = np.isfinite(x) & np.isfinite(y)
    largest = np.full(n_groups, -np.inf)
    for name in (columns.numerator, columns.reference):
        share = np.asarray(
            pairs[borrowed_name(name)].transpose(TIME_COORD, WINDOW_DIM, QUANTILE_DIM).values,
            dtype=np.float64,
        )[rows.cell, rows.window, rows.quantile]
        np.maximum.at(largest, rows.group[used], np.nan_to_num(share[used], nan=-np.inf))
    return np.where(np.isfinite(largest), largest, np.nan)


def _persistence(
    n: npt.NDArray[np.int64], spacing_s: float, tau_s: float
) -> npt.NDArray[np.float64]:
    """Return sqrt(N / N_eff) for each fit's N points (§5.4); 1 with no decorrelation time."""
    points = np.maximum(np.asarray(n, dtype=np.float64), 1.0)
    if not np.isfinite(tau_s) or not np.isfinite(spacing_s):
        return np.ones_like(points)
    return np.sqrt(points / n_effective(points, spacing_s, tau_s))


def _table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Return the fit table with its declared types, empty when there is nothing."""
    table = pd.concat(frames) if frames else pd.DataFrame(columns=list(FIT_COLUMNS))
    return table.loc[:, list(FIT_COLUMNS)].astype(_FIT_DTYPES)
