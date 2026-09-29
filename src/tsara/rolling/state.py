"""Assembling the rolling state: one product per instrument, on its own cells.

For every reading of a stream and every point of the sweep, the state holds
the baseline, the enhancement, and their uncertainties, beside the reading
itself (``docs/METHODS.md`` §6.2). It lives on the stream's own ``time`` and
``time_bnds``: a window is only the interval a statistic looked at, and the
value it yields belongs to the cell the window was centred on. The sweep is
two more dimensions, ``baseline_window`` (seconds) and ``baseline_quantile``.

What this module decides, and what it leaves to the methods
-----------------------------------------------------------
A method (:mod:`tsara.rolling.methods`) applies its definition to every
window and reports what it found. This module applies the *rules*: the count
rule of §6.4 and the width rule of §6.3 blank a sweep point at a reading,
with the reason recorded three ways (the count column beside it, the rule in
the attributes, and the blank fraction of every sweep point), and one
warning per stream names the sweep points that are blank at every reading.
It forms the enhancement, unclipped, and its uncertainties under the rules
of §6.7. And it writes what a baseline records (§6.6): no ``cell_methods``,
since a statistic over a window stated at a cell inside it has two supports
and no CF string is true of both; the enhancement inherits the reading's.

Which variables
---------------
By default every non-circular ``role: gas`` variable of the stream; a caller
may name any scalar variable over ``time``. A direction is refused: it has
no low quantile.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import xarray as xr

from tsara._version import __version__
from tsara.core.bundle import pin_time_encoding
from tsara.core.naming import (
    CELL_METHODS_ATTR,
    TIME_BOUNDS_VAR,
    TIME_COORD,
    baseline_name,
    coverage_window_name,
    enhancement_name,
    is_circular,
    is_companion_name,
    n_readings_window_name,
    sigma_rand_name,
    sigma_sys_name,
)
from tsara.core.support import COPY_RATIO, stream_cells
from tsara.core.timebase import NS_PER_S
from tsara.rolling.methods import (
    QUANTILE_DIM,
    WINDOW_DIM,
    BaselineRequest,
    BaselineResult,
    get_baseline_method,
)
from tsara.rolling.quantile import MAX_BLOCK_ELEMENTS
from tsara.rolling.windows import TsaraRollingError, duration_ns

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    from tsara.config.analysis import BaselineConfig

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_BLANK_FRACTION_ATTR",
    "BASELINE_MEMBERSHIP_ATTR",
    "BASELINE_METHOD_ATTR",
    "BASELINE_MIN_READINGS_ATTR",
    "BASELINE_TOO_WIDE_ATTR",
    "BASELINE_WINDOWS_ATTR",
    "ROLLING_STAGE",
    "SIGMA_ASSUMPTION_ATTR",
    "SIGMA_RULE_ATTR",
    "rolling_state",
    "rolling_states",
]


#: What a rolling state's ``tsara_stage`` says.
ROLLING_STAGE = "rolling"

#: Attrs the baseline column carries instead of a ``cell_methods`` (§6.6):
#: the method that made it, the membership rule, the windows as configured,
#: the count each quantile required, and how much of every sweep point is
#: blank and how much of every window failed the width rule.
BASELINE_METHOD_ATTR = "tsara_baseline_method"
BASELINE_MEMBERSHIP_ATTR = "tsara_baseline_membership"
BASELINE_WINDOWS_ATTR = "tsara_baseline_windows"
BASELINE_MIN_READINGS_ATTR = "tsara_baseline_min_readings"
BASELINE_BLANK_FRACTION_ATTR = "tsara_baseline_blank_fraction"
BASELINE_TOO_WIDE_ATTR = "tsara_baseline_too_wide_fraction"

#: Attrs on the uncertainty companions the state writes: the assumption a
#: sampling sigma rests on, and which rule of §6.7 formed an enhancement's.
SIGMA_ASSUMPTION_ATTR = "tsara_sigma_assumption"
SIGMA_RULE_ATTR = "tsara_sigma_rule"

#: How readings meet a window, the one rule (§6.3).
_MEMBERSHIP = "overlap"

#: At most this many variable names in the one warning about blank sweep
#: points: a canister's fifty-odd VOCs share one sampling pattern and would
#: otherwise repeat one sentence fifty times (measured: 18 716 characters for
#: the 2024-07-18 iWAS file). The same cap, for the same reason, as the join's
#: support warning in `align.binning`.
_MAX_NAMED = 8


class _BlankPoint(NamedTuple):
    """One sweep point blank at every reading of one variable, and why.

    ``kind`` is the rule that blanked it: ``count`` (every window held fewer
    readings than the quantile needs), ``width`` (every window held a reading
    at least ``COPY_RATIO`` times as wide as itself), ``count or width`` (each
    window failed one or the other), or ``adopted`` (an adopted baseline,
    blank because the donor's was). ``held`` is the most readings any window
    held, for the count rule's message.
    """

    variable: str
    point: str
    kind: str
    needs: int
    held: int


#: What Woodruff's figure assumes, written where the figure is (§6.7).
_WOODRUFF_ASSUMPTION = (
    "Woodruff order-statistic interval: readings in a window treated as exchangeable "
    "draws; a floor on the sampling error when the air is autocorrelated (METHODS 6.7)"
)


# ---------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------


def rolling_state(
    stream: xr.Dataset,
    *,
    instrument: str,
    baseline: BaselineConfig,
    variables: Sequence[str] | None = None,
    provided: Mapping[str, xr.Dataset] | None = None,
    block_elements: int = MAX_BLOCK_ELEMENTS,
) -> xr.Dataset:
    """Compute one stream's rolling state.

    Parameters
    ----------
    stream : xarray.Dataset
        The instrument's stream, at native rate, with its cells.
    instrument : str
        The stream's name in the campaign, as the manifest and the
        configuration's ``baseline.methods`` keys spell it.
    baseline : BaselineConfig
        The sweep, the count rule and the method per variable.
    variables : sequence of str, optional
        Which variables to roll. ``None`` takes every non-circular
        ``role: gas`` variable.
    provided : mapping of str to xarray.Dataset, optional
        For each ``from_field`` variable, the donor's baseline joined onto
        this stream's cells (see :func:`~tsara.rolling.methods.baseline_from_field`).
    block_elements : int, optional
        The rolling engine's per-block budget; not a configuration.

    Returns
    -------
    xarray.Dataset
        The rolling state: on the stream's ``time`` and ``time_bnds``, with
        ``baseline_window`` (seconds) and ``baseline_quantile`` as further
        dimensions; per variable ``x`` the reading and its sigmas copied,
        ``baseline_x``, ``enhancement_x``, their sigma companions, and for a
        rolling quantile ``n_readings_window_x`` and ``coverage_window_x``.
        The noise scale detection quotes thresholds in is not here: it is
        detection's, since its fix is a loop with detection (§6.8).

    Raises
    ------
    TsaraRollingError
        If a named variable is missing, circular or not over ``time``; if no
        variable is selected; or if a method refuses its request.
    """
    readings = stream_cells(stream, instrument)
    selection = _select(stream, instrument, variables)
    windows_ns = tuple(duration_ns(window) for window in baseline.windows)
    quantiles = tuple(float(q) for q in baseline.quantiles)
    minimum = tuple(baseline.min_readings_for(q) for q in quantiles)
    data_vars: dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
    blank_everywhere: list[_BlankPoint] = []
    for variable in selection:
        request = BaselineRequest(
            stream=stream,
            instrument=instrument,
            variable=variable,
            readings=readings,
            values=np.asarray(stream[variable].values, dtype=np.float64),
            windows_ns=windows_ns,
            quantiles=quantiles,
            method=baseline.method_for(instrument, variable),
            provided=None if provided is None else provided.get(variable),
            block_elements=block_elements,
        )
        result = get_baseline_method(request.method.method)(request)
        columns, fully_blank = _one_variable(stream, request, result, baseline, minimum)
        data_vars.update(columns)
        blank_everywhere.extend(fully_blank)
    coords: dict[str, object] = {
        str(name): stream.coords[name] for name in stream.coords if name != TIME_BOUNDS_VAR
    }
    coords[WINDOW_DIM] = (
        (WINDOW_DIM,),
        np.asarray(windows_ns, dtype=np.float64) / NS_PER_S,
        {"units": "s", "long_name": "baseline window duration"},
    )
    coords[QUANTILE_DIM] = (
        (QUANTILE_DIM,),
        np.asarray(quantiles, dtype=np.float64),
        {"units": "1", "long_name": "baseline quantile"},
    )
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            **stream.attrs,
            "tsara_version": __version__,
            "tsara_stage": ROLLING_STAGE,
            "tsara_instrument": instrument,
            BASELINE_WINDOWS_ATTR: ", ".join(baseline.windows),
        },
    )
    # The stream's cells, exactly: the state lives on them (§6.2).
    dataset.coords[TIME_BOUNDS_VAR] = stream[TIME_BOUNDS_VAR]
    dataset[TIME_COORD].attrs.update(stream[TIME_COORD].attrs)
    if blank_everywhere:
        _warn_blank_everywhere(instrument, blank_everywhere)
    pin_time_encoding(dataset)
    return dataset


def rolling_states(
    streams: Mapping[str, xr.Dataset],
    baseline: BaselineConfig,
    *,
    provided: Mapping[str, Mapping[str, xr.Dataset]] | None = None,
    block_elements: int = MAX_BLOCK_ELEMENTS,
) -> dict[str, xr.Dataset]:
    """Compute the rolling state of every stream of a campaign.

    The default variables of each stream; a stream with none is skipped
    with a note rather than refused, since a GPS or met stream has no gas.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    baseline : BaselineConfig
        The sweep, the count rule and the method per variable.
    provided : mapping of str to mapping, optional
        Per instrument, per ``from_field`` variable, the joined donor
        baseline.
    block_elements : int, optional
        The rolling engine's per-block budget.

    Returns
    -------
    dict of str to xarray.Dataset
        One rolling state per stream that had a variable to roll.
    """
    states: dict[str, xr.Dataset] = {}
    for instrument, stream in streams.items():
        if not _default_variables(stream):
            logger.info("Stream '%s' has no gas variable to roll; skipped.", instrument)
            continue
        states[instrument] = rolling_state(
            stream,
            instrument=instrument,
            baseline=baseline,
            provided=None if provided is None else provided.get(instrument),
            block_elements=block_elements,
        )
    return states


# ---------------------------------------------------------------------------
# Its steps
# ---------------------------------------------------------------------------


def _select(stream: xr.Dataset, instrument: str, variables: Sequence[str] | None) -> list[str]:
    """Return the variables to roll, refusing what cannot be rolled."""
    if variables is None:
        chosen = _default_variables(stream)
        if not chosen:
            raise TsaraRollingError(
                f"Stream '{instrument}' has no non-circular role='gas' variable to roll; "
                "name the variables to roll, or roll another stream."
            )
        return chosen
    if not variables:
        raise TsaraRollingError("No variables named to roll.")
    for variable in variables:
        if variable not in stream.data_vars:
            raise TsaraRollingError(
                f"Stream '{instrument}' has no variable '{variable}'; its variables: "
                f"{sorted(map(str, stream.data_vars))}."
            )
        if is_circular(stream[variable].attrs):
            raise TsaraRollingError(
                f"'{variable}' on '{instrument}' is circular; a direction has no low "
                "quantile and no baseline."
            )
        if stream[variable].dims != (TIME_COORD,):
            raise TsaraRollingError(
                f"'{variable}' on '{instrument}' has dimensions {stream[variable].dims}; "
                f"a rolling state is computed for one value per cell, over ('{TIME_COORD}',)."
            )
    return list(dict.fromkeys(variables))


def _default_variables(stream: xr.Dataset) -> list[str]:
    """Return the stream's non-circular gas variables over ``time``."""
    return [
        str(name)
        for name in stream.data_vars
        if stream[name].attrs.get("role") == "gas"
        and not is_circular(stream[name].attrs)
        and not is_companion_name(str(name))
        and stream[name].dims == (TIME_COORD,)
    ]


def _one_variable(
    stream: xr.Dataset,
    request: BaselineRequest,
    result: BaselineResult,
    baseline: BaselineConfig,
    minimum: tuple[int, ...],
) -> tuple[
    dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray, dict[str, object]]],
    list[_BlankPoint],
]:
    """Return one variable's columns of the state, and its sweep points blank everywhere.

    Applies the count and width rules to what the method found, forms the
    enhancement and its uncertainties, and writes what each column records.
    """
    variable = request.variable
    values = request.values
    reading = stream[variable]
    dims = (TIME_COORD, WINDOW_DIM, QUANTILE_DIM)
    n_windows, n_quantiles = len(request.windows_ns), len(request.quantiles)
    # 1. The rules: blank a sweep point at a reading whose window holds too few
    # readings for the quantile, or holds a reading too wide for the window.
    blank = np.zeros(result.baseline.shape, dtype=np.bool_)
    if result.n_readings is not None:
        for column, count in enumerate(minimum):
            blank[:, :, column] = result.n_readings < count
    if result.too_wide is not None:
        blank |= result.too_wide[:, :, None]
    baseline_values = np.where(blank, np.nan, result.baseline)
    # The blank share of every sweep point, from the baseline itself, so that an
    # adopted baseline blank where its donor was reports it the same way.
    blank_fraction = np.isnan(baseline_values).mean(axis=0)  # (windows, quantiles)
    fully_blank = [
        _blank_reason(
            result, variable, f"{baseline.windows[w]}/{request.quantiles[q]:g}", w, minimum[q]
        )
        for w in range(n_windows)
        for q in range(n_quantiles)
        if blank_fraction[w, q] == 1.0
    ]
    # 2. What the baseline records (§6.6): no cell method, the attrs instead.
    # The window rules are recorded only where windows exist: a constant has
    # none and an adopted baseline's are the donor's.
    base_attrs: dict[str, object] = {
        "units": reading.attrs.get("units", ""),
        "field": reading.attrs.get("field", variable),
        "description": f"Baseline of {variable} at each window and quantile of the sweep.",
        BASELINE_METHOD_ATTR: request.method.method,
        BASELINE_WINDOWS_ATTR: ", ".join(baseline.windows),
        BASELINE_BLANK_FRACTION_ATTR: blank_fraction.reshape(-1),
        **result.attrs,
    }
    if result.n_readings is not None and result.too_wide is not None:
        base_attrs[BASELINE_MEMBERSHIP_ATTR] = _MEMBERSHIP
        base_attrs[BASELINE_MIN_READINGS_ATTR] = np.asarray(minimum, dtype=np.int64)
        base_attrs[BASELINE_TOO_WIDE_ATTR] = np.asarray(
            result.too_wide.mean(axis=0), dtype=np.float64
        )
    columns: dict[str, xr.DataArray | tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
    # 3. The reading and its declared sigmas, copied, so the state stands alone.
    columns[variable] = reading
    for name in (sigma_rand_name(variable), sigma_sys_name(variable)):
        if name in stream.data_vars:
            columns[name] = stream[name]
    columns[baseline_name(variable)] = (dims, baseline_values, base_attrs)
    # 4. The window's qualifiers, when the method has windows.
    if result.n_readings is not None and result.coverage is not None:
        columns[n_readings_window_name(variable)] = (
            (TIME_COORD, WINDOW_DIM),
            result.n_readings,
            {
                "units": "1",
                "description": (
                    f"Readings of {variable} contributing to the window centred on each "
                    "cell: a finite value overlapping the window by a positive amount."
                ),
            },
        )
        columns[coverage_window_name(variable)] = (
            (TIME_COORD, WINDOW_DIM),
            result.coverage,
            {
                "units": "1",
                "description": (
                    f"Share of the window centred on each cell covered by contributing "
                    f"{variable} readings."
                ),
            },
        )
    # 5. The enhancement, unclipped, inheriting the reading's cell method.
    enhancement = values[:, None, None] - baseline_values
    enhancement_attrs: dict[str, object] = {
        key: value for key, value in reading.attrs.items() if key in ("units", "field", "role")
    }
    enhancement_attrs["description"] = (
        f"Enhancement of {variable} over its baseline at each window and quantile; "
        "never clipped at zero."
    )
    enhancement_attrs[BASELINE_METHOD_ATTR] = request.method.method
    if CELL_METHODS_ATTR in reading.attrs:
        enhancement_attrs[CELL_METHODS_ATTR] = reading.attrs[CELL_METHODS_ATTR]
    # 6. The uncertainties, under the rules of §6.7.
    sigmas = _sigma_columns(stream, request, result, baseline_values, enhancement, blank, dims)
    # A component with no sigma column is said so on the enhancement, not left
    # to silence: the reading's own provenance for that component, `unknown`
    # when the reading states none (§6.7). No estimate stands in for it here:
    # the noise scale that could is detection's (§6.8).
    enh = enhancement_name(variable)
    for key, companion in (
        ("uncertainty_provenance_random", sigma_rand_name(enh)),
        ("uncertainty_provenance_systematic", sigma_sys_name(enh)),
    ):
        if companion not in sigmas:
            enhancement_attrs[key] = str(reading.attrs.get(key, "unknown"))
    columns[enh] = (dims, enhancement, enhancement_attrs)
    columns.update(sigmas)
    return columns, fully_blank


def _blank_reason(
    result: BaselineResult, variable: str, point: str, window: int, needs: int
) -> _BlankPoint:
    """Return which rule left one sweep point blank at every reading of a variable.

    Asked per reading and summarized: the count rule where every window held
    fewer readings than ``needs``, the width rule where every window held a
    reading too wide for it, both where each window failed one of them. A
    baseline with no windows of its own is blank only because the product it
    adopted is.
    """
    if result.n_readings is None or result.too_wide is None:
        # No windows of its own: an adopted baseline, blank because the product is.
        return _BlankPoint(variable, point, "adopted", needs, 0)
    counts = result.n_readings[:, window]
    held = int(counts.max())
    short = counts < needs
    if short.all():
        kind = "count"
    elif result.too_wide[:, window].all():
        kind = "width"
    else:
        kind = "count or width"
    return _BlankPoint(variable, point, kind, needs, held)


def _sigma_columns(
    stream: xr.Dataset,
    request: BaselineRequest,
    result: BaselineResult,
    baseline_values: np.ndarray,
    enhancement: np.ndarray,
    blank: np.ndarray,
    dims: tuple[str, ...],
) -> dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]]:
    """Return the sigma companions of a baseline and its enhancement (§6.7)."""
    variable = request.variable
    units = str(stream[variable].attrs.get("units", ""))
    reading_random = _companion(stream, sigma_rand_name(variable))
    reading_systematic = _companion(stream, sigma_sys_name(variable))
    base = baseline_name(variable)
    enh = enhancement_name(variable)
    columns: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
    # The baseline's own sigmas, blanked with it.
    baseline_random = (
        None if result.sigma_rand is None else np.where(blank, np.nan, result.sigma_rand)
    )
    baseline_systematic = (
        None if result.sigma_sys is None else np.where(blank, np.nan, result.sigma_sys)
    )
    if baseline_random is not None:
        columns[sigma_rand_name(base)] = (
            dims,
            baseline_random,
            {
                "units": units,
                "description": f"Random (sampling) 1-sigma of {base}.",
                "uncertainty_component": "random",
                "uncertainty_provenance": "empirical",
                SIGMA_ASSUMPTION_ATTR: _WOODRUFF_ASSUMPTION,
            }
            if result.same_instrument
            else {
                "units": units,
                "description": f"Random 1-sigma of {base}, joined from the donor's.",
                "uncertainty_component": "random",
                "uncertainty_provenance": "empirical",
            },
        )
    if baseline_systematic is not None:
        columns[sigma_sys_name(base)] = (
            dims,
            baseline_systematic,
            {
                "units": units,
                "description": (
                    f"Systematic 1-sigma of {base}: the reading's, at the quantile position."
                    if result.same_instrument
                    else f"Systematic 1-sigma of {base}, joined from the donor's."
                ),
                "uncertainty_component": "systematic",
                "uncertainty_provenance": str(
                    stream[variable].attrs.get("uncertainty_provenance_systematic", "unknown")
                ),
            },
        )
    # The enhancement's random sigma: the reading's and the baseline's in quadrature.
    if reading_random is not None:
        random_term = reading_random[:, None, None] ** 2
        if baseline_random is not None:
            random_term = random_term + baseline_random**2
        columns[sigma_rand_name(enh)] = (
            dims,
            np.where(np.isfinite(enhancement), np.sqrt(random_term), np.nan),
            {
                "units": units,
                "description": (
                    f"Random 1-sigma of {enh}: the reading's and the baseline's in quadrature."
                ),
                "uncertainty_component": "random",
                "uncertainty_provenance": str(
                    stream[variable].attrs.get("uncertainty_provenance_random", "unknown")
                ),
                SIGMA_RULE_ATTR: "quadrature",
            },
        )
    # The enhancement's systematic sigma: by where the baseline came from.
    if reading_systematic is not None:
        reading_values = request.values
        if result.same_instrument:
            # An offset common to reading and baseline cancels; a gain scales
            # the difference: sigma_sys(x) |delta| / |x|, exact for a pure
            # gain, an upper bound for any mix. Undefined at a reading of
            # exactly zero, where a gain's sigma is zero and says nothing.
            scale = np.divide(
                np.abs(enhancement),
                np.abs(reading_values)[:, None, None],
                out=np.full(enhancement.shape, np.nan),
                where=(reading_values != 0)[:, None, None],
            )
            systematic = reading_systematic[:, None, None] * scale
            rule = "same instrument: sigma_sys(x) |enhancement| / |x|"
        else:
            systematic_term = reading_systematic[:, None, None] ** 2
            if baseline_systematic is not None:
                systematic_term = systematic_term + baseline_systematic**2
            systematic = np.broadcast_to(np.sqrt(systematic_term), enhancement.shape)
            rule = "quadrature with the donor's" if baseline_systematic is not None else "reading"
        columns[sigma_sys_name(enh)] = (
            dims,
            np.where(np.isfinite(enhancement), systematic, np.nan),
            {
                "units": units,
                "description": f"Systematic 1-sigma of {enh}.",
                "uncertainty_component": "systematic",
                "uncertainty_provenance": str(
                    stream[variable].attrs.get("uncertainty_provenance_systematic", "unknown")
                ),
                SIGMA_RULE_ATTR: rule,
            },
        )
    return columns


def _companion(stream: xr.Dataset, name: str) -> np.ndarray | None:
    """Return a stream's sigma companion as a float array, or ``None`` when absent."""
    if name not in stream.data_vars:
        return None
    return np.asarray(stream[name].values, dtype=np.float64)


def _warn_blank_everywhere(instrument: str, blank: Sequence[_BlankPoint]) -> None:
    """Warn once per stream about the sweep points blank at every reading, and why.

    Variables that share one pattern -- the same points, blank for the same
    rule and the same required count, as every variable of one instrument
    normally does, since they share its cells -- are named together, at most
    :data:`_MAX_NAMED` of them, and the pattern is stated once. The most
    readings any window held is the largest over the variables named.
    """
    patterns: dict[tuple[tuple[str, str, int], ...], list[str]] = {}
    held: dict[tuple[str, str, int], int] = {}
    for variable in dict.fromkeys(entry.variable for entry in blank):
        mine = [entry for entry in blank if entry.variable == variable]
        key = tuple((entry.point, entry.kind, entry.needs) for entry in mine)
        patterns.setdefault(key, []).append(variable)
        for entry in mine:
            slot = (entry.point, entry.kind, entry.needs)
            held[slot] = max(held.get(slot, 0), entry.held)
    groups = []
    for key, variables in list(patterns.items())[:_MAX_NAMED]:
        named = ", ".join(variables[:_MAX_NAMED])
        if len(variables) > _MAX_NAMED:
            named += f" and {len(variables) - _MAX_NAMED} more"
        points = "; ".join(
            _describe_blank(point, kind, needs, held[(point, kind, needs)])
            for point, kind, needs in key
        )
        groups.append(f"{named}: {points}")
    if len(patterns) > _MAX_NAMED:
        groups.append(f"and {len(patterns) - _MAX_NAMED} more pattern(s)")
    logger.warning(
        "Rolling state of '%s': sweep points blank at every reading -- %s. A window "
        "holding fewer readings than the count rule asks (METHODS 6.4), or a reading at "
        "least twice as wide as the window (6.3), leaves that point blank with the "
        "reason recorded; choose a longer window, a lower count, or another baseline "
        "method (6.5) for these variables.",
        instrument,
        " | ".join(groups),
    )


def _describe_blank(point: str, kind: str, needs: int, held: int) -> str:
    """Return one blank sweep point and the rule behind it, in a few words."""
    if kind == "count":
        noun = "reading" if held == 1 else "readings"
        return f"{point} (windows held at most {held} {noun}; {needs} needed)"
    if kind == "width":
        return f"{point} (every window held a reading at least {COPY_RATIO:g}x as wide as itself)"
    if kind == "adopted":
        return f"{point} (the adopted baseline is blank there)"
    return f"{point} (too few readings, or a reading too wide, in every window)"
