"""Interpolating smooth fields, which gases are not.

The one exception to the rule
------------------------------
``docs/METHODS.md`` §1.2 states TSARA's central prohibition: **quantified
species are never interpolated, only bin-averaged.** A concentration inside a
plume is not a smooth field, and drawing a straight line through it invents
structure exactly where the science happens.

Auxiliary fields are the exception, and a narrow one. Platform position and
ambient meteorology vary smoothly on sampling timescales, so evaluating them
between samples is physically justified — but only across gaps short enough
that the smoothness assumption survives. That is what
``AlignmentConfig.max_interp_gap`` bounds, and this module is the only place
in TSARA that interpolates anything.

Two guards, both refusals
--------------------------
**What.** A variable declaring ``role: gas`` is refused outright. So is one
declaring no role at all: the prohibition is worth more than the convenience
of guessing, and every stream TSARA produces carries roles.

**How far.** A target instant whose bracketing samples are further apart than
``max_interp_gap`` gets ``nan``, not a bridged value. Nothing is extrapolated
past the ends of a record either. Both counts are reported, because "there
was no position here" and "the position here was too uncertain to state" are
different facts about a cell.

Where a value is placed
------------------------
At the **cell midpoint**. A cell describes an interval, and for a smooth
field under a linear model its midpoint is the representative instant —
identical to the cell mean for a straight, constant-speed path and different
only where the path curves. Interpolation rather than binning is also what
makes a *sparse* auxiliary record usable: a GPS fix every 10 s cannot fill 1 s
cells by averaging, since most cells would contain no fix at all.

The spatial extent of a cell is deliberately not modelled. A 15 s canister at
15 m/s covers about 225 m, and TSARA reports the midpoint position with that
limit recorded (§10.10) rather than inventing a path length it has no
information for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from tsara.align.binning import TsaraAlignError, resolve_variable, stream_cells
from tsara.core.circular import wrap_degrees
from tsara.core.naming import (
    ALTITUDE_COORD,
    LATITUDE_COORD,
    LONGITUDE_COORD,
    TIME_COORD,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from tsara.core.support import CellBounds

logger = logging.getLogger(__name__)

__all__ = [
    "InterpolatedField",
    "attach_positions",
    "interpolate_onto_cells",
]

#: Roles this module refuses to interpolate, and why in one word.
#:
#: ``gas`` is the prohibition of §1.2. A variable with no role at all is
#: refused for the same reason rather than admitted on the assumption that
#: someone would have said so: TSARA's own streams always carry a role, so an
#: absent one means the dataset came from somewhere else and its smoothness is
#: unknown.
REFUSED_ROLES = frozenset({"gas"})

#: Attrs an interpolated field carries, documented in ``docs/METHODS.md`` §11.6.
INTERPOLATED_FROM_ATTR = "tsara_interpolated_from"
INTERP_GAP_ATTR = "tsara_max_interp_gap"
INTERP_MASKED_ATTR = "tsara_interp_gap_masked"
INTERP_OUTSIDE_ATTR = "tsara_interp_outside_record"


@dataclass(frozen=True)
class InterpolatedField:
    """One auxiliary field evaluated at a set of cell midpoints.

    Attributes
    ----------
    values : numpy.ndarray
        The field at each target cell's midpoint, ``nan`` where the guard
        refused to bridge or the target lies outside the source record.
    n_exact : int
        Targets that landed exactly on a source sample. Those are measured
        rather than interpolated, so the gap guard does not apply to them.
    n_interpolated : int
        Targets whose value was interpolated between two samples.
    n_gap_masked : int
        Targets refused because their bracketing samples were further apart
        than ``max_interp_gap``.
    n_outside : int
        Targets before the first or after the last sample. Never
        extrapolated.
    """

    values: np.ndarray
    n_exact: int
    n_interpolated: int
    n_gap_masked: int
    n_outside: int


def _as_seconds(duration: str | pd.Timedelta) -> float:
    """Return a duration in seconds, or raise naming what was given.

    Positivity is tested by comparing ``Timedelta`` objects rather than by
    reading seconds off the result, which matches
    :func:`tsara.config.base.validate_positive_timedelta` and avoids a real
    trap: ``pd.Timedelta("1ns").total_seconds()`` is ``0.0``, so *any*
    sub-microsecond duration would read as non-positive and be rejected.
    TSARA works in integer nanoseconds everywhere else for the same family of
    reasons.
    """
    try:
        delta = pd.Timedelta(duration)
    except (ValueError, TypeError) as error:
        raise TsaraAlignError(f"Could not read '{duration}' as a duration.") from error
    if delta <= pd.Timedelta(0):
        raise TsaraAlignError(
            f"max_interp_gap must be positive, got {duration}. Use a positive "
            "duration, or bin the field instead of interpolating it."
        )
    return float(delta.value) / 1e9


def _check_role(stream: xr.Dataset, variable: str, instrument: str) -> None:
    """Refuse to interpolate anything that is not declared smooth."""
    role = stream[variable].attrs.get("role")
    if role is None:
        raise TsaraAlignError(
            f"'{variable}' on '{instrument}' declares no role, so TSARA cannot tell "
            "whether interpolating it is physically justified. Only auxiliary "
            "fields may be interpolated (METHODS §1.2); declare a role, or bin the "
            "variable instead."
        )
    if str(role) in REFUSED_ROLES:
        raise TsaraAlignError(
            f"'{variable}' on '{instrument}' has role '{role}' and will not be "
            "interpolated. A concentration inside a plume is not a smooth field, "
            "and interpolating it invents structure exactly where the science is "
            "(METHODS §1.2). Bin it onto the cells instead."
        )


def _interpolate(
    source_ns: np.ndarray,
    values: np.ndarray,
    target_ns: np.ndarray,
    max_gap_s: float,
    *,
    circular: bool,
) -> InterpolatedField:
    """Evaluate a field at ``target_ns``, refusing to bridge long gaps."""
    finite = np.isfinite(values)
    source_ns, values = source_ns[finite], values[finite]
    if source_ns.size == 0:
        return InterpolatedField(
            values=np.full(target_ns.size, np.nan),
            n_exact=0,
            n_interpolated=0,
            n_gap_masked=0,
            n_outside=int(target_ns.size),
        )
    order = np.argsort(source_ns, kind="stable")
    source_ns, values = source_ns[order], values[order]

    # Epoch nanoseconds do not fit in a float64 mantissa: at 2024 epochs the
    # spacing between representable values is about 378 ns, so converting
    # directly would quantise every timestamp onto a coarse grid. Subtracting
    # a reference first keeps the arithmetic exact.
    reference = int(source_ns[0])
    source_s = (source_ns - reference).astype(np.float64) / 1e9
    target_s = (target_ns - reference).astype(np.float64) / 1e9

    if circular:
        radians = np.radians(values)
        sin_part = np.interp(target_s, source_s, np.sin(radians))
        cos_part = np.interp(target_s, source_s, np.cos(radians))
        interpolated = wrap_degrees(np.degrees(np.arctan2(sin_part, cos_part)))
    else:
        interpolated = np.interp(target_s, source_s, values)

    # Where each target sits among the sources. `right` is the first sample at
    # or after it, so `right - 1` is the last one before.
    right = np.searchsorted(source_ns, target_ns, side="left")
    exact = (right < source_ns.size) & (
        source_ns[np.minimum(right, source_ns.size - 1)] == target_ns
    )
    outside = ((target_ns < source_ns[0]) | (target_ns > source_ns[-1])) & ~exact
    left = np.clip(right - 1, 0, source_ns.size - 1)
    right_clipped = np.clip(right, 0, source_ns.size - 1)
    gap_ns = (source_ns[right_clipped] - source_ns[left]).astype(np.float64)
    too_far = (gap_ns / 1e9 > max_gap_s) & ~exact & ~outside

    result = np.asarray(interpolated, dtype=np.float64)
    result[outside | too_far] = np.nan
    return InterpolatedField(
        values=result,
        n_exact=int(np.count_nonzero(exact)),
        n_interpolated=int(np.count_nonzero(~exact & ~outside & ~too_far)),
        n_gap_masked=int(np.count_nonzero(too_far)),
        n_outside=int(np.count_nonzero(outside)),
    )


def interpolate_onto_cells(
    streams: Mapping[str, xr.Dataset],
    variable: str | tuple[str, str],
    target: CellBounds,
    *,
    max_interp_gap: str | pd.Timedelta = "10s",
) -> InterpolatedField:
    """Evaluate one smooth auxiliary field at a set of cell midpoints.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        A campaign's streams.
    variable : str or tuple of (str, str)
        Which field. A bare name must be unambiguous across the streams.
    target : CellBounds
        The cells whose midpoints the field is wanted at.
    max_interp_gap : str or pandas.Timedelta, optional
        Longest gap between bracketing samples that may be bridged.
        Config: ``AlignmentConfig.max_interp_gap``.

    Returns
    -------
    InterpolatedField
        The values, and how each one was arrived at.

    Raises
    ------
    TsaraAlignError
        If the variable is a gas, declares no role, cannot be resolved, or
        the gap is not a positive duration.
    """
    instrument, name = resolve_variable(streams, variable)
    stream = streams[instrument]
    _check_role(stream, name, instrument)
    max_gap_s = _as_seconds(max_interp_gap)
    source = stream_cells(stream, instrument)
    circular = str(stream[name].attrs.get("circular", 0)) not in ("0", "False", "None", "")
    result = _interpolate(
        source.midpoint_ns,
        np.asarray(stream[name].values, dtype=np.float64),
        target.midpoint_ns,
        max_gap_s,
        circular=circular,
    )
    if result.n_gap_masked or result.n_outside:
        logger.info(
            "Interpolated %s from '%s' onto %d cells: %d exact, %d interpolated, "
            "%d refused as gaps longer than %s, %d outside the record.",
            name,
            instrument,
            len(target),
            result.n_exact,
            result.n_interpolated,
            result.n_gap_masked,
            max_interp_gap,
            result.n_outside,
        )
    return result


def attach_positions(
    stream: xr.Dataset,
    streams: Mapping[str, xr.Dataset],
    *,
    max_interp_gap: str | pd.Timedelta = "10s",
) -> xr.Dataset:
    """Put a mobile platform's track onto one instrument's cells.

    The join ingestion deliberately left undone. A stationary site has one
    position, so attaching it is exact and free and ingestion does it. A
    moving platform's position lives on the GPS instrument's clock, and
    putting it on a gas instrument's clock is *interpolation* — permitted for
    a smooth field, but only under the guard that lives in
    :class:`~tsara.config.analysis.AlignmentConfig`, which ingestion has no
    business reading. So ingestion recorded the binding in attributes and
    left the join to here, which keeps the interpolation rule enforced in
    exactly one place.

    The resulting coordinates are named exactly as a stationary platform's
    are, so downstream code reads position identically and only has to care
    about the shape.

    Parameters
    ----------
    stream : xarray.Dataset
        The instrument stream to attach positions to.
    streams : mapping of str to xarray.Dataset
        The campaign, which must contain the GPS instrument the stream's
        attributes name.
    max_interp_gap : str or pandas.Timedelta, optional
        Longest GPS gap that may be bridged.

    Returns
    -------
    xarray.Dataset
        A copy carrying ``latitude`` and ``longitude`` coordinates on its own
        time axis, plus ``altitude`` where the platform declared one. A
        stream that already has positions — a stationary site, or one joined
        earlier — is returned unchanged.

    Raises
    ------
    TsaraAlignError
        If the stream names a GPS instrument the campaign does not contain,
        or the track crosses the antimeridian.
    """
    if LATITUDE_COORD in stream.coords:
        return stream
    gps_name = stream.attrs.get("platform_gps_instrument")
    if gps_name is None:
        logger.debug("Stream has no mobile-platform binding, so there is no track to attach.")
        return stream
    if gps_name not in streams:
        raise TsaraAlignError(
            f"This stream names '{gps_name}' as its GPS instrument, but the campaign "
            f"holds {sorted(map(str, streams))}. A mobile platform's track cannot be "
            "attached without it."
        )
    target = stream_cells(stream, str(stream.attrs.get("instrument", "<stream>")))
    joined = stream.copy()
    wanted = [
        (LATITUDE_COORD, stream.attrs.get("platform_lat_variable")),
        (LONGITUDE_COORD, stream.attrs.get("platform_lon_variable")),
        (ALTITUDE_COORD, stream.attrs.get("platform_alt_variable")),
    ]
    for coord, source_name in wanted:
        if source_name is None:
            continue
        if coord == LONGITUDE_COORD:
            _refuse_antimeridian(streams[gps_name], str(source_name), gps_name)
        result = interpolate_onto_cells(
            streams, (gps_name, str(source_name)), target, max_interp_gap=max_interp_gap
        )
        joined.coords[coord] = (
            TIME_COORD,
            result.values,
            {
                **dict(streams[gps_name][str(source_name)].attrs),
                INTERPOLATED_FROM_ATTR: f"{gps_name}.{source_name}",
                INTERP_GAP_ATTR: str(max_interp_gap),
                INTERP_MASKED_ATTR: result.n_gap_masked,
                INTERP_OUTSIDE_ATTR: result.n_outside,
            },
        )
    return joined


def _refuse_antimeridian(gps: xr.Dataset, variable: str, instrument: str) -> None:
    """Refuse a track that wraps at 180 degrees rather than interpolating it wrong.

    Longitude is interpolated linearly, which is right everywhere except
    across the antimeridian, where a step from 179.9 to -179.9 would be read
    as a journey most of the way round the planet. TSARA does not model that
    case; it detects it and says so, because a silently wrong position cannot
    be recovered downstream. The alternative — treating longitude as circular
    always — would put a sine and cosine round trip into every position on
    Earth to serve a campaign none of the target archive contains.
    """
    values = np.asarray(gps[variable].values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.ptp(finite)) > 180.0:
        raise TsaraAlignError(
            f"'{variable}' on '{instrument}' spans {float(np.ptp(finite)):.1f} degrees "
            "of longitude, which means the track crosses the antimeridian. Linear "
            "interpolation would read that step as a journey round the planet, and "
            "TSARA does not model the wrap (METHODS §11.6). Split the record at the "
            "crossing, or shift the longitudes to a continuous branch."
        )
