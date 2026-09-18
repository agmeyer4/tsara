"""The uniform grid: one support for a whole campaign.

What it is for, and what it is not for
---------------------------------------
``docs/METHODS.md`` §1.4: a single uniform ``(time × variable)`` cube is built
**only** for the products that inherently need one — the continuous rolling
state, and the matrix a receptor model such as PMF consumes. Baselines,
detection and cross-species regression all run at native rate and never see
it (§1.1, "synchronize late").

It is not an alternative to pairing (§11.4) and the difference is resolution.
A grid's period is set by the **worst** instrument in it; a pair's clock by
the worse of *that pair*. With 60 s stationary means in a campaign, every
1 Hz-against-1 Hz ratio computed off a single grid would come from sixty times
fewer points, and every sub-minute plume would be flattened before the fit saw
it. The grid is what makes a whole campaign comparable; the pair clock is what
keeps a fast pair from being punished for a slow instrument elsewhere.

There is deliberately no "PMF matrix" object
---------------------------------------------
A receptor-model matrix is this function called with a chosen set of columns.
Which columns is a scientific decision — raw concentrations or the
enhancements §5 will compute, which species, whether met belongs in the same
cube — and hard-coding any of it would make the block need editing every time
that decision changed. Hence the ``variables`` argument, and hence a grid
period validated against *the variables being gridded* rather than against
every stream the campaign happens to contain.

The period rule
---------------
**No selected reading may be twice as wide as a grid cell it touches.** A
60 s mean evaluated on 1 s cells is the same value repeated sixty times:
resolution the instrument never had, and sixty points where there is one
measurement. That is the prohibition the whole interval model exists to
enforce (§10), so it is an error naming the offending instrument and a period
that would work, not a warning -- unless the caller passes
``finer_support="allow"``, in which case the copies are made, labelled and
warned about (§11.2.4). The test is the binner's own
(:func:`~tsara.align.binning.pair_width_ratios`, at
:data:`~tsara.align.binning.COPY_RATIO`), run against the grid's actual cells
before anything is built, so the grid and the operation it calls can never
disagree about the same data.

It replaced two earlier rules: a comparison of median widths, which refused a
one-second grid for an analyzer whose cells measure 1.023 s -- a hair's worth
of jitter the binner itself accepts -- and a sum of each reading's shared time
over all cells, which was right for a uniform grid and wrong for the
overlapping targets §5 and §6 use.

Which variables are selected is therefore a real lever: a run that excludes
the canisters can be gridded far finer than one that includes them.

What the rule does not refuse is recorded instead
--------------------------------------------------
Below that line a reading may still be wider than a cell (narrowed: the
canisters' longest fills reach 1.34 cells on a 15 s grid) or land in more
than one row (shared: a 15 s fill crossing a minute boundary is one
measurement in two rows of a 60 s grid, and on the ten 2024 drive days that
happens to 68 of 261 fills). The binner records both on every column --
``tsara_support_transform``, ``tsara_width_ratio_max``,
``tsara_borrowed_share``, ``tsara_readings`` and a ``borrowed_<name>`` column
per cell -- and warns once per grid naming the columns affected, because a
receptor model treating rows as independent observations counts a shared
reading more than once and a narrowed value stands for time its instrument
never resolved.

A uniform grid spans the gaps
------------------------------
The grid runs from the first selected cell to the last, so a campaign of
separate drives puts every hour between them into the matrix too. Measured:
one-second cells over the ten 2024 drive days span 29.5 days, 2.55 million
rows of which 92 % hold no data, about 95 MB per variable with its companion
columns. Grid each drive with ``start`` and ``end`` when the matrix is for a
receptor model rather than a continuous state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tsara.align.binning import (
    COPY_RATIO,
    FinerSupport,
    TsaraAlignError,
    VariableRef,
    bin_streams_onto_cells,
    pair_width_ratios,
    phase_offset_s,
    select_variables,
    stream_cells,
)
from tsara.core.support import CellBounds, overlap_pairs
from tsara.core.timebase import NS_PER_S, to_utc_naive_stamp

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import xarray as xr

    from tsara.config.analysis import OutputGridConfig
    from tsara.core.propagation import PropagationForm

logger = logging.getLogger(__name__)

__all__ = ["build_output_grid", "grid_cells"]

#: Attrs the gridded product carries, documented in ``docs/METHODS.md`` §11.7.
GRID_FREQ_ATTR = "tsara_grid_freq"
GRID_WIDEST_CELL_ATTR = "tsara_grid_widest_reading_cell_s"
GRID_VARIABLES_ATTR = "tsara_grid_variables"


def grid_cells(
    streams: Mapping[str, xr.Dataset],
    config: OutputGridConfig,
    variables: Sequence[VariableRef] | None = None,
    *,
    finer_support: FinerSupport = "refuse",
) -> CellBounds:
    """Build the uniform cells a campaign's output grid sits on.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    config : OutputGridConfig
        Grid period and optional explicit start and end.
    variables : sequence of str or (str, str), optional
        Which variables the grid is for. ``None`` means all of them, and the
        period rule is checked against all of them accordingly.
    finer_support : {'refuse', 'allow'}, optional
        Whether a period at which a selected reading would be copied across
        cells is refused (the default) or allowed, labelled and warned about
        (``docs/METHODS.md`` §11.2.4).

    Returns
    -------
    CellBounds
        Abutting cells of exactly ``config.freq``.

    Raises
    ------
    TsaraAlignError
        If the period is so fine that one selected reading would be twice as
        wide as a grid cell it touches (and ``finer_support`` is ``refuse``),
        if the requested window is empty, or if no variable was selected.

    Notes
    -----
    With no explicit ``start``, the grid begins at the largest whole multiple
    of the period at or before the earliest selected cell. Anchoring to the
    epoch rather than to whenever the data happened to start means two runs
    over overlapping periods produce cells that line up, which is what lets
    their outputs be compared at all.
    """
    # The same selection the binner will use, so the period is checked against
    # exactly the variables that will be gridded.
    selection = select_variables(streams, variables)
    if not selection:
        raise TsaraAlignError("No variables selected, so there is nothing to build a grid for.")
    period_ns = int(pd.Timedelta(config.freq).value)
    instruments = sorted({instrument for instrument, _ in selection})
    cells = {
        instrument: stream_cells(streams[instrument], instrument) for instrument in instruments
    }

    # The window: an explicit start and end, or the selected data's own extent.
    earliest = min(int(bounds.start_ns.min()) for bounds in cells.values())
    latest = max(int(bounds.stop_ns.max()) for bounds in cells.values())
    if config.start is not None:
        start_ns = int(to_utc_naive_stamp(pd.Timestamp(config.start)).value)
    else:
        # Floor division on a negative epoch still rounds toward minus
        # infinity in Python, which is what "the multiple at or before" means
        # on both sides of 1970.
        start_ns = (earliest // period_ns) * period_ns
    stop_ns = (
        int(to_utc_naive_stamp(pd.Timestamp(config.end)).value)
        if config.end is not None
        else latest
    )
    if stop_ns <= start_ns:
        raise TsaraAlignError(
            f"The requested grid window is empty: it starts at "
            f"{pd.Timestamp(start_ns)} and ends at {pd.Timestamp(stop_ns)}."
        )
    # Abutting cells of exactly one period, enough to reach the end of the window.
    n_cells = int(np.ceil((stop_ns - start_ns) / period_ns))
    edges = start_ns + np.arange(n_cells + 1, dtype=np.int64) * period_ns
    target = CellBounds(start_ns=edges[:-1], stop_ns=edges[1:])
    _warn_if_out_of_phase(cells, target)
    # The period rule, measured against these actual cells before anything is binned.
    _refuse_a_period_too_fine(cells, target, str(config.freq), finer_support)
    return target


def _refuse_a_period_too_fine(
    cells: Mapping[str, CellBounds], target: CellBounds, freq: str, finer_support: FinerSupport
) -> None:
    """Raise if any selected reading would be copied across grid cells.

    The binner's own ratio (:func:`~tsara.align.binning.pair_width_ratios`),
    run here against the grid's actual cells so the refusal can name a period
    that would work before anything is binned. Checked against the cells
    inside the requested window, so a wide instrument whose data all falls
    outside it constrains nothing. Under ``finer_support="allow"`` nothing is
    refused here; the binner makes the copies and says so.
    """
    if finer_support == "allow":
        return
    # Test every selected instrument; name the one with the widest reading.
    worst: tuple[float, float, str] | None = None
    for instrument, bounds in cells.items():
        pairs = overlap_pairs(bounds, target)
        touching = pairs.overlap_ns > 0
        if not touching.any():
            continue
        ratio = float(pair_width_ratios(bounds, target, pairs)[touching].max())
        widest_s = float(bounds.width_ns[pairs.reading_index[touching]].max()) / NS_PER_S
        if ratio >= COPY_RATIO and (worst is None or widest_s > worst[1]):
            worst = (ratio, widest_s, instrument)
    if worst is None:
        return
    ratio, widest_s, instrument = worst
    raise TsaraAlignError(
        f"Grid period {freq} is too fine for '{instrument}': one of its readings is "
        f"{ratio:.3g} times as wide as a grid cell. Evaluating it on a finer grid would "
        "copy one measurement across rows, which is resolution the instrument never had "
        f"(METHODS §11.7). Use a period longer than {widest_s / COPY_RATIO:.6g} s, leave "
        f"'{instrument}' out of the selection, or pass finer_support='allow' to copy it, "
        "labelled."
    )


def _warn_if_out_of_phase(cells: Mapping[str, CellBounds], target: CellBounds) -> None:
    """Note any stream whose cells are as wide as the grid but offset from it.

    A legitimate but easily-missed situation. When a stream's cells are the
    same width as the grid period and share its phase, each grid value comes
    from exactly one reading. When the phases differ, each grid value is a
    *blend* of two adjacent readings -- still an honest weighted mean, but
    a smoothed one, and a ratio taken across two columns treated differently
    that way carries a small bias.

    Measured on the acceptance campaign: a 60 s instrument whose cells are
    centred on its timestamps sits half a cell off an epoch-anchored 60 s
    grid, every grid value then draws from two readings, and the recovered
    ratio moves from 0.250000034 to 0.249940526. Aligning the grid start to
    that instrument's own boundaries removes it entirely.

    The ``n_readings`` and ``borrowed_`` columns already report the blend, so
    this is a note rather than a refusal: blending is sometimes unavoidable,
    and which instrument the grid should be in phase with is the user's
    choice. The test itself is shared with pairing
    (:func:`~tsara.align.binning.phase_offset_s`), which says the same thing
    about a partner half a cell from the clock.
    """
    for instrument, bounds in cells.items():
        offset_s = phase_offset_s(bounds, target)
        if offset_s is None:
            continue
        # In seconds, not the raw nanoseconds the arithmetic runs in: a
        # reader of the log is being asked to act on these two numbers by
        # moving the grid start, and "60000000000 ns" is a period nobody
        # recognises as one minute.
        logger.warning(
            "Stream '%s' has cells exactly as wide as the %.6g s grid period but "
            "offset from it by %.6g s, so every grid value will blend two of its "
            "cells (n_readings = 2) rather than reproduce one. Set the grid start "
            "to one of its cell boundaries to avoid the smoothing.",
            instrument,
            int(target.width_ns[0]) / NS_PER_S,
            offset_s,
        )


def build_output_grid(
    streams: Mapping[str, xr.Dataset],
    config: OutputGridConfig,
    variables: Sequence[VariableRef] | None = None,
    *,
    propagation_form: PropagationForm = "ar1_neff",
    finer_support: FinerSupport = "refuse",
) -> xr.Dataset:
    """Put a campaign's variables on one uniform support.

    A thin layer over :func:`~tsara.align.binning.bin_streams_onto_cells`: the
    only thing this adds is *which cells*, and the rule that the period has to
    respect the data going into it.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    config : OutputGridConfig
        Grid period and optional window.
    variables : sequence of str or (str, str), optional
        Which variables the grid holds. ``None`` takes every non-sigma
        variable in every stream.
    propagation_form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form reduces a correlated random component (§3.4).
    finer_support : {'refuse', 'allow'}, optional
        Whether a period at which a selected reading would be copied across
        cells is refused (the default) or allowed, labelled and warned about
        (``docs/METHODS.md`` §11.2.4).

    Returns
    -------
    xarray.Dataset
        The grid, with each variable's uncertainty, contributing count,
        coverage and borrowed share alongside it, its support record in its
        attrs (§11.2.4), and CF cell boundaries. Cells with no data are
        ``nan`` with a count of zero — never interpolated (§1.2).

    Raises
    ------
    TsaraAlignError
        If the period is so fine that one selected reading would be twice as
        wide as a grid cell it touches, or the selection or window is empty.
    """
    selection = select_variables(streams, variables)
    # Which cells (refusing a period too fine), then the one operation onto them.
    target = grid_cells(streams, config, variables, finer_support=finer_support)
    gridded = bin_streams_onto_cells(
        streams,
        target,
        selection,
        propagation_form=propagation_form,
        finer_support=finer_support,
    )
    # Provenance: what was gridded, at what period, against which widest cell.
    instruments = sorted({instrument for instrument, _ in selection})
    cells = {name: stream_cells(streams[name], name) for name in instruments}
    widest = max(float(bounds.width_ns.max()) for bounds in cells.values()) / NS_PER_S
    gridded.attrs["tsara_stage"] = "gridded"
    gridded.attrs[GRID_FREQ_ATTR] = str(config.freq)
    gridded.attrs[GRID_WIDEST_CELL_ATTR] = float(widest)
    # Recorded because a reader cannot tell from the columns alone whether a
    # variable is absent because it was excluded or because it had no data.
    gridded.attrs[GRID_VARIABLES_ATTR] = ", ".join(
        f"{instrument}.{name}" for instrument, name in selection
    )
    logger.info(
        "Built a %s grid of %d cells over %d variable(s); widest reading cell %.6g s.",
        config.freq,
        len(target),
        len(selection),
        widest,
    )
    return gridded
