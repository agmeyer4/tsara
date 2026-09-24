"""Pairing two species measured on different clocks.

A thin layer, deliberately
---------------------------
Pairing is not its own operation. It is
:func:`~tsara.align.binning.bin_streams_onto_cells` with three decisions
layered on top:

1. **which cells** -- those of the wider-supported of the two streams, or on
   a tie, of the one with fewer measured values;
2. **which stretch** -- optionally restricted to an event or window;
3. **which rows survive** -- a pair needs a real measurement of *both*
   species, so a cell missing either is dropped.

Everything else, including the averaging, the uncertainty propagation and the
counts and coverage that qualify each value, belongs to the binner and is
shared with the campaign-wide grid. Writing pairing as its own implementation
was a design inversion: the two-species case is the general case with two
variables selected, and building the special one first would have left two
implementations of one idea to drift apart.

Why *wider*, not *slower*
--------------------------
Before Phase 3.5 the rule in ``docs/METHODS.md`` §1.3 said "the slower
instrument", and rate and support can disagree. Measured on the 2024 drives,
the iWAS canisters sample every 530 s but each sample integrates for only
14.9 s -- medians over all 261 fills of the ten 2024 drive days. Against a 60 s
stationary mean the canister is about nine times slower by rate (530 s against
60 s) and four times *narrower* by support (14.9 s against 60 s). Pairing on
the canister's clock would evaluate a 60 s mean over 15 s, which is exactly
what the interval model forbids; pairing on the mean's clock is admissible,
and the coverage of 0.25 is what says how much to trust it.

So the direction is always the same: a value may be averaged onto a wider
support, never copied onto a narrower one. Where a reading must still be
narrowed or straddled -- a clock whose cells vary in width, a partner half a
cell out of phase -- the product records how much of each value was borrowed
from beyond its cell and says so (``docs/METHODS.md`` §11.2.4), and a caller
who wants different cells altogether passes ``target=``.

When the widths tie
--------------------
Width cannot decide between two 1 s instruments, and in a 1 s drive suite
that is the ordinary case. It matters when the two sets of cells are out of
phase -- one file labelling each second by its start and another by its
middle is enough -- because every cell of one then straddles two of the
other. If the straddling instrument is also the sparser one, each of its
readings lands in two pairs.

Measured on the 2024-07-18 drive, NOy-LIF (``Time_Mid``) against the Picarro
(``Time_Start``, a CO2 value in every second or third row of a 1 s file):
on the LIF clock 8,447 CO2 readings became 16,893 pairs, on the Picarro's
8,447. The slope is the same either way, but the reading's error is counted
twice, and a naive standard error comes out up to 1/sqrt(2) too narrow
(§11.4.1). This module used to break the tie by argument order, so which of
the two you got depended on which species you named first.

So a tie goes to the member with **fewer measured values where the two
records overlap**: each of its readings is then one pair, and the denser
member is averaged across it. Where even that ties, the instrument names
decide, so the answer never depends on argument order.

Why a pair-specific clock exists at all
----------------------------------------
A campaign-wide grid's period is set by the *worst* instrument in it; a pair's
clock by the worse of *that pair*. With 60 s stationary means in the archive,
every 1 Hz-against-1 Hz ratio computed off a single grid would come from sixty
times fewer points -- roughly eight times the confidence interval -- with every
sub-minute plume flattened before the fit saw it. The grid is what makes a
whole campaign comparable; the pair clock is what keeps a fast pair from being
punished for a slow instrument elsewhere. Both are wanted, and they are the
same code.

What N means afterwards
------------------------
Every returned pair contains at least one real measurement of each species.
That is the whole reason gases are binned rather than interpolated:
interpolated points pose as independent samples and silently inflate the
degrees of freedom of every fit downstream (§1.2).

It does not follow that every pair is *independent*. A sparse member whose
cells straddle the clock's boundaries puts one reading into two pairs, which
choosing the clock well avoids on a tie but cannot always avoid -- a 15 s
canister fill across a minute boundary of a 60 s mean is the standing
example. So the product records, per species, how many distinct readings
stand behind its pairs, and warns when that is fewer than the pairs
themselves. That count, not the number of pairs, is a ceiling on a fit's N.

A ceiling, not an estimate. Two *dense* instruments half a cell apart
duplicate no reading on either clock, yet each reading of the averaged member
is shared between two neighbouring pairs, and a naive standard error is then
too narrow by up to 1/sqrt(2) whichever clock is chosen (§11.4.1). Neither the
readings count nor the borrowed share can see that. A *sparse* partner half a
cell off blends the same way (borrowed share 0.5), but its pairs sit two or
three cells apart, no reading reaches two of them, and the pairs are
independent -- §11.4.1 has the measurement. So the product counts the thing
itself, per species, in ``tsara_shared_readings``: the distinct readings that
formed more than one surviving pair. The blend is always said aloud; the
remedy is named only when that count is not zero, and the remedy is
``target=``: both members on a common clock five to ten times coarser, where
the naive standard error is honest again at no cost in real scatter. Modelling
the correlation from the overlap weights belongs to the regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import ValidationError

from tsara.align.binning import (
    BINNED_ATTR,
    BORROWED_ATTR,
    READINGS_ATTR,
    FinerSupport,
    bin_streams_onto_cells,
)
from tsara.align.cells import (
    median_width_s,
    phase_offset_s,
    readings_behind,
    shared_readings,
    stream_cells,
    targets_overlap,
)
from tsara.align.grid import grid_cells
from tsara.align.variables import TsaraAlignError, VariableRef, resolve_variable
from tsara.config.analysis import OutputGridConfig
from tsara.core.naming import TIME_COORD, coverage_name
from tsara.core.support import CellBounds

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import xarray as xr

    from tsara.core.propagation import PropagationForm

logger = logging.getLogger(__name__)

__all__ = ["PairedSpecies", "pair_species"]

#: Attrs the paired product adds on top of the binner's, documented in §11.4.
PAIRING_CLOCK_ATTR = "tsara_pairing_clock"
PAIRING_REASON_ATTR = "tsara_pairing_clock_reason"
PAIRING_COVERAGE_ATTR = "tsara_pairing_min_coverage"
PAIRING_DROPPED_ATTR = "tsara_pairing_cells_dropped"
PAIRING_CANDIDATE_ATTR = "tsara_pairing_cells_considered"
#: Per species: how many distinct readings formed more than one surviving
#: pair, so that neighbouring pairs share an error (§11.4.1). Zero for a
#: member on the clock, by construction.
SHARED_READINGS_ATTR = "tsara_shared_readings"
#: What ``tsara_pairing_clock`` and :attr:`PairedSpecies.clock` say when the
#: caller supplied the cells rather than letting the clock rule choose.
EXPLICIT_CLOCK = "explicit"


@dataclass(frozen=True)
class PairedSpecies:
    """Two species on one clock, with everything needed to fit them.

    Attributes
    ----------
    dataset : xarray.Dataset
        The pairs. One entry per surviving cell, carrying both species'
        values, whatever uncertainty components were available, and per
        species the contributing sample count and cell coverage. Cells are
        described by CF ``time_bnds`` exactly as a stream's are, so the
        product says what interval each pair describes.
    y_name, x_name : str
        Names of the two species as they appear in the dataset.
    clock : str
        Instrument whose cells were used: the wider-supported one, or on a
        tie the one with fewer measured values -- or ``'explicit'`` when the
        caller supplied ``target=`` and both species were averaged onto it.
    n_pairs : int
        How many pairs survived.
    y_readings, x_readings : int
        How many distinct readings of each species stand behind those pairs.
        Fewer readings than pairs means some reading appears in more than one
        pair, so the smaller of these and ``n_pairs`` is a ceiling on the
        number of independent points a fit really has -- a ceiling only,
        since readings shared between neighbouring pairs lower the
        independent information without lowering the count (§11.4.1). How
        many readings were shared that way is counted separately, in each
        species' ``tsara_shared_readings`` attribute.
    """

    dataset: xr.Dataset
    y_name: str
    x_name: str
    clock: str
    n_pairs: int
    y_readings: int
    x_readings: int

    def __len__(self) -> int:
        """Return the number of surviving pairs."""
        return self.n_pairs


def _restrict(cells: CellBounds, interval: tuple[pd.Timestamp, pd.Timestamp]) -> CellBounds:
    """Return the cells overlapping ``interval``.

    Overlap rather than containment: an event boundary rarely falls on a cell
    edge, and dropping the two partly-covered end cells would quietly shorten
    every event by up to one cell on each side.
    """
    # The window's edges in integer nanoseconds, the unit cells are stored in.
    start, stop = (int(pd.Timestamp(edge).value) for edge in interval)
    if stop <= start:
        raise TsaraAlignError(
            f"Pairing interval must have positive duration; got {interval[0]} to {interval[1]}."
        )
    # A cell overlaps [start, stop) by a positive amount exactly when it ends
    # after the window starts and starts before the window ends. Whole cells are
    # kept, never clipped: a clipped cell would describe a different interval.
    keep = np.flatnonzero((cells.stop_ns > start) & (cells.start_ns < stop))
    return CellBounds(start_ns=cells.start_ns[keep], stop_ns=cells.stop_ns[keep])


def _measured_where_records_overlap(
    members: tuple[tuple[xr.Dataset, str, CellBounds], tuple[xr.Dataset, str, CellBounds]],
) -> tuple[int, int]:
    """Return how many finite values each member has where both records run.

    Counted over the *shared* span rather than each whole record, because the
    question is which member is sparser in the air the two actually have in
    common. A sparse analyzer logging all day beside a dense one switched on
    for ten minutes has more readings in total and fewer where it matters.
    Over the whole record rather than an event's ``interval``, so that one
    pair of instruments keeps one clock from event to event.
    """
    # The span both records cover: the later of the two starts to the earlier end.
    span_start = max(int(cells.start_ns.min()) for _, _, cells in members)
    span_stop = min(int(cells.stop_ns.max()) for _, _, cells in members)
    counts = []
    for stream, variable, cells in members:
        # Cells inside the shared span that hold a value.
        inside = (cells.stop_ns > span_start) & (cells.start_ns < span_stop)
        finite = np.isfinite(np.asarray(stream[variable].values, dtype=np.float64))
        counts.append(int(np.count_nonzero(inside & finite)))
    return counts[0], counts[1]


def _choose_clock(
    y: tuple[str, str, CellBounds],
    x: tuple[str, str, CellBounds],
    streams: Mapping[str, xr.Dataset],
) -> tuple[str, str]:
    """Return the instrument whose cells a pair sits on, and why.

    In order: a shared instrument is its own clock; otherwise the wider cells;
    on a tie in width, the member with fewer measured values where the
    records overlap; on a tie in that too, the first instrument by name. No
    step looks at which species was named first, so swapping the arguments
    can never change which air is compared.
    """
    (y_instrument, y_variable, y_cells), (x_instrument, x_variable, x_cells) = y, x
    # Rule 1: two gases from one instrument already share cells.
    if y_instrument == x_instrument:
        return y_instrument, "both species share one instrument"
    # Rule 2: the wider cells, so nothing is copied onto a finer support.
    y_width, x_width = median_width_s(y_cells), median_width_s(x_cells)
    if y_width != x_width:
        wider, narrower = (y_width, x_width) if y_width > x_width else (x_width, y_width)
        clock = y_instrument if y_width > x_width else x_instrument
        return clock, f"wider cells ({wider:.6g} s vs {narrower:.6g} s)"
    # Rule 3, a tie in width: the member with fewer values where the records
    # overlap, so each of its readings becomes exactly one pair.
    y_count, x_count = _measured_where_records_overlap(
        (
            (streams[y_instrument], y_variable, y_cells),
            (streams[x_instrument], x_variable, x_cells),
        )
    )
    if y_count != x_count:
        clock = y_instrument if y_count < x_count else x_instrument
        fewer, more = sorted((y_count, x_count))
        return clock, (
            f"equal cells ({y_width:.6g} s); fewer measured values where the records "
            f"overlap ({fewer} vs {more})"
        )
    # Rule 4, a tie in that too: alphabetical, so argument order never decides.
    return min(y_instrument, x_instrument), (
        f"equal cells ({y_width:.6g} s) and equal measured values ({y_count}); "
        "first instrument by name"
    )


def _readings_behind(
    joined: xr.Dataset,
    column: str,
    stream: xr.Dataset,
    variable: str,
    readings: CellBounds,
    pairs: CellBounds,
) -> int:
    """Return how many distinct readings of one species the pairs draw on.

    A member already on the clock contributes exactly one reading per pair by
    construction. A binned member contributes every finite reading that
    overlaps a surviving pair by a positive amount -- the same membership rule
    the binner used to form the value, so the count describes the number that
    was actually reported.
    """
    if joined[column].attrs.get(BINNED_ATTR) == 0:
        # On the clock itself: one reading per surviving pair, by construction.
        return len(pairs)
    # Averaged onto the clock: count the distinct readings the pairs overlap.
    return readings_behind(stream, variable, readings, pairs)


def _shared_readings(
    joined: xr.Dataset,
    column: str,
    stream: xr.Dataset,
    variable: str,
    readings: CellBounds,
    pairs: CellBounds,
) -> int:
    """Return how many distinct readings of one species formed more than one pair.

    A member already on the clock forms exactly one pair per reading by
    construction, so none of its readings is shared. A binned member's count
    comes from the same overlap search that counted the readings behind the
    pairs, restricted to the pairs that survived.
    """
    if joined[column].attrs.get(BINNED_ATTR) == 0:
        return 0
    return shared_readings(stream, variable, readings, pairs)


def _explicit_target(
    streams: Mapping[str, xr.Dataset],
    target: CellBounds | str,
    selection: list[tuple[str, str]],
    finer_support: FinerSupport,
) -> tuple[CellBounds, str, str]:
    """Return the cells a caller asked to pair on, and how to describe that choice.

    Explicit cells are taken as given. A period string becomes a uniform grid
    over the two records through the grid builder, so it is anchored,
    windowed and checked against the copy rule exactly as a campaign grid is.
    """
    if isinstance(target, CellBounds):
        if len(target) == 0:
            raise TsaraAlignError("target has no cells to pair on.")
        return (
            target,
            EXPLICIT_CLOCK,
            f"explicit target cells ({len(target)} cells, median {median_width_s(target):.6g} s)",
        )
    try:
        config = OutputGridConfig(freq=str(target))
    except ValidationError as error:
        raise TsaraAlignError(
            f"target must be a CellBounds or a period such as '10s'; got {target!r}."
        ) from error
    return (
        grid_cells(streams, config, selection, finer_support=finer_support),
        EXPLICIT_CLOCK,
        f"explicit target period {target}",
    )


def _warn_if_a_partner_blends(
    dataset: xr.Dataset,
    y: tuple[str, CellBounds, int],
    x: tuple[str, CellBounds, int],
    clock_cells: CellBounds,
) -> None:
    """Say so when a binned member's cells are as wide as the clock's but offset from them.

    The 2024 drive suite's shape on every pair (§11.4.1): each value of the
    averaged member blends the two readings straddling its cell, so it rests
    on air outside the cell by the borrowed share (0.5 at half a cell). That
    much follows from the geometry alone -- the same exact test the grid
    applies to its own cells (:func:`~tsara.align.cells.phase_offset_s`)
    -- and is always said.

    What does not follow from the geometry is whether the pairs are
    independent. A *dense* member lends each reading to two neighbouring
    pairs, so neighbouring pairs share an error and a fit treating them as
    independent has a standard error too narrow by up to 1/sqrt(2); the
    remedy is a coarser common clock, which the message then names. A
    *sparse* member's pairs sit cells apart, no reading reaches two of them,
    and naming the remedy would send the caller to throw away resolution for
    nothing. So that clause is gated on the exact count of readings that
    formed more than one surviving pair
    (:func:`~tsara.align.cells.shared_readings`), never on the geometry.
    """
    for name, member_cells, shared in (y, x):
        if dataset[name].attrs.get(BINNED_ATTR) != 1:
            continue
        offset_s = phase_offset_s(member_cells, clock_cells)
        if offset_s is None:
            continue
        width_s = median_width_s(clock_cells)
        if shared > 0:
            consequence = (
                f"{shared} of its readings formed two neighbouring pairs, so neighbouring "
                "pairs share an error and a fit treating the pairs as independent has a "
                "standard error too narrow by up to 1/sqrt(2) (METHODS §11.4.1). Pair both "
                f"species on a coarser common clock with target=, e.g. target='{10 * width_s:g}s'."
            )
        else:
            consequence = (
                f"No reading formed more than one of the {dataset.sizes[TIME_COORD]} pairs, so "
                "the pairs are independent and no coarser clock is needed (METHODS §11.4.1)."
            )
        logger.warning(
            "Every pair blends two readings of '%s': its cells are as wide as the clock's "
            "(%.6g s) but offset by %.6g s, so each value blends the two readings straddling "
            "its cell and rests on air outside it by the borrowed share (%.2f). %s",
            name,
            width_s,
            offset_s,
            float(dataset[name].attrs[BORROWED_ATTR]),
            consequence,
        )


def pair_species(
    streams: Mapping[str, xr.Dataset],
    y: VariableRef,
    x: VariableRef,
    *,
    interval: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    target: CellBounds | str | None = None,
    min_coverage: float = 0.0,
    propagation_form: PropagationForm = "ar1_neff",
    finer_support: FinerSupport = "refuse",
) -> PairedSpecies:
    """Put two species on one clock, dropping any pair that is not real.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        A campaign's native-rate streams.
    y, x : str or tuple of (str, str)
        The two species. A bare canonical name is looked up across the
        streams and must be unambiguous; a tuple names the instrument.
    interval : tuple of pandas.Timestamp, optional
        Restrict pairing to cells overlapping this window -- an event, in
        Phase 6. ``None`` pairs the whole record.
    target : CellBounds or str, optional
        Cells to pair on instead of the wider-supported member's own: explicit
        cells, or a period such as ``'10s'`` for a uniform grid over the two
        records (built by :func:`~tsara.align.grid.grid_cells`, under the same
        copy rule). Both species are then averaged onto it. The measured use
        (§11.4.1): two dense equal-width instruments half a cell apart share
        every reading between neighbouring pairs, and a naive standard error
        is up to 1/sqrt(2) too narrow on *either* clock; on a common clock five
        to ten times coarser it is honest again while the real scatter is
        unchanged. ``None`` uses the clock rule.
    min_coverage : float, optional
        Drop pairs whose cell was covered by less than this fraction of
        contributing data. Default 0.0 drops nothing; coverage is recorded
        either way (:class:`~tsara.config.analysis.PairingConfig`).
    propagation_form : {'ar1_neff', 'ar1_asymptotic', 'ar1_double_sum'}, optional
        Which registered form reduces a correlated random component (§3.4).
    finer_support : {'refuse', 'allow'}, optional
        The clock is the wider-supported member's cells, so its partner is
        normally averaged, never copied. A clock whose cells vary in width --
        a canister's fills run from 1.8 s to 20 s on the 2024 drives -- can
        still hold one cell less than half as wide as a partner reading, and
        that pair is then refused (the default) or copied, labelled and warned
        about (``docs/METHODS.md`` §11.2.4).

    Returns
    -------
    PairedSpecies
        The pairs, and which stream's clock they are on.

    Raises
    ------
    TsaraAlignError
        If a species cannot be resolved, a stream has no cells, the interval
        selects nothing, no cell holds a real measurement of both species, or
        a partner reading would be copied across clock cells.
    """
    # Resolve both references, refusing a species paired with itself.
    if not streams:
        raise TsaraAlignError("No streams to pair.")
    y_instrument, y_variable = resolve_variable(streams, y)
    x_instrument, x_variable = resolve_variable(streams, x)
    if (y_instrument, y_variable) == (x_instrument, x_variable):
        raise TsaraAlignError(
            f"Cannot pair '{y_variable}' on '{y_instrument}' with itself; a ratio of "
            "a species to itself is 1 by construction."
        )

    # Decision 1, which cells: the clock rule and the target it selects, or
    # the cells the caller asked for.
    y_cells = stream_cells(streams[y_instrument], y_instrument)
    x_cells = stream_cells(streams[x_instrument], x_instrument)
    if target is None:
        clock, reason = _choose_clock(
            (y_instrument, y_variable, y_cells), (x_instrument, x_variable, x_cells), streams
        )
        cells = y_cells if clock == y_instrument else x_cells
    else:
        cells, clock, reason = _explicit_target(
            streams,
            target,
            [(y_instrument, y_variable), (x_instrument, x_variable)],
            finer_support,
        )
    target = cells

    # Decision 2, which stretch: optionally only the clock cells overlapping a window.
    if interval is not None:
        target = _restrict(target, interval)
        if len(target) == 0:
            raise TsaraAlignError(
                f"No cells of '{clock}' overlap the interval {interval[0]} to {interval[1]}."
            )
    n_considered = len(target)

    # The one operation: both species onto the chosen cells.
    joined = bin_streams_onto_cells(
        streams,
        target,
        [(y_instrument, y_variable), (x_instrument, x_variable)],
        propagation_form=propagation_form,
        finer_support=finer_support,
    )
    # The binner names a column after its variable, suffixing with the
    # instrument only when two selected streams claim the same name -- which
    # for a pair means one species compared across two analyzers.
    collides = y_variable == x_variable
    y_name = f"{y_variable}_{y_instrument}" if collides else y_variable
    x_name = f"{x_variable}_{x_instrument}" if collides else x_variable

    # Decision 3, which rows survive: both species measured, and each cell
    # covered at least `min_coverage` by each species' contributing data.
    surviving = np.flatnonzero(
        np.isfinite(joined[y_name].values)
        & np.isfinite(joined[x_name].values)
        & (joined[coverage_name(y_name)].values >= min_coverage)
        & (joined[coverage_name(x_name)].values >= min_coverage)
    )
    if surviving.size == 0:
        raise TsaraAlignError(
            f"'{y_name}' and '{x_name}' produced no usable pairs over {n_considered} "
            f"candidate cell(s) of '{clock}'. Either the records do not overlap, or "
            f"every candidate cell was masked or below min_coverage={min_coverage}."
        )
    # The binner has already said, over every candidate cell, which columns
    # hold values in more rows than they have readings (§11.2.4). Dropping
    # rows can only lower both counts, but not always in step: keep the two
    # rows a straddling reading fills and drop the row its neighbours shared,
    # and rows outnumber readings where before they did not. So the question
    # is asked again of the surviving pairs, and answered aloud only where the
    # binner had nothing to say.
    already_said = {
        name: not targets_overlap(target)
        and int(np.count_nonzero(np.isfinite(joined[name].values)))
        > int(joined[name].attrs[READINGS_ATTR])
        for name in (y_name, x_name)
    }
    dataset = joined.isel({TIME_COORD: surviving})
    # Count the readings behind the surviving pairs, per species, and warn when
    # fewer readings than pairs means some reading sits in more than one pair.
    kept = CellBounds(start_ns=target.start_ns[surviving], stop_ns=target.stop_ns[surviving])
    y_readings = _readings_behind(dataset, y_name, streams[y_instrument], y_variable, y_cells, kept)
    x_readings = _readings_behind(dataset, x_name, streams[x_instrument], x_variable, x_cells, kept)
    # The same attribute the binner wrote, now counted behind the rows that
    # survived: one name, one meaning, on every product.
    dataset[y_name].attrs[READINGS_ATTR] = y_readings
    dataset[x_name].attrs[READINGS_ATTR] = x_readings
    # And the independence question, asked exactly: how many readings formed
    # more than one surviving pair. It is what separates a dense partner half
    # a cell off (every reading in two pairs) from a sparse one (pairs cells
    # apart, no reading in two), which the borrowed share and the readings
    # count cannot tell apart (§11.4.1).
    y_shared = _shared_readings(dataset, y_name, streams[y_instrument], y_variable, y_cells, kept)
    x_shared = _shared_readings(dataset, x_name, streams[x_instrument], x_variable, x_cells, kept)
    dataset[y_name].attrs[SHARED_READINGS_ATTR] = y_shared
    dataset[x_name].attrs[SHARED_READINGS_ATTR] = x_shared
    if clock != EXPLICIT_CLOCK:
        # The clock rule chose these cells, so the caller has not seen what a
        # partner half a cell from them does; with an explicit target the
        # caller chose, and a period target was already checked by the grid.
        _warn_if_a_partner_blends(
            dataset, (y_name, y_cells, y_shared), (x_name, x_cells, x_shared), target
        )
    if min(y_readings, x_readings) < surviving.size:
        sparse_name, sparse_count = (
            (y_name, y_readings) if y_readings <= x_readings else (x_name, x_readings)
        )
    if min(y_readings, x_readings) < surviving.size and not already_said[sparse_name]:
        logger.warning(
            "%d pairs of %s vs %s rest on only %d distinct readings of %s, so some "
            "readings appear in more than one pair. A fit that treats the pairs as "
            "independent will count those readings' errors more than once and "
            "understate its uncertainty (METHODS §11.4.1).",
            surviving.size,
            y_name,
            x_name,
            sparse_count,
            sparse_name,
        )
    # Record every decision made, so the product explains itself.
    dataset.attrs.update(
        {
            "tsara_stage": "paired",
            PAIRING_CLOCK_ATTR: clock,
            PAIRING_REASON_ATTR: reason,
            PAIRING_COVERAGE_ATTR: float(min_coverage),
            PAIRING_CANDIDATE_ATTR: int(n_considered),
            PAIRING_DROPPED_ATTR: int(n_considered - surviving.size),
        }
    )
    logger.info(
        "Paired %s vs %s on '%s' (%s): %d of %d candidate cells kept.",
        y_name,
        x_name,
        clock,
        reason,
        surviving.size,
        n_considered,
    )
    return PairedSpecies(
        dataset=dataset,
        y_name=y_name,
        x_name=x_name,
        clock=clock,
        n_pairs=int(surviving.size),
        y_readings=y_readings,
        x_readings=x_readings,
    )
