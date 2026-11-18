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

So a tie goes to the variable with **fewer measured values where the two
records overlap**: each of its readings is then one pair, and the denser
variable is averaged across it. Where even that ties, the instrument names
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

It does not follow that every pair is *independent*. A sparse variable whose
cells straddle the clock's boundaries puts one reading into two pairs, which
choosing the clock well avoids on a tie but cannot always avoid -- a 15 s
canister fill across a minute boundary of a 60 s mean is the standing
example. So the product records, per species, how many distinct readings
stand behind its pairs, and warns when that is fewer than the pairs
themselves. That count, not the number of pairs, is a ceiling on a fit's N.

A ceiling, not an estimate. Two *dense* instruments half a cell apart
duplicate no reading on either clock, yet each reading of the averaged variable
is shared between two neighbouring pairs, and a naive standard error is then
too narrow by up to 1/sqrt(2) whichever clock is chosen (§11.4.1). Neither the
readings count nor the borrowed share can see that. A *sparse* partner half a
cell off blends the same way (borrowed share 0.5), but its pairs sit two or
three cells apart, no reading reaches two of them, and the pairs are
independent -- §11.4.1 has the measurement. So the product counts the thing
itself, per species, in ``tsara_shared_readings``: the distinct readings that
formed more than one surviving pair. The blend is always said aloud; the
remedy is named only when that count is not zero, and the remedy is
``target=``: both variables on a common clock five to ten times coarser, where
the naive standard error is honest again at no cost in real scatter. Modelling
the correlation from the overlap weights belongs to the regression.

Swept pairs, and variables carried beside a pair
--------------------------------------------------
The ratio stage pairs enhancements, and an enhancement carries the baseline's
sweep: ``(time, baseline_window, baseline_quantile)``, one version per sweep
point (``docs/METHODS.md`` §6.2, §5.2). The binner already averages every
version at once; what pairing adds is the pair rule, asked of each version:
a row is a pair *at a sweep point* when both variables hold a value there and
both are covered at least ``min_coverage``. A row is kept when it is a pair at
one sweep point or more, and at a sweep point where it is not, both variables'
values are blanked, so that every finite value in the product is half of a
real pair at its own sweep point, as for an unswept pair it always was. The
counts are per sweep point too, and each variable records the most any one
point holds, as the binner does for a column's readings. Both variables must
carry the same sweep dimensions: a pair compares one sweep point of each, and
a variable without the dimension has no version to compare.

The weights a fit uses are the readings' own random sigma, not the
enhancement's (§4.3), so the readings must reach the pair cells too. They are
*carried*: ``carry=`` names more variables, of either instrument or any other,
that are put on the same cells by the same join and kept on the surviving
rows. A carried variable never decides which rows survive, and its own blank
values stay blank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import ValidationError

from tsara.align.binning import (
    FinerSupport,
    bin_streams_onto_cells,
    output_names,
)
from tsara.align.cells import phase_offset_s, readings_behind, shared_readings
from tsara.align.grid import grid_cells
from tsara.align.variables import TsaraAlignError, VariableRef, resolve_variable
from tsara.config.analysis import OutputGridConfig
from tsara.core.naming import (
    BINNED_ATTR,
    BORROWED_ATTR,
    READINGS_ATTR,
    TIME_COORD,
    coverage_name,
)
from tsara.core.support import CellBounds, median_width_s, stream_cells, targets_overlap

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

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
#: variable on the clock, by construction.
SHARED_READINGS_ATTR = "tsara_shared_readings"
#: Swept pairs only: the pairs at each sweep point, in the order of the
#: product's sweep dimensions with the last varying fastest (§5.2).
PAIRS_PER_SWEEP_POINT_ATTR = "tsara_pairing_pairs_per_sweep_point"
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
        How many pairs survived: for a swept pair, how many rows are a pair
        at one sweep point or more (each point's count is in the product's
        ``tsara_pairing_pairs_per_sweep_point``).
    y_readings, x_readings : int
        How many distinct readings of each species stand behind those pairs.
        Fewer readings than pairs means some reading appears in more than one
        pair, so the smaller of these and ``n_pairs`` is a ceiling on the
        number of independent points a fit really has -- a ceiling only,
        since readings shared between neighbouring pairs lower the
        independent information without lowering the count (§11.4.1). How
        many readings were shared that way is counted separately, in each
        species' ``tsara_shared_readings`` attribute. For a swept pair, the
        most any one sweep point draws on.
    carried : tuple of str
        The columns of the variables named in ``carry=``, in that order.
    """

    dataset: xr.Dataset
    y_name: str
    x_name: str
    clock: str
    n_pairs: int
    y_readings: int
    x_readings: int
    carried: tuple[str, ...] = ()

    def __len__(self) -> int:
        """Return the number of surviving pairs."""
        return self.n_pairs


def pair_species(
    streams: Mapping[str, xr.Dataset],
    y: VariableRef,
    x: VariableRef,
    *,
    carry: Sequence[VariableRef] = (),
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
        streams and must be unambiguous; a tuple names the instrument. Either
        may carry sweep dimensions (an enhancement from a baseline state),
        and then both must carry the same ones: a row is a pair at each sweep
        point where both hold a value.
    carry : sequence of str or tuple of (str, str), optional
        More variables put on the same cells and kept on the surviving rows,
        such as the readings whose own sigma weighs a fit (METHODS §4.3). They
        never decide which rows survive. Their columns are named in
        :attr:`PairedSpecies.carried`.
    interval : tuple of pandas.Timestamp, optional
        Restrict pairing to cells overlapping this window, such as an
        event's interval from the event catalog. ``None`` pairs the whole
        record.
    target : CellBounds or str, optional
        Cells to pair on instead of the wider-supported variable's own: explicit
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
        The clock is the wider-supported variable's cells, so its partner is
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
        selects nothing, no cell holds a real measurement of both species, a
        partner reading would be copied across clock cells, the two variables
        carry different sweep dimensions, or a carried variable is one of the
        pair's variables or is named twice.
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
    # A pair compares one sweep point of each variable, so both must have the same
    # points; the binner then checks that their coordinates agree.
    y_sweep = _sweep_of(streams[y_instrument], y_variable)
    x_sweep = _sweep_of(streams[x_instrument], x_variable)
    if y_sweep != x_sweep:
        raise TsaraAlignError(
            f"'{y_instrument}.{y_variable}' carries sweep dimensions {dict(y_sweep)} and "
            f"'{x_instrument}.{x_variable}' {dict(x_sweep)}; a pair compares one sweep point "
            "of each variable, so both must carry the same ones."
        )
    pair = [(y_instrument, y_variable), (x_instrument, x_variable)]
    carried = [resolve_variable(streams, reference) for reference in carry]
    for reference in carried:
        if reference in pair:
            raise TsaraAlignError(
                f"'{reference[0]}.{reference[1]}' is one of the pair's variables and cannot also "
                "be carried beside it."
            )
    if len(set(carried)) != len(carried):
        raise TsaraAlignError(f"carry names a variable twice: {carried}.")
    selection = [*pair, *carried]

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
        cells, clock, reason = _explicit_target(streams, target, pair, finer_support)
    target = cells

    # Decision 2, which stretch: optionally only the clock cells overlapping a window.
    if interval is not None:
        target = _restrict(target, interval)
        if len(target) == 0:
            raise TsaraAlignError(
                f"No cells of '{clock}' overlap the interval {interval[0]} to {interval[1]}."
            )
    n_considered = len(target)

    # The one operation: both species, and anything carried, onto the chosen cells.
    joined = bin_streams_onto_cells(
        streams,
        target,
        selection,
        propagation_form=propagation_form,
        finer_support=finer_support,
    )
    # The binner names a column after its variable, suffixing with the
    # instrument only when two selected streams claim the same name -- for a
    # pair, one species compared across two analyzers -- and pairing finds its
    # columns by that same rule rather than a copy of it.
    names = output_names(selection)
    y_name, x_name = names[pair[0]], names[pair[1]]

    # Decision 3, which rows survive: both species measured, and each cell
    # covered at least `min_coverage` by each species' contributing data --
    # asked of every sweep point when the variables carry a sweep, and a row
    # kept when it is a pair at one point or more.
    paired = (
        _finite(joined[y_name])
        & _finite(joined[x_name])
        & (joined[coverage_name(y_name)] >= min_coverage)
        & (joined[coverage_name(x_name)] >= min_coverage)
    )
    sweep = [str(dim) for dim in paired.dims if dim != TIME_COORD]
    surviving = np.flatnonzero((paired.any(dim=sweep) if sweep else paired).values)
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
        and _most_finite_rows(joined, name) > int(joined[name].attrs[READINGS_ATTR])
        for name in (y_name, x_name)
    }
    dataset = joined.isel({TIME_COORD: surviving})
    kept_pairs = paired.isel({TIME_COORD: surviving})
    if sweep:
        # A kept row that is no pair at some sweep point holds neither variable
        # there, so that every finite value is half of a real pair at its point.
        for name in (y_name, x_name):
            dataset[name] = dataset[name].where(kept_pairs)
    # Count the readings behind the surviving pairs, per species and per sweep
    # point, and warn when fewer readings than pairs means some reading sits in
    # more than one pair.
    kept = CellBounds(start_ns=target.start_ns[surviving], stop_ns=target.stop_ns[surviving])
    counts = _counts_per_sweep_point(
        dataset,
        kept_pairs,
        sweep,
        streams,
        ((y_name, y_instrument, y_variable, y_cells), (x_name, x_instrument, x_variable, x_cells)),
        kept,
    )
    y_readings, x_readings = int(counts[:, 1].max()), int(counts[:, 2].max())
    # The same attribute the binner wrote, now counted behind the rows that
    # survived: one name, one meaning, on every product.
    dataset[y_name].attrs[READINGS_ATTR] = y_readings
    dataset[x_name].attrs[READINGS_ATTR] = x_readings
    # And the independence question, asked exactly: how many readings formed
    # more than one surviving pair. It is what separates a dense partner half
    # a cell off (every reading in two pairs) from a sparse one (pairs cells
    # apart, no reading in two), which the borrowed share and the readings
    # count cannot tell apart (§11.4.1).
    y_shared, x_shared = int(counts[:, 3].max()), int(counts[:, 4].max())
    dataset[y_name].attrs[SHARED_READINGS_ATTR] = y_shared
    dataset[x_name].attrs[SHARED_READINGS_ATTR] = x_shared
    if clock != EXPLICIT_CLOCK:
        # The clock rule chose these cells, so the caller has not seen what a
        # partner half a cell from them does; with an explicit target the
        # caller chose, and a period target was already checked by the grid.
        _warn_if_a_partner_blends(
            dataset, (y_name, y_cells, y_shared), (x_name, x_cells, x_shared), target
        )
    # The first sweep point (the only one, unswept) at which pairs outnumber
    # the readings of one variable: there a reading is in more than one pair.
    # One warning names one point; the counts at the others are the same
    # question asked of other baselines.
    short = np.flatnonzero(np.minimum(counts[:, 1], counts[:, 2]) < counts[:, 0])
    if short.size:
        n_pairs_there, y_there, x_there = (int(value) for value in counts[short[0], :3])
        sparse_name, sparse_count = (y_name, y_there) if y_there <= x_there else (x_name, x_there)
    if short.size and not already_said[sparse_name]:
        logger.warning(
            "%d pairs of %s vs %s rest on only %d distinct readings of %s%s, so some "
            "readings appear in more than one pair. A fit that treats the pairs as "
            "independent will count those readings' errors more than once and "
            "understate its uncertainty (METHODS §11.4.1).",
            n_pairs_there,
            y_name,
            x_name,
            sparse_count,
            sparse_name,
            f" at sweep point {int(short[0])} (of {len(counts)})" if sweep else "",
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
    if sweep:
        dataset.attrs[PAIRS_PER_SWEEP_POINT_ATTR] = [int(n) for n in counts[:, 0]]
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
        carried=tuple(names[reference] for reference in carried),
    )


def _choose_clock(
    y: tuple[str, str, CellBounds],
    x: tuple[str, str, CellBounds],
    streams: Mapping[str, xr.Dataset],
) -> tuple[str, str]:
    """Return the instrument whose cells a pair sits on, and why.

    In order: a shared instrument is its own clock; otherwise the wider cells;
    on a tie in width, the variable with fewer measured values where the
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
    # Rule 3, a tie in width: the variable with fewer values where the records
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


def _measured_where_records_overlap(
    pair: tuple[tuple[xr.Dataset, str, CellBounds], tuple[xr.Dataset, str, CellBounds]],
) -> tuple[int, int]:
    """Return how many finite values each variable has where both records run.

    Counted over the *shared* span rather than each whole record, because the
    question is which variable is sparser in the air the two actually have in
    common. A sparse analyzer logging all day beside a dense one switched on
    for ten minutes has more readings in total and fewer where it matters.
    Over the whole record rather than an event's ``interval``, so that one
    pair of instruments keeps one clock from event to event.
    """
    # The span both records cover: the later of the two starts to the earlier end.
    span_start = max(int(cells.start_ns.min()) for _, _, cells in pair)
    span_stop = min(int(cells.stop_ns.max()) for _, _, cells in pair)
    counts = []
    for stream, variable, cells in pair:
        # Cells inside the shared span that hold a value -- at any sweep point
        # of a swept variable, so that one pair of instruments keeps one clock
        # whichever variables of theirs are paired.
        inside = (cells.stop_ns > span_start) & (cells.start_ns < span_stop)
        values = np.asarray(stream[variable].transpose(TIME_COORD, ...).values, dtype=np.float64)
        finite = np.isfinite(values.reshape(len(values), -1)).any(axis=1)
        counts.append(int(np.count_nonzero(inside & finite)))
    return counts[0], counts[1]


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


def _finite(array: xr.DataArray) -> xr.DataArray:
    """Return where an array holds a finite value, keeping its dimensions and coordinates."""
    return array.copy(data=np.isfinite(np.asarray(array.values, dtype=np.float64)))


def _sweep_of(stream: xr.Dataset, variable: str) -> tuple[tuple[str, int], ...]:
    """Return a variable's sweep dimensions and their sizes, in its own order."""
    array = stream[variable]
    return tuple((str(dim), int(array.sizes[dim])) for dim in array.dims if dim != TIME_COORD)


def _most_finite_rows(joined: xr.Dataset, column: str) -> int:
    """Return the most rows any one sweep point of a column holds a value in.

    The rows a value occupies, counted as the binner counts a column's
    readings: per version, the largest. Unswept, simply the finite rows.
    """
    values = np.asarray(joined[column].transpose(TIME_COORD, ...).values, dtype=np.float64)
    return int(np.isfinite(values.reshape(len(values), -1)).sum(axis=0).max())


def _counts_per_sweep_point(
    dataset: xr.Dataset,
    paired: xr.DataArray,
    sweep: Sequence[str],
    streams: Mapping[str, xr.Dataset],
    pair: tuple[tuple[str, str, str, CellBounds], tuple[str, str, str, CellBounds]],
    kept: CellBounds,
) -> np.ndarray:
    """Return, per sweep point, the pairs and the readings behind them.

    One row per sweep point, in the product's order with the last dimension
    varying fastest (one row in all, unswept): the pairs there, then each
    variable's distinct readings behind them, then each variable's readings that
    formed more than one of them. At a sweep point the pairs are the kept
    rows that are a pair *there*, and each variable is read at that point of its
    own sweep, so every count is the one an unswept pairing of that sweep
    point alone would give (tested).
    """
    shape = tuple(int(paired.sizes[dim]) for dim in sweep)
    points = [dict(zip(sweep, index, strict=True)) for index in np.ndindex(*shape)]
    is_pair = np.asarray(paired.transpose(TIME_COORD, *sweep).values).reshape(len(kept), -1)
    counts = np.zeros((len(points), 5), dtype=np.int64)
    for column, point in enumerate(points):
        rows = np.flatnonzero(is_pair[:, column])
        counts[column, 0] = rows.size
        # A point with no pair asks the questions of no cells, and both answer 0.
        cells = CellBounds(start_ns=kept.start_ns[rows], stop_ns=kept.stop_ns[rows])
        for offset, (name, instrument, variable, readings) in enumerate(pair):
            # The variable at this point of its sweep, so that a reading blank
            # here (its baseline window failed) is not counted here.
            stream = streams[instrument].isel(point) if point else streams[instrument]
            counts[column, 1 + offset] = _readings_behind(
                dataset, name, stream, variable, readings, cells
            )
            counts[column, 3 + offset] = _shared_readings(
                dataset, name, stream, variable, readings, cells
            )
    return counts


def _readings_behind(
    joined: xr.Dataset,
    column: str,
    stream: xr.Dataset,
    variable: str,
    readings: CellBounds,
    pairs: CellBounds,
) -> int:
    """Return how many distinct readings of one species the pairs draw on.

    A variable already on the clock contributes exactly one reading per pair by
    construction. A binned variable contributes every finite reading that
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

    A variable already on the clock forms exactly one pair per reading by
    construction, so none of its readings is shared. A binned variable's count
    comes from the same overlap search that counted the readings behind the
    pairs, restricted to the pairs that survived.
    """
    if joined[column].attrs.get(BINNED_ATTR) == 0:
        return 0
    return shared_readings(stream, variable, readings, pairs)


def _warn_if_a_partner_blends(
    dataset: xr.Dataset,
    y: tuple[str, CellBounds, int],
    x: tuple[str, CellBounds, int],
    clock_cells: CellBounds,
) -> None:
    """Say so when a binned variable's cells are as wide as the clock's but offset from them.

    The 2024 drive suite's shape on every pair (§11.4.1): each value of the
    averaged variable blends the two readings straddling its cell, so it rests
    on air outside the cell by the borrowed share (0.5 at half a cell). That
    much follows from the geometry alone -- the same exact test the grid
    applies to its own cells (:func:`~tsara.align.cells.phase_offset_s`)
    -- and is always said.

    What does not follow from the geometry is whether the pairs are
    independent. A *dense* variable lends each reading to two neighbouring
    pairs, so neighbouring pairs share an error and a fit treating them as
    independent has a standard error too narrow by up to 1/sqrt(2); the
    remedy is a coarser common clock, which the message then names. A
    *sparse* variable's pairs sit cells apart, no reading reaches two of them,
    and naming the remedy would send the caller to throw away resolution for
    nothing. So that clause is gated on the exact count of readings that
    formed more than one surviving pair
    (:func:`~tsara.align.cells.shared_readings`), never on the geometry.
    """
    for name, variable_cells, shared in (y, x):
        if dataset[name].attrs.get(BINNED_ATTR) != 1:
            continue
        offset_s = phase_offset_s(variable_cells, clock_cells)
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
