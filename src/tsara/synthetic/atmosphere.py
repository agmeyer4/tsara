"""The synthetic atmosphere: what is true, at any time, before anyone measures it.

A synthetic campaign is manufactured in two halves that this module keeps
apart. First the **air**: every field's background realized once, and every
plume event drawn once (:func:`realize_atmosphere`). Then the **instruments**,
each of which samples that air through its own clock and cells and adds its
own error (:mod:`tsara.synthetic.generator`). An :class:`Atmosphere` is the
first half made queryable: ask it for a field's true value at any instants,
or its true mean over any cells, and it answers from the same realization
every instrument sampled.

Why the halves are kept apart
-----------------------------
Before Phase 4.5 each instrument rendered its own copy of each species, which
made the answer to "what was the methane at 14:03:07?" depend on which
instrument was asked. For analytic backgrounds the copies agreed; for a
random walk they were independent draws and disagreed by about as much as the
background varied, so two analyzers on one inlet were measuring two different
atmospheres. Rendering the air once removes that by construction, and it
gives the answer key a form the instruments' truth columns cannot: a function
of time that exists between samples and beyond any one instrument's cells.

The two identities that make the claim checkable
------------------------------------------------
Because an instrument's noise-free signal is computed *by* this module rather
than alongside it, two identities hold exactly, to the last bit:

* a noise-free ``point`` instrument's values equal :meth:`Atmosphere.value`
  at the stream's timestamps;
* a noise-free ``mean`` instrument's values equal
  :meth:`Atmosphere.mean_over` its cells, at its subsample count.

A test of either is therefore a test that the instrument sampled this
atmosphere and nothing else.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tsara.core.timebase import NS_PER_S
from tsara.core.timebase import epoch_ns as _epoch_ns
from tsara.core.timebase import timestamp_epoch_ns as _stamp_ns
from tsara.core.timebase import timestamp_epoch_s as _stamp_s
from tsara.core.timebase import to_utc_naive_stamp as _to_utc_naive_stamp
from tsara.synthetic.background import RealizedBackground, realize_background
from tsara.synthetic.plumes import RealizedEvent, schedule_events

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    import pandas as pd

    from tsara.core.support import CellBounds
    from tsara.synthetic.config import FieldSpec, SyntheticConfig
    from tsara.synthetic.profiling import RealDataProfile

logger = logging.getLogger(__name__)

__all__ = ["Atmosphere", "CellGrid", "CellTruth", "EventPeak", "realize_atmosphere"]

#: How far beyond an event's support window a query instant is still handed
#: to the kernel, in nanoseconds. The kernel itself returns exactly zero
#: outside its support, judged in float seconds; the window is judged in
#: integer nanoseconds. At epoch-scale seconds one float64 step is ~0.24 us,
#: so an instant a few hundred nanoseconds outside the window can still fall
#: inside the support by the kernel's arithmetic. The cell path always
#: evaluates such an instant -- its cell overlaps the window -- so the instant
#: path widens its window by this margin to evaluate it too. 1 ms is far above
#: that rounding and far below any support.
_WINDOW_MARGIN_NS = 1_000_000


@dataclass(frozen=True, eq=False)
class CellGrid:
    """Cells, plus the instants inside each that a cell mean is evaluated at.

    The midpoint rule: ``subsamples`` instants per cell at the midpoints of
    equal sub-intervals, whose error falls as the inverse square of the count.
    Integer nanoseconds throughout, so that with one subsample the single
    instant lands exactly on the cell midpoint -- which is what lets one code
    path serve a ``point`` instrument (one subsample) and a ``mean`` one.

    Attributes
    ----------
    cells : CellBounds
        The cells.
    fine_ns : numpy.ndarray
        Evaluation instants, epoch nanoseconds, shape ``(len(cells), subsamples)``.
    """

    cells: CellBounds
    fine_ns: npt.NDArray[np.int64]

    @classmethod
    def build(cls, cells: CellBounds, subsamples: int) -> CellGrid:
        """Place ``subsamples`` midpoint-rule instants in every cell.

        Parameters
        ----------
        cells : CellBounds
            Cells of any widths.
        subsamples : int
            Instants per cell, at least 1.

        Returns
        -------
        CellGrid
            The cells and their evaluation instants.

        Raises
        ------
        ValueError
            If ``subsamples`` is below 1.
        """
        if subsamples < 1:
            raise ValueError(f"A cell mean needs at least 1 subsample; got {subsamples}.")
        k = np.arange(subsamples, dtype=np.int64)
        # The k-th instant sits (2k + 1) / (2n) of the way across its cell,
        # computed per cell so cells of different widths are each divided
        # evenly. Integer division keeps it exact to the nanosecond.
        offsets = ((2 * k + 1)[None, :] * cells.width_ns[:, None]) // (2 * subsamples)
        return cls(cells=cells, fine_ns=cells.start_ns[:, None] + offsets)

    @property
    def subsamples(self) -> int:
        """Evaluation instants per cell."""
        return int(self.fine_ns.shape[1])


@dataclass(frozen=True, eq=False)
class EventPeak:
    """How much of one event a set of cells could see.

    Attributes
    ----------
    event : RealizedEvent
        The event.
    sampled_peak : float
        The largest cell mean of this event's contribution alone: the event as
        these cells record it, diluted if they are wide. NaN when no cell
        overlaps the event, which is how an event lost inside a data gap is
        told apart from a small one.
    """

    event: RealizedEvent
    sampled_peak: float


@dataclass(frozen=True, eq=False)
class CellTruth:
    """One field's truth averaged over a set of cells, with its decomposition.

    Attributes
    ----------
    background, enhancement : numpy.ndarray
        Cell means of the plume-free signal and of the plumes, one per cell.
    peaks : tuple of EventPeak
        One entry per event carrying this field, in event order.
    """

    background: npt.NDArray[np.float64]
    enhancement: npt.NDArray[np.float64]
    peaks: tuple[EventPeak, ...]

    @property
    def value(self) -> npt.NDArray[np.float64]:
        """The cell means of the field itself: background plus enhancement."""
        return np.asarray(self.background + self.enhancement, dtype=np.float64)


@dataclass(frozen=True, eq=False)
class Atmosphere:
    """Every field's true value as a function of time, for one campaign.

    Built by :func:`realize_atmosphere`, never by hand: every random draw is
    already made, so every query below is deterministic and can be repeated,
    reordered or asked at times no instrument sampled.

    Attributes
    ----------
    start, end : pandas.Timestamp
        The campaign, tz-naive UTC; ``end`` is exclusive for instrument clocks.
    fields : mapping of str to FieldSpec
        Each field's configuration (role, units, circularity, background).
    backgrounds : mapping of str to RealizedBackground
        Each field's realized background.
    events : tuple of RealizedEvent
        Every plume event, sorted by center time.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    fields: Mapping[str, FieldSpec]
    backgrounds: Mapping[str, RealizedBackground]
    events: tuple[RealizedEvent, ...]

    def background(self, field: str, times: object) -> npt.NDArray[np.float64]:
        """Return the field's true plume-free value at ``times``.

        Parameters
        ----------
        field : str
            Field name.
        times : array-like of datetimes
            Query instants, in any order: a ``DatetimeIndex``, a
            ``datetime64`` array such as ``stream["time"].values``, or a list
            of timestamps. Timezone-aware values are converted to UTC.

        Returns
        -------
        numpy.ndarray
            One value per query instant.

        Raises
        ------
        KeyError
            If the atmosphere has no such field.
        """
        return self._background_of(field).at(_query_ns(times) / NS_PER_S)

    def enhancement(self, field: str, times: object) -> npt.NDArray[np.float64]:
        """Return the sum of every plume's contribution to the field at ``times``.

        Parameters
        ----------
        field : str
            Field name. Only gas fields carry plumes; any other returns zeros.
        times : array-like of datetimes
            Query instants, in any order (see :meth:`background`).

        Returns
        -------
        numpy.ndarray
            One value per query instant.

        Raises
        ------
        KeyError
            If the atmosphere has no such field.
        """
        self._background_of(field)  # a typo'd field fails here, not as zeros
        query_ns = _query_ns(times)
        query_s = query_ns / NS_PER_S
        # Sorted once so each event finds its instants by binary search, which
        # keeps this O(events x window) rather than O(events x instants).
        order = np.argsort(query_ns, kind="stable")
        sorted_ns = query_ns[order]
        sorted_s = query_s[order]
        total = np.zeros(query_ns.size, dtype=np.float64)
        for event in self.events:
            amplitude = event.amplitudes.get(field)
            if amplitude is None:
                continue
            window_start, window_end = event.species_window(field)
            lo = int(
                np.searchsorted(sorted_ns, _stamp_ns(window_start) - _WINDOW_MARGIN_NS, "left")
            )
            hi = int(np.searchsorted(sorted_ns, _stamp_ns(window_end) + _WINDOW_MARGIN_NS, "right"))
            if hi > lo:
                # Same expression, operand for operand, as the cell path below:
                # that is what makes a point instrument equal this exactly.
                dt_s = sorted_s[lo:hi] - _stamp_s(event.species_center(field))
                total[lo:hi] += amplitude * event.kernel.evaluate(dt_s)
        result = np.empty_like(total)
        result[order] = total
        return result

    def value(self, field: str, times: object) -> npt.NDArray[np.float64]:
        """Return the field's true value at ``times``: background plus enhancement.

        What a perfect ``point`` instrument would read at those instants.

        Parameters
        ----------
        field : str
            Field name.
        times : array-like of datetimes
            Query instants, in any order (see :meth:`background`).

        Returns
        -------
        numpy.ndarray
            One value per query instant.

        Raises
        ------
        KeyError
            If the atmosphere has no such field.
        """
        return np.asarray(
            self.background(field, times) + self.enhancement(field, times), dtype=np.float64
        )

    def mean_over(
        self, field: str, cells: CellBounds, subsamples: int = 64
    ) -> npt.NDArray[np.float64]:
        """Return the field's true mean over each cell.

        What a perfect ``mean`` instrument with these cells would read,
        evaluated by the midpoint rule at ``subsamples`` instants per cell --
        the same quadrature, at the same default count, a generated ``mean``
        instrument uses (:class:`~tsara.synthetic.config.TrueSupport`).

        Parameters
        ----------
        field : str
            Field name.
        cells : CellBounds
            Cells of any widths.
        subsamples : int, optional
            Midpoint-rule instants per cell. 1 evaluates each cell at its
            midpoint, which is what a ``point`` instrument does.

        Returns
        -------
        numpy.ndarray
            One mean per cell.

        Raises
        ------
        KeyError
            If the atmosphere has no such field.
        """
        return self.over_cells(field, CellGrid.build(cells, subsamples)).value

    def over_cells(self, field: str, grid: CellGrid) -> CellTruth:
        """Average the field over a grid of cells, keeping its decomposition.

        The one place an instrument's noise-free signal is computed. The
        generator calls it for every measurement, and :meth:`mean_over` is a
        thin view of it, so the two cannot drift apart.

        Parameters
        ----------
        field : str
            Field name.
        grid : CellGrid
            Cells and their evaluation instants.

        Returns
        -------
        CellTruth
            Cell means of background and enhancement, and what the cells saw
            of each event.

        Raises
        ------
        KeyError
            If the atmosphere has no such field.
        """
        cells = grid.cells
        n_sub = grid.subsamples
        fine_s = grid.fine_ns.reshape(-1) / NS_PER_S
        # Evaluated at every instant and averaged per cell, not once per cell:
        # a mean instrument must actually average the background's wander,
        # otherwise `mean` would be a label rather than an operation.
        background = self._background_of(field).at(fine_s).reshape(-1, n_sub).mean(axis=1)

        enhancement = np.zeros(len(cells), dtype=np.float64)
        peaks: list[EventPeak] = []
        # Events are located against the CELL BOUNDARIES, never against the
        # flattened instants. The flat grid looks sortable and is not:
        # timestamp jitter is permitted up to just under half the sampling
        # interval, so full-width cells centred on jittered stamps overlap and
        # the flattened instants then descend. Measured on a 1 Hz stream with
        # 0.4 s jitter, 151 of 300 adjacent cells overlap and the flat grid
        # has 146 descending steps -- enough for a binary search over it to
        # select the wrong cells, by up to 0.8 % of a plume's peak.
        #
        # Cell starts are sorted whatever the jitter, since they are a
        # constant shift of an increasing clock, so the running maximum of the
        # stops makes a valid lower bound. Same pattern as `bin_onto_cells`.
        cell_start = cells.start_ns
        running_stop = np.maximum.accumulate(cells.stop_ns)
        for event in self.events:
            amplitude = event.amplitudes.get(field)
            if amplitude is None:
                continue
            window_start, window_end = event.species_window(field)
            # Every cell that OVERLAPS the support window, which is the unit an
            # instrument emits: a cell partly inside the window is partly
            # affected by the event. The kernel returns exactly zero outside
            # its support, so cells at the edge of the selection contribute
            # exact zeros rather than small numbers.
            lo = int(np.searchsorted(running_stop, _stamp_ns(window_start), side="right"))
            hi = int(np.searchsorted(cell_start, _stamp_ns(window_end), side="left"))
            sampled_peak = float("nan")
            if hi > lo:
                dt_s = fine_s[lo * n_sub : hi * n_sub] - _stamp_s(event.species_center(field))
                contribution = amplitude * event.kernel.evaluate(dt_s)
                per_cell = contribution.reshape(-1, n_sub).mean(axis=1)
                enhancement[lo:hi] += per_cell
                # The peak as these cells could actually see it. For a wide
                # cell that is the diluted peak, not the true amplitude, which
                # is exactly the quantity a later stage needs to decide whether
                # an event was resolvable at all.
                sampled_peak = float(per_cell.max())
            peaks.append(EventPeak(event=event, sampled_peak=sampled_peak))

        return CellTruth(background=background, enhancement=enhancement, peaks=tuple(peaks))

    def _background_of(self, field: str) -> RealizedBackground:
        """Look a field up, naming the fields that do exist when it is absent."""
        if field not in self.backgrounds:
            raise KeyError(
                f"The atmosphere has no field '{field}'; its fields are {sorted(self.backgrounds)}."
            )
        return self.backgrounds[field]


def realize_atmosphere(
    config: SyntheticConfig,
    rng: np.random.Generator,
    profiles: Mapping[str, RealDataProfile] | None = None,
) -> Atmosphere:
    """Make every random draw the air needs: the events, then each background.

    The order is part of the contract. Events are drawn first and backgrounds
    second, both before anything about a platform or an instrument, so the
    atmosphere a seed produces does not depend on who measures it -- and so
    it can be rebuilt from a saved config alone, by replaying this function
    on a fresh generator from the same seed (which is what loading a bundle
    does). A background with no stochastic term draws nothing, which is why a
    configuration without one produces the same noise, draw for draw, that it
    did before backgrounds moved here.

    Parameters
    ----------
    config : SyntheticConfig
        The campaign.
    rng : numpy.random.Generator
        The run's generator, freshly seeded from ``config.seed``.
    profiles : mapping of str to RealDataProfile, optional
        Real-data profiles, required only by bootstrap backgrounds.

    Returns
    -------
    Atmosphere
        The realized atmosphere.

    Raises
    ------
    TsaraSyntheticError
        If a bootstrap background names a profile that was not supplied.
    """
    import pandas as pd

    start = _to_utc_naive_stamp(pd.Timestamp(config.start))
    end = start + pd.Timedelta(config.duration)
    events = schedule_events(config, rng)

    start_ns = _stamp_ns(start)
    end_ns = _stamp_ns(end)
    resolution_ns = int(pd.Timedelta(config.atmosphere.truth_resolution).value)
    backgrounds = {
        name: realize_background(
            spec.background,
            start_ns=start_ns,
            end_ns=end_ns,
            truth_resolution_ns=resolution_ns,
            rng=rng,
            profiles=profiles,
            field=name,
        )
        for name, spec in config.atmosphere.fields.items()
    }
    return Atmosphere(
        start=start,
        end=end,
        fields=dict(config.atmosphere.fields),
        backgrounds=backgrounds,
        events=tuple(events),
    )


def _query_ns(times: object) -> npt.NDArray[np.int64]:
    """Return query instants as epoch nanoseconds, whatever array-like they came as."""
    import pandas as pd

    index = times if isinstance(times, pd.DatetimeIndex) else pd.DatetimeIndex(times)  # type: ignore[arg-type]
    return _epoch_ns(index)
