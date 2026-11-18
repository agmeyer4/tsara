"""The ratio stage over a configuration: pair, then fit, for every ratio a ``ratios:`` names.

A ratio set names a reference and its numerators (``RatiosConfig``,
``docs/METHODS.md`` §5.2). Each numerator is fitted against the reference over
every member of every one of the reference's candidates, on each pair clock
the configuration lists: one pairing of the two enhancements
(``tsara.align.pair_species``), carrying both readings so that the fits are
weighted by the readings' own sigma (§4.3) and each gas's clean spread from
the event states so that they hold the baseline's wobble (§5.4), then one
call of ``tsara.ratios.member_fits``. The two belong to two stages that never import
each other, which is why this loop lives on the layer above them.

Nothing is computed here. The table is the fits' own, one call's rows after
another in the order the configuration writes them: ratio set, then
numerator (an ``'<instrument>'`` entry expanded over the variables that
instrument's baseline state holds an enhancement of, the reference left out,
by ``RatioSetConfig.numerator_refs``), then pair clock.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import pandas as pd

from tsara.align import pair_species
from tsara.config.analysis import NATIVE_PERIOD
from tsara.core.naming import clean_spread_name, enhanced_variables, enhancement_name
from tsara.ratios import FIT_COLUMNS, FIT_DTYPES, PairColumns, TsaraRatioError, member_fits

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

    from tsara.config.analysis import RatiosConfig

logger = logging.getLogger(__name__)

__all__ = ["ratio_fits"]


def ratio_fits(
    states: Mapping[str, xr.Dataset],
    events: Mapping[str, xr.Dataset],
    catalog: pd.DataFrame,
    ratios: RatiosConfig,
) -> pd.DataFrame:
    """Fit every numerator of every ratio set against its reference, on every pair clock.

    Parameters
    ----------
    states : mapping of str to xarray.Dataset
        The baseline states, keyed by instrument (``tsara.baseline``), each
        holding its readings beside their enhancements.
    events : mapping of str to xarray.Dataset
        The event states, keyed by instrument (``tsara.events.event_states``):
        a variable's clean spread there gives its baseline's wobble (§5.4). A
        variable without one (a triggered variable, or an instrument with no
        event state) has no wobble, and its fits say so.
    catalog : pandas.DataFrame
        The event catalog with its candidates linked (``tsara.events``).
    ratios : RatiosConfig
        Which ratios to fit, with which estimators, on which pair clocks.

    Returns
    -------
    pandas.DataFrame
        The fits' table (``tsara.ratios.FIT_COLUMNS``), one call's rows after
        another in the order of ratio set, numerator and pair clock; empty,
        with those columns and their types, when the configuration names no
        ratio set.

    Raises
    ------
    TsaraRatioError
        If a reference is not a variable some baseline state holds an
        enhancement of, or a numerator names an instrument or variable that
        none does, or the fits refuse what they are handed.
    TsaraAlignError
        If pairing refuses a numerator and its reference (no cell holding
        both, or a pair clock that would copy a reading onto narrower cells).
    """
    # Per instrument, the variables its baseline state holds an enhancement
    # of: what a ratio can be formed from, and what '<instrument>' expands to.
    gases = {instrument: enhanced_variables(state) for instrument, state in states.items()}
    tables = []
    for ratio_set in ratios.sets:
        instrument, _, variable = ratio_set.reference.partition(".")
        if variable not in gases.get(instrument, ()):
            raise TsaraRatioError(
                f"The ratio set's reference '{ratio_set.reference}' is not a variable any "
                f"baseline state holds an enhancement of; they hold {gases}."
            )
        try:
            numerators = ratio_set.numerator_refs(gases)
        except ValueError as error:
            raise TsaraRatioError(str(error)) from error
        for numerator in numerators:
            for period in ratios.periods:
                tables.append(
                    _fit_one(
                        states,
                        events,
                        catalog,
                        (instrument, variable),
                        numerator,
                        period,
                        ratios.methods,
                    )
                )
    if not tables:
        return pd.DataFrame(columns=list(FIT_COLUMNS)).astype(FIT_DTYPES)
    return pd.concat(tables, ignore_index=True)


def _fit_one(
    states: Mapping[str, xr.Dataset],
    events: Mapping[str, xr.Dataset],
    catalog: pd.DataFrame,
    reference: tuple[str, str],
    numerator: tuple[str, str],
    period: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Pair one numerator with its reference on one pair clock, and fit every member."""
    # Each gas's clean spread rides beside its baseline state into the pairing,
    # where an event state holds one.
    streams = dict(states)
    spreads: dict[tuple[str, str], tuple[str, str]] = {}
    for instrument, variable in (numerator, reference):
        name = clean_spread_name(variable)
        event_state = events.get(instrument)
        if event_state is not None and name in event_state:
            streams[instrument] = streams[instrument].assign({name: event_state[name]})
            spreads[(instrument, variable)] = (instrument, name)
    carry = [numerator, reference, *spreads.values()]
    paired = pair_species(
        streams,
        (numerator[0], enhancement_name(numerator[1])),
        (reference[0], enhancement_name(reference[1])),
        carry=carry,
        target=None if period == NATIVE_PERIOD else period,
    )
    named = dict(zip(carry, paired.carried, strict=True))
    columns = PairColumns(
        paired.y_name,
        paired.x_name,
        named[numerator],
        named[reference],
        named[spreads[numerator]] if numerator in spreads else None,
        named[spreads[reference]] if reference in spreads else None,
    )
    logger.info(
        "Fitting %s.%s against %s.%s on the %s clock (%s): %d pair cells.",
        *numerator,
        *reference,
        period,
        paired.clock,
        paired.n_pairs,
    )
    return member_fits(
        paired.dataset,
        catalog,
        columns,
        reference=".".join(reference),
        numerator=".".join(numerator),
        period=period,
        methods=methods,
    )
