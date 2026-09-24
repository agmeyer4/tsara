"""Which variables a call acts on.

Every joining operation starts by resolving what the caller meant: a bare
canonical name searched across the campaign's streams, or an explicit
``(instrument, variable)`` pair; and, when nothing is named, the default
selection of every value column in every stream, leaving out the companions
that travel with a value rather than being values themselves. Both answers
are given here once, so that the binner joins exactly the set the output
grid validated (``docs/METHODS.md`` §11.7).

The one refusal that belongs to selection rather than to arithmetic is here
too: a dataset that is itself the product of an earlier join may not be
joined again, because its rows are not readings (§11.2.3).

This module and :mod:`tsara.align.cells` are the vocabulary of the
subpackage; :mod:`tsara.align.binning` is the operation spoken in it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import xarray as xr

from tsara.core.exceptions import TsaraError
from tsara.core.naming import TIME_BOUNDS_VAR, is_companion_name

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

__all__ = [
    "JOINED_STAGES",
    "TsaraAlignError",
    "VariableRef",
    "resolve_variable",
    "select_variables",
]


#: How a caller names a variable: by its canonical name, or by the instrument
#: that measured it when more than one did.
#:
#: A real type alias, not a string that looks like one. It was written as
#: ``"str | tuple[str, str]"`` — an ordinary ``str`` object — and exported, so
#: it described the convention in prose while checking nothing and appearing in
#: no signature. Every function here and in the modules layered on top now
#: annotates with it, which is what makes the name worth having.
VariableRef: TypeAlias = str | tuple[str, str]


#: What ``tsara_stage`` says on a dataset this package built by *joining*
#: measurements, as opposed to one it read from an archive (``ingest``) or
#: manufactured (``synthetic``).
#:
#: The list is deliberately of the three join stages rather than an allow-list
#: of the two source stages. A later phase may well produce a native-rate
#: product of its own — Phase 5's baselines are computed cell by cell over each
#: stream's own cells, so every baseline value still stands for one reading —
#: and binning that is honest. What is refused is joining something whose rows
#: are already the *output* of a join (§11.2.3).
JOINED_STAGES = frozenset({"binned", "paired", "gridded"})


class TsaraAlignError(TsaraError):
    """Raised when variables cannot be put on a common support as asked.

    Its own type because the failures are about *combining* streams rather
    than about reading them: a variable no stream declares, a name two
    streams both declare where one was expected, or a stream carrying no
    cells to bin from.
    """


def select_variables(
    streams: Mapping[str, xr.Dataset],
    variables: Sequence[VariableRef] | None = None,
) -> list[tuple[str, str]]:
    """Resolve which variables a call acts on, as ``(instrument, name)`` pairs.

    Public because more than one stage needs the *same* answer: the binner
    uses it to decide what to join, and the output grid uses it to decide
    which cell widths its period has to respect (``docs/METHODS.md`` §11.7).
    Two implementations of "which variables did the caller mean" would let
    the grid validate one set and bin another.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    variables : sequence of str or (str, str), optional
        Explicit selection. ``None`` takes every non-sigma variable in every
        stream, in stream order.

    Returns
    -------
    list of tuple of (str, str)
        Instrument and variable name for each selection.

    Raises
    ------
    TsaraAlignError
        If a reference names nothing or names it ambiguously, or if a selected
        dataset is a product of an earlier join (:func:`_refuse_a_joined_product`).

    Notes
    -----
    Companion columns are excluded from the default because they are not
    variables in their own right: each travels automatically with the value it
    describes, and selecting one directly would produce a column with no parent
    and no meaning (:func:`~tsara.core.naming.is_companion_name`). The
    everyday case is the uncertainty components, which arrive on every stream
    that declares a budget and must not be averaged as though they were data;
    the counts, coverage fractions and angular quality numbers this module adds
    are covered by the same predicate, so nothing that is handed a dataset of
    TSARA's own making can bin a ``coverage_ch4`` into a
    ``coverage_coverage_ch4``.

    The cell boundaries are excluded too. They are metadata describing the
    rows rather than a variable over them, and they are *usually* a coordinate
    and therefore invisible here; a stream that carries them as a data variable
    is a shape :func:`~tsara.align.cells.stream_cells` deliberately accepts, so this must accept
    it as well rather than trying to average a set of boundaries.
    """
    if variables is not None:
        # An explicit selection: resolve each reference, refusing ambiguity.
        selection = [resolve_variable(streams, reference) for reference in variables]
    else:
        # The default: every value column in every stream, in stream order, skipping
        # the companions (sigmas, counts, coverage, angular quality) and the bounds.
        selection = [
            (instrument, str(name))
            for instrument, stream in streams.items()
            for name in stream.data_vars
            if not is_companion_name(str(name)) and str(name) != TIME_BOUNDS_VAR
        ]
    # Both routes end here, and so does every caller: the binner joins what this
    # returns, and the grid checks its period against it (§11.2.3).
    for instrument in dict.fromkeys(name for name, _ in selection):
        _refuse_a_joined_product(streams[instrument], instrument)
    return selection


def resolve_variable(streams: Mapping[str, xr.Dataset], reference: VariableRef) -> tuple[str, str]:
    """Return ``(instrument, variable)`` for a caller's reference.

    A bare name is searched across the streams. Two instruments measuring the
    same canonical name is a real situation — it is how a campaign compares
    two analyzers — so ambiguity is an error naming both rather than a silent
    choice of the first.

    Parameters
    ----------
    streams : mapping of str to xarray.Dataset
        The campaign's streams.
    reference : str or tuple of (str, str)
        Canonical name, or an explicit instrument and variable.

    Returns
    -------
    tuple of (str, str)
        Instrument name and variable name.

    Raises
    ------
    TsaraAlignError
        If the reference names nothing, or names something ambiguously.
    """
    if isinstance(reference, tuple):
        # Explicit (instrument, variable): nothing to search for, only to check.
        instrument, variable = reference
        if instrument not in streams:
            raise TsaraAlignError(
                f"No stream named '{instrument}'; available: {sorted(map(str, streams))}."
            )
        if variable not in streams[instrument].data_vars:
            raise TsaraAlignError(
                f"Stream '{instrument}' has no variable '{variable}'; available: "
                f"{sorted(map(str, streams[instrument].data_vars))}."
            )
        return instrument, variable
    # A bare name: find every stream that carries it, then insist on exactly one.
    holders = [name for name, stream in streams.items() if reference in stream.data_vars]
    if not holders:
        raise TsaraAlignError(
            f"No stream measures '{reference}'. Streams and their variables: "
            + "; ".join(f"{n}: {sorted(map(str, s.data_vars))}" for n, s in streams.items())
        )
    if len(holders) > 1:
        raise TsaraAlignError(
            f"'{reference}' is measured by more than one instrument ({sorted(holders)}), "
            "so naming it alone is ambiguous. Pass an (instrument, variable) pair to "
            "say which one."
        )
    return holders[0], reference


def _refuse_a_joined_product(stream: xr.Dataset, instrument: str) -> None:
    """Raise if a dataset is itself the product of an earlier join.

    A join reads three things from each input row: the value, the interval the
    row describes, and how much of that interval holds data. On a *stream* all
    three are properties of a measurement. On a product they are properties of
    a **row**, and the second pass cannot tell the difference — it takes each
    row as a measurement covering its whole cell, so a minute row holding one
    15 s canister fill is weighted, counted and reported as a fully measured
    minute.

    Measured on a one-hour campaign of a 1 Hz analyzer with outages beside a
    canister (§11.2.3): five-minute rows built from the campaign's 60 s rows
    are up to 2.9 ppb from the same rows built from the streams, and report a
    coverage of 1.000 for a row where the streams say 0.933. A *nested*
    re-join can be made exact by weighting each input row by its own coverage
    (1.6e-12 ppb in the same test); a non-nested one cannot be repaired at all,
    since cutting a row assigns its mean to both sides and the air in the two
    halves differs — 5.8 ppb apart, weighted or not. That is narrowing, and the
    information is gone from the input rather than mishandled by the join.

    So this is a refusal rather than a warning, on the precedent of the
    antimeridian: refuse the case, do not model it. Nothing needs the
    capability — every product can be rebuilt from the native streams, exactly,
    and a sweep over grid period does precisely that.
    """
    stage = str(stream.attrs.get("tsara_stage", ""))
    if stage not in JOINED_STAGES:
        return
    raise TsaraAlignError(
        f"'{instrument}' is a product this package built (tsara_stage is '{stage}'), "
        "and a product may not be joined again. Its rows are not readings: their "
        "coverage, counts and overlap weights describe the rows of the first join "
        "rather than the air, so a second pass would report a fully covered row over "
        "an interval that was barely measured (METHODS §11.2.3). Build what you want "
        "directly from the native streams, which is always possible and exact."
    )
