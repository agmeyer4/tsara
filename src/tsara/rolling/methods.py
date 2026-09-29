"""The baseline methods, registered by name, and what each hands the state.

A baseline is not a single thing (``docs/METHODS.md`` §6.1), so an
instrument too sparse to see a background at the windows that matter gets a
choice rather than a rule (§6.5). Each choice is a *method*: a function that
takes one variable's readings, the sweep, and the configuration entry that
chose it, and returns the baseline at every reading for every point of the
sweep, with whatever qualifiers and uncertainties that method can honestly
give. Three ship:

``rolling_quantile``
    The weighted low quantile of the variable's own readings over each
    window (§6.3), the default; with the window's count and coverage, the
    width flag, Woodruff's sigma, and the reading's systematic sigma read at
    the quantile position (§6.7).
``from_field``
    Another instrument's baseline of the same field, joined onto this
    variable's cells by :func:`tsara.align.bin_streams_onto_cells` and
    handed in as ``provided``; the method checks the product is that join,
    on exactly these cells, at this sweep, and adopts it with the join's
    record.
``constant``
    A declared number, zero included.

Why a registry rather than ``if method == "rolling_quantile": ...``
--------------------------------------------------------------------
The same reasons the file readers are registered
(:mod:`tsara.ingest.registry`): a swappable implementation selected by a
string in a config file is one kind of thing throughout TSARA; a new method
is a decorated function, not a dispatch table to edit; and a group with a
baseline of its own can register it from its own code and use a stock
TSARA. The schema's ``method`` discriminator and this registry name the
same three methods, and the state refuses a configuration naming one the
registry lacks, listing what is registered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

from tsara.config.analysis import BaselineMethod, ConstantMethod, FromFieldMethod
from tsara.core.naming import (
    BASELINE_PREFIX,
    JOIN_RECORD_ATTRS,
    TIME_COORD,
    sigma_rand_name,
    sigma_sys_name,
)
from tsara.core.support import CellBounds, same_cells, stream_cells
from tsara.core.timebase import NS_PER_S
from tsara.rolling.quantile import rolling_quantile
from tsara.rolling.windows import TsaraRollingError, window_cells

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt
    import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_FROM_ATTR",
    "BASELINE_VALUE_ATTR",
    "BaselineMethodFunction",
    "BaselineRequest",
    "BaselineResult",
    "QUANTILE_DIM",
    "WINDOW_DIM",
    "available_baseline_methods",
    "baseline_by_constant",
    "baseline_by_rolling_quantile",
    "baseline_from_field",
    "get_baseline_method",
    "register_baseline_method",
]


#: The two sweep dimensions of the rolling state (``docs/METHODS.md`` §6.2).
WINDOW_DIM = "baseline_window"
QUANTILE_DIM = "baseline_quantile"

#: Attrs a method writes on the baseline it made: which instrument a
#: ``from_field`` baseline was adopted from, and what a ``constant`` is.
BASELINE_FROM_ATTR = "tsara_baseline_from"
BASELINE_VALUE_ATTR = "tsara_baseline_value"


@dataclass(frozen=True, eq=False)
class BaselineRequest:
    """Everything a method is given for one variable.

    Attributes
    ----------
    stream : xarray.Dataset
        The instrument's stream, for the variable's attributes and its sigma
        companions.
    instrument, variable : str
        Which variable, for messages and for the record.
    readings : CellBounds
        The stream's cells.
    values : numpy.ndarray
        The variable's readings, one per cell, ``nan`` where masked.
    windows_ns : tuple of int
        Every window of the sweep, in nanoseconds, in the configuration's
        order.
    quantiles : tuple of float
        Every quantile of the sweep, in the configuration's order.
    method : RollingQuantileMethod or FromFieldMethod or ConstantMethod
        The configuration entry that chose this method.
    provided : xarray.Dataset or None
        For ``from_field``: the donor's baseline joined onto these cells.
    block_elements : int
        The rolling engine's per-block budget.
    """

    stream: xr.Dataset
    instrument: str
    variable: str
    readings: CellBounds
    values: npt.NDArray[np.float64]
    windows_ns: tuple[int, ...]
    quantiles: tuple[float, ...]
    method: BaselineMethod
    provided: xr.Dataset | None
    block_elements: int


@dataclass(frozen=True, eq=False)
class BaselineResult:
    """What a method hands back: the baseline and what it can say about it.

    Every array is over ``(readings, windows, quantiles)`` unless stated;
    ``None`` means the method has nothing honest to say -- a constant has no
    sampling uncertainty and no window, an adopted baseline's windows are
    the donor's.

    Attributes
    ----------
    baseline : numpy.ndarray
        The baseline at every reading and sweep point; ``nan`` where the
        method could not give one.
    sigma_rand : numpy.ndarray or None
        The baseline's own random (sampling) one-sigma, §6.7.
    sigma_sys : numpy.ndarray or None
        The baseline's systematic one-sigma: the reading's, at the quantile,
        for a rolling quantile; the donor's, joined, for ``from_field``.
    n_readings, coverage : numpy.ndarray or None
        Over ``(readings, windows)``: the window's contributing count and
        its covered share (§6.4).
    too_wide : numpy.ndarray or None
        Over ``(readings, windows)``: whether a contributing reading was at
        least ``COPY_RATIO`` times as wide as the window (§6.3).
    same_instrument : bool
        Whether a systematic error common to the reading and the baseline
        cancels in their difference, which decides the enhancement's
        systematic sigma (§6.7).
    attrs : dict
        Method-specific attributes for the baseline column.
    """

    baseline: npt.NDArray[np.float64]
    sigma_rand: npt.NDArray[np.float64] | None
    sigma_sys: npt.NDArray[np.float64] | None
    n_readings: npt.NDArray[np.int64] | None
    coverage: npt.NDArray[np.float64] | None
    too_wide: npt.NDArray[np.bool_] | None
    same_instrument: bool
    attrs: dict[str, object]


class BaselineMethodFunction(Protocol):
    """The signature every registered baseline method has."""

    def __call__(self, request: BaselineRequest, /) -> BaselineResult:
        """Return the baseline for one variable."""
        ...  # pragma: no cover


#: Registered methods keyed by the configuration's ``method`` discriminator.
#: Module-private: mutation goes through :func:`register_baseline_method`.
_METHODS: dict[str, BaselineMethodFunction] = {}

#: Bound to the protocol so the decorator returns the same function type it
#: received, keeping direct calls typed.
M = TypeVar("M", bound=BaselineMethodFunction)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def register_baseline_method(name: str, *, replace: bool = False) -> Callable[[M], M]:
    """Register a baseline method under its configuration name.

    Used as a decorator::

        @register_baseline_method("rolling_quantile")
        def baseline_by_rolling_quantile(request: BaselineRequest, /) -> BaselineResult:
            ...

    Re-registering a name is refused unless ``replace=True`` says so, for the
    reason the reader registry gives (:func:`tsara.ingest.registry.register_reader`):
    a silent overwrite makes the baseline depend on import order. The
    override is logged, since re-running a notebook cell is the case that
    needs it and a run log should still show it.

    Parameters
    ----------
    name : str
        The method's name, the ``method`` value in ``baseline.methods``.
    replace : bool, optional
        Replace a method already registered under ``name``.

    Returns
    -------
    callable
        Decorator returning its argument unchanged.

    Raises
    ------
    ValueError
        If ``name`` is blank, or already registered and ``replace`` is False.
    """
    if not name or not name.strip():
        raise ValueError("Baseline method name must be a non-empty string.")

    def decorator(func: M) -> M:
        existing = _METHODS.get(name)
        if existing is not None:
            if not replace:
                raise ValueError(
                    f"A baseline method is already registered as '{name}' "
                    f"({_describe(existing)}). Pass replace=True if you meant to override it."
                )
            logger.warning(
                "Replacing baseline method '%s' (%s) with %s.",
                name,
                _describe(existing),
                _describe(func),
            )
        _METHODS[name] = func
        return func

    return decorator


def _describe(func: object) -> str:
    """Return ``module.qualname`` for a method, for messages."""
    return f"{getattr(func, '__module__', '?')}.{getattr(func, '__qualname__', '?')}"


def available_baseline_methods() -> tuple[str, ...]:
    """List the registered method names, sorted.

    Returns
    -------
    tuple of str
        e.g. ``('constant', 'from_field', 'rolling_quantile')``.
    """
    return tuple(sorted(_METHODS))


def get_baseline_method(name: str) -> BaselineMethodFunction:
    """Look up a baseline method by name.

    Parameters
    ----------
    name : str
        The ``method`` value of a ``baseline.methods`` entry.

    Returns
    -------
    BaselineMethodFunction
        The registered method.

    Raises
    ------
    TsaraRollingError
        If nothing is registered under ``name``; the message lists what is.
    """
    try:
        return _METHODS[name]
    except KeyError:
        raise TsaraRollingError(
            f"No baseline method registered as '{name}'. Available: "
            f"{list(available_baseline_methods())}. A method must have been imported to "
            "be registered."
        ) from None


# ---------------------------------------------------------------------------
# The three methods
# ---------------------------------------------------------------------------


@register_baseline_method("rolling_quantile")
def baseline_by_rolling_quantile(request: BaselineRequest, /) -> BaselineResult:
    """Return the weighted low quantile of the variable's own readings over each window.

    One call of the rolling engine per window of the sweep, every quantile
    read from that window's one sort (§6.3). The reading's systematic sigma,
    when the stream has one, rides through the sort and is read at the same
    positions, so that the baseline's systematic sigma is the sigma of the
    readings that define it (§6.7).

    Parameters
    ----------
    request : BaselineRequest
        The variable and the sweep.

    Returns
    -------
    BaselineResult
        With every qualifier: count, coverage, width flag, both sigmas (the
        systematic one only when the reading has one).
    """
    n = request.values.size
    n_windows, n_quantiles = len(request.windows_ns), len(request.quantiles)
    baseline = np.full((n, n_windows, n_quantiles), np.nan)
    sigma_rand = np.full((n, n_windows, n_quantiles), np.nan)
    n_readings = np.zeros((n, n_windows), dtype=np.int64)
    coverage = np.zeros((n, n_windows), dtype=np.float64)
    too_wide = np.zeros((n, n_windows), dtype=np.bool_)
    systematic = sigma_sys_name(request.variable)
    carry = (
        np.asarray(request.stream[systematic].values, dtype=np.float64)
        if systematic in request.stream.data_vars
        else None
    )
    sigma_sys = None if carry is None else np.full((n, n_windows, n_quantiles), np.nan)
    for index, window_ns in enumerate(request.windows_ns):
        rolled = rolling_quantile(
            request.readings,
            request.values,
            window_cells(request.readings, window_ns),
            request.quantiles,
            carry=carry,
            block_elements=request.block_elements,
        )
        baseline[:, index, :] = rolled.values
        sigma_rand[:, index, :] = rolled.sigma
        n_readings[:, index] = rolled.n_readings
        coverage[:, index] = rolled.coverage
        too_wide[:, index] = rolled.too_wide
        if sigma_sys is not None and rolled.carried is not None:
            sigma_sys[:, index, :] = rolled.carried
    return BaselineResult(
        baseline=baseline,
        sigma_rand=sigma_rand,
        sigma_sys=sigma_sys,
        n_readings=n_readings,
        coverage=coverage,
        too_wide=too_wide,
        same_instrument=True,
        attrs={},
    )


@register_baseline_method("from_field")
def baseline_from_field(request: BaselineRequest, /) -> BaselineResult:
    """Another instrument's baseline of the same field, joined onto these cells.

    The join itself is not made here: stages hand each other Datasets and
    never import each other (CONTRIBUTING, "Four layers"), so the caller
    rolls the donor, joins its baseline onto this stream's cells with
    :func:`tsara.align.bin_streams_onto_cells`, and passes the product as
    ``provided``. This method checks that the product is that join -- built by
    a join, on exactly these cells, at this sweep, carrying the donor's
    baseline of this field -- and adopts it with the join's own record of
    what it did to the donor's support (§11.2.4).

    Parameters
    ----------
    request : BaselineRequest
        The variable, the sweep, and the provided product.

    Returns
    -------
    BaselineResult
        The adopted baseline and its joined sigmas; no window qualifiers,
        since the windows are the donor's.

    Raises
    ------
    TsaraRollingError
        If no product was provided, or the product is not the join described.
    """
    method = request.method
    assert isinstance(method, FromFieldMethod)  # the registry name guarantees it
    where = f"'{request.variable}' on '{request.instrument}'"
    if request.provided is None:
        raise TsaraRollingError(
            f"{where} adopts its baseline from '{method.instrument}' (from_field), and no "
            "product was provided. Roll the donor stream first, join its baseline onto this "
            "stream's cells with tsara.align.bin_streams_onto_cells(donor_state, "
            "target=this stream's cells), and pass the result as "
            f"provided={{'{request.variable}': joined}}."
        )
    provided = request.provided
    stage = provided.attrs.get("tsara_stage")
    if stage != "binned":
        raise TsaraRollingError(
            f"The product provided for {where} has tsara_stage '{stage}', not 'binned'; "
            "a from_field baseline is the donor's baseline joined onto this stream's cells."
        )
    field = str(request.stream[request.variable].attrs.get("field", request.variable))
    columns = [
        str(name)
        for name in provided.data_vars
        if str(name).startswith(BASELINE_PREFIX)
        and provided[name].attrs.get("field") == field
        and provided[name].attrs.get("tsara_instrument") == method.instrument
    ]
    if len(columns) != 1:
        raise TsaraRollingError(
            f"The product provided for {where} holds {len(columns)} baseline column(s) of "
            f"field '{field}' from '{method.instrument}' ({columns}); exactly one is adopted. "
            f"Its columns: {sorted(map(str, provided.data_vars))}."
        )
    column = columns[0]
    if not same_cells(stream_cells(provided, "the provided product"), request.readings):
        raise TsaraRollingError(
            f"The product provided for {where} does not sit on this stream's cells; a "
            "from_field baseline is joined onto exactly the cells of the stream that adopts "
            "it (target=stream_cells(stream, instrument))."
        )
    _check_sweep(provided, request, where)
    array = provided[column].transpose(TIME_COORD, WINDOW_DIM, QUANTILE_DIM)
    baseline = np.asarray(array.values, dtype=np.float64)
    sigmas: dict[str, npt.NDArray[np.float64] | None] = {}
    for label, name in (
        ("random", sigma_rand_name(column)),
        ("systematic", sigma_sys_name(column)),
    ):
        sigmas[label] = (
            np.asarray(
                provided[name].transpose(TIME_COORD, WINDOW_DIM, QUANTILE_DIM).values,
                dtype=np.float64,
            )
            if name in provided.data_vars
            else None
        )
    # The join's record of what it did to the donor's support travels with
    # the adopted baseline, under the names METHODS §11.2.4 documents. The
    # donor's own baseline attributes do not: they describe the donor's
    # product, which is where a reader finds them, and the adopter writes
    # its own (§6.6).
    attrs: dict[str, object] = {
        key: provided[column].attrs[key]
        for key in JOIN_RECORD_ATTRS
        if key in provided[column].attrs
    }
    attrs[BASELINE_FROM_ATTR] = method.instrument
    return BaselineResult(
        baseline=baseline,
        sigma_rand=sigmas["random"],
        sigma_sys=sigmas["systematic"],
        n_readings=None,
        coverage=None,
        too_wide=None,
        same_instrument=False,
        attrs=attrs,
    )


def _check_sweep(provided: xr.Dataset, request: BaselineRequest, where: str) -> None:
    """Refuse a provided product whose sweep is not this configuration's."""
    wanted = {
        WINDOW_DIM: np.asarray(request.windows_ns, dtype=np.float64) / NS_PER_S,
        QUANTILE_DIM: np.asarray(request.quantiles, dtype=np.float64),
    }
    for dim, values in wanted.items():
        if dim not in provided.coords:
            raise TsaraRollingError(
                f"The product provided for {where} carries no '{dim}' coordinate; a "
                "from_field baseline is a swept baseline joined onto this stream's cells."
            )
        found = np.asarray(provided[dim].values, dtype=np.float64)
        if found.shape != values.shape or not np.array_equal(found, values):
            raise TsaraRollingError(
                f"The product provided for {where} was rolled at {dim} = {found.tolist()}, "
                f"but this configuration sweeps {values.tolist()}; the donor and the adopter "
                "must be rolled at the same sweep (METHODS §6.5)."
            )


@register_baseline_method("constant")
def baseline_by_constant(request: BaselineRequest, /) -> BaselineResult:
    """Return a declared number at every reading and sweep point, zero included.

    At zero the enhancement is the concentration (§6.5). A constant has no
    sampling uncertainty and no window, and nothing about it cancels a
    systematic error of the reading, so the enhancement inherits the
    reading's whole systematic sigma (§6.7).

    Parameters
    ----------
    request : BaselineRequest
        The variable and the sweep; the method's ``value``.

    Returns
    -------
    BaselineResult
        The constant, and nothing else.
    """
    method = request.method
    assert isinstance(method, ConstantMethod)  # the registry name guarantees it
    shape = (request.values.size, len(request.windows_ns), len(request.quantiles))
    return BaselineResult(
        baseline=np.full(shape, float(method.value), dtype=np.float64),
        sigma_rand=None,
        sigma_sys=None,
        n_readings=None,
        coverage=None,
        too_wide=None,
        same_instrument=False,
        attrs={BASELINE_VALUE_ATTR: float(method.value)},
    )
