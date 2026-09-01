"""Converting raw values to canonical units.

TSARA deliberately does **not** depend on a unit library (CLAUDE.md §5). A
manifest states the arithmetic directly — ``canonical = native * scale +
offset`` — which covers ppm→ppb, °C→K, mg→µg and essentially everything else
a trace-gas campaign needs. The declared ``from_unit``/``to_unit`` strings
are carried as metadata, leaving room to plug in ``pint`` later without
touching a single manifest.

The one subtlety: a spread is not a value
-----------------------------------------
Converting a *measurement* applies both scale and offset. Converting an
*uncertainty* applies only the scale. An uncertainty is a difference between
two values on the same axis, so the offset cancels:

    σ_canonical = |(x + σ)·s + c − (x·s + c)| = |σ·s|

Applying the offset to a sigma would be a straightforward disaster in the
°C→K case: every uncertainty would gain 273.15. This is stated in
``docs/METHODS.md`` §2.2 and implemented in exactly one place —
:func:`convert_spread` — so that no reader has to remember it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from tsara.config.manifest import UnitConversion

__all__ = ["canonical_units", "convert_spread", "convert_values"]


def convert_values(
    values: npt.NDArray[np.float64], conversion: UnitConversion | None
) -> npt.NDArray[np.float64]:
    """Apply a manifest unit conversion to measured values.

    Parameters
    ----------
    values : numpy.ndarray
        Values as they appear in the raw file.
    conversion : UnitConversion or None
        Conversion to apply. ``None`` means the file already reports
        canonical units, and the values are returned unchanged.

    Returns
    -------
    numpy.ndarray
        Values in canonical units. ``NaN`` is preserved by the arithmetic,
        so masked samples stay masked.
    """
    if conversion is None:
        return values
    return values * conversion.scale + conversion.offset


def convert_spread(
    spread: npt.NDArray[np.float64], conversion: UnitConversion | None
) -> npt.NDArray[np.float64]:
    """Apply a manifest unit conversion to an uncertainty (1-sigma spread).

    Scale only — never the offset. See the module docstring for why, and
    ``docs/METHODS.md`` §2.2 for the decision.

    Parameters
    ----------
    spread : numpy.ndarray
        Per-point 1-sigma values in the raw file's units.
    conversion : UnitConversion or None
        Conversion whose ``scale`` applies.

    Returns
    -------
    numpy.ndarray
        Spread in canonical units, non-negative. The absolute value matters
        for a negative scale (a legitimate way to flip a sign convention),
        where the magnitude of the spread must survive but its sign must not.
    """
    if conversion is None:
        return spread
    return np.abs(spread * conversion.scale)


def canonical_units(declared_units: str, conversion: UnitConversion | None) -> str:
    """Return the units a variable has after conversion.

    Parameters
    ----------
    declared_units : str
        ``VariableConfig.units`` — what the raw file reports.
    conversion : UnitConversion or None
        Conversion applied, if any.

    Returns
    -------
    str
        ``conversion.to_unit`` when converting, else the declared units.
        This string is what lands in the output dataset's attrs, so it must
        describe the numbers actually stored rather than the ones read.
    """
    return declared_units if conversion is None else conversion.to_unit
