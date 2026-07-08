"""Shared foundations for all TSARA configuration models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for all config models: reject unknown keys, freeze after load.

    * ``extra="forbid"``: a typo like ``quantles:`` in a science config
      silently changing results is the worst failure mode a config-driven
      package can have, so every model rejects unknown keys loudly.
    * ``frozen=True``: once validated, a config object is a fact about the
      run. Anything that needs to "modify" one (e.g. the loader resolving
      relative paths) must build a new object via ``model_copy(update=...)``,
      keeping provenance auditable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def validate_timedelta(value: str, *, field: str) -> None:
    """Validate a pandas-style positive timedelta string ('30s', '10min').

    Raises ``ValueError`` (which Pydantic converts into a field-scoped
    validation error) if the string doesn't parse or is non-positive. The
    pandas import is deferred to call time so that merely importing the
    config schemas stays fast — relevant for CLI ``--help`` responsiveness.

    Parameters
    ----------
    value : str
        Candidate duration string.
    field : str
        Dotted field name used in the error message, e.g.
        ``'GridConfig.freq'``.
    """
    import pandas as pd

    try:
        td = pd.Timedelta(value)
    except ValueError as exc:
        raise ValueError(
            f"{field}: '{value}' is not a valid timedelta string (try '30s', '10min', '1h')."
        ) from exc
    if td <= pd.Timedelta(0):
        raise ValueError(f"{field}: duration must be positive, got '{value}'.")
