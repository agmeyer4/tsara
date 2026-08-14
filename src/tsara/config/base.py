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


def validate_stream_name(value: str, *, field: str) -> None:
    """Validate a name that becomes an xarray key *and* a filename.

    Two kinds of TSARA name need this rule, for two reasons that happen to
    coincide:

    * **Species names** become ``xarray`` variable names, where a
      non-identifier defeats attribute access (``ds.ch4``) and reads badly in
      every downstream selection.
    * **Instrument (stream) names** become files inside a bundle,
      ``streams/<name>.nc``. A name containing a path separator sends the
      write into a directory that was never created, and the failure surfaces
      only at save time — after a full generate or ingest — as a bare
      ``PermissionError``/``FileNotFoundError`` from the netCDF backend that
      never mentions which instrument caused it. Naming an instrument after
      the archive path it came from is an easy mistake to make; this turns it
      into an immediate, named config error instead.

    The Python identifier rule is stricter than either use strictly requires,
    which is the point: one rule, checkable in one call, that is obviously
    sufficient for both.

    Parameters
    ----------
    value : str
        Candidate name.
    field : str
        Dotted field name used in the error message, e.g.
        ``'SyntheticConfig.instruments'``.

    Raises
    ------
    ValueError
        If ``value`` is not a valid Python identifier.
    """
    if not value.isidentifier():
        raise ValueError(
            f"{field}: '{value}' must be a valid identifier (letters, digits, "
            "underscores; not starting with a digit). Names become xarray "
            "variables and bundle filenames, so separators, spaces and dots "
            "are not usable."
        )


def validate_positive_timedelta(value: str, *, field: str) -> None:
    """Validate a pandas-style strictly positive timedelta string ('30s', '10min').

    For *durations* — window lengths, grid spacings, gap tolerances — where a
    non-positive value is physically meaningless. Signed time quantities
    (e.g. a future per-instrument ``time_shift`` for inlet-lag correction,
    which is legitimately negative) are a different physical animal and must
    NOT use this validator; add a parse-only ``validate_signed_timedelta``
    sibling when such a field first appears.

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


def validate_signed_timedelta(value: str, *, field: str) -> None:
    """Validate a pandas-style timedelta string of *either* sign ('30s', '-15s').

    The sibling anticipated by :func:`validate_positive_timedelta`, added when
    the first genuinely signed time quantity appeared: the synthetic
    generator's ``inter_species_lag`` (:mod:`tsara.synthetic.config`). A lag is
    an *offset*, not a duration — a species that arrives at the inlet 15 s
    *before* the reference species is a physically meaningful configuration
    (different stacks, different transport paths), so zero and negative values
    must both be accepted here. The same will apply to a per-instrument
    ``time_shift`` for inlet-lag correction whenever that lands.

    Raises ``ValueError`` (which Pydantic converts into a field-scoped
    validation error) if the string doesn't parse. Unlike its positive-only
    sibling it imposes no sign constraint at all; the pandas import is
    likewise deferred to call time to keep schema import fast.

    Parameters
    ----------
    value : str
        Candidate signed offset string.
    field : str
        Dotted field name used in the error message, e.g.
        ``'SourceSpec.inter_species_lag'``.
    """
    import pandas as pd

    try:
        pd.Timedelta(value)
    except ValueError as exc:
        raise ValueError(
            f"{field}: '{value}' is not a valid timedelta string (try '30s', '-15s', '2min')."
        ) from exc
