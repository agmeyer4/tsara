"""Exception hierarchy for TSARA.

Every exception TSARA raises deliberately derives from :class:`TsaraError`,
so downstream users (and our own CLI) can distinguish "TSARA rejected your
input / hit a scientific edge case" from genuine programming bugs with a
single ``except TsaraError`` clause. Keep this module dependency-free: it is
imported by everything else in the package.
"""

from __future__ import annotations


class TsaraError(Exception):
    """Base class for all errors raised by TSARA."""


class TsaraConfigError(TsaraError):
    """Raised when a YAML configuration file cannot be loaded or validated.

    This wraps lower-level failures (missing file, YAML syntax error,
    Pydantic validation error) so that users always get the *path* of the
    offending file alongside the underlying cause. The original exception is
    chained via ``raise ... from`` so full detail is never lost.
    """
