"""The file-format reader registry.

Why a registry rather than ``if loader.format == "csv": ...``
--------------------------------------------------------------
Three reasons, in increasing order of importance.

1. **It matches the pattern the rest of TSARA already uses.** Noise
   estimators and regression estimators are registered by name
   (``docs/METHODS.md`` §2.5, §4), and every registered name owes a METHODS
   section. File formats are the same kind of thing: a swappable
   implementation selected by a string in a config file.
2. **New formats become additive.** The campaign archives this package
   targets already contain parquet and GPX alongside the delimited text and
   ICARTT that Phase 3 implements. Each of those is a future one-file
   change: write the reader, decorate it, done — no dispatch table to
   remember to update, and nothing in QA/QC or stream assembly to touch.
3. **Extension without forking.** A group with an in-house binary format can
   register a reader from their own code and use a stock TSARA. That is only
   true if the extension point is public, which a private ``if`` chain is
   not.

The cost is one thing to remember: **a reader only exists once its module has
been imported.** :mod:`tsara.ingest` imports the built-in reader modules for
exactly this reason, so importing the package is sufficient for everything
TSARA ships.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from tsara.ingest.base import RawTable, Reader, TsaraIngestError, check_raw_table

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from tsara.config.manifest import LoaderConfig

logger = logging.getLogger(__name__)

__all__ = [
    "available_readers",
    "get_reader",
    "read_file",
    "register_reader",
]

#: Registered readers keyed by the manifest's ``loader.format`` discriminator.
#: Module-private: mutation goes through :func:`register_reader` so that the
#: duplicate-name check can never be bypassed by accident.
_READERS: dict[str, Reader] = {}

#: TypeVar bound to the protocol so the decorator returns the *same* function
#: type it received. Without this, decorating a reader would erase its precise
#: signature and callers would lose type checking on direct calls.
R = TypeVar("R", bound=Reader)


def register_reader(name: str, *, replace: bool = False) -> Callable[[R], R]:
    """Register a file reader under a ``loader.format`` name.

    Used as a decorator::

        @register_reader("csv")
        def read_csv(path: Path, loader: LoaderConfig, /) -> RawTable:
            ...

    The name must match the ``format`` discriminator of the corresponding
    :data:`~tsara.config.manifest.LoaderConfig` member, since that string is
    what the manifest supplies at dispatch time.

    Re-registering a name
    ---------------------
    Registration refuses to overwrite by default, because a *silent* overwrite
    makes ingestion depend on module import order — importing a plugin twice
    would change which code reads your files, with nothing to see. That
    argument condemns silence, not overwriting, so ``replace=True`` is
    available for the case where the caller means it and says so.

    The case that needs it is interactive: re-running a notebook cell that
    defines a reader, or ``importlib.reload`` on a plugin module, re-executes
    the decorator. Without an override those raise, and the only way forward
    is a kernel restart. TSARA is meant to be driven step-by-step from a
    notebook, so making that a dead end would be a real cost. An override is
    logged at warning level: stated intent is fine, but it should still be
    visible in a run log when someone is working out why a file parsed oddly.

    Parameters
    ----------
    name : str
        Format name, e.g. ``'csv'``.
    replace : bool, optional
        Replace an existing reader registered under ``name``. Default
        ``False``, which raises instead.

    Returns
    -------
    callable
        Decorator returning its argument unchanged, so the decorated function
        stays directly callable and testable without going through dispatch.

    Raises
    ------
    ValueError
        If ``name`` is blank, or already registered and ``replace`` is False.
    """
    if not name or not name.strip():
        raise ValueError("Reader name must be a non-empty string.")

    def decorator(func: R) -> R:
        existing = _READERS.get(name)
        if existing is not None:
            if not replace:
                raise ValueError(
                    f"A reader is already registered for format '{name}' "
                    f"({_describe(existing)}). Pass replace=True if you meant to "
                    "override it, so that ingestion never depends on import order."
                )
            logger.warning(
                "Replacing the reader registered for format '%s': %s -> %s",
                name,
                _describe(existing),
                _describe(func),
            )
        _READERS[name] = func
        logger.debug("Registered reader '%s' -> %s", name, getattr(func, "__qualname__", func))
        return func

    return decorator


def _describe(func: Reader) -> str:
    """Return ``module.qualname`` for a reader, for use in messages."""
    return f"{getattr(func, '__module__', '?')}.{getattr(func, '__qualname__', '?')}"


def available_readers() -> tuple[str, ...]:
    """List the registered format names, sorted.

    Returns
    -------
    tuple of str
        Format names currently available, e.g. ``('csv', 'icartt')``.
    """
    return tuple(sorted(_READERS))


def get_reader(name: str) -> Reader:
    """Look up a reader by format name.

    Parameters
    ----------
    name : str
        Format name from the manifest's ``loader.format``.

    Returns
    -------
    Reader
        The registered reader.

    Raises
    ------
    TsaraIngestError
        If no reader is registered under ``name``. The message lists what
        *is* available, because the usual cause is a plugin module that was
        never imported, and seeing the actual list makes that obvious.
    """
    try:
        return _READERS[name]
    except KeyError:
        raise TsaraIngestError(
            f"No reader registered for format '{name}'. Available: "
            f"{list(available_readers())}. A reader must have been imported "
            "to be registered."
        ) from None


def read_file(path: str | Path, loader: LoaderConfig) -> RawTable:
    """Read one raw file using the reader its loader config selects.

    The single entry point through which all file reading flows, which is
    what lets :func:`~tsara.ingest.base.check_raw_table` police the
    :class:`~tsara.ingest.base.RawTable` contract for every reader — TSARA's
    own and anyone else's — rather than trusting each to police itself.

    Parameters
    ----------
    path : str or pathlib.Path
        File to read.
    loader : LoaderConfig
        Validated loader configuration; its ``format`` selects the reader.

    Returns
    -------
    RawTable
        Contract-checked parsed file.

    Raises
    ------
    TsaraIngestError
        If the file does not exist, no reader is registered for the format,
        the reader fails, or its output violates the contract.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise TsaraIngestError(f"Cannot read '{file_path}': not an existing file.")

    reader = get_reader(loader.format)
    logger.debug("Reading %s with reader '%s'", file_path, loader.format)
    try:
        table = reader(file_path, loader)
    except TsaraIngestError:
        # Already carries file context from the reader; re-wrapping would only
        # bury the specific message under a generic one.
        raise
    except Exception as exc:
        # Anything else (a pandas ParserError, a UnicodeDecodeError, an
        # IndexError on a truncated header) becomes a TsaraError naming the
        # file. On an archive of several hundred files, "which one?" is the
        # only question worth answering first.
        raise TsaraIngestError(f"Reader '{loader.format}' failed on '{file_path}': {exc}") from exc

    return check_raw_table(table, reader_name=loader.format)
