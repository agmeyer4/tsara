"""Logging configuration helpers for TSARA.

Design
------
TSARA follows the standard library-vs-application logging split:

* **As a library**, the package must never configure the root logger or
  emit output the host application didn't ask for. The package root logger
  (``"tsara"``) therefore gets a :class:`logging.NullHandler` (installed in
  ``tsara/__init__.py``), and every module obtains its logger with
  ``logging.getLogger(__name__)`` so messages are namespaced like
  ``tsara.io.crawler``.

* **As an application** (the CLI, or an interactive user who wants to see
  progress in a notebook), :func:`setup_logging` attaches real handlers to
  the ``"tsara"`` logger only — never the root — so TSARA's verbosity can be
  tuned without drowning out (or being drowned out by) other packages.
"""

from __future__ import annotations

import logging
from pathlib import Path

#: Name of the package's root logger; all module loggers are children of it.
PACKAGE_LOGGER_NAME = "tsara"

#: Timestamped format with the module path — on an HPC batch job the module
#: path is often the only clue to *where* in the pipeline a message came from.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int | str = logging.INFO,
    logfile: str | Path | None = None,
) -> logging.Logger:
    """Attach console (and optionally file) handlers to the ``tsara`` logger.

    Safe to call repeatedly (e.g., re-running a notebook cell): existing
    handlers attached by a previous call are removed first, so log lines are
    never duplicated.

    Parameters
    ----------
    level : int or str, default logging.INFO
        Threshold for the ``tsara`` logger, e.g. ``logging.DEBUG`` or
        ``"DEBUG"``.
    logfile : str or pathlib.Path, optional
        If given, messages are *also* appended to this file — useful for
        headless batch runs where stderr scrolls away.

    Returns
    -------
    logging.Logger
        The configured ``tsara`` package logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)

    # Idempotency: drop handlers we previously attached (identified by the
    # marker attribute below) but leave any user-attached handlers alone.
    for handler in list(logger.handlers):
        if getattr(handler, "_tsara_managed", False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console._tsara_managed = True  # type: ignore[attr-defined]
    logger.addHandler(console)

    if logfile is not None:
        filehandler = logging.FileHandler(Path(logfile), mode="a", encoding="utf-8")
        filehandler.setFormatter(formatter)
        filehandler._tsara_managed = True  # type: ignore[attr-defined]
        logger.addHandler(filehandler)

    # Messages should not also bubble to the root logger once we have our own
    # handlers; otherwise applications with a configured root logger would
    # see every line twice.
    logger.propagate = False
    return logger
