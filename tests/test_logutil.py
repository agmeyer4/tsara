"""Tests for tsara.logutil (library-vs-application logging configuration).

setup_logging() mutates a process-global object (the "tsara" logger
singleton), so every test here uses the clean_tsara_logger fixture to
snapshot and restore it -- otherwise one test's handlers/level would leak
into the next test, and into the rest of the suite.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tsara.logutil import PACKAGE_LOGGER_NAME, setup_logging

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_tsara_logger():
    """Snapshot the 'tsara' logger's state and restore it after the test.

    Only handlers *added during the test* are removed (and closed, so file
    handlers release their file descriptor); anything present beforehand --
    e.g. the NullHandler tsara/__init__.py installs at import time -- is left
    exactly as found.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    yield logger
    for handler in list(logger.handlers):
        if handler not in original_handlers:
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(original_level)
    logger.propagate = original_propagate


# ---------------------------------------------------------------------------
# Level handling
# ---------------------------------------------------------------------------


def test_level_applied_as_int(clean_tsara_logger):
    logger = setup_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_level_applied_as_string(clean_tsara_logger):
    """The stdlib itself validates level names; setup_logging adds no
    parsing of its own, so passing 'WARNING' straight through must work."""
    logger = setup_logging(level="WARNING")
    assert logger.level == logging.WARNING


# ---------------------------------------------------------------------------
# Idempotency (safe to call repeatedly, e.g. a re-run notebook cell)
# ---------------------------------------------------------------------------


def test_repeated_calls_do_not_duplicate_handlers(clean_tsara_logger):
    setup_logging()
    count_after_first_call = len(clean_tsara_logger.handlers)

    setup_logging()
    setup_logging()

    assert len(clean_tsara_logger.handlers) == count_after_first_call


def test_repeated_calls_with_logfile_do_not_duplicate_handlers(
    clean_tsara_logger, tmp_path: Path
):
    logfile = tmp_path / "tsara.log"
    setup_logging(logfile=logfile)
    count_after_first_call = len(clean_tsara_logger.handlers)

    setup_logging(logfile=logfile)

    assert len(clean_tsara_logger.handlers) == count_after_first_call


def test_user_attached_handlers_survive_repeated_setup(clean_tsara_logger):
    """setup_logging must remove only the handlers *it* previously attached,
    never a handler the host application attached itself."""
    user_handler = logging.NullHandler()
    clean_tsara_logger.addHandler(user_handler)

    setup_logging()
    setup_logging()

    assert user_handler in clean_tsara_logger.handlers


# ---------------------------------------------------------------------------
# Propagation and actual message delivery
# ---------------------------------------------------------------------------


def test_does_not_propagate_to_root(clean_tsara_logger):
    """Prevents double-logging in a host application that configures its own
    root logger."""
    setup_logging()
    assert clean_tsara_logger.propagate is False


def test_console_handler_writes_to_stderr(clean_tsara_logger, capsys):
    """pytest's caplog fixture only captures records that reach the *root*
    logger's handler; since setup_logging sets propagate=False, records
    never get there no matter which logger= is passed to caplog.at_level
    (confirmed empirically -- that's not how at_level works, it only raises
    the level threshold on the named logger). capsys, which intercepts the
    stream StreamHandler actually writes to, is the correct tool here.
    """
    setup_logging(level=logging.INFO)
    logging.getLogger(f"{PACKAGE_LOGGER_NAME}.somemodule").info("via console handler")
    captured = capsys.readouterr()
    assert "via console handler" in captured.err


def test_logfile_receives_formatted_messages(clean_tsara_logger, tmp_path: Path):
    logfile = tmp_path / "tsara.log"
    setup_logging(level=logging.INFO, logfile=logfile)

    logging.getLogger(f"{PACKAGE_LOGGER_NAME}.somemodule").info("hello from a submodule")

    contents = logfile.read_text(encoding="utf-8")
    assert "hello from a submodule" in contents
    # LOG_FORMAT includes %(name)s -- confirms the real formatter ran, not
    # just that *some* text landed in the file.
    assert f"{PACKAGE_LOGGER_NAME}.somemodule" in contents
    assert "INFO" in contents
