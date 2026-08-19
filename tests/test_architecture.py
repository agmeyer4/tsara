"""Tests that enforce architectural decisions rather than behaviour.

Every other test file asks "does this code compute the right answer?". These
ask "is the package still shaped the way we decided it should be?" — which no
amount of behavioural testing can answer, because a violation of either rule
below breaks nothing at all today. It only makes the next phase harder, which
is precisely why it needs a test rather than good intentions.

Both rules are already written down in prose (``tsara/core/__init__.py`` and
each module's ``__all__``). What was missing was anything that fails when the
prose stops being true.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "tsara"

#: Every module that re-exports a public surface. Adding a subpackage with an
#: ``__all__`` means adding it here.
PACKAGES_WITH_EXPORTS = ["tsara", "tsara.ingest", "tsara.synthetic"]


def _tsara_imports(module_path: Path) -> list[str]:
    """Return every ``tsara.*`` module name imported by one source file.

    Parsed with :mod:`ast` rather than by importing, so the check is static:
    it sees imports guarded by ``TYPE_CHECKING`` and imports nested inside
    functions (TSARA defers several heavy ones that way), both of which an
    import-and-inspect approach would miss entirely.

    Parameters
    ----------
    module_path : pathlib.Path
        Python source file to scan.

    Returns
    -------
    list of str
        Imported module names beginning ``tsara``.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tsara"):
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names if alias.name.startswith("tsara"))
    return found


@pytest.mark.parametrize("module_path", sorted((SRC / "core").glob("*.py")), ids=lambda p: p.name)
def test_core_imports_nothing_from_tsara_outside_core(module_path: Path) -> None:
    """``tsara.core`` must stay a leaf of the dependency graph.

    The invariant documented in ``tsara/core/__init__.py``: core modules may
    import each other and third-party libraries, but nothing else from
    ``tsara``. It is what distinguishes ``core`` from the usual "utils"
    package that nothing can ever be excluded from, and it guarantees that
    adding an import to core can never create a cycle with ``config``,
    ``synthetic``, or any stage added later.

    The failure this prevents is silent: importing ``tsara.config`` from a
    core module works fine right up until some future config module wants a
    core primitive, at which point the cycle appears far from its cause.
    """
    offenders = [name for name in _tsara_imports(module_path) if not name.startswith("tsara.core")]
    assert offenders == [], (
        f"{module_path.name} imports {offenders} from outside tsara.core. "
        "If a core module genuinely needs a stage, it is not core — move it "
        "to the stage that owns it."
    )


@pytest.mark.parametrize("package", PACKAGES_WITH_EXPORTS)
def test_every_exported_name_resolves(package: str) -> None:
    """``__all__`` must not promise names the package does not have.

    ``from tsara.synthetic import *`` raises AttributeError on a stale entry,
    but nothing in the suite does a star-import, so a name left behind by a
    rename would otherwise sit undetected until a user hit it.
    """
    module = importlib.import_module(package)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == [], f"{package}.__all__ lists names that do not exist: {missing}"


@pytest.mark.parametrize("package", PACKAGES_WITH_EXPORTS)
def test_exports_are_sorted_and_unique(package: str) -> None:
    """A sorted ``__all__`` keeps additions from colliding in review.

    Cosmetic on its own; the real value is that an alphabetical list makes a
    duplicate entry (the usual result of two branches adding an export) an
    obvious diff rather than an invisible one.
    """
    exported = list(importlib.import_module(package).__all__)
    assert exported == sorted(exported), f"{package}.__all__ is not alphabetically sorted."
    assert len(exported) == len(set(exported)), f"{package}.__all__ contains duplicates."


def test_the_two_synthetic_loaders_are_distinct_names() -> None:
    """Guards the Stage 7 rename against a well-meaning "consistency" revert.

    ``tsara.config.loader.load_synthetic`` reads a *config*;
    ``tsara.synthetic.bundle.load_bundle`` reads a *dataset*. Both take a
    path. They were briefly both called ``load_synthetic``, which made the
    meaning of a notebook line depend on which import was in scope.
    """
    import tsara
    import tsara.synthetic

    assert not hasattr(tsara.synthetic, "load_synthetic")
    assert not hasattr(tsara.synthetic, "save_synthetic")
    # Compared by defining module rather than by identity: mypy can prove two
    # differently-typed functions are never the same object, so an `is not`
    # check here is a tautology it rightly flags. Where each name *lives* is
    # the thing that actually has to stay true.
    assert tsara.load_synthetic.__module__ == "tsara.config.loader"
    assert tsara.synthetic.load_bundle.__module__ == "tsara.synthetic.bundle"
