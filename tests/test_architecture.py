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
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "tsara"

#: Every module that re-exports a public surface. Adding a subpackage with an
#: ``__all__`` means adding it here.
PACKAGES_WITH_EXPORTS = ["tsara", "tsara.ingest", "tsara.synthetic"]


def _modules_declaring_all() -> list[str]:
    """Return every ``tsara`` module that declares ``__all__``, discovered.

    Enumerated by walking the package tree rather than by listing names,
    because a hand-maintained list only guards what someone remembered to
    add to it. Measured when this replaced a three-item list: 21 modules
    declare ``__all__`` and 3 were being checked, and two of the unchecked
    ones had been unsorted for a whole phase without anything noticing.

    Parsed with :mod:`ast` rather than imported, so a module is discovered
    whether or not importing it has side effects.

    Returns
    -------
    list of str
        Importable dotted module names, sorted.
    """
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        declares = any(
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) == "__all__" for target in node.targets)
            for node in tree.body
        )
        if not declares:
            continue
        dotted = ".".join(path.relative_to(SRC.parent).with_suffix("").parts)
        found.append(dotted.removesuffix(".__init__"))
    return found


#: Every module with a public surface, found rather than remembered.
MODULES_WITH_EXPORTS = _modules_declaring_all()


def _tsara_imports(module_path: Path) -> list[str]:
    """Return every ``tsara.*`` module name imported by one module.

    Parsed with :mod:`ast` rather than by importing, so the check is static:
    it sees imports guarded by ``TYPE_CHECKING`` and imports nested inside
    functions (TSARA defers several heavy ones that way), both of which an
    import-and-inspect approach would miss entirely.

    Parameters
    ----------
    module_path : pathlib.Path
        Python file to scan.

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


@pytest.mark.parametrize("package", MODULES_WITH_EXPORTS)
def test_every_exported_name_resolves(package: str) -> None:
    """``__all__`` must not promise names the package does not have.

    ``from tsara.synthetic import *`` raises AttributeError on a stale entry,
    but nothing in the suite does a star-import, so a name left behind by a
    rename would otherwise sit undetected until a user hit it.
    """
    module = importlib.import_module(package)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == [], f"{package}.__all__ lists names that do not exist: {missing}"


@pytest.mark.parametrize("package", MODULES_WITH_EXPORTS)
def test_exports_are_sorted_and_unique(package: str) -> None:
    """A sorted ``__all__`` keeps additions from colliding in review.

    Cosmetic on its own; the real value is that an alphabetical list makes a
    duplicate entry (the usual result of two branches adding an export) an
    obvious diff rather than an invisible one.

    Sorted by Python's own ordering, which puts every capitalized name
    before every lowercase one -- so ``RawTable`` precedes
    ``TIME_INDEX_NAME`` precedes ``check_raw_table``. Worth stating because
    the intuitive case-insensitive reading disagrees, and that disagreement
    is what two of these lists drifted on.
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


def test_the_ingest_time_index_name_is_the_stream_time_coordinate() -> None:
    """The reader contract and the finished stream must name one axis.

    ``ingest.base.TIME_INDEX_NAME`` is what every reader's raw index is
    checked against; ``core.naming.TIME_COORD`` is what the stream assembler
    names the finished coordinate. They were once two independent literals
    that both happened to say ``"time"`` -- the precise coupling
    ``tsara.core.naming`` exists to remove, reproduced one directory over.
    Identity, not equality, is the thing worth asserting: equal literals is
    the state this guards against.
    """
    from tsara.core.naming import TIME_COORD
    from tsara.ingest.base import TIME_INDEX_NAME

    assert TIME_INDEX_NAME is TIME_COORD


def test_the_two_bundle_writers_declare_different_stages() -> None:
    """Both layouts share a skeleton, so the stage key is what separates them.

    A synthetic bundle and an ingest bundle are both a ``bundle.json`` at
    format version 1 beside a ``streams/`` directory. Nothing in the shared
    skeleton distinguishes them, which is why each loader checks the stage
    before touching its own files -- and why the two stage values must not
    collide.
    """
    from tsara.ingest.bundle import _STAGE as ingest_stage
    from tsara.synthetic.bundle import _STAGE as synthetic_stage

    assert ingest_stage != synthetic_stage


def test_export_discovery_finds_more_than_the_packages() -> None:
    """Guards the discovery above from silently degrading to an empty list.

    A parametrized test over zero cases passes, so a walk that stops
    matching would turn this whole file green while checking nothing.
    """
    assert len(MODULES_WITH_EXPORTS) > len(PACKAGES_WITH_EXPORTS)
    assert set(PACKAGES_WITH_EXPORTS) <= set(MODULES_WITH_EXPORTS)


#: Public names that legitimately have no caller inside ``src/``.
#:
#: Kept explicit and small. Two dead helpers reached review in one phase --
#: ``floor_width``, which guarded a case that occurs on 644 real rows, and
#: ``weakest``, which reconciled something the architecture never assembles --
#: and both were written, documented and unit-tested. A test that only their
#: authors' own expectations guard is not much of a guard, so the rule is now
#: enforced and every exception has to say why.
UNREFERENCED: dict[str, str] = {
    "bin_onto_cells": (
        "The cross-rate pairing primitive. Phase 4 is its first caller; until "
        "then it is reached only by its own tests."
    ),
    "read_icartt": (
        "Reached by name through the reader registry (@register_reader), "
        "which is a lookup rather than a call site the parser can see."
    ),
    "propagate_systematic": (
        "The readable reference implementation of §3.3, which the vectorized "
        "`propagate_systematic_binned` is scored against by test and which "
        "the Monte Carlo checks measure. Its random sibling gained a caller "
        "when the pairwise form became selectable through the binned path -- "
        "which is exactly why these were listed separately. This one has no "
        "equivalent, since a systematic component has no pairwise form; if it "
        "never gains a caller, it goes on the ballot at the walkthrough."
    ),
    "circular_mean": (
        "The single-window form of angular averaging, for a caller holding "
        "one set of directions rather than a stream. Its binning sibling is "
        "now wired into `align.binning`, which is exactly why these were "
        "listed separately; if no caller ever wants the single-window form, "
        "it goes on the ballot at the walkthrough."
    ),
}


def _public_names() -> dict[str, str]:
    """Return every module-level ``__all__`` entry, mapped to its module."""
    found: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(t, "id", None) == "__all__" for t in node.targets):
                continue
            assert isinstance(node.value, ast.List)  # noqa: S101 - __all__ is a list
            for element in node.value.elts:
                assert isinstance(element, ast.Constant)  # noqa: S101
                found[str(element.value)] = str(path.relative_to(SRC))
    return found


def _referenced() -> set[str]:
    """Return every name loaded anywhere in ``src/``, plus package re-exports.

    A name re-exported from a package ``__init__`` is a public entry point by
    definition, so it needs no caller inside the package.
    """
    names: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if path.name == "__init__.py":
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                if not any(getattr(t, "id", None) == "__all__" for t in node.targets):
                    continue
                assert isinstance(node.value, ast.List)  # noqa: S101
                names |= {str(e.value) for e in node.value.elts if isinstance(e, ast.Constant)}
        for expr in ast.walk(tree):
            if isinstance(expr, ast.Name):
                names.add(expr.id)
            elif isinstance(expr, ast.Attribute):
                names.add(expr.attr)
    return names


def test_no_public_name_is_unreachable() -> None:
    """A helper nothing calls is a claim about the architecture, not a guard.

    Asked of the syntax tree rather than the text: a first attempt at this
    audit counted the word 'weakest' inside its own docstring and concluded
    the function was used. Prose is not a call site.
    """
    declared = _public_names()
    used = _referenced()
    orphans = sorted(n for n in declared if n not in used and n not in UNREFERENCED)
    assert orphans == [], (
        "These public names are never referenced from src/: "
        f"{ {n: declared[n] for n in orphans} }. Either wire them in, delete "
        "them, or record why in UNREFERENCED."
    )


def test_no_unreferenced_exemption_is_stale() -> None:
    """An exemption for a name that has gained a caller, or that no longer
    exists, makes the list less trustworthy than no list."""
    declared = _public_names()
    used = _referenced()
    gone = sorted(n for n in UNREFERENCED if n not in declared)
    now_used = sorted(n for n in UNREFERENCED if n in used)
    assert gone == [], f"UNREFERENCED names that no longer exist: {gone}"
    assert now_used == [], f"UNREFERENCED names that now have a caller: {now_used}. Drop them."


# ---------------------------------------------------------------------------
# Every attribute a product carries is documented
# ---------------------------------------------------------------------------

#: The methods document is where a reader looks up what an attribute means, so
#: an attribute nobody wrote down is a number in a file with no definition.
#: Exemptions need a reason, and `test_no_attr_exemption_is_stale` deletes the
#: reason's shelf life.
EXEMPT_ATTRS: dict[str, str] = {}

#: Attribute names are TSARA's own vocabulary when they carry one of these
#: prefixes. CF's own names (`bounds`, `cell_methods`, `units`) are governed by
#: CF and documented where they are used.
ATTR_PREFIXES = ("tsara_", "uncertainty_")


def _attr_names() -> dict[str, str]:
    """Every namespaced attribute name written anywhere in ``src/``.

    Discovered from string constants in the syntax tree rather than from a
    list, because the two families are spelled differently in the code: the
    support attributes are constants in ``core.naming``, the uncertainty ones
    are literals at the point of use. A test that enumerated either one by
    hand would cover half the vocabulary and look complete.
    """
    found: dict[str, str] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            name = node.value
            if name.startswith(ATTR_PREFIXES) and name.replace("_", "").isalnum():
                found.setdefault(name, f"{path.relative_to(SRC)}:{node.lineno}")
    return found


def _methods_text() -> str:
    return (SRC.parent.parent / "docs" / "METHODS.md").read_text()


def test_discovery_finds_the_attribute_vocabulary() -> None:
    """A test parametrized over nothing passes. Guard the discovery itself."""
    names = _attr_names()
    assert len(names) >= 10, f"attribute discovery found only {sorted(names)}"


def test_every_attribute_a_product_carries_is_documented() -> None:
    """An attribute is a promise to a reader six months later, and the only
    place that promise is explained is METHODS.md.

    This has failed for real: both provenance families were documented by
    halves -- the support section named the label and width attributes but
    abbreviated the method one, and the uncertainty
    section named the species-level label while the per-component attributes
    that carry it went unmentioned.
    """
    doc = _methods_text()
    undocumented = {
        name: where
        for name, where in _attr_names().items()
        if f"`{name}`" not in doc and name not in EXEMPT_ATTRS
    }
    assert undocumented == {}, (
        "These attributes are written into products but named nowhere in "
        f"docs/METHODS.md: {undocumented}. Document them where their family "
        "is discussed, or record why not in EXEMPT_ATTRS."
    )


#: Attributes METHODS names deliberately although nothing writes them, e.g. a
#: renamed or removed one discussed as history. Each needs a reason, and
#: `test_no_documented_attr_exemption_is_stale` gives the reason a shelf life.
DOCUMENTED_ONLY_ATTRS: dict[str, str] = {}


def test_the_document_names_no_attribute_that_nothing_writes() -> None:
    """The reverse of the check above, and the direction a rename breaks.

    `test_every_attribute_a_product_carries_is_documented` walks from the
    code to the document, so it notices a *new* attribute with no
    definition. It cannot notice the opposite: rename an attribute and its old
    name sits in the document forever, looking documented and describing
    nothing.

    That is not hypothetical. `tsara_pairing_binned` became `tsara_binned`
    when pairing was refactored into a thin layer over the binner, and the
    old name survived in the pairing section's attribute table because every
    test looked the other way.
    """
    doc = _methods_text()
    written = set(_attr_names())
    documented = {
        name
        for name in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", doc)
        if name.startswith(ATTR_PREFIXES)
    }
    orphans = sorted(documented - written - set(DOCUMENTED_ONLY_ATTRS))
    assert orphans == [], (
        f"docs/METHODS.md documents attributes nothing in src/ writes: {orphans}. "
        "Rename them to match, delete them, or record the reason in "
        "DOCUMENTED_ONLY_ATTRS."
    )


def test_no_documented_attr_exemption_is_stale() -> None:
    """An excuse for an attribute that is written again, or never named, rots."""
    doc = _methods_text()
    written = set(_attr_names())
    documented = {
        name
        for name in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", doc)
        if name.startswith(ATTR_PREFIXES)
    }
    revived = sorted(a for a in DOCUMENTED_ONLY_ATTRS if a in written)
    absent = sorted(a for a in DOCUMENTED_ONLY_ATTRS if a not in documented)
    assert revived == [], f"DOCUMENTED_ONLY_ATTRS excuses attributes now written: {revived}"
    assert absent == [], f"DOCUMENTED_ONLY_ATTRS names attributes the document lost: {absent}"


def test_no_attr_exemption_is_stale() -> None:
    """An exemption for an attribute that is gone, or that is now documented
    anyway, makes the list less trustworthy than no list."""
    doc = _methods_text()
    names = _attr_names()
    gone = sorted(a for a in EXEMPT_ATTRS if a not in names)
    documented = sorted(a for a in EXEMPT_ATTRS if f"`{a}`" in doc)
    assert gone == [], f"EXEMPT_ATTRS names attributes no longer written: {gone}"
    assert documented == [], (
        f"EXEMPT_ATTRS excuses attributes that ARE documented: {documented}. Drop them."
    )
