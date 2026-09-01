"""Every shipped example config must actually load.

These files are the first thing a new user copies, and they are the only
documentation that can be *executed*. Nothing else in the suite reads them,
so without this a schema change would leave them quietly broken until
someone hit the error in a tutorial — the worst possible place to find it.

The check is deliberately about loading, not about content: an example's job
is to be a correct, current demonstration of the schema, and validation is
exactly what proves that.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from tsara import load_analysis, load_manifest, load_synthetic
from tsara.config import manifest as manifest_schema

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "configs"

#: Filename prefix -> the loader that file is meant to be read with. Keeping
#: the mapping explicit (rather than trying each loader until one works)
#: means an example that validates as the *wrong* kind of config still fails.
_LOADERS: dict[str, Callable[[Path], object]] = {
    "manifest_": load_manifest,
    "analysis_": load_analysis,
    "synthetic_": load_synthetic,
}


def _examples() -> list[Path]:
    return sorted(EXAMPLES.glob("*.yaml"))


def test_there_are_examples_to_check() -> None:
    """Guards against this whole file silently passing on an empty glob."""
    assert _examples()


@pytest.mark.parametrize("path", _examples(), ids=lambda p: p.name)
def test_example_loads(path: Path) -> None:
    for prefix, loader in _LOADERS.items():
        if path.name.startswith(prefix):
            assert loader(path) is not None
            return
    pytest.fail(
        f"'{path.name}' does not start with any known prefix {sorted(_LOADERS)}, "
        "so no loader claims it. Rename it or add its kind to _LOADERS."
    )


# ---------------------------------------------------------------------------
# The examples must stay representative, not merely valid
# ---------------------------------------------------------------------------
#
# `test_example_loads` proves an example is *correct*. It cannot notice an
# example going *stale*: a field added to the schema and demonstrated
# nowhere leaves the executable documentation quietly behind the code. When
# this check was written, seven manifest fields had no example at all,
# including `max_dropped_fraction` on all three loaders -- the control that
# decides whether a misparsed file raises or warns.

#: Manifest fields deliberately not demonstrated, each with the reason.
#:
#: Note that a field inherited from a shared base (``max_dropped_fraction``,
#: ``exclude``) is demonstrated for every loader at once, because the check
#: searches YAML keys and the key is the same. Exempting one loader's copy
#: while another shows it is therefore not a thing to do -- and
#: ``test_no_exemption_is_stale`` refuses it, which is how this comment came
#: to be written.
#:
#: An exemption is a *judgement*, so it is recorded rather than implied: the
#: alternative to this list is not a shorter test but an unwritten rule about
#: which fields "do not really need" an example, which is exactly the rule
#: that let seven of them disappear. Adding a schema field means either
#: showing it or saying here why not.
EXEMPT_FIELDS: dict[str, str] = {
    "CSVLoader.comment": "One-character comment marker; the ICARTT reader covers header skipping.",
    "FlagRule.bad_values": "The complement of good_values, which is demonstrated; identical shape.",
    "MobilePlatform.alt_variable": "Optional third GPS column; lat/lon show the binding.",
}

#: Models whose fields a manifest example is expected to reach.
_MANIFEST_MODELS = [
    "Manifest",
    "InstrumentConfig",
    "VariableConfig",
    "UnitConversion",
    "CSVLoader",
    "ICARTTLoader",
    "ParquetLoader",
    "TimeParsing",
    "StationaryPlatform",
    "MobilePlatform",
]


def _manifest_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in EXAMPLES.glob("manifest_*.yaml"))


def _spellings(model: type[BaseModel], name: str) -> set[str]:
    """Return every YAML key that sets this field, including aliases."""
    field = model.model_fields[name]
    names = {name}
    alias = getattr(field, "validation_alias", None)
    if alias is not None and hasattr(alias, "choices"):
        names |= {str(choice) for choice in alias.choices}
    return names


def test_every_manifest_field_is_demonstrated_or_exempt() -> None:
    """A schema field with no example is documentation that stopped tracking code."""
    text = _manifest_text()
    undemonstrated: list[str] = []
    for model_name in _MANIFEST_MODELS:
        model = getattr(manifest_schema, model_name)
        for field_name in model.model_fields:
            key = f"{model_name}.{field_name}"
            if key in EXEMPT_FIELDS:
                continue
            if not any(f"{spelling}:" in text for spelling in _spellings(model, field_name)):
                undemonstrated.append(key)

    assert undemonstrated == [], (
        "These manifest fields appear in no shipped example: "
        f"{sorted(undemonstrated)}. Add one to examples/configs/, or record "
        "the reason in EXEMPT_FIELDS."
    )


def test_no_exemption_is_stale() -> None:
    """An exemption for a field that no longer exists, or that is now shown
    anyway, is a note nobody will re-read. Both make the list less trustworthy."""
    text = _manifest_text()
    unknown: list[str] = []
    redundant: list[str] = []
    for key in EXEMPT_FIELDS:
        model_name, _, field_name = key.partition(".")
        model = getattr(manifest_schema, model_name, None)
        if model is None or field_name not in model.model_fields:
            unknown.append(key)
            continue
        if any(f"{spelling}:" in text for spelling in _spellings(model, field_name)):
            redundant.append(key)

    assert unknown == [], f"EXEMPT_FIELDS names fields that no longer exist: {unknown}"
    assert redundant == [], (
        f"EXEMPT_FIELDS excuses fields that ARE now demonstrated: {redundant}. Drop them."
    )
