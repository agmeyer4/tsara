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

from tsara import load_analysis, load_manifest, load_synthetic

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
