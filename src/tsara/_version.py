"""The one place TSARA's version number is written.

Every product this package saves records the version that made it (the
``tsara_version`` attribute on streams, bundles and joined datasets), so the
stages that write provenance need the number. They import it from here and
never from the package root: :mod:`tsara` imports the config package to
re-export the loaders, and the day it re-exports a stage as well, a stage
importing the root would be a circular import that surfaces as a mysterious
``ImportError`` on ``from tsara import __version__``. A leaf module with one
assignment cannot take part in any cycle.

``pyproject.toml`` reads the same attribute (``dynamic = ["version"]``), so a
release is one edit, here.
"""

from __future__ import annotations

__version__ = "0.1.0"
