"""Foundational primitives shared by every TSARA subsystem.

What belongs here
-----------------
Modules that the rest of the package is built *on top of*, rather than
modules that do science. Two kinds currently live here:

* **Infrastructure** — :mod:`tsara.core.exceptions` (the ``TsaraError``
  hierarchy) and :mod:`tsara.core.logutil` (the library/application logging
  split).
* **Domain primitives** — :mod:`tsara.core.timebase` (how TSARA represents
  time: UTC-internal, integer nanoseconds where exactness matters) and
  :mod:`tsara.core.geodesy` (how it represents position: a local
  equirectangular approximation). These encode *representational decisions*
  that every later stage inherits, which is exactly why they cannot belong
  to any one stage.

The rule that keeps this from becoming a junk drawer
----------------------------------------------------
**Modules in this package may import each other and third-party libraries,
but nothing else from** ``tsara``. That is a checkable invariant, unlike the
usual "utilities" convention where nothing can ever be excluded on principle.
It also makes the dependency direction unambiguous: ``core`` is a leaf, so
adding an import here can never create a cycle with ``config``,
``synthetic``, or any future stage.

A concrete consequence: if a would-be "core" module needs to import
:mod:`tsara.config`, it is not core — it is a stage, and belongs with the
stage that owns it.

Submodules, in reading order
----------------------------
exceptions
    The ``TsaraError`` hierarchy every stage's own error derives from.
logutil
    The library/application logging split: nothing is printed until the
    application asks.
timebase
    How TSARA represents time: UTC-internal, integer nanoseconds where
    exactness matters, and the one home of every time-unit conversion.
naming
    The vocabulary every stage must agree on: coordinate and companion-column
    names, the attribute keys products carry, the support and provenance
    labels.
geodesy
    How TSARA represents position: a local equirectangular approximation.
support
    Cells: the interval of air a value describes, CF bounds, and the one
    overlap-weighted binning arithmetic every join calls.
propagation
    Uncertainty through a mean: the random component that averages down, the
    systematic one that does not, and the correlated forms between.
circular
    Angular statistics: wrapping, the mean resultant length, the exact
    dispersion, and vector averaging onto cells.
bundle
    The on-disk bundle convention shared by every stage that saves itself.

Public API
----------
Nothing here is re-exported from this package's own ``__init__``; the
user-facing names (``TsaraError``, ``setup_logging``, ...) are re-exported
from the top-level :mod:`tsara` namespace instead, so users never need to
know this layout exists.
"""

from __future__ import annotations
