"""TSARA: Time Series and Ratio Analyses.

Ingests raw, multi-rate trace gas timeseries (stationary and mobile
platforms) as native-rate per-instrument streams, computes rolling baselines
and enhancements, detects plume events (including nested multi-scale plumes),
and extracts enhancement ratios with combined uncertainty quantification —
producing synchronized matrices ready for downstream receptor modeling
(e.g., PMF). The mathematics and rationale for every algorithm live in
``docs/METHODS.md``.

The public API is re-exported here so users can write ``from tsara import
load_manifest`` without memorizing the internal module layout. Each phase of
development extends this surface.
"""

from __future__ import annotations

import logging

from tsara.config.analysis import AnalysisConfig
from tsara.config.loader import (
    TsaraConfig,
    load_analysis,
    load_config,
    load_manifest,
    load_synthetic,
)
from tsara.config.manifest import Manifest
from tsara.core.exceptions import TsaraConfigError, TsaraError
from tsara.core.logutil import setup_logging

__version__ = "0.1.0"

# Library etiquette: never emit log output unless the application opts in
# (via tsara.setup_logging() or its own handler configuration).
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AnalysisConfig",
    "Manifest",
    "TsaraConfig",
    "TsaraConfigError",
    "TsaraError",
    "__version__",
    "load_analysis",
    "load_config",
    "load_manifest",
    "load_synthetic",
    "setup_logging",
]
