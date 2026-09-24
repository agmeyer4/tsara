"""Synthetic trace-gas data with exactly known ground truth.

TSARA has no controlled-release measurements to validate against (CLAUDE.md
§5), so injected synthetic truth is the only arbiter of detection and ratio
correctness in v1. This subpackage manufactures that truth: multi-rate
instrument streams carrying plumes of known amplitude, shape, nesting, and
inter-species ratio, contaminated by error of known random/systematic
decomposition — plus the answer key needed to score any algorithm against it.

It doubles as the test harness for every later phase: alignment, baselines,
detection, regression, and UQ are all developed against data generated here
before real files are readable.

The schema this stage consumes -- the atmosphere (fields, backgrounds,
sources) and the instruments measuring it -- is
:class:`~tsara.config.synthetic.SyntheticConfig`, which lives with the other
schemas in :mod:`tsara.config` and is re-exported here.

Submodules
----------
profiling
    Measuring real data's statistical shape, so synthetic parameters are
    grounded in reality rather than guessed.
background
    Realizing one field's plume-free signal, parametrically or by
    block-bootstrap from a real-data profile.
plumes
    Plume shape kernels, Poisson event scheduling, and the ground-truth
    catalog.
atmosphere
    The realized air: every field's true value at any time, which every
    instrument samples.
noise
    Two-component error injection (random, optionally AR(1)-correlated;
    systematic, correlated by construction) and quantization.
platform
    Fixed-site coordinates and synthetic mobile GPS tracks.
generator
    The orchestrator: realize the atmosphere, then let each instrument
    sample it.
export
    A generated dataset written out as the raw files and manifest an archive
    would hold: the round-trip harness ingestion is checked against.
bundle
    Reading and writing the on-disk TSARA bundle directory.
"""

from __future__ import annotations

from tsara.config.synthetic import (
    AtmosphereSpec,
    BootstrapBackground,
    DropoutSpec,
    EMGShape,
    FieldSpec,
    GaussianShape,
    InstrumentSpec,
    LognormalAmplitude,
    MeasurementSpec,
    MobileTrack,
    NestedSpec,
    ParametricBackground,
    RatioSpec,
    SourceSpec,
    StationarySite,
    SyntheticConfig,
    TrueComponent,
    TrueUncertainty,
    UniformAmplitude,
)
from tsara.synthetic.atmosphere import Atmosphere, realize_atmosphere
from tsara.synthetic.background import TsaraSyntheticError
from tsara.synthetic.bundle import TsaraBundleError, load_bundle, save_bundle
from tsara.synthetic.export import export_raw
from tsara.synthetic.generator import SyntheticDataset, generate
from tsara.synthetic.plumes import GroundTruth, GroundTruthEvent
from tsara.synthetic.profiling import (
    RealDataProfile,
    TsaraProfilingError,
    profile_series,
)

__all__ = [
    "Atmosphere",
    "AtmosphereSpec",
    "BootstrapBackground",
    "DropoutSpec",
    "EMGShape",
    "FieldSpec",
    "GaussianShape",
    "GroundTruth",
    "GroundTruthEvent",
    "InstrumentSpec",
    "LognormalAmplitude",
    "MeasurementSpec",
    "MobileTrack",
    "NestedSpec",
    "ParametricBackground",
    "RatioSpec",
    "RealDataProfile",
    "SourceSpec",
    "StationarySite",
    "SyntheticConfig",
    "SyntheticDataset",
    "TrueComponent",
    "TrueUncertainty",
    "TsaraBundleError",
    "TsaraProfilingError",
    "TsaraSyntheticError",
    "UniformAmplitude",
    "export_raw",
    "generate",
    "load_bundle",
    "profile_series",
    "realize_atmosphere",
    "save_bundle",
]
