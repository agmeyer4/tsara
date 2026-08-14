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

Submodules
----------
config
    Pydantic schemas describing the dataset to manufacture.
profiling
    Measuring real data's statistical shape, so synthetic parameters are
    grounded in reality rather than guessed.
background
    Rendering the plume-free signal, parametrically or by block-bootstrap
    from a real-data profile.
plumes
    Plume shape kernels, Poisson event scheduling, and the ground-truth
    catalog.
noise
    Two-component error injection (random, optionally AR(1)-correlated;
    systematic, correlated by construction) and quantization.
platform
    Fixed-site coordinates and synthetic mobile GPS tracks.
generator
    The orchestrator turning one config into streams plus ground truth.
bundle
    Reading and writing the on-disk TSARA bundle directory.
"""

from __future__ import annotations

from tsara.synthetic.background import TsaraSyntheticError
from tsara.synthetic.bundle import TsaraBundleError, load_bundle, save_bundle
from tsara.synthetic.config import (
    BootstrapBackground,
    DropoutSpec,
    EMGShape,
    GaussianShape,
    InstrumentSpec,
    LognormalAmplitude,
    MobileTrack,
    NestedSpec,
    ParametricBackground,
    RatioSpec,
    SourceSpec,
    SpeciesSpec,
    StationarySite,
    SyntheticConfig,
    TrueComponent,
    TrueUncertainty,
    UniformAmplitude,
)
from tsara.synthetic.generator import SyntheticDataset, generate
from tsara.synthetic.plumes import GroundTruth, GroundTruthEvent
from tsara.synthetic.profiling import (
    RealDataProfile,
    TsaraProfilingError,
    profile_series,
)

__all__ = [
    "BootstrapBackground",
    "DropoutSpec",
    "EMGShape",
    "GaussianShape",
    "GroundTruth",
    "GroundTruthEvent",
    "InstrumentSpec",
    "LognormalAmplitude",
    "MobileTrack",
    "NestedSpec",
    "ParametricBackground",
    "RatioSpec",
    "RealDataProfile",
    "SourceSpec",
    "SpeciesSpec",
    "StationarySite",
    "SyntheticConfig",
    "SyntheticDataset",
    "TrueComponent",
    "TrueUncertainty",
    "TsaraBundleError",
    "TsaraProfilingError",
    "TsaraSyntheticError",
    "UniformAmplitude",
    "generate",
    "load_bundle",
    "profile_series",
    "save_bundle",
]
