"""Pydantic configuration schemas and YAML loading for TSARA.

Submodules
----------
base
    Shared foundations: the strict/frozen Pydantic base model and duration
    validation helpers used by every schema.
manifest
    *What the data is*: instruments, file loaders, path templates, variables,
    QA/QC rules, unit conversions, and platform (stationary/mobile) metadata.
analysis
    *What to do with it*: master grid, baseline parameter sweeps, plume
    detection, smoothing, clustering, and regression settings.
loader
    YAML → validated config objects, with readable error reporting and
    manifest-relative path resolution.
"""
