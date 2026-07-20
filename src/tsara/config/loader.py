"""YAML → validated configuration objects.

This is the only module that touches YAML. Everything downstream works with
validated Pydantic objects, so a config error can *only* surface here — with
the file path attached — never deep inside the engine.

Three entry points:

* :func:`load_manifest` — a YAML file containing a manifest.
* :func:`load_analysis` — a YAML file containing analysis settings.
* :func:`load_config` — one combined file with top-level ``manifest:`` and
  ``analysis:`` keys (the form the CLI consumes), returning a
  :class:`TsaraConfig` that also cross-validates the two halves (e.g. the
  regression reference species must actually be a declared gas).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError, model_validator

from tsara.config.analysis import AnalysisConfig
from tsara.config.base import StrictModel as _StrictModel
from tsara.config.manifest import Manifest
from tsara.exceptions import TsaraConfigError

logger = logging.getLogger(__name__)

#: Bound to StrictModel so `_validate(Manifest, ...)` types as Manifest, not
#: Any — callers keep full attribute/type checking on the returned object.
_ModelT = TypeVar("_ModelT", bound=_StrictModel)


class TsaraConfig(_StrictModel):
    """A complete, cross-validated run configuration (manifest + analysis).

    Cross-field checks that need *both* halves live here — neither the
    manifest nor the analysis schema alone can know whether
    ``regression.reference_species`` names a real gas variable.
    """

    manifest: Manifest
    analysis: AnalysisConfig

    @model_validator(mode="after")
    def _reference_species_is_a_declared_gas(self) -> TsaraConfig:
        ref = self.analysis.regression.reference_species
        gases = self.manifest.gas_species
        if ref not in gases:
            raise ValueError(
                f"regression.reference_species '{ref}' is not a role='gas' variable "
                f"in manifest '{self.manifest.name}'; declared gases: {sorted(gases)}."
            )
        return self


# ---------------------------------------------------------------------------
# YAML plumbing
# ---------------------------------------------------------------------------


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict, converting failures to TsaraConfigError.

    ``yaml.safe_load`` (never ``load``) — config files must not be able to
    instantiate arbitrary Python objects.
    """
    path = Path(path)
    if not path.is_file():
        raise TsaraConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        # PyYAML errors carry line/column info in their str(); keep it.
        raise TsaraConfigError(f"YAML syntax error in {path}:\n{exc}") from exc

    if not isinstance(data, dict):
        raise TsaraConfigError(
            f"Top level of {path} must be a mapping (key: value pairs), "
            f"got {type(data).__name__}."
        )
    return data


def _validate(model_cls: type[_ModelT], data: dict[str, Any], path: Path) -> _ModelT:
    """Run Pydantic validation, re-raising with the file path attached.

    Pydantic's ValidationError already pinpoints the offending field
    ('instruments.picarro.variables.ch4.units'); we add *which file* so a
    multi-config batch run on the cluster fails with an actionable message.
    """
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise TsaraConfigError(f"Invalid configuration in {path}:\n{exc}") from exc


def _resolve_base_path(manifest: Manifest, anchor: Path) -> Manifest:
    """Resolve a relative ``base_path`` against the manifest file's directory.

    Rationale: a manifest checked into a project repo should be able to say
    ``base_path: ../data`` and work for every collaborator regardless of
    their working directory when they launch TSARA. Absolute paths pass
    through untouched. Configs are frozen, so this returns a *new* Manifest
    rather than mutating.
    """
    if manifest.base_path.is_absolute():
        return manifest
    resolved = (anchor / manifest.base_path).resolve()
    logger.debug("Resolved relative base_path %s -> %s", manifest.base_path, resolved)
    return manifest.model_copy(update={"base_path": resolved})


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a manifest YAML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the manifest YAML. A relative ``base_path`` inside the file
        is resolved against this file's directory.

    Returns
    -------
    Manifest
        Frozen, validated manifest.

    Raises
    ------
    TsaraConfigError
        If the file is missing, is not valid YAML, or fails validation.
    """
    path = Path(path)
    manifest = _validate(Manifest, _read_yaml(path), path)
    manifest = _resolve_base_path(manifest, path.parent.resolve())
    logger.info(
        "Loaded manifest '%s': %d instrument(s), %d gas species, platform=%s",
        manifest.name,
        len(manifest.instruments),
        len(manifest.gas_species),
        manifest.platform.kind,
    )
    return manifest


def load_analysis(path: str | Path) -> AnalysisConfig:
    """Load and validate an analysis-settings YAML file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the analysis YAML.

    Returns
    -------
    AnalysisConfig
        Frozen, validated analysis configuration.

    Raises
    ------
    TsaraConfigError
        If the file is missing, is not valid YAML, or fails validation.
    """
    path = Path(path)
    analysis = _validate(AnalysisConfig, _read_yaml(path), path)
    logger.info(
        "Loaded analysis config: grid=%s, %d baseline window(s) x %d quantile(s)",
        analysis.grid.freq,
        len(analysis.baseline.windows),
        len(analysis.baseline.quantiles),
    )
    return analysis


def load_config(path: str | Path) -> TsaraConfig:
    """Load a combined run configuration (``manifest:`` + ``analysis:``).

    This is the single-file form the CLI consumes for headless batch runs.
    Both sections are validated individually, then cross-validated (e.g.
    the regression reference species must be a declared gas variable).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the combined YAML with top-level ``manifest`` and
        ``analysis`` mappings.

    Returns
    -------
    TsaraConfig
        Frozen, cross-validated combined configuration.

    Raises
    ------
    TsaraConfigError
        If the file is missing, malformed, missing a section, or fails
        validation.
    """
    path = Path(path)
    data = _read_yaml(path)

    missing = {"manifest", "analysis"} - data.keys()
    if missing:
        raise TsaraConfigError(
            f"Combined config {path} is missing required section(s): {sorted(missing)}. "
            "Expected top-level 'manifest:' and 'analysis:' mappings."
        )

    config: TsaraConfig = _validate(TsaraConfig, data, path)
    # Re-anchor the manifest's relative base_path exactly as load_manifest
    # would; frozen models mean we rebuild rather than mutate.
    resolved_manifest = _resolve_base_path(config.manifest, path.parent.resolve())
    if resolved_manifest is not config.manifest:
        config = config.model_copy(update={"manifest": resolved_manifest})
    logger.info("Loaded combined config for campaign '%s'", config.manifest.name)
    return config
