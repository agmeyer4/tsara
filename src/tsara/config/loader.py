"""YAML → validated configuration objects.

This is the only module that parses YAML: the stage bundles that save a
config beside their products read it back through :func:`read_yaml` too.
Everything downstream works with validated Pydantic objects, so a config
error can *only* surface here — with the file path attached — never deep
inside the engine.

Two kinds of typo are refused rather than absorbed. A key the schema does
not know is refused by the schema (``extra="forbid"`` on every model, see
``tsara.config.base``). A key written *twice* in one mapping is refused by
the reader (:class:`_UniqueKeyLoader`): the YAML specification requires
unique keys, but PyYAML's ``safe_load`` keeps whichever value came last and
says nothing, so a variable pasted twice under one instrument would have
silently lost its first definition.

Four entry points:

* :func:`load_manifest` — a YAML file containing a manifest.
* :func:`load_analysis` — a YAML file containing analysis settings.
* :func:`load_config` — one combined file with top-level ``manifest:`` and
  ``analysis:`` keys (the form the CLI consumes), returning a
  :class:`TsaraConfig` that also cross-validates the two halves (e.g. the
  regression reference species must actually be a declared gas).
* :func:`load_synthetic` — a YAML file describing a synthetic dataset to
  manufacture (Phase 2).
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError, model_validator
from yaml.constructor import ConstructorError

from tsara.config.analysis import AnalysisConfig, FromFieldMethod
from tsara.config.base import StrictModel as _StrictModel
from tsara.config.manifest import Manifest
from tsara.config.synthetic import SyntheticConfig
from tsara.core.exceptions import TsaraConfigError

logger = logging.getLogger(__name__)

#: Bound to StrictModel so `_validate(Manifest, ...)` types as Manifest, not
#: Any — callers keep full attribute/type checking on the returned object.
_ModelT = TypeVar("_ModelT", bound=_StrictModel)

#: The tag YAML gives a merge key (``<<``). PyYAML spells it as this literal
#: inside ``flatten_mapping`` and exposes no constant for it. A merge key is
#: not a key of the mapping — it is an instruction to splice another mapping
#: in — so the duplicate check has to step over it: constructing it as an
#: ordinary key fails (no constructor for the tag), and two ``<<`` entries
#: are PyYAML's business, not ours.
_MERGE_TAG = "tag:yaml.org,2002:merge"


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
        """Check the reference species against fields, because it names a gas.

        A ratio's denominator is a physical quantity rather than one
        analyzer's column, so the comparison is with each variable's
        ``field`` (which is its name unless declared otherwise). Two methane
        analyzers declaring ``field: ch4`` make ``ch4`` a valid reference
        whatever their variables are called.
        """
        ref = self.analysis.regression.reference_species
        gases = self.manifest.gas_species
        if ref not in gases:
            raise ValueError(
                f"regression.reference_species '{ref}' is not the field of any role='gas' "
                f"variable in manifest '{self.manifest.name}'; declared gases: {sorted(gases)}."
            )
        return self

    @model_validator(mode="after")
    def _baseline_methods_name_declared_variables(self) -> TsaraConfig:
        """Check every per-variable baseline method against the manifest.

        The analysis schema validates the *shape* of a ``baseline.methods`` key
        and nothing more, because it cannot see the manifest. Here both halves
        are present, so each key must name a declared instrument and one of
        its variables, and the variable must not be circular -- a direction
        has no low quantile. A ``from_field`` entry is checked further: the
        instrument it names must exist, be a different one, and declare a
        variable of the same ``field`` (METHODS.md §1.6), and that variable's
        own baseline must not be ``from_field`` too, since a chain would adopt
        a baseline that is itself adopted, and the record of where a baseline
        came from would then name the wrong instrument.
        """
        instruments = self.manifest.instruments
        for key, method in self.analysis.baseline.methods.items():
            instrument_name, _, variable_name = key.partition(".")
            instrument = instruments.get(instrument_name)
            if instrument is None:
                raise ValueError(
                    f"baseline.methods names '{key}', but manifest '{self.manifest.name}' "
                    f"declares no instrument '{instrument_name}'; instruments: "
                    f"{sorted(instruments)}."
                )
            variable = instrument.variables.get(variable_name)
            if variable is None:
                raise ValueError(
                    f"baseline.methods names '{key}', but instrument '{instrument_name}' "
                    f"declares no variable '{variable_name}'; its variables: "
                    f"{sorted(instrument.variables)}."
                )
            if variable.circular:
                raise ValueError(
                    f"baseline.methods names '{key}', which is circular; a direction has "
                    "no low quantile and no baseline."
                )
            if not isinstance(method, FromFieldMethod):
                continue
            if method.instrument == instrument_name:
                raise ValueError(
                    f"baseline.methods['{key}'] takes its baseline from its own instrument; "
                    "from_field adopts another instrument's baseline of the same field."
                )
            donor = instruments.get(method.instrument)
            if donor is None:
                raise ValueError(
                    f"baseline.methods['{key}'] names instrument '{method.instrument}', "
                    f"which manifest '{self.manifest.name}' does not declare; instruments: "
                    f"{sorted(instruments)}."
                )
            field = instrument.field_of(variable_name)
            matching = [name for name in donor.variables if donor.field_of(name) == field]
            if not matching:
                raise ValueError(
                    f"baseline.methods['{key}'] adopts a baseline from '{method.instrument}', "
                    f"which measures no variable with field '{field}'; its fields: "
                    f"{sorted({donor.field_of(name) for name in donor.variables})}."
                )
            if len(matching) > 1:
                raise ValueError(
                    f"baseline.methods['{key}'] adopts a baseline from '{method.instrument}', "
                    f"which declares more than one variable with field '{field}' "
                    f"({sorted(matching)}), so which baseline to adopt is ambiguous."
                )
            donor_method = self.analysis.baseline.methods.get(f"{method.instrument}.{matching[0]}")
            if isinstance(donor_method, FromFieldMethod):
                raise ValueError(
                    f"baseline.methods['{key}'] adopts the baseline of "
                    f"'{method.instrument}.{matching[0]}', whose own baseline is from_field "
                    f"(from '{donor_method.instrument}'); a chain is refused so that the "
                    "record of where a baseline came from names the instrument that made it."
                )
        return self

    @model_validator(mode="after")
    def _event_triggers_name_declared_gases(self) -> TsaraConfig:
        """Check every ``events.triggers`` entry against the manifest.

        The analysis schema validates the *spelling* of a key and a trigger;
        here both must exist. Both sides must be ``role: gas`` variables,
        since those are the ones with enhancements to find events on (a key
        naming an instrument must declare at least one); a direction is
        ``role: met``, so no trigger is circular. A variable may not take its
        own events, and a trigger may not itself be triggered: a chain would
        make the record of where a variable's events came from name the wrong
        variable, the reason ``from_field`` refuses one (METHODS.md §6.8).
        """
        events = self.analysis.events
        instruments = self.manifest.instruments

        def gases_of(key: str, instrument_name: str) -> list[str]:
            instrument = instruments.get(instrument_name)
            if instrument is None:
                raise ValueError(
                    f"events.triggers names '{key}', but manifest '{self.manifest.name}' "
                    f"declares no instrument '{instrument_name}'; instruments: "
                    f"{sorted(instruments)}."
                )
            return [name for name, spec in instrument.variables.items() if spec.role == "gas"]

        for key, trigger in events.triggers.items():
            instrument_name, _, variable_name = key.partition(".")
            gases = gases_of(key, instrument_name)
            if variable_name and variable_name not in gases:
                raise ValueError(
                    f"events.triggers names '{key}', but instrument '{instrument_name}' "
                    f"declares no gas variable '{variable_name}'; its gases: "
                    f"{sorted(gases)}."
                )
            if not gases:
                raise ValueError(
                    f"events.triggers names instrument '{key}', which declares no gas "
                    "variable, so the entry would apply to nothing."
                )
            trigger_instrument, _, trigger_variable = trigger.partition(".")
            if trigger_variable not in gases_of(trigger, trigger_instrument):
                raise ValueError(
                    f"events.triggers['{key}'] names '{trigger}', which is not a gas "
                    f"variable of instrument '{trigger_instrument}'."
                )
            covered = [variable_name] if variable_name else gases
            if trigger_instrument == instrument_name and trigger_variable in covered:
                raise ValueError(
                    f"events.triggers['{key}'] makes '{trigger}' its own trigger; name the "
                    "variables that take its events one by one instead."
                )
            upstream = events.trigger_for(trigger_instrument, trigger_variable)
            if upstream is not None:
                raise ValueError(
                    f"events.triggers['{key}'] names '{trigger}', which itself takes its "
                    f"events from '{upstream}'; a chain is refused so that the record of "
                    "where a variable's events came from names the variable that found them."
                )
        return self


# ---------------------------------------------------------------------------
# YAML plumbing
# ---------------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    """A ``SafeLoader`` that refuses a mapping defining the same key twice.

    The YAML specification requires the keys of a mapping to be unique, but
    PyYAML does not enforce it: ``safe_load`` keeps whichever value came
    *last* and says nothing. For a science config that is the worst kind of
    failure — a variable pasted twice under one instrument silently loses
    its first definition, a sweep parameter listed twice silently runs the
    second. Measured before this class existed:
    ``variables: {ch4: {column: A}, ch4: {column: B}}`` loaded as one
    variable reading ``B``.

    The check runs on the raw key nodes *before* the base class flattens
    YAML merge keys (``<<: *defaults``) into the mapping, because an entry
    overriding a merged default is the legitimate use of a merge key and
    must not read as a duplicate.

    Subclassing ``SafeLoader`` rather than ``Loader`` keeps the property the
    old ``safe_load`` call had: a config file cannot instantiate arbitrary
    Python objects.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        """Refuse a repeated key, naming both places it appears, then delegate."""
        first_seen: dict[Hashable, yaml.Mark] = {}
        for key_node, _value_node in node.value:
            if key_node.tag == _MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                # A list or a mapping cannot be a dict key at all; the base
                # class reports that one, with its own mark.
                continue
            if key in first_seen:
                # PyYAML's own two-mark format: the context mark is the first
                # definition, the problem mark the repeat, each with its line
                # and the offending text.
                raise ConstructorError(
                    "while constructing a mapping",
                    first_seen[key],
                    f"found duplicate key {key!r}; YAML would silently keep only the last value",
                    key_node.start_mark,
                )
            first_seen[key] = key_node.start_mark
        return super().construct_mapping(node, deep=deep)


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict, converting every failure to TsaraConfigError.

    Every YAML TSARA reads comes through here — the loaders below and the
    two stage bundles that read back the config saved beside their products —
    so the rules are stated once: a missing file, a syntax error, a duplicate
    key (:class:`_UniqueKeyLoader`) and a top level that is not a mapping are
    each refused with the file path in the message. ``SafeLoader`` semantics
    throughout: a config file cannot instantiate arbitrary Python objects.

    Parameters
    ----------
    path : str or pathlib.Path
        The YAML file.

    Returns
    -------
    dict
        The file's top-level mapping, untouched by any schema.

    Raises
    ------
    TsaraConfigError
        With the path attached, for every failure named above.
    """
    path = Path(path)
    if not path.is_file():
        raise TsaraConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            # `yaml.load` with a SafeLoader *subclass* is `safe_load` plus the
            # duplicate-key refusal; it is not the unsafe `yaml.Loader`.
            data = yaml.load(fh, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        # PyYAML errors carry line/column info and the offending text in
        # their str(); keep it. A duplicate key arrives here too.
        raise TsaraConfigError(f"Invalid YAML in {path}:\n{exc}") from exc

    if not isinstance(data, dict):
        raise TsaraConfigError(
            f"Top level of {path} must be a mapping (key: value pairs), got {type(data).__name__}."
        )
    return data


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
    manifest = _validate(Manifest, read_yaml(path), path)
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
    analysis = _validate(AnalysisConfig, read_yaml(path), path)
    logger.info(
        "Loaded analysis config: %d baseline window(s) x %d quantile(s), %d per-variable "
        "baseline method(s), output_grid=%s",
        len(analysis.baseline.windows),
        len(analysis.baseline.quantiles),
        len(analysis.baseline.methods),
        analysis.output_grid.freq if analysis.output_grid is not None else "none",
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
    data = read_yaml(path)

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


def load_synthetic(path: str | Path) -> SyntheticConfig:
    """Load and validate a synthetic-dataset YAML file.

    Synthetic configs are deliberately loadable through the same door as
    manifests and analysis settings: a generated dataset should be
    specifiable, reviewable, and diffable as a checked-in text file, not only
    constructible in Python.

    Note there is no ``base_path`` resolution here — a synthetic config
    references no files by design. Real-data profiles for bootstrap
    backgrounds are named, not embedded, and are supplied to
    :func:`tsara.synthetic.generate` at call time, which is what keeps these
    files free of real-data-derived numbers.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the synthetic-config YAML.

    Returns
    -------
    SyntheticConfig
        Frozen, validated synthetic dataset specification.

    Raises
    ------
    TsaraConfigError
        If the file is missing, is not valid YAML, or fails validation.
    """
    path = Path(path)
    config: SyntheticConfig = _validate(SyntheticConfig, read_yaml(path), path)
    logger.info("Loaded synthetic config '%s' from %s", config.name, path)
    return config


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
