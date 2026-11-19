"""Saving and reloading the ratio stage's tables (METHODS §5.7).

Under ``ratios/`` in a bundle directory, beside whatever else that bundle
holds:

* ``fits.parquet``, the fit table: one row per member, numerator, estimator
  and pair clock (``tsara.pipeline.ratio_fits``), keyed by the member's
  ``event_id`` and naming its ``candidate_id``;
* ``ratios.parquet``, every candidate's ratio at every gate setting of the
  configuration (:func:`~tsara.ratios.candidate_ratio.candidate_ratios`),
  each row naming its candidate's ``parent_candidate_id``. It is built from
  the fits, the events stage's candidate table and the configuration's gates
  and confidence level as it is written, as the events stage builds its
  candidate table from the catalog, so that it cannot disagree with the fits
  beside it, the configuration beside both, or the candidate table it copied
  the parents from. A rule across windows (which window's ratio belongs to
  which plume, §6.1) can then be added later from these tables without
  refitting;
* ``analysis.yaml``, the ``ratios`` section of the analysis configuration
  alone, for the reason the baseline state gives (§6.6): it is all this stage
  read, and a copy of the whole would stop loading the first time another
  stage's settings changed.

The sweep summary and a candidate's fingerprint are not saved: each is a
function of these tables (§5.6), and a saved sweep summary would freeze one
population. ``bundle.json`` is not touched, for the grid's reason
(:mod:`tsara.core.bundle`).

The fits must be what the configuration beside them asks for: a fit naming a
reference no ratio set holds, a numerator its set does not name, or an
estimator or pair clock the configuration does not list is refused, since the
configuration is the record of what ran. A part of a run's fits (one
estimator, say) is saved as it is, its ratios built from that part; nothing in
a candidate's ratio depends on another candidate's fits, so no number in the
part changes by leaving the rest out.

Why this ships now rather than with the Phase-9 pipeline: every stage product
gains persistence in the phase that introduces it (CLAUDE.md §5), and fitting
every member of every candidate is the slowest step of the stage.

What a reload preserves
-----------------------
Both tables column for column and type for type (each declares its types,
``FIT_DTYPES`` and ``RATIO_DTYPES``, so Parquet has nothing to infer), and the
configuration as it was validated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from tsara.config.analysis import RatiosConfig
from tsara.config.loader import read_yaml
from tsara.core.bundle import (
    BUNDLE_ANALYSIS_CONFIG,
    BUNDLE_FITS_FILE,
    BUNDLE_RATIOS_DIR,
    BUNDLE_RATIOS_FILE,
    TsaraBundleError,
)
from tsara.ratios.candidate_ratio import RATIO_COLUMNS, candidate_ratios
from tsara.ratios.estimators import TsaraRatioError
from tsara.ratios.fits import FIT_COLUMNS

logger = logging.getLogger(__name__)

__all__ = ["RatioTables", "load_ratios", "save_ratios"]

#: What :func:`~tsara.ratios.candidate_ratio.candidate_ratios` reads from the
#: events stage's candidate table.
_CANDIDATE_COLUMNS = ("candidate_id", "parent_candidate_id")


@dataclass(frozen=True, eq=False)
class RatioTables:
    """What a bundle's ratios directory holds, reloaded.

    Attributes
    ----------
    fits : pandas.DataFrame
        The fit table (``tsara.ratios.FIT_COLUMNS``).
    candidate_ratios : pandas.DataFrame
        Every candidate's ratio at every gate setting (``tsara.ratios.RATIO_COLUMNS``).
    ratios : RatiosConfig
        The ratios configuration the fits were made and the ratios read under.
    """

    fits: pd.DataFrame
    candidate_ratios: pd.DataFrame
    ratios: RatiosConfig


def save_ratios(
    fits: pd.DataFrame,
    path: str | Path,
    *,
    candidates: pd.DataFrame,
    ratios: RatiosConfig,
) -> Path:
    """Write the fits, every candidate's ratio and the ratios configuration into a bundle.

    Parameters
    ----------
    fits : pandas.DataFrame
        The fit table (``tsara.pipeline.ratio_fits``).
    path : str or pathlib.Path
        Bundle directory, created if absent. The three files are written
        over any left by an earlier run; anything else is left alone.
    candidates : pandas.DataFrame
        The events stage's candidate table (``tsara.events.event_candidates``,
        or ``EventStates.candidates``) of the catalog the fits were made
        from: each ratio row's parent is copied from it.
    ratios : RatiosConfig
        The configuration the fits were made under: its gates and confidence
        level are those the ratio table is read at, and it is written beside
        them under a ``ratios`` key.

    Returns
    -------
    pathlib.Path
        The ``ratios`` directory written.

    Raises
    ------
    TsaraBundleError
        If the fits do not have the fit table's columns, ask for something the
        configuration does not (a reference, a numerator, an estimator or a
        pair clock), or name a candidate the candidate table does not hold;
        if the candidate table has no candidate and parent columns; or if
        ``path`` exists and is not a directory. Nothing is written then.
    """
    if tuple(fits.columns) != FIT_COLUMNS:
        raise TsaraBundleError(
            "save_ratios was handed fits whose columns are not the fit table's; build them "
            f"with ratio_fits. Columns: {list(fits.columns)[:8]}..."
        )
    missing = [name for name in _CANDIDATE_COLUMNS if name not in candidates.columns]
    if missing:
        raise TsaraBundleError(
            f"save_ratios was handed a candidate table without {missing}; hand the events "
            "stage's (event_candidates, or EventStates.candidates)."
        )
    _check_asked_for(fits, ratios)
    # Built before anything is written, so fits that do not belong to these
    # candidates leave the directory as it was.
    try:
        table = candidate_ratios(
            fits, candidates, ratios.gates, confidence_level=ratios.confidence_level
        )
    except TsaraRatioError as exc:
        raise TsaraBundleError(f"save_ratios could not read the candidates' ratios: {exc}") from exc
    bundle = Path(path)
    if bundle.exists() and not bundle.is_dir():
        raise TsaraBundleError(f"Bundle path '{bundle}' exists and is not a directory.")
    target = bundle / BUNDLE_RATIOS_DIR
    target.mkdir(parents=True, exist_ok=True)
    fits.to_parquet(target / BUNDLE_FITS_FILE, index=False)
    table.to_parquet(target / BUNDLE_RATIOS_FILE, index=False)
    payload = {"ratios": ratios.model_dump(mode="json", exclude_none=False)}
    (target / BUNDLE_ANALYSIS_CONFIG).write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    logger.info(
        "Wrote %d fit(s) and %d candidate ratio row(s) to %s.", len(fits), len(table), target
    )
    return target


def load_ratios(path: str | Path) -> RatioTables:
    """Read the tables and configuration written by :func:`save_ratios`.

    Parameters
    ----------
    path : str or pathlib.Path
        Bundle directory, or its ``ratios`` directory.

    Returns
    -------
    RatioTables
        The fits, the candidates' ratios and the ratios configuration.

    Raises
    ------
    TsaraBundleError
        If the directory is missing or lacks one of the three files
        :func:`save_ratios` always writes, a table does not have its columns,
        or the configuration has no ``ratios`` section or does not validate.
    """
    given = Path(path)
    target = given if given.name == BUNDLE_RATIOS_DIR else given / BUNDLE_RATIOS_DIR
    if not target.is_dir():
        raise TsaraBundleError(
            f"'{target}' is not an existing directory; this bundle holds no ratios."
        )
    for name in (BUNDLE_FITS_FILE, BUNDLE_RATIOS_FILE, BUNDLE_ANALYSIS_CONFIG):
        if not (target / name).is_file():
            raise TsaraBundleError(
                f"'{target}' has no '{name}'; save_ratios writes the fits, the ratios and "
                "the configuration together, so this directory is incomplete."
            )
    fits = _read_table(target / BUNDLE_FITS_FILE, FIT_COLUMNS, "the fit table's", "ratio_fits")
    table = _read_table(
        target / BUNDLE_RATIOS_FILE, RATIO_COLUMNS, "the ratio table's", "candidate_ratios"
    )
    config_file = target / BUNDLE_ANALYSIS_CONFIG
    try:
        # The one YAML door, so a key written twice is refused here too.
        section = read_yaml(config_file).get("ratios")
        if section is None:
            raise TsaraBundleError("it has no 'ratios' section")
        ratios = RatiosConfig.model_validate(section)
    except Exception as exc:
        raise TsaraBundleError(
            f"Could not read the ratios configuration in '{config_file}': {exc}"
        ) from exc
    logger.info(
        "Loaded %d fit(s) and %d candidate ratio row(s) from %s.", len(fits), len(table), target
    )
    return RatioTables(fits=fits, candidate_ratios=table, ratios=ratios)


def _check_asked_for(fits: pd.DataFrame, ratios: RatiosConfig) -> None:
    """Refuse fits the configuration does not ask for, by reference, numerator, estimator or clock.

    A numerator is asked for when its set names it, or names its instrument
    (an ``'<instrument>'`` entry stands for every gas of that instrument but
    the reference; which those are is the baseline states' to say, so here
    the instrument is enough).
    """
    named = {ratio_set.reference: set(ratio_set.numerators) for ratio_set in ratios.sets}
    for reference, numerator in (
        fits[["reference", "numerator"]].drop_duplicates().itertuples(index=False)
    ):
        if reference not in named:
            raise TsaraBundleError(
                f"save_ratios was handed fits against '{reference}', which no ratio set of "
                f"the configuration names; its references are {sorted(named)}. Save the fits "
                "beside the configuration they were made under."
            )
        if (
            numerator not in named[reference]
            and numerator.partition(".")[0] not in named[reference]
        ):
            raise TsaraBundleError(
                f"save_ratios was handed fits of '{numerator}' against '{reference}', which "
                f"that ratio set does not name; it names {sorted(named[reference])}."
            )
    for column, listed in (("method", ratios.methods), ("period", ratios.periods)):
        strangers = sorted(set(fits[column]) - set(listed))
        if strangers:
            raise TsaraBundleError(
                f"save_ratios was handed fits whose {column} {strangers} the configuration "
                f"does not list; it lists {list(listed)}."
            )


def _read_table(file: Path, columns: tuple[str, ...], whose: str, builder: str) -> pd.DataFrame:
    """Read one table, refusing one whose columns are not those it was written with."""
    table = pd.read_parquet(file)
    if tuple(table.columns) != columns:
        raise TsaraBundleError(
            f"'{file}' is a table whose columns are not {whose}; build it with {builder}. "
            f"Columns: {list(table.columns)[:8]}..."
        )
    return table
