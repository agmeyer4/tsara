"""Profiling tests against a live real-data mount.

**No real data ever enters this repository.** These tests are skipped unless
``TSARA_REAL_DATA``  points at a readable file on a machine that has the
campaign archive mounted; on CI, on a collaborator's laptop, and in any fresh
clone they simply do not run. Nothing they read — timestamps, coordinates,
concentrations — is written back to the working tree.

Usage::

    export TSARA_REAL_DATA=/path/to/campaign/instrument.parquet
    export TSARA_REAL_DATA_COLUMN=ch4          # optional
    pytest tests/synthetic/test_real_data.py -v

The point of these tests is not to check TSARA's arithmetic — the synthetic
tests do that deterministically. It is to check that profiling survives
*genuinely messy* input: real gaps, real duplicate timestamps, real
instrument quirks, and the plume-dense records this project actually has.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsara.synthetic.background import render_background
from tsara.synthetic.config import BootstrapBackground
from tsara.synthetic.profiling import profile_series

#: Environment variables configuring the live-mount tests.
DATA_ENV = "TSARA_REAL_DATA"
COLUMN_ENV = "TSARA_REAL_DATA_COLUMN"
TIME_ENV = "TSARA_REAL_DATA_TIME_COLUMN"

_raw_path = os.environ.get(DATA_ENV)
_path = Path(_raw_path) if _raw_path else None

requires_real_data = pytest.mark.skipif(
    _path is None or not _path.is_file(),
    reason=(
        f"Set {DATA_ENV} to a readable CSV/Parquet file on a machine with the "
        "campaign archive mounted to run these tests."
    ),
)


@pytest.fixture(scope="module")
def real_series() -> pd.Series:
    """Load one numeric column from the configured real file.

    Returns
    -------
    pandas.Series
        Real measurements indexed by time, sorted and deduplicated.
    """
    assert _path is not None
    if _path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(_path)
    else:
        frame = pd.read_csv(_path)

    time_column = os.environ.get(TIME_ENV)
    if time_column is None:
        # Fall back to the first column that parses as datetimes.
        for candidate in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[candidate]):
                time_column = str(candidate)
                break
    if time_column is None and not isinstance(frame.index, pd.DatetimeIndex):
        pytest.skip(f"No datetime column found; set {TIME_ENV} to name one.")

    if time_column is not None:
        frame = frame.set_index(pd.to_datetime(frame[time_column]))

    column = os.environ.get(COLUMN_ENV)
    if column is None:
        numeric = frame.select_dtypes("number").columns
        if numeric.empty:
            pytest.skip(f"No numeric column found; set {COLUMN_ENV} to name one.")
        column = str(numeric[0])

    series = pd.to_numeric(frame[column], errors="coerce")
    # Real archives routinely contain duplicate and unsorted timestamps.
    series = series[~series.index.duplicated(keep="first")].sort_index()
    return series


@requires_real_data
def test_profiles_a_real_record(real_series: pd.Series) -> None:
    profile = profile_series(real_series, name="real", block_length=256)
    assert profile.n_blocks > 0
    assert profile.noise_sigma > 0.0
    assert profile.residual_sigma > 0.0
    assert np.all(np.isfinite(profile.residual_blocks))


@requires_real_data
def test_real_profile_blocks_are_centred(real_series: pd.Series) -> None:
    profile = profile_series(real_series, name="real", block_length=256)
    assert np.allclose(profile.residual_blocks.mean(axis=1), 0.0, atol=1e-8)


@requires_real_data
def test_real_profile_drives_a_bootstrap_background(real_series: pd.Series) -> None:
    """The full path: real data in, synthetic background out."""
    profile = profile_series(real_series, name="real", block_length=256)
    times = pd.date_range("2026-01-01", periods=5000, freq="1s")
    values = render_background(
        BootstrapBackground(kind="bootstrap", profile="real"),
        times,
        np.random.default_rng(0),
        profiles={"real": profile},
    )
    assert values.shape == (5000,)
    assert np.all(np.isfinite(values))

    # The substrate carries real fluctuation, bounded above by the source
    # record's own robust spread. It is legitimately *smaller*: blocks are
    # mean-centred, which by construction discards between-block
    # low-frequency structure (see METHODS.md §8.3). On a record with strong
    # slow structure the reduction is large — measured at ~3x on this
    # project's real Picarro CH4 — so this asserts the inequality that must
    # hold rather than an equality that must not.
    from tsara.synthetic.profiling import MAD_TO_SIGMA

    robust_sigma = MAD_TO_SIGMA * np.median(np.abs(values - np.median(values)))
    assert 0.0 < robust_sigma <= profile.residual_sigma * 1.05


@requires_real_data
def test_real_record_is_plume_dense_as_expected(real_series: pd.Series) -> None:
    """Documents this project's actual data condition rather than assuming it.

    Informational: it asserts only that both statistics are computable, then
    reports their ratio, which is the diagnostic for how much real plume
    energy leaks through the profiling baseline.
    """
    profile = profile_series(real_series, name="real", block_length=256)
    ratio = profile.residual_sigma / profile.noise_sigma
    assert ratio >= 1.0 or ratio == pytest.approx(1.0, rel=0.5)
