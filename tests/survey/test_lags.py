"""Tests for one segment's lag (``tsara.survey.lags``), on signals checkable by hand.

The sign is what matters most: a lag is positive when the instrument shows the
air *later* than its partner, so the shift that corrects it is negative
(``docs/METHODS.md`` §12.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tsara.survey.lags import (
    SecondsSeries,
    SegmentLag,
    TsaraSurveyError,
    pooled_lag,
    seconds_series,
)

T0 = int(pd.Timestamp("2026-01-01T00:00:00").value // 1_000_000_000)


def _stream(seconds: np.ndarray, values: np.ndarray, **attrs: object) -> xr.Dataset:
    times = pd.to_datetime(seconds, unit="s").as_unit("ns")
    stream = xr.Dataset({"ch4": ("time", values)}, coords={"time": np.asarray(times)})
    stream["ch4"].attrs.update(attrs)
    stream.attrs["instrument"] = "test"
    return stream


def _plumes(n: int, seed: int = 0) -> np.ndarray:
    """Narrow plumes at random seconds on a flat background: a sharp correlation peak."""
    rng = np.random.default_rng(seed)
    signal = np.zeros(n)
    for centre in rng.choice(n, size=n // 40, replace=False):
        t = np.arange(n) - centre
        signal += rng.lognormal(3, 0.5) * np.exp(-0.5 * (t / 3.0) ** 2)
    return signal + rng.normal(0, 0.2, n)


# ---------------------------------------------------------------------------
# A variable on whole seconds
# ---------------------------------------------------------------------------


def test_times_are_floored_and_the_first_reading_of_a_second_is_kept() -> None:
    """Two readings in second 0 (at .2 and .9 s): the first, 1.0, is kept. The
    rest of the record sits at 7.0, so the running median is 7.0 and each
    wiggle is its value less 7; second 2 holds no reading."""
    seconds = np.r_[T0 + 0.2, T0 + 0.9, T0 + 1.5, T0 + np.arange(3.0, 200.0)]
    values = np.r_[1.0, 9.0, 8.0, np.full(197, 7.0)]
    series = seconds_series(_stream(seconds, values), "ch4", running_median_s=61)
    assert series.first_s == T0
    assert series.values[[0, 1, 3]].tolist() == [-6.0, 1.0, 0.0]
    assert np.isnan(series.values[2])


def test_a_flat_series_has_no_wiggles() -> None:
    seconds = T0 + np.arange(600.0)
    series = seconds_series(_stream(seconds, np.full(600, 1950.0)), "ch4", running_median_s=61)
    assert np.allclose(series.values, 0.0)


def test_a_sparse_record_still_gets_a_running_median() -> None:
    """A quarter of the window's possible readings is enough: a 5 s record
    holds 12 in a 61 s window, so 3 are needed, where a fixed 30 would leave it
    all NaN."""
    seconds = T0 + np.arange(0.0, 600.0, 5.0)
    series = seconds_series(_stream(seconds, np.arange(120.0)), "ch4", running_median_s=61)
    assert np.isfinite(series.values[::5]).all()


def test_missing_values_are_not_readings() -> None:
    values = np.ones(200)
    values[1] = np.nan
    series = seconds_series(_stream(T0 + np.arange(200.0), values), "ch4", running_median_s=61)
    assert np.isnan(series.values[1]) and np.isfinite(series.values[[0, 2, 3]]).all()


def test_a_variable_with_no_reading_gives_an_empty_series() -> None:
    series = seconds_series(
        _stream(T0 + np.arange(3.0), np.full(3, np.nan)), "ch4", running_median_s=61
    )
    assert len(series.values) == 0


def test_an_absent_variable_is_refused() -> None:
    with pytest.raises(TsaraSurveyError, match="has no variable 'co'"):
        seconds_series(_stream(T0 + np.arange(3.0), np.ones(3)), "co", running_median_s=61)


def test_a_circular_variable_is_refused() -> None:
    with pytest.raises(TsaraSurveyError, match="circular"):
        seconds_series(
            _stream(T0 + np.arange(3.0), np.ones(3), circular=1), "ch4", running_median_s=61
        )


def test_a_window_is_nan_outside_the_record() -> None:
    series = SecondsSeries(first_s=10, values=np.array([1.0, 2.0, 3.0]))
    window = series.window(8, 14)
    assert np.isnan(window[[0, 1, 5]]).all()
    assert window[2:5].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(series.window(20, 22)).all()


# ---------------------------------------------------------------------------
# The lag of one segment
# ---------------------------------------------------------------------------


def _lag(pairs: list[tuple[SecondsSeries, SecondsSeries]], **settings: float) -> SegmentLag:
    """One 1000 s segment, search 30 s, 200 pairs, decided at 0.6, tolerance 2 s."""
    options = {"search_s": 30, "min_pairs": 200, "min_r": 0.6, "tolerance_s": 2.0} | settings
    return pooled_lag(pairs, T0 + 100, T0 + 1100, **options)  # type: ignore[arg-type]


def _pair(lag_s: int, n: int = 1200, seed: int = 0) -> tuple[SecondsSeries, SecondsSeries]:
    """The partner shows the plumes at t; the instrument at t + lag."""
    truth = _plumes(n + 200, seed)
    partner = SecondsSeries(first_s=T0, values=truth[100 : 100 + n])
    instrument = SecondsSeries(first_s=T0, values=truth[100 - lag_s : 100 - lag_s + n])
    return instrument, partner


@pytest.mark.parametrize("lag_s", [-17, 0, 5, 29])
def test_the_lag_is_how_much_later_the_instrument_shows_the_air(lag_s: int) -> None:
    lag = _lag([_pair(lag_s)])
    assert lag.lag_s == lag_s
    assert lag.r == pytest.approx(1.0)
    assert not lag.at_edge and not lag.pairs_disagree
    assert lag.n_pairs == 1000


def test_a_best_lag_at_the_search_edge_is_flagged() -> None:
    """A true lag just beyond the search shows as its peak's shoulder at the edge."""
    lag = _lag([_pair(32)])
    assert lag.lag_s == 30
    assert lag.at_edge


def test_a_true_lag_far_beyond_the_search_finds_no_peak() -> None:
    """Measured while writing these tests: 40 s against a 30 s search does not
    show at the edge but as a weak peak somewhere inside, which min_r refuses."""
    lag = _lag([_pair(40)])
    assert not lag.at_edge
    assert lag.r < 0.6


def test_too_few_paired_seconds_measure_nothing() -> None:
    lag = _lag([_pair(5)], min_pairs=1001)
    assert np.isnan(lag.lag_s) and np.isnan(lag.r)
    assert lag.n_pairs == 0


def test_a_flat_segment_measures_nothing() -> None:
    flat = SecondsSeries(first_s=T0, values=np.zeros(1200))
    _, partner = _pair(5)
    lag = _lag([(flat, partner)])
    assert np.isnan(lag.lag_s)


def test_a_tier_pools_its_pairs_curves() -> None:
    """Two pairs agreeing on 6 s pool to 6 s with their paired seconds summed;
    a pair with no data in the segment does not move the pooled curve."""
    one = _pair(6, seed=1)
    two = _pair(6, seed=2)
    empty = (SecondsSeries(first_s=T0, values=np.full(1200, np.nan)), one[1])
    lag = _lag([one, two, empty])
    assert lag.lag_s == 6
    assert lag.n_pairs == 2000
    assert lag.r == pytest.approx(1.0)
    assert not lag.pairs_disagree


def test_pairs_that_disagree_are_flagged_rather_than_averaged_between() -> None:
    """Found writing these tests: pairs at 4 s and 10 s, each certain alone,
    pool to a curve peaking at 7 s with r 0.76, which neither supports."""
    lag = _lag([_pair(4, seed=3), _pair(10, seed=4)])
    assert lag.lag_s == 7
    assert lag.r > 0.6
    assert lag.pairs_disagree


def test_a_pair_too_weak_to_decide_alone_cannot_veto() -> None:
    """A strong pair at 5 s and a weak one (mostly noise) peaking elsewhere."""
    strong = _pair(5, seed=5)
    rng = np.random.default_rng(6)
    noise = SecondsSeries(first_s=T0, values=rng.normal(0, 1, 1200))
    lag = _lag([strong, (noise, strong[1])])
    assert lag.lag_s == 5
    assert not lag.pairs_disagree
