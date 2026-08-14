"""Tests for timestamp -> numeric-epoch conversion.

TSARA is UTC internally, so tz-aware and tz-naive spellings of the same
instant must convert to identical numbers. Getting this wrong would shift a
whole record by the UTC offset without raising anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tsara.core.timebase import (
    epoch_ns,
    epoch_s,
    timestamp_epoch_ns,
    timestamp_epoch_s,
    to_utc_naive,
    to_utc_naive_stamp,
)

EPOCH_NS_2026 = 1767225600_000_000_000  # 2026-01-01T00:00:00Z


def test_naive_index_is_returned_unchanged() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="1s")
    assert to_utc_naive(index) is index


def test_aware_index_is_converted_to_naive_utc() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="1s", tz="UTC")
    converted = to_utc_naive(index)
    assert converted.tz is None
    assert converted[0] == pd.Timestamp("2026-01-01")


def test_non_utc_index_is_shifted_to_utc() -> None:
    """The case a naive .astype() would silently get wrong."""
    index = pd.DatetimeIndex(["2026-01-01 02:00:00"]).tz_localize("Etc/GMT-2")
    assert to_utc_naive(index)[0] == pd.Timestamp("2026-01-01 00:00:00")


def test_epoch_ns_agrees_across_timezone_spellings() -> None:
    naive = pd.date_range("2026-01-01", periods=3, freq="1s")
    aware = pd.date_range("2026-01-01", periods=3, freq="1s", tz="UTC")
    assert list(epoch_ns(naive)) == list(epoch_ns(aware))
    assert epoch_ns(naive)[0] == EPOCH_NS_2026


def test_epoch_s_is_nanoseconds_scaled() -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="30s")
    assert epoch_s(index)[1] - epoch_s(index)[0] == pytest.approx(30.0)


def test_naive_stamp_is_returned_unchanged() -> None:
    stamp = pd.Timestamp("2026-01-01")
    assert to_utc_naive_stamp(stamp) == stamp
    assert to_utc_naive_stamp(stamp).tz is None


def test_aware_stamp_is_converted_to_naive_utc() -> None:
    converted = to_utc_naive_stamp(pd.Timestamp("2026-01-01", tz="UTC"))
    assert converted.tz is None
    assert converted == pd.Timestamp("2026-01-01")


def test_non_utc_stamp_is_shifted_to_utc() -> None:
    converted = to_utc_naive_stamp(pd.Timestamp("2026-01-01 02:00:00", tz="Etc/GMT-2"))
    assert converted == pd.Timestamp("2026-01-01 00:00:00")


def test_scalar_and_index_normalization_agree() -> None:
    """The pair exists so catalog scalars and stream clocks stay comparable."""
    aware = pd.date_range("2026-01-01", periods=3, freq="1s", tz="Etc/GMT-2")
    assert to_utc_naive_stamp(aware[1]) == to_utc_naive(aware)[1]


def test_timestamp_helpers_handle_a_naive_stamp() -> None:
    stamp = pd.Timestamp("2026-01-01")
    assert timestamp_epoch_ns(stamp) == EPOCH_NS_2026
    assert timestamp_epoch_s(stamp) == pytest.approx(EPOCH_NS_2026 / 1e9)


def test_timestamp_helpers_handle_an_aware_stamp() -> None:
    stamp = pd.Timestamp("2026-01-01", tz="UTC")
    assert timestamp_epoch_ns(stamp) == EPOCH_NS_2026
    assert timestamp_epoch_s(stamp) == pytest.approx(EPOCH_NS_2026 / 1e9)


def test_timestamp_helpers_shift_a_non_utc_stamp() -> None:
    stamp = pd.Timestamp("2026-01-01 02:00:00", tz="Etc/GMT-2")
    assert timestamp_epoch_ns(stamp) == EPOCH_NS_2026
    assert timestamp_epoch_s(stamp) == pytest.approx(EPOCH_NS_2026 / 1e9)
