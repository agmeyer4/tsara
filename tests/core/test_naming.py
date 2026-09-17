"""The shared vocabulary's own predicates.

Most of `tsara.core.naming` is constants and one-line name builders, exercised
by every stage that reads a stream. `is_circular` is the exception: it encodes
a contract subtle enough to get backwards, so it is pinned here rather than
only through its callers.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsara.core.naming import is_circular

#: Every spelling the flag is known to arrive in, and what it means.
#:
#: The int is what both producers write, the `numpy.int64` is what a netCDF
#: round trip returns (measured), and the strings are what a dataset built
#: elsewhere may carry. The absent case matters as much as the present ones:
#: a variable saying nothing is not a direction.
SPELLINGS = [
    (1, True),
    (0, False),
    (True, True),
    (False, False),
    (np.int64(1), True),
    (np.int32(0), False),
    ("1", True),
    ("0", False),
    ("True", True),
    ("False", False),
    (None, False),
    ("", False),
]


@pytest.mark.parametrize(("flag", "expected"), SPELLINGS)
def test_every_spelling_of_the_circular_flag_reads_the_same(flag: object, expected: bool) -> None:
    assert is_circular({"circular": flag}) is expected


def test_a_variable_that_says_nothing_is_not_a_direction() -> None:
    assert is_circular({}) is False
    assert is_circular({"units": "ppb"}) is False


def test_the_trap_this_predicate_exists_for() -> None:
    """`bool("0")` is True, which is why truthiness cannot be used here.

    A non-angular variable whose flag came back as the string "0" would be
    vector-averaged: its values would be read as bearings, wrapped into
    [0, 360), and reported with a resultant length instead of a sigma. The
    numbers would all be plausible.
    """
    assert bool("0") is True
    assert is_circular({"circular": "0"}) is False
