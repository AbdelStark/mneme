from __future__ import annotations

import argparse

import pytest

from mneme.cli._arguments import non_negative_float, non_negative_int, positive_int


def test_positive_int_accepts_positive_decimal_text() -> None:
    assert positive_int("3") == 3


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("0", "positive integer"),
        ("-1", "positive integer"),
        ("abc", "integer"),
    ],
)
def test_positive_int_rejects_invalid_values(value: str, match: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=match):
        positive_int(value)


def test_non_negative_int_accepts_zero() -> None:
    assert non_negative_int("0") == 0


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("-1", "non-negative integer"),
        ("abc", "integer"),
    ],
)
def test_non_negative_int_rejects_invalid_values(value: str, match: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=match):
        non_negative_int(value)


def test_non_negative_float_accepts_zero_and_positive_values() -> None:
    assert non_negative_float("0") == 0.0
    assert non_negative_float("0.25") == 0.25


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("-0.1", "non-negative finite number"),
        ("nan", "finite number"),
        ("inf", "finite number"),
        ("abc", "finite number"),
    ],
)
def test_non_negative_float_rejects_invalid_values(
    value: str,
    match: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=match):
        non_negative_float(value)
