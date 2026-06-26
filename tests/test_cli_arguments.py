from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mneme.cli._arguments import (
    add_eval_output_seed_arguments,
    add_eval_profile_arguments,
    add_eval_receipts_arguments,
    non_negative_float,
    non_negative_int,
    positive_int,
)


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


def test_eval_output_seed_arguments_parse_common_report_fields() -> None:
    parser = argparse.ArgumentParser()
    add_eval_output_seed_arguments(parser)

    args = parser.parse_args(["--out", "reports/eval.json", "--seed", "7"])

    assert args.out == Path("reports/eval.json")
    assert args.seed == 7


def test_eval_profile_arguments_share_validation_contract() -> None:
    parser = argparse.ArgumentParser(exit_on_error=False)
    add_eval_profile_arguments(parser)

    args = parser.parse_args(
        [
            "--store",
            "store",
            "--out",
            "reports/profile.json",
            "--k",
            "2",
            "--metric",
            "cosine",
            "--queries",
            "3",
            "--warmup",
            "0",
            "--measurements",
            "4",
            "--approx-backend",
            "none",
            "--seed",
            "11",
        ]
    )

    assert args.store == Path("store")
    assert args.out == Path("reports/profile.json")
    assert args.k == 2
    assert args.metric == "cosine"
    assert args.queries == 3
    assert args.warmup == 0
    assert args.measurements == 4
    assert args.approx_backend == "none"
    assert args.seed == 11
    with pytest.raises(argparse.ArgumentError, match="positive integer"):
        parser.parse_args(
            [
                "--store",
                "store",
                "--out",
                "reports/profile.json",
                "--queries",
                "0",
            ]
        )


def test_eval_receipts_arguments_share_validation_contract() -> None:
    parser = argparse.ArgumentParser(exit_on_error=False)
    add_eval_receipts_arguments(parser)

    args = parser.parse_args(
        [
            "--store",
            "store",
            "--out",
            "reports/receipts.json",
            "--k",
            "2",
            "--metric",
            "l2",
            "--queries",
            "3",
            "--warmup",
            "0",
            "--measurements",
            "4",
            "--seed",
            "11",
        ]
    )

    assert args.store == Path("store")
    assert args.out == Path("reports/receipts.json")
    assert args.k == 2
    assert args.metric == "l2"
    assert args.queries == 3
    assert args.warmup == 0
    assert args.measurements == 4
    assert args.seed == 11
    assert not hasattr(args, "approx_backend")
    with pytest.raises(argparse.ArgumentError, match="non-negative integer"):
        parser.parse_args(
            [
                "--store",
                "store",
                "--out",
                "reports/receipts.json",
                "--warmup",
                "-1",
            ]
        )
