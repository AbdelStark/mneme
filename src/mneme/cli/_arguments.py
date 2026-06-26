"""Argparse value parsers shared by Mneme CLI entry points."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from mneme.core import Metric


def positive_int(value: str) -> int:
    """Parse a positive integer for CLI arguments."""

    parsed = _parse_int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for CLI arguments."""

    parsed = _parse_int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def non_negative_float(value: str) -> float:
    """Parse a non-negative finite float for CLI arguments."""

    parsed = _parse_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def add_eval_output_seed_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common output path and seed arguments for eval reports."""

    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)


def add_eval_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared recall, latency, and footprint profile arguments."""

    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    _add_query_profile_arguments(parser)
    parser.add_argument(
        "--approx-backend",
        default="faiss_hnsw",
        help="approximate backend to compare, or 'none'",
    )
    parser.add_argument("--seed", default=0, type=int)


def add_eval_receipts_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared receipt-overhead profile arguments."""

    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    _add_query_profile_arguments(parser)
    parser.add_argument("--seed", default=0, type=int)


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc


def _parse_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite number")
    return parsed


def _add_query_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k", default=4, type=positive_int)
    parser.add_argument(
        "--metric",
        default=Metric.L2.value,
        choices=[metric.value for metric in Metric],
    )
    parser.add_argument("--queries", default=8, type=positive_int)
    parser.add_argument("--warmup", default=2, type=non_negative_int)
    parser.add_argument("--measurements", default=20, type=positive_int)


__all__ = [
    "add_eval_output_seed_arguments",
    "add_eval_profile_arguments",
    "add_eval_receipts_arguments",
    "non_negative_float",
    "non_negative_int",
    "positive_int",
]
