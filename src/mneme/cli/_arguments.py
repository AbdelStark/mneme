"""Argparse value parsers shared by Mneme CLI entry points."""

from __future__ import annotations

import argparse
import math


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


__all__ = ["non_negative_float", "non_negative_int", "positive_int"]
