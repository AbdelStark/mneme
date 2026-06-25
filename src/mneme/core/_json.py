"""Strict JSON helpers for public persistence and transport boundaries."""

from __future__ import annotations

import json
import math
from typing import NoReturn


def dumps_strict_json(
    value: object,
    *,
    sort_keys: bool = False,
    indent: int | str | None = None,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialize JSON while rejecting non-standard NaN and Infinity constants."""

    return json.dumps(
        value,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        allow_nan=False,
    )


def loads_strict_json(value: str | bytes | bytearray) -> object:
    """Decode JSON while rejecting non-finite numeric values."""

    decoded: object = json.loads(
        value,
        parse_constant=_reject_json_constant,
        parse_float=_parse_json_float,
    )
    return decoded


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant: {value}")


def _parse_json_float(value: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"JSON number must be finite: {value}")
    return numeric
