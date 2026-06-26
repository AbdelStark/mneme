"""Internal path coercion helpers for public boundary validators."""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike, fspath
from pathlib import Path

ErrorFactory = Callable[[str], Exception]


def coerce_text_path(
    value: object,
    field_name: str,
    *,
    type_error: ErrorFactory = TypeError,
    value_error: ErrorFactory = ValueError,
) -> Path:
    """Return a non-empty text path or raise caller-owned boundary errors."""

    if not isinstance(value, str | PathLike):
        raise type_error(f"{field_name} must be a path-like value")
    raw = fspath(value)
    if not isinstance(raw, str):
        raise type_error(f"{field_name} must resolve to a text path")
    if not raw:
        raise value_error(f"{field_name} must not be empty")
    return Path(raw)
