"""Strict JSON helpers for public persistence and transport boundaries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn


def dumps_strict_json(
    value: object,
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
    indent: int | str | None = None,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialize JSON while rejecting weak object keys and non-standard constants."""

    _require_string_object_keys(value)
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        allow_nan=False,
    )


def loads_strict_json(value: str | bytes | bytearray) -> object:
    """Decode JSON while rejecting non-finite numeric values."""

    decoded: object = json.loads(
        value,
        object_pairs_hook=_reject_duplicate_object_keys,
        parse_constant=_reject_json_constant,
        parse_float=_parse_json_float,
    )
    return decoded


def write_strict_json_file(
    path: str | Path,
    value: object,
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
    indent: int | str | None = None,
    separators: tuple[str, str] | None = None,
) -> Path:
    """Write strict JSON after serialization succeeds."""

    payload = (
        dumps_strict_json(
            value,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
        )
        + "\n"
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return target


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, item in pairs:
        if key in decoded:
            raise ValueError(f"duplicate JSON object key: {key}")
        decoded[key] = item
    return decoded


def _require_string_object_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _require_string_object_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _require_string_object_keys(item)


def _parse_json_float(value: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"JSON number must be finite: {value}")
    return numeric
