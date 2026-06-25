"""Append-only value-log records."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import numpy as np
from blake3 import blake3

from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    StoreCorruptionError,
    Transition,
    ValidationError,
    build_item,
    content_id,
)
from mneme.core._ids import cid_from_hex

VALUE_RECORD_SCHEMA: Final = "mneme.value_record.v1"
_HEADER_SIZE: Final = 40
_LENGTH_SIZE: Final = 8
_CHECKSUM_SIZE: Final = 32


def append_value_record(path: Path, item: MemoryItem) -> tuple[int, int]:
    """Append one value record and return `(start_offset, end_offset)`."""

    payload = _encode_record(item)
    header = len(payload).to_bytes(_LENGTH_SIZE, "big") + blake3(payload).digest(
        length=_CHECKSUM_SIZE
    )
    with path.open("ab") as handle:
        start = handle.tell()
        handle.write(header)
        handle.write(payload)
        handle.flush()
        try:
            import os

            os.fsync(handle.fileno())
        except OSError:
            pass
        end = handle.tell()
    return start, end


def read_value_records(path: Path) -> Iterator[MemoryItem]:
    """Yield validated MemoryItems from a value log."""

    for item, _, _ in read_value_records_with_offsets(path):
        yield item


def read_value_records_with_offsets(
    path: Path,
    *,
    start_offset: int = 0,
) -> Iterator[tuple[MemoryItem, int, int]]:
    """Yield validated MemoryItems with `(start, end)` offsets."""

    if start_offset < 0:
        raise StoreCorruptionError("value log start offset must be non-negative")
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        if start_offset > size:
            raise StoreCorruptionError("value log start offset is beyond end of file")
        handle.seek(start_offset)
        while True:
            start = handle.tell()
            header = handle.read(_HEADER_SIZE)
            if not header:
                return
            if len(header) != _HEADER_SIZE:
                raise StoreCorruptionError("value log has a partial record header")
            length = int.from_bytes(header[:_LENGTH_SIZE], "big")
            expected_checksum = header[_LENGTH_SIZE:]
            payload = handle.read(length)
            if len(payload) != length:
                raise StoreCorruptionError("value log has a partial record payload")
            if blake3(payload).digest(length=_CHECKSUM_SIZE) != expected_checksum:
                raise StoreCorruptionError("value log record checksum mismatch")
            end = handle.tell()
            yield _decode_record(payload), start, end


def _encode_record(item: MemoryItem) -> bytes:
    if item.content_id is None:
        item = build_item(item.value, item.key, item.encoder_fp, item.meta)
    expected = content_id(item)
    if item.content_id != expected:
        raise ValidationError("content_id does not match canonical item bytes")
    data = {
        "schema_version": VALUE_RECORD_SCHEMA,
        "content_id": expected.hex(),
        "item": {
            "schema_version": item.schema_version,
            "key": _array_to_json(item.key),
            "value_kind": "transition",
            "value": _transition_to_json(item.value),
            "meta": _json_ready(item.meta),
            "encoder_fp": asdict(item.encoder_fp),
        },
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_record(payload: bytes) -> MemoryItem:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StoreCorruptionError("value log record is not valid JSON") from exc
    if not isinstance(data, Mapping):
        raise StoreCorruptionError("value record must be an object")
    if data.get("schema_version") != VALUE_RECORD_SCHEMA:
        raise StoreCorruptionError("unsupported value record schema")
    item_data = data.get("item")
    if not isinstance(item_data, Mapping):
        raise StoreCorruptionError("value record item must be an object")
    if item_data.get("value_kind") != "transition":
        raise StoreCorruptionError("unsupported value record value_kind")
    encoder_fp = _fingerprint_from_json(item_data.get("encoder_fp"))
    value = _transition_from_json(item_data.get("value"))
    content_id_text = _require_string(data.get("content_id"), "content_id")
    cid = cid_from_hex(
        content_id_text,
        "content_id",
        error_type=StoreCorruptionError,
    )
    try:
        item = MemoryItem(
            content_id=cid,
            key=_array_from_json(item_data.get("key")),
            value=value,
            meta=_require_mapping(item_data.get("meta"), "meta"),
            encoder_fp=encoder_fp,
            schema_version=_require_string(
                item_data.get("schema_version"), "item schema_version"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise StoreCorruptionError("invalid memory item payload") from exc
    if item.content_id != content_id(item):
        raise StoreCorruptionError("value record content_id does not match item bytes")
    return item


def _transition_to_json(value: Transition) -> dict[str, Any]:
    return {
        "schema_version": value.schema_version,
        "z_src": _array_to_json(value.z_src),
        "action": _array_to_json(value.action),
        "z_next": _array_to_json(value.z_next),
        "delta": _array_to_json(value.delta),
        "t": value.t,
        "episode_id": str(value.episode_id),
        "reward": value.reward,
    }


def _transition_from_json(data: object) -> Transition:
    mapping = _require_mapping(data, "transition")
    from uuid import UUID

    try:
        episode_id = UUID(_require_string(mapping.get("episode_id"), "episode_id"))
    except ValueError as exc:
        raise StoreCorruptionError("episode_id must be a UUID string") from exc
    try:
        return Transition(
            z_src=_array_from_json(mapping.get("z_src")),
            action=_array_from_json(mapping.get("action")),
            z_next=_array_from_json(mapping.get("z_next")),
            delta=_array_from_json(mapping.get("delta")),
            t=_require_int(mapping.get("t"), "transition t"),
            episode_id=episode_id,
            reward=_optional_float(mapping.get("reward"), "reward"),
            schema_version=_require_string(
                mapping.get("schema_version"), "transition schema_version"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise StoreCorruptionError("invalid transition payload") from exc


def _fingerprint_from_json(data: object) -> EncoderFingerprint:
    mapping = _require_mapping(data, "encoder fingerprint")
    weights_digest = mapping.get("weights_digest")
    if weights_digest is not None and not isinstance(weights_digest, str):
        raise StoreCorruptionError("weights_digest must be a string or null")
    try:
        return EncoderFingerprint(
            encoder_id=_require_string(mapping.get("encoder_id"), "encoder_id"),
            summarizer_id=_require_string(
                mapping.get("summarizer_id"), "summarizer_id"
            ),
            weights_digest=weights_digest,
            config_digest=_require_string(
                mapping.get("config_digest"), "config_digest"
            ),
            schema_version=_require_string(
                mapping.get("schema_version"), "schema_version"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise StoreCorruptionError("invalid encoder fingerprint") from exc


def _array_to_json(value: object) -> dict[str, Any]:
    array = _as_numpy_array(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValidationError("array must have a numeric dtype")
    if not bool(np.isfinite(array).all()):
        raise ValidationError("array must contain only finite values")
    canonical = np.ascontiguousarray(array)
    return {
        "dtype": str(canonical.dtype),
        "shape": list(canonical.shape),
        "data": base64.b64encode(canonical.tobytes(order="C")).decode("ascii"),
    }


def _array_from_json(data: object) -> np.ndarray:
    mapping = _require_mapping(data, "array")
    try:
        dtype = np.dtype(_require_string(mapping.get("dtype"), "array dtype"))
    except (TypeError, ValueError) as exc:
        raise StoreCorruptionError("array dtype is unsupported") from exc
    if not np.issubdtype(dtype, np.number):
        raise StoreCorruptionError("array dtype must be numeric")
    shape_data = mapping.get("shape")
    if not isinstance(shape_data, list) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
        for dim in shape_data
    ):
        raise StoreCorruptionError("array shape must be a list of non-negative ints")
    try:
        raw = base64.b64decode(
            _require_string(mapping.get("data"), "array data"),
            validate=True,
        )
    except binascii.Error as exc:
        raise StoreCorruptionError("array data must be base64") from exc
    try:
        return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape_data)).copy()
    except ValueError as exc:
        raise StoreCorruptionError("array data does not match dtype and shape") from exc


def _as_numpy_array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
        if isinstance(converted, np.ndarray):
            return converted
    raise TypeError("value must be a numpy array or tensor-like object")


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValidationError("float metadata must be finite")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready(item) for item in value]
    raise ValidationError(f"unsupported metadata value: {type(value).__name__}")


def _require_mapping(data: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise StoreCorruptionError(f"{field_name} must be an object")
    return data


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StoreCorruptionError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreCorruptionError(f"{field_name} must be an integer")
    return value


def _optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StoreCorruptionError(f"{field_name} must be a number or null")
    return float(value)


__all__ = [
    "VALUE_RECORD_SCHEMA",
    "append_value_record",
    "read_value_records",
    "read_value_records_with_offsets",
]
