"""Canonical serialization and content ids."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Final

import numpy as np
from blake3 import blake3

from mneme.core._errors import DTypeError, ShapeError, ValidationError
from mneme.core._types import (
    Cid,
    EncoderFingerprint,
    MemoryItem,
    SummaryVec,
    Transition,
)

_MAGIC: Final = b"mneme.canonical.v1"
_DIGEST_SIZE: Final = 32


def canonical_bytes(value: object) -> bytes:
    """Return deterministic canonical bytes for a supported core object."""

    if isinstance(value, MemoryItem):
        return _document("memory_item", _serialize_memory_item(value))
    if isinstance(value, Transition):
        return _document("transition", _serialize_transition(value))
    if isinstance(value, EncoderFingerprint):
        return _document("encoder_fingerprint", _serialize_fingerprint(value))
    raise TypeError(f"unsupported canonical object: {type(value).__name__}")


def content_id(item: MemoryItem) -> Cid:
    """Compute a BLAKE3 content id for a memory item, excluding content_id."""

    if not isinstance(item, MemoryItem):
        raise TypeError("item must be a MemoryItem")
    return blake3(canonical_bytes(item)).digest(length=_DIGEST_SIZE)


def build_item(
    value: Transition,
    key: SummaryVec,
    encoder_fp: EncoderFingerprint,
    meta: Mapping[str, Any] | None = None,
) -> MemoryItem:
    """Build a validated MemoryItem and fill its deterministic content id."""

    item = MemoryItem(
        content_id=None,
        key=key,
        value=value,
        meta={} if meta is None else meta,
        encoder_fp=encoder_fp,
    )
    return replace(item, content_id=content_id(item))


def _document(kind: str, payload: bytes) -> bytes:
    return _frame(b"doc", _string(kind) + _bytes(_MAGIC) + payload)


def _serialize_memory_item(item: MemoryItem) -> bytes:
    return _record(
        "memory_item",
        [
            ("schema_version", _string(item.schema_version)),
            ("encoder_fp", _serialize_fingerprint(item.encoder_fp)),
            ("key", _array(item.key, "key")),
            ("value_kind", _string("transition")),
            ("value", _serialize_transition(item.value)),
            ("meta", _metadata_mapping(item.meta)),
        ],
    )


def _serialize_fingerprint(fp: EncoderFingerprint) -> bytes:
    return _record(
        "encoder_fingerprint",
        [
            ("schema_version", _string(fp.schema_version)),
            ("encoder_id", _string(fp.encoder_id)),
            ("summarizer_id", _string(fp.summarizer_id)),
            ("weights_digest", _optional_string(fp.weights_digest)),
            ("config_digest", _string(fp.config_digest)),
        ],
    )


def _serialize_transition(value: Transition) -> bytes:
    return _record(
        "transition",
        [
            ("schema_version", _string(value.schema_version)),
            ("z_src", _array(value.z_src, "z_src")),
            ("action", _array(value.action, "action")),
            ("z_next", _array(value.z_next, "z_next")),
            ("delta", _array(value.delta, "delta")),
            ("t", _int(value.t)),
            ("episode_id", _bytes(value.episode_id.bytes)),
            ("reward", _optional_float(value.reward)),
        ],
    )


def _metadata_mapping(meta: Mapping[str, Any]) -> bytes:
    fields = []
    for key, value in meta.items():
        if not isinstance(key, str):
            raise ValidationError("metadata keys must be strings")
        fields.append((key, _metadata_value(value)))
    fields.sort(key=lambda item: item[0].encode("utf-8"))
    payload = _u32(len(fields)) + b"".join(
        _string(key) + value for key, value in fields
    )
    return _frame(b"m", payload)


def _metadata_value(value: object) -> bytes:
    if value is None:
        return _frame(b"none", b"")
    if isinstance(value, bool):
        return _frame(b"bool", b"\x01" if value else b"\x00")
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return _int(value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, Mapping):
        return _metadata_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        payload = _u32(len(value)) + b"".join(_metadata_value(item) for item in value)
        return _frame(b"list", payload)
    raise ValidationError(f"unsupported metadata value: {type(value).__name__}")


def _record(name: str, fields: Sequence[tuple[str, bytes]]) -> bytes:
    payload = _string(name) + _u32(len(fields))
    for field_name, field_payload in fields:
        payload += _string(field_name) + field_payload
    return _frame(b"record", payload)


def _optional_string(value: str | None) -> bytes:
    if value is None:
        return _frame(b"none", b"")
    return _frame(b"some", _string(value))


def _optional_float(value: float | None) -> bytes:
    if value is None:
        return _frame(b"none", b"")
    return _frame(b"some", _float(value))


def _array(value: object, field_name: str) -> bytes:
    array = _as_numpy_array(value, field_name)
    if not np.issubdtype(array.dtype, np.number):
        raise DTypeError(f"{field_name} must have a numeric dtype")
    if array.ndim == 0:
        raise ShapeError(f"{field_name} must have at least one dimension")
    if any(int(dim) <= 0 for dim in array.shape):
        raise ShapeError(f"{field_name} shape dimensions must be positive")
    if not bool(np.isfinite(array).all()):
        raise ValidationError(f"{field_name} must contain only finite values")

    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and np.little_endian is False):
        dtype = dtype.newbyteorder("<")
        array = array.astype(dtype, copy=False)
    canonical = np.ascontiguousarray(array)
    payload = (
        _string(str(canonical.dtype))
        + _u32(canonical.ndim)
        + b"".join(_i64(int(dim)) for dim in canonical.shape)
        + _bytes(canonical.tobytes(order="C"))
    )
    return _frame(b"array", payload)


def _as_numpy_array(value: object, field_name: str) -> np.ndarray:
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
    raise TypeError(f"{field_name} must be a numpy array or tensor-like object")


def _string(value: str) -> bytes:
    return _frame(b"str", value.encode("utf-8"))


def _int(value: int) -> bytes:
    return _frame(b"int", str(value).encode("ascii"))


def _float(value: float) -> bytes:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValidationError("float values must be finite")
    return _frame(b"float64", struct.pack("<d", numeric))


def _bytes(value: bytes) -> bytes:
    return _frame(b"bytes", value)


def _frame(tag: bytes, payload: bytes) -> bytes:
    return _u8(len(tag)) + tag + _u64(len(payload)) + payload


def _u8(value: int) -> bytes:
    return value.to_bytes(1, "big", signed=False)


def _u32(value: int) -> bytes:
    return value.to_bytes(4, "big", signed=False)


def _u64(value: int) -> bytes:
    return value.to_bytes(8, "big", signed=False)


def _i64(value: int) -> bytes:
    return value.to_bytes(8, "big", signed=True)


__all__ = ["build_item", "canonical_bytes", "content_id"]
