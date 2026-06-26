"""Schema-versioned remote store protocol messages."""

from __future__ import annotations

import base64
import binascii
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

import numpy as np

from mneme.core import (
    Cid,
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    Retrieval,
    SchemaVersionError,
    StoreCorruptionError,
    ValidationError,
    content_id,
)
from mneme.core._ids import cid_from_hex, require_cid_bytes
from mneme.receipts import InclusionProof, RetrievalReceipt
from mneme.store import StoreStats
from mneme.store._value_log import (
    _fingerprint_from_json,
    _json_ready,
    _transition_from_json,
    _transition_to_json,
)

PUT_REQUEST_SCHEMA: Final = "mneme.put.request.v1"
PUT_RESPONSE_SCHEMA: Final = "mneme.put.response.v1"
QUERY_REQUEST_SCHEMA: Final = "mneme.query.request.v1"
QUERY_RESPONSE_SCHEMA: Final = "mneme.query.response.v1"
PROVE_REQUEST_SCHEMA: Final = "mneme.prove.request.v1"
PROVE_RESPONSE_SCHEMA: Final = "mneme.prove.response.v1"
ROOT_REQUEST_SCHEMA: Final = "mneme.root.request.v1"
ROOT_RESPONSE_SCHEMA: Final = "mneme.root.response.v1"
STATS_REQUEST_SCHEMA: Final = "mneme.stats.request.v1"
STATS_RESPONSE_SCHEMA: Final = "mneme.stats.response.v1"
ERROR_SCHEMA: Final = "mneme.error.v1"

ByteOrder = Literal["little", "big", "not_applicable"]


@dataclass(frozen=True)
class RemoteArray:
    """Wire representation for numeric arrays."""

    dtype: str
    shape: tuple[int, ...]
    byte_order: ByteOrder
    data: str
    encoding: Literal["base64"] = "base64"

    def __post_init__(self) -> None:
        _require_string(self.dtype, "array dtype")
        object.__setattr__(self, "shape", _array_shape(self.shape))
        _byte_order_value(self.byte_order)
        _encoding(self.encoding)
        _require_string(self.data, "array data")

    @classmethod
    def from_array(cls, value: object) -> RemoteArray:
        """Encode a numeric array as explicit dtype/shape/base64 bytes."""

        if not isinstance(value, np.ndarray):
            raise ValidationError("array value must be a numpy.ndarray")
        if not np.issubdtype(value.dtype, np.number):
            raise ValidationError("array dtype must be numeric")
        if not bool(np.isfinite(value).all()):
            raise ValidationError("array values must be finite")
        shape = _array_shape(value.shape)
        array = np.ascontiguousarray(value)
        return cls(
            dtype=str(array.dtype),
            shape=shape,
            byte_order=_byte_order(array.dtype),
            data=base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        )

    def to_array(self) -> np.ndarray:
        """Decode the array payload and preserve dtype and shape."""

        if self.encoding != "base64":
            raise ValidationError("array encoding must be base64")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValidationError("array dtype is unsupported") from exc
        if not np.issubdtype(dtype, np.number):
            raise ValidationError("array dtype must be numeric")
        if _byte_order(dtype) != self.byte_order:
            raise ValidationError("array byte_order does not match dtype")
        try:
            raw = base64.b64decode(self.data, validate=True)
        except binascii.Error as exc:
            raise ValidationError("array data must be base64") from exc
        expected_bytes = math.prod(self.shape) * dtype.itemsize
        if len(raw) != expected_bytes:
            raise ValidationError("array data does not match dtype and shape")
        try:
            array = np.frombuffer(raw, dtype=dtype).reshape(self.shape).copy()
        except ValueError as exc:
            raise ValidationError("array data does not match dtype and shape") from exc
        if not bool(np.isfinite(array).all()):
            raise ValidationError("array values must be finite")
        return array

    def to_json(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "byte_order": self.byte_order,
            "encoding": self.encoding,
            "data": self.data,
        }

    @classmethod
    def from_json(cls, data: object) -> RemoteArray:
        mapping = _require_mapping(data, "array")
        return cls(
            dtype=_require_string(mapping.get("dtype"), "array dtype"),
            shape=_array_shape(mapping.get("shape")),
            byte_order=_byte_order_value(mapping.get("byte_order")),
            encoding=_encoding(mapping.get("encoding")),
            data=_require_string(mapping.get("data"), "array data"),
        )


@dataclass(frozen=True)
class MemoryItemEnvelope:
    """Wire envelope for a content-addressed memory item."""

    item: MemoryItem

    def __post_init__(self) -> None:
        if not isinstance(self.item, MemoryItem):
            raise ValidationError("item must be a MemoryItem")

    def to_json(self) -> dict[str, object]:
        cid = _canonical_item_content_id(self.item)
        return {
            "schema_version": self.item.schema_version,
            "content_id": cid.hex(),
            "key": RemoteArray.from_array(self.item.key).to_json(),
            "value_kind": "transition",
            "value": _transition_to_json(self.item.value),
            "meta": _json_ready(self.item.meta),
            "encoder_fp": asdict(self.item.encoder_fp),
        }

    @classmethod
    def from_json(cls, data: object) -> MemoryItemEnvelope:
        mapping = _require_mapping(data, "item")
        if mapping.get("value_kind") != "transition":
            raise ValidationError("unsupported memory item value_kind")
        try:
            item = MemoryItem(
                content_id=_bytes_from_hex(mapping.get("content_id"), "content_id"),
                key=RemoteArray.from_json(mapping.get("key")).to_array(),
                value=_transition_from_json(mapping.get("value")),
                meta=_require_mapping(mapping.get("meta"), "meta"),
                encoder_fp=_fingerprint_from_json(mapping.get("encoder_fp")),
                schema_version=_require_string(
                    mapping.get("schema_version"),
                    "item schema_version",
                ),
            )
        except (TypeError, ValueError, ValidationError, StoreCorruptionError) as exc:
            raise ValidationError("invalid memory item payload") from exc
        if item.content_id != content_id(item):
            raise ValidationError("content_id does not match canonical item bytes")
        return cls(item)


@dataclass(frozen=True)
class PutRequest:
    items: tuple[MemoryItem, ...]
    schema_version: str = PUT_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, PUT_REQUEST_SCHEMA)
        object.__setattr__(self, "items", _memory_item_sequence(self.items, "items"))

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "items": [MemoryItemEnvelope(item).to_json() for item in self.items],
        }

    @classmethod
    def from_json(cls, data: object) -> PutRequest:
        mapping = _message_mapping(data, PUT_REQUEST_SCHEMA)
        items = _require_sequence(mapping.get("items"), "items")
        return cls(tuple(MemoryItemEnvelope.from_json(item).item for item in items))


@dataclass(frozen=True)
class PutResponse:
    ids: tuple[Cid, ...]
    schema_version: str = PUT_RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, PUT_RESPONSE_SCHEMA)
        object.__setattr__(self, "ids", _cid_sequence(self.ids, "ids"))

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ids": [cid.hex() for cid in self.ids],
        }

    @classmethod
    def from_json(cls, data: object) -> PutResponse:
        mapping = _message_mapping(data, PUT_RESPONSE_SCHEMA)
        return cls(_ids_from_json(mapping.get("ids")))


@dataclass(frozen=True)
class QueryRequest:
    spec: QuerySpec
    schema_version: str = QUERY_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, QUERY_REQUEST_SCHEMA)
        if not isinstance(self.spec, QuerySpec):
            raise ValidationError("query spec must be a QuerySpec")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query": _query_to_json(self.spec),
        }

    @classmethod
    def from_json(cls, data: object) -> QueryRequest:
        mapping = _message_mapping(data, QUERY_REQUEST_SCHEMA)
        return cls(_query_from_json(mapping.get("query")))


@dataclass(frozen=True)
class QueryResponse:
    retrieval: Retrieval
    schema_version: str = QUERY_RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, QUERY_RESPONSE_SCHEMA)
        if not isinstance(self.retrieval, Retrieval):
            raise ValidationError("retrieval must be a Retrieval")
        if self.retrieval.receipt is not None and not isinstance(
            self.retrieval.receipt,
            RetrievalReceipt,
        ):
            raise ValidationError("receipt must be a RetrievalReceipt")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "items": [
                MemoryItemEnvelope(item).to_json() for item in self.retrieval.items
            ],
            "distances": [float(distance) for distance in self.retrieval.distances],
            "receipt": None
            if self.retrieval.receipt is None
            else _receipt_to_json(self.retrieval.receipt),
        }

    @classmethod
    def from_json(cls, data: object) -> QueryResponse:
        mapping = _message_mapping(data, QUERY_RESPONSE_SCHEMA)
        items = _require_sequence(mapping.get("items"), "items")
        distances = _require_sequence(mapping.get("distances"), "distances")
        return cls(
            Retrieval(
                items=tuple(MemoryItemEnvelope.from_json(item).item for item in items),
                distances=tuple(_require_float(item, "distance") for item in distances),
                receipt=_optional_receipt(mapping.get("receipt")),
            )
        )


@dataclass(frozen=True)
class ProveRequest:
    ids: tuple[Cid, ...]
    schema_version: str = PROVE_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, PROVE_REQUEST_SCHEMA)
        object.__setattr__(self, "ids", _cid_sequence(self.ids, "ids"))

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ids": [cid.hex() for cid in self.ids],
        }

    @classmethod
    def from_json(cls, data: object) -> ProveRequest:
        mapping = _message_mapping(data, PROVE_REQUEST_SCHEMA)
        return cls(_ids_from_json(mapping.get("ids")))


@dataclass(frozen=True)
class ProveResponse:
    proofs: tuple[InclusionProof, ...]
    schema_version: str = PROVE_RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, PROVE_RESPONSE_SCHEMA)
        object.__setattr__(self, "proofs", _proof_sequence(self.proofs, "proofs"))

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "proofs": [proof.to_json() for proof in self.proofs],
        }

    @classmethod
    def from_json(cls, data: object) -> ProveResponse:
        mapping = _message_mapping(data, PROVE_RESPONSE_SCHEMA)
        proofs = _require_sequence(mapping.get("proofs"), "proofs")
        return cls(tuple(InclusionProof.from_json(proof) for proof in proofs))


@dataclass(frozen=True)
class RootRequest:
    schema_version: str = ROOT_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, ROOT_REQUEST_SCHEMA)

    def to_json(self) -> dict[str, object]:
        return {"schema_version": self.schema_version}

    @classmethod
    def from_json(cls, data: object) -> RootRequest:
        _message_mapping(data, ROOT_REQUEST_SCHEMA)
        return cls()


@dataclass(frozen=True)
class RootResponse:
    root: bytes
    schema_version: str = ROOT_RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, ROOT_RESPONSE_SCHEMA)
        object.__setattr__(
            self,
            "root",
            require_cid_bytes(
                self.root,
                "root",
                type_error=ValidationError,
                value_error=ValidationError,
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "root": self.root.hex()}

    @classmethod
    def from_json(cls, data: object) -> RootResponse:
        mapping = _message_mapping(data, ROOT_RESPONSE_SCHEMA)
        return cls(_bytes_from_hex(mapping.get("root"), "root"))


@dataclass(frozen=True)
class StatsRequest:
    schema_version: str = STATS_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, STATS_REQUEST_SCHEMA)

    def to_json(self) -> dict[str, object]:
        return {"schema_version": self.schema_version}

    @classmethod
    def from_json(cls, data: object) -> StatsRequest:
        _message_mapping(data, STATS_REQUEST_SCHEMA)
        return cls()


@dataclass(frozen=True)
class StatsResponse:
    stats: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = STATS_RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, STATS_RESPONSE_SCHEMA)
        object.__setattr__(self, "stats", _json_object(self.stats, "stats"))

    @classmethod
    def from_store_stats(cls, stats: StoreStats) -> StatsResponse:
        return cls(
            {
                "store_id": str(stats.store_id),
                "schema_version": stats.schema_version,
                "active_fingerprint_count": stats.active_fingerprint_count,
                "value_log_count": stats.value_log_count,
                "value_record_count": stats.value_record_count,
                "visible_record_count": stats.visible_record_count,
                "value_bytes": stats.value_bytes,
                "index_backend": stats.index_backend,
                "retention_policy": stats.retention_policy,
                "tombstone_count": stats.tombstone_count,
                "last_completed_transaction": stats.last_completed_transaction,
                "commitments_enabled": stats.commitments_enabled,
            }
        )

    def to_json(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "stats": _json_ready(self.stats)}

    @classmethod
    def from_json(cls, data: object) -> StatsResponse:
        mapping = _message_mapping(data, STATS_RESPONSE_SCHEMA)
        return cls(_require_mapping(mapping.get("stats"), "stats"))


@dataclass(frozen=True)
class ErrorMessage:
    error_type: str
    message: str
    retryable: bool = False
    schema_version: str = ERROR_SCHEMA
    extra: str | None = None
    package: str | None = None

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, ERROR_SCHEMA)
        _require_string(self.error_type, "error_type")
        _require_string(self.message, "message")
        _require_bool(self.retryable, "retryable")
        _optional_string(self.extra, "extra")
        _optional_string(self.package, "package")

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "error_type": self.error_type,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.extra is not None:
            payload["extra"] = self.extra
        if self.package is not None:
            payload["package"] = self.package
        return payload

    @classmethod
    def from_json(cls, data: object) -> ErrorMessage:
        mapping = _message_mapping(data, ERROR_SCHEMA)
        retryable = mapping.get("retryable")
        if not isinstance(retryable, bool):
            raise ValidationError("retryable must be a bool")
        return cls(
            error_type=_require_string(mapping.get("error_type"), "error_type"),
            message=_require_string(mapping.get("message"), "message"),
            retryable=retryable,
            extra=_optional_string(mapping.get("extra"), "extra"),
            package=_optional_string(mapping.get("package"), "package"),
        )


def _query_to_json(spec: QuerySpec) -> dict[str, object]:
    return {
        "schema_version": spec.schema_version,
        "vector": RemoteArray.from_array(spec.vector).to_json(),
        "k": spec.k,
        "metric": spec.metric.value,
        "ef": spec.ef,
        "filters": None if spec.filters is None else _json_ready(spec.filters),
        "temporal_decay": spec.temporal_decay,
        "with_receipt": spec.with_receipt,
        "encoder_fp": None if spec.encoder_fp is None else asdict(spec.encoder_fp),
    }


def _query_from_json(data: object) -> QuerySpec:
    mapping = _require_mapping(data, "query")
    return QuerySpec(
        vector=RemoteArray.from_json(mapping.get("vector")).to_array(),
        k=_require_positive_int(mapping.get("k"), "k"),
        metric=_metric(mapping.get("metric")),
        ef=_optional_positive_int(mapping.get("ef"), "ef"),
        filters=_optional_mapping(mapping.get("filters"), "filters"),
        temporal_decay=_optional_non_negative_float(
            mapping.get("temporal_decay"),
            "temporal_decay",
        ),
        with_receipt=_require_bool(mapping.get("with_receipt"), "with_receipt"),
        encoder_fp=_optional_fingerprint(mapping.get("encoder_fp")),
        schema_version=_require_string(mapping.get("schema_version"), "schema_version"),
    )


def _canonical_item_content_id(item: MemoryItem) -> Cid:
    cid = content_id(item)
    if item.content_id is not None and item.content_id != cid:
        raise ValidationError("content_id does not match canonical item bytes")
    return cid


def _message_mapping(data: object, expected_schema: str) -> Mapping[str, Any]:
    mapping = _require_mapping(data, "message")
    _validate_schema(
        _require_string(mapping.get("schema_version"), "schema_version"),
        expected_schema,
    )
    return mapping


def _validate_schema(schema_version: str, expected: str) -> None:
    if not isinstance(schema_version, str):
        raise SchemaVersionError("message schema_version must be a string")
    prefix, _, major_text = schema_version.rpartition(".v")
    expected_prefix, _, _ = expected.rpartition(".v")
    if prefix != expected_prefix or not major_text.isdigit():
        raise SchemaVersionError(f"unsupported message schema: {schema_version!r}")
    if int(major_text) != 1 or schema_version != expected:
        raise SchemaVersionError(f"unsupported message schema: {schema_version!r}")


def _byte_order(dtype: np.dtype[Any]) -> ByteOrder:
    if dtype.byteorder == "|":
        return "not_applicable"
    little = dtype.byteorder == "<" or (dtype.byteorder == "=" and np.little_endian)
    return "little" if little else "big"


def _byte_order_value(value: object) -> ByteOrder:
    if value == "little" or value == "big" or value == "not_applicable":
        return value
    raise ValidationError("array byte_order is unsupported")


def _encoding(value: object) -> Literal["base64"]:
    if value == "base64":
        return "base64"
    raise ValidationError("array encoding must be base64")


def _array_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValidationError("array shape must be a sequence of integers")
    shape: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValidationError("array shape must be a sequence of integers")
        if dim <= 0:
            raise ValidationError("array shape dimensions must be positive")
        shape.append(dim)
    if not shape:
        raise ValidationError("array shape must include at least one dimension")
    return tuple(shape)


def _ids_from_json(data: object) -> tuple[Cid, ...]:
    values = _require_sequence(data, "ids")
    return tuple(
        cid_from_hex(item, "content id", error_type=ValidationError) for item in values
    )


def _cid_sequence(value: object, field_name: str) -> tuple[Cid, ...]:
    values = _require_sequence(value, field_name)
    return tuple(
        require_cid_bytes(
            item,
            "content id",
            type_error=ValidationError,
            value_error=ValidationError,
        )
        for item in values
    )


def _memory_item_sequence(value: object, field_name: str) -> tuple[MemoryItem, ...]:
    values = _require_sequence(value, field_name)
    items: list[MemoryItem] = []
    for item in values:
        if not isinstance(item, MemoryItem):
            raise ValidationError(f"{field_name} must contain only MemoryItem values")
        items.append(item)
    return tuple(items)


def _proof_sequence(value: object, field_name: str) -> tuple[InclusionProof, ...]:
    values = _require_sequence(value, field_name)
    proofs: list[InclusionProof] = []
    for proof in values:
        if not isinstance(proof, InclusionProof):
            raise ValidationError(
                f"{field_name} must contain only InclusionProof values"
            )
        proofs.append(proof)
    return tuple(proofs)


def _receipt_to_json(value: object) -> dict[str, object]:
    if not isinstance(value, RetrievalReceipt):
        raise ValidationError("receipt must be a RetrievalReceipt")
    return value.to_json()


def _optional_receipt(value: object) -> RetrievalReceipt | None:
    if value is None:
        return None
    return RetrievalReceipt.from_json(value)


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return value


def _json_object(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    ready = _json_ready(value)
    if not isinstance(ready, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return ready


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValidationError(f"{field_name} must be a sequence")
    return value


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a bool")
    return value


def _require_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field_name} must be a finite number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValidationError(f"{field_name} must be finite")
    return numeric


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field_name)


def _optional_non_negative_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    numeric = _require_float(value, field_name)
    if numeric < 0.0:
        raise ValidationError(f"{field_name} must be non-negative")
    return numeric


def _optional_mapping(
    value: object,
    field_name: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _require_mapping(value, field_name)


def _metric(value: object) -> Metric:
    text = _require_string(value, "metric")
    try:
        return Metric(text)
    except ValueError as exc:
        raise ValidationError(f"unsupported metric: {text}") from exc


def _optional_fingerprint(value: object) -> EncoderFingerprint | None:
    if value is None:
        return None
    try:
        return _fingerprint_from_json(value)
    except StoreCorruptionError as exc:
        raise ValidationError("invalid encoder_fp") from exc


def _bytes_from_hex(value: object, field_name: str) -> bytes:
    return cid_from_hex(value, field_name, error_type=ValidationError)


__all__ = [
    "ERROR_SCHEMA",
    "PROVE_REQUEST_SCHEMA",
    "PROVE_RESPONSE_SCHEMA",
    "PUT_REQUEST_SCHEMA",
    "PUT_RESPONSE_SCHEMA",
    "QUERY_REQUEST_SCHEMA",
    "QUERY_RESPONSE_SCHEMA",
    "ROOT_REQUEST_SCHEMA",
    "ROOT_RESPONSE_SCHEMA",
    "STATS_REQUEST_SCHEMA",
    "STATS_RESPONSE_SCHEMA",
    "ErrorMessage",
    "MemoryItemEnvelope",
    "ProveRequest",
    "ProveResponse",
    "PutRequest",
    "PutResponse",
    "QueryRequest",
    "QueryResponse",
    "RemoteArray",
    "RootRequest",
    "RootResponse",
    "StatsRequest",
    "StatsResponse",
]
