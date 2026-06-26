"""Retrieval receipt construction and verification."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from blake3 import blake3

from mneme.core import (
    Cid,
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    ReceiptVerificationError,
    SchemaVersionError,
    ValidationError,
    content_id,
)
from mneme.receipts._mmr import InclusionProof, verify_inclusion_proof

QUERY_RECEIPT_PARAMS_SCHEMA: Final = "mneme.query_receipt_params.v1"
RETRIEVAL_RECEIPT_SCHEMA: Final = "mneme.receipt.v1"
_DIGEST_SIZE: Final = 32
_QUERY_VECTOR_PREFIX: Final = b"mneme.receipt.v1.query_vector"


@dataclass(frozen=True)
class QueryReceiptParams:
    """Receipt-bound query parameters."""

    vector_digest: bytes
    vector_shape: tuple[int, ...]
    vector_dtype: str
    k: int
    metric: Metric
    ef: int | None = None
    filters: Mapping[str, Any] | None = None
    temporal_decay: float | None = None
    encoder_fp: EncoderFingerprint | None = None
    schema_version: str = QUERY_RECEIPT_PARAMS_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(
            self.schema_version,
            QUERY_RECEIPT_PARAMS_SCHEMA,
            "query receipt params",
        )
        _require_digest(self.vector_digest, "query vector digest")
        object.__setattr__(
            self,
            "vector_shape",
            _require_shape(self.vector_shape, "query vector shape"),
        )
        _require_string(self.vector_dtype, "query vector dtype")
        _require_positive_int(self.k, "k")
        if not isinstance(self.metric, Metric):
            raise ValidationError("metric must be a Metric")
        if self.ef is not None:
            _require_positive_int(self.ef, "ef")
            if self.ef < self.k:
                raise ValidationError("ef must be greater than or equal to k")
        if self.filters is not None:
            object.__setattr__(
                self,
                "filters",
                _freeze_json_mapping(self.filters, "filters"),
            )
        if self.temporal_decay is not None:
            _require_non_negative_float(self.temporal_decay, "temporal_decay")
        if self.encoder_fp is not None and not isinstance(
            self.encoder_fp,
            EncoderFingerprint,
        ):
            raise ValidationError("encoder_fp must be an EncoderFingerprint or None")

    @classmethod
    def from_query(cls, spec: QuerySpec) -> QueryReceiptParams:
        """Build receipt parameters from a query request."""

        if not isinstance(spec, QuerySpec):
            raise ValidationError("spec must be a QuerySpec")
        vector = np.ascontiguousarray(spec.vector, dtype=np.float32)
        return cls(
            vector_digest=_query_vector_digest(vector),
            vector_shape=tuple(int(dim) for dim in vector.shape),
            vector_dtype=str(vector.dtype),
            k=spec.k,
            metric=spec.metric,
            ef=spec.ef,
            filters=spec.filters,
            temporal_decay=spec.temporal_decay,
            encoder_fp=spec.encoder_fp,
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready params object."""

        return {
            "schema_version": self.schema_version,
            "vector_digest": self.vector_digest.hex(),
            "vector_shape": list(self.vector_shape),
            "vector_dtype": self.vector_dtype,
            "k": self.k,
            "metric": self.metric.value,
            "ef": self.ef,
            "filters": None
            if self.filters is None
            else _json_ready_mapping(self.filters),
            "temporal_decay": self.temporal_decay,
            "encoder_fp": None if self.encoder_fp is None else asdict(self.encoder_fp),
        }

    @classmethod
    def from_json(cls, data: object) -> QueryReceiptParams:
        mapping = _require_mapping(data, "query receipt params")
        return cls(
            schema_version=_require_string(
                mapping.get("schema_version"),
                "query receipt params schema_version",
            ),
            vector_digest=_bytes_from_hex(
                mapping.get("vector_digest"),
                "query vector digest",
            ),
            vector_shape=_shape_from_json(mapping.get("vector_shape")),
            vector_dtype=_require_string(
                mapping.get("vector_dtype"),
                "query vector dtype",
            ),
            k=_require_positive_int(mapping.get("k"), "k"),
            metric=_metric_from_json(mapping.get("metric")),
            ef=_optional_positive_int(mapping.get("ef"), "ef"),
            filters=_optional_mapping(mapping.get("filters"), "filters"),
            temporal_decay=_optional_non_negative_float(
                mapping.get("temporal_decay"),
                "temporal_decay",
            ),
            encoder_fp=_optional_encoder_fingerprint(mapping.get("encoder_fp")),
        )


@dataclass(frozen=True)
class RetrievalReceipt:
    """Membership receipt for one retrieval result set."""

    root: bytes
    ids: tuple[Cid, ...]
    proofs: tuple[InclusionProof, ...]
    params: QueryReceiptParams
    store_id: str
    created_at: str
    signer: str | None = None
    signature: bytes | None = None
    schema_version: str = RETRIEVAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, RETRIEVAL_RECEIPT_SCHEMA, "receipt")
        _require_digest(self.root, "receipt root")
        object.__setattr__(
            self,
            "ids",
            tuple(
                _require_digest(cid, "receipt id")
                for cid in _require_sequence(self.ids, "receipt ids")
            ),
        )
        object.__setattr__(
            self,
            "proofs",
            tuple(
                _require_inclusion_proof(proof)
                for proof in _require_sequence(self.proofs, "receipt proofs")
            ),
        )
        if len(self.ids) != len(self.proofs):
            raise ValidationError("receipt ids and proofs must have matching lengths")
        if not isinstance(self.params, QueryReceiptParams):
            raise ValidationError("params must be QueryReceiptParams")
        _require_string(self.store_id, "store_id")
        _require_utc_timestamp(self.created_at, "created_at")
        if (self.signer is None) != (self.signature is None):
            raise ValidationError("signer and signature must both be set or both null")
        if self.signer is not None:
            _require_string(self.signer, "signer")
        if self.signature is not None:
            object.__setattr__(
                self,
                "signature",
                _require_signature(self.signature),
            )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready receipt object."""

        return {
            "schema_version": self.schema_version,
            "root": self.root.hex(),
            "ids": [cid.hex() for cid in self.ids],
            "proofs": [proof.to_json() for proof in self.proofs],
            "params": self.params.to_json(),
            "store_id": self.store_id,
            "created_at": self.created_at,
            "signer": self.signer,
            "signature": None if self.signature is None else self.signature.hex(),
        }

    @classmethod
    def from_json(cls, data: object) -> RetrievalReceipt:
        mapping = _require_mapping(data, "receipt")
        ids = mapping.get("ids")
        if not isinstance(ids, list):
            raise ValidationError("receipt ids must be a list")
        proofs = mapping.get("proofs")
        if not isinstance(proofs, list):
            raise ValidationError("receipt proofs must be a list")
        return cls(
            schema_version=_require_string(
                mapping.get("schema_version"),
                "receipt schema_version",
            ),
            root=_bytes_from_hex(mapping.get("root"), "receipt root"),
            ids=tuple(_bytes_from_hex(cid, "receipt id") for cid in ids),
            proofs=tuple(InclusionProof.from_json(proof) for proof in proofs),
            params=QueryReceiptParams.from_json(mapping.get("params")),
            store_id=_require_string(mapping.get("store_id"), "store_id"),
            created_at=_require_string(mapping.get("created_at"), "created_at"),
            signer=_optional_string(mapping.get("signer"), "signer"),
            signature=_optional_bytes_from_hex(mapping.get("signature"), "signature"),
        )


def build_retrieval_receipt(
    *,
    root: bytes,
    ids: Sequence[Cid],
    proofs: Sequence[InclusionProof],
    query: QuerySpec,
    store_id: str,
    created_at: str | None = None,
    signer: str | None = None,
    signature: bytes | None = None,
) -> RetrievalReceipt:
    """Build a retrieval receipt from committed ids and query parameters."""

    return RetrievalReceipt(
        root=root,
        ids=tuple(ids),
        proofs=tuple(proofs),
        params=QueryReceiptParams.from_query(query),
        store_id=store_id,
        created_at=_utc_now() if created_at is None else created_at,
        signer=signer,
        signature=signature,
    )


def verify_retrieval_receipt(
    receipt: RetrievalReceipt,
    items: Sequence[MemoryItem] | None = None,
    *,
    root: bytes | None = None,
    query: QuerySpec | None = None,
) -> bool:
    """Return whether an unsigned receipt verifies for root, items, and query."""

    if not isinstance(receipt, RetrievalReceipt):
        raise ReceiptVerificationError("receipt must be a RetrievalReceipt")
    # Signature fields are schema-reserved for a later signing backend. Treat a
    # signed receipt as unverifiable until that backend is available.
    if receipt.signature is not None:
        return False
    if root is not None and _require_digest(root, "expected root") != receipt.root:
        return False
    if query is not None and QueryReceiptParams.from_query(query) != receipt.params:
        return False
    if items is not None:
        try:
            item_ids = tuple(_canonical_item_id(item) for item in items)
        except (TypeError, ValueError, ValidationError):
            return False
        if item_ids != receipt.ids:
            return False
    return all(
        verify_inclusion_proof(cid, proof, receipt.root)
        for cid, proof in zip(receipt.ids, receipt.proofs, strict=True)
    )


def _canonical_item_id(item: MemoryItem) -> Cid:
    if not isinstance(item, MemoryItem):
        raise ValidationError("items must contain MemoryItem instances")
    cid = content_id(item)
    if item.content_id is not None and item.content_id != cid:
        raise ValidationError("item content_id does not match canonical item bytes")
    return cid


def _query_vector_digest(vector: np.ndarray) -> bytes:
    metadata = json.dumps(
        {
            "dtype": str(vector.dtype),
            "shape": [int(dim) for dim in vector.shape],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return blake3(
        _QUERY_VECTOR_PREFIX
        + len(metadata).to_bytes(8, "big")
        + metadata
        + vector.tobytes(order="C")
    ).digest(length=_DIGEST_SIZE)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_schema(schema_version: str, expected: str, name: str) -> None:
    if not isinstance(schema_version, str):
        raise SchemaVersionError(f"{name} schema_version must be a string")
    if schema_version != expected:
        raise SchemaVersionError(f"unsupported {name} schema: {schema_version!r}")


def _require_digest(value: object, field_name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != _DIGEST_SIZE:
        raise ValidationError(f"{field_name} must be {_DIGEST_SIZE} bytes")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field_name)


def _require_non_negative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field_name} must be a non-negative finite number")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValidationError(f"{field_name} must be a non-negative finite number")
    return numeric


def _optional_non_negative_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_non_negative_float(value, field_name)


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_utc_timestamp(value: object, field_name: str) -> str:
    text = _require_string(value, field_name)
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError as exc:
        raise ValidationError(
            f"{field_name} must be an ISO 8601 UTC timestamp",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValidationError(f"{field_name} must be an ISO 8601 UTC timestamp")
    return text


def _require_shape(value: object, field_name: str) -> tuple[int, ...]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValidationError(f"{field_name} must be a sequence")
    shape = tuple(value)
    if not shape:
        raise ValidationError(f"{field_name} must not be empty")
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in shape
    ):
        raise ValidationError(f"{field_name} must contain positive integers")
    return shape


def _shape_from_json(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValidationError("query vector shape must be a list")
    return _require_shape(value, "query vector shape")


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return value


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValidationError(f"{field_name} must be a sequence")
    return value


def _optional_mapping(
    value: object,
    field_name: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _freeze_json_mapping(_require_mapping(value, field_name), field_name)


def _freeze_json_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    frozen: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str) or not key:
            raise ValidationError(f"{field_name} keys must be non-empty strings")
        frozen[key] = _freeze_json_value(nested, f"{field_name}.{key}")
    return MappingProxyType(frozen)


def _freeze_json_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValidationError(f"{field_name} must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze_json_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValidationError(f"{field_name} must be JSON-compatible")


def _json_ready_mapping(value: Mapping[str, Any]) -> dict[str, object]:
    return {key: _json_ready_value(nested) for key, nested in value.items()}


def _json_ready_value(value: object) -> object:
    if value is None or isinstance(value, bool | str | int | float):
        return value
    if isinstance(value, Mapping):
        return _json_ready_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready_value(item) for item in value]
    raise ValidationError(f"unsupported JSON value: {type(value).__name__}")


def _metric_from_json(value: object) -> Metric:
    text = _require_string(value, "metric")
    try:
        return Metric(text)
    except ValueError as exc:
        raise ValidationError(f"unsupported metric: {text}") from exc


def _optional_encoder_fingerprint(value: object) -> EncoderFingerprint | None:
    if value is None:
        return None
    mapping = _require_mapping(value, "encoder_fp")
    weights_digest = mapping.get("weights_digest")
    if weights_digest is not None and not isinstance(weights_digest, str):
        raise ValidationError("encoder_fp weights_digest must be a string or null")
    try:
        return EncoderFingerprint(
            encoder_id=_require_string(mapping.get("encoder_id"), "encoder_id"),
            summarizer_id=_require_string(
                mapping.get("summarizer_id"),
                "summarizer_id",
            ),
            weights_digest=weights_digest,
            config_digest=_require_string(
                mapping.get("config_digest"),
                "config_digest",
            ),
            schema_version=_require_string(
                mapping.get("schema_version"),
                "encoder_fp schema_version",
            ),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValidationError("invalid encoder_fp") from exc


def _require_inclusion_proof(value: object) -> InclusionProof:
    if not isinstance(value, InclusionProof):
        raise ValidationError("proofs must contain InclusionProof instances")
    return value


def _require_signature(value: object) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise ValidationError("signature must be non-empty bytes")
    return value


def _bytes_from_hex(value: object, field_name: str) -> bytes:
    text = _require_string(value, field_name)
    try:
        return bytes.fromhex(text)
    except ValueError as exc:
        raise ValidationError(f"{field_name} must be hex bytes") from exc


def _optional_bytes_from_hex(value: object, field_name: str) -> bytes | None:
    if value is None:
        return None
    return _bytes_from_hex(value, field_name)


__all__ = [
    "QUERY_RECEIPT_PARAMS_SCHEMA",
    "RETRIEVAL_RECEIPT_SCHEMA",
    "QueryReceiptParams",
    "RetrievalReceipt",
    "build_retrieval_receipt",
    "verify_retrieval_receipt",
]
