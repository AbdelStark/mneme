"""Fail-closed validation helpers for remote store responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from mneme.core import (
    EvaluationError,
    FingerprintMismatchError,
    MnemeError,
    OptionalDependencyError,
    QueryError,
    QuerySpec,
    ReceiptVerificationError,
    Retrieval,
    SchemaVersionError,
    StoreCorruptionError,
    StoreError,
    UnsupportedOperationError,
    ValidationError,
    content_id,
)
from mneme.receipts import RetrievalReceipt, verify_retrieval_receipt
from mneme.remote._messages import (
    ERROR_SCHEMA,
    QUERY_RESPONSE_SCHEMA,
    ErrorMessage,
    QueryRequest,
    QueryResponse,
)

QueryResponsePayload: TypeAlias = QueryResponse | Mapping[str, object]
RemoteErrorPayload: TypeAlias = ErrorMessage | Mapping[str, object]


def validate_query_response(
    response: QueryResponsePayload,
    request: QueryRequest | QuerySpec,
) -> Retrieval:
    """Validate a remote query response before a caller can condition on it."""

    parsed = (
        response
        if isinstance(response, QueryResponse)
        else QueryResponse.from_json(response)
    )
    _ensure_schema(parsed.schema_version, QUERY_RESPONSE_SCHEMA)
    spec = request.spec if isinstance(request, QueryRequest) else request
    if not isinstance(spec, QuerySpec):
        raise ValidationError("request must be a QueryRequest or QuerySpec")

    retrieval = parsed.retrieval
    for item in retrieval.items:
        cid = content_id(item)
        if item.content_id != cid:
            raise ValidationError("remote item content_id does not match item bytes")
        if spec.encoder_fp is not None and item.encoder_fp != spec.encoder_fp:
            raise FingerprintMismatchError(
                "remote item fingerprint does not match query fingerprint"
            )

    receipt = retrieval.receipt
    if spec.with_receipt and not isinstance(receipt, RetrievalReceipt):
        raise ReceiptVerificationError("remote query response did not include receipt")
    if isinstance(receipt, RetrievalReceipt) and not verify_retrieval_receipt(
        receipt,
        retrieval.items,
        root=receipt.root,
        query=spec,
    ):
        raise ReceiptVerificationError("remote receipt verification failed")
    return retrieval


def raise_for_remote_error(error: RemoteErrorPayload) -> None:
    """Raise the local typed error represented by a remote error message."""

    parsed = error if isinstance(error, ErrorMessage) else ErrorMessage.from_json(error)
    _ensure_schema(parsed.schema_version, ERROR_SCHEMA)
    error_type = parsed.error_type
    message = f"remote {error_type}: {parsed.message}"
    if error_type == "SchemaVersionError":
        raise SchemaVersionError(message)
    if error_type == "FingerprintMismatchError":
        raise FingerprintMismatchError(message)
    if error_type == "QueryError":
        raise QueryError(message)
    if error_type == "ReceiptVerificationError":
        raise ReceiptVerificationError(message)
    if error_type == "UnsupportedOperationError":
        raise UnsupportedOperationError(message)
    if error_type == "StoreCorruptionError":
        raise StoreCorruptionError(message)
    if error_type == "EvaluationError":
        raise EvaluationError(message)
    if error_type == "OptionalDependencyError":
        raise OptionalDependencyError(message)
    if error_type == "ValidationError":
        raise ValidationError(message)
    if error_type == "StoreError":
        raise StoreError(message)
    if error_type == "MnemeError":
        raise MnemeError(message)
    raise StoreError(message)


def _ensure_schema(schema_version: str, expected: str) -> None:
    if schema_version != expected:
        raise SchemaVersionError(f"unsupported message schema: {schema_version!r}")


__all__ = ["raise_for_remote_error", "validate_query_response"]
