from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from mneme.core import (
    EncoderFingerprint,
    FingerprintMismatchError,
    MemoryItem,
    Metric,
    QueryError,
    QuerySpec,
    ReceiptVerificationError,
    Retrieval,
    SchemaVersionError,
    Transition,
    ValidationError,
    content_id,
)
from mneme.receipts import CommitmentState, build_retrieval_receipt
from mneme.remote import (
    ErrorMessage,
    QueryResponse,
    raise_for_remote_error,
    validate_query_response,
)


def test_validate_query_response_accepts_verified_receipt() -> None:
    item = _built_item(1.0)
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        with_receipt=True,
        encoder_fp=_fingerprint("meanpool-v1"),
    )
    response = _response_with_receipt(spec, item)

    retrieval = validate_query_response(response.to_json(), spec)

    assert retrieval.items[0].content_id == item.content_id
    assert retrieval.receipt is not None


def test_validate_query_response_fails_closed_on_malformed_schema() -> None:
    item = _built_item(1.0)
    response = QueryResponse(Retrieval(items=(item,), distances=(0.0,))).to_json()
    response["schema_version"] = "mneme.query.response.v2"

    with pytest.raises(SchemaVersionError, match="unsupported message schema"):
        validate_query_response(response, QuerySpec(item.key, k=1, metric=Metric.L2))


def test_validate_query_response_rejects_typed_response_schema_bypass() -> None:
    item = _built_item(1.0)
    response = QueryResponse(
        Retrieval(items=(item,), distances=(0.0,)),
        schema_version="mneme.query.response.v2",
    )

    with pytest.raises(SchemaVersionError, match="unsupported message schema"):
        validate_query_response(response, QuerySpec(item.key, k=1, metric=Metric.L2))


def test_validate_query_response_rejects_forged_content_id() -> None:
    item = replace(_built_item(1.0), content_id=b"\x11" * 32)
    response = QueryResponse(Retrieval(items=(item,), distances=(0.0,)))

    with pytest.raises(ValidationError, match="content_id"):
        validate_query_response(response, QuerySpec(item.key, k=1, metric=Metric.L2))


def test_validate_query_response_rejects_fingerprint_mismatch() -> None:
    item = _built_item(1.0, fingerprint=_fingerprint("right"))
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        encoder_fp=_fingerprint("left"),
    )
    response = QueryResponse(Retrieval(items=(item,), distances=(0.0,)))

    with pytest.raises(FingerprintMismatchError, match="fingerprint"):
        validate_query_response(response, spec)


def test_validate_query_response_maps_receipt_failure_to_typed_error() -> None:
    item = _built_item(1.0)
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        with_receipt=True,
    )
    response = _response_with_receipt(spec, item)
    assert response.retrieval.receipt is not None
    bad_receipt = replace(response.retrieval.receipt, root=b"\x00" * 32)
    bad_response = QueryResponse(
        Retrieval(
            items=response.retrieval.items,
            distances=response.retrieval.distances,
            receipt=bad_receipt,
        )
    )

    with pytest.raises(ReceiptVerificationError, match="verification failed"):
        validate_query_response(bad_response, spec)


def test_validate_query_response_requires_receipt_when_requested() -> None:
    item = _built_item(1.0)
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        with_receipt=True,
    )
    response = QueryResponse(Retrieval(items=(item,), distances=(0.0,)))

    with pytest.raises(ReceiptVerificationError, match="did not include receipt"):
        validate_query_response(response, spec)


def test_remote_error_mapping_raises_local_typed_errors() -> None:
    with pytest.raises(QueryError, match="remote QueryError"):
        raise_for_remote_error(ErrorMessage("QueryError", "bad query"))
    with pytest.raises(ReceiptVerificationError, match="remote"):
        raise_for_remote_error(
            ErrorMessage("ReceiptVerificationError", "bad proof").to_json()
        )


def _response_with_receipt(spec: QuerySpec, item: MemoryItem) -> QueryResponse:
    cid = item.content_id or content_id(item)
    state = CommitmentState.from_cids((cid,))
    receipt = build_retrieval_receipt(
        root=state.root,
        ids=(cid,),
        proofs=(state.prove(cid),),
        query=spec,
        store_id="00000000-0000-0000-0000-000000000000",
        created_at="2026-06-24T00:00:00Z",
    )
    return QueryResponse(Retrieval(items=(item,), distances=(0.0,), receipt=receipt))


def _fingerprint(summarizer_id: str = "meanpool-v1") -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id=summarizer_id,
        weights_digest=None,
        config_digest="blake3:config",
    )


def _built_item(
    key_value: float,
    *,
    fingerprint: EncoderFingerprint | None = None,
) -> MemoryItem:
    z_src = np.array([key_value, 0.0], dtype=np.float32)
    z_next = np.array([key_value + 1.0, 0.0], dtype=np.float32)
    item = MemoryItem(
        content_id=None,
        key=np.array([key_value, 0.0], dtype=np.float32),
        value=Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_next,
            delta=z_next - z_src,
            t=0,
            episode_id=uuid4(),
        ),
        meta={"source": "remote-validation-fixture"},
        encoder_fp=_fingerprint() if fingerprint is None else fingerprint,
    )
    return MemoryItem(
        content_id=content_id(item),
        key=item.key,
        value=item.value,
        meta=item.meta,
        encoder_fp=item.encoder_fp,
    )
