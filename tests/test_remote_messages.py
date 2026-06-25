from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from blake3 import blake3

from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    Retrieval,
    SchemaVersionError,
    Transition,
    ValidationError,
    content_id,
)
from mneme.receipts import CommitmentState
from mneme.remote import (
    ERROR_SCHEMA,
    PROVE_REQUEST_SCHEMA,
    PROVE_RESPONSE_SCHEMA,
    PUT_REQUEST_SCHEMA,
    PUT_RESPONSE_SCHEMA,
    QUERY_REQUEST_SCHEMA,
    QUERY_RESPONSE_SCHEMA,
    ROOT_REQUEST_SCHEMA,
    ROOT_RESPONSE_SCHEMA,
    STATS_REQUEST_SCHEMA,
    STATS_RESPONSE_SCHEMA,
    ErrorMessage,
    ProveRequest,
    ProveResponse,
    PutRequest,
    PutResponse,
    QueryRequest,
    QueryResponse,
    RemoteArray,
    RootRequest,
    RootResponse,
    StatsRequest,
    StatsResponse,
)


def test_remote_array_round_trip_preserves_dtype_shape_and_byte_order() -> None:
    array = np.ascontiguousarray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.dtype(">f4"))
    )

    payload = RemoteArray.from_array(array)
    restored = RemoteArray.from_json(payload.to_json()).to_array()

    assert payload.byte_order == "big"
    assert restored.dtype == array.dtype
    assert restored.shape == array.shape
    np.testing.assert_array_equal(restored, array)


def test_remote_messages_round_trip_every_operation() -> None:
    item = _built_item(1.0)
    cid = item.content_id or content_id(item)
    root = blake3(b"root").digest()
    state = CommitmentState.from_cids((cid,))
    proof = state.prove(cid)
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        with_receipt=True,
        encoder_fp=_fingerprint(),
    )
    retrieval = Retrieval(items=(item,), distances=(0.0,))
    messages = (
        (PUT_REQUEST_SCHEMA, PutRequest((item,)), PutRequest),
        (PUT_RESPONSE_SCHEMA, PutResponse((cid,)), PutResponse),
        (QUERY_REQUEST_SCHEMA, QueryRequest(spec), QueryRequest),
        (QUERY_RESPONSE_SCHEMA, QueryResponse(retrieval), QueryResponse),
        (PROVE_REQUEST_SCHEMA, ProveRequest((cid,)), ProveRequest),
        (PROVE_RESPONSE_SCHEMA, ProveResponse((proof,)), ProveResponse),
        (ROOT_REQUEST_SCHEMA, RootRequest(), RootRequest),
        (ROOT_RESPONSE_SCHEMA, RootResponse(root), RootResponse),
        (
            STATS_REQUEST_SCHEMA,
            StatsRequest(),
            StatsRequest,
        ),
        (
            STATS_RESPONSE_SCHEMA,
            StatsResponse({"store_id": "fixture", "value_record_count": 1}),
            StatsResponse,
        ),
        (
            ERROR_SCHEMA,
            ErrorMessage("ValidationError", "invalid request", retryable=False),
            ErrorMessage,
        ),
    )

    for schema, message, message_type in messages:
        decoded = message_type.from_json(message.to_json())
        assert decoded.to_json() == message.to_json()
        assert decoded.to_json()["schema_version"] == schema


def test_query_response_round_trips_receipt_payload() -> None:
    item = _built_item(1.0)
    cid = item.content_id or content_id(item)
    state = CommitmentState.from_cids((cid,))
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        with_receipt=True,
    )
    from mneme.receipts import build_retrieval_receipt

    receipt = build_retrieval_receipt(
        root=state.root,
        ids=(cid,),
        proofs=(state.prove(cid),),
        query=spec,
        store_id="00000000-0000-0000-0000-000000000000",
        created_at="2026-06-24T00:00:00Z",
    )
    response = QueryResponse(
        Retrieval(items=(item,), distances=(0.0,), receipt=receipt)
    )

    decoded = QueryResponse.from_json(response.to_json())

    assert decoded.retrieval.receipt == receipt
    assert decoded.retrieval.items[0].content_id == cid
    assert decoded.retrieval.distances == (0.0,)


def test_remote_messages_reject_non_digest_ids_and_roots() -> None:
    with pytest.raises(ValidationError, match="content id must be 32 bytes"):
        PutResponse((b"x",))
    with pytest.raises(ValidationError, match="content id must be 32 bytes"):
        ProveRequest.from_json({"schema_version": PROVE_REQUEST_SCHEMA, "ids": ["00"]})
    with pytest.raises(ValidationError, match="root must be 32 bytes"):
        RootResponse(b"x")
    with pytest.raises(ValidationError, match="root must be 32 bytes"):
        RootResponse.from_json({"schema_version": ROOT_RESPONSE_SCHEMA, "root": "00"})


def test_unknown_major_message_schema_fails_closed() -> None:
    payload = QueryRequest(
        QuerySpec(vector=np.array([1.0], dtype=np.float32), k=1)
    ).to_json()
    payload["schema_version"] = "mneme.query.request.v2"

    with pytest.raises(SchemaVersionError, match="unsupported message schema"):
        QueryRequest.from_json(payload)


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _built_item(key_value: float) -> MemoryItem:
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
        meta={"source": "remote-message-fixture"},
        encoder_fp=_fingerprint(),
    )
    return MemoryItem(
        content_id=content_id(item),
        key=item.key,
        value=item.value,
        meta=item.meta,
        encoder_fp=item.encoder_fp,
    )
