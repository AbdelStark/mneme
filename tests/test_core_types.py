from __future__ import annotations

from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pytest

from mneme.core import (
    ENCODER_FINGERPRINT_SCHEMA,
    MEMORY_ITEM_SCHEMA,
    QUERY_SPEC_SCHEMA,
    RETRIEVAL_SCHEMA,
    TRANSITION_SCHEMA,
    Cid,
    EncoderFingerprint,
    Latent,
    MemoryItem,
    Metric,
    QueryError,
    QuerySpec,
    Retrieval,
    SummaryVec,
    Transition,
)


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder",
        summarizer_id="mean_pool",
        weights_digest=None,
        config_digest="sha256:config",
    )


def _transition() -> Transition:
    z_src = np.array([[1.0, 2.0]], dtype=np.float32)
    z_next = np.array([[1.5, 2.5]], dtype=np.float32)
    return Transition(
        z_src=z_src,
        action=np.array([0.1, 0.2], dtype=np.float32),
        z_next=z_next,
        delta=z_next - z_src,
        t=3,
        episode_id=uuid4(),
        reward=1.0,
    )


def _key() -> np.ndarray:
    return np.array([0.6, 0.8], dtype=np.float32)


def _item() -> MemoryItem:
    return MemoryItem(
        content_id=None,
        key=_key(),
        value=_transition(),
        meta={"episode": "demo", "tags": ["fixture", 1], "score": 1.5},
        encoder_fp=_fingerprint(),
    )


def test_public_core_types_import() -> None:
    assert ENCODER_FINGERPRINT_SCHEMA == "mneme.encoder_fingerprint.v1"
    assert TRANSITION_SCHEMA == "mneme.transition.v1"
    assert MEMORY_ITEM_SCHEMA == "mneme.memory_item.v1"
    assert QUERY_SPEC_SCHEMA == "mneme.query_spec.v1"
    assert RETRIEVAL_SCHEMA == "mneme.retrieval.v1"
    assert Cid is bytes
    assert Latent is not None
    assert SummaryVec is not None


def test_valid_construction_freezes_metadata_and_sequences() -> None:
    item = _item()
    query = QuerySpec(vector=_key(), k=2, ef=4, encoder_fp=item.encoder_fp)
    retrieval = Retrieval(items=[item], distances=[0.25])

    assert item.schema_version == MEMORY_ITEM_SCHEMA
    assert query.metric is Metric.COSINE
    assert retrieval.items == (item,)
    assert retrieval.distances == (0.25,)
    assert isinstance(item.meta, MappingProxyType)
    assert item.meta["tags"] == ("fixture", 1)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (EncoderFingerprint, {"schema_version": "mneme.encoder_fingerprint.v2"}),
        (Transition, {"schema_version": "mneme.transition.v2"}),
        (MemoryItem, {"schema_version": "mneme.memory_item.v2"}),
        (QuerySpec, {"schema_version": "mneme.query_spec.v2"}),
        (Retrieval, {"schema_version": "mneme.retrieval.v2"}),
    ],
)
def test_invalid_schema_major_versions_are_rejected(
    factory: object, kwargs: dict
) -> None:
    base_kwargs = {
        EncoderFingerprint: {
            "encoder_id": "encoder",
            "summarizer_id": "mean_pool",
            "weights_digest": None,
            "config_digest": "sha256:config",
        },
        Transition: {
            "z_src": np.array([1.0], dtype=np.float32),
            "action": np.array([0.1], dtype=np.float32),
            "z_next": np.array([2.0], dtype=np.float32),
            "delta": np.array([1.0], dtype=np.float32),
            "t": 0,
            "episode_id": uuid4(),
        },
        MemoryItem: {
            "content_id": None,
            "key": _key(),
            "value": _transition(),
            "meta": {},
            "encoder_fp": _fingerprint(),
        },
        QuerySpec: {"vector": _key(), "k": 1},
        Retrieval: {"items": [], "distances": []},
    }
    call_kwargs = base_kwargs[factory] | kwargs

    with pytest.raises(ValueError, match="unsupported schema version"):
        factory(**call_kwargs)


def test_summary_vec_validation_rejects_invalid_dtype_shape_and_values() -> None:
    with pytest.raises(QueryError, match="dtype float32"):
        QuerySpec(vector=np.array([1.0], dtype=np.float64), k=1)
    with pytest.raises(QueryError, match="one-dimensional"):
        QuerySpec(vector=np.array([[1.0]], dtype=np.float32), k=1)
    with pytest.raises(QueryError, match="finite"):
        QuerySpec(vector=np.array([np.nan], dtype=np.float32), k=1)


def test_transition_validation_rejects_invalid_shape_dtype_uuid_and_step() -> None:
    valid = {
        "z_src": np.array([1.0], dtype=np.float32),
        "action": np.array([0.1], dtype=np.float32),
        "z_next": np.array([2.0], dtype=np.float32),
        "delta": np.array([1.0], dtype=np.float32),
        "t": 0,
        "episode_id": uuid4(),
    }

    with pytest.raises(ValueError, match="share shape and dtype"):
        Transition(**(valid | {"delta": np.array([[1.0]], dtype=np.float32)}))
    with pytest.raises(TypeError, match="numeric dtype"):
        Transition(**(valid | {"z_src": np.array(["bad"])}))
    with pytest.raises(ValueError, match="t must be >= 0"):
        Transition(**(valid | {"t": -1}))
    with pytest.raises(TypeError, match="episode_id must be a UUID"):
        Transition(**(valid | {"episode_id": str(uuid4())}))


def test_query_validation_rejects_bad_k_ef_metric_and_temporal_decay() -> None:
    with pytest.raises(QueryError, match="k must be >= 1"):
        QuerySpec(vector=_key(), k=0)
    with pytest.raises(QueryError, match="ef must be None"):
        QuerySpec(vector=_key(), k=3, ef=2)
    with pytest.raises(QueryError, match="metric must be a Metric"):
        QuerySpec(vector=_key(), k=1, metric="cosine")
    with pytest.raises(QueryError, match="temporal_decay must be >= 0"):
        QuerySpec(vector=_key(), k=1, temporal_decay=-0.1)


def test_memory_item_metadata_rejects_reserved_and_non_json_values() -> None:
    with pytest.raises(ValueError, match="reserved"):
        MemoryItem(
            content_id=None,
            key=_key(),
            value=_transition(),
            meta={"schema_version": "bad"},
            encoder_fp=_fingerprint(),
        )
    with pytest.raises(TypeError, match="JSON-compatible"):
        MemoryItem(
            content_id=None,
            key=_key(),
            value=_transition(),
            meta={"raw": b"bytes"},
            encoder_fp=_fingerprint(),
        )


def test_memory_item_rejects_malformed_content_ids() -> None:
    with pytest.raises(TypeError, match="content_id must be bytes"):
        MemoryItem(
            content_id="not-bytes",
            key=_key(),
            value=_transition(),
            meta={},
            encoder_fp=_fingerprint(),
        )
    with pytest.raises(ValueError, match="content_id must be 32 bytes"):
        MemoryItem(
            content_id=b"short",
            key=_key(),
            value=_transition(),
            meta={},
            encoder_fp=_fingerprint(),
        )


def test_retrieval_validation_rejects_length_mismatch_and_nonfinite_distance() -> None:
    item = _item()

    with pytest.raises(ValueError, match="matching lengths"):
        Retrieval(items=[item], distances=[])
    with pytest.raises(ValueError, match="finite"):
        Retrieval(items=[item], distances=[float("inf")])
