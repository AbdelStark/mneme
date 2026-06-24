from __future__ import annotations

import json
from collections.abc import Mapping
from math import log
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mneme.condition import CondCtx, KnnCorrector
from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QueryError,
    QuerySpec,
    Retrieval,
    Transition,
    ValidationError,
)
from mneme.eval import run_fixture_evaluation
from mneme.index import FlatIndex
from mneme.observability import (
    EVENT_SCHEMA_VERSION,
    REQUIRED_EVENT_NAMES,
    ObservabilityConfig,
    emit_event,
    has_event_sink,
)
from mneme.store import init_store, verify_store


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, event: Mapping[str, object]) -> None:
        self.events.append(dict(event))


def _vec(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _item(key_value: float, *, step: int = 1) -> MemoryItem:
    z_src = _vec([key_value, 0.0])
    z_next = _vec([key_value + 1.0, 0.0])
    return MemoryItem(
        content_id=None,
        key=_vec([key_value, 0.0]),
        value=Transition(
            z_src=z_src,
            action=_vec([0.1]),
            z_next=z_next,
            delta=z_next - z_src,
            t=step,
            episode_id=uuid4(),
        ),
        meta={"source": "fixture", "unsafe_note": "not emitted"},
        encoder_fp=_fingerprint(),
    )


def _retrieval() -> Retrieval:
    return Retrieval(
        items=(
            _item(1.0, step=1),
            _item(2.0, step=2),
        ),
        distances=(0.0, log(3.0)),
    )


def _strip_duration(event: Mapping[str, object]) -> dict[str, object]:
    stable = dict(event)
    stable.pop("duration_ms", None)
    return stable


def test_observability_config_and_required_event_names() -> None:
    assert "mneme.receipt.verify" in REQUIRED_EVENT_NAMES
    assert not has_event_sink(ObservabilityConfig())

    with pytest.raises(ValueError, match="content_id_prefix_bytes"):
        ObservabilityConfig(content_id_prefix_bytes=-1)


def test_index_search_event_snapshot_contains_required_fields() -> None:
    sink = RecordingSink()
    index = FlatIndex(ObservabilityConfig(event_sink=sink))
    index.add(b"a", _vec([1.0, 0.0]))
    index.add(b"b", _vec([2.0, 0.0]))

    assert index.search(_vec([1.0, 0.0]), 1, metric=Metric.L2) == [(b"a", 0.0)]

    assert _strip_duration(sink.events[-1]) == {
        "event": "mneme.index.search",
        "schema_version": EVENT_SCHEMA_VERSION,
        "operation": "index.search",
        "status": "ok",
        "backend": "flat",
        "metric": "l2",
        "k": 1,
        "ef": None,
        "index_size": 2,
        "hit_count": 1,
        "distance_min": 0.0,
        "distance_mean": 0.0,
    }


def test_structured_events_cover_store_condition_verify_and_eval(
    tmp_path: Path,
) -> None:
    sink = RecordingSink()
    observability = ObservabilityConfig(
        event_sink=sink,
        content_id_prefix_bytes=4,
    )
    root = tmp_path / "store"
    store = init_store(
        root, active_fingerprints=[_fingerprint()], observability=observability
    )
    cid = store.put(_item(1.0))
    retrieval = store.query(
        QuerySpec(
            _vec([1.0, 0.0]),
            k=1,
            metric=Metric.L2,
            encoder_fp=_fingerprint(),
        )
    )
    corrector = KnnCorrector(
        tau=1.0,
        lambda_max=1.0,
        alpha=0.0,
        mode="delta",
        observability=observability,
    )
    corrector.condition(
        _vec([10.0, 10.0]),
        _retrieval(),
        CondCtx(_vec([1.0, 1.0])),
    )
    verify_store(root, observability=observability)
    run_fixture_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
        observability=observability,
    )

    names = [event["event"] for event in sink.events]
    assert "mneme.store.put" in names
    assert "mneme.store.commit" in names
    assert "mneme.index.search" in names
    assert "mneme.store.query" in names
    assert "mneme.condition.apply" in names
    assert "mneme.store.verify" in names
    assert "mneme.eval.run" in names
    for event in sink.events:
        assert {"event", "schema_version", "operation", "duration_ms", "status"} <= set(
            event
        )
        json.dumps(event, sort_keys=True)

    put_event = next(
        event for event in sink.events if event["event"] == "mneme.store.put"
    )
    assert put_event["content_id_prefixes"] == [cid[:4].hex()]
    query_event = next(
        event for event in sink.events if event["event"] == "mneme.store.query"
    )
    assert query_event["hit_count"] == len(retrieval.items)
    assert query_event["fingerprint_match"] is True
    assert "unsafe_note" not in json.dumps(sink.events, sort_keys=True)


def test_event_redaction_removes_arrays_paths_secrets_and_unsafe_metadata() -> None:
    sink = RecordingSink()
    observability = ObservabilityConfig(event_sink=sink)

    emit_event(
        observability,
        event="mneme.test.redaction",
        operation="test.redaction",
        status="ok",
        started=None,
        latent=np.array([9.0, 8.0], dtype=np.float32),
        summary_vector=[1.0, 2.0],
        action=np.array([0.25], dtype=np.float32),
        observation={"pixels": [255, 0], "room": "private-lab"},
        store_path=Path("/Users/abdel/private/store"),
        secret_token="super-secret-token",
        dataset_id="private-dataset-name",
        metadata={
            "unsafe_note": "leak-me",
            "api_key": "secret-api-key",
            "safe_label": "fixture",
            "safe_path": "/Users/abdel/private/dataset",
        },
        raw_content_id=b"abcdef",
    )

    event = sink.events[-1]
    assert event["latent"] == {
        "redacted": "array",
        "shape": [2],
        "dtype": "float32",
    }
    assert event["summary_vector"] == {
        "redacted": "array",
        "shape": [2],
        "dtype": "sequence",
    }
    assert event["action"] == {
        "redacted": "array",
        "shape": [1],
        "dtype": "float32",
    }
    assert event["observation"] == "<redacted:observation>"
    assert event["store_path"] == "<redacted:path>"
    assert event["secret_token"] == "<redacted:secret>"
    assert event["dataset_id"] == "<redacted:dataset>"
    assert event["metadata"] == {
        "safe_label": "fixture",
        "safe_path": "<redacted:path>",
    }
    assert event["raw_content_id"] == {"redacted": "bytes", "length": 6}

    dumped = json.dumps(event, sort_keys=True)
    assert "super-secret-token" not in dumped
    assert "secret-api-key" not in dumped
    assert "leak-me" not in dumped
    assert "private-dataset-name" not in dumped
    assert "/Users/abdel" not in dumped
    assert "9.0" not in dumped


def test_content_id_prefixes_can_be_disabled(tmp_path: Path) -> None:
    sink = RecordingSink()
    observability = ObservabilityConfig(
        event_sink=sink,
        include_content_id_prefixes=False,
    )
    store = init_store(tmp_path / "store", observability=observability)

    cid = store.put(_item(1.0))
    store.query(QuerySpec(_vec([1.0, 0.0]), k=1, metric=Metric.L2))

    dumped = json.dumps(sink.events, sort_keys=True)
    assert cid[:6].hex() not in dumped
    put_event = next(
        event for event in sink.events if event["event"] == "mneme.store.put"
    )
    query_event = next(
        event for event in sink.events if event["event"] == "mneme.store.query"
    )
    assert put_event["content_id_prefixes"] == []
    assert query_event["content_id_prefixes"] == []


def test_fixture_eval_events_do_not_log_raw_arrays() -> None:
    sink = RecordingSink()
    run_fixture_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
        observability=ObservabilityConfig(event_sink=sink),
    )

    dumped = json.dumps(sink.events, sort_keys=True)
    assert '"z_src":' not in dumped
    assert '"z_next":' not in dumped
    assert '"delta":' not in dumped
    assert '"action":' not in dumped
    assert "10.0, 10.0" not in dumped
    assert "[1.0, 1.0]" not in dumped


def test_error_events_include_typed_error_fields() -> None:
    sink = RecordingSink()
    observability = ObservabilityConfig(event_sink=sink)
    index = FlatIndex(observability)
    index.add(b"a", _vec([1.0, 0.0]))

    with pytest.raises(QueryError, match="k must be >= 1"):
        index.search(_vec([1.0, 0.0]), 0, metric=Metric.L2)

    corrector = KnnCorrector(observability=observability)
    with pytest.raises(ValidationError, match="current_latent"):
        corrector.condition(_vec([1.0, 0.0]), _retrieval(), CondCtx(None))

    error_events = [event for event in sink.events if event["status"] == "error"]
    assert [event["error_type"] for event in error_events] == [
        "QueryError",
        "ValidationError",
    ]
