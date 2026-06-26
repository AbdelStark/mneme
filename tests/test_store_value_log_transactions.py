from __future__ import annotations

import base64
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from blake3 import blake3

from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    StoreCorruptionError,
    Transition,
    ValidationError,
    content_id,
)
from mneme.store import (
    STORE_MANIFEST_SCHEMA,
    StoreRecoveryEvent,
    StoreStats,
    init_store,
    open_store,
    rebuild_index,
)
from mneme.store._value_log import VALUE_RECORD_SCHEMA, append_value_record


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _item(key_value: float, *, step: int = 0) -> MemoryItem:
    z_src = np.array([key_value, 0.0], dtype=np.float32)
    z_next = np.array([key_value + 1.0, 0.0], dtype=np.float32)
    return MemoryItem(
        content_id=None,
        key=np.array([key_value, 0.0], dtype=np.float32),
        value=Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_next,
            delta=z_next - z_src,
            t=step,
            episode_id=uuid4(),
        ),
        meta={"source": "fixture", "step": step},
        encoder_fp=_fingerprint(),
    )


def test_put_returns_content_id_and_updates_manifest_and_transaction(
    tmp_path: Path,
) -> None:
    store = init_store(tmp_path / "store")
    item = _item(1.0)

    cid = store.put(item)

    assert cid == content_id(item)
    assert store.stats().value_record_count == 1
    assert store.stats().value_bytes > 0
    assert store.manifest.active_fingerprints == (_fingerprint(),)
    assert store.manifest.last_completed_transaction is not None

    transaction_path = (
        store.path
        / "transactions"
        / f"{store.manifest.last_completed_transaction}.json"
    )
    transaction = json.loads(transaction_path.read_text())
    assert transaction["state"] == "committed"
    assert transaction["item_count"] == 1
    assert (
        transaction["written_offsets"][0]["end"]
        > transaction["written_offsets"][0]["start"]
    )


def test_restart_after_committed_write_preserves_queryability(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    cid = store.put(_item(1.0))

    reopened = open_store(root)
    retrieval = reopened.query(
        QuerySpec(
            vector=np.array([1.0, 0.0], dtype=np.float32),
            k=1,
            metric=Metric.L2,
            encoder_fp=_fingerprint(),
        )
    )

    assert retrieval.items[0].content_id == cid
    assert retrieval.distances == (0.0,)
    assert reopened.stats().value_record_count == 1


def test_put_batch_appends_multiple_records_and_queries_nearest(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    cids = store.put_batch([_item(1.0, step=1), _item(3.0, step=2)])

    reopened = open_store(tmp_path / "store")
    retrieval = reopened.query(
        QuerySpec(np.array([2.9, 0.0], dtype=np.float32), k=1, metric=Metric.L2)
    )

    assert len(cids) == 2
    assert reopened.stats().value_record_count == 2
    assert retrieval.items[0].content_id == cids[1]


def test_interrupted_transaction_before_value_append_rolls_back(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    transaction_path = root / "transactions" / "txn-interrupted.json"
    transaction_path.write_text(
        json.dumps(
            {
                "schema_version": "mneme.transaction.v1",
                "transaction_id": "txn-interrupted",
                "state": "intent",
                "item_count": 1,
                "value_log": "values/log-000000.mnv",
                "previous_size_bytes": 0,
                "previous_record_count": 0,
            }
        ),
        encoding="utf-8",
    )

    recovered = open_store(root)
    transaction = json.loads(transaction_path.read_text())
    event = recovered.recovery_events[0].to_json()
    duration_ms = event.pop("duration_ms")

    assert recovered.stats().value_record_count == 0
    assert transaction["state"] == "rolled_back"
    assert duration_ms >= 0.0
    assert event == {
        "event": "mneme.store.recover",
        "schema_version": "mneme.store_recovery_event.v1",
        "store_id": str(recovered.manifest.store_id),
        "operation": "store.recover",
        "status": "rolled_back",
        "transaction_id": "txn-interrupted",
        "value_log": "values/log-000000.mnv",
        "previous_size_bytes": 0,
        "recovered_size_bytes": 0,
        "previous_record_count": 0,
        "recovered_record_count": 0,
    }
    assert open_store(root).recovery_events == ()


def test_interrupted_transaction_after_complete_append_is_completed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    item = _item(1.0)
    transaction_path = root / "transactions" / "txn-appended.json"
    transaction_path.write_text(
        json.dumps(
            {
                "schema_version": "mneme.transaction.v1",
                "transaction_id": "txn-appended",
                "state": "intent",
                "item_count": 1,
                "value_log": "values/log-000000.mnv",
                "previous_size_bytes": 0,
                "previous_record_count": 0,
            }
        ),
        encoding="utf-8",
    )
    append_value_record(root / "values" / "log-000000.mnv", item)

    recovered = open_store(root)
    transaction = json.loads(transaction_path.read_text())
    retrieval = recovered.query(
        QuerySpec(np.array([1.0, 0.0], dtype=np.float32), k=1, metric=Metric.L2)
    )

    assert recovered.stats().value_record_count == 1
    assert recovered.manifest.last_completed_transaction == "txn-appended"
    assert rebuild_index(root).ok
    assert transaction["state"] == "committed"
    assert transaction["written_offsets"][0]["start"] == 0
    assert retrieval.items[0].content_id == content_id(item)
    assert recovered.recovery_events[0].status == "completed"
    assert open_store(root).recovery_events == ()


def test_malformed_transaction_file_raises_store_corruption(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store(root)
    (root / "transactions" / "txn-bad.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="malformed JSON"):
        open_store(root)


def test_unreadable_transaction_file_raises_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    (root / "transactions" / "txn-bad.json").mkdir()

    with pytest.raises(
        StoreCorruptionError, match="transaction file could not be read"
    ):
        open_store(root)


def test_value_log_checksum_corruption_is_detected_on_open(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    log_path = root / "values" / "log-000000.mnv"
    payload = bytearray(log_path.read_bytes())
    payload[-1] ^= 0x01
    log_path.write_bytes(payload)

    with pytest.raises(StoreCorruptionError, match="checksum mismatch"):
        open_store(root)


def test_unreadable_value_log_path_raises_store_corruption(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    log_path = root / "values" / "log-000000.mnv"
    log_path.unlink()
    log_path.mkdir()

    with pytest.raises(StoreCorruptionError, match="value log could not be read"):
        open_store(root)


def test_value_log_invalid_content_id_hex_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    _write_raw_value_record(
        root / "values" / "log-000000.mnv",
        _value_record(content_id="not hex"),
    )

    with pytest.raises(StoreCorruptionError, match="content_id must be hex bytes"):
        open_store(root)


def test_value_log_short_content_id_is_store_corruption(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store(root)
    _write_raw_value_record(
        root / "values" / "log-000000.mnv",
        _value_record(content_id="00"),
    )

    with pytest.raises(StoreCorruptionError, match="content_id must be 32 bytes"):
        open_store(root)


def test_store_stats_constructor_normalizes_path() -> None:
    stats = StoreStats(
        store_id=uuid4(),
        path="store",
        schema_version=STORE_MANIFEST_SCHEMA,
        active_fingerprint_count=1,
        value_log_count=1,
        value_record_count=2,
        visible_record_count=2,
        value_bytes=128,
        index_backend="flat",
        retention_policy="none",
        tombstone_count=0,
        last_completed_transaction=None,
        commitments_enabled=False,
    )

    assert stats.path == Path("store")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"store_id": "store"}, "store_id must be a UUID"),
        ({"path": object()}, "path must be a path-like value"),
        ({"path": ""}, "path must not be empty"),
        ({"schema_version": ""}, "schema_version must be a non-empty string"),
        (
            {"active_fingerprint_count": -1},
            "active_fingerprint_count must be a non-negative integer",
        ),
        ({"value_log_count": True}, "value_log_count must be a non-negative integer"),
        (
            {"value_record_count": -1},
            "value_record_count must be a non-negative integer",
        ),
        (
            {"visible_record_count": -1},
            "visible_record_count must be a non-negative integer",
        ),
        ({"value_bytes": -1}, "value_bytes must be a non-negative integer"),
        ({"index_backend": ""}, "index_backend must be a non-empty string"),
        ({"retention_policy": object()}, "retention_policy must be"),
        ({"tombstone_count": -1}, "tombstone_count must be a non-negative integer"),
        (
            {"last_completed_transaction": ""},
            "last_completed_transaction must be a non-empty string",
        ),
        ({"commitments_enabled": 1}, "commitments_enabled must be a bool"),
    ),
)
def test_store_stats_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = _store_stats_values()
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        StoreStats(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"schema_version": "mneme.store_recovery_event.v2"}, "unsupported store"),
        ({"event": ""}, "event must be a non-empty string"),
        ({"store_id": object()}, "store_id must be a non-empty string"),
        ({"operation": ""}, "operation must be a non-empty string"),
        ({"status": ""}, "status must be a non-empty string"),
        ({"transaction_id": ""}, "transaction_id must be a non-empty string"),
        ({"value_log": object()}, "value_log must be a non-empty string"),
        (
            {"previous_size_bytes": -1},
            "previous_size_bytes must be a non-negative integer",
        ),
        (
            {"recovered_size_bytes": True},
            "recovered_size_bytes must be a non-negative integer",
        ),
        (
            {"previous_record_count": -1},
            "previous_record_count must be a non-negative integer",
        ),
        (
            {"recovered_record_count": -1},
            "recovered_record_count must be a non-negative integer",
        ),
        ({"duration_ms": float("nan")}, "duration_ms must be"),
        ({"duration_ms": -0.1}, "duration_ms must be"),
    ),
)
def test_store_recovery_event_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = _recovery_event_values()
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        StoreRecoveryEvent(**values)


def _store_stats_values() -> dict[str, object]:
    return {
        "store_id": uuid4(),
        "path": Path("store"),
        "schema_version": STORE_MANIFEST_SCHEMA,
        "active_fingerprint_count": 1,
        "value_log_count": 1,
        "value_record_count": 1,
        "visible_record_count": 1,
        "value_bytes": 128,
        "index_backend": "flat",
        "retention_policy": "none",
        "tombstone_count": 0,
        "last_completed_transaction": "txn-fixture",
        "commitments_enabled": False,
    }


def _recovery_event_values() -> dict[str, object]:
    return {
        "event": "mneme.store.recover",
        "store_id": str(uuid4()),
        "operation": "store.recover",
        "status": "completed",
        "transaction_id": "txn-fixture",
        "value_log": "values/log-000000.mnv",
        "previous_size_bytes": 0,
        "recovered_size_bytes": 0,
        "previous_record_count": 0,
        "recovered_record_count": 0,
        "duration_ms": 0.0,
    }


def test_value_log_non_object_record_is_store_corruption(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store(root)
    _write_raw_value_record(root / "values" / "log-000000.mnv", ["not", "object"])

    with pytest.raises(StoreCorruptionError, match="value record must be an object"):
        open_store(root)


def test_value_log_invalid_base64_array_data_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["key"]["data"] = "not base64!"
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(StoreCorruptionError, match="array data must be base64"):
        open_store(root)


def test_value_log_boolean_array_shape_is_store_corruption(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["key"]["shape"] = [True, 2]
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(StoreCorruptionError, match="array shape"):
        open_store(root)


@pytest.mark.parametrize("shape", ([1], [3]))
def test_value_log_array_byte_count_mismatch_is_store_corruption(
    tmp_path: Path,
    shape: list[int],
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["key"]["shape"] = shape
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(
        StoreCorruptionError, match="array data does not match dtype and shape"
    ):
        open_store(root)


def test_value_log_nonfinite_array_payload_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["key"] = _array_payload(
        np.array([float("nan"), 0.0], dtype=np.float32)
    )
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(
        StoreCorruptionError, match="array must contain only finite values"
    ):
        open_store(root)


def test_value_log_nonstandard_reward_constant_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["value"]["reward"] = float("inf")
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(
        StoreCorruptionError,
        match="value log record is not valid JSON",
    ):
        open_store(root)


def test_value_log_invalid_memory_item_payload_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["key"] = _array_payload(np.array([[1.0, 0.0]], dtype=np.float32))
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(StoreCorruptionError, match="invalid memory item payload"):
        open_store(root)


def test_value_log_unsupported_memory_item_schema_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["schema_version"] = "mneme.memory_item.v2"
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(StoreCorruptionError, match="invalid memory item payload"):
        open_store(root)


def test_value_log_unsupported_transition_schema_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["value"]["schema_version"] = "mneme.transition.v2"
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(StoreCorruptionError, match="invalid transition payload"):
        open_store(root)


def test_value_log_unsupported_encoder_fingerprint_schema_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["encoder_fp"]["schema_version"] = "mneme.encoder_fingerprint.v2"
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(StoreCorruptionError, match="invalid encoder fingerprint"):
        open_store(root)


def test_value_log_unsupported_value_kind_is_store_corruption(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    record = _value_record(content_id="00" * 32)
    record["item"]["value_kind"] = "frame"
    _write_raw_value_record(root / "values" / "log-000000.mnv", record)

    with pytest.raises(
        StoreCorruptionError, match="unsupported value record value_kind"
    ):
        open_store(root)


def _write_raw_value_record(path: Path, record: object) -> None:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header = len(payload).to_bytes(8, "big") + blake3(payload).digest(length=32)
    path.write_bytes(header + payload)


def _value_record(*, content_id: str) -> dict[str, object]:
    return {
        "schema_version": VALUE_RECORD_SCHEMA,
        "content_id": content_id,
        "item": {
            "schema_version": "mneme.memory_item.v1",
            "key": _array_payload(np.array([1.0, 0.0], dtype=np.float32)),
            "value_kind": "transition",
            "value": {
                "schema_version": "mneme.transition.v1",
                "z_src": _array_payload(np.array([1.0, 0.0], dtype=np.float32)),
                "action": _array_payload(np.array([0.1], dtype=np.float32)),
                "z_next": _array_payload(np.array([2.0, 0.0], dtype=np.float32)),
                "delta": _array_payload(np.array([1.0, 0.0], dtype=np.float32)),
                "t": 1,
                "episode_id": str(uuid4()),
                "reward": None,
            },
            "meta": {},
            "encoder_fp": {
                "schema_version": "mneme.encoder_fingerprint.v1",
                "encoder_id": "encoder.fixture",
                "summarizer_id": "meanpool-v1",
                "weights_digest": None,
                "config_digest": "blake3:config",
            },
        },
    }


def _array_payload(array: np.ndarray) -> dict[str, object]:
    canonical = np.ascontiguousarray(array)
    return {
        "dtype": str(canonical.dtype),
        "shape": list(canonical.shape),
        "data": base64.b64encode(canonical.tobytes(order="C")).decode("ascii"),
    }
