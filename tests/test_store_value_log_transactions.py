from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    StoreCorruptionError,
    TransactionError,
    Transition,
    content_id,
)
from mneme.store import init_store, open_store


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


def test_interrupted_transaction_intent_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store(root)
    (root / "transactions" / "txn-interrupted.json").write_text(
        json.dumps(
            {
                "schema_version": "mneme.transaction.v1",
                "transaction_id": "txn-interrupted",
                "state": "intent",
                "item_count": 1,
                "value_log": "values/log-000000.mnv",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(TransactionError, match="interrupted transaction"):
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
