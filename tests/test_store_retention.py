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
    Transition,
)
from mneme.store import (
    age_retention,
    count_retention,
    init_store,
    open_store,
    rebuild_index,
    verify_store,
)


def test_count_retention_caps_visible_items_and_persists_tombstones(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = init_store(root, retention_policy=count_retention(2))
    cids = store.put_batch([_item(1.0, step=1), _item(2.0, step=2), _item(3.0, step=3)])

    stats = store.stats()
    retrieval = store.query(
        QuerySpec(np.array([1.0, 0.0], dtype=np.float32), k=3, metric=Metric.L2)
    )
    manifest_json = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert stats.value_record_count == 3
    assert stats.visible_record_count == 2
    assert stats.tombstone_count == 1
    assert stats.retention_policy == "count"
    assert [item.content_id for item in retrieval.items] == [cids[1], cids[2]]
    assert manifest_json["retention_policy"]["tombstones"] == [
        {
            "content_id": cids[0].hex(),
            "created_at": manifest_json["retention_policy"]["tombstones"][0][
                "created_at"
            ],
            "reason": "count",
            "transaction_id": store.manifest.last_completed_transaction,
        }
    ]

    reopened = open_store(root)
    reopened_retrieval = reopened.query(
        QuerySpec(np.array([1.0, 0.0], dtype=np.float32), k=3, metric=Metric.L2)
    )

    assert [item.content_id for item in reopened_retrieval.items] == [cids[1], cids[2]]
    assert verify_store(root).ok
    assert rebuild_index(root).item_count == 2
    assert verify_store(root).ok


def test_age_retention_excludes_expired_event_time_items(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root, retention_policy=age_retention(5))
    cids = store.put_batch(
        [_item(0.0, step=0), _item(6.0, step=6), _item(10.0, step=10)]
    )

    retrieval = store.query(
        QuerySpec(np.array([0.0, 0.0], dtype=np.float32), k=3, metric=Metric.L2)
    )
    tombstones = store.manifest.retention_policy["tombstones"]

    assert store.stats().visible_record_count == 2
    assert store.stats().tombstone_count == 1
    assert [item.content_id for item in retrieval.items] == [cids[1], cids[2]]
    assert tombstones[0]["content_id"] == cids[0].hex()
    assert tombstones[0]["reason"] == "age"


def test_retention_manifest_validation_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["retention_policy"] = {
        "policy": "count",
        "max_items": -1,
        "tombstones": [],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="max_items"):
        open_store(root)


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _item(key_value: float, *, step: int) -> MemoryItem:
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
