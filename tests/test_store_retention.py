from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mneme.core import (
    Cid,
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    StoreCorruptionError,
    Transition,
    build_item,
)
from mneme.store import (
    age_retention,
    count_retention,
    init_store,
    open_store,
    rebuild_index,
    verify_store,
)
from mneme.store._retention import apply_retention_policy


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


def test_retention_manifest_rejects_non_digest_tombstones(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["retention_policy"] = {
        "policy": "none",
        "tombstones": [
            {
                "content_id": "00",
                "reason": "count",
                "created_at": "2026-06-24T00:00:00Z",
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="tombstone content_id must be 32"):
        open_store(root)


def test_count_retention_helper_uses_deterministic_tie_breaks() -> None:
    items = dict(
        _prepared_item_pairs(
            [
                _item(1.0, step=1),
                _item(2.0, step=2),
                _item(3.0, step=2),
            ]
        )
    )
    ordered = sorted(
        items.items(),
        key=lambda entry: (entry[1].value.t, entry[0]),
        reverse=True,
    )

    policy = apply_retention_policy(
        {"policy": "count", "max_items": 2, "tombstones": []},
        items,
        transaction_id="txn-retention",
    )

    assert [stone["content_id"] for stone in policy["tombstones"]] == [
        ordered[2][0].hex()
    ]
    assert policy["tombstones"][0]["reason"] == "count"
    assert policy["tombstones"][0]["transaction_id"] == "txn-retention"


def test_retention_helper_rejects_unvalidated_policy_numbers() -> None:
    item = _prepared_item(_item(1.0, step=1))

    with pytest.raises(StoreCorruptionError, match="policy is unsupported"):
        apply_retention_policy(
            {"policy": "density", "tombstones": []},
            {_prepared_cid(item): item},
            transaction_id="txn-invalid",
        )
    with pytest.raises(StoreCorruptionError, match="policy is unsupported"):
        apply_retention_policy(
            {"policy": False, "tombstones": []},
            {_prepared_cid(item): item},
            transaction_id="txn-invalid",
        )
    with pytest.raises(StoreCorruptionError, match="max_items"):
        apply_retention_policy(
            {"policy": "count", "max_items": True, "tombstones": []},
            {_prepared_cid(item): item},
            transaction_id="txn-invalid",
        )
    with pytest.raises(StoreCorruptionError, match="tombstone content_id must be 32"):
        apply_retention_policy(
            {"policy": "none", "tombstones": [{"content_id": "00"}]},
            {_prepared_cid(item): item},
            transaction_id="txn-invalid",
        )


def test_retention_helper_defaults_missing_policy_to_none() -> None:
    item = _prepared_item(_item(1.0, step=1))

    policy = apply_retention_policy(
        {"tombstones": []},
        {_prepared_cid(item): item},
        transaction_id="txn-default",
    )

    assert policy == {"policy": "none", "tombstones": []}


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


def _prepared_item(item: MemoryItem) -> MemoryItem:
    return build_item(item.value, item.key, item.encoder_fp, item.meta)


def _prepared_item_pairs(items: list[MemoryItem]) -> list[tuple[bytes, MemoryItem]]:
    prepared = [_prepared_item(item) for item in items]
    return [(_prepared_cid(item), item) for item in prepared]


def _prepared_cid(item: MemoryItem) -> Cid:
    if item.content_id is None:
        raise AssertionError("prepared test item must have a content id")
    return item.content_id
