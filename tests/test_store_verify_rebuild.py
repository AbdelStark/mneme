from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from _cli_runner import run_cli

from mneme.core import (
    CliExitCode,
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    StoreCorruptionError,
    Transition,
)
from mneme.receipts import RetrievalReceipt, verify_retrieval_receipt
from mneme.store import (
    COMMIT_INIT_SCHEMA,
    INDEX_DATA_SCHEMA,
    commit_init_store,
    init_store,
    open_store,
    rebuild_index,
    verify_store,
)


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


def test_verify_store_succeeds_on_healthy_fixture(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))

    report = verify_store(root)

    assert report.ok
    assert report.item_count == 1
    assert report.value_log_count == 1
    assert report.index_backend == "flat"
    assert report.errors == ()

    cli = run_cli("store", "verify", root)
    assert cli.returncode == int(CliExitCode.SUCCESS)
    assert json.loads(cli.stdout)["ok"] is True


def test_verify_store_reports_corrupt_value_log_with_typed_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    log_path = root / "values" / "log-000000.mnv"
    payload = bytearray(log_path.read_bytes())
    payload[-1] ^= 0x01
    log_path.write_bytes(payload)

    report = verify_store(root)

    assert not report.ok
    assert any("checksum mismatch" in error for error in report.errors)
    with pytest.raises(StoreCorruptionError, match="checksum mismatch"):
        verify_store(root, raise_on_error=True)

    cli = run_cli("store", "verify", root)
    assert cli.returncode == int(CliExitCode.DATA_VALIDATION)
    assert json.loads(cli.stdout)["ok"] is False


def test_rebuild_index_restores_snapshot_and_query_results(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    cids = store.put_batch([_item(1.0, step=1), _item(3.0, step=2)])
    (root / "index" / "backend.json").unlink()

    before = verify_store(root)
    assert not before.ok
    assert any("index backend missing" in error for error in before.errors)

    rebuild = rebuild_index(root)

    assert rebuild.ok
    assert rebuild.item_count == 2
    snapshot = json.loads((root / "index" / "data.json").read_text())
    assert snapshot["schema_version"] == INDEX_DATA_SCHEMA
    assert snapshot["item_count"] == 2
    assert sorted(item["content_id"] for item in snapshot["items"]) == sorted(
        cid.hex() for cid in cids
    )

    assert verify_store(root).ok
    reopened = open_store(root)
    retrieval = reopened.query(
        QuerySpec(np.array([2.9, 0.0], dtype=np.float32), k=1, metric=Metric.L2)
    )
    assert retrieval.items[0].content_id == cids[1]


def test_verify_store_rejects_non_digest_index_content_ids(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    assert rebuild_index(root).ok
    data_path = root / "index" / "data.json"
    snapshot = json.loads(data_path.read_text(encoding="utf-8"))
    snapshot["items"][0]["content_id"] = "00"
    data_path.write_text(json.dumps(snapshot), encoding="utf-8")

    report = verify_store(root)

    assert not report.ok
    assert any(
        "index data item 0 content_id must be 32 bytes" in error
        for error in report.errors
    )


def test_index_rebuild_cli_returns_documented_success(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))

    cli = run_cli("index", "rebuild", root)

    assert cli.returncode == int(CliExitCode.SUCCESS)
    report = json.loads(cli.stdout)
    assert report["ok"] is True
    assert report["data_path"] == "index/data.json"
    assert (root / "index" / "data.json").is_file()


def test_commit_init_upgrades_healthy_v0_1_store_and_preserves_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    cids = store.put_batch([_item(1.0, step=1), _item(3.0, step=2)])
    assert not store.stats().commitments_enabled

    report = commit_init_store(root)
    reopened = open_store(root)
    spec = QuerySpec(
        np.array([3.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        with_receipt=True,
    )
    retrieval = reopened.query(spec)

    assert report.schema_version == COMMIT_INIT_SCHEMA
    assert report.ok
    assert report.item_count == 2
    assert report.root == reopened.root().hex()
    assert report.commitment_path == "receipts/commitment-mmr-v1.json"
    assert not report.already_initialized
    assert reopened.commitment_state().leaf_ids == tuple(cids)
    assert isinstance(retrieval.receipt, RetrievalReceipt)
    assert verify_retrieval_receipt(
        retrieval.receipt,
        retrieval.items,
        root=reopened.root(),
        query=spec,
    )


def test_commit_init_refuses_corrupt_value_log_without_writing_sidecar(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    log_path = root / "values" / "log-000000.mnv"
    payload = bytearray(log_path.read_bytes())
    payload[-1] ^= 0x01
    log_path.write_bytes(payload)

    report = commit_init_store(root)

    assert not report.ok
    assert report.root is None
    assert report.item_count == 0
    assert any("checksum mismatch" in error for error in report.errors)
    assert not (root / "receipts" / "commitment-mmr-v1.json").exists()
    manifest_json = json.loads((root / "manifest.json").read_text())
    assert manifest_json["commitment"] == {
        "enabled": False,
        "backend": None,
        "root": None,
        "files": [],
    }


def test_commit_init_cli_returns_upgrade_report(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put_batch([_item(1.0, step=1), _item(2.0, step=2)])

    cli = run_cli("store", "commit-init", root)

    assert cli.returncode == int(CliExitCode.SUCCESS), cli.stdout + cli.stderr
    report = json.loads(cli.stdout)
    assert report["schema_version"] == COMMIT_INIT_SCHEMA
    assert report["ok"] is True
    assert report["item_count"] == 2
    assert report["root"] == open_store(root).root().hex()
    assert report["commitment_path"] == "receipts/commitment-mmr-v1.json"


def test_cli_invalid_args_return_user_input_exit_code() -> None:
    cli = run_cli("store")

    assert cli.returncode == int(CliExitCode.USER_INPUT)
