from __future__ import annotations

import json
from collections.abc import Callable
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
    ValidationError,
)
from mneme.receipts import RetrievalReceipt, verify_retrieval_receipt
from mneme.store import (
    COMMIT_INIT_SCHEMA,
    INDEX_DATA_SCHEMA,
    INDEX_REBUILD_SCHEMA,
    STORE_VERIFICATION_SCHEMA,
    CommitInitReport,
    IndexRebuildReport,
    StoreVerificationReport,
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


class _BytesPath:
    def __fspath__(self) -> bytes:
        return b"store"


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


def test_verify_and_rebuild_report_unreadable_value_log(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    log_path = root / "values" / "log-000000.mnv"
    log_path.unlink()
    log_path.mkdir()

    report = verify_store(root)

    assert not report.ok
    assert any("value log could not be read" in error for error in report.errors)
    with pytest.raises(StoreCorruptionError, match="value log could not be read"):
        verify_store(root, raise_on_error=True)

    rebuild = rebuild_index(root)
    assert not rebuild.ok
    assert any("value log could not be read" in error for error in rebuild.errors)

    cli = run_cli("store", "verify", root)
    assert cli.returncode == int(CliExitCode.DATA_VALIDATION)
    payload = json.loads(cli.stdout)
    assert payload["ok"] is False
    assert any("value log could not be read" in error for error in payload["errors"])


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


@pytest.mark.parametrize("entrypoint", (verify_store, rebuild_index, commit_init_store))
def test_store_maintenance_entrypoints_reject_empty_paths_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: Callable[[str], object],
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="path must not be empty"):
        entrypoint("")

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "values").exists()
    assert not (tmp_path / "index").exists()
    assert not (tmp_path / "transactions").exists()
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("entrypoint", (verify_store, rebuild_index, commit_init_store))
def test_store_maintenance_entrypoints_reject_non_path_values(
    entrypoint: Callable[[object], object],
) -> None:
    with pytest.raises(ValidationError, match="path must be a path-like value"):
        entrypoint(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("entrypoint", (verify_store, rebuild_index, commit_init_store))
def test_store_maintenance_entrypoints_reject_bytes_pathlike_values(
    entrypoint: Callable[[object], object],
) -> None:
    with pytest.raises(ValidationError, match="path must resolve to a text path"):
        entrypoint(_BytesPath())  # type: ignore[arg-type]


def test_verify_store_rejects_nonstandard_index_backend_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    (root / "index" / "backend.json").write_text(
        '{"backend": NaN}',
        encoding="utf-8",
    )

    report = verify_store(root)

    assert not report.ok
    assert any("index backend is malformed JSON" in error for error in report.errors)


def test_verify_store_reports_unreadable_index_backend(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store(root)
    backend_path = root / "index" / "backend.json"
    backend_path.unlink()
    backend_path.mkdir()

    report = verify_store(root)

    assert not report.ok
    assert any("index backend could not be read" in error for error in report.errors)


def test_verify_store_rejects_nonstandard_index_data_json(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    assert rebuild_index(root).ok
    (root / "index" / "data.json").write_text(
        '{"schema_version": NaN}',
        encoding="utf-8",
    )

    report = verify_store(root)

    assert not report.ok
    assert any("index data is malformed JSON" in error for error in report.errors)


def test_verify_store_reports_unreadable_index_data(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    assert rebuild_index(root).ok
    data_path = root / "index" / "data.json"
    data_path.unlink()
    data_path.mkdir()

    report = verify_store(root)

    assert not report.ok
    assert any("index data could not be read" in error for error in report.errors)


def test_verify_store_rejects_overflowed_index_data_json(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    assert rebuild_index(root).ok
    content_id = "00" * 32
    (root / "index" / "data.json").write_text(
        f'{{"schema_version": "mneme.flat_index_snapshot.v1",'
        f'"backend": "flat",'
        f'"item_count": 1,'
        f'"items": [{{"content_id": "{content_id}", "key": [1e999]}}]}}',
        encoding="utf-8",
    )

    report = verify_store(root)

    assert not report.ok
    assert any("JSON number must be finite: 1e999" in error for error in report.errors)


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


@pytest.mark.parametrize("item_count", [True, 1.0])
def test_verify_store_rejects_malformed_index_item_count(
    tmp_path: Path,
    item_count: object,
) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    assert rebuild_index(root).ok
    data_path = root / "index" / "data.json"
    snapshot = json.loads(data_path.read_text(encoding="utf-8"))
    snapshot["item_count"] = item_count
    data_path.write_text(json.dumps(snapshot), encoding="utf-8")

    report = verify_store(root)

    assert not report.ok
    assert any(
        "index data item_count must be a non-negative integer" in error
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


def test_rebuild_index_reports_snapshot_write_failure(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    data_path = root / "index" / "data.json"
    data_path.mkdir()

    report = rebuild_index(root)

    assert not report.ok
    assert report.item_count == 0
    assert report.index_backend == "flat"
    assert any(
        "index rebuild could not write snapshot" in error for error in report.errors
    )

    cli = run_cli("index", "rebuild", root)
    assert cli.returncode == int(CliExitCode.DATA_VALIDATION)
    payload = json.loads(cli.stdout)
    assert payload["ok"] is False
    assert any(
        "index rebuild could not write snapshot" in error for error in payload["errors"]
    )


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


def test_commit_init_reports_sidecar_write_failure(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))
    receipts_path = root / "receipts"
    receipts_path.rmdir()
    receipts_path.write_text("occupied", encoding="utf-8")

    report = commit_init_store(root)

    assert not report.ok
    assert report.item_count == 1
    assert report.root is None
    assert any(
        "commitment state could not be written" in error for error in report.errors
    )
    manifest_json = json.loads((root / "manifest.json").read_text())
    assert manifest_json["commitment"] == {
        "enabled": False,
        "backend": None,
        "root": None,
        "files": [],
    }

    cli = run_cli("store", "commit-init", root)
    assert cli.returncode == int(CliExitCode.DATA_VALIDATION)
    payload = json.loads(cli.stdout)
    assert payload["ok"] is False
    assert any(
        "commitment state could not be written" in error for error in payload["errors"]
    )


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


def test_store_report_constructors_normalize_sequences_and_roots() -> None:
    verification = StoreVerificationReport(
        ok=False,
        store_id=None,
        item_count=0,
        value_log_count=0,
        index_backend=None,
        errors=["manifest missing"],
    )
    rebuild = IndexRebuildReport(
        ok=True,
        item_count=0,
        index_backend="flat",
        data_path="index/data.json",
        errors=[],
    )
    commit_init = CommitInitReport(
        ok=True,
        store_id="store",
        item_count=1,
        root=("AA" * 32),
        commitment_path="receipts/commitment-mmr-v1.json",
        already_initialized=False,
        errors=[],
    )

    assert verification.errors == ("manifest missing",)
    assert rebuild.errors == ()
    assert commit_init.root == "aa" * 32
    assert commit_init.errors == ()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        (
            {"schema_version": "mneme.store_verification.v2"},
            "unsupported store verification",
        ),
        ({"ok": "yes"}, "ok must be a bool"),
        ({"store_id": ""}, "store_id must be a non-empty string"),
        ({"item_count": -1}, "item_count must be a non-negative integer"),
        (
            {"value_log_count": True},
            "value_log_count must be a non-negative integer",
        ),
        ({"index_backend": object()}, "index_backend must be a non-empty string"),
        ({"errors": "bad"}, "errors must be a sequence"),
        ({"errors": ("",)}, "errors item must be a non-empty string"),
    ),
)
def test_store_verification_report_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = _store_verification_report_values()
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        StoreVerificationReport(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"schema_version": "mneme.index_rebuild.v2"}, "unsupported index rebuild"),
        ({"ok": 1}, "ok must be a bool"),
        ({"item_count": -1}, "item_count must be a non-negative integer"),
        ({"index_backend": ""}, "index_backend must be a non-empty string"),
        ({"data_path": object()}, "data_path must be a non-empty string"),
        ({"errors": object()}, "errors must be a sequence"),
        ({"errors": ("",)}, "errors item must be a non-empty string"),
    ),
)
def test_index_rebuild_report_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = _index_rebuild_report_values()
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        IndexRebuildReport(**values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"schema_version": "mneme.store_commit_init.v2"}, "unsupported commit init"),
        ({"ok": "yes"}, "ok must be a bool"),
        ({"store_id": ""}, "store_id must be a non-empty string"),
        ({"item_count": -1}, "item_count must be a non-negative integer"),
        ({"root": "00"}, "root must be 32 bytes"),
        ({"root": object()}, "root must be a non-empty string"),
        ({"commitment_path": ""}, "commitment_path must be a non-empty string"),
        ({"already_initialized": 0}, "already_initialized must be a bool"),
        ({"errors": object()}, "errors must be a sequence"),
        ({"errors": ("",)}, "errors item must be a non-empty string"),
    ),
)
def test_commit_init_report_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = _commit_init_report_values()
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        CommitInitReport(**values)


def _store_verification_report_values() -> dict[str, object]:
    return {
        "ok": True,
        "store_id": "store",
        "item_count": 1,
        "value_log_count": 1,
        "index_backend": "flat",
        "errors": (),
        "schema_version": STORE_VERIFICATION_SCHEMA,
    }


def _index_rebuild_report_values() -> dict[str, object]:
    return {
        "ok": True,
        "item_count": 1,
        "index_backend": "flat",
        "data_path": "index/data.json",
        "errors": (),
        "schema_version": INDEX_REBUILD_SCHEMA,
    }


def _commit_init_report_values() -> dict[str, object]:
    return {
        "ok": True,
        "store_id": "store",
        "item_count": 1,
        "root": "00" * 32,
        "commitment_path": "receipts/commitment-mmr-v1.json",
        "already_initialized": False,
        "errors": (),
        "schema_version": COMMIT_INIT_SCHEMA,
    }
