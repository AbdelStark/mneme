from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mneme.core import (
    CliExitCode,
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    StoreCorruptionError,
    Transition,
)
from mneme.store import (
    INDEX_DATA_SCHEMA,
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

    cli = _run_cli("store", "verify", root)
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

    cli = _run_cli("store", "verify", root)
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


def test_index_rebuild_cli_returns_documented_success(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))

    cli = _run_cli("index", "rebuild", root)

    assert cli.returncode == int(CliExitCode.SUCCESS)
    report = json.loads(cli.stdout)
    assert report["ok"] is True
    assert report["data_path"] == "index/data.json"
    assert (root / "index" / "data.json").is_file()


def test_cli_invalid_args_return_user_input_exit_code() -> None:
    cli = _run_cli("store")

    assert cli.returncode == int(CliExitCode.USER_INPUT)


def _run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mneme.cli", *(str(arg) for arg in args)],
        check=False,
        text=True,
        capture_output=True,
    )
