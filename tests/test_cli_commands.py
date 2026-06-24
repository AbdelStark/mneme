from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

from mneme.core import (
    CliExitCode,
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    Transition,
)
from mneme.eval import validate_report_json
from mneme.store import init_store


def test_store_init_and_stats_cli_emit_schema_versioned_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"

    init = _run_cli("store", "init", root)
    stats = _run_cli("store", "stats", root, "--json")

    assert init.returncode == int(CliExitCode.SUCCESS), init.stdout + init.stderr
    assert stats.returncode == int(CliExitCode.SUCCESS), stats.stdout + stats.stderr
    init_json = _stdout_json(init)
    stats_json = _stdout_json(stats)
    assert init_json["ok"] is True
    assert init_json["schema_version"] == "mneme.store_stats.v1"
    assert stats_json["schema_version"] == "mneme.store_stats.v1"
    assert stats_json["store_id"] == init_json["store_id"]
    assert stats_json["value_record_count"] == 0
    assert stats_json["visible_record_count"] == 0
    assert stats_json["retention_policy"] == "none"
    assert stats_json["tombstone_count"] == 0
    assert "path" not in stats_json
    assert str(root) not in stats.stdout


def test_store_init_cli_reports_typed_duplicate_store_error(tmp_path: Path) -> None:
    root = tmp_path / "store"
    assert _run_cli("store", "init", root).returncode == int(CliExitCode.SUCCESS)

    duplicate = _run_cli("store", "init", root)

    assert duplicate.returncode == int(CliExitCode.INTERNAL)
    error = _stdout_json(duplicate)
    assert error["schema_version"] == "mneme.cli_error.v1"
    assert error["ok"] is False
    assert error["error_type"] == "StoreError"


def test_store_verify_cli_success_and_validation_failure(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))

    ok = _run_cli("store", "verify", root)
    missing = _run_cli("store", "verify", tmp_path / "missing")

    assert ok.returncode == int(CliExitCode.SUCCESS), ok.stdout + ok.stderr
    assert _stdout_json(ok)["schema_version"] == "mneme.store_verification.v1"
    assert missing.returncode == int(CliExitCode.DATA_VALIDATION)
    missing_json = _stdout_json(missing)
    assert missing_json["ok"] is False
    assert missing_json["errors"]


def test_index_rebuild_cli_success_and_validation_failure(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put(_item(1.0))

    ok = _run_cli("index", "rebuild", root)
    missing = _run_cli("index", "rebuild", tmp_path / "missing")

    assert ok.returncode == int(CliExitCode.SUCCESS), ok.stdout + ok.stderr
    ok_json = _stdout_json(ok)
    assert ok_json["schema_version"] == "mneme.index_rebuild.v1"
    assert ok_json["ok"] is True
    assert missing.returncode == int(CliExitCode.DATA_VALIDATION)
    missing_json = _stdout_json(missing)
    assert missing_json["ok"] is False
    assert missing_json["errors"]


def test_query_cli_emits_result_without_raw_memory_fields(tmp_path: Path) -> None:
    root = tmp_path / "store"
    vector_path = tmp_path / "query.json"
    store = init_store(root)
    cid = store.put(_item(1.0))
    vector_path.write_text(json.dumps({"vector": [1.0, 0.0]}), encoding="utf-8")

    result = _run_cli(
        "query",
        root,
        "--vector",
        vector_path,
        "--k",
        "1",
        "--metric",
        "l2",
        "--json",
    )

    assert result.returncode == int(CliExitCode.SUCCESS), result.stdout + result.stderr
    report = _stdout_json(result)
    assert report == {
        "schema_version": "mneme.query_result.v1",
        "ok": True,
        "item_count": 1,
        "content_id_prefixes": [cid[:6].hex()],
        "distances": [0.0],
        "receipt": None,
    }
    assert "z_src" not in result.stdout
    assert "action" not in result.stdout
    assert "z_next" not in result.stdout
    assert "metadata" not in result.stdout


def test_query_cli_reports_typed_query_error(tmp_path: Path) -> None:
    root = tmp_path / "store"
    vector_path = tmp_path / "query.json"
    init_store(root)
    vector_path.write_text("[1.0, 0.0]", encoding="utf-8")

    result = _run_cli(
        "query",
        root,
        "--vector",
        vector_path,
        "--k",
        "0",
        "--metric",
        "l2",
        "--json",
    )

    assert result.returncode == int(CliExitCode.USER_INPUT)
    error = _stdout_json(result)
    assert error["schema_version"] == "mneme.cli_error.v1"
    assert error["error_type"] == "QueryError"


def test_eval_fixtures_cli_writes_and_prints_valid_report(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "fixtures.json"

    result = _run_cli("eval", "fixtures", "--out", output, "--seed", "42")

    assert result.returncode == int(CliExitCode.SUCCESS), result.stdout + result.stderr
    printed = validate_report_json(_stdout_json(result))
    written = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert printed == written
    assert printed.schema_version == "mneme.eval_report.v1"
    assert printed.seed == 42


def test_eval_fixtures_cli_reports_typed_write_error(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")

    result = _run_cli("eval", "fixtures", "--out", blocked_parent / "report.json")

    assert result.returncode == int(CliExitCode.INTERNAL)
    error = _stdout_json(result)
    assert error["schema_version"] == "mneme.cli_error.v1"
    assert error["error_type"] == "EvaluationError"


def test_eval_profile_cli_writes_and_prints_valid_report(tmp_path: Path) -> None:
    root = tmp_path / "store"
    output = tmp_path / "reports" / "profile.json"
    store = init_store(root)
    store.put_batch([_item(float(index), step=index) for index in range(3)])

    result = _run_cli(
        "eval",
        "profile",
        "--store",
        root,
        "--out",
        output,
        "--k",
        "2",
        "--metric",
        "l2",
        "--queries",
        "2",
        "--warmup",
        "0",
        "--measurements",
        "2",
        "--approx-backend",
        "none",
    )

    assert result.returncode == int(CliExitCode.SUCCESS), result.stdout + result.stderr
    printed = validate_report_json(_stdout_json(result))
    written = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert printed == written
    assert printed.metrics["flat_recall_at_k"] == 1.0
    assert printed.metrics["query_latency_p50_ms"] >= 0.0
    assert printed.metrics["conditioning_latency_p50_ms"] >= 0.0
    assert printed.metrics["memory_footprint_bytes_per_item"] > 0.0


def test_eval_recall_and_latency_aliases_emit_profile_reports(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = init_store(root)
    store.put_batch([_item(float(index), step=index) for index in range(3)])

    for command in ("recall", "latency"):
        output = tmp_path / "reports" / f"{command}.json"
        result = _run_cli(
            "eval",
            command,
            "--store",
            root,
            "--out",
            output,
            "--k",
            "2",
            "--metric",
            "l2",
            "--queries",
            "2",
            "--warmup",
            "0",
            "--measurements",
            "1",
            "--approx-backend",
            "none",
        )

        assert result.returncode == int(CliExitCode.SUCCESS), (
            result.stdout + result.stderr
        )
        report = validate_report_json(_stdout_json(result))
        assert report.command[:3] == ("mneme", "eval", command)
        assert report.metrics["flat_recall_at_k"] == 1.0
        assert report.metrics["query_latency_p50_ms"] >= 0.0


def test_eval_receipts_cli_writes_and_prints_valid_report(tmp_path: Path) -> None:
    root = tmp_path / "store"
    output = tmp_path / "reports" / "receipts.json"
    store = init_store(root)
    store.put_batch([_item(float(index), step=index) for index in range(4)])
    store.commit()

    result = _run_cli(
        "eval",
        "receipts",
        "--store",
        root,
        "--out",
        output,
        "--k",
        "2",
        "--metric",
        "l2",
        "--queries",
        "2",
        "--warmup",
        "0",
        "--measurements",
        "1",
    )

    assert result.returncode == int(CliExitCode.SUCCESS), result.stdout + result.stderr
    printed = validate_report_json(_stdout_json(result))
    written = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert printed == written
    assert printed.artifacts["report_kind"] == "receipt-overhead"
    assert printed.metrics["receipt_proof_count_mean"] == 2.0


def test_eval_remote_conformance_cli_writes_and_prints_valid_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reports" / "remote-conformance.json"

    result = _run_cli(
        "eval",
        "remote-conformance",
        "--out",
        output,
        "--seed",
        "9",
    )

    assert result.returncode == int(CliExitCode.SUCCESS), result.stdout + result.stderr
    printed = validate_report_json(_stdout_json(result))
    written = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert printed == written
    assert printed.seed == 9
    assert printed.artifacts["report_kind"] == "remote-conformance"
    assert printed.artifacts["transport"] == "http-json-asgi"
    assert printed.metrics["scenario_count"] == printed.metrics["passed_scenario_count"]
    assert printed.metrics["error_case_count"] == 2


def test_eval_benchmark_dry_run_cli_writes_valid_external_report(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    output = tmp_path / "reports" / "benchmark.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "mneme.dataset_ref.v1",
                "dataset_id": "loopnav-dry-run",
                "kind": "external",
                "split": "dry-run",
                "version": "v0",
                "uri": "https://example.invalid/loopnav",
                "metadata": {"dry_run": True},
            }
        ),
        encoding="utf-8",
    )

    result = _run_cli(
        "eval",
        "benchmark",
        "--dry-run",
        "--dataset",
        dataset_path,
        "--out",
        output,
        "--checkpoint",
        "checkpoints/base.json",
        "--modes",
        "no_memory,corrector,in_context,adapter",
        "--seed",
        "5",
    )

    assert result.returncode == int(CliExitCode.SUCCESS), result.stdout + result.stderr
    printed = validate_report_json(_stdout_json(result))
    written = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert printed == written
    assert printed.dataset.kind == "external"
    assert printed.dataset.split == "dry-run"
    assert printed.seed == 5
    assert printed.artifacts["report_kind"] == "external-benchmark-dry-run"
    assert printed.metrics["adapter_status"] == "dry_run"
    assert printed.caveats


def test_eval_benchmark_cli_reports_missing_dataset(tmp_path: Path) -> None:
    result = _run_cli(
        "eval",
        "benchmark",
        "--dry-run",
        "--dataset",
        tmp_path / "missing.json",
        "--out",
        tmp_path / "benchmark.json",
        "--checkpoint",
        "checkpoints/base.json",
    )

    assert result.returncode == int(CliExitCode.INTERNAL)
    error = _stdout_json(result)
    assert error["schema_version"] == "mneme.cli_error.v1"
    assert error["error_type"] == "EvaluationError"
    assert "benchmark dataset file not found" in str(error["errors"][0])


def test_receipts_verify_cli_verifies_committed_receipt(tmp_path: Path) -> None:
    root = tmp_path / "store"
    receipt_path = tmp_path / "receipt.json"
    store = init_store(root)
    store.put_batch([_item(float(index), step=index) for index in range(2)])
    committed_root = store.commit()
    retrieval = store.query(
        QuerySpec(
            vector=np.array([0.0, 0.0], dtype=np.float32),
            k=1,
            metric=Metric.L2,
            with_receipt=True,
        )
    )
    assert retrieval.receipt is not None
    receipt_path.write_text(
        json.dumps(retrieval.receipt.to_json()),
        encoding="utf-8",
    )

    ok = _run_cli("receipts", "verify", receipt_path, "--root", committed_root.hex())
    wrong_root = _run_cli(
        "receipts",
        "verify",
        receipt_path,
        "--root",
        "00" * 32,
    )

    assert ok.returncode == int(CliExitCode.SUCCESS), ok.stdout + ok.stderr
    ok_json = _stdout_json(ok)
    assert ok_json["schema_version"] == "mneme.receipt_verification.v1"
    assert ok_json["ok"] is True
    assert ok_json["proof_count"] == 1
    assert wrong_root.returncode == int(CliExitCode.DATA_VALIDATION)
    assert _stdout_json(wrong_root)["ok"] is False


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


def _run_cli(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mneme.cli", *(str(arg) for arg in args)],
        check=False,
        text=True,
        capture_output=True,
    )


def _stdout_json(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    return data
