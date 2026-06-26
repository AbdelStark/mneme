from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from _entrypoint_runner import run_entrypoint

from mneme.core import (
    CliExitCode,
    EncoderFingerprint,
    EvaluationError,
    MemoryItem,
    Metric,
    Transition,
)
from mneme.eval import run_receipt_profile_evaluation, validate_report_json
from mneme.eval.receipts import main as receipts_main
from mneme.store import init_store, open_store


def test_receipt_profile_report_records_latency_and_proof_size_trend(
    tmp_path: Path,
) -> None:
    store = init_store(tmp_path / "store")
    store.put_batch([_item(float(index), step=index) for index in range(8)])
    store.commit()

    report = run_receipt_profile_evaluation(
        open_store(store.path),
        k=2,
        metric=Metric.L2,
        query_count=3,
        warmup_count=0,
        measurement_count=2,
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    decoded = validate_report_json(report.to_json())

    assert decoded == report
    assert report.report_id == "mneme-receipt-overhead-v1"
    assert report.dataset.kind == "fixture"
    assert report.dataset.metadata["fixture_scale"] is True
    assert report.metrics["item_count"] == 8
    assert report.metrics["committed_item_count"] == 8
    assert report.metrics["k"] == 2
    assert report.metrics["metric"] == "l2"
    assert report.metrics["receipt_proof_count_mean"] == 2.0
    assert report.metrics["receipt_proof_bytes_mean"] > 0.0
    assert report.metrics["receipt_bytes_per_returned_item_mean"] > 0.0
    assert report.metrics["disabled_query_latency_p50_ms"] >= 0.0
    assert report.metrics["receipt_query_latency_p50_ms"] >= 0.0
    assert report.metrics["receipt_query_overhead_p50_ms"] >= 0.0
    assert report.metrics["receipt_build_latency_p50_ms"] >= 0.0
    assert report.metrics["receipt_verify_latency_p50_ms"] >= 0.0
    trend_counts = _csv_ints(report.metrics["proof_size_trend_item_counts"])
    trend_bytes = _csv_ints(report.metrics["proof_size_trend_proof_bytes"])
    trend_steps = _csv_ints(report.metrics["proof_size_trend_proof_steps"])
    assert trend_counts == [1, 2, 4, 8]
    assert len(trend_bytes) == len(trend_counts)
    assert len(trend_steps) == len(trend_counts)
    assert report.artifacts["report_kind"] == "receipt-overhead"
    assert any("do not prove search correctness" in item for item in report.caveats)
    assert report.passed is True


def test_receipt_profile_rejects_uncommitted_store(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    store.put(_item(1.0, step=1))

    with pytest.raises(EvaluationError, match="requires a committed store"):
        run_receipt_profile_evaluation(store)


def test_receipt_eval_module_writes_valid_report_json(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    output = tmp_path / "reports" / "receipts.json"
    store.put_batch([_item(float(index), step=index) for index in range(4)])
    store.commit()

    completed = run_entrypoint(
        receipts_main,
        "--store",
        store.path,
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

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == str(output)
    report = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert report.command[:4] == ("mneme", "eval", "receipts", "--store")
    assert report.metrics["receipt_proof_count_mean"] == 2.0
    assert report.artifacts["proof_size_trend"] == "metrics:proof_size_trend_*"


def test_receipt_eval_module_rejects_negative_warmup_count(
    tmp_path: Path,
) -> None:
    completed = run_entrypoint(
        receipts_main,
        "--store",
        tmp_path / "missing-store",
        "--out",
        tmp_path / "receipts.json",
        "--warmup",
        "-1",
    )

    assert completed.returncode == int(CliExitCode.USER_INPUT)
    assert completed.stdout == ""
    assert "argument --warmup: must be a non-negative integer" in completed.stderr


def _csv_ints(value: object) -> list[int]:
    assert isinstance(value, str)
    return [int(part) for part in value.split(",")]


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
        meta={"source": "receipt-profile-fixture", "step": step},
        encoder_fp=_fingerprint(),
    )
