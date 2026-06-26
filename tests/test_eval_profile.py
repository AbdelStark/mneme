from __future__ import annotations

import importlib
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
from mneme.eval import run_profile_evaluation, validate_report_json
from mneme.eval.profile import main as profile_main
from mneme.store import count_retention, init_store, open_store


def test_profile_report_records_recall_latency_and_footprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_missing_faiss(monkeypatch)
    store = init_store(tmp_path / "store")
    store.put_batch([_item(float(index), step=index) for index in range(4)])

    report = run_profile_evaluation(
        open_store(store.path),
        k=2,
        metric=Metric.L2,
        query_count=3,
        warmup_count=1,
        measurement_count=3,
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    decoded = validate_report_json(report.to_json())

    assert decoded == report
    assert report.report_id == "mneme-profile-recall-latency-footprint-v1"
    assert report.dataset.kind == "fixture"
    assert report.dataset.metadata["fixture_scale"] is True
    assert report.metrics["item_count"] == 4
    assert report.metrics["visible_record_count"] == 4
    assert report.metrics["dimension"] == 2
    assert report.metrics["k"] == 2
    assert report.metrics["metric"] == "l2"
    assert report.metrics["backend"] == "flat"
    assert report.metrics["ground_truth_backend"] == "flat"
    assert report.metrics["flat_recall_at_k"] == 1.0
    assert report.metrics["approx_backend"] == "faiss_hnsw"
    assert report.metrics["approx_backend_available"] == 0
    assert report.metrics["approx_recall_at_k"] == "unavailable"
    assert report.metrics["query_latency_p50_ms"] >= 0.0
    assert (
        report.metrics["query_latency_p99_ms"] >= report.metrics["query_latency_p50_ms"]
    )
    assert report.metrics["conditioning_latency_p50_ms"] >= 0.0
    assert report.metrics["memory_footprint_bytes_per_item"] > 0.0
    assert report.platform["cpu_count"]
    assert report.platform["memory"] == "unknown"
    assert report.artifacts["ground_truth_backend"] == "flat"
    assert any("Approximate backend was unavailable" in item for item in report.caveats)
    assert report.passed is True


def test_profile_report_respects_visible_items_after_retention(
    tmp_path: Path,
) -> None:
    store = init_store(tmp_path / "store", retention_policy=count_retention(2))
    store.put_batch([_item(float(index), step=index) for index in range(4)])

    report = run_profile_evaluation(
        open_store(store.path),
        k=4,
        metric=Metric.L2,
        query_count=4,
        warmup_count=0,
        measurement_count=1,
        approximate_backend=None,
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    assert report.metrics["item_count"] == 2
    assert report.metrics["visible_record_count"] == 2
    assert report.metrics["value_record_count"] == 4
    assert report.metrics["k"] == 2
    assert report.metrics["approx_backend"] == "none"
    assert report.metrics["approx_recall_at_k"] == "not_requested"
    assert report.dataset.metadata["tombstone_count"] == 2


def test_profile_report_rejects_empty_store(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")

    with pytest.raises(EvaluationError, match="at least one visible item"):
        run_profile_evaluation(store)


def test_profile_eval_module_writes_valid_report_json(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    output = tmp_path / "reports" / "profile.json"
    store.put_batch([_item(float(index), step=index) for index in range(3)])

    completed = run_entrypoint(
        profile_main,
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
        "2",
        "--approx-backend",
        "none",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == str(output)
    report = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert report.command[:4] == ("mneme", "eval", "profile", "--store")
    assert report.metrics["flat_recall_at_k"] == 1.0
    assert report.metrics["memory_footprint_bytes_per_item"] > 0.0


def test_profile_eval_module_rejects_non_positive_query_count(
    tmp_path: Path,
) -> None:
    completed = run_entrypoint(
        profile_main,
        "--store",
        tmp_path / "missing-store",
        "--out",
        tmp_path / "profile.json",
        "--queries",
        "0",
    )

    assert completed.returncode == int(CliExitCode.USER_INPUT)
    assert completed.stdout == ""
    assert "argument --queries: must be a positive integer" in completed.stderr


def _force_missing_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = importlib.import_module

    def missing_faiss(name: str, package: str | None = None) -> object:
        if name == "faiss":
            raise ImportError("missing faiss")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_faiss)


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
        meta={"source": "profile-fixture", "step": step},
        encoder_fp=_fingerprint(),
    )
