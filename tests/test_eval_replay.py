from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mneme.core import (
    EncoderFingerprint,
    EvaluationError,
    MemoryItem,
    Metric,
    QuerySpec,
    Transition,
)
from mneme.eval import (
    RECEIPT_REPLAY_REPORT_SCHEMA,
    RECEIPT_REPLAY_TRACE_SCHEMA,
    KnnReplayConfig,
    ReceiptReplayReport,
    ReceiptReplayTrace,
    build_receipt_replay_trace,
    load_replay_trace_json,
    replay_receipt_trace,
    write_replay_report_json,
    write_replay_trace_json,
)
from mneme.store import init_store, open_store


def test_receipt_replay_trace_reproduces_prediction_and_reports_root_ids(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)

    report = replay_receipt_trace(trace)
    reloaded = ReceiptReplayTrace.from_json(trace.to_json())
    reloaded_report = replay_receipt_trace(reloaded)

    assert trace.schema_version == RECEIPT_REPLAY_TRACE_SCHEMA
    assert report.schema_version == RECEIPT_REPLAY_REPORT_SCHEMA
    assert report.ok
    assert report.conditioned
    assert report.root == trace.receipt.root
    assert report.ids == trace.receipt.ids
    assert report.mismatch_causes == ()
    assert report.max_abs_error == 0.0
    assert report.replayed_prediction is not None
    np.testing.assert_allclose(report.replayed_prediction, trace.expected_prediction)
    assert reloaded_report.ok


def test_receipt_replay_altered_item_fails_before_conditioning(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path)
    altered = replace(
        trace.items[0],
        meta={**dict(trace.items[0].meta), "tampered": True},
    )
    tampered = replace(trace, items=(altered, *trace.items[1:]))

    report = replay_receipt_trace(tampered)

    assert not report.ok
    assert not report.conditioned
    assert report.replayed_prediction is None
    assert report.max_abs_error is None
    assert report.mismatch_causes == ("receipt_verification_failed",)


def test_receipt_replay_shape_mismatch_writes_strict_json(tmp_path: Path) -> None:
    trace = replace(
        _trace(tmp_path),
        expected_prediction=np.array([[1.5, 0.0]], dtype=np.float32),
    )
    output = tmp_path / "shape-mismatch-report.json"

    report = replay_receipt_trace(trace)
    write_replay_report_json(report, output)

    payload = output.read_text(encoding="utf-8")
    written = json.loads(payload)
    assert not report.ok
    assert report.mismatch_causes == ("prediction_shape_mismatch",)
    assert report.max_abs_error is None
    assert "Infinity" not in payload
    assert written["max_abs_error"] is None


def test_receipt_replay_cli_writes_report(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    trace_path = tmp_path / "trace.json"
    report_path = tmp_path / "report.json"
    write_replay_trace_json(trace, trace_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mneme.cli",
            "eval",
            "replay",
            "--trace",
            str(trace_path),
            "--out",
            str(report_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    printed = json.loads(completed.stdout)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert printed == written
    assert written["schema_version"] == RECEIPT_REPLAY_REPORT_SCHEMA
    assert written["ok"] is True
    assert written["root"] == trace.receipt.root.hex()
    assert written["ids"] == [cid.hex() for cid in trace.receipt.ids]


def test_receipt_replay_loader_rejects_non_digest_item_ids(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace = _trace(tmp_path)
    payload = trace.to_json()
    payload["items"][0]["content_id"] = "00"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="content_id must be 32 bytes"):
        load_replay_trace_json(trace_path)


def test_receipt_replay_loader_rejects_nonstandard_json_constants(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text('{"schema_version": NaN}', encoding="utf-8")

    with pytest.raises(EvaluationError, match="replay trace is not valid JSON"):
        load_replay_trace_json(trace_path)


def test_receipt_replay_loader_wraps_invalid_item_payloads(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace = _trace(tmp_path)
    payload = trace.to_json()
    payload["items"][0]["key"]["shape"] = [True, 2]
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="invalid replay item payload"):
        load_replay_trace_json(trace_path)


def test_receipt_replay_loader_wraps_invalid_receipt_payloads(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace = _trace(tmp_path)
    payload = trace.to_json()
    payload["receipt"] = {"schema_version": "mneme.receipt.v1"}
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="invalid replay receipt payload"):
        load_replay_trace_json(trace_path)


def test_receipt_replay_loader_wraps_invalid_array_payloads(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace = _trace(tmp_path)
    payload = trace.to_json()
    payload["parametric_prediction"]["shape"] = [True, 2]
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="parametric_prediction is invalid"):
        load_replay_trace_json(trace_path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tau": 0.0}, "tau must be positive"),
        ({"lambda_max": 1.5}, "lambda_max must be between 0 and 1"),
        ({"alpha": -1.0}, "alpha must be non-negative"),
        ({"delta0": float("nan")}, "delta0 must be finite"),
        ({"mode": "unknown"}, "mode must be 'delta' or 'absolute'"),
    ],
)
def test_knn_replay_config_rejects_invalid_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(EvaluationError, match=match):
        KnnReplayConfig(**kwargs)


def test_receipt_replay_loader_rejects_invalid_conditioner_config(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    trace = _trace(tmp_path)
    payload = trace.to_json()
    payload["conditioner"]["tau"] = 0.0
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="tau must be positive"):
        load_replay_trace_json(trace_path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"schema_version": "mneme.receipt_replay_report.v2"}, "unsupported replay"),
        ({"ok": "yes"}, "ok must be a bool"),
        ({"root": object()}, "root must be bytes"),
        ({"root": b"short"}, "root must be 32 bytes"),
        ({"ids": object()}, "ids must be a sequence"),
        ({"ids": (b"short",)}, "ids item must be 32 bytes"),
        ({"conditioned": "yes"}, "conditioned must be a bool"),
        ({"mismatch_causes": object()}, "mismatch_causes must be a sequence"),
        (
            {"mismatch_causes": ("",)},
            "mismatch_causes item must be a non-empty string",
        ),
        ({"max_abs_error": float("nan")}, "max_abs_error must be finite"),
        ({"max_abs_error": -1.0}, "max_abs_error must be non-negative"),
        ({"expected_prediction": object()}, "expected_prediction must be"),
        ({"replayed_prediction": object()}, "replayed_prediction must be"),
    ],
)
def test_receipt_replay_report_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = _report_values()
    values.update(kwargs)

    with pytest.raises(EvaluationError, match=match):
        ReceiptReplayReport(**values)


def _report_values() -> dict[str, object]:
    return {
        "ok": True,
        "root": b"\x00" * 32,
        "ids": (b"\x01" * 32,),
        "conditioned": True,
        "mismatch_causes": (),
        "max_abs_error": 0.0,
        "expected_prediction": np.array([1.5, 0.0], dtype=np.float32),
        "replayed_prediction": np.array([1.5, 0.0], dtype=np.float32),
    }


def _trace(tmp_path: Path) -> ReceiptReplayTrace:
    root = tmp_path / "store"
    store = init_store(root)
    store.put_batch([_item(float(index), step=index) for index in range(3)])
    store.commit()
    reopened = open_store(root)
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=2,
        metric=Metric.L2,
        with_receipt=True,
    )
    retrieval = reopened.query(spec)
    return build_receipt_replay_trace(
        query=spec,
        retrieval=retrieval,
        current_latent=np.array([1.0, 0.0], dtype=np.float32),
        parametric_prediction=np.array([1.5, 0.0], dtype=np.float32),
    )


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
        meta={"source": "receipt-replay-fixture", "step": step},
        encoder_fp=_fingerprint(),
    )
