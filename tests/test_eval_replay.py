from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np

from mneme.core import EncoderFingerprint, MemoryItem, Metric, QuerySpec, Transition
from mneme.eval import (
    RECEIPT_REPLAY_REPORT_SCHEMA,
    RECEIPT_REPLAY_TRACE_SCHEMA,
    ReceiptReplayTrace,
    build_receipt_replay_trace,
    replay_receipt_trace,
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
