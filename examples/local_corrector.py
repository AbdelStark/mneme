"""Minimal local-store and kNN-corrector example.

Run from a checkout with:

    python3 examples/local_corrector.py

The output is a JSON success signal for a synthetic fixture only. It is not
external benchmark evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import numpy as np

from mneme.condition import CondCtx, KnnCorrector
from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    Transition,
    build_item,
)
from mneme.store import init_store, verify_store


def main() -> int:
    fingerprint = EncoderFingerprint(
        encoder_id="example.encoder",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:example-config",
    )
    query = np.array([1.0, 0.0], dtype=np.float32)
    parametric = np.array([5.0, 5.0], dtype=np.float32)
    target = np.array([2.0, 0.0], dtype=np.float32)

    with TemporaryDirectory(prefix="mneme-local-corrector-") as tmp:
        store = init_store(Path(tmp) / "store", active_fingerprints=[fingerprint])
        store.put_batch(_fixture_items(fingerprint))
        retrieval = store.query(
            QuerySpec(
                vector=query,
                k=2,
                metric=Metric.L2,
                encoder_fp=fingerprint,
            )
        )
        corrected = KnnCorrector(
            tau=1.0,
            lambda_max=1.0,
            alpha=0.0,
            mode="absolute",
        ).condition(parametric, retrieval, CondCtx(current_latent=query))
        corrected_array = _as_array(corrected)
        no_memory_l2 = _l2(parametric, target)
        corrected_l2 = _l2(corrected_array, target)
        store_ok = verify_store(store.path).ok

    payload = {
        "ok": bool(store_ok and corrected_l2 < no_memory_l2),
        "example": "local-corrector",
        "retrieved": len(retrieval.items),
        "no_memory_l2": no_memory_l2,
        "corrected_l2": corrected_l2,
        "success_signal": "corrected_l2 < no_memory_l2 on synthetic fixture",
        "claim_boundary": "fixture-only; not external benchmark evidence",
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0 if payload["ok"] else 1


def _fixture_items(fingerprint: EncoderFingerprint) -> list[MemoryItem]:
    return [
        build_item(
            Transition(
                z_src=np.array([1.0, 0.0], dtype=np.float32),
                action=np.array([0.0], dtype=np.float32),
                z_next=np.array([2.0, 0.0], dtype=np.float32),
                delta=np.array([1.0, 0.0], dtype=np.float32),
                t=1,
                episode_id=UUID("12345678-1234-5678-1234-567812345001"),
            ),
            key=np.array([1.0, 0.0], dtype=np.float32),
            encoder_fp=fingerprint,
            meta={"source": "example", "kind": "near"},
        ),
        build_item(
            Transition(
                z_src=np.array([2.0, 0.0], dtype=np.float32),
                action=np.array([0.0], dtype=np.float32),
                z_next=np.array([2.5, 0.0], dtype=np.float32),
                delta=np.array([0.5, 0.0], dtype=np.float32),
                t=2,
                episode_id=UUID("12345678-1234-5678-1234-567812345002"),
            ),
            key=np.array([2.0, 0.0], dtype=np.float32),
            encoder_fp=fingerprint,
            meta={"source": "example", "kind": "neighbor"},
        ),
    ]


def _l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _as_array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("corrector returned a non-NumPy value")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
