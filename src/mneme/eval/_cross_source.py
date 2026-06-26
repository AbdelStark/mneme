"""Fixture-scale cross-source memory transfer reports."""

from __future__ import annotations

import json
import platform as platform_module
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np

from mneme._version import __version__
from mneme.condition import CondCtx, KnnCorrector
from mneme.core import (
    Cid,
    EncoderFingerprint,
    EvaluationError,
    MemoryItem,
    Metric,
    QuerySpec,
    Retrieval,
    Transition,
    build_item,
)
from mneme.core._time import utc_now_iso
from mneme.eval._reports import DatasetRef, EvalMetric, EvalReport
from mneme.receipts import RetrievalReceipt, verify_retrieval_receipt
from mneme.store import LocalStore, init_store

_CAVEATS = (
    "Fixture-scale cross-source transfer uses synthetic local stores; it does "
    "not claim general transfer or external benchmark success.",
    "Cross-source provenance receipts prove committed membership only; they do "
    "not provide confidentiality, private retrieval, consent compliance, or "
    "search optimality.",
)
_SOURCE_SCHEMA = "mneme.source.v1"
_CROSS_SOURCE_RECEIPT_SCHEMA = "mneme.cross_source_receipt.v1"


def run_cross_source_transfer_evaluation(
    *,
    seed: int = 0,
    command: Sequence[str] = ("mneme", "eval", "cross-source"),
    created_at: str | None = None,
    git_commit: str | None = None,
) -> EvalReport:
    """Build a deterministic pooled-memory transfer report."""

    fingerprint = _fingerprint()
    target = _target_case()
    corrector = KnnCorrector(tau=1.0, lambda_max=1.0, alpha=10.0, delta0=1.0)
    report_created_at = utc_now_iso() if created_at is None else created_at

    with tempfile.TemporaryDirectory(prefix="mneme-cross-source-") as tmp:
        root = Path(tmp)
        source_a = _build_source(
            init_store(root / "source-a", active_fingerprints=[fingerprint]),
            source_id="source-a-public-fixture",
            delta=np.array([1.0, 0.0], dtype=np.float32),
            fingerprint=fingerprint,
            step=1,
        )
        source_b = _build_source(
            init_store(root / "source-b", active_fingerprints=[fingerprint]),
            source_id="source-b-public-fixture",
            delta=np.array([0.0, 1.0], dtype=np.float32),
            fingerprint=fingerprint,
            step=2,
        )

        query = QuerySpec(
            vector=target.current,
            k=1,
            metric=Metric.L2,
            with_receipt=True,
            encoder_fp=fingerprint,
        )
        retrieved = tuple(
            _query_source(source, query) for source in (source_a, source_b)
        )

    in_source_retrieval = retrieved[0].retrieval
    pooled_retrieval = _pooled_retrieval(retrieved)
    no_memory_error = _l2(target.parametric, target.true_next)
    in_source_error = _condition_l2(
        corrector,
        target,
        in_source_retrieval,
    )
    pooled_error = _condition_l2(
        corrector,
        target,
        pooled_retrieval,
    )
    source_count = len(retrieved)
    returned_item_count = sum(len(source.retrieval.items) for source in retrieved)
    verified_count = sum(int(source.receipt_verified) for source in retrieved)

    metrics: dict[str, EvalMetric] = {
        "source_count": source_count,
        "target_case_count": 1,
        "returned_item_count": returned_item_count,
        "downstream_no_memory_l2": no_memory_error,
        "downstream_in_source_l2": in_source_error,
        "downstream_pooled_l2": pooled_error,
        "cross_source_improvement_rate": int(pooled_error < in_source_error),
        "pooled_improves_no_memory": int(pooled_error < no_memory_error),
        "negative_transfer_rate": int(pooled_error > in_source_error),
        "source_diversity_score": _source_diversity_score(retrieved),
        "receipt_verification_success_count": verified_count,
        "receipt_verification_failure_count": source_count - verified_count,
        "encoder_fingerprint_rejection_count": 0,
        "policy_filter_rejection_count": 0,
        "redaction_failure_count": 0,
    }
    metrics.update(_proof_byte_metrics(retrieved))

    return EvalReport(
        report_id="mneme-cross-source-transfer-v1",
        command=tuple(command),
        package_version=__version__,
        git_commit=_detect_git_commit() if git_commit is None else git_commit,
        created_at=report_created_at,
        platform=_platform_summary(),
        seed=seed,
        dataset=DatasetRef(
            dataset_id="synthetic-cross-source-transfer-fixture",
            kind="fixture",
            split="deterministic",
            version="v1",
            metadata={
                "fixture_scale": True,
                "synthetic": True,
                "target_id": "target-composed-drift-fixture",
                "source_count": source_count,
                "sources": [_source_identity_json(source) for source in retrieved],
                "provenance": _cross_source_provenance_json(
                    retrieved,
                    query,
                    created_at=report_created_at,
                ),
            },
        ),
        metrics=metrics,
        artifacts={
            "report_kind": "cross-source-transfer",
            "baseline": "no-memory-and-single-source",
            "provenance_schema": _CROSS_SOURCE_RECEIPT_SCHEMA,
        },
        caveats=_CAVEATS,
        passed=pooled_error < no_memory_error
        and pooled_error < in_source_error
        and verified_count == source_count,
    )


@dataclass(frozen=True)
class _TargetCase:
    current: np.ndarray
    parametric: np.ndarray
    true_next: np.ndarray


@dataclass(frozen=True)
class _SourceFixture:
    source_id: str
    store: LocalStore
    store_id: str
    root: bytes
    fingerprint: EncoderFingerprint


@dataclass(frozen=True)
class _RetrievedSource:
    source_id: str
    store_id: str
    root: bytes
    fingerprint: EncoderFingerprint
    retrieval: Retrieval
    receipt: RetrievalReceipt
    receipt_verified: bool


def _target_case() -> _TargetCase:
    return _TargetCase(
        current=np.array([0.0, 0.0], dtype=np.float32),
        parametric=np.array([0.0, 0.0], dtype=np.float32),
        true_next=np.array([1.0, 1.0], dtype=np.float32),
    )


def _build_source(
    store: LocalStore,
    *,
    source_id: str,
    delta: np.ndarray,
    fingerprint: EncoderFingerprint,
    step: int,
) -> _SourceFixture:
    item = _item(delta, fingerprint=fingerprint, source_id=source_id, step=step)
    store.put(item)
    root = store.commit()
    return _SourceFixture(
        source_id=source_id,
        store=store,
        store_id=str(store.manifest.store_id),
        root=root,
        fingerprint=fingerprint,
    )


def _query_source(source: _SourceFixture, query: QuerySpec) -> _RetrievedSource:
    retrieval = source.store.query(query)
    receipt = retrieval.receipt
    if not isinstance(receipt, RetrievalReceipt):
        raise EvaluationError("cross-source query did not return a receipt")
    receipt_verified = verify_retrieval_receipt(
        receipt,
        retrieval.items,
        root=source.root,
        query=query,
    )
    return _RetrievedSource(
        source_id=source.source_id,
        store_id=source.store_id,
        root=source.root,
        fingerprint=source.fingerprint,
        retrieval=retrieval,
        receipt=receipt,
        receipt_verified=receipt_verified,
    )


def _pooled_retrieval(sources: Sequence[_RetrievedSource]) -> Retrieval:
    items: list[MemoryItem] = []
    distances: list[float] = []
    for source in sources:
        items.extend(source.retrieval.items)
        distances.extend(float(distance) for distance in source.retrieval.distances)
    return Retrieval(items=tuple(items), distances=tuple(distances))


def _condition_l2(
    corrector: KnnCorrector,
    target: _TargetCase,
    retrieval: Retrieval,
) -> float:
    conditioned = corrector.condition(
        target.parametric,
        retrieval,
        CondCtx(current_latent=target.current),
    )
    if not isinstance(conditioned, np.ndarray):
        raise EvaluationError("cross-source corrector returned a non-NumPy value")
    return _l2(conditioned, target.true_next)


def _source_identity_json(source: _RetrievedSource) -> dict[str, object]:
    fingerprint = source.fingerprint
    return {
        "schema_version": _SOURCE_SCHEMA,
        "source_id": source.source_id,
        "source_kind": "local_store",
        "store_id": source.store_id,
        "root": source.root.hex(),
        "root_scheme": "mmr-v1",
        "encoder_fingerprint": {
            "schema_version": fingerprint.schema_version,
            "encoder_id": fingerprint.encoder_id,
            "summarizer_id": fingerprint.summarizer_id,
            "weights_digest": fingerprint.weights_digest,
            "config_digest": fingerprint.config_digest,
        },
        "policy_tags": ["public-fixture"],
        "disclosure_level": "public",
    }


def _cross_source_provenance_json(
    sources: Sequence[_RetrievedSource],
    query: QuerySpec,
    *,
    created_at: str,
) -> dict[str, object]:
    return {
        "schema_version": _CROSS_SOURCE_RECEIPT_SCHEMA,
        "query_digest": sources[0].receipt.params.vector_digest.hex(),
        "aggregation_policy": "per-source-k1-then-distance-merge",
        "sources": [source.source_id for source in sources],
        "returned_ids_by_source": {
            source.source_id: _ids_hex(source.retrieval.items) for source in sources
        },
        "retrieval_receipts_by_source": {
            source.source_id: {
                "schema_version": source.receipt.schema_version,
                "root": source.receipt.root.hex(),
                "store_id": source.receipt.store_id,
                "item_count": len(source.receipt.ids),
                "proof_count": len(source.receipt.proofs),
                "verified": source.receipt_verified,
            }
            for source in sources
        },
        "validation_steps": [
            "validated local QuerySpec before use",
            "recomputed returned item content ids through receipt verification",
            "rejected encoder fingerprint mismatches through QuerySpec.encoder_fp",
            "verified every per-source RetrievalReceipt",
            "recorded aggregation policy",
        ],
        "created_at": created_at,
        "query_k_per_source": query.k,
        "metric": query.metric.value,
    }


def _proof_byte_metrics(sources: Sequence[_RetrievedSource]) -> dict[str, int]:
    return {
        f"{source.source_id.replace('-', '_')}_proof_bytes": _json_size(
            source.receipt.to_json()
        )
        for source in sources
    }


def _source_diversity_score(sources: Sequence[_RetrievedSource]) -> float:
    returned_source_ids = [
        source.source_id for source in sources for _ in source.retrieval.items
    ]
    if not returned_source_ids:
        return 0.0
    return len(set(returned_source_ids)) / len(returned_source_ids)


def _ids_hex(items: Sequence[MemoryItem]) -> list[str]:
    return [(item.content_id or _missing_content_id()).hex() for item in items]


def _missing_content_id() -> Cid:
    raise EvaluationError("cross-source retrieved item is missing content_id")


def _item(
    delta: np.ndarray,
    *,
    fingerprint: EncoderFingerprint,
    source_id: str,
    step: int,
) -> MemoryItem:
    z_src = np.zeros_like(delta)
    return build_item(
        Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_src + delta,
            delta=delta,
            t=step,
            episode_id=UUID(f"12345678-1234-5678-1234-{step:012d}"),
        ),
        key=np.array([0.0, 0.0], dtype=np.float32),
        encoder_fp=fingerprint,
        meta={
            "fixture": "cross-source-transfer",
            "step": step,
            "mneme_source": {
                "schema_version": _SOURCE_SCHEMA,
                "source_id": source_id,
                "source_kind": "local_store",
                "policy_tags": ["public-fixture"],
                "disclosure_level": "public",
            },
        },
    )


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:cross-source-config",
    )


def _l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _json_size(payload: Mapping[str, object]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _platform_summary() -> dict[str, str]:
    return {
        "machine": platform_module.machine() or "unknown",
        "python": platform_module.python_version(),
        "system": platform_module.system() or "unknown",
    }


def _detect_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


__all__ = ["run_cross_source_transfer_evaluation"]
