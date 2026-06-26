"""Receipt overhead and proof-size profiling reports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from mneme._version import __version__
from mneme.core import (
    Cid,
    EvaluationError,
    Metric,
    QuerySpec,
    Retrieval,
    content_id,
)
from mneme.core._json import dumps_strict_json
from mneme.core._time import utc_now_iso
from mneme.eval._profile import (
    _detect_git_commit,
    _latency_metrics,
    _measure_ms,
    _per_item,
    _profile_platform_summary,
    _query_vectors,
    _require_non_negative_int,
    _require_positive_int,
    _visible_items,
)
from mneme.eval._reports import DatasetRef, EvalMetric, EvalReport
from mneme.receipts import (
    CommitmentState,
    RetrievalReceipt,
    build_retrieval_receipt,
    verify_retrieval_receipt,
)
from mneme.store import LocalStore

_RECEIPT_PROFILE_CAVEAT: Final = (
    "Receipt profiling reports membership-proof overhead for the supplied local "
    "store only; receipts do not prove search correctness or control-loop "
    "suitability."
)
_COMMITTED_STORE_CAVEAT: Final = (
    "Run this report on target hardware before citing receipt latency or proof "
    "size in release notes."
)


def run_receipt_profile_evaluation(
    store: LocalStore,
    *,
    k: int = 4,
    metric: Metric = Metric.L2,
    query_count: int = 8,
    warmup_count: int = 2,
    measurement_count: int = 20,
    seed: int = 0,
    command: Sequence[str] = ("mneme", "eval", "receipts"),
    created_at: str | None = None,
    git_commit: str | None = None,
) -> EvalReport:
    """Build a local receipt overhead and proof-size report."""

    _require_positive_int(k, "k")
    _require_positive_int(query_count, "query_count")
    _require_non_negative_int(warmup_count, "warmup_count")
    _require_positive_int(measurement_count, "measurement_count")
    if not isinstance(metric, Metric):
        raise EvaluationError("metric must be a Metric")

    stats = store.stats()
    if not stats.commitments_enabled:
        raise EvaluationError(
            "receipt profile requires a committed store; run "
            "`mneme store commit-init PATH` first"
        )
    state = store.commitment_state()
    if state.item_count != stats.value_record_count:
        raise EvaluationError(
            "receipt profile requires commitments to cover all value-log records; "
            "run `mneme store commit-init PATH` after writes"
        )

    visible_items = _visible_items(store)
    if not visible_items:
        raise EvaluationError("receipt profile requires at least one visible item")
    effective_k = min(k, len(visible_items))
    queries = _query_vectors(visible_items, query_count)
    disabled_specs = [
        QuerySpec(vector=query, k=effective_k, metric=metric) for query in queries
    ]
    receipt_specs = [
        QuerySpec(vector=query, k=effective_k, metric=metric, with_receipt=True)
        for query in queries
    ]
    base_retrievals = [store.query(spec) for spec in disabled_specs]
    receipt_retrievals = [store.query(spec) for spec in receipt_specs]
    receipts = tuple(_require_receipt(retrieval) for retrieval in receipt_retrievals)
    root = state.root

    disabled_query_latencies = _measure_ms(
        lambda index: store.query(disabled_specs[index % len(disabled_specs)]),
        warmup_count=warmup_count,
        measurement_count=measurement_count,
    )
    receipt_query_latencies = _measure_ms(
        lambda index: store.query(receipt_specs[index % len(receipt_specs)]),
        warmup_count=warmup_count,
        measurement_count=measurement_count,
    )
    build_latencies = _measure_ms(
        lambda index: _build_receipt_for_retrieval(
            store,
            receipt_specs[index % len(receipt_specs)],
            base_retrievals[index % len(base_retrievals)],
        ),
        warmup_count=warmup_count,
        measurement_count=measurement_count,
    )
    verify_latencies = _measure_ms(
        lambda index: _verify_receipt_once(
            receipts[index % len(receipts)],
            receipt_retrievals[index % len(receipt_retrievals)],
            receipt_specs[index % len(receipt_specs)],
            root,
        ),
        warmup_count=warmup_count,
        measurement_count=measurement_count,
    )
    query_overheads = tuple(
        max(enabled - disabled, 0.0)
        for enabled, disabled in zip(
            receipt_query_latencies,
            disabled_query_latencies,
            strict=True,
        )
    )

    proof_counts = tuple(len(receipt.proofs) for receipt in receipts)
    proof_step_counts = tuple(
        sum(len(proof.steps) for proof in receipt.proofs) for receipt in receipts
    )
    proof_bytes = tuple(
        sum(_json_size(proof.to_json()) for proof in receipt.proofs)
        for receipt in receipts
    )
    receipt_bytes = tuple(_json_size(receipt.to_json()) for receipt in receipts)
    trend = _proof_size_trend(state)
    dimension = int(visible_items[0][1].key.shape[0])

    metrics: dict[str, EvalMetric] = {
        "item_count": len(visible_items),
        "value_record_count": stats.value_record_count,
        "visible_record_count": stats.visible_record_count,
        "committed_item_count": state.item_count,
        "dimension": dimension,
        "k": effective_k,
        "metric": metric.value,
        "backend": stats.index_backend,
        "query_count": len(queries),
        "warmup_count": warmup_count,
        "measurement_count": measurement_count,
        "receipt_proof_count_mean": _mean(proof_counts),
        "receipt_proof_count_max": max(proof_counts),
        "receipt_proof_step_count_mean": _mean(proof_step_counts),
        "receipt_proof_step_count_max": max(proof_step_counts),
        "receipt_proof_bytes_mean": _mean(proof_bytes),
        "receipt_proof_bytes_max": max(proof_bytes),
        "receipt_bytes_mean": _mean(receipt_bytes),
        "receipt_bytes_max": max(receipt_bytes),
        "receipt_bytes_per_returned_item_mean": _per_item(
            int(sum(receipt_bytes)),
            int(sum(proof_counts)),
        ),
        "proof_size_trend_item_counts": ",".join(
            str(row["item_count"]) for row in trend
        ),
        "proof_size_trend_proof_bytes": ",".join(
            str(row["proof_bytes"]) for row in trend
        ),
        "proof_size_trend_proof_steps": ",".join(
            str(row["proof_steps"]) for row in trend
        ),
    }
    metrics.update(_latency_metrics("disabled_query_latency", disabled_query_latencies))
    metrics.update(_latency_metrics("receipt_query_latency", receipt_query_latencies))
    metrics.update(_latency_metrics("receipt_query_overhead", query_overheads))
    metrics.update(_latency_metrics("receipt_build_latency", build_latencies))
    metrics.update(_latency_metrics("receipt_verify_latency", verify_latencies))

    return EvalReport(
        report_id="mneme-receipt-overhead-v1",
        command=tuple(command),
        package_version=__version__,
        git_commit=_detect_git_commit() if git_commit is None else git_commit,
        created_at=utc_now_iso() if created_at is None else created_at,
        platform=_profile_platform_summary(),
        seed=seed,
        dataset=DatasetRef(
            dataset_id="local-store-receipt-profile",
            kind="fixture",
            split="local",
            version="v1",
            metadata={
                "store_id": str(stats.store_id),
                "fixture_scale": True,
                "synthetic": False,
                "commitment_backend": store.manifest.commitment.backend,
                "commitment_root": store.manifest.commitment.root,
            },
        ),
        metrics=metrics,
        artifacts={
            "report_kind": "receipt-overhead",
            "proof_size_trend": "metrics:proof_size_trend_*",
        },
        caveats=(_RECEIPT_PROFILE_CAVEAT, _COMMITTED_STORE_CAVEAT),
        passed=all(
            _verify_receipt_once(receipt, retrieval, spec, root)
            for receipt, retrieval, spec in zip(
                receipts,
                receipt_retrievals,
                receipt_specs,
                strict=True,
            )
        ),
    )


def _build_receipt_for_retrieval(
    store: LocalStore,
    spec: QuerySpec,
    retrieval: Retrieval,
) -> RetrievalReceipt:
    state = store.commitment_state()
    ids = _retrieval_ids(retrieval)
    return build_retrieval_receipt(
        root=state.root,
        ids=ids,
        proofs=tuple(state.prove(cid) for cid in ids),
        query=spec,
        store_id=str(store.manifest.store_id),
    )


def _verify_receipt_once(
    receipt: RetrievalReceipt,
    retrieval: Retrieval,
    spec: QuerySpec,
    root: bytes,
) -> bool:
    return verify_retrieval_receipt(receipt, retrieval.items, root=root, query=spec)


def _require_receipt(retrieval: Retrieval) -> RetrievalReceipt:
    if not isinstance(retrieval.receipt, RetrievalReceipt):
        raise EvaluationError("receipt query did not return a RetrievalReceipt")
    return retrieval.receipt


def _retrieval_ids(retrieval: Retrieval) -> tuple[Cid, ...]:
    return tuple(item.content_id or content_id(item) for item in retrieval.items)


def _proof_size_trend(state: CommitmentState) -> tuple[dict[str, int], ...]:
    rows: list[dict[str, int]] = []
    for item_count in _trend_item_counts(state.item_count):
        prefix = CommitmentState.from_cids(state.leaf_ids[:item_count])
        proof = prefix.prove(prefix.leaf_ids[-1])
        rows.append(
            {
                "item_count": item_count,
                "proof_steps": len(proof.steps),
                "proof_bytes": _json_size(proof.to_json()),
            }
        )
    return tuple(rows)


def _trend_item_counts(item_count: int) -> tuple[int, ...]:
    counts: list[int] = []
    current = 1
    while current < item_count:
        counts.append(current)
        current *= 2
    if not counts or counts[-1] != item_count:
        counts.append(item_count)
    return tuple(counts)


def _json_size(data: object) -> int:
    return len(
        dumps_strict_json(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _mean(values: Sequence[int | float]) -> float:
    if not values:
        raise EvaluationError("mean requires at least one value")
    return float(sum(float(value) for value in values) / len(values))


__all__ = ["run_receipt_profile_evaluation"]
