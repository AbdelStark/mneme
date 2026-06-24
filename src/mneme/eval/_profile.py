"""Fixture-scale recall, latency, and footprint profiling reports."""

from __future__ import annotations

import math
import os
import platform as platform_module
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Final

import numpy as np

from mneme._version import __version__
from mneme.condition import CondCtx, KnnCorrector
from mneme.core import (
    Cid,
    DTypeError,
    EvaluationError,
    MemoryItem,
    Metric,
    OptionalDependencyError,
    QuerySpec,
    Retrieval,
    SummaryVec,
    Transition,
)
from mneme.eval._reports import DatasetRef, EvalMetric, EvalReport
from mneme.index import FlatIndex, create_index_backend
from mneme.store import LocalStore

_PROFILE_CAVEAT: Final = (
    "Local profiling reports characterize the supplied store only and cannot "
    "prove external task success or broad benchmark improvement."
)
_APPROX_UNAVAILABLE_CAVEAT: Final = (
    "Approximate backend was unavailable; exact-only recall was reported "
    "against FlatIndex ground truth."
)


def run_profile_evaluation(
    store: LocalStore,
    *,
    k: int = 4,
    metric: Metric = Metric.L2,
    query_count: int = 8,
    warmup_count: int = 2,
    measurement_count: int = 20,
    approximate_backend: str | None = "faiss_hnsw",
    seed: int = 0,
    command: Sequence[str] = ("mneme", "eval", "profile"),
    created_at: str | None = None,
    git_commit: str | None = None,
) -> EvalReport:
    """Build a local profile report for recall, latency, and footprint."""

    _require_positive_int(k, "k")
    _require_positive_int(query_count, "query_count")
    _require_non_negative_int(warmup_count, "warmup_count")
    _require_positive_int(measurement_count, "measurement_count")
    if not isinstance(metric, Metric):
        raise EvaluationError("metric must be a Metric")

    visible_items = _visible_items(store)
    if not visible_items:
        raise EvaluationError("profile evaluation requires at least one visible item")

    effective_k = min(k, len(visible_items))
    queries = _query_vectors(visible_items, query_count)
    flat_index = _flat_ground_truth(visible_items)
    ground_truth = [
        flat_index.search(query, effective_k, metric=metric) for query in queries
    ]
    flat_recall = _mean_recall(ground_truth, ground_truth)

    caveats = [_PROFILE_CAVEAT]
    approx_available = 0
    approx_backend_name = "none" if approximate_backend is None else approximate_backend
    approx_recall: EvalMetric = "not_requested"
    if approximate_backend is not None:
        try:
            approx_index = create_index_backend(approximate_backend)
            approx_index.add_batch(_index_items(visible_items))
            approx_results = [
                approx_index.search(query, effective_k, metric=metric)
                for query in queries
            ]
        except OptionalDependencyError:
            approx_recall = "unavailable"
            caveats.append(_APPROX_UNAVAILABLE_CAVEAT)
        else:
            approx_available = 1
            approx_recall = _mean_recall(ground_truth, approx_results)

    query_specs = [
        QuerySpec(vector=query, k=effective_k, metric=metric) for query in queries
    ]
    query_latencies = _measure_ms(
        lambda index: store.query(query_specs[index % len(query_specs)]),
        warmup_count=warmup_count,
        measurement_count=measurement_count,
    )
    conditioning_inputs = [
        _conditioning_input(store.query(spec)) for spec in query_specs
    ]
    condition_latencies = _measure_ms(
        lambda index: _condition_once(
            conditioning_inputs[index % len(conditioning_inputs)]
        ),
        warmup_count=warmup_count,
        measurement_count=measurement_count,
    )

    stats = store.stats()
    key_bytes = sum(item.key.nbytes for _, item in visible_items)
    value_bytes = stats.value_bytes
    total_estimated_bytes = key_bytes + value_bytes
    item_count = len(visible_items)
    dimension = int(visible_items[0][1].key.shape[0])
    metrics: dict[str, EvalMetric] = {
        "item_count": item_count,
        "value_record_count": stats.value_record_count,
        "visible_record_count": stats.visible_record_count,
        "dimension": dimension,
        "k": effective_k,
        "metric": metric.value,
        "backend": stats.index_backend,
        "ground_truth_backend": "flat",
        "approx_backend": approx_backend_name,
        "approx_backend_available": approx_available,
        "query_count": len(queries),
        "warmup_count": warmup_count,
        "measurement_count": measurement_count,
        "flat_recall_at_k": flat_recall,
        "approx_recall_at_k": approx_recall,
        "key_bytes": key_bytes,
        "value_log_bytes": value_bytes,
        "total_estimated_footprint_bytes": total_estimated_bytes,
        "memory_footprint_bytes_per_item": _per_item(
            total_estimated_bytes,
            item_count,
        ),
        "key_bytes_per_item": _per_item(key_bytes, item_count),
        "value_log_bytes_per_visible_item": _per_item(value_bytes, item_count),
    }
    metrics.update(_latency_metrics("query_latency", query_latencies))
    metrics.update(_latency_metrics("conditioning_latency", condition_latencies))

    return EvalReport(
        report_id="mneme-profile-recall-latency-footprint-v1",
        command=tuple(command),
        package_version=__version__,
        git_commit=_detect_git_commit() if git_commit is None else git_commit,
        created_at=_utc_now() if created_at is None else created_at,
        platform=_profile_platform_summary(),
        seed=seed,
        dataset=DatasetRef(
            dataset_id="local-store-profile",
            kind="fixture",
            split="local",
            version="v1",
            metadata={
                "store_id": str(stats.store_id),
                "fixture_scale": True,
                "synthetic": False,
                "retention_policy": stats.retention_policy,
                "tombstone_count": stats.tombstone_count,
            },
        ),
        metrics=metrics,
        artifacts={
            "report_kind": "local-profile",
            "ground_truth_backend": "flat",
            "approx_backend": approx_backend_name,
        },
        caveats=tuple(caveats),
        passed=flat_recall == 1.0 and all(value >= 0.0 for value in query_latencies),
    )


def _visible_items(store: LocalStore) -> tuple[tuple[Cid, MemoryItem], ...]:
    tombstoned = _tombstoned_cids(store.manifest.retention_policy)
    return tuple(
        (cid, store._items[cid]) for cid in sorted(set(store._items) - tombstoned)
    )


def _tombstoned_cids(retention_policy: Mapping[str, object]) -> set[Cid]:
    raw_tombstones = retention_policy.get("tombstones", [])
    if not isinstance(raw_tombstones, list):
        return set()
    cids: set[Cid] = set()
    for raw in raw_tombstones:
        if not isinstance(raw, Mapping):
            continue
        content_id_hex = raw.get("content_id")
        if not isinstance(content_id_hex, str):
            continue
        try:
            cids.add(bytes.fromhex(content_id_hex))
        except ValueError:
            continue
    return cids


def _query_vectors(
    items: Sequence[tuple[Cid, MemoryItem]],
    query_count: int,
) -> tuple[SummaryVec, ...]:
    selected = items[: min(query_count, len(items))]
    return tuple(
        np.ascontiguousarray(item.key, dtype=np.float32) for _, item in selected
    )


def _flat_ground_truth(items: Sequence[tuple[Cid, MemoryItem]]) -> FlatIndex:
    flat = FlatIndex()
    flat.add_batch(_index_items(items))
    return flat


def _index_items(
    items: Sequence[tuple[Cid, MemoryItem]],
) -> tuple[tuple[Cid, SummaryVec], ...]:
    return tuple((cid, item.key) for cid, item in items)


def _mean_recall(
    expected: Sequence[Sequence[tuple[Cid, float]]],
    actual: Sequence[Sequence[tuple[Cid, float]]],
) -> float:
    recalls: list[float] = []
    for expected_row, actual_row in zip(expected, actual, strict=True):
        expected_ids = {cid for cid, _ in expected_row}
        actual_ids = {cid for cid, _ in actual_row}
        if not expected_ids:
            recalls.append(1.0)
        else:
            recalls.append(len(expected_ids & actual_ids) / len(expected_ids))
    if not recalls:
        return 1.0
    return float(sum(recalls) / len(recalls))


def _measure_ms(
    operation: Callable[[int], object],
    *,
    warmup_count: int,
    measurement_count: int,
) -> tuple[float, ...]:
    for index in range(warmup_count):
        operation(index)
    timings: list[float] = []
    for index in range(measurement_count):
        started = time.perf_counter_ns()
        operation(index)
        elapsed_ns = time.perf_counter_ns() - started
        timings.append(elapsed_ns / 1_000_000.0)
    return tuple(timings)


def _conditioning_input(
    retrieval: Retrieval,
) -> tuple[np.ndarray, Retrieval, CondCtx]:
    compatible = _compatible_retrieval(retrieval)
    first = compatible.items[0].value
    if not isinstance(first, Transition):
        raise EvaluationError("conditioning profile requires Transition values")
    current = _numpy_latent(first.z_src, "z_src")
    parametric = _numpy_latent(first.z_next, "z_next")
    return parametric, compatible, CondCtx(current)


def _compatible_retrieval(retrieval: Retrieval) -> Retrieval:
    if not retrieval.items:
        raise EvaluationError("conditioning profile requires non-empty retrievals")
    first_value = retrieval.items[0].value
    if not isinstance(first_value, Transition):
        raise EvaluationError("conditioning profile requires Transition values")
    expected_shape = _numpy_latent(first_value.delta, "delta").shape
    items: list[MemoryItem] = []
    distances: list[float] = []
    for item, distance in zip(retrieval.items, retrieval.distances, strict=True):
        value = item.value
        if not isinstance(value, Transition):
            continue
        if _numpy_latent(value.delta, "delta").shape == expected_shape:
            items.append(item)
            distances.append(distance)
    if not items:
        raise EvaluationError("conditioning profile found no shape-compatible items")
    return Retrieval(items=tuple(items), distances=tuple(distances))


def _condition_once(payload: tuple[np.ndarray, Retrieval, CondCtx]) -> object:
    parametric, retrieval, ctx = payload
    return KnnCorrector().condition(parametric, retrieval, ctx)


def _numpy_latent(value: object, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise DTypeError(f"{field_name} must be a numpy.ndarray")
    return np.ascontiguousarray(value)


def _latency_metrics(prefix: str, values: Sequence[float]) -> dict[str, EvalMetric]:
    if not values:
        raise EvaluationError(f"{prefix} requires at least one measurement")
    return {
        f"{prefix}_p50_ms": _percentile(values, 50.0),
        f"{prefix}_p95_ms": _percentile(values, 95.0),
        f"{prefix}_p99_ms": _percentile(values, 99.0),
        f"{prefix}_mean_ms": float(sum(values) / len(values)),
        f"{prefix}_max_ms": float(max(values)),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return ordered[lower_index]
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return float(lower + (upper - lower) * (rank - lower_index))


def _per_item(total: int, item_count: int) -> float:
    return float(total / item_count)


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationError(f"{field_name} must be a positive integer")


def _require_non_negative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{field_name} must be a non-negative integer")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _profile_platform_summary() -> dict[str, str]:
    return {
        "cpu_count": str(os.cpu_count() or "unknown"),
        "gpu": "none-detected",
        "machine": platform_module.machine() or "unknown",
        "memory": "unknown",
        "processor": platform_module.processor() or "unknown",
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


__all__ = ["run_profile_evaluation"]
