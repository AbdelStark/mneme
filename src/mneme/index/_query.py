"""QuerySpec post-search semantics shared by index backends and stores."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import SupportsFloat

from mneme.core import (
    Cid,
    EncoderFingerprint,
    FingerprintMismatchError,
    QueryError,
    QuerySpec,
)
from mneme.index._protocols import Index
from mneme.observability import ObservabilityConfig, emit_event

FilterPredicate = Callable[[Cid], bool]


def planned_search_k(
    spec: QuerySpec,
    *,
    has_filters: bool = False,
    overfetch_multiplier: int = 4,
) -> int:
    """Return the backend k to request before store-level filtering."""

    _validate_query_spec(spec)
    if (
        isinstance(overfetch_multiplier, bool)
        or not isinstance(overfetch_multiplier, int)
        or overfetch_multiplier < 1
    ):
        raise QueryError("overfetch_multiplier must be >= 1")
    breadth = max(spec.k, spec.ef or spec.k)
    if has_filters or spec.filters:
        return max(breadth, spec.k * overfetch_multiplier)
    return spec.k


def search_index(
    index: Index,
    spec: QuerySpec,
    *,
    index_fingerprint: EncoderFingerprint | None = None,
    filter_predicate: FilterPredicate | None = None,
    timestamps: Mapping[Cid, float] | None = None,
    now: float | None = None,
    overfetch_multiplier: int = 4,
) -> list[tuple[Cid, float]]:
    """Search an index and apply QuerySpec post-search semantics."""

    _validate_query_spec(spec)
    if (
        spec.encoder_fp is not None
        and index_fingerprint is not None
        and spec.encoder_fp != index_fingerprint
    ):
        raise FingerprintMismatchError("query fingerprint does not match index")

    index_size = len(index)
    fetch_k = planned_search_k(
        spec,
        has_filters=filter_predicate is not None,
        overfetch_multiplier=overfetch_multiplier,
    )
    if spec.temporal_decay is not None and index_size > 0:
        fetch_k = max(fetch_k, index_size)

    while True:
        raw = index.search(spec.vector, fetch_k, metric=spec.metric, ef=spec.ef)
        results, duplicate_count = _deduplicate_results_with_count(raw)
        _emit_duplicate_warning(index, raw, results, duplicate_count)
        if filter_predicate is not None:
            results = [result for result in results if filter_predicate(result[0])]
        if spec.temporal_decay is not None:
            results = apply_temporal_decay(
                results,
                decay=spec.temporal_decay,
                timestamps=timestamps,
                now=now,
            )
        if len(results) >= spec.k or index_size == 0 or fetch_k >= index_size:
            return results[: spec.k]
        fetch_k = min(index_size, max(fetch_k + 1, fetch_k * 2))


def deduplicate_results(results: list[tuple[Cid, float]]) -> list[tuple[Cid, float]]:
    """Remove duplicate ids while preserving first occurrence order."""

    deduped, _ = _deduplicate_results_with_count(results)
    return deduped


def _deduplicate_results_with_count(
    results: list[tuple[Cid, float]],
) -> tuple[list[tuple[Cid, float]], int]:
    deduped: list[tuple[Cid, float]] = []
    seen: set[Cid] = set()
    duplicate_count = 0
    for cid, distance in results:
        distance_value = _finite_float(distance, "distance")
        if cid in seen:
            duplicate_count += 1
            continue
        seen.add(cid)
        deduped.append((cid, distance_value))
    return deduped, duplicate_count


def _emit_duplicate_warning(
    index: Index,
    raw: list[tuple[Cid, float]],
    deduped: list[tuple[Cid, float]],
    duplicate_count: int,
) -> None:
    if duplicate_count <= 0:
        return
    observability = getattr(index, "observability", None)
    if not isinstance(observability, ObservabilityConfig):
        return
    emit_event(
        observability,
        event="mneme.index.search",
        operation="index.search.deduplicate",
        status="warning",
        started=None,
        backend=getattr(index, "backend", type(index).__name__),
        raw_hit_count=len(raw),
        hit_count=len(deduped),
        duplicate_result_count=duplicate_count,
    )


def apply_temporal_decay(
    results: list[tuple[Cid, float]],
    *,
    decay: float,
    timestamps: Mapping[Cid, float] | None,
    now: float | None,
) -> list[tuple[Cid, float]]:
    """Apply deterministic age penalty and return sorted results."""

    if timestamps is None or now is None:
        raise QueryError("timestamps and now are required for temporal decay")
    decay_value = _finite_float(decay, "temporal_decay")
    now_value = _finite_float(now, "now")
    decayed: list[tuple[Cid, float]] = []
    for cid, distance in results:
        if cid not in timestamps:
            raise QueryError(f"missing timestamp for result id {cid.hex()}")
        distance_value = _finite_float(distance, "distance")
        timestamp_value = _finite_float(timestamps[cid], "timestamp")
        age = max(0.0, now_value - timestamp_value)
        score = distance_value + decay_value * age
        decayed.append((cid, _finite_float(score, "decayed distance")))
    decayed.sort(key=lambda item: (item[1], item[0]))
    return decayed


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise QueryError(f"{field_name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise QueryError(f"{field_name} must be a finite number")
    return numeric


def _validate_query_spec(spec: QuerySpec) -> None:
    if not isinstance(spec, QuerySpec):
        raise QueryError("spec must be a QuerySpec")


__all__ = [
    "FilterPredicate",
    "apply_temporal_decay",
    "deduplicate_results",
    "planned_search_k",
    "search_index",
]
