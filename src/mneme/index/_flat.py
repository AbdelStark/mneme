"""Exact NumPy flat index backend."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from mneme.core import Cid, DTypeError, Metric, QueryError, ShapeError, SummaryVec
from mneme.core._ids import require_cid_bytes
from mneme.observability import (
    ObservabilityConfig,
    distance_mean,
    distance_min,
    emit_event,
    start_event_timer,
)

_COSINE_NORM_TOLERANCE = 1e-4


class FlatIndex:
    """Exact in-memory index used as the recall ground truth."""

    def __init__(self, observability: ObservabilityConfig | None = None) -> None:
        self.observability = observability
        self._keys: dict[Cid, SummaryVec] = {}
        self._dim: int | None = None

    def add(self, cid: Cid, key: SummaryVec) -> None:
        _validate_cid(cid)
        vector = _validate_vector(key, field_name="key")
        self._validate_or_set_dim(vector)
        self._keys[cid] = np.ascontiguousarray(vector, dtype=np.float32)

    def add_batch(self, items: Sequence[tuple[Cid, SummaryVec]]) -> None:
        for cid, key in items:
            self.add(cid, key)

    def search(
        self,
        q: SummaryVec,
        k: int,
        *,
        metric: Metric,
        ef: int | None = None,
    ) -> list[tuple[Cid, float]]:
        started = start_event_timer(self.observability)
        try:
            results = self._search(q, k, metric=metric, ef=ef)
        except Exception as exc:
            if started is not None:
                emit_event(
                    self.observability,
                    event="mneme.index.search",
                    operation="index.search",
                    status="error",
                    started=started,
                    error=exc,
                    backend="flat",
                    metric=str(metric),
                    k=k,
                    ef=ef,
                    index_size=len(self._keys),
                )
            raise
        if started is not None:
            distances = tuple(distance for _, distance in results)
            emit_event(
                self.observability,
                event="mneme.index.search",
                operation="index.search",
                status="ok",
                started=started,
                backend="flat",
                metric=str(metric),
                k=k,
                ef=ef,
                index_size=len(self._keys),
                hit_count=len(results),
                distance_min=distance_min(distances),
                distance_mean=distance_mean(distances),
            )
        return results

    def _search(
        self,
        q: SummaryVec,
        k: int,
        *,
        metric: Metric,
        ef: int | None = None,
    ) -> list[tuple[Cid, float]]:
        _validate_k(k)
        if not isinstance(metric, Metric):
            raise QueryError("metric must be a Metric")
        if ef is not None and (
            isinstance(ef, bool) or not isinstance(ef, int) or ef < k
        ):
            raise QueryError("ef must be None or an integer greater than or equal to k")
        query = _validate_vector(q, field_name="query")
        if self._dim is not None and query.shape[0] != self._dim:
            raise QueryError(
                f"query dimension {query.shape[0]} "
                f"does not match index dimension {self._dim}"
            )
        if not self._keys:
            return []

        if metric is Metric.COSINE:
            _validate_unit_norm(query, "query")
            scored = [
                (cid, _cosine_distance(query, _require_unit_key(cid, key)))
                for cid, key in self._keys.items()
            ]
            scored.sort(key=lambda item: (item[1], item[0]))
        elif metric is Metric.L2:
            scored = [
                (cid, _l2_distance(query, key)) for cid, key in self._keys.items()
            ]
            scored.sort(key=lambda item: (item[1], item[0]))
        else:
            scored = [
                (cid, _inner_product_distance(query, key))
                for cid, key in self._keys.items()
            ]
            scored.sort(key=lambda item: (item[1], item[0]))
        return scored[:k]

    def __len__(self) -> int:
        return len(self._keys)

    def _validate_or_set_dim(self, vector: SummaryVec) -> None:
        dim = int(vector.shape[0])
        if self._dim is None:
            self._dim = dim
        elif dim != self._dim:
            raise ShapeError(
                f"key dimension {dim} does not match index dimension {self._dim}"
            )


def _validate_cid(cid: object) -> None:
    require_cid_bytes(cid, "cid", type_error=TypeError, value_error=ValueError)


def _validate_vector(value: object, *, field_name: str) -> SummaryVec:
    if not isinstance(value, np.ndarray):
        raise QueryError(f"{field_name} must be a numpy.ndarray")
    if value.dtype != np.float32:
        raise DTypeError(f"{field_name} must have dtype float32")
    if value.ndim != 1:
        raise ShapeError(f"{field_name} must be one-dimensional")
    if value.shape[0] <= 0:
        raise ShapeError(f"{field_name} must not be empty")
    if not value.flags.c_contiguous:
        raise ShapeError(f"{field_name} must be contiguous")
    if not bool(np.isfinite(value).all()):
        raise QueryError(f"{field_name} must contain only finite values")
    return value


def _validate_k(k: object) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise QueryError("k must be an integer")
    if k < 1:
        raise QueryError("k must be >= 1")


def _validate_unit_norm(vector: SummaryVec, field_name: str) -> None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or abs(norm - 1.0) > _COSINE_NORM_TOLERANCE:
        raise QueryError(f"{field_name} must be L2-normalized for cosine search")


def _require_unit_key(cid: Cid, key: SummaryVec) -> SummaryVec:
    try:
        _validate_unit_norm(key, f"key {cid.hex()}")
    except QueryError as exc:
        raise QueryError("stored key must be L2-normalized for cosine search") from exc
    return key


def _cosine_distance(left: SummaryVec, right: SummaryVec) -> float:
    return float(1.0 - float(np.dot(left, right)))


def _l2_distance(left: SummaryVec, right: SummaryVec) -> float:
    return float(np.linalg.norm(left - right))


def _inner_product_distance(left: SummaryVec, right: SummaryVec) -> float:
    return float(-float(np.dot(left, right)))


__all__ = ["FlatIndex"]
