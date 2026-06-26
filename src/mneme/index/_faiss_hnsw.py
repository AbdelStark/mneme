"""Optional FAISS HNSW index backend."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mneme.core import (
    Cid,
    DTypeError,
    Metric,
    OptionalDependencyError,
    QueryError,
    ShapeError,
    SummaryVec,
)
from mneme.core._ids import require_cid_bytes
from mneme.index._protocols import Index
from mneme.observability import ObservabilityConfig

_COSINE_NORM_TOLERANCE = 1e-4


class FaissHnswIndex:
    """Approximate FAISS HNSW backend behind the ``index`` extra."""

    backend = "faiss_hnsw"

    def __init__(
        self,
        *,
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int | None = None,
        observability: ObservabilityConfig | None = None,
    ) -> None:
        self._faiss = _load_faiss()
        self.observability = observability
        self.m = _positive_int(m, "m")
        self.ef_construction = _positive_int(ef_construction, "ef_construction")
        self.ef_search = (
            None
            if ef_search is None
            else _positive_int(
                ef_search,
                "ef_search",
            )
        )
        self._keys: dict[Cid, SummaryVec] = {}
        self._dim: int | None = None
        self._metric: Metric | None = None
        self._index: Any | None = None
        self._ordinal_to_cid: dict[int, Cid] = {}
        self._dirty = True

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None,
        *,
        observability: ObservabilityConfig | None = None,
    ) -> FaissHnswIndex:
        """Build a backend from manifest-style index parameters."""

        params = {} if params is None else dict(params)
        allowed = {"m", "ef_construction", "ef_search"}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise QueryError(f"unsupported faiss_hnsw params: {', '.join(unknown)}")
        return cls(
            m=params.get("m", 32),
            ef_construction=params.get("ef_construction", 40),
            ef_search=params.get("ef_search"),
            observability=observability,
        )

    def add(self, cid: Cid, key: SummaryVec) -> None:
        _validate_cid(cid)
        vector = _validate_vector(key, field_name="key")
        self._validate_or_set_dim(vector)
        self._keys[cid] = np.ascontiguousarray(vector, dtype=np.float32)
        self._dirty = True

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
            for cid, key in self._keys.items():
                _require_unit_key(cid, key)

        self._ensure_index(metric)
        assert self._index is not None
        search_k = min(k, len(self._keys))
        search_ef = ef or self.ef_search
        if search_ef is not None and hasattr(self._index, "index"):
            self._index.index.hnsw.efSearch = max(search_ef, search_k)
        raw_distances, raw_labels = self._index.search(
            query.reshape(1, -1),
            search_k,
        )
        results: list[tuple[Cid, float]] = []
        for label, distance in zip(raw_labels[0], raw_distances[0], strict=True):
            ordinal = int(label)
            if ordinal < 0:
                continue
            cid = self._ordinal_to_cid[ordinal]
            results.append((cid, _portable_distance(float(distance), metric)))
        results.sort(key=lambda item: (item[1], item[0]))
        return results[:k]

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

    def _ensure_index(self, metric: Metric) -> None:
        if not self._dirty and self._metric is metric:
            return
        assert self._dim is not None
        faiss_metric = (
            self._faiss.METRIC_L2
            if metric is Metric.L2
            else self._faiss.METRIC_INNER_PRODUCT
        )
        base = self._faiss.IndexHNSWFlat(self._dim, self.m, faiss_metric)
        base.hnsw.efConstruction = self.ef_construction
        if self.ef_search is not None:
            base.hnsw.efSearch = self.ef_search
        index = self._faiss.IndexIDMap2(base)
        ordered = sorted(self._keys.items(), key=lambda item: item[0])
        vectors = np.stack([key for _, key in ordered]).astype(np.float32, copy=False)
        ids = np.arange(len(ordered), dtype=np.int64)
        index.add_with_ids(vectors, ids)
        self._ordinal_to_cid = {int(idx): cid for idx, (cid, _) in enumerate(ordered)}
        self._index = index
        self._metric = metric
        self._dirty = False


def create_index_backend(
    backend: str = "flat",
    params: Mapping[str, Any] | None = None,
    *,
    observability: Any | None = None,
) -> Index:
    """Create an index backend from a manifest backend name and params."""

    from mneme.core import IndexUnavailableError
    from mneme.index._flat import FlatIndex

    if backend == "flat":
        return FlatIndex(observability=observability)
    if backend == FaissHnswIndex.backend:
        return FaissHnswIndex.from_params(params, observability=observability)
    raise IndexUnavailableError(f"unsupported index backend: {backend}")


def _load_faiss() -> Any:
    try:
        return importlib.import_module("faiss")
    except ImportError as exc:
        raise OptionalDependencyError(
            "FAISS HNSW index requires the 'index' extra",
            extra="index",
            package="faiss-cpu",
        ) from exc


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


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QueryError(f"{field_name} must be a positive integer")
    return value


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


def _portable_distance(raw_distance: float, metric: Metric) -> float:
    if metric is Metric.L2:
        return math.sqrt(max(raw_distance, 0.0))
    if metric is Metric.COSINE:
        return 1.0 - raw_distance
    return -raw_distance


__all__ = ["FaissHnswIndex", "create_index_backend"]
