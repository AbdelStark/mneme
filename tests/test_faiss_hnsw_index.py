from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mneme.core import Metric, OptionalDependencyError, QueryError
from mneme.index import FaissHnswIndex, FlatIndex, Index, create_index_backend
from mneme.store import init_store


def _vec(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def test_faiss_hnsw_index_matches_flat_l2_order_with_fake_faiss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_faiss(monkeypatch)
    flat = FlatIndex()
    faiss_index = FaissHnswIndex(m=4, ef_construction=8, ef_search=5)
    for cid, key in [(b"b", _vec([1.0, 0.0])), (b"a", _vec([0.0, 1.0]))]:
        flat.add(cid, key)
        faiss_index.add(cid, key)

    result = faiss_index.search(_vec([0.9, 0.1]), 2, metric=Metric.L2)
    expected = flat.search(_vec([0.9, 0.1]), 2, metric=Metric.L2)

    assert isinstance(faiss_index, Index)
    assert [cid for cid, _ in result] == [cid for cid, _ in expected]
    assert [distance for _, distance in result] == pytest.approx(
        [distance for _, distance in expected]
    )


def test_faiss_hnsw_index_maps_inner_product_and_cosine_distances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_faiss(monkeypatch)
    index = FaissHnswIndex(m=4)
    index.add(b"x", _vec([1.0, 0.0]))
    index.add(b"y", _vec([0.0, 1.0]))

    inner = index.search(_vec([2.0, 0.0]), 2, metric=Metric.INNER_PRODUCT)
    cosine = index.search(_vec([1.0, 0.0]), 2, metric=Metric.COSINE)

    assert inner == [(b"x", -2.0), (b"y", -0.0)]
    assert cosine == [(b"x", 0.0), (b"y", 1.0)]


def test_faiss_hnsw_missing_dependency_raises_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "faiss", raising=False)
    original_import = importlib.import_module

    def missing_faiss(name: str, package: str | None = None) -> Any:
        if name == "faiss":
            raise ImportError("missing faiss")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_faiss)

    with pytest.raises(OptionalDependencyError) as raised:
        FaissHnswIndex()

    assert raised.value.extra == "index"
    assert raised.value.package == "faiss-cpu"


def test_store_init_fails_before_manifest_when_faiss_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "faiss", raising=False)
    original_import = importlib.import_module

    def missing_faiss(name: str, package: str | None = None) -> Any:
        if name == "faiss":
            raise ImportError("missing faiss")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_faiss)
    root = tmp_path / "store"

    with pytest.raises(OptionalDependencyError):
        init_store(root, index_backend="faiss_hnsw")

    assert not (root / "manifest.json").exists()


def test_faiss_hnsw_backend_params_and_package_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_faiss(monkeypatch)

    backend = create_index_backend(
        "faiss_hnsw",
        {"m": 8, "ef_construction": 12, "ef_search": 10},
    )

    assert isinstance(backend, FaissHnswIndex)
    with pytest.raises(QueryError, match="unsupported faiss_hnsw params"):
        create_index_backend("faiss_hnsw", {"unknown": 1})


def test_project_index_extra_declares_faiss_cpu() -> None:
    import tomllib

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "faiss-cpu>=1.8" in pyproject["project"]["optional-dependencies"]["index"]


def test_public_api_docs_describe_faiss_optional_boundary() -> None:
    public_api = Path("docs/spec/02-public-api.md").read_text(encoding="utf-8")

    assert "FaissHnswIndex" in public_api
    assert "faiss_hnsw" in public_api
    assert "OptionalDependencyError" in public_api
    assert "faiss-cpu" in public_api


def _install_fake_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(
        METRIC_INNER_PRODUCT=0,
        METRIC_L2=1,
        IndexHNSWFlat=_FakeIndexHNSWFlat,
        IndexIDMap2=_FakeIndexIDMap2,
    )
    monkeypatch.setitem(sys.modules, "faiss", fake)


class _FakeIndexHNSWFlat:
    def __init__(self, dim: int, m: int, metric: int) -> None:
        self.dim = dim
        self.m = m
        self.metric = metric
        self.hnsw = SimpleNamespace(efConstruction=None, efSearch=None)


class _FakeIndexIDMap2:
    def __init__(self, index: _FakeIndexHNSWFlat) -> None:
        self.index = index
        self._vectors = np.empty((0, index.dim), dtype=np.float32)
        self._ids = np.empty((0,), dtype=np.int64)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._ids = np.asarray(ids, dtype=np.int64)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        all_distances: list[list[float]] = []
        all_ids: list[list[int]] = []
        for query in np.asarray(queries, dtype=np.float32):
            scored: list[tuple[float, int]] = []
            for vector, item_id in zip(self._vectors, self._ids, strict=True):
                if self.index.metric == 1:
                    score = float(np.sum((vector - query) ** 2))
                    scored.append((score, int(item_id)))
                else:
                    score = float(np.dot(vector, query))
                    scored.append((-score, int(item_id)))
            scored.sort()
            selected = scored[:k]
            if self.index.metric == 1:
                all_distances.append([score for score, _ in selected])
            else:
                all_distances.append([-score for score, _ in selected])
            all_ids.append([item_id for _, item_id in selected])
        return (
            np.asarray(all_distances, dtype=np.float32),
            np.asarray(all_ids, dtype=np.int64),
        )
