from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from mneme.core import DTypeError, Metric, QueryError, ShapeError
from mneme.index import FlatIndex, Index


def _vec(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _cid(rank: int) -> bytes:
    return bytes([rank]) * 32


def test_flat_index_l2_known_vectors() -> None:
    index = FlatIndex()
    cid_a = _cid(1)
    cid_b = _cid(2)
    cid_c = _cid(3)
    index.add(cid_b, _vec([1.0, 0.0]))
    index.add(cid_a, _vec([0.0, 1.0]))
    index.add(cid_c, _vec([2.0, 0.0]))

    result = index.search(_vec([0.9, 0.1]), 2, metric=Metric.L2)

    assert isinstance(index, Index)
    assert len(index) == 3
    assert [cid for cid, _ in result] == [cid_b, cid_c]


def test_flat_index_cosine_known_vectors() -> None:
    index = FlatIndex()
    cid_x = _cid(1)
    cid_y = _cid(2)
    index.add(cid_x, _vec([1.0, 0.0]))
    index.add(cid_y, _vec([0.0, 1.0]))

    result = index.search(_vec([1.0, 0.0]), 2, metric=Metric.COSINE)

    assert result[0] == (cid_x, 0.0)
    assert result[1][0] == cid_y
    assert result[1][1] == pytest.approx(1.0)


def test_flat_index_inner_product_orders_by_descending_score() -> None:
    index = FlatIndex()
    cid_low = _cid(1)
    cid_high = _cid(2)
    index.add(cid_low, _vec([1.0, 0.0]))
    index.add(cid_high, _vec([3.0, 0.0]))

    result = index.search(_vec([2.0, 0.0]), 2, metric=Metric.INNER_PRODUCT)

    assert result == [(cid_high, -6.0), (cid_low, -2.0)]


def test_flat_index_ties_are_ordered_by_content_id_bytes() -> None:
    index = FlatIndex()
    cid_a = _cid(1)
    cid_b = _cid(2)
    index.add(cid_b, _vec([1.0, 0.0]))
    index.add(cid_a, _vec([1.0, 0.0]))

    result = index.search(_vec([1.0, 0.0]), 2, metric=Metric.L2)

    assert [cid for cid, _ in result] == [cid_a, cid_b]


def test_flat_index_empty_search_returns_empty_result() -> None:
    assert FlatIndex().search(_vec([1.0, 0.0]), 1, metric=Metric.L2) == []


def test_flat_index_add_batch_and_replace_existing_id() -> None:
    index = FlatIndex()
    cid_a = _cid(1)
    cid_b = _cid(2)
    index.add_batch([(cid_a, _vec([1.0, 0.0])), (cid_b, _vec([0.0, 1.0]))])
    index.add(cid_a, _vec([2.0, 0.0]))

    result = index.search(_vec([2.0, 0.0]), 2, metric=Metric.L2)

    assert len(index) == 2
    assert result[0] == (cid_a, 0.0)


@pytest.mark.parametrize(
    ("query", "error_type", "match"),
    [
        (np.array([1.0], dtype=np.float64), DTypeError, "float32"),
        (np.array([[1.0]], dtype=np.float32), ShapeError, "one-dimensional"),
        (np.array([np.nan], dtype=np.float32), QueryError, "finite"),
    ],
)
def test_flat_index_invalid_query_vectors_raise_typed_errors(
    query: object, error_type: type[Exception], match: str
) -> None:
    with pytest.raises(error_type, match=match):
        FlatIndex().search(query, 1, metric=Metric.L2)


def test_flat_index_rejects_bad_k_metric_ef_and_dimensions() -> None:
    index = FlatIndex()
    index.add(_cid(1), _vec([1.0, 0.0]))

    with pytest.raises(QueryError, match="k must be >= 1"):
        index.search(_vec([1.0, 0.0]), 0, metric=Metric.L2)
    with pytest.raises(QueryError, match="metric must be a Metric"):
        index.search(_vec([1.0, 0.0]), 1, metric="l2")
    with pytest.raises(QueryError, match="ef must be None"):
        index.search(_vec([1.0, 0.0]), 2, metric=Metric.L2, ef=1)
    with pytest.raises(QueryError, match="dimension"):
        index.search(_vec([1.0, 0.0, 0.0]), 1, metric=Metric.L2)


def test_flat_index_rejects_non_normalized_cosine_vectors() -> None:
    index = FlatIndex()
    index.add(_cid(1), _vec([2.0, 0.0]))

    with pytest.raises(QueryError, match="L2-normalized"):
        index.search(_vec([1.0, 0.0]), 1, metric=Metric.COSINE)


def test_flat_index_rejects_non_digest_content_ids() -> None:
    with pytest.raises(ValueError, match="cid must be 32 bytes"):
        FlatIndex().add(b"a", _vec([1.0, 0.0]))


def test_flat_index_import_does_not_load_optional_backends() -> None:
    script = (
        "import sys; "
        "import mneme.index; "
        "blocked = {'faiss', 'torch', 'cryptography', 'pydantic'}; "
        "loaded = sorted(blocked & set(sys.modules)); "
        "print(','.join(loaded)); "
        "raise SystemExit(1 if loaded else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
