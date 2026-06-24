from __future__ import annotations

import numpy as np
import pytest

from mneme.core import (
    EncoderFingerprint,
    FingerprintMismatchError,
    Metric,
    QueryError,
    QuerySpec,
)
from mneme.index import (
    FlatIndex,
    apply_temporal_decay,
    deduplicate_results,
    planned_search_k,
    search_index,
)


class DuplicateIndex:
    def __init__(self, results: list[tuple[bytes, float]]) -> None:
        self.results = results
        self.calls: list[int] = []

    def add(self, cid: bytes, key: np.ndarray) -> None:
        raise NotImplementedError

    def add_batch(self, items: list[tuple[bytes, np.ndarray]]) -> None:
        raise NotImplementedError

    def search(
        self,
        q: np.ndarray,
        k: int,
        *,
        metric: Metric,
        ef: int | None = None,
    ) -> list[tuple[bytes, float]]:
        self.calls.append(k)
        return self.results[:k]

    def __len__(self) -> int:
        return len(self.results)


def _query(**overrides: object) -> QuerySpec:
    kwargs = {
        "vector": np.array([1.0, 0.0], dtype=np.float32),
        "k": 2,
        "metric": Metric.L2,
    } | overrides
    return QuerySpec(**kwargs)


def _fingerprint(summarizer_id: str = "summary") -> EncoderFingerprint:
    return EncoderFingerprint("encoder", summarizer_id, None, "blake3:config")


def test_invalid_query_construction_raises_query_error() -> None:
    with pytest.raises(QueryError, match="k must be >= 1"):
        _query(k=0)
    with pytest.raises(QueryError, match="ef must be None"):
        _query(k=3, ef=2)
    with pytest.raises(QueryError, match="metric must be a Metric"):
        _query(metric="l2")
    with pytest.raises(QueryError, match="reserved"):
        _query(filters={"schema_version": "mneme.query_filter.v1"})
    with pytest.raises(QueryError, match="JSON-compatible"):
        _query(filters={"source": object()})


def test_search_index_de_duplicates_stably() -> None:
    index = DuplicateIndex([(b"a", 0.1), (b"a", 0.0), (b"b", 0.2)])

    result = search_index(index, _query(k=2))

    assert result == [(b"a", 0.1), (b"b", 0.2)]


def test_deduplicate_results_preserves_first_occurrence() -> None:
    assert deduplicate_results([(b"b", 0.2), (b"a", 0.1), (b"b", 0.0)]) == [
        (b"b", 0.2),
        (b"a", 0.1),
    ]


def test_planned_search_k_overfetches_for_filters() -> None:
    unfiltered = _query(k=3, ef=5)
    filtered = _query(k=3, ef=5, filters={"source": "fixture"})

    assert planned_search_k(unfiltered) == 3
    assert planned_search_k(filtered) == 12
    assert planned_search_k(filtered, overfetch_multiplier=2) == 6


def test_search_index_applies_filter_predicate_after_overfetch() -> None:
    index = DuplicateIndex([(b"a", 0.1), (b"b", 0.2), (b"c", 0.3), (b"d", 0.4)])
    spec = _query(k=2, filters={"source": "fixture"})

    result = search_index(index, spec, filter_predicate=lambda cid: cid in {b"b", b"c"})

    assert index.calls == [8]
    assert result == [(b"b", 0.2), (b"c", 0.3)]


def test_temporal_decay_matches_hand_computed_ordering() -> None:
    results = [(b"a", 0.1), (b"b", 0.2), (b"c", 0.0)]
    timestamps = {b"a": 0.0, b"b": 9.0, b"c": 2.0}

    decayed = apply_temporal_decay(results, decay=0.05, timestamps=timestamps, now=10.0)

    assert decayed == [(b"b", 0.25), (b"c", 0.4), (b"a", 0.6)]


def test_search_index_applies_temporal_decay_before_final_top_k() -> None:
    index = DuplicateIndex([(b"a", 0.1), (b"b", 0.2), (b"c", 0.0)])
    spec = _query(k=2, temporal_decay=0.05)

    result = search_index(
        index,
        spec,
        timestamps={b"a": 0.0, b"b": 9.0, b"c": 2.0},
        now=10.0,
    )

    assert result == [(b"b", 0.25), (b"c", 0.4)]


def test_temporal_decay_requires_timestamps_and_now() -> None:
    with pytest.raises(QueryError, match="timestamps and now"):
        apply_temporal_decay([(b"a", 0.1)], decay=0.1, timestamps=None, now=1.0)
    with pytest.raises(QueryError, match="missing timestamp"):
        apply_temporal_decay([(b"a", 0.1)], decay=0.1, timestamps={}, now=1.0)


def test_query_fingerprint_mismatch_fails_closed() -> None:
    spec = _query(encoder_fp=_fingerprint("left"))

    with pytest.raises(FingerprintMismatchError, match="does not match"):
        search_index(DuplicateIndex([]), spec, index_fingerprint=_fingerprint("right"))


def test_search_index_with_flat_index_uses_query_spec() -> None:
    index = FlatIndex()
    index.add(b"a", np.array([1.0, 0.0], dtype=np.float32))
    index.add(b"b", np.array([0.0, 1.0], dtype=np.float32))

    assert search_index(index, _query(k=1)) == [(b"a", 0.0)]
