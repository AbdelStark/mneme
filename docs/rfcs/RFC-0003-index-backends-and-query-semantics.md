# RFC-0003: Index Backends and Query Semantics

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme exposes a backend-neutral index protocol with a required flat exact backend and an approximate backend. Query semantics are defined in `QuerySpec`: vector, k, metric, search breadth, filters, temporal decay, receipt flag, and optional fingerprint. Exact search is the correctness reference; approximate search is an acceleration structure.

## Motivation

[Performance Budget](../spec/08-performance-budget.md) requires latency characterization at different store sizes, and [Data Model](../spec/03-data-model.md#queryspec) defines query invariants. The PRD selects approximate nearest-neighbor search for scalable retrieval but also needs exact ground truth for recall tests.

## Goals

- Provide a common `Index` protocol.
- Require a flat exact backend for correctness tests.
- Provide one approximate backend for v0.1.
- Define stable ordering, duplicate handling, filtering, and temporal decay.
- Make recall measurement part of the contract.

## Non-Goals

- Prove approximate top-k correctness.
- Support deletes from all indexes in v0.1.
- Hide backend recall and latency trade-offs.

## Proposed Design

Protocol:

```python
class Index(Protocol):
    def add(self, cid: Cid, key: SummaryVec) -> None: ...
    def add_batch(self, items: Sequence[tuple[Cid, SummaryVec]]) -> None: ...
    def search(
        self,
        q: SummaryVec,
        k: int,
        *,
        metric: Metric,
        ef: int | None = None,
    ) -> list[tuple[Cid, float]]: ...
    def __len__(self) -> int: ...
```

Backends:

- `FlatIndex`: exact search in NumPy, required in core tests.
- `ApproxIndex`: first approximate backend selected during implementation, behind an optional extra.

Search returns ids and distances sorted by ascending distance for cosine and L2, and descending score converted to a distance for inner product. Ties are broken by content id bytes to make results stable.

Filtering occurs in the store, not the index, unless a backend can support equivalent pre-filtering. The store may over-fetch from the index to satisfy filters. Temporal decay is applied after raw distances are returned and before final top-k truncation.

Empty index search returns an empty result. Invalid query vectors, `k < 1`, unsupported metric, or `ef < k` raise `QueryError`.

## Alternatives Considered

- Use only an approximate backend: faster to ship at scale, but lacks exact recall ground truth.
- Put filters in every index backend: efficient for some backends, but complicates the protocol and slows v0.1.
- Return backend-native distances directly: preserves details but breaks conditioner portability.
- Treat inner product as a separate score ordering: leaks backend differences into callers.

## Drawbacks

- Store-level filtering may require over-fetching and can be inefficient for selective filters.
- Exact flat search does not scale, but it is intentionally a reference backend.
- Approximate backend choice may need platform-specific handling.

## Migration / Rollout

v0.1 implements `FlatIndex` and one approximate backend. The manifest records backend name and parameters so indexes can be rebuilt or migrated. Later backends must pass the same conformance tests.

## Testing Strategy

- Exact nearest-neighbor tests with known vectors.
- Approximate recall@k measured against `FlatIndex`.
- Stable tie ordering by content id.
- Query validation tests for k, metric, ef, vector shape, dtype, and non-finite values.
- Filter and temporal decay integration tests at the store layer.

## Open Questions

- OPEN QUESTION: Which approximate backend is selected as the default optional backend for v0.1. Owner: maintainer. Target: v0.1 index implementation.

## References

- [Performance Budget](../spec/08-performance-budget.md)
- [Testing Strategy](../spec/07-testing-strategy.md#index)
- [PRD Section 9.1](../../prd.md#91-index)
