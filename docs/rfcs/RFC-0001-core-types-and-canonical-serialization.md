# RFC-0001: Core Types and Canonical Serialization

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme defines immutable core data carriers and a deterministic canonical serialization format for memory items, queries, retrievals, manifests, receipts, and evaluation reports. This RFC locks the identity rules that make content ids, persistence, receipt verification, and replay possible.

## Motivation

[SPEC.md](https://github.com/AbdelStark/mneme/blob/main/SPEC.md#scope) requires a reusable memory layer with auditable stored transitions. [Data Model](../spec/03-data-model.md#memoryitem) requires every memory item to have a content id derived from canonical bytes. Without a canonical format, two processes can disagree about the same item identity and receipts cannot be verified reliably.

## Goals

- Define stable dataclass-style public carriers for v0.1.
- Define deterministic canonical serialization for content-id inputs.
- Require schema versions for all persisted objects.
- Make serialization independent of Python process state and dictionary insertion order.
- Keep optional ML and index backends out of core imports.

## Non-Goals

- Define every future value kind.
- Define the remote wire protocol; RFC-0008 covers it.
- Define cryptographic receipt structures; RFC-0007 covers them.
- Optimize binary tensor storage for all large-scale cases in v0.1.

## Proposed Design

Core objects live in `mneme.core` and are frozen dataclasses or enums. Public constructors validate and normalize inputs before objects enter the store.

Required v0.1 carriers:

```python
@dataclass(frozen=True)
class EncoderFingerprint:
    encoder_id: str
    summarizer_id: str
    weights_digest: str | None
    config_digest: str
    schema_version: str = "mneme.encoder_fingerprint.v1"

@dataclass(frozen=True)
class Transition:
    z_src: Latent
    action: np.ndarray
    z_next: Latent
    delta: Latent
    t: int
    episode_id: UUID
    reward: float | None = None

@dataclass(frozen=True)
class MemoryItem:
    content_id: Cid | None
    key: SummaryVec
    value: Transition
    meta: Mapping[str, Any]
    encoder_fp: EncoderFingerprint
    schema_version: str = "mneme.memory_item.v1"
```

Canonical serialization uses a deterministic tagged binary format:

- strings are UTF-8 with byte length prefixes;
- integers are signed big-endian two's-complement with minimal width tags;
- floats are IEEE 754 little-endian `float64` for scalar metadata;
- arrays include backend tag, dtype, shape, byte order, and contiguous raw bytes;
- mappings are sorted by normalized UTF-8 key bytes;
- dataclasses serialize fields in schema-declared order, not `__dict__` order;
- optional values serialize as explicit `none` or `some` tags;
- UUIDs serialize as 16 raw bytes.

`content_id(item)` computes the digest over:

```text
schema_version || encoder_fp || key || value_kind || value || normalized_meta
```

The `content_id` field itself is excluded from the digest. `build_item` fills it after validation. Content IDs are 32-byte BLAKE3 digests at public, persisted, and remote boundaries.

Core validation rejects non-finite keys, unsupported dtypes, negative transition steps, invalid schema major versions, and metadata that cannot be normalized. Serialization functions are pure and do not read files, environment variables, or global state.

## Alternatives Considered

- JSON-only serialization: easy to inspect, but inefficient and ambiguous for binary arrays, dtype, byte order, and NaN handling.
- Pickle: preserves Python objects, but is unsafe for untrusted data and unstable as a public persistence contract.
- Arrow or HDF5 as the only canonical layer: mature, but heavier than needed for content-id bytes and harder to keep as a minimal core dependency.
- Hash only value bytes and ignore metadata: simpler, but metadata and fingerprints affect retrieval safety and audit meaning.

## Drawbacks

- A custom canonical format adds implementation and test surface.
- Array byte-order normalization must be handled carefully.
- Users cannot insert arbitrary Python metadata without normalization.

## Migration / Rollout

v0.1 implements schema `v1` only and rejects unknown major versions. Minor schema migrations are registered in `mneme.core.migrations`. Future schema changes must include golden serialization fixtures.

## Testing Strategy

- Golden bytes and content ids for representative memory items.
- Property tests for mapping-order independence.
- Cross-process test that content ids match after restart.
- Validation tests for NaN, dtype, shape, metadata, UUID, and schema errors.
- Import-boundary test proving `import mneme.core` does not import optional ML or index backends.

## Resolved Bootstrap Decisions

- Value tensor persistence uses a separate chunked value-log format, not the small-object canonical serialization format. Each record has a canonical header and normalized numeric payload bytes, so content ids remain deterministic without forcing large tensors through a metadata-heavy object encoder.

## References

- [Data Model](../spec/03-data-model.md)
- [Error Model](../spec/04-error-model.md)
- [PRD Section 6](https://github.com/AbdelStark/mneme/blob/main/prd.md#6-data-model-and-core-concepts)
