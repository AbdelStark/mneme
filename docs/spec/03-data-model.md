# Data Model

- Status: Accepted
- Created: 2026-06-24
- Source: [https://github.com/AbdelStark/mneme/blob/main/prd.md](https://github.com/AbdelStark/mneme/blob/main/prd.md#6-data-model-and-core-concepts)

## Schema Versioning

Every persisted object carries an explicit schema version string. Readers must reject unknown major versions and may migrate older minor versions through registered migration functions.

Required initial schema identifiers:

- `mneme.encoder_fingerprint.v1`
- `mneme.summary_vec.v1`
- `mneme.transition.v1`
- `mneme.memory_item.v1`
- `mneme.query_spec.v1`
- `mneme.retrieval.v1`
- `mneme.store_manifest.v1`
- `mneme.receipt.v1`
- `mneme.eval_report.v1`

## Latent

A `Latent` is opaque to Mneme except for shape, dtype, backend, and subtraction compatibility for transition deltas. Valid backends are `numpy.ndarray` and `torch.Tensor`.

Invariants:

- INV-DATA-001: A latent stored in one `Transition` has a stable shape across `z_src`, `z_next`, and `delta`.
- INV-DATA-002: `delta = z_next - z_src` is stored only when both operands share shape and numeric dtype.
- INV-DATA-003: Mneme does not assume pixels, patches, tokens, or simulator state from a latent unless an adapter declares that metadata.

## SummaryVec

A `SummaryVec` is the index key.

Contract:

```python
SummaryVec = np.ndarray  # shape: (dim,), dtype: float32, contiguous, finite
```

Invariants:

- INV-DATA-010: `dim > 0`.
- INV-DATA-011: all values are finite.
- INV-DATA-012: cosine keys are normalized to unit L2 norm within `1e-4`.
- INV-DATA-013: keys are comparable only when their `EncoderFingerprint` matches.

## EncoderFingerprint

`EncoderFingerprint` binds a key to the encoder and summarizer that produced it.

Canonical fields:

- `encoder_id`: human-readable stable adapter id.
- `summarizer_id`: stable summarizer id.
- `weights_digest`: digest of encoder weights when available.
- `config_digest`: digest of adapter and summarizer configuration.
- `schema_version`: fingerprint schema version.

The fingerprint is part of the memory item content id. Changing the encoder or summarizer changes item identity because old keys cannot be compared safely with new keys.

## Memory Values

`Transition` is required in v0.1. `Frame` and `Window` are public data concepts and may be implemented after the transition path.

```python
@dataclass(frozen=True)
class Frame:
    z: Latent
    t: int
    episode_id: UUID

@dataclass(frozen=True)
class Window:
    frames: tuple[Frame, ...]
    episode_id: UUID
```

Invariants:

- INV-DATA-020: `Transition.t >= 0`.
- INV-DATA-021: `episode_id` is a valid UUID.
- INV-DATA-022: `action` is one-dimensional or adapter-declared structured action data.
- INV-DATA-023: `reward` is optional and not used by default conditioning.

## MemoryItem

`MemoryItem` is the committed unit of storage.

Canonical content-id input order:

1. schema version
2. encoder fingerprint
3. key bytes
4. value kind
5. value bytes
6. normalized metadata

Metadata values must be JSON-compatible scalars, arrays, or objects after
normalization. `bytes` and `bytearray` metadata are rejected rather than
implicitly base64-encoded; callers that need binary metadata must encode it into
an explicit JSON object or string before constructing a `MemoryItem`.

Invariants:

- INV-DATA-030: `content_id` equals the digest of the canonical item bytes when present.
- INV-DATA-031: store append order is independent of content id.
- INV-DATA-032: two byte-identical items have the same content id.
- INV-DATA-033: metadata cannot override reserved fields such as `schema_version`, `content_id`, or `encoder_fp`.
- INV-DATA-034: every `content_id` at public, persisted, and remote boundaries is a 32-byte BLAKE3 digest.

## QuerySpec

`QuerySpec` defines retrieval semantics.

Invariants:

- INV-DATA-040: `k >= 1`.
- INV-DATA-041: `ef is None or ef >= k`.
- INV-DATA-042: `temporal_decay is None or temporal_decay >= 0`.
- INV-DATA-043: filters use a schema-versioned allowlist of fields.
- INV-DATA-044: `with_receipt=True` requires a store that supports receipts or raises `UnsupportedOperationError`.

## RetrievalReceipt

A receipt proves returned items are members of a committed store root. It does not prove approximate search optimality.

Required fields:

- `schema_version`
- `root`
- `ids`
- `proofs`
- `params`
- `store_id`
- `created_at`
- `signer`
- `signature`

`params` stores a schema-versioned query-parameter digest, including vector
digest, vector shape and dtype, `k`, metric, optional `ef`, filters, temporal
decay, and optional encoder fingerprint. The receipt binds a replay request
without embedding raw latent vectors in the receipt JSON.

## Persistence Manifest

The manifest records:

- store schema version
- store id
- created and updated timestamps
- active encoder fingerprints
- value log file names and offsets
- index backend and backend parameters
- retention policy
- commitment state when enabled
- last completed transaction id

## Resolved Bootstrap Decisions

- Default tensor persistence: value-log records use a chunked binary format with a canonical metadata header and raw numeric array payloads. Torch tensors are detached, moved to CPU, converted to contiguous numeric arrays, and stored without Python pickle. The content id is computed over the canonical header plus normalized tensor bytes.
