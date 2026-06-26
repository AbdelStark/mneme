# RFC-0002: Encoder Fingerprints and Summary Keys

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme keys every memory item with a compact `float32` summary vector and binds that vector to an `EncoderFingerprint`. Stores reject unsafe comparison across fingerprints by default. The v0.1 summarizer is deterministic mean pooling plus normalization, with room for learned summarizers later.

## Motivation

[Architecture](../spec/01-architecture.md#write-path) separates large latent values from compact search keys. [Data Model](../spec/03-data-model.md#encoderfingerprint) states that keys are comparable only within a matching encoder and summarizer. This RFC defines that boundary so index behavior cannot silently mix incompatible representation spaces.

## Goals

- Define the `Encoder` and `Summarizer` contracts.
- Specify fingerprint fields and mismatch behavior.
- Ship a deterministic default summarizer in v0.1.
- Preserve enough metadata to rebuild keys when summarizers change.
- Keep adapters model-agnostic.

## Non-Goals

- Train a learned summarizer in v0.1.
- Define model-specific encoder internals.
- Guarantee semantic nearest-neighbor quality for all latent spaces.

## Proposed Design

Public protocols:

```python
class Encoder(Protocol):
    def encode(self, obs: object) -> Latent: ...
    def fingerprint(self) -> EncoderFingerprint: ...

class Summarizer(Protocol):
    @property
    def id(self) -> str: ...
    def summarize(self, z: Latent) -> SummaryVec: ...
```

Default summarizer:

```python
@dataclass(frozen=True)
class MeanPoolSummarizer:
    normalize: bool = True
    output_dim: int | None = None
    id: str = "meanpool-v1"
```

The default flattens all non-feature axes by averaging, converts to CPU `float32`, and L2-normalizes for cosine search. Deterministic projection is reserved for v0.2; v0.1 raises `UnsupportedOperationError` when `output_dim` is set.

Fingerprint fields:

- `encoder_id`: adapter identifier such as `custom.encoder`.
- `summarizer_id`: stable summarizer id.
- `weights_digest`: digest of weights when available; `None` only for explicitly unweighted encoders.
- `config_digest`: digest of adapter config, summarizer config, preprocessing, and projection seed.

Query behavior:

- A store with one active fingerprint rejects mismatched query fingerprints.
- A store with multiple fingerprints routes to the matching index.
- If no query fingerprint is provided, the store may search only when it has exactly one active fingerprint; otherwise it raises `FingerprintMismatchError`.

Lazy re-encoding is a store maintenance operation: read stored values, re-run a summarizer under a new fingerprint, rebuild the index, and record both fingerprints in the manifest.

## Alternatives Considered

- Single global store without fingerprints: simpler, but unsafe after encoder or summarizer changes.
- Store only summarizer id: insufficient when model weights change under the same summarizer.
- Use full latent as key: avoids summarization but makes indexing expensive and backend-specific.
- Require learned summary tokens from every model: better for some models, but blocks v0.1 adoption for arbitrary frozen predictors.

## Drawbacks

- Mean pooling may be a weak semantic key for some latent spaces.
- Fingerprints create migration work when adapters change.
- Weight digests can be expensive for very large models; adapters may need cached digest files.

## Migration / Rollout

v0.1 supports one fingerprint per local store and rejects mismatches. Multi-fingerprint routing and lazy re-encoding can land after the initial store is stable, but the manifest schema reserves the fields now.

## Testing Strategy

- Mean-pool summarizer returns contiguous finite `float32` vectors.
- Cosine keys are normalized within tolerance.
- Fingerprint digest changes when config or summarizer config changes.
- Store rejects mismatched fingerprints.
- Index rebuild test proves values can be re-summarized into a fresh index.

## Resolved Bootstrap Decisions

- Deterministic projection waits until v0.2 or the first large-latent adapter that needs it. v0.1 ships mean pooling plus L2 normalization only, which keeps the first summarizer auditable and avoids adding projection-matrix migration state before it is needed.

## References

- [Architecture](../spec/01-architecture.md)
- [Data Model](../spec/03-data-model.md#summaryvec)
- [PRD Section 9.3](https://github.com/AbdelStark/mneme/blob/main/prd.md#93-encoder-versioning)
