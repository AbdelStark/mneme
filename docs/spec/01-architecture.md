# Architecture

- Status: Accepted
- Created: 2026-06-24
- Source: [https://github.com/AbdelStark/mneme/blob/main/prd.md](https://github.com/AbdelStark/mneme/blob/main/prd.md#5-system-overview)

## System Shape

Mneme has a write path that records realized experience and a read path that conditions prediction. The encoder and predictor are external by design. Mneme owns the contracts between latent values, summary keys, indexes, stores, conditioners, receipts, and evaluation reports.

```text
observe -> Encoder -> Latent
                     |
                     +--> Summarizer -> SummaryVec -> Index
                     |
real step result ----+--> Transition -> MemoryItem -> Store -> optional Commitment

rollout step:
Latent -> Summarizer -> QuerySpec -> Store.query -> Retrieval
Predictor(z, action) -> z_hat
Conditioner(z_hat, Retrieval, CondCtx) -> z_pred
Planner consumes z_pred
```

## Package Boundaries

`mneme.core`
: Owns public data carriers, protocols, enums, schema constants, canonical serialization, validation helpers, and base errors. It must not import optional ML or index backends at module import time.

`mneme.encode`
: Owns `Encoder`, `PredictorAdapter`, `Summarizer`, `EncoderFingerprint`, and reference summarizers. Backend-specific adapters are optional extras and must fail with actionable dependency errors when unavailable.

`mneme.index`
: Owns the `Index` protocol and backends. A flat exact index is required for tests and recall measurement. Approximate backends must expose their recall and latency knobs through `QuerySpec`.

`mneme.store`
: Owns `MemoryStore`, append log persistence, value loading, manifest files, retention, index rebuilds, and eventually commitments. Store operations must validate schema versions before mutation.

`mneme.condition`
: Owns `Conditioner`, `KnnCorrector`, `InContextConditioner`, and `CrossAttnAdapter`. Conditioners must reduce to the parametric prediction when retrieval is empty or gated off.

`mneme.wmcp`
: Owns remote message schemas, client/server adapters, and conformance tests. It must call the same core protocols as the local store.

`mneme.eval`
: Owns reproducible evaluation commands and report schemas. It must separate fixture tests from external benchmark runs.

`mneme.cli`
: Owns user-facing commands for store inspection, index rebuild, query, receipt verification, evaluation, and environment checks.

## Write Path

1. The caller encodes an observation into `z_src`.
2. The caller executes an action and encodes the next observation into `z_next`.
3. Mneme builds a `Transition` with `delta = z_next - z_src` when shape and dtype permit subtraction.
4. The summarizer maps `z_src` to a contiguous `float32` `SummaryVec`.
5. Canonical serialization computes a content id over key, value, metadata, schema version, and encoder fingerprint.
6. The store appends the item to the log, writes value bytes or references, adds the key to the index, updates the manifest, and optionally updates the commitment root.

## Read Path

1. The caller summarizes the current latent into a query vector.
2. `MemoryStore.query` validates the query shape, metric, fingerprint, filters, and `k`.
3. The store searches the index, loads values for candidate ids, applies filters and temporal decay, and returns a `Retrieval`.
4. If requested and supported, the store attaches inclusion proofs and a root to the retrieval receipt.
5. The caller obtains the base prediction `z_hat` from its predictor.
6. The conditioner fuses `z_hat` with the retrieval into `z_pred`.

## Runtime Modes

- `per_real_step`: one retrieval per real control step; default for v0.1.
- `per_imagined_step`: batched retrieval for each imagined rollout step; supported after batched query tests.
- `offline_replay`: deterministic replay over a logged episode for evaluation and receipt verification.
- `index_rebuild`: rebuild summary keys and index files after encoder or summarizer changes.

Offline receipt replay reconstructs the logged query, returned items, receipt,
conditioner configuration, parametric prediction, and current latent, then
recomputes conditioning after receipt verification. It does not replay the full
environment and does not prove exact nearest-neighbor selection.

## Invariants

- INV-ARCH-001: Core types do not depend on optional runtime extras.
- INV-ARCH-002: A store never compares keys with different encoder fingerprints unless an explicit migration adapter produced compatible keys.
- INV-ARCH-003: Empty retrieval returns the parametric prediction unchanged.
- INV-ARCH-004: Query and conditioning can be replayed from persisted inputs when deterministic flags are enabled.
- INV-ARCH-005: Persisted files are self-describing through schema version, content id, and manifest entries.

## Cross-Cutting Concerns

- Determinism: canonical serialization, stable sorting on equal distances, fixed seeds in evaluation, and documented nondeterminism for backend kernels.
- Device and dtype: summaries are CPU `float32`; values may be CPU arrays or tensors; conditioners explicitly move values to the prediction device.
- Error handling: public operations raise typed errors from `mneme.core.errors`.
- Observability: every query can emit structured fields for k, latency, backend, fingerprint, hit count, gate value, and receipt status.

## Risks

- RISK: Optional extras can leak imports into core modules and make installation brittle. Resolution: import-boundary tests in v0.1.
- RISK: The store and index can diverge after partial writes. Resolution: manifest transaction states and repair commands in RFC-0004.
