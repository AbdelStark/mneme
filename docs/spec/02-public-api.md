# Public API

- Status: Accepted
- Created: 2026-06-24
- Source: [../../prd.md](../../prd.md#11-reference-implementation-mneme-python)

## API Stability

Public APIs are objects exported from `mneme.__init__`, documented under `docs/spec/`, or used in examples. Internal modules may change between minor versions. Public APIs follow semantic versioning after v1.0; before v1.0, breaking changes are allowed only with changelog entries and migration notes.

## Core Types

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Mapping, Protocol, Sequence
from uuid import UUID

import numpy as np

Latent = Any  # np.ndarray or shape/dtype-compatible tensor without importing optional backends
SummaryVec = np.ndarray
Cid = bytes
MerkleRoot = bytes

class Metric(StrEnum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"

@dataclass(frozen=True)
class EncoderFingerprint:
    encoder_id: str
    summarizer_id: str
    weights_digest: str | None
    config_digest: str
    schema_version: str = "mneme.encoder_fingerprint.v1"

@dataclass(frozen=True)
class QuerySpec:
    vector: SummaryVec
    k: int
    metric: Metric = Metric.COSINE
    ef: int | None = None
    filters: Mapping[str, Any] | None = None
    temporal_decay: float | None = None
    with_receipt: bool = False
    encoder_fp: EncoderFingerprint | None = None

@dataclass(frozen=True)
class Transition:
    z_src: Latent
    action: np.ndarray
    z_next: Latent
    delta: Latent
    t: int
    episode_id: UUID
    reward: float | None = None
    schema_version: str = "mneme.transition.v1"

@dataclass(frozen=True)
class MemoryItem:
    content_id: Cid | None
    key: SummaryVec
    value: Transition
    meta: Mapping[str, Any]
    encoder_fp: EncoderFingerprint
    schema_version: str = "mneme.memory_item.v1"

@dataclass(frozen=True)
class QuerySpec:
    vector: SummaryVec
    k: int
    metric: Metric = Metric.COSINE
    ef: int | None = None
    filters: Mapping[str, Any] | None = None
    temporal_decay: float | None = None
    with_receipt: bool = False
    encoder_fp: EncoderFingerprint | None = None
    schema_version: str = "mneme.query_spec.v1"

@dataclass(frozen=True)
class Retrieval:
    items: tuple[MemoryItem, ...]
    distances: tuple[float, ...]
    receipt: object | None = None
    schema_version: str = "mneme.retrieval.v1"
```

These v0.1 carriers are exported from `mneme.core`: `Latent`, `SummaryVec`,
`Cid`, `Metric`, `EncoderFingerprint`, `Transition`, `MemoryItem`,
`QuerySpec`, and `Retrieval`. `mneme.core` may import NumPy, but it must not
import optional ML, index, receipt, or remote backends.

## Protocols

```python
from mneme.encode import Encoder, MeanPoolSummarizer, Summarizer
from mneme.encode import build_encoder_fingerprint, ensure_fingerprint_match

class Encoder(Protocol):
    def encode(self, obs: object) -> Latent: ...
    def fingerprint(self) -> EncoderFingerprint: ...

class Summarizer(Protocol):
    @property
    def id(self) -> str: ...
    def summarize(self, z: Latent) -> SummaryVec: ...

from mneme.index import FaissHnswIndex, FlatIndex, Index, planned_search_k, search_index
from mneme.condition import CondCtx, Conditioner, InContextConditioner, KnnCorrector
from mneme.adapter import AdapterCheckpointMetadata, load_adapter_checkpoint
from mneme.eval import BenchmarkResult, BenchmarkRunner, BenchmarkSpec, DryRunBenchmarkRunner

class Index(Protocol):
    def add(self, cid: Cid, key: SummaryVec) -> None: ...
    def add_batch(self, items: Sequence[tuple[Cid, SummaryVec]]) -> None: ...
    def search(self, q: SummaryVec, k: int, *, metric: Metric, ef: int | None = None) -> list[tuple[Cid, float]]: ...
    def __len__(self) -> int: ...

class MemoryStore(Protocol):
    def put(self, item: MemoryItem) -> Cid: ...
    def put_batch(self, items: Sequence[MemoryItem]) -> list[Cid]: ...
    def query(self, spec: QuerySpec) -> Retrieval: ...
    def commit(self) -> MerkleRoot: ...
    def prove(self, ids: Sequence[Cid]) -> list[InclusionProof]: ...
    def root(self) -> MerkleRoot: ...
    def stats(self) -> StoreStats: ...

@dataclass(frozen=True)
class CondCtx:
    current_latent: Latent | None
    step: int | None = None
    goal_latent: Latent | None = None
    metadata: Mapping[str, Any] | None = None
    schema_version: str = "mneme.cond_ctx.v1"

class Conditioner(Protocol):
    def condition(self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx) -> Latent: ...

@dataclass(frozen=True)
class KnnCorrector:
    tau: float = 0.1
    lambda_max: float = 0.5
    alpha: float = 10.0
    delta0: float = 0.2
    mode: Literal["delta", "absolute"] = "delta"

    def condition(self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx) -> Latent: ...

class InContextPredictor(Protocol):
    def predict_with_context(
        self,
        parametric: Latent,
        retrieved_tokens: Sequence[Latent],
        ctx: CondCtx,
    ) -> Latent: ...

@dataclass(frozen=True)
class InContextConditioner:
    predictor: InContextPredictor | Callable[[Latent, Sequence[Latent], CondCtx], Latent]
    max_tokens: int | None = None

    def condition(self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx) -> Latent: ...

class CrossAttnAdapter(torch.nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None: ...

    def forward(
        self,
        predictor_hidden: torch.Tensor,
        retrieved_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

@dataclass(frozen=True)
class AdapterTrainingBatch:
    predictor_input: object
    retrieved_values: object
    target_hidden: object
    attention_mask: object | None = None

@dataclass(frozen=True)
class AdapterCheckpointMetadata:
    adapter_kind: str
    adapter_config: Mapping[str, Any]
    base_fingerprint: EncoderFingerprint
    training_report_uri: str
    weights_file: str = "adapter.safetensors"
    package_version: str = mneme.__version__
    schema_version: str = "mneme.adapter_checkpoint.v1"

    def to_json(self) -> dict[str, object]: ...

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> AdapterCheckpointMetadata: ...

def load_adapter_checkpoint(
    path: Path | str,
    *,
    expected_base_fingerprint: EncoderFingerprint | None = None,
    require_weights: bool = True,
) -> AdapterCheckpoint: ...

class BenchmarkRunner(Protocol):
    def run(self, spec: BenchmarkSpec) -> BenchmarkResult: ...

@dataclass(frozen=True)
class BenchmarkSpec:
    dataset: DatasetRef
    checkpoint_uri: str
    modes: Sequence[Literal["no_memory", "corrector", "in_context", "adapter"]]
    command: Sequence[str]
    seed: int | None = None
    hardware: Mapping[str, str] = field(default_factory=dict)
```

Minimum custom adapter example:

```python
import numpy as np

from mneme.core import EncoderFingerprint
from mneme.encode import Encoder, Summarizer

class MyEncoder:
    def encode(self, obs: object) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def fingerprint(self) -> EncoderFingerprint:
        return EncoderFingerprint(
            encoder_id="example.encoder",
            summarizer_id="example.mean",
            weights_digest=None,
            config_digest="sha256:example",
        )

class MySummarizer:
    @property
    def id(self) -> str:
        return "example.mean"

    def summarize(self, z: object) -> np.ndarray:
        array = np.asarray(z, dtype=np.float32)
        return np.ascontiguousarray(array.reshape(-1).mean(keepdims=True))

encoder: Encoder = MyEncoder()
summarizer: Summarizer = MySummarizer()
```

Adapter implementations own model-specific imports. Importing `mneme.encode`
must not import optional ML backends such as torch.
Importing `mneme.adapter` must also avoid importing torch; requesting
`CrossAttnAdapter` requires the `ml` extra and raises
`OptionalDependencyError(extra="ml", package="torch")` when torch is missing.

`MeanPoolSummarizer` is the v0.1 default summarizer. It mean-pools all
non-feature axes, returns a contiguous finite `float32` vector, and L2-normalizes
by default for cosine keys. Deterministic projection is intentionally deferred
until v0.2 or the first large-latent adapter that needs it.

Encoder adapters should build fingerprints with `build_encoder_fingerprint`.
The helper binds the summarizer id and summarizer config into
`config_digest`, computes optional BLAKE3 weight digests, and requires
`unweighted=True` when no weight digest exists. Stores and indexes should use
`ensure_fingerprint_match` before comparing keys from different sources.

`FlatIndex` is the required exact reference backend. It performs NumPy flat
search, returns stable results by breaking equal-distance ties with content-id
bytes, and does not require optional index extras.
`FaissHnswIndex` is the optional v0.1 approximate backend behind the `index`
extra. Its manifest backend name is `faiss_hnsw`; missing FAISS support raises
`OptionalDependencyError(extra="index", package="faiss-cpu")`. The backend
returns portable Mneme distances: L2 distance, cosine distance, and negative
inner-product score, matching the `Index` protocol.

`search_index` applies shared `QuerySpec` semantics around an index backend:
fail-closed fingerprint checks, deterministic over-fetch before store filters,
stable de-duplication by first occurrence, optional temporal decay, and final
top-k truncation. `planned_search_k` returns `k` for unfiltered exact search and
`max(k * 4, ef or k)` when filters require over-fetching unless callers provide
a different multiplier.

`Conditioner` is the public protocol for memory-conditioned prediction. It
accepts an already computed parametric latent, a retrieval, and `CondCtx`; it
does not own or require gradients through a base model. Conditioners must return
the parametric latent unchanged for empty retrievals unless they document a
stricter typed failure mode. `CondCtx` carries the current latent, optional goal
latent, optional step, and JSON-safe metadata for one conditioning call.
`KnnCorrector` is the v0.1 training-free reference conditioner. It computes a
distance-softmax nonparametric estimate from retrieved `Transition` values,
supports `delta` and `absolute` modes, and gates the memory estimate toward zero
as nearest-neighbor distance grows.
The default gate parameters are fixture baselines, not universal safety
guarantees; deployment-safe fallback depends on calibrating nearest-neighbor
distance distributions for the target encoder, summarizer, and task.

`InContextConditioner` is a v0.2 comparison baseline for compatible predictor
wrappers. It validates each retrieved `Transition.z_next` against the
parametric prediction shape, passes those successor latents as appended context
tokens to `predict_with_context(...)` or an equivalent callable, and preserves
the empty-retrieval identity fallback. It is not the default conditioner:
self-attention cost grows with `k`, and longer retrieved contexts can dilute the
signal.

`CrossAttnAdapter` is the v0.2 trained memory module behind the `ml` extra.
`predictor_hidden` has shape `(batch, predictor_tokens, hidden_dim)`.
`retrieved_values` has shape `(batch, retrieved, latent_dim)` and is projected
into hidden space before cross-attention. `attention_mask`, when provided, has
shape `(batch, retrieved)` and uses `True`/`1` for valid retrieved slots; masked
slots are not attended. Retrieved values and masks are moved to the
`predictor_hidden` dtype and device before attention. The module returns
`(batch, predictor_tokens, hidden_dim)` and does not own base predictor weights.

`train_frozen_base_adapter` is the fixture-scale offline training harness for
adapter-only training. It requires torch from the `ml` extra, a callable frozen
base model whose `parameters()` can be frozen, a callable adapter with
trainable `parameters()`, and non-empty `train`, `calibration`, and
`validation` splits of `AdapterTrainingBatch`. The harness freezes and clears
base parameters before training, runs the base model under `torch.no_grad()`,
steps only adapter optimizer parameters, asserts base gradients remain absent
after backward, and returns `mneme.eval_report.v1` with split counts, seed,
loss metrics, caveats, and a `base_gradients_absent` metric.

`AdapterCheckpointMetadata` is the JSON sidecar schema for adapter artifacts. It
records `schema_version`, `adapter_kind`, JSON-compatible `adapter_config`,
`base_fingerprint`, `training_report_uri`, `weights_file`, and
`package_version`. `load_adapter_checkpoint` accepts either a checkpoint
directory containing `adapter.json` or a metadata JSON path, validates the
sidecar, rejects absolute or parent-traversing weight paths, checks that the
weight file exists by default, and raises `FingerprintMismatchError` when
`expected_base_fingerprint` does not match the sidecar.

`BenchmarkRunner` is the opt-in external benchmark interface. `BenchmarkSpec`
requires an external `DatasetRef`, split, model checkpoint URI, comparison modes
for no-memory/corrector/in-context/adapter, command, optional seed, and hardware
metadata. Runner results are wrapped in `mneme.eval_report.v1` with caveats; the
built-in `DryRunBenchmarkRunner` validates report plumbing only and is not
benchmark evidence.

## Constructors

The v0.1 package must provide:

```python
from mneme.store import age_retention, count_retention, init_store, open_store, rebuild_index, verify_store

def init_store(path: Path | str, *, ...) -> LocalStore: ...
def open_store(path: Path | str, *, create: bool = False) -> LocalStore: ...
def verify_store(path: Path | str, *, raise_on_error: bool = False) -> StoreVerificationReport: ...
def rebuild_index(path: Path | str) -> IndexRebuildReport: ...
def build_item(value: Transition, key: SummaryVec, encoder_fp: EncoderFingerprint, meta: Mapping[str, Any] | None = None) -> MemoryItem: ...
def content_id(item: MemoryItem) -> Cid: ...
def canonical_bytes(item: MemoryItem | Transition | EncoderFingerprint) -> bytes: ...
```

These constructors centralize validation and prevent callers from bypassing schema and content-id rules.
`content_id` computes a BLAKE3 digest over canonical bytes and excludes the
`content_id` field itself from the digest.
`init_store` creates the local v0.1 directory layout and schema-versioned
manifest. `open_store(...).stats()` returns manifest-derived store id, value-log,
index, transaction, and commitment-reservation fields.
`LocalStore.put` and `put_batch` append length-prefixed, checksummed value
records under a transaction intent/commit file, update the manifest, and rebuild
queryability from the value log on restart.
`count_retention(max_items)` caps visible items by newest `Transition.t` and
records tombstones in the manifest. `age_retention(max_age_seconds)` excludes
items older than the newest visible event-time `Transition.t` minus the age
window. Tombstoned records remain in value logs until a future compaction
command; queries and rebuilt indexes exclude them.
`LocalStore.recovery_events` is a tuple of schema-versioned
`StoreRecoveryEvent` objects produced by `open_store` when it completes or rolls
back pending transactions.
`verify_store` returns a schema-versioned JSON-ready report for manifest,
value-log checksum/content-id/fingerprint, and index-reference validation; with
`raise_on_error=True`, failed reports raise `StoreCorruptionError`.
`rebuild_index` rewrites non-destructive index metadata from value logs only,
including `index/backend.json` and a `mneme.flat_index_snapshot.v1`
`index/data.json` snapshot. It never deletes value logs.

## Command-Line Surface

```text
mneme store init PATH
mneme store stats PATH --json
mneme store verify PATH
mneme index rebuild PATH
mneme query PATH --vector VECTOR_FILE --k 16 --metric cosine --json
mneme eval fixtures --out reports/fixtures.json
mneme receipts verify RECEIPT_FILE --root ROOT_HEX
```

Commands return exit code 0 on success, 2 for invalid user input, 3 for data validation failure, 4 for unavailable optional dependency, and 5 for internal errors.
The implemented v0.1 module entry point is `python -m mneme.cli ...`.
Store stats, verification, index rebuild, query, fixture-eval, profile-eval,
recall-eval, and latency-eval commands print schema-versioned JSON reports.
CLI error responses print
`mneme.cli_error.v1` JSON with `ok: false`, typed `error_type`, and `errors`.
`receipts verify` is wired as the documented command shape but returns
`UnsupportedOperationError` until the receipt implementation lands.

CLI implementations translate public errors with:

```python
from mneme.core import CliExitCode, cli_exit_code
```

`cli_exit_code(None)` returns `CliExitCode.SUCCESS`. `QueryError` and
`UnsupportedOperationError` map to invalid user input, validation and
verification failures map to data validation failure, `OptionalDependencyError`
maps to unavailable optional dependency, and other errors map to internal
failure.

## Numerical Contracts

- Summary vectors are one-dimensional, contiguous, CPU `float32` arrays.
- Cosine search requires L2-normalized vectors within tolerance `1e-4`.
- Returned distances are finite Python floats.
- Conditioners return the same latent backend type as `parametric` when possible.
- `KnnCorrector` preserves NumPy output dtype from `parametric`.
- `KnnCorrector` restores torch outputs to the `parametric` tensor dtype and
  device.
- `InContextConditioner` requires retrieved successor tokens and predictor
  results to match the parametric prediction shape.
- Torch inputs are detached, copied through CPU NumPy for deterministic
  fixture-scale weighting, and restored to the parametric device.
- Torch conditioners run under inference mode unless explicitly training.
- Device movement is explicit and recorded in debug logs when observability is
  enabled.

## Deprecation Policy

Before v1.0, public breaking changes require a changelog entry and migration note. After v1.0, a public API is deprecated for at least one minor release before removal unless a security issue requires faster removal.

## Resolved Bootstrap Decisions

- `Frame` and `Window` remain documented data concepts but do not ship as required v0.1 public types. v0.1 implements `Transition` only; `Frame` and `Window` move to v0.2 or later when a concrete conditioning path needs them.
