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
from typing import Any, Mapping, Protocol, Sequence
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

from mneme.index import FlatIndex, Index, planned_search_k, search_index

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

class Conditioner(Protocol):
    def condition(self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx) -> Latent: ...
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

`search_index` applies shared `QuerySpec` semantics around an index backend:
fail-closed fingerprint checks, deterministic over-fetch before store filters,
stable de-duplication by first occurrence, optional temporal decay, and final
top-k truncation. `planned_search_k` returns `k` for unfiltered exact search and
`max(k * 4, ef or k)` when filters require over-fetching unless callers provide
a different multiplier.

## Constructors

The v0.1 package must provide:

```python
from mneme.store import init_store, open_store, rebuild_index, verify_store

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
The implemented v0.1 module entry points are `python -m mneme.cli store verify PATH`
and `python -m mneme.cli index rebuild PATH`; both print JSON reports.

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
- Torch conditioners run under inference mode unless explicitly training.
- Device movement is explicit and recorded in debug logs.

## Deprecation Policy

Before v1.0, public breaking changes require a changelog entry and migration note. After v1.0, a public API is deprecated for at least one minor release before removal unless a security issue requires faster removal.

## Resolved Bootstrap Decisions

- `Frame` and `Window` remain documented data concepts but do not ship as required v0.1 public types. v0.1 implements `Transition` only; `Frame` and `Window` move to v0.2 or later when a concrete conditioning path needs them.
