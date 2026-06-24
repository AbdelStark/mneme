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
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import UUID

import numpy as np
import torch

Latent = torch.Tensor | np.ndarray
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

@dataclass(frozen=True)
class MemoryItem:
    content_id: Cid | None
    key: SummaryVec
    value: Transition | Frame | Window
    meta: Mapping[str, Any]
    encoder_fp: EncoderFingerprint
    schema_version: str = "mneme.memory_item.v1"
```

## Protocols

```python
class Encoder(Protocol):
    def encode(self, obs: object) -> Latent: ...
    def fingerprint(self) -> EncoderFingerprint: ...

class Summarizer(Protocol):
    @property
    def id(self) -> str: ...
    def summarize(self, z: Latent) -> SummaryVec: ...

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

## Constructors

The v0.1 package must provide:

```python
def open_store(path: Path | str, *, create: bool = False) -> MemoryStore: ...
def build_item(value: Transition | Frame | Window, key: SummaryVec, encoder_fp: EncoderFingerprint, meta: Mapping[str, Any] | None = None) -> MemoryItem: ...
def content_id(item: MemoryItem) -> Cid: ...
```

These constructors centralize validation and prevent callers from bypassing schema and content-id rules.

## Command-Line Surface

```text
mneme store init PATH
mneme store stats PATH --json
mneme store verify PATH
mneme index rebuild PATH --encoder-fingerprint FINGERPRINT
mneme query PATH --vector VECTOR_FILE --k 16 --metric cosine --json
mneme eval fixtures --out reports/fixtures.json
mneme receipts verify RECEIPT_FILE --root ROOT_HEX
```

Commands return exit code 0 on success, 2 for invalid user input, 3 for data validation failure, 4 for unavailable optional dependency, and 5 for internal errors.

## Numerical Contracts

- Summary vectors are one-dimensional, contiguous, CPU `float32` arrays.
- Cosine search requires L2-normalized vectors within tolerance `1e-4`.
- Returned distances are finite Python floats.
- Conditioners return the same latent backend type as `parametric` when possible.
- Torch conditioners run under inference mode unless explicitly training.
- Device movement is explicit and recorded in debug logs.

## Deprecation Policy

Before v1.0, public breaking changes require a changelog entry and migration note. After v1.0, a public API is deprecated for at least one minor release before removal unless a security issue requires faster removal.

## Open Questions

- OPEN QUESTION: Whether `Frame` and `Window` ship in v0.1 or start as v0.2 types. Owner: maintainer. Target: v0.1 API implementation.
