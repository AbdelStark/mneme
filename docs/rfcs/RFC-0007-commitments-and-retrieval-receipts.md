# RFC-0007: Commitments and Retrieval Receipts

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.3

## Summary

Mneme commits memory item content ids in append order and returns retrieval receipts with inclusion proofs. Receipts prove returned items are committed members of the store at a root and that replay can reconstruct the conditioning set. They do not prove approximate top-k optimality.

## Motivation

[Security](../spec/06-security.md#integrity-controls) requires tamper-evident memory. The PRD's verifiability layer depends on content addressing, append-only roots, inclusion proofs, and eventually signed roots. This RFC defines the v0.3 integrity tier and explicitly scopes out verifiable search.

## Goals

- Commit content ids in append order.
- Return inclusion proofs for retrieved ids.
- Bind receipts to query parameters and store root.
- Reserve optional root-signature fields without claiming signed provenance.
- Preserve the distinction between membership proof and search correctness.

## Non-Goals

- Prove exact top-k correctness.
- Provide confidentiality.
- Implement private retrieval.
- Require commitments for v0.1 local stores.

## Proposed Design

Commitment state:

```python
@dataclass(frozen=True)
class CommitmentState:
    scheme: Literal["mmr-v1"]
    root: MerkleRoot
    item_count: int
    peaks: tuple[bytes, ...]
    leaf_ids: tuple[Cid, ...]
    schema_version: str = "mneme.commitment.v1"
```

The sidecar persists `leaf_ids` so local stores can produce inclusion proofs
after reopening. `peaks` and `root` are recomputed from the leaf ids during load
and fail closed if the sidecar is inconsistent.

Receipt:

```python
@dataclass(frozen=True)
class RetrievalReceipt:
    schema_version: str
    root: MerkleRoot
    ids: tuple[Cid, ...]
    proofs: tuple[InclusionProof, ...]
    params: QueryReceiptParams
    store_id: str
    created_at: str
    signer: str | None = None
    signature: bytes | None = None
```

The store appends content ids to a Merkle Mountain Range. `commit()` seals the current peak set, persists a sidecar, updates manifest commitment fields, and returns the root. `prove(ids)` returns inclusion proofs for ids present at the current root. `query(with_receipt=True)` snapshots the current root, builds proofs for returned ids, and attaches query parameters sufficient for deterministic replay.

Enabled manifest commitments require backend `mmr-v1`, a 32-byte hex root, and
at least one relative sidecar file. Disabled commitments keep backend, root, and
files empty.

The implemented local receipt binds query parameters with a vector digest,
vector shape and dtype, `k`, metric, optional `ef`, filters, temporal decay, and
optional encoder fingerprint. The receipt JSON does not embed raw latent query
vectors.

Verification:

1. canonicalize returned items and recompute content ids;
2. verify each content id is listed in the receipt;
3. verify each inclusion proof against the root;
4. reject signed receipts until a signing backend is implemented and tested;
5. verify query parameters match the replay request.

The receipt does not prove the index returned the true top-k neighbors. It proves returned items were committed and unaltered.

The initial local verifier supports unsigned receipts. If `signer` and
`signature` are present, verification fails closed until a signing backend is
implemented and tested. Signed-root verification is a compatible schema
extension and remains separate from membership proof semantics.

## Alternatives Considered

- Hash a sorted set of ids for each commit: simpler, but cannot prove append history efficiently.
- Use a plain Merkle tree rebuilt on every append: conceptually simple, but inefficient for append-heavy stores.
- Sign every item instead of roots: increases signature overhead and does not bind append history.
- Wait for verifiable search before adding receipts: blocks useful provenance on a harder research problem.

## Drawbacks

- Inclusion proofs add storage and query overhead.
- Receipt verification requires returned values or their canonical bytes.
- Users may overinterpret receipts as search-optimality proofs unless docs are explicit.

## Migration / Rollout

v0.3 adds commitment files to existing stores through an offline `mneme store commit-init PATH` command that verifies the store, scans value logs in append order, records the commitment root in the manifest, and reports the item count and root. Stores without commitments continue to work but cannot satisfy `with_receipt=True`.

## Testing Strategy

- MMR append and proof golden tests.
- Receipt verification success and mutation failure cases.
- Root mismatch tests and signed-receipt fail-closed tests.
- Store upgrade test from uncommitted v0.1 store.
- Receipt overhead benchmark with proof size as a function of item count.

## Resolved Bootstrap Decisions

- Optional root signatures are reserved for a later signing backend. The current
  receipt schema stores `signer` and `signature` fields only; any Ed25519
  payload format, algorithm identifier, key identifier, and verification policy
  must be added with tests before documentation can claim signed provenance.

## References

- [Security](../spec/06-security.md)
- [Data Model](../spec/03-data-model.md#retrievalreceipt)
- [PRD Section 10](https://github.com/AbdelStark/mneme/blob/main/prd.md#10-verifiability-layer)
