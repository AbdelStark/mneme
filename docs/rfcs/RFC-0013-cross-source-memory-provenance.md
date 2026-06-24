# RFC-0013: Cross-Source Memory Provenance

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.5

## Summary

Mneme v0.5 may experiment with memory items drawn from multiple local, remote,
or dataset-backed sources. Cross-source use requires explicit source identity,
per-source provenance receipts, and conservative privacy limits. This RFC
defines the design boundary before any federation service exists.

Cross-source provenance is an integrity and audit feature. It does not provide
confidentiality, private retrieval, encrypted search, consent automation, or
proof that approximate search returned the exact top-k set. A cross-source
receipt does not prove private retrieval.

## Motivation

[Overview](../spec/00-overview.md#goals) allows remote or shared stores through
schema-versioned messages. [Security](../spec/06-security.md) and
[RFC-0012](RFC-0012-security-boundaries-and-privacy-tiers.md) state that remote
responses are untrusted and v0.x stores are not confidential by default.
Combining memories from multiple sources can make those boundaries ambiguous
unless source identity, receipt validation, and metrics are explicit.

## Goals

- Define source identity metadata for retrieved memory items.
- Define provenance receipt requirements across multiple sources.
- Preserve fail-closed validation before conditioning.
- Document cross-source trust and privacy assumptions.
- Identify metrics for cross-source transfer evaluation.

## Non-Goals

- Implement a federation service.
- Add private retrieval or encrypted stores.
- Automate consent, licensing, or data-subject policy compliance.
- Prove exact top-k correctness across stores.
- Require cross-source support for v1.0.

## Proposed Design

### Source Identity Metadata

Every source that contributes memory to a cross-source retrieval must have a
JSON-safe `mneme.source.v1` identity object. The object is carried in reports,
trace artifacts, or a reserved metadata field such as `meta["mneme_source"]`;
it must not replace the canonical `MemoryItem.content_id`.

Minimum fields:

```python
@dataclass(frozen=True)
class SourceIdentity:
    schema_version: Literal["mneme.source.v1"]
    source_id: str
    source_kind: Literal["local_store", "remote_store", "dataset", "derived"]
    store_id: str | None
    root: str | None
    root_scheme: str | None
    encoder_fingerprint: EncoderFingerprint
    policy_tags: tuple[str, ...]
    disclosure_level: Literal["opaque", "internal", "public"]
```

Field rules:

- `source_id` is stable and opaque. Public reports should use a generated id or
  digest, not raw hostnames, user ids, private dataset names, or secret paths.
- `source_kind` distinguishes local stores, remote stores, offline datasets, and
  derived/generated sources.
- `store_id`, `root`, and `root_scheme` bind the identity to a committed store
  when commitments exist. Uncommitted sources must set these fields to `None`
  and cannot satisfy receipt-backed provenance claims.
- `encoder_fingerprint` prevents unsafe key comparison across incompatible
  encoders or summarizers.
- `policy_tags` records operator-supplied tags such as `public-fixture`,
  `internal`, `no-redistribution`, or `expires-YYYY-MM-DD`. Mneme records tags
  but does not enforce legal or consent policy by itself.
- `disclosure_level` controls how much source detail may appear in public
  reports and logs.

### Cross-Source Provenance Receipt

A cross-source retrieval must keep per-source receipts rather than flattening
all sources into an implicit global trust root.

```python
@dataclass(frozen=True)
class CrossSourceProvenanceReceipt:
    schema_version: Literal["mneme.cross_source_receipt.v1"]
    query_digest: str
    aggregation_policy: str
    sources: tuple[SourceIdentity, ...]
    returned_ids_by_source: Mapping[str, tuple[Cid, ...]]
    retrieval_receipts_by_source: Mapping[str, RetrievalReceipt]
    validation_steps: tuple[str, ...]
    created_at: str
```

Required validation steps:

1. validate each remote response schema before use;
2. recompute returned item content ids;
3. reject encoder fingerprint mismatches unless a documented migration adapter
   produced compatible keys;
4. verify every requested per-source `RetrievalReceipt`;
5. verify source roots and signatures when the deployment claims signed
   provenance;
6. record the aggregation policy that merged or ranked results across sources.

Cross-source receipts prove only that the returned item bytes were members of
the stated source roots and that the aggregator recorded its validation inputs.
They do not prove private retrieval, encrypted storage, consent compliance, or
search optimality. If an aggregator needs a single root, it must commit the
aggregated returned ids into a new local root and record that as a derived
source.

### Trust And Privacy Assumptions

- Source identities, policy tags, roots, and item ids can reveal sensitive
  relationships between operators, datasets, and tasks. Default logs and public
  reports must redact or hash source details unless `disclosure_level` is
  `public`.
- Cross-source sharing does not relax the
  [shared-store deployment checklist](../spec/06-security.md#shared-store-deployment-checklist).
  Operators still own authenticated transport, credential management, backup
  controls, root publication, and log retention.
- Mneme v0.x does not provide confidentiality. Any cross-source deployment that
  needs confidentiality must add access control, transport security, storage
  encryption, and secret management outside Mneme until a separate encrypted
  store RFC lands.
- Private retrieval is research-only in v0.x. Public examples must not imply
  that a source can hide queries or returned ids from another operator.

### Metrics For Cross-Source Transfer

The v0.5 evaluation work should report at least:

- source count and returned item count per source;
- in-source baseline error versus cross-source conditioned error;
- cross-source improvement rate on fixture tasks;
- negative-transfer rate where cross-source memory worsens the target metric;
- source-diversity score for retrieved conditioning sets;
- encoder fingerprint rejection count;
- receipt verification failure count by source;
- policy-filter rejection count by tag;
- query latency overhead and proof bytes by source;
- redaction failure count for source metadata in public reports.

These metrics are evaluation signals, not benchmark claims. External task claims
still require an external benchmark report under RFC-0009.

## Alternatives Considered

- Single global root for all shared memories: simple for verification, but it
  hides source-specific trust and retention boundaries.
- Trust transport authentication alone: insufficient because payload schemas,
  content ids, fingerprints, and receipts still need validation.
- Treat policy tags as enforcement: too broad for Mneme; policy enforcement is
  operator and deployment specific.

## Drawbacks

- Provenance metadata increases report and receipt size.
- Opaque source ids make debugging harder unless operators keep a private
  mapping.
- Cross-source metrics can be misread as external task evidence without clear
  caveats.

## Migration / Rollout

v0.5 starts with fixture-scale cross-source transfer reports and no federation
service. The immediate implementation issue is #48, which measures
cross-source memory transfer under these metrics. The v1.0 security review is
#50. If implementation expands beyond those issues, create separate issues for
source identity serialization, cross-source receipt construction, and redaction
regression coverage before coding the feature.

## Testing Strategy

- Source identity JSON round-trip rejects missing schema or unsafe public fields.
- Cross-source receipt validation fails closed on missing source receipts,
  fingerprint mismatch, root mismatch, or altered item bytes.
- Evaluation reports include source counts, transfer metrics, caveats, and
  redacted source metadata.
- Public docs continue to state that cross-source sharing does not provide
  confidentiality or private retrieval.

## References

- [Overview](../spec/00-overview.md)
- [Security](../spec/06-security.md)
- [RFC-0007](RFC-0007-commitments-and-retrieval-receipts.md)
- [RFC-0009](RFC-0009-evaluation-and-reproducibility-harness.md)
- [RFC-0012](RFC-0012-security-boundaries-and-privacy-tiers.md)
