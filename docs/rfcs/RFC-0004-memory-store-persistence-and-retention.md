# RFC-0004: Memory Store Persistence and Retention

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme stores memory items in an append-mostly local store with a manifest, value logs, index files, transaction records, and retention policies. The value log is the source of truth; indexes are rebuildable acceleration structures.

## Motivation

[Architecture](../spec/01-architecture.md#write-path) requires durable writes and restartable query. [Error Model](../spec/04-error-model.md#failure-modes) identifies index/log divergence and partial transactions as first-class failure modes. A local persistence contract is required before commitments, remote stores, or release examples can be credible.

## Goals

- Define the local store directory layout.
- Make writes recoverable after interruption.
- Treat the value log as authoritative.
- Support index rebuild and store verification commands.
- Provide retention without corrupting index or manifest state.

## Non-Goals

- Multi-writer distributed concurrency in v0.1.
- Remote-store persistence; RFC-0008 covers remote messages.
- Cryptographic commitments; RFC-0007 layers on top.

## Proposed Design

Directory layout:

```text
store/
  manifest.json
  values/
    log-000000.mnv
  index/
    backend.json
    data.*
  transactions/
    txn-<id>.json
  receipts/
    optional exported receipts
```

`manifest.json` is schema-versioned and records store id, active fingerprints, value logs, index backend, backend parameters, retention policy, and last completed transaction.

Write transaction:

1. validate item;
2. compute or verify content id;
3. write transaction intent with item count and target files;
4. append value records with length prefix and checksum;
5. add keys to index;
6. fsync value log and manifest where supported;
7. mark transaction committed.

Recovery reads pending transactions on open. If all value records are present and checksums pass, it completes index addition or requires rebuild. If records are incomplete, it rolls back by truncating to the previous manifest offset.

Retention policies:

- `CountRetention(max_items=N)`
- `AgeRetention(max_age_seconds=N)` over stored transition event time `t`
- `DensityRetention(max_items_per_bucket=N)` after summary bucketing exists

Retention writes tombstones in the manifest. Physical compaction is a separate maintenance command and must produce a new value log and rebuilt index.

## Alternatives Considered

- Use a single embedded database for everything: simpler transactional semantics, but less transparent for content-id logs and rebuildable indexes.
- Keep all values in memory: easy for tests but contradicts the PRD's large-latent footprint concerns.
- Mutate index as source of truth: fast for search, but index formats are backend-specific and harder to audit.

## Drawbacks

- Append logs plus manifest recovery add implementation complexity.
- Tombstones mean retained stores may not shrink until compaction.
- Single-writer v0.1 limits concurrent ingestion.

## Migration / Rollout

v0.1 implements local single-writer stores, verification, and index rebuild. v0.3 adds commitment files without changing value-log item identity. v0.4 can wrap the same store behind remote protocol handlers.

## Testing Strategy

- Restart tests after writes.
- Simulated interrupted transaction before and after value append.
- Manifest checksum and value record checksum tests.
- Index rebuild from value log.
- Retention tests that prove no dangling index ids after query.
- Store verification CLI success and corruption cases.

## Resolved Bootstrap Decisions

- v0.1 retention implements count caps and event-time age windows, and may leave tombstoned records. Physical compaction is deferred to v0.2 or later because the v0.1 durability requirement is safer with append-only value logs and rebuildable indexes.

## References

- [Architecture](../spec/01-architecture.md#write-path)
- [Error Model](../spec/04-error-model.md)
- [PRD Section 9.2](../../prd.md#92-storage-and-footprint)
