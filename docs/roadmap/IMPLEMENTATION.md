# Implementation Ledger - 2026-06-25

Generated from the spec corpus in PR #1: https://github.com/AbdelStark/mneme/pull/1.
Every implementable unit below was filed as a GitHub issue and linked back to a
spec section or RFC in its issue body. The v0.1.0 backlog through #63 is closed;
this page is now a release ledger, not an open work queue.

## Milestone: v0.1

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #2 | release: scaffold installable Python package | area:release | priority:p0 | effort:m | spec:rfc-0010 | Closed |
| #3 | core: implement schema-versioned public data types | area:core | priority:p0 | effort:m | spec:rfc-0001 | Closed |
| #4 | core: implement canonical serialization and content ids | area:core | priority:p0 | effort:m | spec:rfc-0001 | Closed |
| #5 | core: define typed errors and exit-code mapping | area:core | priority:p0 | effort:s | spec:rfc-0001, spec:rfc-0010 | Closed |
| #6 | encode: implement encoder and summarizer protocols | area:encode | priority:p0 | effort:s | spec:rfc-0002 | Closed |
| #7 | encode: implement mean-pool summary key generation | area:encode | priority:p0 | effort:m | spec:rfc-0002 | Closed |
| #8 | encode: implement fingerprint digests and mismatch checks | area:encode | priority:p0 | effort:m | spec:rfc-0002, spec:rfc-0001 | Closed |
| #9 | index: implement exact flat search backend | area:index | priority:p0 | effort:m | spec:rfc-0003 | Closed |
| #10 | index: add approximate backend behind optional extra | area:index | priority:p1 | effort:m | spec:rfc-0003, spec:rfc-0010 | Closed |
| #11 | index: implement QuerySpec search semantics | area:index | priority:p0 | effort:m | spec:rfc-0003, spec:rfc-0002 | Closed |
| #12 | store: implement local layout and manifest schema | area:store | priority:p0 | effort:m | spec:rfc-0004 | Closed |
| #13 | store: implement append-only value log transactions | area:store | priority:p0 | effort:l | spec:rfc-0004, spec:rfc-0001 | Closed |
| #14 | store: add verification and index rebuild commands | area:store | priority:p0 | effort:m | spec:rfc-0004, spec:rfc-0010 | Closed |
| #15 | store: implement retention policies and tombstones | area:store | priority:p1 | effort:m | spec:rfc-0004 | Closed |
| #16 | store: recover from partial transactions | area:store | priority:p0 | effort:m | spec:rfc-0004 | Closed |
| #17 | condition: implement Conditioner and CondCtx contracts | area:condition | priority:p0 | effort:s | spec:rfc-0005 | Closed |
| #18 | condition: implement kNN corrector modes | area:condition | priority:p0 | effort:m | spec:rfc-0005 | Closed |
| #19 | condition: implement distance gate fallback tests | area:condition | priority:p0 | effort:s | spec:rfc-0005, spec:rfc-0009 | Closed |
| #20 | condition: add torch and numpy parity coverage | area:condition | priority:p1 | effort:m | spec:rfc-0005 | Closed |
| #21 | eval: implement schema-versioned report model | area:eval | priority:p0 | effort:m | spec:rfc-0009 | Closed |
| #22 | eval: add fixture drift and gate reports | area:eval | priority:p0 | effort:m | spec:rfc-0009, spec:rfc-0005 | Closed |
| #23 | eval: add recall, latency, and footprint reports | area:eval | priority:p1 | effort:m | spec:rfc-0009, spec:rfc-0003 | Closed |
| #24 | observability: emit structured operation events | area:observability | priority:p1 | effort:m | spec:rfc-0011 | Closed |
| #25 | observability: enforce default redaction rules | area:observability | priority:p0 | effort:s | spec:rfc-0011, spec:rfc-0012 | Closed |
| #26 | security: implement validation and public security boundary | area:security | priority:p0 | effort:m | spec:rfc-0012 | Closed |
| #27 | cli: implement store, query, eval, and receipt command surface | area:cli | priority:p1 | effort:l | spec:rfc-0010, spec:rfc-0004, spec:rfc-0009 | Closed |
| #28 | docs: add README, security, contributing, and changelog | area:docs | priority:p0 | effort:m | spec:rfc-0010, spec:rfc-0012 | Closed |
| #29 | ci: add lint, test, build, install, and fixture gates | area:release | priority:p0 | effort:m | spec:rfc-0010, spec:rfc-0009 | Closed |
| #30 | release: add release checklist and artifact validation | area:release | priority:p1 | effort:s | spec:rfc-0010 | Closed |

## Milestone: v0.2

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #31 | adapter: implement cross-attention memory module | area:adapter | priority:p1 | effort:l | spec:rfc-0006 | Closed |
| #32 | adapter: add frozen-base training harness | area:adapter | priority:p1 | effort:l | spec:rfc-0006, spec:rfc-0009 | Closed |
| #33 | condition: add in-context retrieved-token baseline | area:condition | priority:p2 | effort:m | spec:rfc-0006 | Closed |
| #34 | adapter: define checkpoint metadata and loading | area:adapter | priority:p1 | effort:m | spec:rfc-0006, spec:rfc-0010 | Closed |
| #35 | eval: add external benchmark runner interface | area:eval | priority:p1 | effort:m | spec:rfc-0009 | Closed |

## Milestone: v0.3

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #36 | receipts: implement MMR commitment state | area:receipts | priority:p1 | effort:l | spec:rfc-0007 | Closed |
| #37 | receipts: build and verify retrieval receipts | area:receipts | priority:p1 | effort:l | spec:rfc-0007 | Closed |
| #38 | store: add committed-store upgrade path | area:store | priority:p1 | effort:m | spec:rfc-0007, spec:rfc-0004 | Closed |
| #39 | eval: measure receipt overhead and proof size | area:eval | priority:p2 | effort:m | spec:rfc-0007, spec:rfc-0009 | Closed |
| #40 | eval: add receipt-backed replay harness | area:eval | priority:p1 | effort:m | spec:rfc-0007, spec:rfc-0009 | Closed |

## Milestone: v0.4

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #41 | remote: define schema-versioned store messages | area:remote | priority:p1 | effort:m | spec:rfc-0008 | Closed |
| #42 | remote: validate responses before conditioning | area:remote | priority:p1 | effort:m | spec:rfc-0008, spec:rfc-0012 | Closed |
| #43 | remote: implement first transport adapter | area:remote | priority:p1 | effort:l | spec:rfc-0008 | Closed |
| #44 | remote: add local-vs-remote conformance suite | area:remote | priority:p1 | effort:m | spec:rfc-0008, spec:rfc-0009 | Closed |
| #45 | security: document shared-store deployment guidance | area:security | priority:p1 | effort:s | spec:rfc-0008, spec:rfc-0012 | Closed |
| #46 | docs: add end-to-end public examples | area:docs | priority:p1 | effort:m | spec:rfc-0010, spec:rfc-0008 | Closed |

## Milestone: v0.5

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #47 | security: design cross-source memory provenance | area:security | priority:p2 | effort:m | spec:rfc-0012, spec:rfc-0007 | Closed |
| #48 | eval: measure cross-source memory transfer | area:eval | priority:p2 | effort:l | spec:rfc-0009, spec:rfc-0012 | Closed |

## Milestone: v1.0

| # | Title | Area | Priority | Effort | RFC | Status |
|---|-------|------|----------|--------|-----|--------|
| #49 | release: add public API compatibility checks | area:release | priority:p1 | effort:m | spec:rfc-0010, spec:rfc-0001 | Closed |
| #50 | security: complete v1.0 integrity and privacy review | area:security | priority:p1 | effort:m | spec:rfc-0012, spec:rfc-0007 | Closed |
| #51 | release: run v1.0 release readiness gate | area:release | priority:p0 | effort:l | spec:rfc-0010, spec:rfc-0009 | Closed |

## Tracking Issues

- #52 [Tracking] Core types and serialization
- #53 [Tracking] Encoder fingerprints and summary keys
- #54 [Tracking] Index backends and query semantics
- #55 [Tracking] Memory store persistence and retention
- #56 [Tracking] Training-free conditioning
- #57 [Tracking] Trained memory adapter
- #58 [Tracking] Commitments and retrieval receipts
- #59 [Tracking] Remote store protocol
- #60 [Tracking] Evaluation and reproducibility
- #61 [Tracking] Packaging, CI, and release
- #62 [Tracking] Observability and redaction
- #63 [Tracking] Security boundaries and privacy

## Cross-Cutting Dependencies

- #2 blocks most v0.1 implementation because package metadata and source layout must exist before module work lands.
- #3 and #4 block store, receipt, remote, and compatibility work because all persisted identity flows through core types and canonical serialization.
- #8 and #11 block store query behavior because keys must be fingerprint-safe and query semantics must be stable before persistence can rely on them.
- #13 blocks retention, recovery, receipts, and committed-store upgrade work because value-log transactions define append order and durability.
- #18 blocks v0.1 fixture evaluation because the training-free corrector is the first measurable memory behavior.
- #21 blocks all evaluation reports because every evidence artifact uses the shared report schema.
- #25 and #26 block public docs and remote work because logs, reports, and persisted stores must preserve the explicit privacy boundary.
- #37 blocks remote response validation and replay because receipt verification is the integrity primitive used by later audit paths.
- #41 and #42 block the first remote transport because schema and fail-closed client validation must precede transport-specific behavior.
- #49 and #50 block #51 because v1.0 readiness requires API compatibility and final security review.
