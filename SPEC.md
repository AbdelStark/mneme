# Mneme Specification

- Status: Accepted for implementation planning
- Created: 2026-06-24
- Source of intent: [prd.md](prd.md)
- Target implementation language: Python 3.11+
- First shippable milestone: v0.1

Mneme is an episodic memory and retrieval layer for latent world models. It stores realized transitions in an append-mostly memory, retrieves relevant transitions during rollout, and conditions a frozen predictor so imagined futures can stay anchored to observed experience.

This corpus is the canonical execution source for the project. The PRD remains preserved as source material; implementation decisions, public contracts, milestones, and issue decomposition are governed by the files indexed here.

## Scope

The v1 line covers a reusable Python package with typed interfaces, a local store, approximate and exact vector indexes, training-free conditioning, a trained adapter path, committed retrieval receipts, remote-store protocol messages, evaluation harnesses, release automation, and public documentation.

The first shippable milestone, v0.1, is narrower: core data types, deterministic serialization, summary keys, local flat and approximate indexes, a persistence-backed memory store without commitments, the training-free kNN corrector, a minimal PyTorch encoder adapter contract, and tests that demonstrate safe fallback behavior and characterize drift on fixture-scale rollouts.

## Corpus Index

- [Overview](docs/spec/00-overview.md)
- [Architecture](docs/spec/01-architecture.md)
- [Public API](docs/spec/02-public-api.md)
- [Data Model](docs/spec/03-data-model.md)
- [Error Model](docs/spec/04-error-model.md)
- [Observability](docs/spec/05-observability.md)
- [Security](docs/spec/06-security.md)
- [Testing Strategy](docs/spec/07-testing-strategy.md)
- [Performance Budget](docs/spec/08-performance-budget.md)
- [Release and Versioning](docs/spec/09-release-and-versioning.md)
- [Glossary](docs/spec/10-glossary.md)

## RFC Index

- [RFC-0001: Core Types and Canonical Serialization](docs/rfcs/RFC-0001-core-types-and-canonical-serialization.md)
- [RFC-0002: Encoder Fingerprints and Summary Keys](docs/rfcs/RFC-0002-encoder-fingerprints-and-summary-keys.md)
- [RFC-0003: Index Backends and Query Semantics](docs/rfcs/RFC-0003-index-backends-and-query-semantics.md)
- [RFC-0004: Memory Store Persistence and Retention](docs/rfcs/RFC-0004-memory-store-persistence-and-retention.md)
- [RFC-0005: Training-Free kNN Conditioning](docs/rfcs/RFC-0005-training-free-knn-conditioning.md)
- [RFC-0006: Trained Memory Adapter](docs/rfcs/RFC-0006-trained-memory-adapter.md)
- [RFC-0007: Commitments and Retrieval Receipts](docs/rfcs/RFC-0007-commitments-and-retrieval-receipts.md)
- [RFC-0008: Remote Store Protocol Messages](docs/rfcs/RFC-0008-remote-store-protocol-messages.md)
- [RFC-0009: Evaluation and Reproducibility Harness](docs/rfcs/RFC-0009-evaluation-and-reproducibility-harness.md)
- [RFC-0010: Packaging, CI, and Release Discipline](docs/rfcs/RFC-0010-packaging-ci-and-release-discipline.md)
- [RFC-0011: Observability and Redaction](docs/rfcs/RFC-0011-observability-and-redaction.md)
- [RFC-0012: Security Boundaries and Privacy Tiers](docs/rfcs/RFC-0012-security-boundaries-and-privacy-tiers.md)

## Milestone Map

- v0.1: local training-free memory wedge.
- v0.2: trained memory adapter and adapter-vs-corrector evaluation.
- v0.3: committed store, receipts, and replay verification.
- v0.4: remote/shared store protocol and documented public release.
- v0.5: cross-source memory sharing experiments.
- v1.0: stable API, complete reliability gates, security review, and public release criteria satisfied.

## Claim Boundary

The repository does not yet contain an implementation or benchmark evidence. Claims about drift reduction, latency, recall, adapter accuracy, receipt overhead, and downstream task success are requirements and validation targets, not achieved results. Any README, paper, release note, or issue must preserve that boundary until the corresponding evidence is produced by the evaluation harness.

## Open Questions

The active open questions are recorded in the relevant RFCs with owners and target milestones. They do not block v0.1 unless explicitly marked as v0.1-critical.
