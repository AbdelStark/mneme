# Overview

- Status: Accepted
- Created: 2026-06-24
- Source: [../../prd.md](../../prd.md)

## Thesis

Latent world models can predict compact representations cheaply, but long rollouts accumulate error and drift away from observed states. Mneme treats realized experience as an external episodic memory. At prediction time, it retrieves nearby transitions and conditions the world model's own prediction with those transitions under a deterministic, inspectable contract.

The project is not a new world-model architecture. It is infrastructure around existing encoders and predictors: storage, retrieval, conditioning, provenance, evaluation, and packaging.

## Primary Users

- ML researchers evaluating whether retrieval reduces latent rollout drift.
- Robotics or embodied-agent engineers who need a memory layer around a frozen predictor.
- Infrastructure contributors implementing indexes, stores, receipts, protocol messages, and release gates.
- Auditors reconstructing which memories conditioned a logged decision.

## Goals

- Provide a model-agnostic Python API for storing latent transitions and retrieving them during rollout.
- Ship a training-free conditioning path in v0.1 that can run against any conforming frozen predictor.
- Add a trained memory adapter in v0.2 without requiring gradients through the base model.
- Make memory items content-addressed and support retrieval receipts in v0.3.
- Support local stores first, then remote or shared stores through schema-versioned messages.
- Preserve numerical contracts for dtype, shape, device movement, determinism, and evaluation hygiene.
- Keep public claims tied to reproducible commands and stored artifacts.

## Non-Goals

- Mneme does not train the base world model.
- Mneme does not generate pixels or own a video-generation pipeline.
- Mneme does not claim confidentiality for v0.x stores.
- Mneme v0.x does not prove approximate search returned the exact top-k set.
- Mneme v0.1 does not target embedded or minimal-footprint deployment.
- Mneme does not define a new benchmark suite; it consumes existing tasks and fixture-scale local tests.

## v0.1 Scope

v0.1 includes:

- `mneme.core`: typed data carriers, protocols, error types, canonical serialization.
- `mneme.encode`: encoder and summarizer protocols, mean-pool summarizer, encoder fingerprints.
- `mneme.index`: flat exact index and one approximate nearest-neighbor backend.
- `mneme.store`: local append-mostly memory store, persistence manifest, retention policies.
- `mneme.condition`: training-free kNN corrector with distance-gated fallback.
- `mneme.eval`: fixture-scale drift, gate, recall, and latency checks.
- Package metadata, linting, typing, unit tests, documentation, and CI.

v0.1 excludes commitments, signatures, remote store messages, trained adapters, large external datasets, and public benchmark claims beyond fixture-scale validation.

## v1.0 Completion Criteria

- Public APIs are typed, documented, and covered by compatibility tests.
- Every persisted artifact carries a schema version and can be validated before use.
- Local and remote query paths have deterministic behavior under fixed inputs.
- Receipt verification can replay a logged conditioning set against a committed root.
- Evaluation commands produce machine-readable reports for drift, gate behavior, recall, latency, and receipt overhead.
- Security, release, and contribution docs exist and are consistent with package behavior.
- The issue tracker has no unclosed v1.0 issues marked `priority:p0` or `priority:p1`.

## Claim Boundary

The strongest current claim is that the PRD defines a plausible retrieval-memory design for latent world models. The repository has not yet shown that Mneme reduces drift or improves task success. Documentation must use "target", "requirement", "evaluation", or "hypothesis" language until the evaluation harness produces evidence.

## Risks

- RISK: Retrieval may not improve drift outside dense store coverage. Resolution: v0.1 gate tests and v0.2 coverage ablations.
- RISK: Summary keys may not preserve the semantics needed for useful neighbors. Resolution: require exact-index recall checks and summary-key ablations.
- RISK: Large latent values may dominate memory footprint. Resolution: separate index keys from values and implement retention before scaling claims.
- RISK: Remote-store semantics may drift from local-store semantics. Resolution: define shared schema and conformance tests before remote release.

## Open Questions

- OPEN QUESTION: Which external benchmark is the first non-fixture target? Owner: maintainer. Target: v0.2 planning.
- OPEN QUESTION: Which approximate index backend ships by default on each supported platform? Owner: maintainer. Target: v0.1 implementation.
