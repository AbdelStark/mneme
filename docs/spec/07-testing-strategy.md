# Testing Strategy

- Status: Accepted
- Created: 2026-06-24
- Source: [../../prd.md](../../prd.md#13-evaluation-plan)

## Test Pyramid

Unit tests cover validation, serialization, indexes, stores, conditioning math, and error mapping. Property tests cover canonical serialization, content ids, query validation, retention, and receipt verification. Integration tests cover store persistence, index rebuild, CLI commands, local query-to-condition loops, and remote protocol conformance when available. Evaluation tests produce fixture-scale reports; external benchmark runs are separate and opt-in.

## Required Test Groups

Core:

- dataclass validation rejects invalid shapes, dtypes, schema versions, metadata, and fingerprints;
- canonical serialization is deterministic across process runs;
- content id changes when key, value, metadata, or fingerprint changes;
- optional extras are not imported by `mneme.core`.

Index:

- flat index returns exact nearest neighbors for known vectors;
- approximate index recall is measured against flat ground truth;
- cosine search validates normalization;
- duplicate ids are stable and de-duplicated;
- filters and temporal decay are applied deterministically.

Store:

- `put`, `put_batch`, `query`, `stats`, and `verify` work across process restart;
- partial transaction recovery completes or rolls back;
- index rebuild restores searchability from value logs;
- retention never leaves dangling index ids;
- fingerprint mismatch fails closed.

Conditioning:

- empty retrieval returns the parametric latent exactly;
- far-neighbor gate approaches zero;
- near-neighbor delta mode moves prediction toward weighted deltas;
- non-finite distances or latents fail validation;
- torch and numpy paths preserve dtype and device contracts where applicable.

Receipts:

- valid inclusion proofs verify;
- altered item bytes fail verification;
- mismatched root fails verification;
- receipt overhead is measured on fixture stores.

Evaluation:

- fixture drift report writes valid `mneme.eval_report.v1`;
- gate behavior report includes in-distribution and out-of-distribution fixtures;
- latency report includes p50 and p99;
- all reports include command, seed, package version, and caveats.

CLI:

- every public command has a success test and at least one typed-error test;
- JSON output is valid and schema-versioned;
- exit codes match [Public API](02-public-api.md#command-line-surface).

## ML-Specific Hygiene

- Deterministic seeds are recorded.
- Evaluation code separates train, calibration, and validation slices.
- Adapter training keeps base model parameters frozen and asserts no base gradients.
- Torch inference runs under inference mode by default.
- Device and dtype conversions are explicit.
- Metrics report confidence intervals or sample counts when comparing methods.
- Synthetic fixtures are labeled as synthetic and cannot support external benchmark claims.

## CI Gates

v0.1 required gates:

```text
ruff check .
ruff format --check .
pytest
python -m mneme.eval.fixtures --out reports/fixtures.json
python -m mneme.cli store verify tests/fixtures/store
```

The exact command module names may change during implementation, but the gates above define the required coverage.

## Open Questions

- OPEN QUESTION: Whether static type checking is enforced from v0.1 or v0.2. Owner: maintainer. Target: v0.1 packaging implementation.
