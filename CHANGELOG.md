# Changelog

All notable user-visible changes are recorded here.

## [Unreleased]

### Changed

- Extracted local-store retention policy application into a dedicated internal
  module shared by store mutation and verification.
- Extracted CLI JSON/error runtime helpers and made CLI contract tests run
  in-process while preserving a subprocess entrypoint smoke.
- Centralized eval module entrypoint report writing and moved eval/release
  module command tests in-process.
- Raised the CI coverage ratchet to 84% for the full pytest coverage gate.

## [0.1.0] - 2026-06-25

### Added

- Installable typed `mneme` package with a `mneme` console script and uv-first
  development, CI, docs, and release workflows.
- Public README with install instructions, minimal local-store usage, current
  limitations, and specification links.
- Contributing guide with local validation gates and claim-boundary rules.
- Security policy describing the v0.x integrity and confidentiality boundary.
- Fixture-scale evaluation report command for deterministic drift and gate
  checks.
- Structured operation events with default redaction rules.
- Hosted CI gates, release checklist, and release artifact validation command.
- Optional FAISS HNSW approximate index backend behind the `index` extra.
- Count and event-time age retention policies with manifest tombstones.
- Torch-compatible `KnnCorrector` latent handling with NumPy parity coverage.
- Local profile evaluation reports for recall, latency, and footprint evidence.
- Optional `CrossAttnAdapter` trained memory module behind the `ml` extra.
- Frozen-base adapter training harness with fixture-scale evaluation reports.
- `InContextConditioner` retrieved-token baseline for compatible predictor
  wrappers.
- Adapter checkpoint metadata sidecar validation and fingerprint-checked
  checkpoint loading.
- Opt-in external benchmark runner protocol, dry-run runner, and benchmark
  report command.
- Merkle Mountain Range commitment state with inclusion proofs and local-store
  commit sidecar persistence.
- Strict MkDocs documentation site and GitHub Pages workflow.
- Code of conduct, support policy, and citation metadata for public release use.

### Security

- Persisted store validation rejects path traversal, malformed value-log records,
  invalid content ids, invalid fingerprints, and malformed array payloads with
  typed errors.
- Event redaction covers arrays, observations, paths, secrets, private dataset
  identifiers, unsafe metadata, generic bytes, and configurable content-id
  prefixes.

### Caveats

- No external benchmark, task-success, or drift-improvement claim is made by the
  fixture reports.
- Stores are not confidential by default; deployments requiring confidentiality
  must add controls outside Mneme.
