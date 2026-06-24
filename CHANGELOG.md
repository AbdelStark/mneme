# Changelog

All notable user-visible changes are recorded here. Mneme is pre-1.0; breaking
changes may occur before the first stable release.

## [Unreleased]

### Added

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
