# Release and Versioning

- Status: Accepted
- Created: 2026-06-24
- Source: [../../prd.md](../../prd.md#14-milestones)

## Version Policy

Before v1.0, minor versions map to milestone capabilities and may include breaking API changes with migration notes. After v1.0, Mneme follows semantic versioning.

Milestones:

- v0.1: local training-free memory wedge.
- v0.2: trained adapter and comparative evaluation.
- v0.3: committed store and retrieval receipts.
- v0.4: remote/shared store protocol and public examples.
- v0.5: cross-source memory sharing experiments.
- v1.0: stable public API and complete release gates.

## Package Metadata

The package metadata must define:

- name: `mneme`
- Python requirement: `>=3.11`
- license file
- README description
- typed package marker
- optional extras for index backends, ML adapters, receipts, remote protocol, docs, and dev tools
- project URLs for source, issues, security, and changelog

## Dependency Policy

Core dependencies must stay minimal. Optional runtime dependencies are loaded through extras and lazy imports.

Initial constraints:

- `numpy>=1.26`
- `torch>=2.3` only for ML adapter extras or any public type path that requires runtime tensor operations
- `faiss-cpu>=1.8` behind an index extra
- `blake3>=0.4` behind receipts or core serialization if selected for content ids
- `cryptography>=42` behind receipt signing if selected
- `pydantic>=2` only if schema validation uses it; otherwise avoid it in core
- `pytest`, `ruff`, and a type checker in dev extras

## Release Gates

Every release candidate must pass:

- lint and format check;
- unit and integration tests;
- fixture evaluation report generation;
- package build and install from artifact;
- import check for minimal core install and each optional extra;
- documentation link check;
- changelog update;
- security and license file presence.

## Changelog Discipline

Every user-visible change is recorded under `CHANGELOG.md` with:

- added, changed, deprecated, removed, fixed, and security sections when relevant;
- migration notes for breaking changes;
- claim evidence links for benchmark or performance changes.

## Deprecation

Deprecations include:

- first version that emits the warning;
- replacement API;
- earliest removal version;
- migration note.

## Open Questions

- OPEN QUESTION: Exact build backend and lockfile policy. Owner: maintainer. Target: v0.1 packaging implementation.
- OPEN QUESTION: Whether v0.1 publishes to package index or only ships source. Owner: maintainer. Target: v0.1 release planning.
