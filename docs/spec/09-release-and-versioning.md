# Release and Versioning

- Status: Accepted
- Created: 2026-06-24
- Source: [https://github.com/AbdelStark/mneme/blob/main/prd.md](https://github.com/AbdelStark/mneme/blob/main/prd.md#14-milestones)

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
- optional extras for index backends, ML adapters, receipts, and the remote
  protocol
- dependency groups for docs and development tooling
- project URLs for source, issues, security, and changelog

## Dependency Policy

Core dependencies must stay minimal. Optional runtime dependencies are loaded through extras and lazy imports.

Initial constraints:

- `numpy>=1.26`
- `torch>=2.3` only for ML adapter extras or any public type path that requires runtime tensor operations
- `faiss-cpu>=1.8` behind an index extra
- `blake3>=0.4` behind receipts or core serialization if selected for content ids
- `cryptography>=42` behind receipt signing if selected
- `uvicorn>=0.30` behind the remote extra for serving the ASGI HTTP adapter
- `pydantic>=2` only if schema validation uses it; otherwise avoid it in core
- `pytest`, `ruff`, and a type checker in dev extras

## Adapter Checkpoint Artifacts

Adapter checkpoints use a JSON sidecar named `adapter.json` plus a relative
weights file reference. The sidecar schema is `mneme.adapter_checkpoint.v1` and
must include:

- `adapter_kind`;
- JSON-compatible `adapter_config`;
- `base_fingerprint`;
- `training_report_uri`;
- `weights_file`;
- `package_version`.

Loaders must validate the sidecar before exposing the checkpoint, reject missing
required fields, reject absolute or parent-traversing weight paths, require the
weights file by default, and fail closed when the expected base fingerprint does
not match the sidecar.

## Release Gates

Every release candidate must pass:

- lint and format check;
- unit and integration tests;
- fixture evaluation report generation;
- package build and install from artifact;
- import check for minimal core install and each optional extra;
- public API compatibility snapshot and schema fixture validation;
- documentation link check;
- changelog update;
- security and license file presence.
- security boundary review confirming public docs do not claim store
  confidentiality by default and link the redaction regression tests.
- cross-source memory sharing claims link to RFC-0013 and do not imply private
  retrieval, encrypted storage, consent automation, or federation support.

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

The release compatibility gate is documented in
[API Compatibility Gate](../release/API_COMPATIBILITY.md). Public export,
signature, or persisted schema changes must update the snapshot digest there and
carry a migration note or changelog entry.

## Resolved Bootstrap Decisions

- Build backend and lockfile policy: use Hatchling as the PEP 517 build
  backend. Commit `uv.lock` for reproducible repository CI, docs, and release
  artifact validation, and include it in source artifacts. Package metadata
  remains the dependency source of truth for the built library.
- v0.1 publication policy: v0.1 ships source and wheel artifacts from the repository release only. Package-index publication starts no earlier than v0.2, after the fixture evidence, README, security boundary, and install checks are stable.
