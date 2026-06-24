# RFC-0010: Packaging, CI, and Release Discipline

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme is packaged as a typed Python project with minimal core dependencies, optional extras, deterministic test gates, release artifacts, changelog discipline, and public documentation that does not overclaim beyond evaluation evidence.

## Motivation

[Release and Versioning](../spec/09-release-and-versioning.md) requires package metadata, optional extras, release gates, and changelog discipline. The PRD targets adoption as a maintained Python package, which requires installable artifacts and reliable CI before model or benchmark work can be trusted.

## Goals

- Define package metadata and optional extras.
- Keep core imports lightweight.
- Establish lint, format, test, typing, and build gates.
- Require release notes and changelog entries.
- Keep documentation claims tied to evidence reports.

## Non-Goals

- Publish v0.1 before the v0.1 implementation issues close.
- Freeze all dependencies forever.
- Require every optional extra in minimal CI.

## Proposed Design

Project files:

```text
pyproject.toml
README.md
LICENSE
CHANGELOG.md
SECURITY.md
CONTRIBUTING.md
src/mneme/
tests/
docs/
```

Package metadata:

- Python `>=3.11`;
- typed package marker `py.typed`;
- optional extras for index backends, ML adapters, receipts, remote protocol, docs, and dev;
- source, issue, changelog, and security URLs;
- license file included in source and wheel.

CI gates:

```text
ruff check .
ruff format --check .
pytest
python -m build
python -m pip install dist/*.whl
python -c "import mneme; print(mneme.__version__)"
python -m mneme.eval.fixtures --out reports/fixtures.json
```

Documentation gates:

- README links to SPEC and does not cite benchmark claims without reports.
- API docs list public protocols and constructors.
- CONTRIBUTING lists setup, tests, style, and issue workflow.
- SECURITY states v0.x integrity/confidentiality boundary.
- CHANGELOG includes user-visible changes.

## Alternatives Considered

- Delay packaging until after research code works: faster experimentation, but makes public API drift harder to control.
- Make all dependencies mandatory: simpler imports, but heavy installs and worse adoption.
- Skip typing until v1.0: less initial work, but public protocol correctness is central to the project.

## Drawbacks

- Packaging and CI add work before algorithmic experiments.
- Optional extras increase test matrix complexity.
- Type gates can slow early refactors if configured too strictly.

## Migration / Rollout

v0.1 creates package scaffolding and minimal CI. Optional extras are tested in a matrix as they are added. Release candidates must pass build and install checks before tag creation.

## Testing Strategy

- Build source and wheel artifacts.
- Install from wheel in a clean environment.
- Import minimal core without optional extras.
- Test each optional extra in at least one CI job when implemented.
- Validate README claim links against evaluation reports.

## Open Questions

- OPEN QUESTION: Exact build backend and lockfile policy. Owner: maintainer. Target: v0.1 packaging implementation.
- OPEN QUESTION: Whether type checking is required in v0.1 CI or introduced in v0.2. Owner: maintainer. Target: v0.1 CI implementation.

## References

- [Release and Versioning](../spec/09-release-and-versioning.md)
- [Testing Strategy](../spec/07-testing-strategy.md#ci-gates)
- [PRD Section 11.3](../../prd.md#113-packaging-tooling-and-a-native-core-later)
