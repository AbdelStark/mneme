# v1.0 Readiness Report

- Status: Accepted for release-candidate preparation
- Reviewed: 2026-06-24
- Gate issue: [#51](https://github.com/AbdelStark/mneme/issues/51)
- Scope: release readiness evidence for Mneme v1.0, not a release
  announcement.

## Result

The v1.0 readiness gate is satisfied when the local commands and hosted CI gates
listed below pass on the release-candidate change. This report records the
required evidence links and claim boundaries for issue #51.

Publishing a release still requires tagging the release candidate, attaching or
retaining the generated artifacts, and linking the hosted CI run from the release
notes.

## Issue Gate

| Issue | Status | Evidence |
|---|---|---|
| [#30](https://github.com/AbdelStark/mneme/issues/30) release checklist and artifact validation | Closed 2026-06-24 | `docs/release/RELEASE_CHECKLIST.md`, `tests/test_release_artifacts.py` |
| [#48](https://github.com/AbdelStark/mneme/issues/48) cross-source transfer measurement | Closed 2026-06-24 | `mneme eval cross-source`, `tests/test_cross_source_provenance_docs.py` |
| [#49](https://github.com/AbdelStark/mneme/issues/49) public API compatibility checks | Closed 2026-06-24 | `docs/release/API_COMPATIBILITY.md`, `tests/test_public_api_compatibility.py` |
| [#50](https://github.com/AbdelStark/mneme/issues/50) integrity and privacy review | Closed 2026-06-24 | `docs/release/SECURITY_REVIEW.md`, `tests/test_security_review.py` |
| [#51](https://github.com/AbdelStark/mneme/issues/51) v1.0 readiness gate | This report | `docs/release/V1_READINESS_REPORT.md`, `tests/test_v1_readiness_report.py` |

Open p0 and p1 issues [#52](https://github.com/AbdelStark/mneme/issues/52)
through [#63](https://github.com/AbdelStark/mneme/issues/63) are tracking
parents with `type:tracking` and `tracking` labels. They remain intentionally
open as subsystem roadmaps and are not actionable release blockers for this
readiness gate unless a non-tracking child issue is reopened.

## Local Evidence Commands

Run these commands from a clean checkout before tagging a release candidate:

```bash
ruff check .
ruff format --check .
pytest
mypy src/mneme
rm -rf dist .artifacts/ci .ci-install
python -m build
python3 -m venv .ci-install
.ci-install/bin/python -m pip install --upgrade pip
.ci-install/bin/python -m pip install dist/*.whl
.ci-install/bin/python -c "import mneme; print(mneme.__version__)"
.ci-install/bin/python -m mneme.cli eval fixtures --out .artifacts/ci/fixtures.json
.ci-install/bin/python -m mneme.cli eval remote-conformance --out .artifacts/ci/remote-conformance.json
.ci-install/bin/python -m mneme.cli eval cross-source --out .artifacts/ci/cross-source.json
.ci-install/bin/python -m mneme.release.validate_artifacts --dist dist --fixture-report .artifacts/ci/fixtures.json --out .artifacts/ci/release-artifacts.json
```

Expected generated artifacts for version `0.1.0.dev0`:

- `dist/mneme-0.1.0.dev0-py3-none-any.whl`
- `dist/mneme-0.1.0.dev0.tar.gz`
- `.artifacts/ci/fixtures.json`
- `.artifacts/ci/remote-conformance.json`
- `.artifacts/ci/cross-source.json`
- `.artifacts/ci/release-artifacts.json`

Hosted CI evidence is the `Python 3.12 gates` workflow on the readiness PR or
release-candidate PR. The workflow builds source and wheel artifacts, installs
the built wheel in `.ci-install`, writes `.artifacts/ci/fixtures.json`, and
validates `.artifacts/ci/release-artifacts.json`.

## Evaluation Claim Boundary

The supported evaluation claims are fixture-scale only:

- `mneme eval fixtures` supports deterministic CI drift and release-gate checks.
- `mneme eval remote-conformance` supports local-vs-remote protocol fixture
  conformance checks.
- `mneme eval cross-source` supports deterministic synthetic cross-source
  transfer measurement with source identities and receipt evidence.

Mneme does not claim external task success, broad benchmark improvement, private
retrieval, encrypted search, encrypted storage, confidential stores,
remote-store security, signed provenance, or exact approximate-search optimality
from these reports. External benchmark claims require separate opt-in benchmark
artifacts.

## Documentation, Security, and Compatibility Evidence

| Area | Evidence |
|---|---|
| Public docs and examples | `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `examples/README.md`, `tests/test_public_docs.py`, `tests/test_examples.py` |
| Release checklist and artifacts | `docs/release/RELEASE_CHECKLIST.md`, `.github/workflows/ci.yml`, `tests/test_release_docs.py`, `tests/test_release_artifacts.py` |
| Security and privacy | `SECURITY.md`, `docs/spec/06-security.md`, `docs/release/SECURITY_REVIEW.md`, `tests/test_security_docs.py`, `tests/test_security_review.py` |
| Public API compatibility | `docs/release/API_COMPATIBILITY.md`, `tests/fixtures/compat/public_api_snapshot.json`, `tests/test_public_api_compatibility.py` |
| Cross-source provenance boundary | `docs/rfcs/RFC-0013-cross-source-memory-provenance.md`, `tests/test_cross_source_provenance_docs.py` |

The public documentation must continue to state that Mneme stores are not
confidential by default and that deployments requiring confidentiality need
external access control, authenticated transport, filesystem or volume
encryption, backup controls, secret management, and host isolation.

## Release Notes Evidence Links

Use the release checklist template and include these links or attached artifacts
in the release notes:

- Closing issue links: #30, #48, #49, #50, and #51.
- Hosted CI run link for the release-candidate commit.
- Built artifact names from `dist/`.
- Fixture report path or attachment: `.artifacts/ci/fixtures.json`.
- Remote conformance report path or attachment:
  `.artifacts/ci/remote-conformance.json`.
- Cross-source report path or attachment: `.artifacts/ci/cross-source.json`.
- Release artifact validation path or attachment:
  `.artifacts/ci/release-artifacts.json`.
- Security review: `docs/release/SECURITY_REVIEW.md`.
- API compatibility gate: `docs/release/API_COMPATIBILITY.md`.
