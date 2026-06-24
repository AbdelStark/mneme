from __future__ import annotations

from pathlib import Path

REPORT = Path("docs/release/V1_READINESS_REPORT.md")


def test_v1_readiness_report_links_dependency_issues_and_tracking_boundary() -> None:
    report = REPORT.read_text(encoding="utf-8")
    normalized = " ".join(report.split())

    for issue in ("#30", "#48", "#49", "#50", "#51"):
        assert f"https://github.com/AbdelStark/mneme/issues/{issue[1:]}" in report

    assert "Closed 2026-06-24" in report
    assert "#52" in report
    assert "#63" in report
    assert "type:tracking" in report
    assert "not actionable release blockers" in normalized
    assert "unless a non-tracking child issue is reopened" in normalized


def test_v1_readiness_report_records_release_artifact_and_install_evidence() -> None:
    report = REPORT.read_text(encoding="utf-8")

    for evidence in (
        "ruff check .",
        "ruff format --check .",
        "pytest",
        "mypy src/mneme",
        "python -m build",
        ".ci-install/bin/python -m pip install dist/*.whl",
        '.ci-install/bin/python -c "import mneme; print(mneme.__version__)"',
        "dist/mneme-0.1.0.dev0-py3-none-any.whl",
        "dist/mneme-0.1.0.dev0.tar.gz",
        ".artifacts/ci/fixtures.json",
        ".artifacts/ci/remote-conformance.json",
        ".artifacts/ci/cross-source.json",
        ".artifacts/ci/release-artifacts.json",
        "mneme.release.validate_artifacts",
        "Python 3.12 gates",
    ):
        assert evidence in report


def test_v1_readiness_report_preserves_evaluation_claim_boundary() -> None:
    report = " ".join(REPORT.read_text(encoding="utf-8").lower().split())

    for boundary in (
        "fixture-scale only",
        "does not claim external task success",
        "broad benchmark improvement",
        "private retrieval",
        "encrypted search",
        "encrypted storage",
        "confidential stores",
        "remote-store security",
        "signed provenance",
        "exact approximate-search optimality",
        "external benchmark claims require separate opt-in benchmark artifacts",
        "stores are not confidential by default",
    ):
        assert boundary in report


def test_v1_readiness_report_links_docs_tests_and_release_notes_evidence() -> None:
    report = REPORT.read_text(encoding="utf-8")

    for evidence in (
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "examples/README.md",
        ".github/workflows/ci.yml",
        "docs/release/RELEASE_CHECKLIST.md",
        "docs/release/SECURITY_REVIEW.md",
        "docs/release/API_COMPATIBILITY.md",
        "docs/rfcs/RFC-0013-cross-source-memory-provenance.md",
        "tests/test_public_docs.py",
        "tests/test_examples.py",
        "tests/test_release_docs.py",
        "tests/test_release_artifacts.py",
        "tests/test_security_docs.py",
        "tests/test_security_review.py",
        "tests/test_public_api_compatibility.py",
        "tests/test_cross_source_provenance_docs.py",
        "Release Notes Evidence Links",
        "Hosted CI run link",
    ):
        assert evidence in report


def test_release_checklist_links_v1_readiness_report() -> None:
    checklist = Path("docs/release/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    normalized = " ".join(checklist.split())

    assert "V1_READINESS_REPORT.md" in checklist
    assert (
        "issue, evaluation, documentation, security, API compatibility, and CI"
        in normalized
    )
