from __future__ import annotations

from pathlib import Path


def test_security_review_checklist_maps_required_evidence() -> None:
    review = Path("docs/release/SECURITY_REVIEW.md").read_text(encoding="utf-8")

    assert "No release-critical security blockers" in review
    for area in (
        "Persisted validation",
        "Receipts",
        "Remote responses",
        "Redaction",
        "Public docs",
        "Release artifacts",
    ):
        assert area in review
    for evidence in (
        "tests/test_remote_validation.py",
        "tests/test_remote_http.py",
        "tests/test_receipt_commitment.py",
        "tests/test_eval_replay.py",
        "tests/test_observability_events.py",
        "tests/test_release_artifacts.py",
    ):
        assert evidence in review


def test_security_review_defers_encryption_with_rationale() -> None:
    review = Path("docs/release/SECURITY_REVIEW.md").read_text(encoding="utf-8").lower()

    assert "built-in encryption at rest is deferred past v1.0" in review
    assert "key management" in review
    assert "stores are not confidential by default" in review
    assert "private retrieval and encrypted search remain research-only" in review
    assert "post-v1.0 rfc" in review


def test_security_docs_and_release_checklist_link_final_review() -> None:
    security = Path("docs/spec/06-security.md").read_text(encoding="utf-8")
    release = Path("docs/release/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    normalized_release = " ".join(release.split())

    assert "SECURITY_REVIEW.md" in security
    assert "SECURITY_REVIEW.md" in release
    assert "security slice" in normalized_release


def test_public_docs_do_not_overclaim_confidentiality() -> None:
    checked_paths = (
        "README.md",
        "SECURITY.md",
        "docs/spec/00-overview.md",
        "docs/spec/06-security.md",
        "docs/release/RELEASE_CHECKLIST.md",
        "docs/release/SECURITY_REVIEW.md",
    )
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8").lower() for path in checked_paths
    )
    normalized = " ".join(combined.split())

    assert "stores are not confidential by default" in combined
    assert "does not provide confidentiality by default" in combined
    assert "no private retrieval" in combined
    assert "no encryption at rest" in combined
    assert "encrypted stores" in combined
    assert "must not claim built-in confidentiality" in normalized
