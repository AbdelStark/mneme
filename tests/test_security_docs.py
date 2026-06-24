from __future__ import annotations

from pathlib import Path


def test_security_docs_state_confidentiality_boundary_and_redaction_link() -> None:
    security = Path("SECURITY.md").read_text(encoding="utf-8").lower()

    assert "does not provide confidentiality by default" in security
    assert "persisted stores are untrusted" in security
    assert "tests/test_observability_events.py" in security
    assert "redacted" in security


def test_release_checklist_includes_security_boundary_review() -> None:
    release = Path("docs/spec/09-release-and-versioning.md").read_text(encoding="utf-8")

    assert "security boundary review" in release
    assert "do not claim store" in release
    assert "redaction regression tests" in release
