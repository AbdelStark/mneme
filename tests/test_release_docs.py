from __future__ import annotations

from pathlib import Path


def test_release_checklist_covers_required_gates() -> None:
    checklist = Path("docs/release/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    for required in (
        "ruff check .",
        "ruff format --check .",
        "pytest",
        "mypy src/mneme",
        "python -m build",
        "Install the built wheel in a clean environment",
        "Generate a fixture report from the installed package",
        "mneme.release.validate_artifacts",
        "CHANGELOG",
        "SECURITY",
        "LICENSE",
    ):
        assert required in checklist


def test_release_notes_template_preserves_claim_boundary() -> None:
    checklist = Path("docs/release/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    normalized = " ".join(checklist.lower().split())

    assert "## release notes template" in checklist.lower()
    assert "does not claim external task success" in normalized
    assert "broad benchmark improvement" in normalized
    assert "private retrieval" in normalized
    assert "encrypted storage" in normalized
    assert "receipt verification" in normalized
    assert "stores are not confidential by default" in normalized
