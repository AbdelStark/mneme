from __future__ import annotations

from pathlib import Path


def test_release_checklist_covers_required_gates() -> None:
    checklist = Path("docs/release/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    for required in (
        "ruff check .",
        "ruff format --check .",
        "pytest --cov=mneme --cov-report=term-missing --cov-fail-under=84",
        "mypy src/mneme",
        "mkdocs build --strict",
        "uv build",
        "Install the built wheel in a clean environment",
        "Generate a fixture report from the installed package",
        "mneme.release.validate_artifacts",
        "uv.lock",
        "mkdocs.yml",
        "examples",
        "CHANGELOG",
        "SECURITY",
        "LICENSE",
    ):
        assert required in checklist


def test_release_specs_match_committed_lockfile_policy() -> None:
    docs = "\n".join(
        [
            Path("docs/spec/09-release-and-versioning.md").read_text(encoding="utf-8"),
            Path("docs/rfcs/RFC-0010-packaging-ci-and-release-discipline.md").read_text(
                encoding="utf-8"
            ),
        ]
    )
    normalized = " ".join(docs.lower().split())

    assert "commit `uv.lock` for reproducible repository ci" in normalized
    assert "package metadata remains the dependency source of truth" in normalized
    assert "dependency groups for docs and development tooling" in normalized
    assert "do not commit a library lockfile" not in normalized
    assert "do not commit a lockfile for the library" not in normalized


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
