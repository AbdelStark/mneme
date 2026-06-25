from __future__ import annotations

from pathlib import Path


def test_readme_preserves_public_claim_boundary() -> None:
    readme = " ".join(Path("README.md").read_text(encoding="utf-8").lower().split())

    assert "does not claim external task success" in readme
    assert "broad benchmark improvement" in readme
    assert "fixture-scale release evidence" in readme
    assert "no external benchmark, task-success, or drift-improvement claim" in readme
    assert "stores are not confidential by default" in readme


def test_readme_includes_install_usage_limitations_and_spec_links() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "uv sync --locked --group dev" in readme
    assert "uv run mneme --help" in readme
    assert "from mneme.store import init_store, verify_store" in readme
    assert "uv run mneme eval fixtures --out .artifacts/fixtures.json" in readme
    assert "https://abdelstark.github.io/mneme/" in readme
    assert "[Public API](docs/spec/02-public-api.md)" in readme
    assert "[Security](docs/spec/06-security.md)" in readme
    assert "[Release checklist](docs/release/RELEASE_CHECKLIST.md)" in readme
    assert "[Examples](examples/README.md)" in readme
    assert "[CONTRIBUTING.md](CONTRIBUTING.md)" in readme
    assert "[Code of Conduct](CODE_OF_CONDUCT.md)" in readme
    assert "[Support](SUPPORT.md)" in readme
    assert "[Citation](CITATION.cff)" in readme


def test_governance_docs_are_present_and_claim_bounded() -> None:
    conduct = Path("CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    support = Path("SUPPORT.md").read_text(encoding="utf-8")
    citation = Path("CITATION.cff").read_text(encoding="utf-8")

    assert "Expected Behavior" in conduct
    assert "Unacceptable Behavior" in conduct
    assert "private datasets" in conduct
    assert "benchmark or production evidence" in conduct
    assert "Use Issues For" in support
    assert "Use Security Reporting For" in support
    assert "does not provide private deployment consulting" in support
    assert "cff-version: 1.2.0" in citation
    assert "version: 0.1.0" in citation


def test_contributing_lists_local_gates() -> None:
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "uv sync --locked --group dev" in contributing
    assert "uv run ruff check ." in contributing
    assert "uv run ruff format --check ." in contributing
    assert (
        "uv run pytest --cov=mneme --cov-report=term-missing --cov-fail-under=80"
        in contributing
    )
    assert "uv run mypy src/mneme" in contributing
    assert "uv run --group docs mkdocs build --strict" in contributing
    assert "uv build --out-dir dist --clear --no-build-logs" in contributing
    assert "uv run mneme eval fixtures" in contributing


def test_changelog_has_initial_unreleased_section() -> None:
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "## [Unreleased]" in changelog
    assert "## [0.1.0] - 2026-06-25" in changelog
    assert "### Added" in changelog
    assert "### Security" in changelog
    assert "### Caveats" in changelog
