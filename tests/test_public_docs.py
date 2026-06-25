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


def test_contributing_lists_local_gates() -> None:
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "uv sync --locked --group dev" in contributing
    assert "uv run ruff check ." in contributing
    assert "uv run ruff format --check ." in contributing
    assert "uv run pytest --cov=mneme --cov-report=term-missing" in contributing
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
