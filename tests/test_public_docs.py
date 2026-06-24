from __future__ import annotations

from pathlib import Path


def test_readme_preserves_public_claim_boundary() -> None:
    readme = " ".join(Path("README.md").read_text(encoding="utf-8").lower().split())

    assert "does not claim external task success" in readme
    assert "broad benchmark improvement" in readme
    assert "fixture reports are useful for ci" in readme
    assert "no external benchmark or drift-improvement claim" in readme
    assert "stores are not confidential by default" in readme


def test_readme_includes_install_usage_limitations_and_spec_links() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "python3 -m pip install -e ." in readme
    assert "from mneme.store import init_store, verify_store" in readme
    assert "python -m mneme.eval.fixtures --out .artifacts/fixtures.json" in readme
    assert "[Public API](docs/spec/02-public-api.md)" in readme
    assert "[Security](docs/spec/06-security.md)" in readme
    assert "[Release checklist](docs/release/RELEASE_CHECKLIST.md)" in readme
    assert "[Examples](examples/README.md)" in readme
    assert "[CONTRIBUTING.md](CONTRIBUTING.md)" in readme


def test_contributing_lists_local_gates() -> None:
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")

    assert ".venv/bin/ruff check ." in contributing
    assert ".venv/bin/ruff format --check ." in contributing
    assert ".venv/bin/pytest" in contributing
    assert ".venv/bin/mypy src/mneme" in contributing
    assert ".venv/bin/python -m build" in contributing
    assert "python -m mneme.eval.fixtures" in contributing


def test_changelog_has_initial_unreleased_section() -> None:
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "## [Unreleased]" in changelog
    assert "### Added" in changelog
    assert "### Security" in changelog
    assert "### Caveats" in changelog
