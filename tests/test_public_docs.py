from __future__ import annotations

import ast
import tomllib
from collections import Counter
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
        "uv run pytest --cov=mneme --cov-report=term-missing --cov-fail-under=84"
        in contributing
    )
    assert "uv run mypy src/mneme" in contributing
    assert "uv run --group docs mkdocs build --strict" in contributing
    assert "uv build --out-dir dist --clear --no-build-logs" in contributing
    assert "uv run mneme eval fixtures" in contributing


def test_docs_use_shipped_remote_package_and_optional_extras() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = sorted(pyproject["project"]["optional-dependencies"])
    docs = "\n".join(
        [
            Path("prd.md").read_text(encoding="utf-8"),
            Path("docs/spec/01-architecture.md").read_text(encoding="utf-8"),
            Path("docs/rfcs/RFC-0008-remote-store-protocol-messages.md").read_text(
                encoding="utf-8"
            ),
        ]
    )

    assert extras == ["index", "ml", "receipts", "remote"]
    for extra in extras:
        assert f"mneme[{extra}]" in docs
    assert "`mneme.remote`" in docs
    assert "mneme.wmcp" not in docs
    assert "mneme[faiss]" not in docs
    assert "mneme[hnswlib]" not in docs
    assert "mneme[verifiable]" not in docs
    assert "hnswlib backend" not in docs


def test_public_api_core_types_block_has_unique_declarations() -> None:
    public_api = Path("docs/spec/02-public-api.md").read_text(encoding="utf-8")
    core_types = public_api.split("## Core Types", maxsplit=1)[1].split(
        "## Protocols",
        maxsplit=1,
    )[0]
    code_block = core_types.split("```python", maxsplit=1)[1].split(
        "```",
        maxsplit=1,
    )[0]
    tree = ast.parse(code_block)
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    duplicates = sorted(
        class_name for class_name, count in Counter(classes).items() if count > 1
    )

    assert duplicates == []
    assert 'schema_version: str = "mneme.query_spec.v1"' in code_block


def test_changelog_has_initial_unreleased_section() -> None:
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "## [Unreleased]" in changelog
    assert "## [0.1.0] - 2026-06-25" in changelog
    assert "### Added" in changelog
    assert "### Security" in changelog
    assert "### Caveats" in changelog
