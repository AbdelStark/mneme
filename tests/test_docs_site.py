from __future__ import annotations

import tomllib
from pathlib import Path


def test_mkdocs_site_declares_public_docsite_and_core_nav() -> None:
    config = Path("mkdocs.yml").read_text(encoding="utf-8")

    assert "site_name: Mneme" in config
    assert "site_url: https://abdelstark.github.io/mneme/" in config
    assert "theme:" in config
    assert "name: material" in config
    for page in (
        "index.md",
        "getting-started.md",
        "cli.md",
        "examples.md",
        "evaluation.md",
        "spec/02-public-api.md",
        "spec/06-security.md",
        "release/RELEASE_CHECKLIST.md",
    ):
        assert page in config


def test_docs_pages_are_uv_first_and_claim_bounded() -> None:
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "docs/index.md",
            "docs/getting-started.md",
            "docs/cli.md",
            "docs/examples.md",
            "docs/evaluation.md",
        )
    )
    normalized = " ".join(combined.lower().split())

    assert "uv sync --locked --group dev" in combined
    assert "uv run mneme" in combined
    assert "uv build --out-dir dist --clear --no-build-logs" in combined
    assert "does not claim external task success" in normalized
    assert "not external benchmark evidence" in normalized
    assert "stores are not confidential by default" in normalized


def test_docs_dependencies_are_uv_dependency_group_not_runtime_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "docs" not in pyproject["project"]["optional-dependencies"]
    assert "docs" in pyproject["dependency-groups"]
    assert "mkdocs-material>=9.5" in pyproject["dependency-groups"]["docs"]
