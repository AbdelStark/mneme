from __future__ import annotations

from pathlib import Path


def test_hosted_ci_workflow_runs_required_gates() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "astral-sh/setup-uv@v8.2.0" in workflow
    assert "uv lock --check" in workflow
    assert "uv sync --locked --group dev" in workflow
    assert "uv run ruff check ." in workflow
    assert "uv run ruff format --check ." in workflow
    assert (
        "uv run pytest --cov=mneme --cov-report=term-missing --cov-fail-under=80"
        in workflow
    )
    assert "uv run mypy src/mneme" in workflow
    assert "uv build --out-dir dist --clear --no-build-logs" in workflow
    assert "uv pip install --python .ci-install/bin/python dist/*.whl" in workflow
    assert ".ci-install/bin/python -c" in workflow
    assert (
        ".ci-install/bin/mneme eval fixtures --out .artifacts/ci/fixtures.json"
        in workflow
    )
    assert "Validate release artifacts against checklist" in workflow
    assert "mneme.release.validate_artifacts --dist dist" in workflow
    assert "release-artifacts.json" in workflow
    assert "actions/upload-artifact@v4" in workflow


def test_docs_workflow_builds_and_deploys_github_pages() -> None:
    workflow = Path(".github/workflows/docs.yml").read_text(encoding="utf-8")

    assert "name: Docs" in workflow
    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "astral-sh/setup-uv@v8.2.0" in workflow
    assert "uv lock --check" in workflow
    assert "uv sync --locked --group docs" in workflow
    assert "uv run mkdocs build --strict" in workflow
    assert "actions/configure-pages@v6.0.0" in workflow
    assert "actions/upload-pages-artifact@v5.0.0" in workflow
    assert "actions/deploy-pages@v5.0.0" in workflow


def test_contributing_documents_ci_reproduction_commands() -> None:
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "## CI Reproduction" in contributing
    assert "Hosted CI runs on pull requests" in contributing
    assert "uv sync --locked --group dev" in contributing
    assert "uv lock --check" in contributing
    assert "uv run ruff check ." in contributing
    assert "uv run ruff format --check ." in contributing
    assert (
        "uv run pytest --cov=mneme --cov-report=term-missing --cov-fail-under=80"
        in contributing
    )
    assert "uv run mypy src/mneme" in contributing
    assert "uv run mkdocs build --strict" in contributing
    assert "uv build --out-dir dist --clear --no-build-logs" in contributing
    assert "uv pip install --python .ci-install/bin/python dist/*.whl" in contributing
    assert ".ci-install/bin/python -c" in contributing
    assert (
        ".ci-install/bin/mneme eval fixtures --out .artifacts/ci/fixtures.json"
        in contributing
    )
    assert "mneme.release.validate_artifacts --dist dist" in contributing
    assert "release-artifacts.json" in contributing
