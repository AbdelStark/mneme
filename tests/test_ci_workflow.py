from __future__ import annotations

from pathlib import Path


def test_hosted_ci_workflow_runs_required_gates() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "python -m ruff check ." in workflow
    assert "python -m ruff format --check ." in workflow
    assert "python -m pytest" in workflow
    assert "python -m mypy src/mneme" in workflow
    assert "python -m build" in workflow
    assert ".ci-install/bin/python -m pip install dist/*.whl" in workflow
    assert ".ci-install/bin/python -c" in workflow
    assert "mneme.cli eval fixtures --out .artifacts/ci/fixtures.json" in workflow
    assert "Validate release artifacts against checklist" in workflow
    assert "mneme.release.validate_artifacts --dist dist" in workflow
    assert "release-artifacts.json" in workflow
    assert "actions/upload-artifact@v4" in workflow


def test_contributing_documents_ci_reproduction_commands() -> None:
    contributing = Path("CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "## CI Reproduction" in contributing
    assert "Hosted CI runs on pull requests" in contributing
    assert ".venv/bin/ruff check ." in contributing
    assert ".venv/bin/ruff format --check ." in contributing
    assert ".venv/bin/pytest" in contributing
    assert ".venv/bin/mypy src/mneme" in contributing
    assert ".venv/bin/python -m build" in contributing
    assert ".ci-install/bin/python -m pip install dist/*.whl" in contributing
    assert ".ci-install/bin/python -c" in contributing
    assert "mneme.cli eval fixtures --out .artifacts/ci/fixtures.json" in contributing
    assert "mneme.release.validate_artifacts --dist dist" in contributing
    assert "release-artifacts.json" in contributing
