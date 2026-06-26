from __future__ import annotations

import ast
import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path

import mneme


def test_package_exposes_version() -> None:
    assert mneme.__version__ == "0.1.0"


def test_project_metadata_declares_required_package_surface() -> None:
    metadata = importlib.metadata.metadata("mneme")

    assert metadata["Name"] == "mneme"
    assert metadata["Requires-Python"] == ">=3.11"
    assert "numpy>=1.26" in metadata.get_all("Requires-Dist", [])

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    assert pyproject["project"]["license"] == {"file": "LICENSE"}
    assert sorted(pyproject["project"]["optional-dependencies"]) == [
        "index",
        "ml",
        "receipts",
        "remote",
    ]
    assert pyproject["project"]["scripts"]["mneme"] == "mneme.cli.__main__:main"
    assert sorted(pyproject["dependency-groups"]) == ["dev", "docs"]
    assert "pytest-cov>=5.0" in pyproject["dependency-groups"]["dev"]
    assert "mkdocs-material>=9.5" in pyproject["dependency-groups"]["docs"]
    assert {"Documentation", "Source", "Issues", "Security", "Changelog"} <= set(
        pyproject["project"]["urls"]
    )


def test_uv_lock_and_docs_site_are_source_artifacts() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert Path("uv.lock").exists()
    assert "/CITATION.cff" in sdist["include"]
    assert "/CODE_OF_CONDUCT.md" in sdist["include"]
    assert "/SUPPORT.md" in sdist["include"]
    assert "/uv.lock" in sdist["include"]
    assert "/mkdocs.yml" in sdist["include"]
    assert "/docs" in sdist["include"]
    assert "/examples" in sdist["include"]


def test_core_import_avoids_optional_runtime_dependencies() -> None:
    script = (
        "import sys; "
        "import mneme; "
        "import mneme.core; "
        "blocked = {'torch', 'faiss', 'cryptography', 'pydantic'}; "
        "loaded = sorted(blocked & set(sys.modules)); "
        "print(','.join(loaded)); "
        "raise SystemExit(1 if loaded else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_production_json_uses_strict_helper_boundary() -> None:
    allowed = Path("src/mneme/core/_json.py")
    offenders: list[str] = []

    for path in Path("src/mneme").rglob("*.py"):
        if path == allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"dumps", "loads"}
                and isinstance(func.value, ast.Name)
                and func.value.id == "json"
            ):
                offenders.append(f"{path}:{node.lineno} json.{func.attr}")

    assert offenders == []
