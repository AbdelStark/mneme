from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path

import mneme


def test_package_exposes_version() -> None:
    assert mneme.__version__ == "0.1.0.dev0"


def test_project_metadata_declares_required_package_surface() -> None:
    metadata = importlib.metadata.metadata("mneme")

    assert metadata["Name"] == "mneme"
    assert metadata["Requires-Python"] == ">=3.11"
    assert "numpy>=1.26" in metadata.get_all("Requires-Dist", [])

    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    assert pyproject["project"]["license"] == {"file": "LICENSE"}
    assert sorted(pyproject["project"]["optional-dependencies"]) == [
        "dev",
        "docs",
        "index",
        "ml",
        "receipts",
        "remote",
    ]
    assert {"Source", "Issues", "Security", "Changelog"} <= set(
        pyproject["project"]["urls"]
    )


def test_core_import_avoids_optional_runtime_dependencies() -> None:
    script = (
        "import sys; "
        "import mneme; "
        "import mneme.core; "
        "blocked = {'torch', 'faiss', 'blake3', 'cryptography', 'pydantic'}; "
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
