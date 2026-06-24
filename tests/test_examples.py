from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


def test_local_corrector_example_runs_and_reports_success_signal() -> None:
    payload = _run_example("local_corrector.py")

    assert payload["ok"] is True
    assert payload["example"] == "local-corrector"
    assert payload["corrected_l2"] < payload["no_memory_l2"]
    assert payload["claim_boundary"] == "fixture-only; not external benchmark evidence"


def test_remote_shared_store_example_runs_and_reports_receipt_validation() -> None:
    payload = _run_example("remote_shared_store.py")

    assert payload["ok"] is True
    assert payload["example"] == "remote-shared-store"
    assert payload["receipt_verified"] is True
    assert "deployment controls" in str(payload["security_boundary"])
    assert "not private retrieval" in str(payload["claim_boundary"])


def test_examples_readme_states_prerequisites_reports_and_boundaries() -> None:
    readme = Path("examples/README.md").read_text(encoding="utf-8")

    assert "python3 examples/local_corrector.py" in readme
    assert "python3 examples/remote_shared_store.py" in readme
    assert "mneme.cli eval fixtures --out .artifacts/examples/fixtures.json" in readme
    assert "remote-conformance" in readme
    assert "SPEC" in readme
    assert "RFC-0005" in readme
    assert "RFC-0007" in readme
    assert "RFC-0008" in readme
    assert "RFC-0010" in readme
    assert "validate_query_response" in readme
    assert "verify_retrieval_receipt" in readme
    assert "does not provide private retrieval" in readme
    assert "encrypted storage" in readme
    assert "not external benchmark evidence" in readme


def test_sdist_includes_examples_directory() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "/examples" in sdist["include"]


def _run_example(name: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(Path("examples") / name)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload
