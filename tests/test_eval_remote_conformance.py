from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mneme._version import __version__
from mneme.eval import run_remote_conformance_evaluation, validate_report_json


def test_remote_conformance_report_validates_and_records_transport() -> None:
    report = run_remote_conformance_evaluation(
        seed=7,
        command=("mneme", "eval", "remote-conformance", "--out", "report.json"),
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    decoded = validate_report_json(report.to_json())

    assert decoded == report
    assert decoded.package_version == __version__
    assert decoded.seed == 7
    assert decoded.passed is True
    assert decoded.artifacts["report_kind"] == "remote-conformance"
    assert decoded.artifacts["transport"] == "http-json-asgi"
    assert decoded.dataset.metadata["fixture_scale"] is True
    assert decoded.dataset.metadata["transport"] == "http-json-asgi"


def test_remote_conformance_report_compares_semantics_and_errors() -> None:
    report = run_remote_conformance_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    assert report.metrics["put_ids_match"] == 1
    assert report.metrics["roots_match"] == 1
    assert report.metrics["query_ids_match"] == 1
    assert report.metrics["query_distances_match"] == 1
    assert report.metrics["proofs_match"] == 1
    assert report.metrics["stats_visible_count_match"] == 1
    assert report.metrics["error_case_count"] == 2
    assert report.metrics["error_types_match"] == 1
    assert report.metrics["scenario_count"] == report.metrics["passed_scenario_count"]


def test_remote_conformance_report_keeps_fixture_claim_boundary() -> None:
    report = run_remote_conformance_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    text = " ".join(report.caveats).lower()

    assert report.dataset.kind == "fixture"
    assert "fixture-scale" in text
    assert "does not certify network deployment" in text
    assert "confidentiality" in text


def test_remote_conformance_eval_module_writes_valid_report_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reports" / "remote-conformance.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mneme.eval.remote_conformance",
            "--out",
            str(output),
            "--seed",
            "11",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    report = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert report.seed == 11
    assert report.command[:3] == ("mneme", "eval", "remote-conformance")
    assert report.artifacts["transport"] == "http-json-asgi"
    assert report.passed is True
