from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mneme.eval import run_fixture_evaluation, validate_report_json


def test_fixture_evaluation_report_validates_and_records_seed_caveats() -> None:
    report = run_fixture_evaluation(
        seed=123,
        command=("mneme", "eval", "fixtures", "--out", "report.json"),
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    decoded = validate_report_json(report.to_json())

    assert decoded == report
    assert decoded.seed == 123
    assert decoded.dataset.kind == "fixture"
    assert decoded.dataset.metadata["synthetic"] is True
    assert decoded.dataset.metadata["fixture_scale"] is True
    assert decoded.caveats


def test_fixture_report_contains_drift_and_gate_metrics() -> None:
    report = run_fixture_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    assert report.metrics["corrector_l2"] < report.metrics["no_memory_l2"]
    assert report.metrics["corrector_improves_fixture"] == 1
    assert report.metrics["gate_in_distribution_near"] > 0.1
    assert report.metrics["gate_out_of_distribution_far"] < 1e-6
    assert report.metrics["gate_in_distribution_case_count"] == 1
    assert report.metrics["gate_out_of_distribution_case_count"] == 1


def test_fixture_report_does_not_imply_broad_benchmark_claim() -> None:
    report = run_fixture_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    text = " ".join(report.caveats).lower()

    assert "fixture" in text
    assert "cannot prove external task success" in text
    assert report.dataset.kind == "fixture"
    assert report.artifacts["report_kind"] == "fixture-scale"


def test_fixture_eval_module_writes_valid_report_json(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "fixtures.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mneme.eval.fixtures",
            "--out",
            str(output),
            "--seed",
            "42",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    report = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert report.seed == 42
    assert report.command[:4] == ("mneme", "eval", "fixtures", "--out")
    assert report.passed is True
