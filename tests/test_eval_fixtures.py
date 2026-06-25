from __future__ import annotations

import json
from pathlib import Path

from _entrypoint_runner import run_entrypoint

from mneme.core import CliExitCode
from mneme.eval import run_fixture_evaluation, validate_report_json
from mneme.eval.fixtures import main as fixtures_main


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

    completed = run_entrypoint(
        fixtures_main,
        "--out",
        output,
        "--seed",
        "42",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    assert completed.stdout.strip() == str(output)
    report = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert report.seed == 42
    assert report.command[:4] == ("mneme", "eval", "fixtures", "--out")
    assert report.passed is True


def test_fixture_eval_module_reports_typed_write_error(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")

    completed = run_entrypoint(
        fixtures_main,
        "--out",
        blocked_parent / "fixtures.json",
    )

    assert completed.returncode == int(CliExitCode.INTERNAL)
    assert completed.stdout == ""
    assert "EvaluationError: failed to write fixture report" in completed.stderr
    assert "Traceback" not in completed.stderr
