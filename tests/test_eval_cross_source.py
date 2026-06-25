from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from _entrypoint_runner import run_entrypoint

from mneme._version import __version__
from mneme.eval import run_cross_source_transfer_evaluation, validate_report_json
from mneme.eval.cross_source import main as cross_source_main


def test_cross_source_report_validates_and_records_provenance() -> None:
    report = run_cross_source_transfer_evaluation(
        seed=17,
        command=("mneme", "eval", "cross-source", "--out", "report.json"),
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    decoded = validate_report_json(report.to_json())
    provenance = decoded.dataset.metadata["provenance"]

    assert decoded == report
    assert decoded.package_version == __version__
    assert decoded.seed == 17
    assert decoded.dataset.kind == "fixture"
    assert decoded.dataset.metadata["fixture_scale"] is True
    assert decoded.dataset.metadata["source_count"] == 2
    assert decoded.artifacts["report_kind"] == "cross-source-transfer"
    assert isinstance(provenance, Mapping)
    assert provenance["schema_version"] == "mneme.cross_source_receipt.v1"
    assert "returned_ids_by_source" in provenance
    assert "retrieval_receipts_by_source" in provenance
    assert (
        "verified every per-source RetrievalReceipt" in provenance["validation_steps"]
    )


def test_cross_source_report_measures_pooling_against_baselines() -> None:
    report = run_cross_source_transfer_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    assert report.metrics["source_count"] == 2
    assert report.metrics["target_case_count"] == 1
    assert report.metrics["returned_item_count"] == 2
    assert (
        report.metrics["downstream_pooled_l2"]
        < report.metrics["downstream_in_source_l2"]
    )
    assert (
        report.metrics["downstream_pooled_l2"]
        < report.metrics["downstream_no_memory_l2"]
    )
    assert report.metrics["cross_source_improvement_rate"] == 1
    assert report.metrics["negative_transfer_rate"] == 0
    assert report.metrics["source_diversity_score"] == 1.0
    assert report.metrics["receipt_verification_failure_count"] == 0
    assert report.metrics["source_a_public_fixture_proof_bytes"] > 0
    assert report.metrics["source_b_public_fixture_proof_bytes"] > 0
    assert report.passed is True


def test_cross_source_report_keeps_fixture_claim_boundary() -> None:
    report = run_cross_source_transfer_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    caveats = " ".join(report.caveats).lower()

    assert "fixture-scale" in caveats
    assert "does not claim general transfer" in caveats
    assert "external benchmark success" in caveats
    assert "do not provide confidentiality" in caveats
    assert "private retrieval" in caveats
    assert "search optimality" in caveats


def test_cross_source_eval_module_writes_valid_report_json(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "cross-source.json"

    completed = run_entrypoint(
        cross_source_main,
        "--out",
        output,
        "--seed",
        "23",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    assert completed.stdout.strip() == str(output)
    report = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert report.seed == 23
    assert report.command[:3] == ("mneme", "eval", "cross-source")
    assert report.artifacts["provenance_schema"] == "mneme.cross_source_receipt.v1"
    assert report.passed is True
