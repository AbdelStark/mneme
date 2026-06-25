from __future__ import annotations

import json
from pathlib import Path

import pytest

from mneme.core import SchemaVersionError, ValidationError
from mneme.eval import (
    DATASET_REF_SCHEMA,
    EVAL_REPORT_SCHEMA,
    DatasetRef,
    EvalReport,
    validate_report_json,
    write_report_json,
)

_GOLDEN_REPORT = Path("tests/fixtures/eval/golden_report.json")


def _dataset() -> DatasetRef:
    return DatasetRef(
        dataset_id="synthetic-gate-fixture",
        kind="fixture",
        split="unit",
        version="v1",
        metadata={"source": "test"},
    )


def _report() -> EvalReport:
    return EvalReport(
        report_id="report-fixture",
        command=("mneme", "eval", "gate", "--fixture"),
        package_version="0.1.0",
        git_commit="abcdef0",
        created_at="2026-06-24T00:00:00Z",
        platform={"system": "test", "python": "3.12"},
        seed=7,
        dataset=_dataset(),
        metrics={"near_gate_min": 0.4, "case_count": 2, "status": "ok"},
        artifacts={"gate_cases": "tests/fixtures/condition/gate_behavior_cases.json"},
        caveats=("Synthetic fixture evidence cannot prove external task success.",),
        passed=True,
    )


def test_eval_report_schema_round_trips_and_validates() -> None:
    report = _report()
    encoded = report.to_json()
    decoded = validate_report_json(encoded)

    assert encoded["schema_version"] == EVAL_REPORT_SCHEMA
    assert encoded["dataset"]["schema_version"] == DATASET_REF_SCHEMA
    assert decoded == report
    assert decoded.command == ("mneme", "eval", "gate", "--fixture")


def test_write_report_json_writes_deterministic_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "gate.json"
    report = _report()

    write_report_json(report, output)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert validate_report_json(loaded) == report
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_write_report_json_rejects_nonfinite_runtime_payload(tmp_path: Path) -> None:
    output = tmp_path / "reports" / "gate.json"
    report = _report()
    object.__setattr__(report, "metrics", {"bad": float("nan")})

    with pytest.raises(ValueError, match="Out of range float values"):
        write_report_json(report, output)

    assert not output.exists()


def test_missing_caveats_fail_for_fixture_reports() -> None:
    with pytest.raises(ValidationError, match="fixture reports must include caveats"):
        EvalReport(
            report_id="missing-caveat",
            command=("mneme", "eval", "fixtures"),
            package_version="0.1.0",
            git_commit=None,
            created_at="2026-06-24T00:00:00Z",
            platform={"system": "test"},
            seed=None,
            dataset=_dataset(),
            metrics={"ok": 1},
            artifacts={},
            caveats=(),
            passed=False,
        )

    data = _report().to_json()
    data["caveats"] = []
    with pytest.raises(ValidationError, match="fixture reports must include caveats"):
        validate_report_json(data)


def test_golden_report_fixture_validates() -> None:
    report = validate_report_json(json.loads(_GOLDEN_REPORT.read_text()))

    assert report.report_id == "fixture-gate-golden"
    assert report.dataset.kind == "fixture"
    assert report.passed is True
    assert report.caveats


def test_invalid_report_schema_and_metrics_fail_closed() -> None:
    data = _report().to_json()
    data["schema_version"] = "mneme.eval_report.v2"
    with pytest.raises(SchemaVersionError, match="unsupported report schema"):
        validate_report_json(data)

    data = _report().to_json()
    data["metrics"]["bad"] = float("nan")
    with pytest.raises(ValidationError, match="metric bad"):
        validate_report_json(data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"command": object()}, "command must be a sequence"),
        ({"metrics": []}, "metrics must be an object"),
        ({"caveats": object()}, "caveats must be a sequence"),
    ),
)
def test_eval_report_constructor_rejects_malformed_collections(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = _report().to_json()
    values["dataset"] = _dataset()
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        EvalReport(**values)


def test_dataset_ref_constructor_rejects_non_string_kind() -> None:
    with pytest.raises(ValidationError, match="dataset kind"):
        DatasetRef(dataset_id="fixture", kind=[])  # type: ignore[arg-type]
