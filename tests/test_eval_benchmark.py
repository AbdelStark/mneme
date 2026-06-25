from __future__ import annotations

import json
from pathlib import Path

import pytest

from mneme.core import EvaluationError
from mneme.eval import (
    BENCHMARK_MODES,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSpec,
    DatasetRef,
    DryRunBenchmarkRunner,
    load_benchmark_dataset_ref,
    parse_benchmark_modes,
    run_external_benchmark,
    validate_report_json,
    write_external_benchmark_report,
)


def _dataset() -> DatasetRef:
    return DatasetRef(
        dataset_id="loopnav-dry-run",
        kind="external",
        split="dry-run",
        version="v0",
        uri="https://example.invalid/loopnav",
        metadata={"dry_run": True},
    )


def _write_dataset(path: Path) -> Path:
    path.write_text(json.dumps(_dataset().to_json()), encoding="utf-8")
    return path


def test_dry_run_benchmark_runner_writes_valid_external_report(
    tmp_path: Path,
) -> None:
    dataset_path = _write_dataset(tmp_path / "dataset.json")
    spec = BenchmarkSpec(
        dataset=load_benchmark_dataset_ref(dataset_path),
        dataset_manifest=str(dataset_path),
        checkpoint_uri="checkpoints/base.json",
        modes=BENCHMARK_MODES,
        command=("mneme", "eval", "benchmark", "--dry-run"),
        seed=11,
        hardware={"gpu": "none", "memory": "fixture"},
    )
    output = tmp_path / "reports" / "benchmark.json"

    report = write_external_benchmark_report(
        DryRunBenchmarkRunner(),
        spec,
        output,
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    written = validate_report_json(json.loads(output.read_text(encoding="utf-8")))

    assert written == report
    assert report.dataset.kind == "external"
    assert report.dataset.split == "dry-run"
    assert report.artifacts["report_kind"] == "external-benchmark-dry-run"
    assert report.artifacts["comparison_modes"] == ",".join(BENCHMARK_MODES)
    assert report.artifacts["model_checkpoint"] == "checkpoints/base.json"
    assert report.metrics["comparison_mode_count"] == 4
    assert report.metrics["no_memory_status"] == "dry_run"
    assert report.metrics["corrector_status"] == "dry_run"
    assert report.metrics["in_context_status"] == "dry_run"
    assert report.metrics["adapter_status"] == "dry_run"
    assert report.platform["gpu"] == "none"
    assert report.platform["memory"] == "fixture"
    assert any("opt-in evidence" in caveat for caveat in report.caveats)
    assert any("Dry-run benchmark runner" in caveat for caveat in report.caveats)


def test_missing_dataset_file_fails_with_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing-dataset.json"

    with pytest.raises(EvaluationError, match="benchmark dataset file not found"):
        load_benchmark_dataset_ref(missing)


def test_dataset_manifest_rejects_nonstandard_json_constants(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text('{"schema_version": NaN}', encoding="utf-8")

    with pytest.raises(EvaluationError, match="benchmark dataset file is not valid"):
        load_benchmark_dataset_ref(dataset_path)


def test_benchmark_spec_requires_external_dataset_split_and_valid_modes() -> None:
    fixture_dataset = DatasetRef(dataset_id="fixture", kind="fixture", split="unit")
    external_without_split = DatasetRef(dataset_id="external", kind="external")

    with pytest.raises(EvaluationError, match="kind must be external"):
        BenchmarkSpec(dataset=fixture_dataset, checkpoint_uri="checkpoint")
    with pytest.raises(EvaluationError, match="split is required"):
        BenchmarkSpec(dataset=external_without_split, checkpoint_uri="checkpoint")
    with pytest.raises(EvaluationError, match="unsupported benchmark modes"):
        BenchmarkSpec(
            dataset=_dataset(),
            checkpoint_uri="checkpoint",
            modes=("bad",),  # type: ignore[list-item]
        )


def test_custom_runner_protocol_and_mode_parser() -> None:
    class CustomRunner:
        def run(self, spec: BenchmarkSpec) -> BenchmarkResult:
            return BenchmarkResult(
                metrics={
                    "mode_count": len(spec.modes),
                    "adapter_delta": "not_measured",
                },
                artifacts={"runner": "custom"},
                caveats=("custom fixture runner only",),
                passed=False,
            )

    runner = CustomRunner()
    modes = parse_benchmark_modes("no_memory,adapter")
    report = run_external_benchmark(
        runner,
        BenchmarkSpec(
            dataset=_dataset(),
            checkpoint_uri="checkpoint",
            modes=modes,
            command=("custom", "benchmark"),
        ),
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    assert isinstance(DryRunBenchmarkRunner(), BenchmarkRunner)
    assert isinstance(runner, BenchmarkRunner)
    assert modes == ("no_memory", "adapter")
    assert report.metrics["mode_count"] == 2
    assert report.artifacts["runner"] == "custom"
    assert report.artifacts["comparison_modes"] == "no_memory,adapter"
    assert report.passed is False
