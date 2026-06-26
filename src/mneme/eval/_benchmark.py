"""Opt-in external benchmark runner interfaces."""

from __future__ import annotations

import os
import platform as platform_module
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias, runtime_checkable

from mneme._version import __version__
from mneme.core import EvaluationError, MnemeError
from mneme.core._json import loads_strict_json
from mneme.core._time import utc_now_iso
from mneme.eval._reports import DatasetRef, EvalMetric, EvalReport, write_report_json

BenchmarkMode: TypeAlias = Literal["no_memory", "corrector", "in_context", "adapter"]
BENCHMARK_MODES: Final[tuple[BenchmarkMode, ...]] = (
    "no_memory",
    "corrector",
    "in_context",
    "adapter",
)

_EXTERNAL_BENCHMARK_CAVEAT: Final = (
    "External benchmark reports are opt-in evidence artifacts and do not by "
    "themselves prove broad task success or drift improvement."
)
_DRY_RUN_CAVEAT: Final = (
    "Dry-run benchmark runner validates report plumbing only; it does not "
    "execute an external benchmark dataset or model checkpoint."
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Inputs supplied to an external benchmark runner."""

    dataset: DatasetRef
    checkpoint_uri: str
    modes: Sequence[BenchmarkMode] = BENCHMARK_MODES
    command: Sequence[str] = ("mneme", "eval", "benchmark")
    seed: int | None = None
    hardware: Mapping[str, str] = field(default_factory=dict)
    dataset_manifest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetRef):
            raise EvaluationError("benchmark dataset must be a DatasetRef")
        if self.dataset.kind != "external":
            raise EvaluationError("benchmark dataset kind must be external")
        if self.dataset.split is None:
            raise EvaluationError("benchmark dataset split is required")
        _require_non_empty_str(self.checkpoint_uri, "checkpoint_uri")
        object.__setattr__(self, "modes", _mode_tuple(self.modes))
        object.__setattr__(self, "command", _string_tuple(self.command, "command"))
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise EvaluationError("seed must be None or an integer")
        object.__setattr__(
            self,
            "hardware",
            _freeze_string_mapping(self.hardware, "hardware"),
        )
        if self.dataset_manifest is not None:
            _require_non_empty_str(self.dataset_manifest, "dataset_manifest")


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics, artifacts, caveats, and pass/fail status from a runner."""

    metrics: Mapping[str, EvalMetric]
    artifacts: Mapping[str, str] = field(default_factory=dict)
    caveats: Sequence[str] = ()
    passed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics))
        object.__setattr__(
            self,
            "artifacts",
            _freeze_string_mapping(self.artifacts, "artifacts"),
        )
        object.__setattr__(self, "caveats", _string_tuple(self.caveats, "caveats"))
        if not isinstance(self.passed, bool):
            raise EvaluationError("passed must be a bool")


@runtime_checkable
class BenchmarkRunner(Protocol):
    """Protocol implemented by opt-in external benchmark runners."""

    def run(self, spec: BenchmarkSpec) -> BenchmarkResult:
        """Run a benchmark and return report-ready metrics."""
        ...


@dataclass(frozen=True)
class DryRunBenchmarkRunner:
    """Fixture runner that validates benchmark report plumbing only."""

    runner_id: str = "dry_run"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runner_id",
            _require_non_empty_str(self.runner_id, "runner_id"),
        )

    def run(self, spec: BenchmarkSpec) -> BenchmarkResult:
        """Return deterministic dry-run metrics for all requested modes."""

        metrics: dict[str, EvalMetric] = {
            "dry_run": 1,
            "comparison_mode_count": len(spec.modes),
        }
        for mode in spec.modes:
            metrics[f"{mode}_status"] = "dry_run"
            metrics[f"{mode}_case_count"] = 0
            metrics[f"{mode}_score"] = "not_run"
        artifacts = {
            "report_kind": "external-benchmark-dry-run",
            "runner": self.runner_id,
            "comparison_modes": ",".join(spec.modes),
            "model_checkpoint": spec.checkpoint_uri,
        }
        if spec.dataset_manifest is not None:
            artifacts["dataset_manifest"] = spec.dataset_manifest
        return BenchmarkResult(
            metrics=metrics,
            artifacts=artifacts,
            caveats=(_DRY_RUN_CAVEAT,),
            passed=True,
        )


def parse_benchmark_modes(value: str) -> tuple[BenchmarkMode, ...]:
    """Parse a comma-separated benchmark comparison mode list."""

    if not isinstance(value, str):
        raise EvaluationError("benchmark modes must be a comma-separated string")
    raw_modes = tuple(part.strip() for part in value.split(",") if part.strip())
    return _mode_tuple(raw_modes)


def load_benchmark_dataset_ref(path: str | Path) -> DatasetRef:
    """Load an external benchmark dataset manifest."""

    manifest_path = Path(path)
    try:
        data = loads_strict_json(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(
            f"benchmark dataset file not found: {manifest_path}"
        ) from exc
    except OSError as exc:
        raise EvaluationError(
            f"benchmark dataset file could not be read: {manifest_path}"
        ) from exc
    except ValueError as exc:
        raise EvaluationError(
            f"benchmark dataset file is not valid JSON: {manifest_path}"
        ) from exc
    if isinstance(data, Mapping) and "dataset" in data:
        data = data["dataset"]
    try:
        dataset = DatasetRef.from_json(data)
    except MnemeError as exc:
        raise EvaluationError(
            f"benchmark dataset manifest is invalid: {manifest_path}"
        ) from exc
    if dataset.kind != "external":
        raise EvaluationError("benchmark dataset kind must be external")
    if dataset.split is None:
        raise EvaluationError("benchmark dataset split is required")
    return dataset


def run_external_benchmark(
    runner: BenchmarkRunner,
    spec: BenchmarkSpec,
    *,
    created_at: str | None = None,
    git_commit: str | None = None,
    package_version: str = __version__,
) -> EvalReport:
    """Run an opt-in benchmark runner and wrap its output in an EvalReport."""

    if not isinstance(spec, BenchmarkSpec):
        raise EvaluationError("spec must be a BenchmarkSpec")
    result = runner.run(spec)
    if not isinstance(result, BenchmarkResult):
        raise EvaluationError("benchmark runner must return BenchmarkResult")
    artifacts = dict(result.artifacts)
    artifacts.setdefault("report_kind", "external-benchmark")
    artifacts.setdefault("comparison_modes", ",".join(spec.modes))
    artifacts.setdefault("model_checkpoint", spec.checkpoint_uri)
    if spec.dataset_manifest is not None:
        artifacts.setdefault("dataset_manifest", spec.dataset_manifest)
    caveats = (_EXTERNAL_BENCHMARK_CAVEAT, *result.caveats)
    return EvalReport(
        report_id="mneme-external-benchmark-v1",
        command=spec.command,
        package_version=package_version,
        git_commit=_detect_git_commit() if git_commit is None else git_commit,
        created_at=utc_now_iso() if created_at is None else created_at,
        platform=_benchmark_platform_summary(spec.hardware),
        seed=spec.seed,
        dataset=spec.dataset,
        metrics=result.metrics,
        artifacts=artifacts,
        caveats=caveats,
        passed=result.passed,
    )


def write_external_benchmark_report(
    runner: BenchmarkRunner,
    spec: BenchmarkSpec,
    path: str | Path,
    *,
    created_at: str | None = None,
    git_commit: str | None = None,
) -> EvalReport:
    """Run a benchmark runner and write a valid EvalReport JSON artifact."""

    report = run_external_benchmark(
        runner,
        spec,
        created_at=created_at,
        git_commit=git_commit,
    )
    write_report_json(report, path)
    return report


def _mode_tuple(values: object) -> tuple[BenchmarkMode, ...]:
    if isinstance(values, str | bytes | bytearray) or not isinstance(
        values,
        Sequence,
    ):
        raise EvaluationError("benchmark modes must be a sequence")
    if not values:
        raise EvaluationError("benchmark modes must include at least one mode")
    modes: list[BenchmarkMode] = []
    unsupported: list[str] = []
    seen: set[BenchmarkMode] = set()
    duplicates: list[str] = []
    for value in values:
        if value in BENCHMARK_MODES:
            if value in seen:
                duplicates.append(value)
            else:
                seen.add(value)
                modes.append(value)
        else:
            unsupported.append(str(value))
    if unsupported:
        raise EvaluationError(
            "unsupported benchmark modes: " + ", ".join(sorted(unsupported))
        )
    if duplicates:
        raise EvaluationError(
            "duplicate benchmark modes: " + ", ".join(sorted(set(duplicates)))
        )
    return tuple(modes)


def _freeze_metrics(metrics: Mapping[str, EvalMetric]) -> Mapping[str, EvalMetric]:
    if not isinstance(metrics, Mapping):
        raise EvaluationError("metrics must be a mapping")
    frozen: dict[str, EvalMetric] = {}
    for key, value in metrics.items():
        _require_non_empty_str(key, "metric name")
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise EvaluationError(f"metric {key} must be int, float, or string")
        if isinstance(value, float) and (
            value == float("inf") or value == float("-inf") or value != value
        ):
            raise EvaluationError(f"metric {key} must be finite")
        if isinstance(value, str) and not value:
            raise EvaluationError(f"metric {key} must not be empty")
        frozen[key] = value
    return MappingProxyType(frozen)


def _freeze_string_mapping(
    values: Mapping[str, str],
    field_name: str,
) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise EvaluationError(f"{field_name} must be a mapping")
    frozen: dict[str, str] = {}
    for key, value in values.items():
        frozen[_require_non_empty_str(key, f"{field_name} key")] = (
            _require_non_empty_str(value, f"{field_name}.{key}")
        )
    return MappingProxyType(frozen)


def _string_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes | bytearray) or not isinstance(
        values,
        Sequence,
    ):
        raise EvaluationError(f"{field_name} must be a sequence")
    result = tuple(
        _require_non_empty_str(value, f"{field_name} item") for value in values
    )
    if field_name == "command" and not result:
        raise EvaluationError("command must not be empty")
    return result


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{field_name} must be a non-empty string")
    return value


def _benchmark_platform_summary(hardware: Mapping[str, str]) -> dict[str, str]:
    summary = {
        "cpu_count": str(os.cpu_count() or "unknown"),
        "gpu": "not-recorded",
        "machine": platform_module.machine() or "unknown",
        "memory": "unknown",
        "processor": platform_module.processor() or "unknown",
        "python": platform_module.python_version(),
        "system": platform_module.system() or "unknown",
    }
    summary.update(dict(hardware))
    return summary


def _detect_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


__all__ = [
    "BENCHMARK_MODES",
    "BenchmarkMode",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSpec",
    "DryRunBenchmarkRunner",
    "load_benchmark_dataset_ref",
    "parse_benchmark_modes",
    "run_external_benchmark",
    "write_external_benchmark_report",
]
