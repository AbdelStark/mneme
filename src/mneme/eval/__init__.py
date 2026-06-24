"""Evaluation report public models."""

from mneme.eval._benchmark import (
    BENCHMARK_MODES,
    BenchmarkMode,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSpec,
    DryRunBenchmarkRunner,
    load_benchmark_dataset_ref,
    parse_benchmark_modes,
    run_external_benchmark,
    write_external_benchmark_report,
)
from mneme.eval._fixtures import run_fixture_evaluation
from mneme.eval._profile import run_profile_evaluation
from mneme.eval._receipts import run_receipt_profile_evaluation
from mneme.eval._remote_conformance import run_remote_conformance_evaluation
from mneme.eval._replay import (
    RECEIPT_REPLAY_REPORT_SCHEMA,
    RECEIPT_REPLAY_TRACE_SCHEMA,
    KnnReplayConfig,
    ReceiptReplayReport,
    ReceiptReplayTrace,
    build_receipt_replay_trace,
    load_replay_trace_json,
    replay_receipt_trace,
    write_replay_report_json,
    write_replay_trace_json,
)
from mneme.eval._reports import (
    DATASET_REF_SCHEMA,
    EVAL_REPORT_SCHEMA,
    DatasetKind,
    DatasetRef,
    EvalMetric,
    EvalReport,
    validate_report_json,
    write_report_json,
)

__all__ = [
    "BENCHMARK_MODES",
    "DATASET_REF_SCHEMA",
    "EVAL_REPORT_SCHEMA",
    "RECEIPT_REPLAY_REPORT_SCHEMA",
    "RECEIPT_REPLAY_TRACE_SCHEMA",
    "BenchmarkMode",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSpec",
    "DatasetKind",
    "DatasetRef",
    "DryRunBenchmarkRunner",
    "EvalMetric",
    "EvalReport",
    "KnnReplayConfig",
    "ReceiptReplayReport",
    "ReceiptReplayTrace",
    "build_receipt_replay_trace",
    "load_benchmark_dataset_ref",
    "load_replay_trace_json",
    "parse_benchmark_modes",
    "replay_receipt_trace",
    "run_external_benchmark",
    "validate_report_json",
    "run_fixture_evaluation",
    "run_profile_evaluation",
    "run_receipt_profile_evaluation",
    "run_remote_conformance_evaluation",
    "write_external_benchmark_report",
    "write_replay_report_json",
    "write_replay_trace_json",
    "write_report_json",
]
