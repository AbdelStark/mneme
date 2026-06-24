"""Evaluation report public models."""

from mneme.eval._fixtures import run_fixture_evaluation
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
    "DATASET_REF_SCHEMA",
    "EVAL_REPORT_SCHEMA",
    "DatasetKind",
    "DatasetRef",
    "EvalMetric",
    "EvalReport",
    "validate_report_json",
    "run_fixture_evaluation",
    "write_report_json",
]
