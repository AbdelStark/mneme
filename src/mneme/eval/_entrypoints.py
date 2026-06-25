"""Shared helpers for evaluation module entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from mneme.core import CliExitCode, EvaluationError
from mneme.eval._reports import EvalReport, write_report_json


def write_report_for_entrypoint(
    report: EvalReport,
    out: Path,
    *,
    report_name: str,
    stdout: TextIO | None = None,
    echo_path: bool = True,
) -> int:
    """Write an eval report and return the documented success exit code."""

    try:
        write_report_json(report, out)
    except OSError as exc:
        raise EvaluationError(f"failed to write {report_name} report: {out}") from exc
    if echo_path:
        target = sys.stdout if stdout is None else stdout
        print(out, file=target)
    return int(CliExitCode.SUCCESS)


__all__ = ["write_report_for_entrypoint"]
