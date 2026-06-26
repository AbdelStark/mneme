"""Shared helpers for evaluation module entrypoints."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TextIO

from mneme.core import CliExitCode, EvaluationError, MnemeError, cli_exit_code
from mneme.eval._reports import EvalReport, write_report_json


class EvalEntrypoint(Protocol):
    """Stream-injectable eval module entrypoint implementation."""

    def __call__(
        self,
        argv: Sequence[str] | None,
        *,
        stdout: TextIO | None = None,
    ) -> int: ...


def run_eval_entrypoint(
    handler: EvalEntrypoint,
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
) -> int:
    """Run an eval module entrypoint with stable CLI-style error handling."""

    try:
        return handler(argv, stdout=stdout)
    except MnemeError as exc:
        _print_entrypoint_error(exc)
        return cli_exit_code(exc)
    except Exception as exc:
        _print_entrypoint_error(exc)
        return int(CliExitCode.INTERNAL)


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
    except (EvaluationError, OSError) as exc:
        raise EvaluationError(
            f"failed to write {report_name} report: {out}: {exc}"
        ) from exc
    if echo_path:
        target = sys.stdout if stdout is None else stdout
        print(out, file=target)
    return int(CliExitCode.SUCCESS)


def _print_entrypoint_error(error: BaseException) -> None:
    print(f"{type(error).__name__}: {error}", file=sys.stderr)


__all__ = ["run_eval_entrypoint", "write_report_for_entrypoint"]
