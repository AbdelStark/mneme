"""Runtime helpers shared by Mneme CLI entrypoints and tests."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from mneme.core import CliExitCode

CLI_ERROR_SCHEMA = "mneme.cli_error.v1"


@dataclass(frozen=True)
class JsonResult:
    """Small JSON result wrapper for CLI-only reports."""

    ok: bool
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, **self.payload}


def report_to_json(report: object) -> dict[str, Any]:
    """Normalize a command handler report into a JSON object."""

    to_json = getattr(report, "to_json", None)
    if callable(to_json):
        data = to_json()
    else:
        data = report
    if not isinstance(data, dict):
        raise TypeError("command handler returned non-object JSON")
    return data


def success_exit_code(payload: dict[str, Any]) -> int:
    """Return the CLI success-path exit code for a JSON payload."""

    ok = payload.get("ok", True)
    if ok is False:
        return int(CliExitCode.DATA_VALIDATION)
    return int(CliExitCode.SUCCESS)


def error_json(error: BaseException) -> dict[str, Any]:
    """Return the stable CLI error envelope for an exception."""

    return {
        "schema_version": CLI_ERROR_SCHEMA,
        "ok": False,
        "errors": [str(error)],
        "error_type": type(error).__name__,
    }


def print_json(data: object, stream: TextIO | None = None) -> None:
    """Print deterministic, pretty JSON to the provided stream."""

    target = sys.stdout if stream is None else stream
    print(json.dumps(data, sort_keys=True, indent=2), file=target)


__all__ = [
    "CLI_ERROR_SCHEMA",
    "JsonResult",
    "error_json",
    "print_json",
    "report_to_json",
    "success_exit_code",
]
