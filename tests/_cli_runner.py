"""In-process CLI runner for command contract tests."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO

from mneme.cli.__main__ import main as cli_main
from mneme.core import CliExitCode


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def run_cli(*args: object) -> CliResult:
    """Run the Mneme CLI without spawning a Python subprocess."""

    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            returncode = cli_main([str(arg) for arg in args], stdout=stdout)
        except SystemExit as exc:
            returncode = _system_exit_code(exc)
    return CliResult(
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _system_exit_code(exc: SystemExit) -> int:
    if isinstance(exc.code, int):
        return exc.code
    if exc.code is None:
        return int(CliExitCode.SUCCESS)
    return int(CliExitCode.INTERNAL)
