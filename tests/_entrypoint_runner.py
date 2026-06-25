"""In-process runner for module entrypoint tests."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Protocol

_SUCCESS_EXIT_CODE = 0
_INTERNAL_EXIT_CODE = 5


class Entrypoint(Protocol):
    def __call__(
        self,
        argv: list[str],
        *,
        stdout: StringIO | None = None,
    ) -> int: ...


@dataclass(frozen=True)
class EntrypointResult:
    returncode: int
    stdout: str
    stderr: str


def run_entrypoint(main: Entrypoint, *args: object) -> EntrypointResult:
    """Run a stream-injectable module entrypoint in-process."""

    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            returncode = main([str(arg) for arg in args], stdout=stdout)
        except SystemExit as exc:
            returncode = _system_exit_code(exc)
    return EntrypointResult(
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def run_plain_entrypoint(
    main: Callable[[list[str]], int],
    *args: object,
) -> EntrypointResult:
    """Run an entrypoint that does not expose stream injection."""

    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            returncode = main([str(arg) for arg in args])
        except SystemExit as exc:
            returncode = _system_exit_code(exc)
    return EntrypointResult(
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def _system_exit_code(exc: SystemExit) -> int:
    if isinstance(exc.code, int):
        return exc.code
    if exc.code is None:
        return _SUCCESS_EXIT_CODE
    return _INTERNAL_EXIT_CODE
