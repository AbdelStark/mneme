"""Command entry point for deterministic fixture-scale evaluation reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from mneme.eval._entrypoints import run_eval_entrypoint, write_report_for_entrypoint
from mneme.eval._fixtures import run_fixture_evaluation


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Write the deterministic fixture evaluation report."""

    return run_eval_entrypoint(_run, argv, stdout=stdout)


def _run(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mneme eval fixtures")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args(argv)
    command = (
        "mneme",
        "eval",
        "fixtures",
        "--out",
        str(args.out),
        "--seed",
        str(args.seed),
    )
    report = run_fixture_evaluation(seed=args.seed, command=command)
    return write_report_for_entrypoint(
        report,
        args.out,
        report_name="fixture",
        stdout=stdout,
    )


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
