"""Command entry point for cross-source transfer evaluation reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from mneme.eval._cross_source import run_cross_source_transfer_evaluation
from mneme.eval._entrypoints import write_report_for_entrypoint


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Write the deterministic cross-source transfer report."""

    parser = argparse.ArgumentParser(prog="mneme eval cross-source")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args(argv)
    command = (
        "mneme",
        "eval",
        "cross-source",
        "--out",
        str(args.out),
        "--seed",
        str(args.seed),
    )
    report = run_cross_source_transfer_evaluation(seed=args.seed, command=command)
    return write_report_for_entrypoint(
        report,
        args.out,
        report_name="cross-source transfer",
        stdout=stdout,
    )


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
