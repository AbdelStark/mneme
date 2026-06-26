"""Command entry point for fixture-scale remote conformance reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TextIO

from mneme.cli._arguments import add_eval_output_seed_arguments
from mneme.eval._entrypoints import run_eval_entrypoint, write_report_for_entrypoint
from mneme.eval._remote_conformance import run_remote_conformance_evaluation


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    return run_eval_entrypoint(_run, argv, stdout=stdout)


def _run(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mneme.eval.remote_conformance")
    add_eval_output_seed_arguments(parser)
    args = parser.parse_args(argv)
    command = (
        "mneme",
        "eval",
        "remote-conformance",
        "--out",
        str(args.out),
        "--seed",
        str(args.seed),
    )
    report = run_remote_conformance_evaluation(seed=args.seed, command=command)
    return write_report_for_entrypoint(
        report,
        args.out,
        report_name="remote conformance",
        stdout=stdout,
        echo_path=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
