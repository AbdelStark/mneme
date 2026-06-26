"""Command entry point for receipt overhead evaluation reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import NoReturn, TextIO

from mneme.cli._arguments import add_eval_receipts_arguments
from mneme.core import Metric
from mneme.eval._entrypoints import (
    run_eval_entrypoint,
    write_report_for_entrypoint,
)
from mneme.eval._receipts import run_receipt_profile_evaluation
from mneme.store import open_store


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Write a local receipt overhead and proof-size report."""

    return run_eval_entrypoint(_run, argv, stdout=stdout)


def _run(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mneme eval receipts")
    add_eval_receipts_arguments(parser)
    args = parser.parse_args(argv)
    command = (
        "mneme",
        "eval",
        "receipts",
        "--store",
        str(args.store),
        "--out",
        str(args.out),
        "--k",
        str(args.k),
        "--metric",
        str(args.metric),
        "--queries",
        str(args.queries),
        "--warmup",
        str(args.warmup),
        "--measurements",
        str(args.measurements),
        "--seed",
        str(args.seed),
    )
    report = run_receipt_profile_evaluation(
        open_store(args.store),
        k=args.k,
        metric=Metric(args.metric),
        query_count=args.queries,
        warmup_count=args.warmup,
        measurement_count=args.measurements,
        seed=args.seed,
        command=command,
    )
    return write_report_for_entrypoint(
        report,
        args.out,
        report_name="receipt profile",
        stdout=stdout,
    )


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
