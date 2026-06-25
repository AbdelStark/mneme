"""Command entry point for receipt overhead evaluation reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from mneme.core import Metric
from mneme.eval._entrypoints import write_report_for_entrypoint
from mneme.eval._receipts import run_receipt_profile_evaluation
from mneme.store import open_store


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Write a local receipt overhead and proof-size report."""

    parser = argparse.ArgumentParser(prog="mneme eval receipts")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", default=4, type=int)
    parser.add_argument(
        "--metric",
        default=Metric.L2.value,
        choices=[metric.value for metric in Metric],
    )
    parser.add_argument("--queries", default=8, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--measurements", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
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
