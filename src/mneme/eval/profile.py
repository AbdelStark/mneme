"""Command entry point for local profile evaluation reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import NoReturn, TextIO

from mneme.cli._arguments import add_eval_profile_arguments
from mneme.core import Metric
from mneme.eval._entrypoints import (
    run_eval_entrypoint,
    write_report_for_entrypoint,
)
from mneme.eval._profile import run_profile_evaluation
from mneme.store import open_store


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Write a local recall, latency, and footprint profile report."""

    return run_eval_entrypoint(_run, argv, stdout=stdout)


def _run(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mneme eval profile")
    add_eval_profile_arguments(parser)
    args = parser.parse_args(argv)
    approx_backend = None if args.approx_backend == "none" else args.approx_backend
    command = (
        "mneme",
        "eval",
        "profile",
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
        "--approx-backend",
        str(args.approx_backend),
        "--seed",
        str(args.seed),
    )
    report = run_profile_evaluation(
        open_store(args.store),
        k=args.k,
        metric=Metric(args.metric),
        query_count=args.queries,
        warmup_count=args.warmup,
        measurement_count=args.measurements,
        approximate_backend=approx_backend,
        seed=args.seed,
        command=command,
    )
    return write_report_for_entrypoint(
        report,
        args.out,
        report_name="profile",
        stdout=stdout,
    )


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
