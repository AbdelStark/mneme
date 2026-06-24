"""Command entry point for local profile evaluation reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from mneme.core import CliExitCode, Metric
from mneme.eval._profile import run_profile_evaluation
from mneme.eval._reports import write_report_json
from mneme.store import open_store


def main(argv: Sequence[str] | None = None) -> int:
    """Write a local recall, latency, and footprint profile report."""

    parser = argparse.ArgumentParser(prog="mneme eval profile")
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
    parser.add_argument(
        "--approx-backend",
        default="faiss_hnsw",
        help="approximate backend to compare, or 'none'",
    )
    parser.add_argument("--seed", default=0, type=int)
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
    write_report_json(report, args.out)
    print(args.out)
    return int(CliExitCode.SUCCESS)


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
