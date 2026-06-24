"""Command entry point for fixture-scale remote conformance reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mneme.core import EvaluationError
from mneme.eval._remote_conformance import run_remote_conformance_evaluation
from mneme.eval._reports import write_report_json


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mneme.eval.remote_conformance")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
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
    try:
        write_report_json(report, args.out)
    except OSError as exc:
        raise EvaluationError(
            f"failed to write remote conformance report: {args.out}"
        ) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
