"""Command-line entry point for release artifact validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from mneme.core import CliExitCode
from mneme.release import validate_release_artifacts


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Validate built release artifacts and fixture evidence."""

    parser = argparse.ArgumentParser(prog="python -m mneme.release.validate_artifacts")
    parser.add_argument("--dist", default="dist", help="directory containing artifacts")
    parser.add_argument(
        "--fixture-report",
        required=True,
        help="fixture evaluation report JSON generated for this release",
    )
    parser.add_argument("--out", help="optional path for the validation report JSON")
    args = parser.parse_args(argv)
    report = validate_release_artifacts(
        args.dist,
        fixture_report=args.fixture_report,
    )
    output = json.dumps(report.to_json(), sort_keys=True, indent=2) + "\n"
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
    target_stream = sys.stdout if stdout is None else stdout
    print(output, end="", file=target_stream)
    if report.ok:
        return int(CliExitCode.SUCCESS)
    return int(CliExitCode.DATA_VALIDATION)


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
