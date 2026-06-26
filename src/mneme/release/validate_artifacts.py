"""Command-line entry point for release artifact validation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NoReturn, TextIO

from mneme.core import CliExitCode
from mneme.core._json import dumps_strict_json, write_strict_json_file
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
    output = dumps_strict_json(report.to_json(), sort_keys=True, indent=2) + "\n"
    if args.out is not None:
        try:
            write_strict_json_file(
                args.out,
                report.to_json(),
                sort_keys=True,
                indent=2,
            )
        except (OSError, TypeError, ValueError) as exc:
            print(
                f"failed to write release artifact report: {args.out}: {exc}",
                file=sys.stderr,
            )
            return int(CliExitCode.INTERNAL)
    target_stream = sys.stdout if stdout is None else stdout
    print(output, end="", file=target_stream)
    if report.ok:
        return int(CliExitCode.SUCCESS)
    return int(CliExitCode.DATA_VALIDATION)


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
