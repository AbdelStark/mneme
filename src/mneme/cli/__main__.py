"""Mneme command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from mneme.core import CliExitCode, StoreCorruptionError, cli_exit_code
from mneme.store import rebuild_index, verify_store


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Mneme command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return int(CliExitCode.USER_INPUT)
    try:
        report = args.handler(args)
    except StoreCorruptionError as exc:
        _print_json({"ok": False, "errors": [str(exc)]})
        return cli_exit_code(exc)
    _print_json(report.to_json())
    return int(CliExitCode.SUCCESS) if report.ok else int(CliExitCode.DATA_VALIDATION)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mneme")
    subparsers = parser.add_subparsers(dest="command", required=True)

    store_parser = subparsers.add_parser("store", help="local store operations")
    store_subparsers = store_parser.add_subparsers(
        dest="store_command",
        required=True,
    )
    verify_parser = store_subparsers.add_parser("verify", help="verify a local store")
    verify_parser.add_argument("path", type=Path)
    verify_parser.set_defaults(command="store verify", handler=_handle_store_verify)

    index_parser = subparsers.add_parser("index", help="local index operations")
    index_subparsers = index_parser.add_subparsers(
        dest="index_command",
        required=True,
    )
    rebuild_parser = index_subparsers.add_parser(
        "rebuild",
        help="rebuild persisted index metadata from value logs",
    )
    rebuild_parser.add_argument("path", type=Path)
    rebuild_parser.set_defaults(command="index rebuild", handler=_handle_index_rebuild)

    return parser


def _handle_store_verify(args: argparse.Namespace) -> object:
    return verify_store(args.path)


def _handle_index_rebuild(args: argparse.Namespace) -> object:
    return rebuild_index(args.path)


def _print_json(data: object) -> None:
    print(json.dumps(data, sort_keys=True, indent=2))


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
