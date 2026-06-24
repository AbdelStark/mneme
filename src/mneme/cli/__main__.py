"""Mneme command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from mneme.core import (
    CliExitCode,
    EvaluationError,
    Metric,
    MnemeError,
    QueryError,
    QuerySpec,
    UnsupportedOperationError,
    cli_exit_code,
    content_id,
)
from mneme.eval import run_fixture_evaluation, run_profile_evaluation, write_report_json
from mneme.store import StoreStats, init_store, open_store, rebuild_index, verify_store

QUERY_RESULT_SCHEMA = "mneme.query_result.v1"
STORE_STATS_SCHEMA = "mneme.store_stats.v1"
CLI_ERROR_SCHEMA = "mneme.cli_error.v1"


@dataclass(frozen=True)
class JsonResult:
    """Small JSON result wrapper for CLI-only reports."""

    ok: bool
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, **self.payload}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Mneme command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return int(CliExitCode.USER_INPUT)
    try:
        report = args.handler(args)
    except MnemeError as exc:
        _print_json(_error_json(exc))
        return cli_exit_code(exc)
    except Exception as exc:
        _print_json(_error_json(exc))
        return int(CliExitCode.INTERNAL)
    payload = _to_json(report)
    _print_json(payload)
    return _success_exit_code(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mneme")
    subparsers = parser.add_subparsers(dest="command", required=True)

    store_parser = subparsers.add_parser("store", help="local store operations")
    store_subparsers = store_parser.add_subparsers(
        dest="store_command",
        required=True,
    )
    init_parser = store_subparsers.add_parser("init", help="initialize a local store")
    init_parser.add_argument("path", type=Path)
    init_parser.set_defaults(command="store init", handler=_handle_store_init)

    stats_parser = store_subparsers.add_parser("stats", help="show local store stats")
    stats_parser.add_argument("path", type=Path)
    stats_parser.add_argument("--json", action="store_true", help="emit JSON")
    stats_parser.set_defaults(command="store stats", handler=_handle_store_stats)

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

    query_parser = subparsers.add_parser("query", help="query a local store")
    query_parser.add_argument("path", type=Path)
    query_parser.add_argument("--vector", required=True, type=Path)
    query_parser.add_argument("--k", required=True, type=int)
    query_parser.add_argument(
        "--metric",
        default=Metric.COSINE.value,
        choices=[metric.value for metric in Metric],
    )
    query_parser.add_argument("--json", action="store_true", help="emit JSON")
    query_parser.set_defaults(command="query", handler=_handle_query)

    eval_parser = subparsers.add_parser("eval", help="evaluation commands")
    eval_subparsers = eval_parser.add_subparsers(
        dest="eval_command",
        required=True,
    )
    fixtures_parser = eval_subparsers.add_parser(
        "fixtures",
        help="write deterministic fixture evaluation report",
    )
    fixtures_parser.add_argument("--out", required=True, type=Path)
    fixtures_parser.add_argument("--seed", default=0, type=int)
    fixtures_parser.set_defaults(command="eval fixtures", handler=_handle_eval_fixtures)

    profile_parser = eval_subparsers.add_parser(
        "profile",
        help="write local recall, latency, and footprint report",
    )
    _add_eval_profile_args(profile_parser)
    profile_parser.set_defaults(command="eval profile", handler=_handle_eval_profile)
    recall_parser = eval_subparsers.add_parser(
        "recall",
        help="write local recall profile report",
    )
    _add_eval_profile_args(recall_parser)
    recall_parser.set_defaults(command="eval recall", handler=_handle_eval_profile)
    latency_parser = eval_subparsers.add_parser(
        "latency",
        help="write local latency profile report",
    )
    _add_eval_profile_args(latency_parser)
    latency_parser.set_defaults(command="eval latency", handler=_handle_eval_profile)

    receipts_parser = subparsers.add_parser("receipts", help="receipt commands")
    receipts_subparsers = receipts_parser.add_subparsers(
        dest="receipts_command",
        required=True,
    )
    receipt_verify_parser = receipts_subparsers.add_parser(
        "verify",
        help="verify a retrieval receipt",
    )
    receipt_verify_parser.add_argument("receipt_file", type=Path)
    receipt_verify_parser.add_argument("--root", required=True)
    receipt_verify_parser.set_defaults(
        command="receipts verify",
        handler=_handle_receipts_verify,
    )

    return parser


def _handle_store_init(args: argparse.Namespace) -> JsonResult:
    store = init_store(args.path)
    return JsonResult(ok=True, payload=_stats_json(store.stats()))


def _handle_store_stats(args: argparse.Namespace) -> JsonResult:
    return JsonResult(ok=True, payload=_stats_json(open_store(args.path).stats()))


def _handle_store_verify(args: argparse.Namespace) -> object:
    return verify_store(args.path)


def _handle_index_rebuild(args: argparse.Namespace) -> object:
    return rebuild_index(args.path)


def _handle_query(args: argparse.Namespace) -> JsonResult:
    vector = _load_vector(args.vector)
    try:
        metric = Metric(args.metric)
    except ValueError as exc:
        raise QueryError(f"unsupported metric: {args.metric}") from exc
    retrieval = open_store(args.path).query(
        QuerySpec(vector=vector, k=args.k, metric=metric)
    )
    return JsonResult(
        ok=True,
        payload={
            "schema_version": QUERY_RESULT_SCHEMA,
            "item_count": len(retrieval.items),
            "content_id_prefixes": [
                (item.content_id or content_id(item))[:6].hex()
                for item in retrieval.items
            ],
            "distances": [float(distance) for distance in retrieval.distances],
            "receipt": None,
        },
    )


def _handle_eval_fixtures(args: argparse.Namespace) -> object:
    command = (
        "mneme",
        "eval",
        "fixtures",
        "--out",
        str(args.out),
        "--seed",
        str(args.seed),
    )
    report = run_fixture_evaluation(seed=args.seed, command=command)
    try:
        write_report_json(report, args.out)
    except OSError as exc:
        raise EvaluationError(f"failed to write fixture report: {args.out}") from exc
    return report


def _handle_eval_profile(args: argparse.Namespace) -> object:
    approx_backend = None if args.approx_backend == "none" else args.approx_backend
    command = (
        "mneme",
        "eval",
        str(args.eval_command),
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
    try:
        write_report_json(report, args.out)
    except OSError as exc:
        raise EvaluationError(f"failed to write profile report: {args.out}") from exc
    return report


def _handle_receipts_verify(args: argparse.Namespace) -> object:
    _ = args
    raise UnsupportedOperationError("receipt verification is not implemented in v0.1")


def _load_vector(path: Path) -> np.ndarray:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise QueryError(f"vector file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise QueryError(f"vector file is not valid JSON: {path}") from exc
    if isinstance(data, dict):
        data = data.get("vector")
    try:
        return np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    except (TypeError, ValueError) as exc:
        raise QueryError("vector file must contain a numeric JSON array") from exc


def _add_eval_profile_args(parser: argparse.ArgumentParser) -> None:
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


def _stats_json(stats: StoreStats) -> dict[str, Any]:
    return {
        "schema_version": STORE_STATS_SCHEMA,
        "store_id": str(stats.store_id),
        "active_fingerprint_count": stats.active_fingerprint_count,
        "value_log_count": stats.value_log_count,
        "value_record_count": stats.value_record_count,
        "visible_record_count": stats.visible_record_count,
        "value_bytes": stats.value_bytes,
        "index_backend": stats.index_backend,
        "retention_policy": stats.retention_policy,
        "tombstone_count": stats.tombstone_count,
        "last_completed_transaction": stats.last_completed_transaction,
        "commitments_enabled": stats.commitments_enabled,
    }


def _to_json(report: object) -> dict[str, Any]:
    to_json = getattr(report, "to_json", None)
    if callable(to_json):
        data = to_json()
    else:
        data = report
    if not isinstance(data, dict):
        raise TypeError("command handler returned non-object JSON")
    return data


def _success_exit_code(payload: dict[str, Any]) -> int:
    ok = payload.get("ok", True)
    if ok is False:
        return int(CliExitCode.DATA_VALIDATION)
    return int(CliExitCode.SUCCESS)


def _error_json(error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": CLI_ERROR_SCHEMA,
        "ok": False,
        "errors": [str(error)],
        "error_type": type(error).__name__,
    }


def _print_json(data: object) -> None:
    print(json.dumps(data, sort_keys=True, indent=2))


def _exit() -> NoReturn:
    raise SystemExit(main())


if __name__ == "__main__":
    _exit()
