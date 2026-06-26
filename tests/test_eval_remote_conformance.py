from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from _entrypoint_runner import run_entrypoint

from mneme._version import __version__
from mneme.core import EvaluationError
from mneme.eval import run_remote_conformance_evaluation, validate_report_json
from mneme.eval._remote_conformance import _call_asgi
from mneme.eval.remote_conformance import main as remote_conformance_main
from mneme.remote import RemoteHttpConfig


def test_remote_conformance_report_validates_and_records_transport() -> None:
    report = run_remote_conformance_evaluation(
        seed=7,
        command=("mneme", "eval", "remote-conformance", "--out", "report.json"),
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    decoded = validate_report_json(report.to_json())

    assert decoded == report
    assert decoded.package_version == __version__
    assert decoded.seed == 7
    assert decoded.passed is True
    assert decoded.artifacts["report_kind"] == "remote-conformance"
    assert decoded.artifacts["transport"] == "http-json-asgi"
    assert decoded.dataset.metadata["fixture_scale"] is True
    assert decoded.dataset.metadata["transport"] == "http-json-asgi"


def test_remote_conformance_report_compares_semantics_and_errors() -> None:
    report = run_remote_conformance_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )

    assert report.metrics["put_ids_match"] == 1
    assert report.metrics["roots_match"] == 1
    assert report.metrics["query_ids_match"] == 1
    assert report.metrics["query_distances_match"] == 1
    assert report.metrics["proofs_match"] == 1
    assert report.metrics["stats_visible_count_match"] == 1
    assert report.metrics["error_case_count"] == 2
    assert report.metrics["error_types_match"] == 1
    assert report.metrics["scenario_count"] == report.metrics["passed_scenario_count"]


def test_remote_conformance_report_keeps_fixture_claim_boundary() -> None:
    report = run_remote_conformance_evaluation(
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    text = " ".join(report.caveats).lower()

    assert report.dataset.kind == "fixture"
    assert "fixture-scale" in text
    assert "does not certify network deployment" in text
    assert "confidentiality" in text


def test_remote_conformance_eval_module_writes_valid_report_json(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reports" / "remote-conformance.json"

    completed = run_entrypoint(
        remote_conformance_main,
        "--out",
        output,
        "--seed",
        "11",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RuntimeWarning" not in completed.stderr
    assert completed.stdout == ""
    report = validate_report_json(json.loads(output.read_text(encoding="utf-8")))
    assert report.seed == 11
    assert report.command[:3] == ("mneme", "eval", "remote-conformance")
    assert report.artifacts["transport"] == "http-json-asgi"
    assert report.passed is True


def test_remote_conformance_asgi_call_wraps_malformed_json_body() -> None:
    class MalformedJsonApp:
        async def __call__(self, scope, receive, send) -> None:
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"{not-json"})

    with pytest.raises(EvaluationError, match="ASGI response body must be valid JSON"):
        asyncio.run(
            _call_asgi(
                MalformedJsonApp(),
                "POST",
                "/stats",
                {},
                RemoteHttpConfig("http://mneme-conformance"),
            )
        )


@pytest.mark.parametrize(
    ("status", "match"),
    (
        (True, "ASGI response status must be an integer"),
        (99, "ASGI response status must be an HTTP status"),
        (600, "ASGI response status must be an HTTP status"),
    ),
)
def test_remote_conformance_asgi_call_rejects_malformed_status(
    status: object,
    match: str,
) -> None:
    class MalformedStatusApp:
        async def __call__(self, scope, receive, send) -> None:
            await send({"type": "http.response.start", "status": status})
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"schema_version": "mneme.stats.response.v1"}',
                }
            )

    with pytest.raises(EvaluationError, match=match):
        asyncio.run(
            _call_asgi(
                MalformedStatusApp(),
                "POST",
                "/stats",
                {},
                RemoteHttpConfig("http://mneme-conformance"),
            )
        )


def test_remote_conformance_asgi_call_requires_response_start() -> None:
    class MissingStartApp:
        async def __call__(self, scope, receive, send) -> None:
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"schema_version": "mneme.stats.response.v1"}',
                }
            )

    with pytest.raises(EvaluationError, match="response start message is missing"):
        asyncio.run(
            _call_asgi(
                MissingStartApp(),
                "POST",
                "/stats",
                {},
                RemoteHttpConfig("http://mneme-conformance"),
            )
        )


def test_remote_conformance_asgi_call_rejects_non_bytes_body_chunks() -> None:
    class MalformedBodyChunkApp:
        async def __call__(self, scope, receive, send) -> None:
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": "not-bytes"})

    with pytest.raises(EvaluationError, match="body chunks must be bytes"):
        asyncio.run(
            _call_asgi(
                MalformedBodyChunkApp(),
                "POST",
                "/stats",
                {},
                RemoteHttpConfig("http://mneme-conformance"),
            )
        )
