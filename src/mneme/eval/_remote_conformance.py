"""Fixture-scale local-vs-remote store conformance reports."""

from __future__ import annotations

import asyncio
import json
import platform as platform_module
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np

from mneme._version import __version__
from mneme.core import (
    Cid,
    EncoderFingerprint,
    EvaluationError,
    MemoryItem,
    Metric,
    MnemeError,
    QuerySpec,
    Transition,
    content_id,
)
from mneme.eval._reports import DatasetRef, EvalReport
from mneme.remote import (
    HttpJsonRequester,
    HttpJsonResponse,
    MemoryStoreASGIApp,
    RemoteHttpClient,
    RemoteHttpConfig,
)
from mneme.store import LocalStore, init_store

_REMOTE_CONFORMANCE_CAVEAT = (
    "Fixture-scale remote conformance uses an in-process ASGI requester. It "
    "does not certify network deployment, authentication operations, load, or "
    "confidentiality."
)
_TRANSPORT = "http-json-asgi"


def run_remote_conformance_evaluation(
    *,
    seed: int = 0,
    command: Sequence[str] = ("mneme", "eval", "remote-conformance"),
    created_at: str | None = None,
    git_commit: str | None = None,
) -> EvalReport:
    """Compare local store semantics with the first remote HTTP adapter."""

    with tempfile.TemporaryDirectory(prefix="mneme-remote-conformance-") as tmp:
        root = Path(tmp)
        local = init_store(root / "local")
        remote_backend = init_store(root / "remote")
        client = RemoteHttpClient(
            RemoteHttpConfig("http://mneme-conformance"),
            requester=_asgi_requester(MemoryStoreASGIApp(remote_backend)),
        )
        items = _fixture_items()

        local_put_ids = tuple(local.put_batch(list(items)))
        remote_put_ids = tuple(client.put_batch(items))
        local_root = local.commit()
        remote_root = remote_backend.commit()
        remote_client_root = client.root()

        query = QuerySpec(
            vector=items[0].key,
            k=2,
            metric=Metric.L2,
            with_receipt=True,
            encoder_fp=items[0].encoder_fp,
        )
        local_retrieval = local.query(query)
        remote_retrieval = client.query(query)
        local_query_ids = _retrieval_ids(local_retrieval.items)
        remote_query_ids = _retrieval_ids(remote_retrieval.items)

        proof_ids = local_query_ids[:1]
        local_proofs = tuple(proof.to_json() for proof in local.prove(proof_ids))
        remote_proofs = tuple(proof.to_json() for proof in client.prove(proof_ids))

        local_stats = local.stats()
        remote_stats = client.stats()
        error_cases = _error_case_names(local, client, items[0])

    checks = {
        "put_ids_match": local_put_ids == remote_put_ids,
        "roots_match": local_root == remote_root == remote_client_root,
        "query_ids_match": local_query_ids == remote_query_ids,
        "query_distances_match": tuple(local_retrieval.distances)
        == tuple(remote_retrieval.distances),
        "query_receipt_returned": remote_retrieval.receipt is not None,
        "proofs_match": local_proofs == remote_proofs,
        "stats_visible_count_match": local_stats.visible_record_count
        == remote_stats.get("visible_record_count"),
        "stats_commitment_flag_match": local_stats.commitments_enabled
        == remote_stats.get("commitments_enabled"),
        "error_types_match": all(left == right for left, right in error_cases),
    }
    passed_count = sum(int(value) for value in checks.values())
    metrics: dict[str, int | str] = {name: int(value) for name, value in checks.items()}
    metrics.update(
        {
            "scenario_count": len(checks),
            "passed_scenario_count": passed_count,
            "item_count": len(items),
            "query_k": 2,
            "error_case_count": len(error_cases),
            "transport": _TRANSPORT,
        }
    )

    return EvalReport(
        report_id="mneme-remote-conformance-v1",
        command=tuple(command),
        package_version=__version__,
        git_commit=_detect_git_commit() if git_commit is None else git_commit,
        created_at=_utc_now() if created_at is None else created_at,
        platform=_platform_summary(),
        seed=seed,
        dataset=DatasetRef(
            dataset_id="local-vs-remote-store-conformance",
            kind="fixture",
            split="deterministic",
            version="v1",
            metadata={
                "fixture_scale": True,
                "synthetic": True,
                "transport": _TRANSPORT,
                "client": "RemoteHttpClient",
                "server": "MemoryStoreASGIApp",
                "item_count": len(items),
                "error_case_count": len(error_cases),
            },
        ),
        metrics=metrics,
        artifacts={
            "report_kind": "remote-conformance",
            "transport": _TRANSPORT,
            "client": "RemoteHttpClient",
            "server": "MemoryStoreASGIApp",
        },
        caveats=(_REMOTE_CONFORMANCE_CAVEAT,),
        passed=passed_count == len(checks),
    )


def _error_case_names(
    local: LocalStore,
    remote: RemoteHttpClient,
    item: MemoryItem,
) -> tuple[tuple[str, str], ...]:
    wrong_query = QuerySpec(
        vector=item.key,
        k=1,
        metric=Metric.L2,
        encoder_fp=_fingerprint("wrong-summary"),
    )
    with tempfile.TemporaryDirectory(prefix="mneme-remote-error-") as tmp:
        uncommitted = init_store(Path(tmp) / "store")
        uncommitted_remote = RemoteHttpClient(
            RemoteHttpConfig("http://mneme-conformance-error"),
            requester=_asgi_requester(MemoryStoreASGIApp(uncommitted)),
        )
        return (
            (
                _error_name(lambda: local.query(wrong_query)),
                _error_name(lambda: remote.query(wrong_query)),
            ),
            (
                _error_name(lambda: uncommitted.root()),
                _error_name(lambda: uncommitted_remote.root()),
            ),
        )


def _error_name(action: Callable[[], object]) -> str:
    try:
        action()
    except MnemeError as exc:
        return type(exc).__name__
    raise EvaluationError("conformance error case did not raise")


def _retrieval_ids(items: Sequence[MemoryItem]) -> tuple[Cid, ...]:
    return tuple(item.content_id or content_id(item) for item in items)


def _fixture_items() -> tuple[MemoryItem, ...]:
    fingerprint = _fingerprint()
    return tuple(
        _built_item(float(index), fingerprint=fingerprint, step=index)
        for index in range(1, 4)
    )


def _built_item(
    key_value: float,
    *,
    fingerprint: EncoderFingerprint,
    step: int,
) -> MemoryItem:
    z_src = np.array([key_value, 0.0], dtype=np.float32)
    z_next = np.array([key_value + 0.5, 0.25], dtype=np.float32)
    item = MemoryItem(
        content_id=None,
        key=np.array([key_value, 0.0], dtype=np.float32),
        value=Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_next,
            delta=z_next - z_src,
            t=step,
            episode_id=UUID(f"12345678-1234-5678-1234-{step:012d}"),
        ),
        meta={"fixture": "remote-conformance", "step": step},
        encoder_fp=fingerprint,
    )
    return replace(item, content_id=content_id(item))


def _fingerprint(summarizer_id: str = "meanpool-v1") -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id=summarizer_id,
        weights_digest=None,
        config_digest="blake3:config",
    )


def _asgi_requester(app: MemoryStoreASGIApp) -> HttpJsonRequester:
    def requester(
        method: str,
        path: str,
        payload: Mapping[str, object],
        config: RemoteHttpConfig,
    ) -> HttpJsonResponse:
        return asyncio.run(_call_asgi(app, method, path, payload, config))

    return requester


async def _call_asgi(
    app: MemoryStoreASGIApp,
    method: str,
    path: str,
    payload: Mapping[str, object],
    config: RemoteHttpConfig,
) -> HttpJsonResponse:
    body = json.dumps(payload).encode()
    sent = False
    messages: list[dict[str, object]] = []
    headers = [(b"content-type", b"application/json")]
    if config.bearer_token is not None:
        headers.append((b"authorization", f"Bearer {config.bearer_token}".encode()))

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await app(
        {"type": "http", "method": method, "path": path, "headers": headers},
        receive,
        send,
    )
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    bodies = [
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    ]
    response_body = b"".join(body for body in bodies if isinstance(body, bytes))
    status = start.get("status")
    if not isinstance(status, int):
        raise EvaluationError("ASGI response status must be an integer")
    return HttpJsonResponse(status, json.loads(response_body))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _platform_summary() -> dict[str, str]:
    return {
        "machine": platform_module.machine() or "unknown",
        "python": platform_module.python_version(),
        "system": platform_module.system() or "unknown",
    }


def _detect_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


__all__ = ["run_remote_conformance_evaluation"]
