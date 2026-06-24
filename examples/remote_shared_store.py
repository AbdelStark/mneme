"""Minimal remote/shared-store example over the HTTP JSON ASGI boundary.

Run from a checkout with:

    python3 examples/remote_shared_store.py

The example uses an in-process ASGI requester so it can run without starting a
server. Serving with uvicorn requires installing the `remote` extra and adding
operator-managed transport and credential controls.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import numpy as np

from mneme.core import EncoderFingerprint, Metric, QuerySpec, Transition, build_item
from mneme.receipts import RetrievalReceipt, verify_retrieval_receipt
from mneme.remote import (
    HttpJsonResponse,
    MemoryStoreASGIApp,
    RemoteHttpClient,
    RemoteHttpConfig,
)
from mneme.store import init_store


def main() -> int:
    fingerprint = EncoderFingerprint(
        encoder_id="example.encoder",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:example-config",
    )
    item = build_item(
        Transition(
            z_src=np.array([1.0, 0.0], dtype=np.float32),
            action=np.array([0.0], dtype=np.float32),
            z_next=np.array([2.0, 0.0], dtype=np.float32),
            delta=np.array([1.0, 0.0], dtype=np.float32),
            t=1,
            episode_id=UUID("12345678-1234-5678-1234-567812345101"),
        ),
        key=np.array([1.0, 0.0], dtype=np.float32),
        encoder_fp=fingerprint,
        meta={"source": "remote-shared-store-example"},
    )
    spec = QuerySpec(
        vector=item.key,
        k=1,
        metric=Metric.L2,
        with_receipt=True,
        encoder_fp=fingerprint,
    )

    with TemporaryDirectory(prefix="mneme-remote-example-") as tmp:
        backend = init_store(Path(tmp) / "remote", active_fingerprints=[fingerprint])
        app = MemoryStoreASGIApp(backend, bearer_token="example-token")
        client = RemoteHttpClient(
            RemoteHttpConfig(
                "http://mneme-example",
                bearer_token="example-token",
            ),
            requester=_asgi_requester(app),
        )
        cid = client.put(item)
        root = backend.commit()
        retrieval = client.query(spec)

    receipt = retrieval.receipt
    receipt_verified = isinstance(
        receipt, RetrievalReceipt
    ) and verify_retrieval_receipt(
        receipt,
        retrieval.items,
        root=root,
        query=spec,
    )
    payload = {
        "ok": cid == item.content_id and receipt_verified,
        "example": "remote-shared-store",
        "transport": "http-json-asgi-inprocess",
        "retrieved": len(retrieval.items),
        "receipt_verified": bool(receipt_verified),
        "success_signal": "receipt_verified is true for returned item",
        "security_boundary": (
            "requires authenticated transport and deployment controls outside Mneme"
        ),
        "claim_boundary": "fixture-only; not private retrieval or benchmark evidence",
    }
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0 if payload["ok"] else 1


def _asgi_requester(
    app: MemoryStoreASGIApp,
) -> Callable[
    [str, str, Mapping[str, object], RemoteHttpConfig],
    HttpJsonResponse,
]:
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
        raise RuntimeError("ASGI response status must be an integer")
    return HttpJsonResponse(status, json.loads(response_body))


if __name__ == "__main__":
    raise SystemExit(main())
