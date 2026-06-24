from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import mneme.remote._http as remote_http
from mneme.core import (
    EncoderFingerprint,
    FingerprintMismatchError,
    MemoryItem,
    Metric,
    OptionalDependencyError,
    QuerySpec,
    Retrieval,
    Transition,
    UnsupportedOperationError,
    ValidationError,
    content_id,
)
from mneme.remote import (
    HttpJsonResponse,
    MemoryStoreASGIApp,
    QueryResponse,
    RemoteHttpClient,
    RemoteHttpConfig,
    serve_asgi_app,
)
from mneme.store import init_store


def test_remote_http_client_asgi_smoke_put_query_root_stats(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    app = MemoryStoreASGIApp(store)
    client = RemoteHttpClient(
        RemoteHttpConfig("http://testserver"),
        requester=_asgi_requester(app),
    )
    item = _built_item(1.0)
    spec = QuerySpec(
        vector=item.key,
        k=1,
        metric=Metric.L2,
        with_receipt=True,
        encoder_fp=item.encoder_fp,
    )

    cid = client.put(item)
    root = store.commit()
    retrieval = client.query(spec)
    stats = client.stats()

    assert cid == item.content_id
    assert client.root() == root
    assert stats["visible_record_count"] == 1
    assert retrieval.items[0].content_id == cid
    assert retrieval.receipt is not None


def test_remote_http_client_validation_is_always_applied() -> None:
    item = _built_item(1.0, fingerprint=_fingerprint("right"))
    response = QueryResponse(Retrieval(items=(item,), distances=(0.0,))).to_json()

    def requester(
        method: str,
        path: str,
        payload: Mapping[str, object],
        config: RemoteHttpConfig,
    ) -> HttpJsonResponse:
        assert (method, path, config.base_url) == (
            "POST",
            "/query",
            "http://testserver",
        )
        assert payload["schema_version"] == "mneme.query.request.v1"
        return HttpJsonResponse(200, response)

    client = RemoteHttpClient(
        RemoteHttpConfig("http://testserver"),
        requester=requester,
    )
    spec = QuerySpec(
        vector=item.key,
        k=1,
        metric=Metric.L2,
        encoder_fp=_fingerprint("left"),
    )

    with pytest.raises(FingerprintMismatchError, match="fingerprint"):
        client.query(spec)


def test_remote_http_server_errors_map_to_local_typed_errors(tmp_path: Path) -> None:
    app = MemoryStoreASGIApp(init_store(tmp_path / "store"))
    client = RemoteHttpClient(
        RemoteHttpConfig("http://testserver"),
        requester=_asgi_requester(app),
    )

    with pytest.raises(UnsupportedOperationError, match="commitments"):
        client.root()


def test_remote_http_bearer_token_required(tmp_path: Path) -> None:
    app = MemoryStoreASGIApp(init_store(tmp_path / "store"), bearer_token="secret")
    missing = RemoteHttpClient(
        RemoteHttpConfig("http://testserver"),
        requester=_asgi_requester(app),
    )
    authorized = RemoteHttpClient(
        RemoteHttpConfig("http://testserver", bearer_token="secret"),
        requester=_asgi_requester(app),
    )

    with pytest.raises(ValidationError, match="bearer token"):
        missing.stats()
    assert authorized.stats()["visible_record_count"] == 0


def test_remote_http_malformed_server_response_fails_closed() -> None:
    def requester(
        method: str,
        path: str,
        payload: Mapping[str, object],
        config: RemoteHttpConfig,
    ) -> HttpJsonResponse:
        return HttpJsonResponse(
            500,
            {"schema_version": "mneme.error.v2", "error_type": "StoreError"},
        )

    client = RemoteHttpClient(
        RemoteHttpConfig("http://testserver"),
        requester=requester,
    )

    with pytest.raises(ValidationError):
        client.stats()


def test_serve_asgi_app_missing_uvicorn_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def import_module(name: str) -> object:
        if name == "uvicorn":
            raise ModuleNotFoundError(name)
        raise AssertionError(name)

    monkeypatch.setattr(remote_http.importlib, "import_module", import_module)

    with pytest.raises(OptionalDependencyError) as raised:
        serve_asgi_app(MemoryStoreASGIApp(object()))  # type: ignore[arg-type]

    assert raised.value.extra == "remote"
    assert raised.value.package == "uvicorn"


def test_remote_extra_declares_uvicorn_and_import_stays_lazy() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "uvicorn>=0.30" in pyproject["project"]["optional-dependencies"]["remote"]

    script = (
        "import sys; "
        "import mneme.remote; "
        "loaded = 'uvicorn' in sys.modules; "
        "print(loaded); "
        "raise SystemExit(1 if loaded else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
    body = json.dumps(payload).encode("utf-8")
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
    return HttpJsonResponse(int(start["status"]), json.loads(response_body))


def _fingerprint(summarizer_id: str = "meanpool-v1") -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id=summarizer_id,
        weights_digest=None,
        config_digest="blake3:config",
    )


def _built_item(
    key_value: float,
    *,
    fingerprint: EncoderFingerprint | None = None,
) -> MemoryItem:
    z_src = np.array([key_value, 0.0], dtype=np.float32)
    z_next = np.array([key_value + 1.0, 0.0], dtype=np.float32)
    item = MemoryItem(
        content_id=None,
        key=np.array([key_value, 0.0], dtype=np.float32),
        value=Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_next,
            delta=z_next - z_src,
            t=0,
            episode_id=uuid4(),
        ),
        meta={"source": "remote-http-fixture"},
        encoder_fp=_fingerprint() if fingerprint is None else fingerprint,
    )
    return replace(item, content_id=content_id(item))
