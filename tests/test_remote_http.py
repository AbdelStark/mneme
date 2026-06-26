from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
    StoreError,
    Transition,
    UnsupportedOperationError,
    ValidationError,
    content_id,
)
from mneme.remote import (
    QUERY_REQUEST_SCHEMA,
    HttpJsonResponse,
    MemoryStoreASGIApp,
    QueryResponse,
    RemoteArray,
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


def test_remote_http_config_normalizes_and_validates_inputs() -> None:
    config = RemoteHttpConfig(
        "http://testserver",
        bearer_token="secret",
        timeout_seconds=5,
    )

    assert config.timeout_seconds == 5.0

    invalid_cases = [
        ({"base_url": ""}, "base_url"),
        ({"base_url": object()}, "base_url"),
        ({"base_url": "testserver"}, "base_url"),
        ({"base_url": "ftp://testserver"}, "base_url"),
        ({"base_url": "file:///tmp/store"}, "base_url"),
        ({"base_url": "http://user:test@testserver"}, "base_url"),
        ({"base_url": "http://testserver?token=secret"}, "base_url"),
        ({"base_url": "http://testserver#fragment"}, "base_url"),
        ({"base_url": "http://testserver", "bearer_token": ""}, "bearer_token"),
        (
            {"base_url": "http://testserver", "bearer_token": object()},
            "bearer_token",
        ),
        ({"base_url": "http://testserver", "timeout_seconds": True}, "timeout"),
        ({"base_url": "http://testserver", "timeout_seconds": "fast"}, "timeout"),
        ({"base_url": "http://testserver", "timeout_seconds": 0}, "timeout"),
        (
            {"base_url": "http://testserver", "timeout_seconds": float("nan")},
            "timeout",
        ),
    ]

    for kwargs, match in invalid_cases:
        with pytest.raises(ValidationError, match=match):
            RemoteHttpConfig(**kwargs)


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


def test_remote_http_asgi_app_validates_bearer_token(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")

    with pytest.raises(ValidationError, match="bearer_token"):
        MemoryStoreASGIApp(store, bearer_token="")
    with pytest.raises(ValidationError, match="bearer_token"):
        MemoryStoreASGIApp(store, bearer_token=object())  # type: ignore[arg-type]


def test_http_json_response_validates_status_and_payload() -> None:
    invalid_cases = [
        (True, {}, "status_code"),
        ("200", {}, "status_code"),
        (99, {}, "status_code"),
        (600, {}, "status_code"),
        (200, [], "payload"),
        (200, {1: "ok"}, "payload keys"),
    ]

    for status_code, payload, match in invalid_cases:
        with pytest.raises(ValidationError, match=match):
            HttpJsonResponse(status_code, payload)  # type: ignore[arg-type]


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


def test_remote_http_rejects_nonstandard_json_constants(tmp_path: Path) -> None:
    app = MemoryStoreASGIApp(init_store(tmp_path / "store"))

    response = asyncio.run(
        _call_asgi_raw(
            app,
            "POST",
            "/stats",
            b'{"schema_version": NaN}',
            RemoteHttpConfig("http://testserver"),
        )
    )

    assert response.status_code == 400
    assert response.payload["schema_version"] == "mneme.error.v1"
    assert response.payload["error_type"] == "ValidationError"
    assert "valid JSON" in str(response.payload["message"])


def test_remote_http_maps_query_errors_to_bad_request(tmp_path: Path) -> None:
    app = MemoryStoreASGIApp(init_store(tmp_path / "store"))
    payload = {
        "schema_version": QUERY_REQUEST_SCHEMA,
        "query": {
            "schema_version": "mneme.query_spec.v1",
            "vector": RemoteArray.from_array(
                np.array([[1.0]], dtype=np.float32)
            ).to_json(),
            "k": 1,
            "metric": "cosine",
            "ef": None,
            "filters": None,
            "temporal_decay": None,
            "with_receipt": False,
            "encoder_fp": None,
        },
    }

    response = asyncio.run(
        _call_asgi_raw(
            app,
            "POST",
            "/query",
            json.dumps(payload, allow_nan=False).encode("utf-8"),
            RemoteHttpConfig("http://testserver"),
        )
    )

    assert response.status_code == 400
    assert response.payload["schema_version"] == "mneme.error.v1"
    assert response.payload["error_type"] == "QueryError"
    assert "one-dimensional" in str(response.payload["message"])


def test_stdlib_request_json_rejects_nonfinite_payload_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def urlopen(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("urlopen should not be called")

    monkeypatch.setattr(remote_http.urllib.request, "urlopen", urlopen)

    with pytest.raises(ValidationError, match="finite numbers"):
        remote_http._stdlib_request_json(
            "POST",
            "/stats",
            {"schema_version": float("nan")},
            RemoteHttpConfig("http://testserver"),
        )

    assert called is False


@pytest.mark.parametrize(
    ("failure", "match"),
    (
        (TimeoutError("timed out"), "timed out"),
        (remote_http.http.client.HTTPException("bad status line"), "bad status line"),
    ),
)
def test_stdlib_request_json_wraps_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    match: str,
) -> None:
    def urlopen(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(remote_http.urllib.request, "urlopen", urlopen)

    with pytest.raises(StoreError, match=f"remote HTTP request failed: {match}"):
        remote_http._stdlib_request_json(
            "POST",
            "/stats",
            {"schema_version": "mneme.stats.request.v1"},
            RemoteHttpConfig("http://testserver"),
        )


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


@pytest.mark.parametrize(
    ("host", "port", "match"),
    (
        ("", 8000, "host"),
        ("   ", 8000, "host"),
        (object(), 8000, "host"),
        ("127.0.0.1", True, "port"),
        ("127.0.0.1", -1, "port"),
        ("127.0.0.1", 65536, "port"),
        ("127.0.0.1", "8000", "port"),
    ),
)
def test_serve_asgi_app_validates_bind_before_optional_import(
    monkeypatch: pytest.MonkeyPatch,
    host: object,
    port: object,
    match: str,
) -> None:
    def import_module(name: str) -> object:
        raise AssertionError(f"unexpected optional import: {name}")

    monkeypatch.setattr(remote_http.importlib, "import_module", import_module)

    with pytest.raises(ValidationError, match=match):
        serve_asgi_app(
            MemoryStoreASGIApp(object()),  # type: ignore[arg-type]
            host=host,  # type: ignore[arg-type]
            port=port,  # type: ignore[arg-type]
        )


def test_serve_asgi_app_forwards_valid_bind_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, str, int]] = []

    def run(app: object, *, host: str, port: int) -> None:
        calls.append((app, host, port))

    def import_module(name: str) -> object:
        if name == "uvicorn":
            return SimpleNamespace(run=run)
        raise AssertionError(name)

    monkeypatch.setattr(remote_http.importlib, "import_module", import_module)
    app = MemoryStoreASGIApp(object())  # type: ignore[arg-type]

    serve_asgi_app(app, host="0.0.0.0", port=0)

    assert calls == [(app, "0.0.0.0", 0)]


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
    return await _call_asgi_raw(
        app,
        method,
        path,
        json.dumps(payload, allow_nan=False).encode("utf-8"),
        config,
    )


async def _call_asgi_raw(
    app: MemoryStoreASGIApp,
    method: str,
    path: str,
    body: bytes,
    config: RemoteHttpConfig,
) -> HttpJsonResponse:
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
