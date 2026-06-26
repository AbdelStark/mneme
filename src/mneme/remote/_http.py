"""HTTP JSON remote store adapter over an ASGI-compatible service boundary."""

from __future__ import annotations

import hmac
import http.client
import importlib
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

from mneme.core import (
    Cid,
    FingerprintMismatchError,
    MemoryItem,
    MnemeError,
    OptionalDependencyError,
    QueryError,
    QuerySpec,
    ReceiptVerificationError,
    Retrieval,
    SchemaVersionError,
    StoreCorruptionError,
    StoreError,
    UnsupportedOperationError,
    ValidationError,
)
from mneme.core._json import dumps_strict_json, loads_strict_json
from mneme.receipts import InclusionProof
from mneme.remote._messages import (
    ErrorMessage,
    ProveRequest,
    ProveResponse,
    PutRequest,
    PutResponse,
    QueryRequest,
    QueryResponse,
    RootRequest,
    RootResponse,
    StatsRequest,
    StatsResponse,
)
from mneme.remote._validation import raise_for_remote_error, validate_query_response
from mneme.store import StoreStats

JsonObject: TypeAlias = Mapping[str, object]
AsgiScope: TypeAlias = Mapping[str, object]
AsgiMessage: TypeAlias = dict[str, object]
AsgiReceive: TypeAlias = Callable[[], Awaitable[AsgiMessage]]
AsgiSend: TypeAlias = Callable[[AsgiMessage], Awaitable[None]]
_BEARER_TOKEN_ERROR = (
    "remote HTTP bearer_token must be a non-empty ASCII token without whitespace"
)


@dataclass(frozen=True)
class RemoteHttpConfig:
    """Configuration for the first HTTP JSON remote store adapter."""

    base_url: str
    bearer_token: str | None = None
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url:
            raise ValidationError("remote HTTP base_url must be non-empty")
        _validate_http_base_url(self.base_url)
        if self.bearer_token is not None:
            _require_bearer_token(self.bearer_token)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int | float)
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0.0
        ):
            raise ValidationError("remote HTTP timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


@dataclass(frozen=True)
class HttpJsonResponse:
    """HTTP status and JSON object returned by an HTTP JSON requester."""

    status_code: int
    payload: JsonObject

    def __post_init__(self) -> None:
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or self.status_code < 100
            or self.status_code > 599
        ):
            raise ValidationError("remote HTTP status_code must be an HTTP status")
        if not isinstance(self.payload, Mapping):
            raise ValidationError("remote HTTP payload must be a JSON object")
        if not all(isinstance(key, str) for key in self.payload):
            raise ValidationError("remote HTTP payload keys must be strings")


class HttpJsonRequester(Protocol):
    """Synchronous JSON requester used by RemoteHttpClient."""

    def __call__(
        self,
        method: str,
        path: str,
        payload: JsonObject,
        config: RemoteHttpConfig,
    ) -> HttpJsonResponse: ...


class RemoteStoreBackend(Protocol):
    """Store methods exposed by the HTTP ASGI wrapper."""

    def put_batch(self, items: list[MemoryItem]) -> list[Cid]: ...

    def query(self, spec: QuerySpec) -> Retrieval: ...

    def prove(self, ids: Sequence[Cid]) -> list[InclusionProof]: ...

    def root(self) -> bytes: ...

    def stats(self) -> StoreStats: ...


class RemoteHttpClient:
    """Client wrapper that mirrors store operations over HTTP JSON."""

    def __init__(
        self,
        config: RemoteHttpConfig,
        *,
        requester: HttpJsonRequester | None = None,
    ) -> None:
        if not isinstance(config, RemoteHttpConfig):
            raise ValidationError("remote HTTP config must be a RemoteHttpConfig")
        if requester is not None and not callable(requester):
            raise ValidationError("remote HTTP requester must be callable")
        self.config = config
        self._requester = _stdlib_request_json if requester is None else requester

    def put(self, item: MemoryItem) -> Cid:
        """Store one memory item and return its content id."""

        return self.put_batch((item,))[0]

    def put_batch(self, items: Sequence[MemoryItem]) -> list[Cid]:
        """Store memory items and return content ids."""

        payload = self._post("/put", PutRequest(tuple(items)).to_json())
        return list(PutResponse.from_json(payload).ids)

    def query(self, spec: QuerySpec) -> Retrieval:
        """Query the remote store after fail-closed response validation."""

        request = QueryRequest(spec)
        payload = self._post("/query", request.to_json())
        return validate_query_response(payload, request)

    def prove(self, ids: Sequence[Cid]) -> list[InclusionProof]:
        """Return remote inclusion proofs for committed content ids."""

        payload = self._post("/prove", ProveRequest(tuple(ids)).to_json())
        return list(ProveResponse.from_json(payload).proofs)

    def root(self) -> bytes:
        """Return the remote store's committed root."""

        payload = self._post("/root", RootRequest().to_json())
        return RootResponse.from_json(payload).root

    def stats(self) -> Mapping[str, Any]:
        """Return JSON-safe remote store statistics."""

        payload = self._post("/stats", StatsRequest().to_json())
        return StatsResponse.from_json(payload).stats

    def _post(self, path: str, payload: JsonObject) -> JsonObject:
        response = self._requester("POST", path, payload, self.config)
        if not isinstance(response, HttpJsonResponse):
            raise ValidationError("remote HTTP requester must return HttpJsonResponse")
        if response.status_code >= 400:
            raise_for_remote_error(response.payload)
        if response.status_code < 200 or response.status_code >= 300:
            raise StoreError(f"remote HTTP returned status {response.status_code}")
        return response.payload


class MemoryStoreASGIApp:
    """ASGI app that exposes a local MemoryStore-compatible object over JSON."""

    def __init__(
        self,
        store: RemoteStoreBackend,
        *,
        bearer_token: str | None = None,
    ) -> None:
        if bearer_token is not None:
            _require_bearer_token(bearer_token)
        self.store = store
        self.bearer_token = bearer_token

    async def __call__(
        self,
        scope: AsgiScope,
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        if scope.get("type") != "http":
            return
        if self.bearer_token is not None and not _authorized(scope, self.bearer_token):
            await _send_json(
                send,
                401,
                ErrorMessage("ValidationError", "missing or invalid bearer token"),
            )
            return
        method = _scope_text(scope, "method").upper()
        path = _scope_text(scope, "path")
        if method != "POST":
            await _send_json(
                send,
                405,
                ErrorMessage("UnsupportedOperationError", "only POST is supported"),
            )
            return
        try:
            request = _json_body(await _read_body(receive))
            response = self._handle(path, request)
        except Exception as exc:
            await _send_json(send, _status_for_error(exc), _error_message(exc))
            return
        await _send_json(send, 200, response)

    def _handle(self, path: str, payload: JsonObject) -> JsonObject:
        if path == "/put":
            put_request = PutRequest.from_json(payload)
            ids = self.store.put_batch(list(put_request.items))
            return PutResponse(tuple(ids)).to_json()
        if path == "/query":
            query_request = QueryRequest.from_json(payload)
            return QueryResponse(self.store.query(query_request.spec)).to_json()
        if path == "/prove":
            prove_request = ProveRequest.from_json(payload)
            return ProveResponse(tuple(self.store.prove(prove_request.ids))).to_json()
        if path == "/root":
            RootRequest.from_json(payload)
            return RootResponse(self.store.root()).to_json()
        if path == "/stats":
            StatsRequest.from_json(payload)
            return StatsResponse.from_store_stats(self.store.stats()).to_json()
        raise UnsupportedOperationError(f"unsupported remote HTTP path: {path}")


def serve_asgi_app(
    app: MemoryStoreASGIApp,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Serve a remote store ASGI app with uvicorn from the remote extra."""

    _validate_asgi_bind(host, port)
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "Remote HTTP serving requires the 'remote' extra",
            extra="remote",
            package="uvicorn",
        ) from exc
    runner = cast(Any, uvicorn).run
    if not callable(runner):
        raise OptionalDependencyError(
            "Remote HTTP serving requires uvicorn.run",
            extra="remote",
            package="uvicorn",
        )
    runner(app, host=host, port=port)


def _validate_asgi_bind(host: object, port: object) -> None:
    if not isinstance(host, str) or not host.strip():
        raise ValidationError("remote HTTP serve host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValidationError("remote HTTP serve port must be an integer 0..65535")


def _require_bearer_token(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(_BEARER_TOKEN_ERROR)
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValidationError(_BEARER_TOKEN_ERROR) from exc
    if any(byte <= 0x20 or byte == 0x7F for byte in encoded):
        raise ValidationError(_BEARER_TOKEN_ERROR)
    return value


def _stdlib_request_json(
    method: str,
    path: str,
    payload: JsonObject,
    config: RemoteHttpConfig,
) -> HttpJsonResponse:
    body = _json_bytes(payload)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if config.bearer_token is not None:
        headers["Authorization"] = f"Bearer {config.bearer_token}"
    request = urllib.request.Request(
        _url(config.base_url, path),
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=config.timeout_seconds
        ) as response:
            return HttpJsonResponse(
                status_code=response.status,
                payload=_json_body(response.read()),
            )
    except urllib.error.HTTPError as exc:
        return HttpJsonResponse(status_code=exc.code, payload=_json_body(exc.read()))
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise StoreError(
            f"remote HTTP request failed: {_request_failure_reason(exc)}"
        ) from exc


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _request_failure_reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        return str(reason) or type(exc).__name__
    return str(exc) or type(exc).__name__


def _validate_http_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise ValidationError(
            "remote HTTP base_url must be an http(s) URL without credentials, "
            "query, or fragment"
        )


async def _read_body(receive: AsgiReceive) -> bytes:
    chunks: list[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        body = message.get("body", b"")
        if not isinstance(body, bytes):
            raise ValidationError("ASGI request body must be bytes")
        chunks.append(body)
        more_body = bool(message.get("more_body", False))
    return b"".join(chunks)


async def _send_json(
    send: AsgiSend,
    status_code: int,
    payload: ErrorMessage | JsonObject,
) -> None:
    json_payload = payload.to_json() if isinstance(payload, ErrorMessage) else payload
    body = _json_bytes(json_payload)
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _json_body(body: bytes) -> JsonObject:
    try:
        value = loads_strict_json(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValidationError("remote HTTP body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError("remote HTTP body must be a JSON object")
    return cast(JsonObject, value)


def _json_bytes(payload: JsonObject) -> bytes:
    try:
        return dumps_strict_json(payload).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "remote HTTP payload must be JSON-serializable with finite numbers"
        ) from exc


def _authorized(scope: AsgiScope, token: str) -> bool:
    expected = f"Bearer {token}".encode()
    seen = False
    matched = False
    for name, value in _scope_headers(scope):
        if name.lower() == b"authorization":
            if seen:
                return False
            seen = True
            matched = hmac.compare_digest(value, expected)
    return seen and matched


def _scope_headers(scope: AsgiScope) -> tuple[tuple[bytes, bytes], ...]:
    value = scope.get("headers", ())
    if not isinstance(value, tuple | list):
        return ()
    headers: list[tuple[bytes, bytes]] = []
    for item in value:
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], bytes)
            and isinstance(item[1], bytes)
        ):
            headers.append((item[0], item[1]))
    return tuple(headers)


def _scope_text(scope: AsgiScope, key: str) -> str:
    value = scope.get(key, "")
    return value if isinstance(value, str) else ""


def _error_message(exc: BaseException) -> ErrorMessage:
    error_type = type(exc).__name__ if isinstance(exc, MnemeError) else "StoreError"
    message = str(exc) or error_type
    extra = exc.extra if isinstance(exc, OptionalDependencyError) else None
    package = exc.package if isinstance(exc, OptionalDependencyError) else None
    return ErrorMessage(
        error_type,
        message,
        retryable=_retryable(exc),
        extra=extra,
        package=package,
    )


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, StoreError) and not isinstance(exc, StoreCorruptionError)


def _status_for_error(exc: BaseException) -> int:
    if isinstance(exc, FingerprintMismatchError | ReceiptVerificationError):
        return 409
    if isinstance(exc, QueryError | SchemaVersionError | ValidationError):
        return 400
    if isinstance(exc, UnsupportedOperationError):
        return 501
    if isinstance(exc, OptionalDependencyError):
        return 503
    return 500


__all__ = [
    "HttpJsonRequester",
    "HttpJsonResponse",
    "MemoryStoreASGIApp",
    "RemoteStoreBackend",
    "RemoteHttpClient",
    "RemoteHttpConfig",
    "serve_asgi_app",
]
