"""Encoder fingerprint construction and mismatch checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from os import PathLike
from typing import Any

from blake3 import blake3

from mneme.core import (
    EncoderFingerprint,
    FingerprintMismatchError,
    ValidationError,
)
from mneme.core._json import dumps_strict_json
from mneme.core._paths import coerce_text_path
from mneme.encode._protocols import Summarizer

_DIGEST_PREFIX = "blake3:"
_CHUNK_SIZE = 1024 * 1024


def digest_config(config: Mapping[str, Any]) -> str:
    """Digest JSON-compatible configuration with stable key ordering."""

    canonical = dumps_strict_json(
        _json_ready(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _digest_bytes(canonical)


def digest_weights(weights: bytes | str | PathLike[str]) -> str:
    """Digest raw weight bytes or a weight file."""

    if isinstance(weights, bytes):
        return _digest_bytes(weights)

    path = coerce_text_path(
        weights,
        "weights file",
        type_error=ValidationError,
        value_error=ValidationError,
    )
    hasher = blake3()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                hasher.update(chunk)
    except FileNotFoundError as exc:
        raise ValidationError(f"weights file not found: {path}") from exc
    except OSError as exc:
        raise ValidationError(f"weights file could not be read: {path}") from exc
    return f"{_DIGEST_PREFIX}{hasher.hexdigest()}"


def build_encoder_fingerprint(
    encoder_id: str,
    summarizer: Summarizer | str,
    *,
    encoder_config: Mapping[str, Any] | None = None,
    weights: bytes | str | PathLike[str] | None = None,
    weights_digest: str | None = None,
    unweighted: bool = False,
) -> EncoderFingerprint:
    """Build an EncoderFingerprint with strict weight-digest policy."""

    if weights is not None and weights_digest is not None:
        raise ValidationError("provide weights or weights_digest, not both")
    if unweighted and (weights is not None or weights_digest is not None):
        raise ValidationError("unweighted fingerprints cannot include weight digests")
    if not unweighted and weights is None and weights_digest is None:
        raise ValidationError("weights_digest is required unless unweighted=True")

    summarizer_id = _summarizer_id(summarizer)
    config_digest = digest_config(
        {
            "encoder_id": encoder_id,
            "encoder_config": {} if encoder_config is None else encoder_config,
            "summarizer": _summarizer_config(summarizer),
        }
    )
    resolved_weights_digest = weights_digest
    if weights is not None:
        resolved_weights_digest = digest_weights(weights)

    return EncoderFingerprint(
        encoder_id=encoder_id,
        summarizer_id=summarizer_id,
        weights_digest=resolved_weights_digest,
        config_digest=config_digest,
    )


def fingerprints_match(left: EncoderFingerprint, right: EncoderFingerprint) -> bool:
    """Return whether two fingerprints are exactly equal."""

    return left == right


def ensure_fingerprint_match(
    expected: EncoderFingerprint,
    actual: EncoderFingerprint,
) -> None:
    """Fail closed when two fingerprints differ."""

    if expected != actual:
        raise FingerprintMismatchError(
            "encoder fingerprint mismatch: "
            f"expected {format_fingerprint(expected)}, "
            f"got {format_fingerprint(actual)}"
        )


def format_fingerprint(fingerprint: EncoderFingerprint) -> str:
    """Return a compact display string for logs, errors, and manifests."""

    weight = fingerprint.weights_digest or "unweighted"
    return (
        f"{fingerprint.encoder_id}/{fingerprint.summarizer_id}"
        f" config={fingerprint.config_digest} weights={weight}"
    )


def _digest_bytes(data: bytes) -> str:
    return f"{_DIGEST_PREFIX}{blake3(data).hexdigest()}"


def _summarizer_id(summarizer: Summarizer | str) -> str:
    if isinstance(summarizer, str):
        if not summarizer:
            raise ValidationError("summarizer id must not be empty")
        return summarizer
    summarizer_id = summarizer.id
    if not isinstance(summarizer_id, str) or not summarizer_id:
        raise ValidationError("summarizer id must be a non-empty string")
    return summarizer_id


def _summarizer_config(summarizer: Summarizer | str) -> Mapping[str, Any]:
    if isinstance(summarizer, str):
        return {"id": summarizer}
    if is_dataclass(summarizer):
        return {"id": summarizer.id, "config": asdict(summarizer)}
    return {
        "id": summarizer.id,
        "class": f"{type(summarizer).__module__}.{type(summarizer).__qualname__}",
    }


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("configuration floats must be finite")
        return value
    if isinstance(value, Mapping):
        ready: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise ValidationError("configuration keys must be non-empty strings")
            ready[key] = _json_ready(nested)
        return ready
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_ready(item) for item in value]
    raise ValidationError(f"unsupported configuration value: {type(value).__name__}")


__all__ = [
    "build_encoder_fingerprint",
    "digest_config",
    "digest_weights",
    "ensure_fingerprint_match",
    "fingerprints_match",
    "format_fingerprint",
]
