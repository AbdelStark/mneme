"""Structured event hooks for Mneme operations."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from mneme.core import ValidationError

EVENT_SCHEMA_VERSION: Final = "mneme.event.v1"
REQUIRED_EVENT_NAMES: Final = (
    "mneme.store.put",
    "mneme.store.query",
    "mneme.store.commit",
    "mneme.store.verify",
    "mneme.index.search",
    "mneme.condition.apply",
    "mneme.receipt.verify",
    "mneme.eval.run",
)
_ARRAY_FIELD_MARKERS: Final = (
    "action",
    "delta",
    "latent",
    "summary",
    "vector",
    "z_next",
    "z_src",
)
_OBSERVATION_FIELD_MARKERS: Final = ("observation", "obs")
_PATH_FIELD_MARKERS: Final = ("path", "uri")
_SECRET_FIELD_MARKERS: Final = (
    "api_key",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
_METADATA_FIELD_MARKERS: Final = ("meta", "metadata")
_PRIVATE_DATASET_FIELDS: Final = frozenset({"dataset_id", "dataset_name"})
_SAFE_METADATA_PREFIX: Final = "safe_"


class EventSink(Protocol):
    """Consumer for JSON-serializable structured events."""

    def emit(self, event: Mapping[str, object]) -> None:
        """Receive one structured event."""
        ...


@dataclass(frozen=True)
class ObservabilityConfig:
    """Configuration for optional structured event emission."""

    event_sink: EventSink | None = None
    redact_metadata: bool = True
    include_content_id_prefixes: bool = True
    content_id_prefix_bytes: int = 6

    def __post_init__(self) -> None:
        if not isinstance(self.redact_metadata, bool):
            raise ValidationError("redact_metadata must be a bool")
        if not isinstance(self.include_content_id_prefixes, bool):
            raise ValidationError("include_content_id_prefixes must be a bool")
        if (
            isinstance(self.content_id_prefix_bytes, bool)
            or not isinstance(self.content_id_prefix_bytes, int)
            or self.content_id_prefix_bytes < 0
        ):
            raise ValidationError(
                "content_id_prefix_bytes must be a non-negative integer"
            )


def has_event_sink(config: ObservabilityConfig | None) -> bool:
    """Return whether an event sink is configured."""

    return config is not None and config.event_sink is not None


def start_event_timer(config: ObservabilityConfig | None) -> float | None:
    """Return a monotonic start time only when emission is enabled."""

    if not has_event_sink(config):
        return None
    return time.perf_counter()


def emit_event(
    config: ObservabilityConfig | None,
    *,
    event: str,
    operation: str,
    status: str,
    started: float | None,
    error: BaseException | None = None,
    **fields: object,
) -> None:
    """Emit a JSON-serializable event if a sink is configured."""

    if config is None or config.event_sink is None:
        return
    payload: dict[str, object] = {
        "event": event,
        "schema_version": EVENT_SCHEMA_VERSION,
        "operation": operation,
        "duration_ms": _duration_ms(started),
        "status": status,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
    for key, value in fields.items():
        payload[key] = redact_event_value(value, field_name=key, config=config)
    config.event_sink.emit(payload)


def redact_event_value(
    value: object,
    *,
    field_name: str = "",
    config: ObservabilityConfig | None = None,
) -> object:
    """Return a JSON-safe value with default redaction rules applied."""

    key = field_name.lower()
    if _is_secret_field(key):
        return "<redacted:secret>"
    if key in _PRIVATE_DATASET_FIELDS:
        return "<redacted:dataset>"
    if _is_metadata_field(key):
        return redact_metadata(value, config=config)
    if _is_observation_field(key):
        return "<redacted:observation>"
    if _is_path_field(key) and isinstance(value, str | Path):
        return "<redacted:path>"
    if _is_array_field(key) and _is_array_like(value):
        return _array_summary(value)
    return _json_safe(value, key=key, config=config)


def redact_metadata(
    metadata: object,
    *,
    config: ObservabilityConfig | None = None,
) -> dict[str, object]:
    """Return metadata safe for default event logs."""

    if not isinstance(metadata, Mapping):
        return {}
    if config is not None and not config.redact_metadata:
        return {
            str(key): redact_event_value(value, field_name=str(key), config=config)
            for key, value in metadata.items()
        }
    safe: dict[str, object] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)
        if not key.startswith(_SAFE_METADATA_PREFIX) or _is_secret_field(key.lower()):
            continue
        safe[key] = redact_event_value(value, field_name=key, config=config)
    return safe


def content_id_prefix(cid: bytes, config: ObservabilityConfig | None) -> str | None:
    """Return a configured content id prefix, or None when prefixes are disabled."""

    if config is not None and not config.include_content_id_prefixes:
        return None
    prefix_bytes = 6 if config is None else config.content_id_prefix_bytes
    if prefix_bytes <= 0:
        return None
    return cid[:prefix_bytes].hex()


def distance_min(distances: Sequence[float]) -> float | None:
    """Return the minimum finite distance, or None for empty input."""

    finite = _finite_distances(distances)
    if not finite:
        return None
    return min(finite)


def distance_mean(distances: Sequence[float]) -> float | None:
    """Return the mean finite distance, or None for empty input."""

    finite = _finite_distances(distances)
    if not finite:
        return None
    return sum(finite) / len(finite)


def _finite_distances(distances: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        numeric for distance in distances if math.isfinite(numeric := float(distance))
    )


def _duration_ms(started: float | None) -> float:
    if started is None:
        return 0.0
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _json_safe(
    value: object,
    *,
    key: str,
    config: ObservabilityConfig | None,
) -> object:
    if value is None or isinstance(value, str | bool | int):
        if isinstance(value, str) and _looks_like_absolute_path(value):
            return "<redacted:path>"
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return "<redacted:path>"
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if isinstance(value, bytes):
        return {"redacted": "bytes", "length": len(value)}
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_event_value(
                item,
                field_name=str(item_key),
                config=config,
            )
            for item_key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        if _is_array_field(key) and _is_array_like(value):
            return _array_summary(value)
        return [_json_safe(item, key=key, config=config) for item in value]
    return str(value)


def _array_summary(value: object) -> dict[str, object]:
    if isinstance(value, np.ndarray):
        return {
            "redacted": "array",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return {
            "redacted": "array",
            "shape": [len(value)],
            "dtype": "sequence",
        }
    return {"redacted": "array", "shape": None, "dtype": "unknown"}


def _is_array_like(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return False
    return all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in value
    )


def _is_array_field(key: str) -> bool:
    return any(marker in key for marker in _ARRAY_FIELD_MARKERS)


def _is_observation_field(key: str) -> bool:
    return any(marker in key for marker in _OBSERVATION_FIELD_MARKERS)


def _is_path_field(key: str) -> bool:
    return any(marker in key for marker in _PATH_FIELD_MARKERS)


def _is_secret_field(key: str) -> bool:
    return any(marker in key for marker in _SECRET_FIELD_MARKERS)


def _is_metadata_field(key: str) -> bool:
    return any(marker in key for marker in _METADATA_FIELD_MARKERS)


def _looks_like_absolute_path(value: str) -> bool:
    return value.startswith("/") or (
        len(value) >= 3 and value[1:3] == ":\\" and value[0].isalpha()
    )


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "REQUIRED_EVENT_NAMES",
    "EventSink",
    "ObservabilityConfig",
    "content_id_prefix",
    "distance_mean",
    "distance_min",
    "emit_event",
    "has_event_sink",
    "redact_event_value",
    "redact_metadata",
    "start_event_timer",
]
