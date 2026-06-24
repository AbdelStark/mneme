"""Structured event hooks for Mneme operations."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

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
            raise TypeError("redact_metadata must be a bool")
        if not isinstance(self.include_content_id_prefixes, bool):
            raise TypeError("include_content_id_prefixes must be a bool")
        if (
            isinstance(self.content_id_prefix_bytes, bool)
            or not isinstance(self.content_id_prefix_bytes, int)
            or self.content_id_prefix_bytes < 0
        ):
            raise ValueError("content_id_prefix_bytes must be a non-negative integer")


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
        payload[key] = _json_safe(value)
    config.event_sink.emit(payload)


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

    if not distances:
        return None
    return float(min(float(distance) for distance in distances))


def distance_mean(distances: Sequence[float]) -> float | None:
    """Return the mean finite distance, or None for empty input."""

    if not distances:
        return None
    return float(sum(float(distance) for distance in distances) / len(distances))


def _duration_ms(started: float | None) -> float:
    if started is None:
        return 0.0
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    return str(value)


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
    "start_event_timer",
]
