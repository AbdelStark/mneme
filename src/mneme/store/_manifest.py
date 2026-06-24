"""Schema-versioned local store manifest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Final
from uuid import UUID

from mneme.core import EncoderFingerprint, SchemaVersionError, StoreCorruptionError

STORE_MANIFEST_SCHEMA: Final = "mneme.store_manifest.v1"
_SUPPORTED_MAJOR: Final = 1
_RETENTION_POLICIES: Final = frozenset({"none", "count", "age"})


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StoreCorruptionError(f"{field_name} must be a non-empty string")
    return value


def _require_store_relative_path(value: object, field_name: str) -> str:
    path = _require_string(value, field_name)
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or "\\" in path
        or ":" in path
        or path.startswith("~")
    ):
        raise StoreCorruptionError(f"{field_name} must be a relative store path")
    return parsed.as_posix()


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreCorruptionError(f"{field_name} must be an integer")
    if value < 0:
        raise StoreCorruptionError(f"{field_name} must be >= 0")
    return value


def count_retention(max_items: int) -> dict[str, Any]:
    """Return a JSON-ready count retention policy."""

    return _require_retention_policy(
        {"policy": "count", "max_items": max_items, "tombstones": []}
    )


def age_retention(max_age_seconds: int) -> dict[str, Any]:
    """Return a JSON-ready event-time age retention policy."""

    return _require_retention_policy(
        {"policy": "age", "max_age_seconds": max_age_seconds, "tombstones": []}
    )


@dataclass(frozen=True)
class ValueLogRef:
    """Manifest reference to one append-only value log."""

    path: str
    size_bytes: int = 0
    record_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _require_store_relative_path(self.path, "value log path"),
        )
        _require_non_negative_int(self.size_bytes, "value log size_bytes")
        _require_non_negative_int(self.record_count, "value log record_count")

    @classmethod
    def from_json(cls, data: object) -> ValueLogRef:
        mapping = _require_mapping(data, "value log")
        path = _require_store_relative_path(mapping.get("path"), "value log path")
        size_bytes = _require_non_negative_int(
            mapping.get("size_bytes"), "value log size_bytes"
        )
        record_count = _require_non_negative_int(
            mapping.get("record_count"), "value log record_count"
        )
        return cls(path=path, size_bytes=size_bytes, record_count=record_count)


@dataclass(frozen=True)
class IndexConfig:
    """Manifest index backend configuration."""

    backend: str = "flat"
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: object) -> IndexConfig:
        mapping = _require_mapping(data, "index")
        backend = _require_string(mapping.get("backend"), "index backend")
        params = _require_json_mapping(mapping.get("params", {}), "index params")
        return cls(backend=backend, params=params)


@dataclass(frozen=True)
class CommitmentState:
    """Reserved commitment fields for RFC-0007 without enabling receipts."""

    enabled: bool = False
    backend: str | None = None
    root: str | None = None
    files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise StoreCorruptionError("commitment enabled must be a bool")
        if self.backend is not None and not isinstance(self.backend, str):
            raise StoreCorruptionError("commitment backend must be a string or null")
        if self.root is not None and not isinstance(self.root, str):
            raise StoreCorruptionError("commitment root must be a string or null")
        if isinstance(self.files, str | bytes | bytearray) or not isinstance(
            self.files, Sequence
        ):
            raise StoreCorruptionError("commitment files must be a sequence")
        object.__setattr__(
            self,
            "files",
            tuple(
                _require_store_relative_path(item, "commitment file")
                for item in self.files
            ),
        )

    @classmethod
    def from_json(cls, data: object) -> CommitmentState:
        mapping = _require_mapping(data, "commitment")
        enabled = mapping.get("enabled")
        if not isinstance(enabled, bool):
            raise StoreCorruptionError("commitment enabled must be a bool")
        backend = mapping.get("backend")
        if backend is not None and not isinstance(backend, str):
            raise StoreCorruptionError("commitment backend must be a string or null")
        root = mapping.get("root")
        if root is not None and not isinstance(root, str):
            raise StoreCorruptionError("commitment root must be a string or null")
        files = mapping.get("files", [])
        if not isinstance(files, list) or not all(
            isinstance(item, str) for item in files
        ):
            raise StoreCorruptionError("commitment files must be a list of strings")
        return cls(
            enabled=enabled,
            backend=backend,
            root=root,
            files=tuple(
                _require_store_relative_path(item, "commitment file") for item in files
            ),
        )


@dataclass(frozen=True)
class StoreManifest:
    """Local store manifest persisted as manifest.json."""

    store_id: UUID
    created_at: str
    updated_at: str
    active_fingerprints: tuple[EncoderFingerprint, ...] = ()
    value_logs: tuple[ValueLogRef, ...] = (ValueLogRef("values/log-000000.mnv"),)
    index: IndexConfig = field(default_factory=IndexConfig)
    retention_policy: Mapping[str, Any] = field(
        default_factory=lambda: {"policy": "none", "tombstones": []}
    )
    last_completed_transaction: str | None = None
    commitment: CommitmentState = field(default_factory=CommitmentState)
    schema_version: str = STORE_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        validate_manifest_schema(self.schema_version)
        if not isinstance(self.store_id, UUID):
            raise StoreCorruptionError("store_id must be a UUID")
        _require_string(self.created_at, "created_at")
        _require_string(self.updated_at, "updated_at")
        object.__setattr__(
            self,
            "active_fingerprints",
            tuple(_require_fingerprint(item) for item in self.active_fingerprints),
        )
        object.__setattr__(
            self,
            "value_logs",
            tuple(_require_value_log(item) for item in self.value_logs),
        )
        if not isinstance(self.index, IndexConfig):
            raise StoreCorruptionError("index must be an IndexConfig")
        object.__setattr__(
            self,
            "retention_policy",
            _require_retention_policy(self.retention_policy),
        )
        if self.last_completed_transaction is not None and not isinstance(
            self.last_completed_transaction, str
        ):
            raise StoreCorruptionError(
                "last_completed_transaction must be string or null"
            )
        if not isinstance(self.commitment, CommitmentState):
            raise StoreCorruptionError("commitment must be a CommitmentState")

    @classmethod
    def from_json(cls, data: object) -> StoreManifest:
        mapping = _require_mapping(data, "manifest")
        validate_manifest_schema(
            _require_string(mapping.get("schema_version"), "schema_version")
        )
        store_id_text = _require_string(mapping.get("store_id"), "store_id")
        try:
            store_id = UUID(store_id_text)
        except ValueError as exc:
            raise StoreCorruptionError("store_id must be a UUID string") from exc
        fingerprints = mapping.get("active_fingerprints", [])
        if not isinstance(fingerprints, list):
            raise StoreCorruptionError("active_fingerprints must be a list")
        value_logs = mapping.get("value_logs", [])
        if not isinstance(value_logs, list):
            raise StoreCorruptionError("value_logs must be a list")
        return cls(
            schema_version=STORE_MANIFEST_SCHEMA,
            store_id=store_id,
            created_at=_require_string(mapping.get("created_at"), "created_at"),
            updated_at=_require_string(mapping.get("updated_at"), "updated_at"),
            active_fingerprints=tuple(
                _fingerprint_from_json(item) for item in fingerprints
            ),
            value_logs=tuple(ValueLogRef.from_json(item) for item in value_logs),
            index=IndexConfig.from_json(mapping.get("index")),
            retention_policy=_require_retention_policy(
                mapping.get("retention_policy", {})
            ),
            last_completed_transaction=_optional_string(
                mapping.get("last_completed_transaction"),
                "last_completed_transaction",
            ),
            commitment=CommitmentState.from_json(mapping.get("commitment")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "store_id": str(self.store_id),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active_fingerprints": [
                asdict(fingerprint) for fingerprint in self.active_fingerprints
            ],
            "value_logs": [asdict(value_log) for value_log in self.value_logs],
            "index": {
                "backend": self.index.backend,
                "params": dict(self.index.params),
            },
            "retention_policy": dict(self.retention_policy),
            "last_completed_transaction": self.last_completed_transaction,
            "commitment": {
                "enabled": self.commitment.enabled,
                "backend": self.commitment.backend,
                "root": self.commitment.root,
                "files": list(self.commitment.files),
            },
        }


def validate_manifest_schema(schema_version: str) -> None:
    if not isinstance(schema_version, str):
        raise SchemaVersionError("manifest schema_version must be a string")
    prefix, _, major_text = schema_version.rpartition(".v")
    expected_prefix, _, _ = STORE_MANIFEST_SCHEMA.rpartition(".v")
    if prefix != expected_prefix or not major_text.isdigit():
        raise SchemaVersionError(f"unsupported manifest schema: {schema_version!r}")
    if int(major_text) != _SUPPORTED_MAJOR or schema_version != STORE_MANIFEST_SCHEMA:
        raise SchemaVersionError(f"unsupported manifest schema: {schema_version!r}")


def _fingerprint_from_json(data: object) -> EncoderFingerprint:
    mapping = _require_mapping(data, "encoder fingerprint")
    try:
        return EncoderFingerprint(
            encoder_id=_require_string(mapping.get("encoder_id"), "encoder_id"),
            summarizer_id=_require_string(
                mapping.get("summarizer_id"), "summarizer_id"
            ),
            weights_digest=_optional_string(
                mapping.get("weights_digest"), "weights_digest"
            ),
            config_digest=_require_string(
                mapping.get("config_digest"), "config_digest"
            ),
            schema_version=_require_string(
                mapping.get("schema_version"), "schema_version"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise StoreCorruptionError("invalid encoder fingerprint") from exc


def _require_mapping(data: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise StoreCorruptionError(f"{field_name} must be an object")
    return data


def _require_json_mapping(data: object, field_name: str) -> Mapping[str, Any]:
    mapping = _require_mapping(data, field_name)
    _validate_json_value(mapping, field_name)
    return dict(mapping)


def _require_retention_policy(data: object) -> dict[str, Any]:
    mapping = dict(_require_json_mapping(data, "retention_policy"))
    policy = mapping.get("policy", "none")
    if not isinstance(policy, str) or policy not in _RETENTION_POLICIES:
        raise StoreCorruptionError("retention_policy policy is unsupported")
    tombstones = _require_tombstones(mapping.get("tombstones", []))
    normalized: dict[str, Any] = {"policy": policy, "tombstones": tombstones}
    if policy == "count":
        normalized["max_items"] = _require_non_negative_int(
            mapping.get("max_items"), "retention_policy max_items"
        )
    elif policy == "age":
        normalized["max_age_seconds"] = _require_non_negative_int(
            mapping.get("max_age_seconds"),
            "retention_policy max_age_seconds",
        )
    return normalized


def _require_tombstones(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise StoreCorruptionError("retention_policy tombstones must be a list")
    tombstones: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        mapping = _require_mapping(raw, f"retention_policy tombstones[{index}]")
        content_id = _require_string(mapping.get("content_id"), "tombstone content_id")
        try:
            bytes.fromhex(content_id)
        except ValueError as exc:
            raise StoreCorruptionError(
                "tombstone content_id must be hex bytes"
            ) from exc
        tombstone: dict[str, Any] = {
            "content_id": content_id,
            "reason": _require_string(mapping.get("reason"), "tombstone reason"),
            "created_at": _require_string(
                mapping.get("created_at"), "tombstone created_at"
            ),
        }
        transaction_id = _optional_string(
            mapping.get("transaction_id"),
            "tombstone transaction_id",
        )
        if transaction_id is not None:
            tombstone["transaction_id"] = transaction_id
        tombstones.append(tombstone)
    tombstones.sort(key=lambda item: item["content_id"])
    return tombstones


def _validate_json_value(value: object, field_name: str) -> None:
    if value is None or isinstance(value, bool | str):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf") or value != value:
            raise StoreCorruptionError(f"{field_name} must contain finite floats")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise StoreCorruptionError(f"{field_name} keys must be strings")
            _validate_json_value(nested, f"{field_name}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]")
        return
    raise StoreCorruptionError(f"{field_name} contains unsupported JSON value")


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise StoreCorruptionError(f"{field_name} must be a non-empty string or null")
    return value


def _require_fingerprint(value: object) -> EncoderFingerprint:
    if not isinstance(value, EncoderFingerprint):
        raise StoreCorruptionError(
            "active_fingerprints must contain EncoderFingerprint"
        )
    return value


def _require_value_log(value: object) -> ValueLogRef:
    if not isinstance(value, ValueLogRef):
        raise StoreCorruptionError("value_logs must contain ValueLogRef")
    return value


__all__ = [
    "STORE_MANIFEST_SCHEMA",
    "CommitmentState",
    "IndexConfig",
    "StoreManifest",
    "ValueLogRef",
    "age_retention",
    "count_retention",
    "validate_manifest_schema",
]
