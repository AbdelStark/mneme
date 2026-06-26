"""Adapter checkpoint metadata and loading contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from mneme._version import __version__
from mneme.core import (
    ENCODER_FINGERPRINT_SCHEMA,
    EncoderFingerprint,
    FingerprintMismatchError,
    SchemaVersionError,
    ValidationError,
)
from mneme.core._json import loads_strict_json, write_strict_json_file

ADAPTER_CHECKPOINT_SCHEMA: Final = "mneme.adapter_checkpoint.v1"
ADAPTER_CHECKPOINT_METADATA_FILE: Final = "adapter.json"
DEFAULT_ADAPTER_WEIGHTS_FILE: Final = "adapter.safetensors"

_METADATA_FIELDS: Final = frozenset(
    {
        "adapter_kind",
        "adapter_config",
        "base_fingerprint",
        "package_version",
        "schema_version",
        "training_report_uri",
        "weights_file",
    }
)


@dataclass(frozen=True)
class AdapterCheckpointMetadata:
    """Schema-versioned sidecar for an adapter checkpoint artifact."""

    adapter_kind: str
    adapter_config: Mapping[str, Any]
    base_fingerprint: EncoderFingerprint
    training_report_uri: str
    weights_file: str = DEFAULT_ADAPTER_WEIGHTS_FILE
    package_version: str = __version__
    schema_version: str = ADAPTER_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _require_non_empty_str(self.adapter_kind, "adapter_kind")
        _require_non_empty_str(self.training_report_uri, "training_report_uri")
        _require_non_empty_str(self.package_version, "package_version")
        object.__setattr__(
            self,
            "weights_file",
            _require_relative_file(self.weights_file, "weights_file"),
        )
        if not isinstance(self.base_fingerprint, EncoderFingerprint):
            raise ValidationError("base_fingerprint must be an EncoderFingerprint")
        object.__setattr__(
            self,
            "adapter_config",
            _freeze_json_mapping(self.adapter_config, "adapter_config"),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready metadata mapping."""

        return {
            "schema_version": self.schema_version,
            "adapter_kind": self.adapter_kind,
            "adapter_config": _thaw_json(self.adapter_config),
            "base_fingerprint": _fingerprint_to_json(self.base_fingerprint),
            "training_report_uri": self.training_report_uri,
            "weights_file": self.weights_file,
            "package_version": self.package_version,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> AdapterCheckpointMetadata:
        """Build and validate checkpoint metadata from a decoded JSON object."""

        if not isinstance(value, Mapping):
            raise ValidationError("adapter checkpoint metadata must be a mapping")
        missing = sorted(_METADATA_FIELDS - set(value))
        if missing:
            raise ValidationError(
                "adapter checkpoint metadata missing fields: " + ", ".join(missing)
            )
        unsupported = sorted(set(value) - _METADATA_FIELDS)
        if unsupported:
            raise ValidationError(
                "adapter checkpoint metadata contains unsupported fields: "
                + ", ".join(unsupported)
            )
        return cls(
            schema_version=_require_string_field(value, "schema_version"),
            adapter_kind=_require_string_field(value, "adapter_kind"),
            adapter_config=_require_mapping_field(value, "adapter_config"),
            base_fingerprint=_fingerprint_from_json(
                _require_mapping_field(value, "base_fingerprint")
            ),
            training_report_uri=_require_string_field(value, "training_report_uri"),
            weights_file=_require_string_field(value, "weights_file"),
            package_version=_require_string_field(value, "package_version"),
        )


@dataclass(frozen=True)
class AdapterCheckpoint:
    """Validated adapter checkpoint sidecar and weight-file location."""

    metadata: AdapterCheckpointMetadata
    metadata_path: Path
    weights_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, AdapterCheckpointMetadata):
            raise ValidationError("metadata must be AdapterCheckpointMetadata")
        object.__setattr__(
            self,
            "metadata_path",
            _require_path(self.metadata_path, "metadata_path"),
        )
        object.__setattr__(
            self,
            "weights_path",
            _require_path(self.weights_path, "weights_path"),
        )


def save_adapter_checkpoint_metadata(
    path: str | Path,
    metadata: AdapterCheckpointMetadata,
) -> Path:
    """Write adapter checkpoint metadata JSON and return the sidecar path."""

    if not isinstance(metadata, AdapterCheckpointMetadata):
        raise ValidationError("metadata must be AdapterCheckpointMetadata")
    metadata_path = _metadata_path(path)
    try:
        return write_strict_json_file(
            metadata_path,
            metadata.to_json(),
            indent=2,
            sort_keys=True,
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise ValidationError(
            f"adapter checkpoint metadata could not be serialized: {metadata_path}"
        ) from exc
    except OSError as exc:
        raise ValidationError(
            f"adapter checkpoint metadata could not be written: {metadata_path}"
        ) from exc


def load_adapter_checkpoint_metadata(path: str | Path) -> AdapterCheckpointMetadata:
    """Load and validate an adapter checkpoint metadata sidecar."""

    metadata_path = _metadata_path(path)
    try:
        raw = loads_strict_json(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(
            f"adapter checkpoint metadata not found: {metadata_path}"
        ) from exc
    except OSError as exc:
        raise ValidationError(
            f"adapter checkpoint metadata could not be read: {metadata_path}"
        ) from exc
    except ValueError as exc:
        raise ValidationError(
            f"adapter checkpoint metadata is not valid JSON: {metadata_path}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ValidationError("adapter checkpoint metadata must be a mapping")
    return AdapterCheckpointMetadata.from_json(raw)


def load_adapter_checkpoint(
    path: str | Path,
    *,
    expected_base_fingerprint: EncoderFingerprint | None = None,
    require_weights: bool = True,
) -> AdapterCheckpoint:
    """Load a checkpoint sidecar and validate its base fingerprint and weights."""

    if not isinstance(require_weights, bool):
        raise ValidationError("require_weights must be a bool")
    metadata_path = _metadata_path(path)
    metadata = load_adapter_checkpoint_metadata(metadata_path)
    if expected_base_fingerprint is not None:
        if not isinstance(expected_base_fingerprint, EncoderFingerprint):
            raise ValidationError(
                "expected_base_fingerprint must be an EncoderFingerprint"
            )
        if metadata.base_fingerprint != expected_base_fingerprint:
            raise FingerprintMismatchError(
                "adapter checkpoint base fingerprint does not match expected base"
            )
    weights_path = metadata_path.parent / metadata.weights_file
    if require_weights and not weights_path.is_file():
        raise ValidationError(f"adapter weights file not found: {weights_path}")
    return AdapterCheckpoint(
        metadata=metadata,
        metadata_path=metadata_path,
        weights_path=weights_path,
    )


def _metadata_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == ".json":
        return candidate
    return candidate / ADAPTER_CHECKPOINT_METADATA_FILE


def _validate_schema_version(schema_version: str) -> None:
    if not isinstance(schema_version, str):
        raise SchemaVersionError("adapter checkpoint schema_version must be a string")
    if schema_version != ADAPTER_CHECKPOINT_SCHEMA:
        raise SchemaVersionError(
            f"unsupported adapter checkpoint schema: {schema_version!r}"
        )


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_string_field(value: Mapping[str, object], field_name: str) -> str:
    return _require_non_empty_str(value[field_name], field_name)


def _require_nested_string_field(
    value: Mapping[str, object],
    field_name: str,
    parent_name: str,
) -> str:
    return _require_non_empty_str(value[field_name], f"{parent_name}.{field_name}")


def _require_mapping_field(
    value: Mapping[str, object],
    field_name: str,
) -> Mapping[str, object]:
    field_value = value[field_name]
    if not isinstance(field_value, Mapping):
        raise ValidationError(f"{field_name} must be a mapping")
    return field_value


def _require_relative_file(value: object, field_name: str) -> str:
    text = _require_non_empty_str(value, field_name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValidationError(f"{field_name} must be a relative file path")
    return text


def _require_path(value: object, field_name: str) -> Path:
    if isinstance(value, str) and not value:
        raise ValidationError(f"{field_name} must not be empty")
    if isinstance(value, str | Path):
        return Path(value)
    raise ValidationError(f"{field_name} must be a path-like value")


def _freeze_json_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be a mapping")
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValidationError(f"{field_name} keys must be non-empty strings")
        frozen[key] = _freeze_json_value(item, f"{field_name}.{key}")
    return MappingProxyType(frozen)


def _freeze_json_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf") or value != value:
            raise ValidationError(f"{field_name} must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze_json_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValidationError(f"{field_name} must be JSON-compatible")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _fingerprint_to_json(fingerprint: EncoderFingerprint) -> dict[str, object]:
    return {
        "schema_version": fingerprint.schema_version,
        "encoder_id": fingerprint.encoder_id,
        "summarizer_id": fingerprint.summarizer_id,
        "weights_digest": fingerprint.weights_digest,
        "config_digest": fingerprint.config_digest,
    }


def _fingerprint_from_json(value: Mapping[str, object]) -> EncoderFingerprint:
    required = {
        "schema_version",
        "encoder_id",
        "summarizer_id",
        "weights_digest",
        "config_digest",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValidationError("base_fingerprint missing fields: " + ", ".join(missing))
    unsupported = sorted(set(value) - required)
    if unsupported:
        raise ValidationError(
            "base_fingerprint contains unsupported fields: " + ", ".join(unsupported)
        )
    schema_version = _require_nested_string_field(
        value,
        "schema_version",
        "base_fingerprint",
    )
    if schema_version != ENCODER_FINGERPRINT_SCHEMA:
        raise SchemaVersionError(
            f"unsupported EncoderFingerprint schema: {schema_version!r}"
        )
    weights_digest = value["weights_digest"]
    if weights_digest is not None and not isinstance(weights_digest, str):
        raise ValidationError(
            "base_fingerprint.weights_digest must be a string or null"
        )
    try:
        return EncoderFingerprint(
            schema_version=schema_version,
            encoder_id=_require_nested_string_field(
                value,
                "encoder_id",
                "base_fingerprint",
            ),
            summarizer_id=_require_nested_string_field(
                value,
                "summarizer_id",
                "base_fingerprint",
            ),
            weights_digest=weights_digest,
            config_digest=_require_nested_string_field(
                value,
                "config_digest",
                "base_fingerprint",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(str(exc)) from exc


__all__ = [
    "ADAPTER_CHECKPOINT_METADATA_FILE",
    "ADAPTER_CHECKPOINT_SCHEMA",
    "DEFAULT_ADAPTER_WEIGHTS_FILE",
    "AdapterCheckpoint",
    "AdapterCheckpointMetadata",
    "load_adapter_checkpoint",
    "load_adapter_checkpoint_metadata",
    "save_adapter_checkpoint_metadata",
]
