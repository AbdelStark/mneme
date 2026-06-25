"""Schema-versioned evaluation reports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias

from mneme.core import SchemaVersionError, ValidationError
from mneme.core._json import dumps_strict_json

DATASET_REF_SCHEMA: Final = "mneme.dataset_ref.v1"
EVAL_REPORT_SCHEMA: Final = "mneme.eval_report.v1"

DatasetKind: TypeAlias = Literal["fixture", "external"]
EvalMetric: TypeAlias = float | int | str


@dataclass(frozen=True)
class DatasetRef:
    """Dataset or fixture identity attached to an evaluation report."""

    dataset_id: str
    kind: DatasetKind
    split: str | None = None
    version: str | None = None
    uri: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DATASET_REF_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, DATASET_REF_SCHEMA, "dataset")
        _require_string(self.dataset_id, "dataset_id")
        object.__setattr__(self, "kind", _dataset_kind(self.kind))
        _optional_string(self.split, "split")
        _optional_string(self.version, "version")
        _optional_string(self.uri, "uri")
        object.__setattr__(
            self,
            "metadata",
            _freeze_json_mapping(self.metadata, "dataset metadata"),
        )

    @classmethod
    def from_json(cls, data: object) -> DatasetRef:
        mapping = _require_mapping(data, "dataset")
        return cls(
            schema_version=_require_string(
                mapping.get("schema_version"), "dataset schema_version"
            ),
            dataset_id=_require_string(mapping.get("dataset_id"), "dataset_id"),
            kind=_dataset_kind(mapping.get("kind")),
            split=_optional_string(mapping.get("split"), "split"),
            version=_optional_string(mapping.get("version"), "version"),
            uri=_optional_string(mapping.get("uri"), "uri"),
            metadata=_freeze_json_mapping(
                mapping.get("metadata", {}),
                "dataset metadata",
            ),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dataset reference."""

        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "kind": self.kind,
            "split": self.split,
            "version": self.version,
            "uri": self.uri,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True)
class EvalReport:
    """Evidence envelope for fixture and benchmark evaluations."""

    report_id: str
    command: Sequence[str]
    package_version: str
    git_commit: str | None
    created_at: str
    platform: Mapping[str, str]
    seed: int | None
    dataset: DatasetRef
    metrics: Mapping[str, EvalMetric]
    artifacts: Mapping[str, str]
    caveats: Sequence[str]
    passed: bool
    schema_version: str = EVAL_REPORT_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, EVAL_REPORT_SCHEMA, "report")
        _require_string(self.report_id, "report_id")
        object.__setattr__(self, "command", _string_tuple(self.command, "command"))
        _require_string(self.package_version, "package_version")
        _optional_string(self.git_commit, "git_commit")
        _require_string(self.created_at, "created_at")
        object.__setattr__(
            self,
            "platform",
            _freeze_string_mapping(self.platform, "platform"),
        )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValidationError("seed must be None or an integer")
        if not isinstance(self.dataset, DatasetRef):
            raise ValidationError("dataset must be a DatasetRef")
        object.__setattr__(
            self,
            "metrics",
            _freeze_metrics(self.metrics),
        )
        object.__setattr__(
            self,
            "artifacts",
            _freeze_string_mapping(self.artifacts, "artifacts"),
        )
        caveats = _string_tuple(self.caveats, "caveats")
        if self.dataset.kind == "fixture" and not caveats:
            raise ValidationError("fixture reports must include caveats")
        object.__setattr__(self, "caveats", caveats)
        if not isinstance(self.passed, bool):
            raise ValidationError("passed must be a bool")

    @classmethod
    def from_json(cls, data: object) -> EvalReport:
        mapping = _require_mapping(data, "report")
        return cls(
            schema_version=_require_string(
                mapping.get("schema_version"), "schema_version"
            ),
            report_id=_require_string(mapping.get("report_id"), "report_id"),
            command=_string_tuple(
                _require_sequence(mapping.get("command"), "command"),
                "command",
            ),
            package_version=_require_string(
                mapping.get("package_version"), "package_version"
            ),
            git_commit=_optional_string(mapping.get("git_commit"), "git_commit"),
            created_at=_require_string(mapping.get("created_at"), "created_at"),
            platform=_freeze_string_mapping(mapping.get("platform"), "platform"),
            seed=_optional_int(mapping.get("seed"), "seed"),
            dataset=DatasetRef.from_json(mapping.get("dataset")),
            metrics=_freeze_metrics(
                _require_mapping(mapping.get("metrics"), "metrics")
            ),
            artifacts=_freeze_string_mapping(mapping.get("artifacts"), "artifacts"),
            caveats=_string_tuple(
                _require_sequence(mapping.get("caveats"), "caveats"),
                "caveats",
            ),
            passed=_require_bool(mapping.get("passed"), "passed"),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "command": list(self.command),
            "package_version": self.package_version,
            "git_commit": self.git_commit,
            "created_at": self.created_at,
            "platform": dict(self.platform),
            "seed": self.seed,
            "dataset": self.dataset.to_json(),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "caveats": list(self.caveats),
            "passed": self.passed,
        }


def validate_report_json(data: object) -> EvalReport:
    """Validate JSON-decoded report data and return an `EvalReport`."""

    return EvalReport.from_json(data)


def write_report_json(report: EvalReport, path: str | Path) -> None:
    """Write a schema-versioned evaluation report as deterministic JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        dumps_strict_json(report.to_json(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_schema(schema_version: str, expected: str, name: str) -> None:
    if not isinstance(schema_version, str):
        raise SchemaVersionError(f"{name} schema_version must be a string")
    if schema_version != expected:
        raise SchemaVersionError(f"unsupported {name} schema: {schema_version!r}")


def _dataset_kind(value: object) -> DatasetKind:
    if value == "fixture" or value == "external":
        return value
    raise ValidationError("dataset kind must be 'fixture' or 'external'")


def _require_mapping(data: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return data


def _require_sequence(data: object, field_name: str) -> Sequence[object]:
    if not isinstance(data, Sequence) or isinstance(data, str | bytes | bytearray):
        raise ValidationError(f"{field_name} must be a list")
    return data


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be None or an integer")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a bool")
    return value


def _string_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes | bytearray) or not isinstance(
        values,
        Sequence,
    ):
        raise ValidationError(f"{field_name} must be a sequence of strings")
    result = tuple(_require_string(value, f"{field_name} item") for value in values)
    if field_name == "command" and not result:
        raise ValidationError("command must not be empty")
    return result


def _freeze_string_mapping(data: object, field_name: str) -> Mapping[str, str]:
    mapping = _require_mapping(data, field_name)
    return MappingProxyType(
        {
            _require_string(key, f"{field_name} key"): _require_string(
                value,
                f"{field_name}.{key}",
            )
            for key, value in mapping.items()
        }
    )


def _freeze_metrics(metrics: object) -> Mapping[str, EvalMetric]:
    mapping = _require_mapping(metrics, "metrics")
    frozen: dict[str, EvalMetric] = {}
    for key, value in mapping.items():
        metric_name = _require_string(key, "metric name")
        if isinstance(value, bool):
            raise ValidationError(f"metric {metric_name} must not be bool")
        if isinstance(value, int):
            frozen[metric_name] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValidationError(f"metric {metric_name} must be finite")
            frozen[metric_name] = value
        elif isinstance(value, str) and value:
            frozen[metric_name] = value
        else:
            raise ValidationError(
                f"metric {metric_name} must be a finite number or string"
            )
    return MappingProxyType(frozen)


def _freeze_json_mapping(data: object, field_name: str) -> Mapping[str, Any]:
    mapping = _require_mapping(data, field_name)
    return MappingProxyType(
        {
            _require_string(key, f"{field_name} key"): _freeze_json_value(
                value,
                f"{field_name}.{key}",
            )
            for key, value in mapping.items()
        }
    )


def _freeze_json_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name} must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, field_name)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze_json_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValidationError(f"{field_name} contains unsupported JSON value")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


__all__ = [
    "DATASET_REF_SCHEMA",
    "EVAL_REPORT_SCHEMA",
    "DatasetKind",
    "DatasetRef",
    "EvalMetric",
    "EvalReport",
    "validate_report_json",
    "write_report_json",
]
