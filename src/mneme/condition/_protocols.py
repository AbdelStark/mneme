"""Protocols and context objects for memory conditioning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from mneme.core import Latent, Retrieval, SchemaVersionError, ValidationError

COND_CTX_SCHEMA: Final = "mneme.cond_ctx.v1"


@dataclass(frozen=True)
class CondCtx:
    """Context supplied to a conditioner for one parametric prediction."""

    current_latent: Latent | None
    step: int | None = None
    goal_latent: Latent | None = None
    metadata: Mapping[str, Any] | None = None
    schema_version: str = COND_CTX_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if self.step is not None and (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 0
        ):
            raise ValidationError("step must be None or a non-negative integer")
        if self.metadata is not None:
            object.__setattr__(
                self,
                "metadata",
                _freeze_metadata(self.metadata),
            )

    @property
    def current(self) -> Latent | None:
        """Alias for the current latent used in issue-level contracts."""

        return self.current_latent

    @property
    def goal(self) -> Latent | None:
        """Alias for the optional goal latent used in issue-level contracts."""

        return self.goal_latent


@runtime_checkable
class Conditioner(Protocol):
    """Contract for blending memory retrievals into a parametric prediction."""

    def condition(
        self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx
    ) -> Latent:
        """Return a conditioned latent prediction.

        Implementations must return ``parametric`` unchanged for empty retrievals
        unless they document a stricter typed failure mode.
        """
        ...


def _validate_schema_version(schema_version: str) -> None:
    if not isinstance(schema_version, str):
        raise SchemaVersionError("CondCtx schema_version must be a string")
    if schema_version != COND_CTX_SCHEMA:
        raise SchemaVersionError(f"unsupported CondCtx schema: {schema_version!r}")


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValidationError("metadata must be a mapping")
    return MappingProxyType(
        {
            _require_metadata_key(key): _freeze_metadata_value(value, f"metadata.{key}")
            for key, value in metadata.items()
        }
    )


def _freeze_metadata_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf") or value != value:
            raise ValidationError(f"{field_name} must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_metadata(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze_metadata_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValidationError(f"{field_name} contains unsupported metadata value")


def _require_metadata_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError("metadata keys must be non-empty strings")
    return value


__all__ = ["COND_CTX_SCHEMA", "CondCtx", "Conditioner"]
