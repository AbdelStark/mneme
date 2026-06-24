"""Schema-versioned public data carriers for Mneme."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, TypeAlias
from uuid import UUID

import numpy as np
import numpy.typing as npt

from mneme.core._errors import QueryError

Latent: TypeAlias = Any
SummaryVec: TypeAlias = npt.NDArray[np.float32]
Cid: TypeAlias = bytes

ENCODER_FINGERPRINT_SCHEMA: Final = "mneme.encoder_fingerprint.v1"
SUMMARY_VEC_SCHEMA: Final = "mneme.summary_vec.v1"
TRANSITION_SCHEMA: Final = "mneme.transition.v1"
MEMORY_ITEM_SCHEMA: Final = "mneme.memory_item.v1"
QUERY_SPEC_SCHEMA: Final = "mneme.query_spec.v1"
RETRIEVAL_SCHEMA: Final = "mneme.retrieval.v1"

_SUPPORTED_MAJOR: Final = 1
_RESERVED_META_KEYS: Final = frozenset({"schema_version", "content_id", "encoder_fp"})


class Metric(StrEnum):
    """Distance metric used by query and index contracts."""

    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


@dataclass(frozen=True)
class EncoderFingerprint:
    """Stable identity for an encoder and summarizer pair."""

    encoder_id: str
    summarizer_id: str
    weights_digest: str | None
    config_digest: str
    schema_version: str = ENCODER_FINGERPRINT_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, ENCODER_FINGERPRINT_SCHEMA)
        _require_non_empty_str(self.encoder_id, "encoder_id")
        _require_non_empty_str(self.summarizer_id, "summarizer_id")
        _require_non_empty_str(self.config_digest, "config_digest")
        if self.weights_digest is not None:
            _require_non_empty_str(self.weights_digest, "weights_digest")


@dataclass(frozen=True)
class Transition:
    """A realized latent transition stored as a memory value."""

    z_src: Latent
    action: npt.NDArray[Any]
    z_next: Latent
    delta: Latent
    t: int
    episode_id: UUID
    reward: float | None = None
    schema_version: str = TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, TRANSITION_SCHEMA)
        src = _validate_latent(self.z_src, "z_src")
        z_next = _validate_latent(self.z_next, "z_next")
        delta = _validate_latent(self.delta, "delta")
        if z_next != src or delta != src:
            raise ValueError("z_src, z_next, and delta must share shape and dtype")
        _validate_action(self.action, "action")
        _validate_non_negative_int(self.t, "t")
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        if self.reward is not None:
            _validate_finite_number(self.reward, "reward")


@dataclass(frozen=True)
class MemoryItem:
    """Committed storage unit before canonical content id calculation."""

    content_id: Cid | None
    key: SummaryVec
    value: Transition
    meta: Mapping[str, Any]
    encoder_fp: EncoderFingerprint
    schema_version: str = MEMORY_ITEM_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, MEMORY_ITEM_SCHEMA)
        if self.content_id is not None:
            _validate_cid(self.content_id, "content_id")
        _validate_summary_vec(self.key, "key")
        if not isinstance(self.value, Transition):
            raise TypeError("value must be a Transition")
        if not isinstance(self.encoder_fp, EncoderFingerprint):
            raise TypeError("encoder_fp must be an EncoderFingerprint")
        object.__setattr__(self, "meta", _freeze_metadata(self.meta))


@dataclass(frozen=True)
class QuerySpec:
    """Retrieval query contract."""

    vector: SummaryVec
    k: int
    metric: Metric = Metric.COSINE
    ef: int | None = None
    filters: Mapping[str, Any] | None = None
    temporal_decay: float | None = None
    with_receipt: bool = False
    encoder_fp: EncoderFingerprint | None = None
    schema_version: str = QUERY_SPEC_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, QUERY_SPEC_SCHEMA)
        try:
            _validate_summary_vec(self.vector, "vector")
            _validate_positive_int(self.k, "k")
        except (TypeError, ValueError) as exc:
            raise QueryError(str(exc)) from exc
        if not isinstance(self.metric, Metric):
            raise QueryError("metric must be a Metric")
        if self.ef is not None and _is_invalid_ef(self.ef, self.k):
            raise QueryError("ef must be None or an integer greater than or equal to k")
        if self.filters is not None:
            try:
                object.__setattr__(self, "filters", _freeze_metadata(self.filters))
            except (TypeError, ValueError) as exc:
                raise QueryError(str(exc)) from exc
        if self.temporal_decay is not None:
            try:
                _validate_non_negative_number(self.temporal_decay, "temporal_decay")
            except (TypeError, ValueError) as exc:
                raise QueryError(str(exc)) from exc
        if not isinstance(self.with_receipt, bool):
            raise QueryError("with_receipt must be a bool")
        if self.encoder_fp is not None and not isinstance(
            self.encoder_fp, EncoderFingerprint
        ):
            raise QueryError("encoder_fp must be an EncoderFingerprint or None")


@dataclass(frozen=True)
class Retrieval:
    """Result shell returned by memory-store queries."""

    items: tuple[MemoryItem, ...]
    distances: tuple[float, ...]
    receipt: object | None = None
    schema_version: str = RETRIEVAL_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version, RETRIEVAL_SCHEMA)
        items = _as_tuple(self.items, "items")
        distances = _as_tuple(self.distances, "distances")
        for item in items:
            if not isinstance(item, MemoryItem):
                raise TypeError("items must contain only MemoryItem instances")
        for distance in distances:
            _validate_finite_number(distance, "distance")
        if len(items) != len(distances):
            raise ValueError("items and distances must have matching lengths")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "distances", distances)


def _validate_schema_version(schema_version: str, expected: str) -> None:
    if not isinstance(schema_version, str):
        raise TypeError("schema_version must be a string")
    prefix, _, major_text = schema_version.rpartition(".v")
    expected_prefix, _, _ = expected.rpartition(".v")
    if prefix != expected_prefix or not major_text.isdigit():
        raise ValueError(f"unsupported schema version: {schema_version!r}")
    major = int(major_text)
    if major != _SUPPORTED_MAJOR or schema_version != expected:
        raise ValueError(f"unsupported schema version: {schema_version!r}")


def _require_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _validate_summary_vec(value: object, field_name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a numpy.ndarray")
    if value.dtype != np.float32:
        raise TypeError(f"{field_name} must have dtype float32")
    if value.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if value.shape[0] <= 0:
        raise ValueError(f"{field_name} must not be empty")
    if not value.flags.c_contiguous:
        raise ValueError(f"{field_name} must be contiguous")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{field_name} must contain only finite values")


def _validate_action(value: object, field_name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field_name} must be a numpy.ndarray")
    if value.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if value.shape[0] <= 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{field_name} must have a numeric dtype")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{field_name} must contain only finite values")


def _validate_cid(value: object, field_name: str) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{field_name} must be bytes")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _validate_latent(value: object, field_name: str) -> tuple[tuple[int, ...], str]:
    if isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"{field_name} must have a numeric dtype")
        shape = tuple(int(dim) for dim in value.shape)
        _validate_shape(shape, field_name)
        return shape, str(value.dtype)

    shape_obj = getattr(value, "shape", None)
    dtype_obj = getattr(value, "dtype", None)
    if shape_obj is None or dtype_obj is None:
        raise TypeError(f"{field_name} must expose shape and dtype")
    try:
        shape = tuple(int(dim) for dim in shape_obj)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} shape must be an integer sequence") from exc
    _validate_shape(shape, field_name)
    dtype = str(dtype_obj)
    if not any(token in dtype for token in ("float", "int", "uint")):
        raise TypeError(f"{field_name} must have a numeric dtype")
    return shape, dtype


def _validate_shape(shape: tuple[int, ...], field_name: str) -> None:
    if not shape:
        raise ValueError(f"{field_name} must have at least one dimension")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"{field_name} shape dimensions must be positive")


def _validate_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")


def _validate_non_negative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_finite_number(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


def _validate_non_negative_number(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    if numeric < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _is_invalid_ef(value: object, k: int) -> bool:
    return isinstance(value, bool) or not isinstance(value, int) or value < k


def _freeze_metadata(meta: object) -> Mapping[str, Any]:
    if not isinstance(meta, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen: dict[str, object] = {}
    for key, value in meta.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if not key:
            raise ValueError("metadata keys must not be empty")
        if key in _RESERVED_META_KEYS:
            raise ValueError(f"metadata key {key!r} is reserved")
        frozen[key] = _freeze_json_value(value, f"metadata[{key!r}]")
    return MappingProxyType(frozen)


def _freeze_json_value(value: object, field_name: str) -> object:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} mapping keys must be strings")
            if not key:
                raise ValueError(f"{field_name} mapping keys must not be empty")
            frozen[key] = _freeze_json_value(nested_value, f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze_json_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{field_name} must be JSON-compatible")


def _as_tuple(value: object, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return tuple(value)


__all__ = [
    "Cid",
    "ENCODER_FINGERPRINT_SCHEMA",
    "Latent",
    "MEMORY_ITEM_SCHEMA",
    "QUERY_SPEC_SCHEMA",
    "RETRIEVAL_SCHEMA",
    "SUMMARY_VEC_SCHEMA",
    "TRANSITION_SCHEMA",
    "EncoderFingerprint",
    "MemoryItem",
    "Metric",
    "QuerySpec",
    "Retrieval",
    "SummaryVec",
    "Transition",
]
