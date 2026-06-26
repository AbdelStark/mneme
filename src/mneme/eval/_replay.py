"""Receipt-backed conditioning replay harness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from mneme.condition import CondCtx, KnnCorrector, KnnMode
from mneme.core import (
    Cid,
    EncoderFingerprint,
    EvaluationError,
    MemoryItem,
    Metric,
    MnemeError,
    QuerySpec,
    Retrieval,
    StoreCorruptionError,
    content_id,
)
from mneme.core._ids import cid_from_hex, require_cid_bytes
from mneme.core._json import loads_strict_json, write_strict_json_file
from mneme.receipts import RetrievalReceipt, verify_retrieval_receipt
from mneme.store._value_log import (
    _array_from_json,
    _array_to_json,
    _fingerprint_from_json,
    _json_ready,
    _transition_from_json,
    _transition_to_json,
)

RECEIPT_REPLAY_TRACE_SCHEMA: Final = "mneme.receipt_replay_trace.v1"
RECEIPT_REPLAY_REPORT_SCHEMA: Final = "mneme.receipt_replay_report.v1"


@dataclass(frozen=True)
class KnnReplayConfig:
    """Replay-safe subset of the KNN conditioner configuration."""

    tau: float = 0.1
    lambda_max: float = 0.5
    alpha: float = 10.0
    delta0: float = 0.2
    mode: KnnMode = "delta"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tau", _require_positive_float(self.tau, "tau"))
        object.__setattr__(
            self,
            "lambda_max",
            _require_probability(self.lambda_max, "lambda_max"),
        )
        object.__setattr__(
            self,
            "alpha",
            _require_non_negative_float(self.alpha, "alpha"),
        )
        object.__setattr__(
            self,
            "delta0",
            _require_finite_float(self.delta0, "delta0"),
        )
        object.__setattr__(self, "mode", _knn_mode(self.mode))

    def conditioner(self) -> KnnCorrector:
        """Return the configured conditioner."""

        return KnnCorrector(
            tau=self.tau,
            lambda_max=self.lambda_max,
            alpha=self.alpha,
            delta0=self.delta0,
            mode=self.mode,
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready conditioner config."""

        return {
            "kind": "knn_corrector",
            "tau": self.tau,
            "lambda_max": self.lambda_max,
            "alpha": self.alpha,
            "delta0": self.delta0,
            "mode": self.mode,
        }

    @classmethod
    def from_json(cls, data: object) -> KnnReplayConfig:
        mapping = _require_mapping(data, "conditioner")
        if mapping.get("kind") != "knn_corrector":
            raise EvaluationError("unsupported replay conditioner")
        return cls(
            tau=_require_finite_float(mapping.get("tau"), "tau"),
            lambda_max=_require_finite_float(
                mapping.get("lambda_max"),
                "lambda_max",
            ),
            alpha=_require_finite_float(mapping.get("alpha"), "alpha"),
            delta0=_require_finite_float(mapping.get("delta0"), "delta0"),
            mode=_knn_mode(mapping.get("mode")),
        )


@dataclass(frozen=True)
class ReceiptReplayTrace:
    """Logged conditioning set bound to a retrieval receipt."""

    query: QuerySpec
    items: tuple[MemoryItem, ...]
    distances: tuple[float, ...]
    receipt: RetrievalReceipt
    conditioner: KnnReplayConfig
    parametric_prediction: np.ndarray
    current_latent: np.ndarray
    expected_prediction: np.ndarray
    schema_version: str = RECEIPT_REPLAY_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_REPLAY_TRACE_SCHEMA:
            raise EvaluationError("unsupported replay trace schema")
        if not isinstance(self.query, QuerySpec):
            raise EvaluationError("query must be a QuerySpec")
        object.__setattr__(
            self, "items", tuple(_require_item(item) for item in self.items)
        )
        distances = tuple(
            _require_finite_float(distance, "distance") for distance in self.distances
        )
        if len(self.items) != len(distances):
            raise EvaluationError("items and distances must have matching lengths")
        object.__setattr__(self, "distances", distances)
        if not isinstance(self.receipt, RetrievalReceipt):
            raise EvaluationError("receipt must be a RetrievalReceipt")
        if not isinstance(self.conditioner, KnnReplayConfig):
            raise EvaluationError("conditioner must be KnnReplayConfig")
        object.__setattr__(
            self,
            "parametric_prediction",
            _require_array(self.parametric_prediction, "parametric_prediction"),
        )
        object.__setattr__(
            self,
            "current_latent",
            _require_array(self.current_latent, "current_latent"),
        )
        object.__setattr__(
            self,
            "expected_prediction",
            _require_array(self.expected_prediction, "expected_prediction"),
        )

    @property
    def root(self) -> bytes:
        """Return the committed root bound to the receipt."""

        return self.receipt.root

    @property
    def ids(self) -> tuple[Cid, ...]:
        """Return receipt ids in retrieval order."""

        return self.receipt.ids

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready replay trace."""

        return {
            "schema_version": self.schema_version,
            "query": _query_to_json(self.query),
            "items": [_item_to_json(item) for item in self.items],
            "distances": list(self.distances),
            "receipt": self.receipt.to_json(),
            "conditioner": self.conditioner.to_json(),
            "parametric_prediction": _array_to_json(self.parametric_prediction),
            "current_latent": _array_to_json(self.current_latent),
            "expected_prediction": _array_to_json(self.expected_prediction),
        }

    @classmethod
    def from_json(cls, data: object) -> ReceiptReplayTrace:
        mapping = _require_mapping(data, "replay trace")
        schema_version = _require_string(
            mapping.get("schema_version"), "schema_version"
        )
        if schema_version != RECEIPT_REPLAY_TRACE_SCHEMA:
            raise EvaluationError("unsupported replay trace schema")
        items = _require_sequence(mapping.get("items"), "items")
        distances = _require_sequence(mapping.get("distances"), "distances")
        try:
            return cls(
                schema_version=schema_version,
                query=_query_from_json(mapping.get("query")),
                items=tuple(_item_from_json(item) for item in items),
                distances=tuple(
                    _require_finite_float(item, "distance") for item in distances
                ),
                receipt=_receipt_from_json(mapping.get("receipt")),
                conditioner=KnnReplayConfig.from_json(mapping.get("conditioner")),
                parametric_prediction=_array_field_from_json(
                    mapping.get("parametric_prediction"),
                    "parametric_prediction",
                ),
                current_latent=_array_field_from_json(
                    mapping.get("current_latent"),
                    "current_latent",
                ),
                expected_prediction=_array_field_from_json(
                    mapping.get("expected_prediction"),
                    "expected_prediction",
                ),
            )
        except EvaluationError:
            raise
        except MnemeError as exc:
            raise EvaluationError("invalid replay trace payload") from exc


@dataclass(frozen=True)
class ReceiptReplayReport:
    """Result of replaying one receipt-bound conditioning trace."""

    ok: bool
    root: bytes
    ids: tuple[Cid, ...]
    conditioned: bool
    mismatch_causes: tuple[str, ...]
    max_abs_error: float | None
    expected_prediction: np.ndarray
    replayed_prediction: np.ndarray | None
    schema_version: str = RECEIPT_REPLAY_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_REPLAY_REPORT_SCHEMA:
            raise EvaluationError("unsupported replay report schema")
        object.__setattr__(self, "ok", _require_bool(self.ok, "ok"))
        object.__setattr__(self, "root", _require_digest(self.root, "root"))
        object.__setattr__(self, "ids", _cid_tuple(self.ids, "ids"))
        object.__setattr__(
            self, "conditioned", _require_bool(self.conditioned, "conditioned")
        )
        object.__setattr__(
            self,
            "mismatch_causes",
            _string_tuple(self.mismatch_causes, "mismatch_causes"),
        )
        object.__setattr__(
            self,
            "max_abs_error",
            _optional_non_negative_float(self.max_abs_error, "max_abs_error"),
        )
        object.__setattr__(
            self,
            "expected_prediction",
            _require_array(self.expected_prediction, "expected_prediction"),
        )
        if self.replayed_prediction is not None:
            object.__setattr__(
                self,
                "replayed_prediction",
                _require_array(self.replayed_prediction, "replayed_prediction"),
            )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready replay report."""

        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "root": self.root.hex(),
            "ids": [cid.hex() for cid in self.ids],
            "conditioned": self.conditioned,
            "mismatch_causes": list(self.mismatch_causes),
            "max_abs_error": self.max_abs_error,
            "expected_prediction": _array_to_json(self.expected_prediction),
            "replayed_prediction": None
            if self.replayed_prediction is None
            else _array_to_json(self.replayed_prediction),
        }


def build_receipt_replay_trace(
    *,
    query: QuerySpec,
    retrieval: Retrieval,
    parametric_prediction: np.ndarray,
    current_latent: np.ndarray,
    conditioner: KnnReplayConfig | None = None,
    expected_prediction: np.ndarray | None = None,
) -> ReceiptReplayTrace:
    """Record a replayable conditioning set from a receipt-bearing retrieval."""

    receipt = _require_receipt(retrieval)
    config = KnnReplayConfig() if conditioner is None else conditioner
    if not verify_retrieval_receipt(
        receipt, retrieval.items, root=receipt.root, query=query
    ):
        raise EvaluationError("retrieval receipt does not verify")
    expected = (
        _condition(config, parametric_prediction, retrieval, current_latent)
        if expected_prediction is None
        else expected_prediction
    )
    return ReceiptReplayTrace(
        query=query,
        items=tuple(retrieval.items),
        distances=tuple(retrieval.distances),
        receipt=receipt,
        conditioner=config,
        parametric_prediction=parametric_prediction,
        current_latent=current_latent,
        expected_prediction=expected,
    )


def replay_receipt_trace(
    trace: ReceiptReplayTrace,
    *,
    atol: float = 1e-6,
) -> ReceiptReplayReport:
    """Replay a receipt-bound trace and report deterministic mismatch causes."""

    _require_non_negative_float(atol, "atol")
    if not verify_retrieval_receipt(
        trace.receipt,
        trace.items,
        root=trace.root,
        query=trace.query,
    ):
        return ReceiptReplayReport(
            ok=False,
            root=trace.root,
            ids=trace.ids,
            conditioned=False,
            mismatch_causes=("receipt_verification_failed",),
            max_abs_error=None,
            expected_prediction=trace.expected_prediction,
            replayed_prediction=None,
        )
    retrieval = Retrieval(
        items=trace.items,
        distances=trace.distances,
        receipt=trace.receipt,
    )
    replayed = _condition(
        trace.conditioner,
        trace.parametric_prediction,
        retrieval,
        trace.current_latent,
    )
    causes: list[str] = []
    max_abs_error = _max_abs_error(replayed, trace.expected_prediction)
    if max_abs_error is None:
        causes.append("prediction_shape_mismatch")
    elif max_abs_error > atol:
        causes.append("prediction_value_mismatch")
    return ReceiptReplayReport(
        ok=not causes,
        root=trace.root,
        ids=trace.ids,
        conditioned=True,
        mismatch_causes=tuple(causes),
        max_abs_error=max_abs_error,
        expected_prediction=trace.expected_prediction,
        replayed_prediction=replayed,
    )


def write_replay_trace_json(trace: ReceiptReplayTrace, path: str | Path) -> None:
    """Write a replay trace JSON artifact."""

    try:
        data = trace.to_json()
    except (MnemeError, TypeError, ValueError) as exc:
        raise EvaluationError(f"replay trace could not be serialized: {path}") from exc
    _write_json(path, data, artifact_name="replay trace")


def load_replay_trace_json(path: str | Path) -> ReceiptReplayTrace:
    """Load and validate a replay trace JSON artifact."""

    try:
        data = loads_strict_json(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"replay trace not found: {path}") from exc
    except OSError as exc:
        raise EvaluationError(f"replay trace could not be read: {path}") from exc
    except ValueError as exc:
        raise EvaluationError("replay trace is not valid JSON") from exc
    return ReceiptReplayTrace.from_json(data)


def write_replay_report_json(report: ReceiptReplayReport, path: str | Path) -> None:
    """Write a replay report JSON artifact."""

    try:
        data = report.to_json()
    except (MnemeError, TypeError, ValueError) as exc:
        raise EvaluationError(f"replay report could not be serialized: {path}") from exc
    _write_json(path, data, artifact_name="replay report")


def _condition(
    config: KnnReplayConfig,
    parametric_prediction: np.ndarray,
    retrieval: Retrieval,
    current_latent: np.ndarray,
) -> np.ndarray:
    result = config.conditioner().condition(
        _require_array(parametric_prediction, "parametric_prediction"),
        retrieval,
        CondCtx(_require_array(current_latent, "current_latent")),
    )
    return _require_array(result, "replayed_prediction")


def _query_to_json(spec: QuerySpec) -> dict[str, object]:
    return {
        "schema_version": spec.schema_version,
        "vector": _array_to_json(spec.vector),
        "k": spec.k,
        "metric": spec.metric.value,
        "ef": spec.ef,
        "filters": None if spec.filters is None else _json_ready(spec.filters),
        "temporal_decay": spec.temporal_decay,
        "with_receipt": spec.with_receipt,
        "encoder_fp": None if spec.encoder_fp is None else asdict(spec.encoder_fp),
    }


def _query_from_json(data: object) -> QuerySpec:
    mapping = _require_mapping(data, "query")
    try:
        return QuerySpec(
            vector=_array_field_from_json(mapping.get("vector"), "query.vector"),
            k=_require_positive_int(mapping.get("k"), "k"),
            metric=_metric(mapping.get("metric")),
            ef=_optional_positive_int(mapping.get("ef"), "ef"),
            filters=_optional_mapping(mapping.get("filters"), "filters"),
            temporal_decay=_optional_non_negative_float(
                mapping.get("temporal_decay"),
                "temporal_decay",
            ),
            with_receipt=_require_bool(mapping.get("with_receipt"), "with_receipt"),
            encoder_fp=_optional_fingerprint(mapping.get("encoder_fp")),
            schema_version=_require_string(
                mapping.get("schema_version"),
                "schema_version",
            ),
        )
    except EvaluationError:
        raise
    except MnemeError as exc:
        raise EvaluationError("invalid replay query payload") from exc


def _item_to_json(item: MemoryItem) -> dict[str, object]:
    cid = item.content_id or content_id(item)
    return {
        "schema_version": item.schema_version,
        "content_id": cid.hex(),
        "key": _array_to_json(item.key),
        "value": _transition_to_json(item.value),
        "meta": _json_ready(item.meta),
        "encoder_fp": asdict(item.encoder_fp),
    }


def _item_from_json(data: object) -> MemoryItem:
    mapping = _require_mapping(data, "item")
    cid = _bytes_from_hex(mapping.get("content_id"), "content_id")
    try:
        item = MemoryItem(
            content_id=cid,
            key=_array_from_json(mapping.get("key")),
            value=_transition_from_json(mapping.get("value")),
            meta=_require_mapping(mapping.get("meta"), "meta"),
            encoder_fp=_fingerprint_from_json(mapping.get("encoder_fp")),
            schema_version=_require_string(
                mapping.get("schema_version"), "schema_version"
            ),
        )
    except EvaluationError:
        raise
    except (MnemeError, TypeError, ValueError) as exc:
        raise EvaluationError("invalid replay item payload") from exc
    if item.content_id != content_id(item):
        raise EvaluationError("item content_id does not match canonical bytes")
    return item


def _receipt_from_json(data: object) -> RetrievalReceipt:
    try:
        return RetrievalReceipt.from_json(data)
    except MnemeError as exc:
        raise EvaluationError("invalid replay receipt payload") from exc


def _array_field_from_json(data: object, field_name: str) -> np.ndarray:
    try:
        return _array_from_json(data)
    except StoreCorruptionError as exc:
        raise EvaluationError(f"{field_name} is invalid") from exc


def _require_item(item: object) -> MemoryItem:
    if not isinstance(item, MemoryItem):
        raise EvaluationError("items must contain MemoryItem instances")
    return item


def _require_receipt(retrieval: Retrieval) -> RetrievalReceipt:
    if not isinstance(retrieval.receipt, RetrievalReceipt):
        raise EvaluationError("replay requires a retrieval receipt")
    return retrieval.receipt


def _require_array(value: object, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise EvaluationError(f"{field_name} must be a numpy.ndarray")
    if not np.issubdtype(value.dtype, np.floating):
        raise EvaluationError(f"{field_name} must have a floating dtype")
    if value.shape == ():
        raise EvaluationError(f"{field_name} must have at least one dimension")
    if any(dim <= 0 for dim in value.shape):
        raise EvaluationError(f"{field_name} dimensions must be positive")
    if not bool(np.isfinite(value).all()):
        raise EvaluationError(f"{field_name} must contain only finite values")
    return np.ascontiguousarray(value)


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape:
        return None
    return float(np.max(np.abs(left - right)))


def _write_json(
    path: str | Path,
    data: Mapping[str, object],
    *,
    artifact_name: str,
) -> None:
    try:
        write_strict_json_file(path, data, sort_keys=True, indent=2)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(
            f"{artifact_name} could not be serialized: {path}"
        ) from exc
    except OSError as exc:
        raise EvaluationError(f"{artifact_name} could not be written: {path}") from exc


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{field_name} must be an object")
    return value


def _require_sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise EvaluationError(f"{field_name} must be a sequence")
    return value


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{field_name} must be a non-empty string")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    return tuple(
        _require_string(item, f"{field_name} item")
        for item in _require_sequence(value, field_name)
    )


def _cid_tuple(value: object, field_name: str) -> tuple[Cid, ...]:
    return tuple(
        _require_digest(item, f"{field_name} item")
        for item in _require_sequence(value, field_name)
    )


def _require_digest(value: object, field_name: str) -> bytes:
    return require_cid_bytes(
        value,
        field_name,
        type_error=EvaluationError,
        value_error=EvaluationError,
    )


def _require_finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationError(f"{field_name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EvaluationError(f"{field_name} must be finite")
    return numeric


def _require_positive_float(value: object, field_name: str) -> float:
    numeric = _require_finite_float(value, field_name)
    if numeric <= 0.0:
        raise EvaluationError(f"{field_name} must be positive")
    return numeric


def _require_non_negative_float(value: object, field_name: str) -> float:
    numeric = _require_finite_float(value, field_name)
    if numeric < 0.0:
        raise EvaluationError(f"{field_name} must be non-negative")
    return numeric


def _require_probability(value: object, field_name: str) -> float:
    numeric = _require_finite_float(value, field_name)
    if numeric < 0.0 or numeric > 1.0:
        raise EvaluationError(f"{field_name} must be between 0 and 1")
    return numeric


def _optional_non_negative_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_non_negative_float(value, field_name)


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, field_name)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationError(f"{field_name} must be a bool")
    return value


def _optional_mapping(
    value: object,
    field_name: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _require_mapping(value, field_name)


def _metric(value: object) -> Metric:
    text = _require_string(value, "metric")
    try:
        return Metric(text)
    except ValueError as exc:
        raise EvaluationError(f"unsupported metric: {text}") from exc


def _knn_mode(value: object) -> KnnMode:
    if value == "delta" or value == "absolute":
        return value
    raise EvaluationError("mode must be 'delta' or 'absolute'")


def _optional_fingerprint(value: object) -> EncoderFingerprint | None:
    if value is None:
        return None
    try:
        return _fingerprint_from_json(value)
    except StoreCorruptionError as exc:
        raise EvaluationError("encoder_fp is invalid") from exc


def _bytes_from_hex(value: object, field_name: str) -> bytes:
    return cid_from_hex(value, field_name, error_type=EvaluationError)


__all__ = [
    "RECEIPT_REPLAY_REPORT_SCHEMA",
    "RECEIPT_REPLAY_TRACE_SCHEMA",
    "KnnReplayConfig",
    "ReceiptReplayReport",
    "ReceiptReplayTrace",
    "build_receipt_replay_trace",
    "load_replay_trace_json",
    "replay_receipt_trace",
    "write_replay_report_json",
    "write_replay_trace_json",
]
