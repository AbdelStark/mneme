"""Local store layout initialization and manifest-backed stats."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

from mneme.core import (
    Cid,
    EncoderFingerprint,
    MemoryItem,
    QuerySpec,
    ReceiptVerificationError,
    Retrieval,
    StoreCorruptionError,
    StoreError,
    TransactionError,
    UnsupportedOperationError,
    ValidationError,
    build_item,
    content_id,
)
from mneme.core._ids import require_cid_bytes
from mneme.core._json import dumps_strict_json, loads_strict_json
from mneme.index import Index, create_index_backend, search_index
from mneme.observability import (
    ObservabilityConfig,
    content_id_prefix,
    distance_mean,
    distance_min,
    emit_event,
    start_event_timer,
)
from mneme.receipts import (
    CommitmentState as MmrCommitmentState,
)
from mneme.receipts import (
    InclusionProof,
    build_retrieval_receipt,
    load_commitment_state,
    save_commitment_state,
)
from mneme.store._manifest import (
    STORE_MANIFEST_SCHEMA,
    IndexConfig,
    StoreManifest,
    ValueLogRef,
)
from mneme.store._manifest import (
    CommitmentState as ManifestCommitmentState,
)
from mneme.store._retention import (
    apply_retention_policy,
    tombstoned_cids,
    visible_cids,
)
from mneme.store._value_log import (
    append_value_record,
    read_value_records,
    read_value_records_with_offsets,
)

_MANIFEST_FILE = "manifest.json"
_VALUE_LOG = "values/log-000000.mnv"
_INDEX_BACKEND_FILE = "index/backend.json"
_COMMITMENT_FILE = "receipts/commitment-mmr-v1.json"
_LAYOUT_DIRS = ("values", "index", "transactions", "receipts")
_TRANSACTION_SCHEMA: Final = "mneme.transaction.v1"
_STORE_RECOVERY_EVENT_SCHEMA: Final = "mneme.store_recovery_event.v1"


@dataclass(frozen=True)
class StoreStats:
    """Manifest-derived local store statistics."""

    store_id: UUID
    path: Path
    schema_version: str
    active_fingerprint_count: int
    value_log_count: int
    value_record_count: int
    visible_record_count: int
    value_bytes: int
    index_backend: str
    retention_policy: str
    tombstone_count: int
    last_completed_transaction: str | None
    commitments_enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.store_id, UUID):
            raise ValidationError("store_id must be a UUID")
        object.__setattr__(self, "path", _require_stats_path(self.path))
        _require_non_empty_stats_string(self.schema_version, "schema_version")
        object.__setattr__(
            self,
            "active_fingerprint_count",
            _require_non_negative_stats_int(
                self.active_fingerprint_count,
                "active_fingerprint_count",
            ),
        )
        object.__setattr__(
            self,
            "value_log_count",
            _require_non_negative_stats_int(self.value_log_count, "value_log_count"),
        )
        object.__setattr__(
            self,
            "value_record_count",
            _require_non_negative_stats_int(
                self.value_record_count,
                "value_record_count",
            ),
        )
        object.__setattr__(
            self,
            "visible_record_count",
            _require_non_negative_stats_int(
                self.visible_record_count,
                "visible_record_count",
            ),
        )
        object.__setattr__(
            self,
            "value_bytes",
            _require_non_negative_stats_int(self.value_bytes, "value_bytes"),
        )
        object.__setattr__(
            self,
            "index_backend",
            _require_non_empty_stats_string(self.index_backend, "index_backend"),
        )
        object.__setattr__(
            self,
            "retention_policy",
            _require_non_empty_stats_string(
                self.retention_policy,
                "retention_policy",
            ),
        )
        object.__setattr__(
            self,
            "tombstone_count",
            _require_non_negative_stats_int(self.tombstone_count, "tombstone_count"),
        )
        object.__setattr__(
            self,
            "last_completed_transaction",
            _optional_non_empty_stats_string(
                self.last_completed_transaction,
                "last_completed_transaction",
            ),
        )
        object.__setattr__(
            self,
            "commitments_enabled",
            _require_stats_bool(self.commitments_enabled, "commitments_enabled"),
        )


@dataclass(frozen=True)
class StoreRecoveryEvent:
    """Structured event emitted when open-store recovery changes state."""

    event: str
    store_id: str
    operation: str
    status: str
    transaction_id: str
    value_log: str
    previous_size_bytes: int
    recovered_size_bytes: int
    previous_record_count: int
    recovered_record_count: int
    duration_ms: float
    schema_version: str = _STORE_RECOVERY_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != _STORE_RECOVERY_EVENT_SCHEMA:
            raise ValidationError("unsupported store recovery event schema")
        object.__setattr__(
            self, "event", _require_non_empty_stats_string(self.event, "event")
        )
        object.__setattr__(
            self,
            "store_id",
            _require_non_empty_stats_string(self.store_id, "store_id"),
        )
        object.__setattr__(
            self,
            "operation",
            _require_non_empty_stats_string(self.operation, "operation"),
        )
        object.__setattr__(
            self, "status", _require_non_empty_stats_string(self.status, "status")
        )
        object.__setattr__(
            self,
            "transaction_id",
            _require_non_empty_stats_string(self.transaction_id, "transaction_id"),
        )
        object.__setattr__(
            self,
            "value_log",
            _require_non_empty_stats_string(self.value_log, "value_log"),
        )
        object.__setattr__(
            self,
            "previous_size_bytes",
            _require_non_negative_stats_int(
                self.previous_size_bytes,
                "previous_size_bytes",
            ),
        )
        object.__setattr__(
            self,
            "recovered_size_bytes",
            _require_non_negative_stats_int(
                self.recovered_size_bytes,
                "recovered_size_bytes",
            ),
        )
        object.__setattr__(
            self,
            "previous_record_count",
            _require_non_negative_stats_int(
                self.previous_record_count,
                "previous_record_count",
            ),
        )
        object.__setattr__(
            self,
            "recovered_record_count",
            _require_non_negative_stats_int(
                self.recovered_record_count,
                "recovered_record_count",
            ),
        )
        object.__setattr__(
            self,
            "duration_ms",
            _require_non_negative_stats_float(self.duration_ms, "duration_ms"),
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable recovery event."""

        return {
            "event": self.event,
            "schema_version": self.schema_version,
            "store_id": self.store_id,
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "transaction_id": self.transaction_id,
            "value_log": self.value_log,
            "previous_size_bytes": self.previous_size_bytes,
            "recovered_size_bytes": self.recovered_size_bytes,
            "previous_record_count": self.previous_record_count,
            "recovered_record_count": self.recovered_record_count,
        }


def _require_stats_path(value: object) -> Path:
    if isinstance(value, str) and not value:
        raise ValidationError("path must not be empty")
    if isinstance(value, str | Path):
        path = Path(value)
    else:
        raise ValidationError("path must be a path-like value")
    if not str(path):
        raise ValidationError("path must not be empty")
    return path


def _require_stats_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a bool")
    return value


def _require_non_negative_stats_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{field_name} must be a non-negative integer")
    return value


def _require_non_negative_stats_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field_name} must be a finite non-negative number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValidationError(f"{field_name} must be a finite non-negative number")
    return numeric


def _require_non_empty_stats_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_non_empty_stats_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_stats_string(value, field_name)


def _require_local_store_path(value: object) -> Path:
    if isinstance(value, str) and not value:
        raise ValidationError("path must not be empty")
    if isinstance(value, str | Path):
        path = Path(value)
    else:
        raise ValidationError("path must be a path-like value")
    if not str(path):
        raise ValidationError("path must not be empty")
    return path


def _require_local_store_items(value: object) -> dict[Cid, MemoryItem]:
    if not isinstance(value, Mapping):
        raise ValidationError("_items must be a mapping")
    items: dict[Cid, MemoryItem] = {}
    for cid, item in value.items():
        valid_cid = require_cid_bytes(
            cid,
            "_items keys",
            type_error=ValidationError,
            value_error=ValidationError,
        )
        if not isinstance(item, MemoryItem):
            raise ValidationError("_items values must be MemoryItem")
        if valid_cid != (item.content_id or content_id(item)):
            raise ValidationError("_items keys must match item content ids")
        items[valid_cid] = item
    return items


def _require_local_store_recovery_events(
    value: object,
) -> tuple[StoreRecoveryEvent, ...]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValidationError("recovery_events must be a sequence")
    events = tuple(value)
    for event in events:
        if not isinstance(event, StoreRecoveryEvent):
            raise ValidationError("recovery_events items must be StoreRecoveryEvent")
    return events


@dataclass
class LocalStore:
    """Local store handle for manifest, write, and query operations."""

    path: Path
    manifest: StoreManifest
    index: Index
    _items: dict[Cid, MemoryItem]
    recovery_events: tuple[StoreRecoveryEvent, ...] = ()
    observability: ObservabilityConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _require_local_store_path(self.path))
        if not isinstance(self.manifest, StoreManifest):
            raise ValidationError("manifest must be a StoreManifest")
        if not isinstance(self.index, Index):
            raise ValidationError("index must implement Index")
        object.__setattr__(
            self,
            "_items",
            _require_local_store_items(self._items),
        )
        object.__setattr__(
            self,
            "recovery_events",
            _require_local_store_recovery_events(self.recovery_events),
        )
        if self.observability is not None and not isinstance(
            self.observability,
            ObservabilityConfig,
        ):
            raise ValidationError("observability must be an ObservabilityConfig")
        if len(self.index) != len(visible_cids(self.manifest, self._items)):
            raise ValidationError("index size must match visible item count")

    def stats(self) -> StoreStats:
        value_record_count = sum(log.record_count for log in self.manifest.value_logs)
        value_bytes = sum(log.size_bytes for log in self.manifest.value_logs)
        tombstoned = tombstoned_cids(self.manifest)
        return StoreStats(
            store_id=self.manifest.store_id,
            path=self.path,
            schema_version=self.manifest.schema_version,
            active_fingerprint_count=len(self.manifest.active_fingerprints),
            value_log_count=len(self.manifest.value_logs),
            value_record_count=value_record_count,
            visible_record_count=len(set(self._items) - tombstoned),
            value_bytes=value_bytes,
            index_backend=self.manifest.index.backend,
            retention_policy=str(self.manifest.retention_policy.get("policy", "none")),
            tombstone_count=len(tombstoned),
            last_completed_transaction=self.manifest.last_completed_transaction,
            commitments_enabled=self.manifest.commitment.enabled,
        )

    def put(self, item: MemoryItem) -> Cid:
        """Append one item and return its content id."""

        return self.put_batch([item])[0]

    def put_batch(self, items: list[MemoryItem]) -> list[Cid]:
        """Append a batch of items under one transaction."""

        started = start_event_timer(self.observability)
        if not items:
            if started is not None:
                stats = self.stats()
                emit_event(
                    self.observability,
                    event="mneme.store.put",
                    operation="store.put",
                    status="ok",
                    started=started,
                    store_id=str(self.manifest.store_id),
                    backend=self.manifest.index.backend,
                    item_count=0,
                    value_record_count=stats.value_record_count,
                    value_bytes=stats.value_bytes,
                )
            return []
        txid: str | None = None
        try:
            prepared = [_prepare_item(item) for item in items]
        except Exception as exc:
            if started is not None:
                emit_event(
                    self.observability,
                    event="mneme.store.put",
                    operation="store.put",
                    status="error",
                    started=started,
                    error=exc,
                    store_id=str(self.manifest.store_id),
                    backend=self.manifest.index.backend,
                    item_count=len(items),
                )
            raise
        txid = f"txn-{uuid4().hex}"
        log_path = self.path / self.manifest.value_logs[0].path
        previous_size = log_path.stat().st_size
        previous_count = self.manifest.value_logs[0].record_count
        _write_transaction(
            self.path,
            txid,
            {
                "schema_version": "mneme.transaction.v1",
                "transaction_id": txid,
                "state": "intent",
                "item_count": len(prepared),
                "value_log": self.manifest.value_logs[0].path,
                "previous_size_bytes": previous_size,
                "previous_record_count": previous_count,
            },
        )
        try:
            written_offsets: list[tuple[int, int]] = []
            for item in prepared:
                written_offsets.append(append_value_record(log_path, item))
            self.index.add_batch(
                [(item.content_id or content_id(item), item.key) for item in prepared]
            )
            for item in prepared:
                self._items[item.content_id or content_id(item)] = item
            previous_tombstones = tombstoned_cids(self.manifest)
            self.manifest = _manifest_after_commit(
                self.manifest,
                txid=txid,
                items=prepared,
                log_size=log_path.stat().st_size,
                all_items=self._items,
            )
            if tombstoned_cids(self.manifest) != previous_tombstones:
                self.index = _index_from_items(
                    self.manifest,
                    self._items,
                    observability=self.observability,
                )
            _write_json_atomic(self.path / _MANIFEST_FILE, self.manifest.to_json())
            _write_transaction(
                self.path,
                txid,
                {
                    "schema_version": "mneme.transaction.v1",
                    "transaction_id": txid,
                    "state": "committed",
                    "item_count": len(prepared),
                    "value_log": self.manifest.value_logs[0].path,
                    "previous_size_bytes": previous_size,
                    "previous_record_count": previous_count,
                    "written_offsets": [
                        {"start": start, "end": end} for start, end in written_offsets
                    ],
                },
            )
        except Exception as exc:
            if started is not None:
                emit_event(
                    self.observability,
                    event="mneme.store.put",
                    operation="store.put",
                    status="error",
                    started=started,
                    error=exc,
                    store_id=str(self.manifest.store_id),
                    backend=self.manifest.index.backend,
                    transaction_id=txid,
                    item_count=len(items),
                )
            raise TransactionError("failed to commit value-log transaction") from exc
        cids = [item.content_id or content_id(item) for item in prepared]
        if started is not None:
            stats = self.stats()
            content_id_prefixes = _content_id_prefixes(cids, self.observability)
            emit_event(
                self.observability,
                event="mneme.store.commit",
                operation="store.commit",
                status="ok",
                started=started,
                store_id=str(self.manifest.store_id),
                backend=self.manifest.index.backend,
                transaction_id=txid,
                item_count=len(cids),
                value_record_count=stats.value_record_count,
                value_bytes=stats.value_bytes,
                content_id_prefixes=content_id_prefixes,
            )
            emit_event(
                self.observability,
                event="mneme.store.put",
                operation="store.put",
                status="ok",
                started=started,
                store_id=str(self.manifest.store_id),
                backend=self.manifest.index.backend,
                transaction_id=txid,
                item_count=len(cids),
                value_record_count=stats.value_record_count,
                value_bytes=stats.value_bytes,
                content_id_prefixes=content_id_prefixes,
            )
        return cids

    def query(self, spec: QuerySpec) -> Retrieval:
        """Query the in-memory index and load values from the value log cache."""

        started = start_event_timer(self.observability)
        commitment_state = self.commitment_state() if spec.with_receipt else None
        index_fp = (
            self.manifest.active_fingerprints[0]
            if len(self.manifest.active_fingerprints) == 1
            else None
        )
        try:
            results = search_index(self.index, spec, index_fingerprint=index_fp)
            items = tuple(self._items[cid] for cid, _ in results)
        except KeyError as exc:
            error = StoreCorruptionError("index contains id missing from value log")
            if started is not None:
                emit_event(
                    self.observability,
                    event="mneme.store.query",
                    operation="store.query",
                    status="error",
                    started=started,
                    error=error,
                    store_id=str(self.manifest.store_id),
                    backend=self.manifest.index.backend,
                    k=spec.k,
                    fingerprint_match=_fingerprint_match(spec, index_fp),
                )
            raise error from exc
        except Exception as exc:
            if started is not None:
                emit_event(
                    self.observability,
                    event="mneme.store.query",
                    operation="store.query",
                    status="error",
                    started=started,
                    error=exc,
                    store_id=str(self.manifest.store_id),
                    backend=self.manifest.index.backend,
                    k=spec.k,
                    fingerprint_match=_fingerprint_match(spec, index_fp),
                )
            raise
        if started is not None:
            distances = tuple(distance for _, distance in results)
            emit_event(
                self.observability,
                event="mneme.store.query",
                operation="store.query",
                status="ok",
                started=started,
                store_id=str(self.manifest.store_id),
                backend=self.manifest.index.backend,
                k=spec.k,
                hit_count=len(results),
                distance_min=distance_min(distances),
                distance_mean=distance_mean(distances),
                fingerprint_match=_fingerprint_match(spec, index_fp),
                content_id_prefixes=_content_id_prefixes(
                    [cid for cid, _ in results],
                    self.observability,
                ),
            )
        receipt = None
        if commitment_state is not None:
            result_ids = tuple(cid for cid, _ in results)
            proofs = tuple(commitment_state.prove(cid) for cid in result_ids)
            receipt = build_retrieval_receipt(
                root=commitment_state.root,
                ids=result_ids,
                proofs=proofs,
                query=spec,
                store_id=str(self.manifest.store_id),
            )
        return Retrieval(
            items=items,
            distances=tuple(distance for _, distance in results),
            receipt=receipt,
        )

    def commit(self) -> bytes:
        """Commit current value-log append order and return the MMR root."""

        started = start_event_timer(self.observability)
        try:
            state = MmrCommitmentState.from_cids(
                _content_ids_in_value_log_order(self.path, self.manifest)
            )
            save_commitment_state(self.path / _COMMITMENT_FILE, state)
            self.manifest = replace(
                self.manifest,
                updated_at=_utc_now(),
                commitment=ManifestCommitmentState(
                    enabled=True,
                    backend=state.scheme,
                    root=state.root_hex,
                    files=(_COMMITMENT_FILE,),
                ),
            )
            _write_json_atomic(self.path / _MANIFEST_FILE, self.manifest.to_json())
        except Exception as exc:
            if started is not None:
                emit_event(
                    self.observability,
                    event="mneme.store.commit",
                    operation="store.commit",
                    status="error",
                    started=started,
                    error=exc,
                    store_id=str(self.manifest.store_id),
                    backend=self.manifest.index.backend,
                )
            raise
        if started is not None:
            emit_event(
                self.observability,
                event="mneme.store.commit",
                operation="store.commit",
                status="ok",
                started=started,
                store_id=str(self.manifest.store_id),
                backend=self.manifest.index.backend,
                commitment_backend=state.scheme,
                commitment_root=state.root_hex,
                item_count=state.item_count,
            )
        return state.root

    def root(self) -> bytes:
        """Return the last persisted commitment root."""

        root_hex = self.manifest.commitment.root
        if not self.manifest.commitment.enabled or root_hex is None:
            raise UnsupportedOperationError("store commitments are not initialized")
        try:
            root = bytes.fromhex(root_hex)
        except ValueError as exc:
            raise StoreCorruptionError("commitment root must be hex bytes") from exc
        if len(root) != 32:
            raise StoreCorruptionError("commitment root must be 32 bytes")
        return root

    def commitment_state(self) -> MmrCommitmentState:
        """Load the persisted commitment state sidecar."""

        if not self.manifest.commitment.enabled:
            raise UnsupportedOperationError("store commitments are not initialized")
        if _COMMITMENT_FILE not in self.manifest.commitment.files:
            raise StoreCorruptionError("commitment sidecar is not referenced")
        state = load_commitment_state(self.path / _COMMITMENT_FILE)
        root_hex = self.manifest.commitment.root
        if root_hex is None or state.root_hex != root_hex:
            raise ReceiptVerificationError("commitment sidecar root mismatch")
        return state

    def prove(self, ids: Sequence[Cid]) -> list[InclusionProof]:
        """Return inclusion proofs for committed content ids."""

        if isinstance(ids, bytes) or not isinstance(ids, Sequence):
            raise ValidationError("ids must be a sequence of content ids")
        state = self.commitment_state()
        return [state.prove(cid) for cid in ids]


def init_store(
    path: str | Path,
    *,
    active_fingerprints: list[EncoderFingerprint] | None = None,
    index_backend: str = "flat",
    index_params: dict[str, Any] | None = None,
    retention_policy: dict[str, Any] | None = None,
    store_id: UUID | None = None,
    exist_ok: bool = False,
    observability: ObservabilityConfig | None = None,
) -> LocalStore:
    """Create a local store layout and schema-versioned manifest."""

    root = Path(path)
    manifest_path = root / _MANIFEST_FILE
    if manifest_path.exists() and not exist_ok:
        raise StoreError(f"store manifest already exists at {manifest_path}")
    index_config = IndexConfig(index_backend, index_params or {})
    create_index_backend(
        index_config.backend,
        index_config.params,
        observability=observability,
    )

    root.mkdir(parents=True, exist_ok=True)
    for dirname in _LAYOUT_DIRS:
        (root / dirname).mkdir(exist_ok=True)
    (root / _VALUE_LOG).touch(exist_ok=True)

    now = _utc_now()
    manifest = StoreManifest(
        schema_version=STORE_MANIFEST_SCHEMA,
        store_id=uuid4() if store_id is None else store_id,
        created_at=now,
        updated_at=now,
        active_fingerprints=tuple(active_fingerprints or ()),
        value_logs=(ValueLogRef(_VALUE_LOG),),
        index=index_config,
        retention_policy=retention_policy or {"policy": "none", "tombstones": []},
        last_completed_transaction=None,
        commitment=ManifestCommitmentState(
            enabled=False,
            backend=None,
            root=None,
            files=(),
        ),
    )
    _write_json_atomic(manifest_path, manifest.to_json())
    _write_json_atomic(
        root / _INDEX_BACKEND_FILE,
        {"backend": manifest.index.backend, "params": dict(manifest.index.params)},
    )
    return _store_from_manifest(root, manifest, observability=observability)


def open_store(
    path: str | Path,
    *,
    create: bool = False,
    observability: ObservabilityConfig | None = None,
) -> LocalStore:
    """Open a local store, optionally initializing it when absent."""

    root = Path(path)
    manifest_path = root / _MANIFEST_FILE
    if not manifest_path.exists():
        if create:
            return init_store(root, observability=observability)
        raise StoreError(f"store manifest not found at {manifest_path}")
    manifest = load_manifest(root)
    manifest, recovery_events = _recover_interrupted_transactions(root, manifest)
    return _store_from_manifest(
        root,
        manifest,
        recovery_events=recovery_events,
        observability=observability,
    )


def load_manifest(path: str | Path) -> StoreManifest:
    """Load and validate manifest.json from a store path or manifest path."""

    candidate = Path(path)
    manifest_path = (
        candidate if candidate.name == _MANIFEST_FILE else candidate / _MANIFEST_FILE
    )
    try:
        raw = loads_strict_json(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StoreError(f"store manifest not found at {manifest_path}") from exc
    except OSError as exc:
        raise StoreCorruptionError(
            f"store manifest could not be read: {manifest_path}"
        ) from exc
    except ValueError as exc:
        raise StoreCorruptionError(
            f"store manifest is malformed JSON: {manifest_path}"
        ) from exc
    return StoreManifest.from_json(raw)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        dumps_strict_json(data, sort_keys=True, indent=2, separators=(",", ": "))
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _store_from_manifest(
    root: Path,
    manifest: StoreManifest,
    *,
    recovery_events: tuple[StoreRecoveryEvent, ...] = (),
    observability: ObservabilityConfig | None = None,
) -> LocalStore:
    items: dict[Cid, MemoryItem] = {}
    for value_log in manifest.value_logs:
        log_path = root / value_log.path
        if not log_path.exists():
            raise StoreCorruptionError(f"value log missing: {log_path}")
        for item in read_value_records(log_path):
            cid = item.content_id or content_id(item)
            items[cid] = item
    index = _index_from_items(manifest, items, observability=observability)
    return LocalStore(
        path=root,
        manifest=manifest,
        index=index,
        _items=items,
        recovery_events=recovery_events,
        observability=observability,
    )


def _prepare_item(item: MemoryItem) -> MemoryItem:
    prepared = build_item(item.value, item.key, item.encoder_fp, item.meta)
    if item.content_id is not None and item.content_id != prepared.content_id:
        raise ValidationError("content_id does not match canonical item bytes")
    return prepared


def _content_id_prefixes(
    cids: list[Cid],
    observability: ObservabilityConfig | None,
) -> list[str]:
    prefixes = [
        prefix
        for cid in cids
        if (prefix := content_id_prefix(cid, observability)) is not None
    ]
    return prefixes


def _fingerprint_match(
    spec: QuerySpec,
    index_fingerprint: EncoderFingerprint | None,
) -> bool | None:
    if spec.encoder_fp is None or index_fingerprint is None:
        return None
    return spec.encoder_fp == index_fingerprint


def _manifest_after_commit(
    manifest: StoreManifest,
    *,
    txid: str,
    items: list[MemoryItem],
    log_size: int,
    all_items: Mapping[Cid, MemoryItem],
) -> StoreManifest:
    current_log = manifest.value_logs[0]
    updated_log = replace(
        current_log,
        size_bytes=log_size,
        record_count=current_log.record_count + len(items),
    )
    active_fingerprints = list(manifest.active_fingerprints)
    for item in items:
        if item.encoder_fp not in active_fingerprints:
            active_fingerprints.append(item.encoder_fp)
    updated = replace(
        manifest,
        updated_at=_utc_now(),
        active_fingerprints=tuple(active_fingerprints),
        value_logs=(updated_log, *manifest.value_logs[1:]),
        last_completed_transaction=txid,
    )
    return replace(
        updated,
        retention_policy=apply_retention_policy(
            updated.retention_policy,
            all_items,
            transaction_id=txid,
        ),
    )


def _index_from_items(
    manifest: StoreManifest,
    items: Mapping[Cid, MemoryItem],
    *,
    observability: ObservabilityConfig | None,
) -> Index:
    index = create_index_backend(
        manifest.index.backend,
        manifest.index.params,
        observability=observability,
    )
    for cid in sorted(visible_cids(manifest, items)):
        index.add(cid, items[cid].key)
    return index


def _write_transaction(root: Path, txid: str, data: dict[str, Any]) -> None:
    _write_json_atomic(root / "transactions" / f"{txid}.json", data)


@dataclass(frozen=True)
class _PendingTransaction:
    path: Path
    transaction_id: str
    item_count: int
    value_log: str
    previous_size_bytes: int
    previous_record_count: int


def _recover_interrupted_transactions(
    root: Path,
    manifest: StoreManifest,
) -> tuple[StoreManifest, tuple[StoreRecoveryEvent, ...]]:
    transactions = root / "transactions"
    if not transactions.exists():
        return manifest, ()
    current = manifest
    events: list[StoreRecoveryEvent] = []
    for path in sorted(transactions.glob("txn-*.json")):
        pending = _load_pending_transaction(path)
        if pending is None:
            continue
        current, event = _recover_pending_transaction(root, current, pending)
        events.append(event)
    return current, tuple(events)


def _load_pending_transaction(path: Path) -> _PendingTransaction | None:
    try:
        data = loads_strict_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StoreCorruptionError(
            f"transaction file could not be read: {path.name}"
        ) from exc
    except ValueError as exc:
        raise StoreCorruptionError(
            f"transaction file is malformed JSON: {path.name}"
        ) from exc
    if not isinstance(data, Mapping):
        raise StoreCorruptionError(f"transaction file must be an object: {path.name}")
    schema_version = data.get("schema_version")
    if schema_version != _TRANSACTION_SCHEMA:
        raise StoreCorruptionError("unsupported transaction schema")
    state = _require_transaction_string(data.get("state"), "transaction state")
    if state in {"committed", "rolled_back"}:
        return None
    if state != "intent":
        raise StoreCorruptionError(f"unsupported transaction state: {state}")
    transaction_id = _require_transaction_string(
        data.get("transaction_id"),
        "transaction_id",
    )
    if transaction_id != path.stem:
        raise StoreCorruptionError("transaction_id does not match file name")
    return _PendingTransaction(
        path=path,
        transaction_id=transaction_id,
        item_count=_require_positive_transaction_int(
            data.get("item_count"), "item_count"
        ),
        value_log=_require_transaction_string(data.get("value_log"), "value_log"),
        previous_size_bytes=_require_non_negative_transaction_int(
            data.get("previous_size_bytes"),
            "previous_size_bytes",
        ),
        previous_record_count=_require_non_negative_transaction_int(
            data.get("previous_record_count"),
            "previous_record_count",
        ),
    )


def _recover_pending_transaction(
    root: Path,
    manifest: StoreManifest,
    transaction: _PendingTransaction,
) -> tuple[StoreManifest, StoreRecoveryEvent]:
    started = time.perf_counter()
    if transaction.value_log != manifest.value_logs[0].path:
        raise StoreCorruptionError("pending transaction targets unknown value log")
    log_path = root / transaction.value_log
    if not log_path.exists():
        raise StoreCorruptionError(f"value log missing: {transaction.value_log}")
    actual_size = log_path.stat().st_size
    if actual_size < transaction.previous_size_bytes:
        raise StoreCorruptionError("value log is shorter than transaction offset")

    appended: list[tuple[MemoryItem, int, int]] = []
    read_error: StoreCorruptionError | None = None
    try:
        appended = list(
            read_value_records_with_offsets(
                log_path,
                start_offset=transaction.previous_size_bytes,
            )
        )
    except StoreCorruptionError as exc:
        read_error = exc

    manifest_log = manifest.value_logs[0]
    manifest_already_completed = (
        manifest.last_completed_transaction == transaction.transaction_id
        and manifest_log.size_bytes == actual_size
        and manifest_log.record_count
        == transaction.previous_record_count + transaction.item_count
    )
    if (
        manifest.last_completed_transaction == transaction.transaction_id
        and not manifest_already_completed
    ):
        raise StoreCorruptionError(
            "manifest completed transaction but offsets do not match"
        )
    if manifest_already_completed and (
        read_error is not None or len(appended) != transaction.item_count
    ):
        raise StoreCorruptionError("completed transaction value log is invalid")
    if (
        read_error is None
        and len(appended) == transaction.item_count
        and (
            actual_size > transaction.previous_size_bytes or manifest_already_completed
        )
    ):
        completed = _complete_recovered_transaction(
            root,
            manifest,
            transaction,
            appended,
            actual_size,
        )
        return completed, _recovery_event(
            manifest=completed,
            transaction=transaction,
            status="completed",
            recovered_size_bytes=actual_size,
            recovered_record_count=transaction.item_count,
            started=started,
        )
    if read_error is None and len(appended) > transaction.item_count:
        raise StoreCorruptionError("transaction appended more records than declared")
    _truncate_value_log(log_path, transaction.previous_size_bytes)
    _write_transaction_state(
        transaction.path,
        transaction,
        state="rolled_back",
        written_offsets=[],
    )
    return manifest, _recovery_event(
        manifest=manifest,
        transaction=transaction,
        status="rolled_back",
        recovered_size_bytes=transaction.previous_size_bytes,
        recovered_record_count=0,
        started=started,
    )


def _complete_recovered_transaction(
    root: Path,
    manifest: StoreManifest,
    transaction: _PendingTransaction,
    appended: list[tuple[MemoryItem, int, int]],
    actual_size: int,
) -> StoreManifest:
    if manifest.last_completed_transaction == transaction.transaction_id:
        completed = manifest
    else:
        manifest_log = manifest.value_logs[0]
        if manifest_log.size_bytes != transaction.previous_size_bytes:
            raise StoreCorruptionError(
                "manifest size does not match transaction offset"
            )
        if manifest_log.record_count != transaction.previous_record_count:
            raise StoreCorruptionError(
                "manifest record count does not match transaction offset"
            )
        completed = _manifest_after_commit(
            manifest,
            txid=transaction.transaction_id,
            items=[item for item, _, _ in appended],
            log_size=actual_size,
            all_items=_items_from_value_log(root / transaction.value_log),
        )
        _write_json_atomic(root / _MANIFEST_FILE, completed.to_json())
    _write_transaction_state(
        transaction.path,
        transaction,
        state="committed",
        written_offsets=[(start, end) for _, start, end in appended],
    )
    return completed


def _items_from_value_log(path: Path) -> dict[Cid, MemoryItem]:
    items: dict[Cid, MemoryItem] = {}
    for item in read_value_records(path):
        items[item.content_id or content_id(item)] = item
    return items


def _content_ids_in_value_log_order(
    root: Path,
    manifest: StoreManifest,
) -> tuple[Cid, ...]:
    cids: list[Cid] = []
    for value_log in manifest.value_logs:
        for item in read_value_records(root / value_log.path):
            cids.append(item.content_id or content_id(item))
    return tuple(cids)


def _truncate_value_log(path: Path, size: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(size)
        handle.flush()
        try:
            import os

            os.fsync(handle.fileno())
        except OSError:
            pass


def _write_transaction_state(
    path: Path,
    transaction: _PendingTransaction,
    *,
    state: str,
    written_offsets: list[tuple[int, int]],
) -> None:
    data: dict[str, Any] = {
        "schema_version": _TRANSACTION_SCHEMA,
        "transaction_id": transaction.transaction_id,
        "state": state,
        "item_count": transaction.item_count,
        "value_log": transaction.value_log,
        "previous_size_bytes": transaction.previous_size_bytes,
        "previous_record_count": transaction.previous_record_count,
    }
    if written_offsets:
        data["written_offsets"] = [
            {"start": start, "end": end} for start, end in written_offsets
        ]
    _write_json_atomic(path, data)


def _recovery_event(
    *,
    manifest: StoreManifest,
    transaction: _PendingTransaction,
    status: str,
    recovered_size_bytes: int,
    recovered_record_count: int,
    started: float,
) -> StoreRecoveryEvent:
    return StoreRecoveryEvent(
        event="mneme.store.recover",
        store_id=str(manifest.store_id),
        operation="store.recover",
        status=status,
        transaction_id=transaction.transaction_id,
        value_log=transaction.value_log,
        previous_size_bytes=transaction.previous_size_bytes,
        recovered_size_bytes=recovered_size_bytes,
        previous_record_count=transaction.previous_record_count,
        recovered_record_count=recovered_record_count,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )


def _require_transaction_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StoreCorruptionError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_transaction_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StoreCorruptionError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_transaction_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StoreCorruptionError(f"{field_name} must be a positive integer")
    return value


__all__ = [
    "LocalStore",
    "StoreRecoveryEvent",
    "StoreStats",
    "init_store",
    "load_manifest",
    "open_store",
]
