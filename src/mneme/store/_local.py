"""Local store layout initialization and manifest-backed stats."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from mneme.core import (
    Cid,
    EncoderFingerprint,
    MemoryItem,
    QuerySpec,
    Retrieval,
    StoreCorruptionError,
    StoreError,
    TransactionError,
    ValidationError,
    build_item,
    content_id,
)
from mneme.index import FlatIndex, search_index
from mneme.store._manifest import (
    STORE_MANIFEST_SCHEMA,
    CommitmentState,
    IndexConfig,
    StoreManifest,
    ValueLogRef,
)
from mneme.store._value_log import append_value_record, read_value_records

_MANIFEST_FILE = "manifest.json"
_VALUE_LOG = "values/log-000000.mnv"
_INDEX_BACKEND_FILE = "index/backend.json"
_LAYOUT_DIRS = ("values", "index", "transactions", "receipts")


@dataclass(frozen=True)
class StoreStats:
    """Manifest-derived local store statistics."""

    store_id: UUID
    path: Path
    schema_version: str
    active_fingerprint_count: int
    value_log_count: int
    value_record_count: int
    value_bytes: int
    index_backend: str
    last_completed_transaction: str | None
    commitments_enabled: bool


@dataclass
class LocalStore:
    """Local store handle for manifest, write, and query operations."""

    path: Path
    manifest: StoreManifest
    index: FlatIndex
    _items: dict[Cid, MemoryItem]

    def stats(self) -> StoreStats:
        value_record_count = sum(log.record_count for log in self.manifest.value_logs)
        value_bytes = sum(log.size_bytes for log in self.manifest.value_logs)
        return StoreStats(
            store_id=self.manifest.store_id,
            path=self.path,
            schema_version=self.manifest.schema_version,
            active_fingerprint_count=len(self.manifest.active_fingerprints),
            value_log_count=len(self.manifest.value_logs),
            value_record_count=value_record_count,
            value_bytes=value_bytes,
            index_backend=self.manifest.index.backend,
            last_completed_transaction=self.manifest.last_completed_transaction,
            commitments_enabled=self.manifest.commitment.enabled,
        )

    def put(self, item: MemoryItem) -> Cid:
        """Append one item and return its content id."""

        return self.put_batch([item])[0]

    def put_batch(self, items: list[MemoryItem]) -> list[Cid]:
        """Append a batch of items under one transaction."""

        if not items:
            return []
        prepared = [_prepare_item(item) for item in items]
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
        written_offsets: list[tuple[int, int]] = []
        try:
            for item in prepared:
                written_offsets.append(append_value_record(log_path, item))
            self.index.add_batch(
                [(item.content_id or content_id(item), item.key) for item in prepared]
            )
            for item in prepared:
                self._items[item.content_id or content_id(item)] = item
            self.manifest = _manifest_after_commit(
                self.manifest,
                txid=txid,
                items=prepared,
                log_size=log_path.stat().st_size,
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
            raise TransactionError("failed to commit value-log transaction") from exc
        return [item.content_id or content_id(item) for item in prepared]

    def query(self, spec: QuerySpec) -> Retrieval:
        """Query the in-memory index and load values from the value log cache."""

        index_fp = (
            self.manifest.active_fingerprints[0]
            if len(self.manifest.active_fingerprints) == 1
            else None
        )
        results = search_index(self.index, spec, index_fingerprint=index_fp)
        try:
            items = tuple(self._items[cid] for cid, _ in results)
        except KeyError as exc:
            raise StoreCorruptionError(
                "index contains id missing from value log"
            ) from exc
        return Retrieval(
            items=items,
            distances=tuple(distance for _, distance in results),
        )


def init_store(
    path: str | Path,
    *,
    active_fingerprints: list[EncoderFingerprint] | None = None,
    index_backend: str = "flat",
    index_params: dict[str, Any] | None = None,
    retention_policy: dict[str, Any] | None = None,
    store_id: UUID | None = None,
    exist_ok: bool = False,
) -> LocalStore:
    """Create a local store layout and schema-versioned manifest."""

    root = Path(path)
    manifest_path = root / _MANIFEST_FILE
    if manifest_path.exists() and not exist_ok:
        raise StoreError(f"store manifest already exists at {manifest_path}")

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
        index=IndexConfig(index_backend, index_params or {}),
        retention_policy=retention_policy or {"policy": "none", "tombstones": []},
        last_completed_transaction=None,
        commitment=CommitmentState(enabled=False, backend=None, root=None, files=()),
    )
    _write_json_atomic(manifest_path, manifest.to_json())
    _write_json_atomic(
        root / _INDEX_BACKEND_FILE,
        {"backend": manifest.index.backend, "params": dict(manifest.index.params)},
    )
    return _store_from_manifest(root, manifest)


def open_store(path: str | Path, *, create: bool = False) -> LocalStore:
    """Open a local store, optionally initializing it when absent."""

    root = Path(path)
    manifest_path = root / _MANIFEST_FILE
    if not manifest_path.exists():
        if create:
            return init_store(root)
        raise StoreError(f"store manifest not found at {manifest_path}")
    manifest = load_manifest(root)
    _detect_interrupted_transactions(root)
    return _store_from_manifest(root, manifest)


def load_manifest(path: str | Path) -> StoreManifest:
    """Load and validate manifest.json from a store path or manifest path."""

    candidate = Path(path)
    manifest_path = (
        candidate if candidate.name == _MANIFEST_FILE else candidate / _MANIFEST_FILE
    )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StoreError(f"store manifest not found at {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise StoreCorruptionError(
            f"store manifest is malformed JSON: {manifest_path}"
        ) from exc
    return StoreManifest.from_json(raw)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _store_from_manifest(root: Path, manifest: StoreManifest) -> LocalStore:
    index = FlatIndex()
    items: dict[Cid, MemoryItem] = {}
    for value_log in manifest.value_logs:
        log_path = root / value_log.path
        if not log_path.exists():
            raise StoreCorruptionError(f"value log missing: {log_path}")
        for item in read_value_records(log_path):
            cid = item.content_id or content_id(item)
            items[cid] = item
            index.add(cid, item.key)
    return LocalStore(path=root, manifest=manifest, index=index, _items=items)


def _prepare_item(item: MemoryItem) -> MemoryItem:
    prepared = build_item(item.value, item.key, item.encoder_fp, item.meta)
    if item.content_id is not None and item.content_id != prepared.content_id:
        raise ValidationError("content_id does not match canonical item bytes")
    return prepared


def _manifest_after_commit(
    manifest: StoreManifest,
    *,
    txid: str,
    items: list[MemoryItem],
    log_size: int,
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
    return replace(
        manifest,
        updated_at=_utc_now(),
        active_fingerprints=tuple(active_fingerprints),
        value_logs=(updated_log, *manifest.value_logs[1:]),
        last_completed_transaction=txid,
    )


def _write_transaction(root: Path, txid: str, data: dict[str, Any]) -> None:
    _write_json_atomic(root / "transactions" / f"{txid}.json", data)


def _detect_interrupted_transactions(root: Path) -> None:
    transactions = root / "transactions"
    if not transactions.exists():
        return
    for path in transactions.glob("txn-*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TransactionError(f"transaction file is malformed: {path}") from exc
        if data.get("state") != "committed":
            raise TransactionError(f"interrupted transaction detected: {path.name}")


__all__ = ["LocalStore", "StoreStats", "init_store", "load_manifest", "open_store"]
