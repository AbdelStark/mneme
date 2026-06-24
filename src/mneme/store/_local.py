"""Local store layout initialization and manifest-backed stats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from mneme.core import EncoderFingerprint, StoreCorruptionError, StoreError
from mneme.store._manifest import (
    STORE_MANIFEST_SCHEMA,
    CommitmentState,
    IndexConfig,
    StoreManifest,
    ValueLogRef,
)

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


@dataclass(frozen=True)
class LocalStore:
    """Local store handle for manifest/layout operations."""

    path: Path
    manifest: StoreManifest

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
    return LocalStore(path=root, manifest=manifest)


def open_store(path: str | Path, *, create: bool = False) -> LocalStore:
    """Open a local store, optionally initializing it when absent."""

    root = Path(path)
    manifest_path = root / _MANIFEST_FILE
    if not manifest_path.exists():
        if create:
            return init_store(root)
        raise StoreError(f"store manifest not found at {manifest_path}")
    return LocalStore(path=root, manifest=load_manifest(root))


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


__all__ = ["LocalStore", "StoreStats", "init_store", "load_manifest", "open_store"]
