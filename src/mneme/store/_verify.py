"""Local store verification and persisted index rebuild helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from mneme.core import Cid, MemoryItem, StoreCorruptionError, StoreError
from mneme.observability import ObservabilityConfig, emit_event, start_event_timer
from mneme.store._local import _write_json_atomic, load_manifest
from mneme.store._manifest import StoreManifest, ValueLogRef
from mneme.store._value_log import read_value_records

STORE_VERIFICATION_SCHEMA: Final = "mneme.store_verification.v1"
INDEX_REBUILD_SCHEMA: Final = "mneme.index_rebuild.v1"
INDEX_DATA_SCHEMA: Final = "mneme.flat_index_snapshot.v1"

_INDEX_BACKEND_FILE: Final = "index/backend.json"
_INDEX_DATA_FILE: Final = "index/data.json"


@dataclass(frozen=True)
class StoreVerificationReport:
    """Structured report returned by store verification."""

    ok: bool
    store_id: str | None
    item_count: int
    value_log_count: int
    index_backend: str | None
    errors: tuple[str, ...]
    schema_version: str = STORE_VERIFICATION_SCHEMA

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "store_id": self.store_id,
            "item_count": self.item_count,
            "value_log_count": self.value_log_count,
            "index_backend": self.index_backend,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class IndexRebuildReport:
    """Structured report returned by index rebuild."""

    ok: bool
    item_count: int
    index_backend: str | None
    data_path: str
    errors: tuple[str, ...]
    schema_version: str = INDEX_REBUILD_SCHEMA

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "item_count": self.item_count,
            "index_backend": self.index_backend,
            "data_path": self.data_path,
            "errors": list(self.errors),
        }


def verify_store(
    path: str | Path,
    *,
    raise_on_error: bool = False,
    observability: ObservabilityConfig | None = None,
) -> StoreVerificationReport:
    """Verify manifest, value logs, content ids, checksums, and index refs."""

    root = Path(path)
    started = start_event_timer(observability)
    try:
        report = _verify_store_report(root)
    except Exception as exc:
        if started is not None:
            emit_event(
                observability,
                event="mneme.store.verify",
                operation="store.verify",
                status="error",
                started=started,
                error=exc,
                store_id=None,
                backend=None,
                item_count=0,
                error_count=1,
            )
        raise
    if started is not None:
        emit_event(
            observability,
            event="mneme.store.verify",
            operation="store.verify",
            status="ok" if report.ok else "error",
            started=started,
            store_id=report.store_id,
            backend=report.index_backend,
            item_count=report.item_count,
            value_log_count=report.value_log_count,
            error_count=len(report.errors),
        )
    if raise_on_error and not report.ok:
        raise StoreCorruptionError("; ".join(report.errors))
    return report


def _verify_store_report(root: Path) -> StoreVerificationReport:
    """Verify a store and return a report without raising for report errors."""

    errors: list[str] = []
    manifest = _load_manifest_for_report(root, errors)
    if manifest is None:
        return _verification_report(
            manifest=None,
            item_count=0,
            value_log_count=0,
            errors=errors,
        )

    items, value_log_errors = _load_items_from_value_logs(root, manifest)
    errors.extend(value_log_errors)
    if not value_log_errors:
        errors.extend(_validate_manifest_value_logs(root, manifest))
        errors.extend(_validate_active_fingerprints(manifest, items))
    errors.extend(_validate_index_backend(root, manifest))
    errors.extend(_validate_index_data(root, manifest, items))
    return _verification_report(
        manifest=manifest,
        item_count=len(items),
        value_log_count=len(manifest.value_logs),
        errors=errors,
    )


def rebuild_index(path: str | Path) -> IndexRebuildReport:
    """Rebuild the persisted flat-index snapshot from append-only value logs."""

    root = Path(path)
    errors: list[str] = []
    try:
        manifest = load_manifest(root)
    except StoreError as exc:
        errors.append(str(exc))
        return IndexRebuildReport(
            ok=False,
            item_count=0,
            index_backend=None,
            data_path=_INDEX_DATA_FILE,
            errors=tuple(errors),
        )
    items, read_errors = _load_items_from_value_logs(root, manifest)
    if read_errors:
        return IndexRebuildReport(
            ok=False,
            item_count=0,
            index_backend=manifest.index.backend,
            data_path=_INDEX_DATA_FILE,
            errors=tuple(read_errors),
        )
    visible_items = _visible_items(manifest, items)

    _write_json_atomic(
        root / _INDEX_BACKEND_FILE,
        {"backend": manifest.index.backend, "params": dict(manifest.index.params)},
    )
    _write_json_atomic(
        root / _INDEX_DATA_FILE,
        {
            "schema_version": INDEX_DATA_SCHEMA,
            "backend": manifest.index.backend,
            "item_count": len(visible_items),
            "items": [
                {"content_id": cid.hex(), "key": _key_to_json(item.key)}
                for cid, item in sorted(
                    visible_items.items(), key=lambda entry: entry[0]
                )
            ],
        },
    )
    return IndexRebuildReport(
        ok=True,
        item_count=len(visible_items),
        index_backend=manifest.index.backend,
        data_path=_INDEX_DATA_FILE,
        errors=(),
    )


def _load_manifest_for_report(
    root: Path,
    errors: list[str],
) -> StoreManifest | None:
    try:
        return load_manifest(root)
    except StoreError as exc:
        errors.append(str(exc))
        return None


def _load_items_from_value_logs(
    root: Path,
    manifest: StoreManifest,
) -> tuple[dict[Cid, MemoryItem], list[str]]:
    items: dict[Cid, MemoryItem] = {}
    errors: list[str] = []
    for value_log in manifest.value_logs:
        log_path = root / value_log.path
        try:
            for item in read_value_records(log_path):
                if item.content_id is None:
                    errors.append(
                        f"value log item missing content_id: {value_log.path}"
                    )
                    continue
                if item.content_id in items:
                    errors.append(
                        f"duplicate content_id in value logs: {item.content_id.hex()}"
                    )
                items[item.content_id] = item
        except (FileNotFoundError, StoreCorruptionError) as exc:
            errors.append(f"{value_log.path}: {exc}")
    return items, errors


def _validate_manifest_value_logs(
    root: Path,
    manifest: StoreManifest,
) -> list[str]:
    errors: list[str] = []
    for value_log in manifest.value_logs:
        errors.extend(_validate_value_log_ref(root, value_log))
    return errors


def _validate_value_log_ref(root: Path, value_log: ValueLogRef) -> list[str]:
    log_path = root / value_log.path
    errors: list[str] = []
    if not log_path.exists():
        return [f"value log missing: {value_log.path}"]
    actual_size = log_path.stat().st_size
    if actual_size != value_log.size_bytes:
        errors.append(
            "value log size mismatch: "
            f"{value_log.path} manifest={value_log.size_bytes} actual={actual_size}"
        )
    try:
        actual_count = sum(1 for _ in read_value_records(log_path))
    except StoreCorruptionError as exc:
        errors.append(f"{value_log.path}: {exc}")
        return errors
    if actual_count != value_log.record_count:
        errors.append(
            "value log record count mismatch: "
            f"{value_log.path} manifest={value_log.record_count} actual={actual_count}"
        )
    return errors


def _validate_active_fingerprints(
    manifest: StoreManifest,
    items: dict[Cid, MemoryItem],
) -> list[str]:
    errors: list[str] = []
    active_fingerprints = set(manifest.active_fingerprints)
    for cid, item in items.items():
        if item.encoder_fp not in active_fingerprints:
            errors.append(
                f"value log item uses fingerprint missing from manifest: {cid.hex()}"
            )
    return errors


def _validate_index_backend(root: Path, manifest: StoreManifest) -> list[str]:
    backend_path = root / _INDEX_BACKEND_FILE
    try:
        data = json.loads(backend_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"index backend missing: {_INDEX_BACKEND_FILE}"]
    except json.JSONDecodeError as exc:
        return [f"index backend is malformed JSON: {exc.msg}"]
    if not isinstance(data, dict):
        return ["index backend must be an object"]
    errors: list[str] = []
    if data.get("backend") != manifest.index.backend:
        errors.append(
            "index backend mismatch: "
            f"manifest={manifest.index.backend!r} index={data.get('backend')!r}"
        )
    params = data.get("params", {})
    if not isinstance(params, dict):
        errors.append("index backend params must be an object")
    elif params != dict(manifest.index.params):
        errors.append("index backend params mismatch")
    return errors


def _validate_index_data(
    root: Path,
    manifest: StoreManifest,
    value_log_items: dict[Cid, MemoryItem],
) -> list[str]:
    data_path = root / _INDEX_DATA_FILE
    if not data_path.exists():
        return []
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"index data is malformed JSON: {exc.msg}"]
    if not isinstance(data, dict):
        return ["index data must be an object"]
    errors: list[str] = []
    if data.get("schema_version") != INDEX_DATA_SCHEMA:
        errors.append("unsupported index data schema")
    if data.get("backend") != manifest.index.backend:
        errors.append(
            "index data backend mismatch: "
            f"manifest={manifest.index.backend!r} index={data.get('backend')!r}"
        )
    items = data.get("items")
    if not isinstance(items, list):
        return [*errors, "index data items must be a list"]
    item_count = data.get("item_count")
    if item_count != len(items):
        errors.append(
            f"index data item_count mismatch: item_count={item_count!r} "
            f"actual={len(items)}"
        )
    seen: set[Cid] = set()
    tombstoned = _tombstoned_cids(manifest)
    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            errors.append(f"index data item {index} must be an object")
            continue
        cid = _cid_from_hex(raw_item.get("content_id"))
        if cid is None:
            errors.append(f"index data item {index} content_id must be hex bytes")
            continue
        if cid in seen:
            errors.append(f"index data contains duplicate content_id: {cid.hex()}")
        seen.add(cid)
        if cid not in value_log_items:
            errors.append(f"index references missing value log item: {cid.hex()}")
        if cid in tombstoned:
            errors.append(f"index references tombstoned value log item: {cid.hex()}")
        errors.extend(
            _validate_index_key(raw_item.get("key"), cid, value_log_items.get(cid))
        )
    missing_from_index = set(value_log_items) - tombstoned - seen
    for cid in sorted(missing_from_index):
        errors.append(f"value log item missing from index data: {cid.hex()}")
    return errors


def _visible_items(
    manifest: StoreManifest,
    items: dict[Cid, MemoryItem],
) -> dict[Cid, MemoryItem]:
    tombstoned = _tombstoned_cids(manifest)
    return {cid: item for cid, item in items.items() if cid not in tombstoned}


def _tombstoned_cids(manifest: StoreManifest) -> set[Cid]:
    tombstones = manifest.retention_policy.get("tombstones", [])
    if not isinstance(tombstones, list):
        return set()
    cids: set[Cid] = set()
    for tombstone in tombstones:
        if not isinstance(tombstone, dict):
            continue
        content_id_hex = tombstone.get("content_id")
        if not isinstance(content_id_hex, str):
            continue
        try:
            cids.add(bytes.fromhex(content_id_hex))
        except ValueError:
            continue
    return cids


def _validate_index_key(
    raw_key: object,
    cid: Cid,
    item: MemoryItem | None,
) -> list[str]:
    if not isinstance(raw_key, list) or not all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in raw_key
    ):
        return [f"index key for {cid.hex()} must be a numeric list"]
    if item is None:
        return []
    key = np.asarray(raw_key, dtype=np.float32)
    if key.shape != item.key.shape:
        return [f"index key shape mismatch for {cid.hex()}"]
    if not bool(np.allclose(key, item.key, rtol=0.0, atol=0.0)):
        return [f"index key mismatch for {cid.hex()}"]
    return []


def _cid_from_hex(value: object) -> Cid | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return bytes.fromhex(value)
    except ValueError:
        return None


def _key_to_json(value: np.ndarray) -> list[float]:
    return [float(item) for item in value.tolist()]


def _verification_report(
    *,
    manifest: StoreManifest | None,
    item_count: int,
    value_log_count: int,
    errors: list[str],
) -> StoreVerificationReport:
    return StoreVerificationReport(
        ok=not errors,
        store_id=str(manifest.store_id) if manifest is not None else None,
        item_count=item_count,
        value_log_count=value_log_count,
        index_backend=manifest.index.backend if manifest is not None else None,
        errors=tuple(errors),
    )


__all__ = [
    "INDEX_DATA_SCHEMA",
    "INDEX_REBUILD_SCHEMA",
    "STORE_VERIFICATION_SCHEMA",
    "IndexRebuildReport",
    "StoreVerificationReport",
    "rebuild_index",
    "verify_store",
]
