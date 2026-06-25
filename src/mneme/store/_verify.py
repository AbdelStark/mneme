"""Local store verification and persisted index rebuild helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from mneme.core import (
    Cid,
    MemoryItem,
    StoreCorruptionError,
    StoreError,
    ValidationError,
)
from mneme.core._ids import cid_from_hex
from mneme.core._json import loads_strict_json
from mneme.observability import ObservabilityConfig, emit_event, start_event_timer
from mneme.store._local import (
    _COMMITMENT_FILE,
    _write_json_atomic,
    load_manifest,
    open_store,
)
from mneme.store._manifest import StoreManifest, ValueLogRef
from mneme.store._retention import tombstoned_cids
from mneme.store._value_log import read_value_records

STORE_VERIFICATION_SCHEMA: Final = "mneme.store_verification.v1"
INDEX_REBUILD_SCHEMA: Final = "mneme.index_rebuild.v1"
INDEX_DATA_SCHEMA: Final = "mneme.flat_index_snapshot.v1"
COMMIT_INIT_SCHEMA: Final = "mneme.store_commit_init.v1"

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

    def __post_init__(self) -> None:
        if self.schema_version != STORE_VERIFICATION_SCHEMA:
            raise ValidationError("unsupported store verification report schema")
        object.__setattr__(self, "ok", _require_bool(self.ok, "ok"))
        object.__setattr__(
            self, "store_id", _optional_non_empty_string(self.store_id, "store_id")
        )
        object.__setattr__(
            self, "item_count", _require_non_negative_int(self.item_count, "item_count")
        )
        object.__setattr__(
            self,
            "value_log_count",
            _require_non_negative_int(self.value_log_count, "value_log_count"),
        )
        object.__setattr__(
            self,
            "index_backend",
            _optional_non_empty_string(self.index_backend, "index_backend"),
        )
        object.__setattr__(self, "errors", _string_tuple(self.errors, "errors"))

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

    def __post_init__(self) -> None:
        if self.schema_version != INDEX_REBUILD_SCHEMA:
            raise ValidationError("unsupported index rebuild report schema")
        object.__setattr__(self, "ok", _require_bool(self.ok, "ok"))
        object.__setattr__(
            self, "item_count", _require_non_negative_int(self.item_count, "item_count")
        )
        object.__setattr__(
            self,
            "index_backend",
            _optional_non_empty_string(self.index_backend, "index_backend"),
        )
        object.__setattr__(
            self, "data_path", _require_non_empty_string(self.data_path, "data_path")
        )
        object.__setattr__(self, "errors", _string_tuple(self.errors, "errors"))

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


@dataclass(frozen=True)
class CommitInitReport:
    """Structured report returned by committed-store upgrade."""

    ok: bool
    store_id: str | None
    item_count: int
    root: str | None
    commitment_path: str
    already_initialized: bool
    errors: tuple[str, ...]
    schema_version: str = COMMIT_INIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COMMIT_INIT_SCHEMA:
            raise ValidationError("unsupported commit init report schema")
        object.__setattr__(self, "ok", _require_bool(self.ok, "ok"))
        object.__setattr__(
            self, "store_id", _optional_non_empty_string(self.store_id, "store_id")
        )
        object.__setattr__(
            self, "item_count", _require_non_negative_int(self.item_count, "item_count")
        )
        object.__setattr__(self, "root", _optional_root_hex(self.root))
        object.__setattr__(
            self,
            "commitment_path",
            _require_non_empty_string(self.commitment_path, "commitment_path"),
        )
        object.__setattr__(
            self,
            "already_initialized",
            _require_bool(self.already_initialized, "already_initialized"),
        )
        object.__setattr__(self, "errors", _string_tuple(self.errors, "errors"))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "store_id": self.store_id,
            "item_count": self.item_count,
            "root": self.root,
            "commitment_path": self.commitment_path,
            "already_initialized": self.already_initialized,
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


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a bool")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{field_name} must be a non-negative integer")
    return value


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_non_empty_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name)


def _optional_root_hex(value: object) -> str | None:
    if value is None:
        return None
    return cid_from_hex(value, "root", error_type=ValidationError).hex()


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValidationError(f"{field_name} must be a sequence")
    return tuple(
        _require_non_empty_string(item, f"{field_name} item") for item in value
    )


def commit_init_store(path: str | Path) -> CommitInitReport:
    """Upgrade a verified local store by initializing its commitment state."""

    verification = verify_store(path)
    if not verification.ok:
        return CommitInitReport(
            ok=False,
            store_id=verification.store_id,
            item_count=0,
            root=None,
            commitment_path=_COMMITMENT_FILE,
            already_initialized=False,
            errors=verification.errors,
        )
    store = open_store(path)
    already_initialized = store.manifest.commitment.enabled
    root = store.commit()
    return CommitInitReport(
        ok=True,
        store_id=str(store.manifest.store_id),
        item_count=verification.item_count,
        root=root.hex(),
        commitment_path=_COMMITMENT_FILE,
        already_initialized=already_initialized,
        errors=(),
    )


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
        data = loads_strict_json(backend_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"index backend missing: {_INDEX_BACKEND_FILE}"]
    except ValueError as exc:
        return [f"index backend is malformed JSON: {exc}"]
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
        data = loads_strict_json(data_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"index data is malformed JSON: {exc}"]
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
    tombstoned = tombstoned_cids(manifest)
    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            errors.append(f"index data item {index} must be an object")
            continue
        try:
            cid = cid_from_hex(
                raw_item.get("content_id"),
                "content_id",
                error_type=StoreCorruptionError,
            )
        except StoreCorruptionError as exc:
            errors.append(f"index data item {index} {exc}")
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
    tombstoned = tombstoned_cids(manifest)
    return {cid: item for cid, item in items.items() if cid not in tombstoned}


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
    "COMMIT_INIT_SCHEMA",
    "INDEX_DATA_SCHEMA",
    "INDEX_REBUILD_SCHEMA",
    "STORE_VERIFICATION_SCHEMA",
    "CommitInitReport",
    "IndexRebuildReport",
    "StoreVerificationReport",
    "commit_init_store",
    "rebuild_index",
    "verify_store",
]
