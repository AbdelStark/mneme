from __future__ import annotations

import json
from types import MappingProxyType
from uuid import UUID

import pytest

from mneme.core import (
    EncoderFingerprint,
    Metric,
    SchemaVersionError,
    StoreCorruptionError,
    StoreError,
    ValidationError,
)
from mneme.store import (
    STORE_MANIFEST_SCHEMA,
    IndexConfig,
    LocalStore,
    init_store,
    load_manifest,
    open_store,
)


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


class _SizedIndex:
    def __init__(self, size: int) -> None:
        self.size = size

    def add(self, _cid: bytes, _key: object) -> None:
        return None

    def add_batch(self, _items: object) -> None:
        return None

    def search(
        self,
        _q: object,
        _k: int,
        *,
        metric: Metric,
        ef: int | None = None,
    ) -> list[tuple[bytes, float]]:
        return []

    def __len__(self) -> int:
        return self.size


def test_local_store_constructor_normalizes_direct_handle_fields(tmp_path) -> None:
    store = init_store(tmp_path / "store")

    handle = LocalStore(
        path=str(store.path),
        manifest=store.manifest,
        index=store.index,
        _items={},
        recovery_events=[],
    )

    assert handle.path == store.path
    assert handle._items == {}
    assert handle.recovery_events == ()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"path": object()}, "path must be a path-like value"),
        ({"path": ""}, "path must not be empty"),
        ({"manifest": object()}, "manifest must be a StoreManifest"),
        ({"index": object()}, "index must implement Index"),
        ({"_items": []}, "_items must be a mapping"),
        ({"_items": {b"0": object()}}, "_items keys must be 32 bytes"),
        ({"_items": {b"0" * 32: object()}}, "_items values must be MemoryItem"),
        ({"recovery_events": object()}, "recovery_events must be a sequence"),
        (
            {"recovery_events": (object(),)},
            "recovery_events items must be StoreRecoveryEvent",
        ),
        ({"observability": object()}, "observability must be an ObservabilityConfig"),
    ),
)
def test_local_store_constructor_rejects_malformed_fields(
    tmp_path,
    kwargs: dict[str, object],
    match: str,
) -> None:
    store = init_store(tmp_path / "store")
    values: dict[str, object] = {
        "path": store.path,
        "manifest": store.manifest,
        "index": store.index,
        "_items": {},
    }
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        LocalStore(**values)


def test_local_store_constructor_rejects_index_item_count_mismatch(tmp_path) -> None:
    store = init_store(tmp_path / "store")

    with pytest.raises(ValidationError, match="index size"):
        LocalStore(
            path=store.path,
            manifest=store.manifest,
            index=_SizedIndex(size=1),
            _items={},
        )


def test_init_store_creates_layout_and_schema_versioned_manifest(tmp_path) -> None:
    root = tmp_path / "store"
    store_id = UUID("12345678-1234-5678-1234-567812345678")

    store = init_store(
        root,
        store_id=store_id,
        active_fingerprints=[_fingerprint()],
        index_backend="flat",
        index_params={"metric": "l2"},
    )

    assert store.path == root
    assert (root / "manifest.json").is_file()
    assert (root / "values").is_dir()
    assert (root / "values/log-000000.mnv").is_file()
    assert (root / "index").is_dir()
    assert (root / "index/backend.json").is_file()
    assert (root / "transactions").is_dir()
    assert (root / "receipts").is_dir()

    manifest_json = json.loads((root / "manifest.json").read_text())
    assert manifest_json["schema_version"] == STORE_MANIFEST_SCHEMA
    assert manifest_json["store_id"] == str(store_id)
    assert manifest_json["active_fingerprints"][0]["summarizer_id"] == "meanpool-v1"
    assert manifest_json["value_logs"] == [
        {"path": "values/log-000000.mnv", "record_count": 0, "size_bytes": 0}
    ]
    assert manifest_json["index"] == {"backend": "flat", "params": {"metric": "l2"}}
    assert manifest_json["retention_policy"] == {"policy": "none", "tombstones": []}
    assert manifest_json["last_completed_transaction"] is None
    assert manifest_json["commitment"] == {
        "backend": None,
        "enabled": False,
        "files": [],
        "root": None,
    }


def test_open_store_create_initializes_missing_store_and_stats_read_manifest(
    tmp_path,
) -> None:
    store = open_store(tmp_path / "store", create=True)

    reopened = open_store(tmp_path / "store")
    stats = reopened.stats()

    assert reopened.manifest == store.manifest
    assert stats.store_id == store.manifest.store_id
    assert stats.schema_version == STORE_MANIFEST_SCHEMA
    assert stats.active_fingerprint_count == 0
    assert stats.value_log_count == 1
    assert stats.value_record_count == 0
    assert stats.value_bytes == 0
    assert stats.index_backend == "flat"
    assert stats.last_completed_transaction is None
    assert not stats.commitments_enabled


def test_store_open_and_init_bool_flags_reject_non_bool_values(tmp_path) -> None:
    missing_root = tmp_path / "missing-store"

    with pytest.raises(ValidationError, match="create must be a bool"):
        open_store(missing_root, create="yes")  # type: ignore[arg-type]

    assert not missing_root.exists()

    root = tmp_path / "store"
    init_store(root)

    with pytest.raises(ValidationError, match="exist_ok must be a bool"):
        init_store(root, exist_ok="yes")  # type: ignore[arg-type]


def test_store_path_entrypoints_reject_empty_paths_without_layout_side_effects(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="path must not be empty"):
        init_store("")
    with pytest.raises(ValidationError, match="path must not be empty"):
        open_store("", create=True)
    with pytest.raises(ValidationError, match="path must not be empty"):
        load_manifest("")

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "values").exists()
    assert not (tmp_path / "index").exists()
    assert not (tmp_path / "transactions").exists()
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("path_value", (object(),))
def test_store_path_entrypoints_reject_non_path_like_values(path_value: object) -> None:
    with pytest.raises(ValidationError, match="path must be a path-like value"):
        init_store(path_value)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="path must be a path-like value"):
        open_store(path_value)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="path must be a path-like value"):
        load_manifest(path_value)  # type: ignore[arg-type]


def test_load_manifest_reconstructs_fingerprints_and_index_config(tmp_path) -> None:
    root = tmp_path / "store"
    init_store(root, active_fingerprints=[_fingerprint()])

    manifest = load_manifest(root)

    assert manifest.active_fingerprints == (_fingerprint(),)
    assert manifest.index.backend == "flat"
    assert manifest.value_logs[0].path == "values/log-000000.mnv"


def test_index_config_rejects_non_json_safe_direct_params() -> None:
    with pytest.raises(
        StoreCorruptionError,
        match="index params.threshold must contain finite",
    ):
        IndexConfig("flat", {"threshold": float("nan")})

    with pytest.raises(StoreCorruptionError, match="index params keys must be strings"):
        IndexConfig("flat", {1: "bad"})
    with pytest.raises(
        StoreCorruptionError,
        match="index params keys must be non-empty strings",
    ):
        IndexConfig("flat", {"": "bad"})
    with pytest.raises(
        StoreCorruptionError,
        match="index params.nested keys must be strings",
    ):
        IndexConfig("flat", {"nested": {1: "bad"}})


def test_index_config_normalizes_nested_json_params() -> None:
    config = IndexConfig(
        "flat",
        {
            "nested": MappingProxyType({"enabled": True}),
            "sequence": (1, MappingProxyType({"name": "fixture"})),
        },
    )

    assert config.params == {
        "nested": {"enabled": True},
        "sequence": [1, {"name": "fixture"}],
    }


def test_init_store_writes_normalized_nested_index_params(tmp_path) -> None:
    root = tmp_path / "store"

    init_store(
        root,
        index_params={
            "nested": MappingProxyType({"enabled": True}),
            "sequence": (1, MappingProxyType({"name": "fixture"})),
        },
    )

    manifest_json = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    backend_json = json.loads(
        (root / "index" / "backend.json").read_text(encoding="utf-8")
    )

    assert manifest_json["index"]["params"] == {
        "nested": {"enabled": True},
        "sequence": [1, {"name": "fixture"}],
    }
    assert backend_json["params"] == manifest_json["index"]["params"]


def test_init_store_rejects_invalid_index_params_before_layout(tmp_path) -> None:
    root = tmp_path / "store"

    with pytest.raises(
        StoreCorruptionError,
        match="index params.threshold must contain finite",
    ):
        init_store(root, index_params={"threshold": float("inf")})

    assert not root.exists()


def test_init_store_wraps_layout_filesystem_errors(tmp_path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")

    with pytest.raises(StoreError, match="layout could not be initialized"):
        init_store(blocked_parent / "store")


def test_unknown_manifest_major_version_fails_closed(tmp_path) -> None:
    root = tmp_path / "store"
    init_store(root)
    manifest_path = root / "manifest.json"
    manifest_json = json.loads(manifest_path.read_text())
    manifest_json["schema_version"] = "mneme.store_manifest.v2"
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    with pytest.raises(SchemaVersionError, match="unsupported manifest schema"):
        open_store(root)


def test_manifest_rejects_unsupported_active_fingerprint_schema_as_corruption(
    tmp_path,
) -> None:
    root = tmp_path / "store"
    init_store(root, active_fingerprints=[_fingerprint()])
    manifest_path = root / "manifest.json"
    manifest_json = json.loads(manifest_path.read_text())
    manifest_json["active_fingerprints"][0]["schema_version"] = (
        "mneme.encoder_fingerprint.v2"
    )
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="invalid encoder fingerprint"):
        open_store(root)


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("created_at", "not-a-timestamp"),
        ("updated_at", "2026-06-24T00:00:00"),
        ("updated_at", "2026-06-24T01:00:00+01:00"),
    ),
)
def test_manifest_rejects_malformed_timestamps(
    tmp_path,
    field_name: str,
    value: object,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    manifest_path = root / "manifest.json"
    manifest_json = json.loads(manifest_path.read_text())
    manifest_json[field_name] = value
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    with pytest.raises(
        StoreCorruptionError,
        match=f"{field_name} must be an ISO 8601 UTC timestamp",
    ):
        open_store(root)


def test_missing_manifest_fails_without_create(tmp_path) -> None:
    root = tmp_path / "store"
    root.mkdir()

    with pytest.raises(StoreError, match="manifest not found"):
        open_store(root)


def test_malformed_manifest_json_raises_store_corruption(tmp_path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="malformed JSON"):
        open_store(root)


def test_unreadable_manifest_raises_store_corruption(tmp_path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.json").mkdir()

    with pytest.raises(StoreCorruptionError, match="manifest could not be read"):
        open_store(root)


def test_manifest_rejects_nonstandard_json_constants(tmp_path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.json").write_text('{"schema_version": NaN}', encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="malformed JSON"):
        open_store(root)


def test_manifest_with_missing_required_fields_is_corrupt(tmp_path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": STORE_MANIFEST_SCHEMA}),
        encoding="utf-8",
    )

    with pytest.raises(StoreCorruptionError, match="store_id"):
        open_store(root)


def test_manifest_rejects_value_log_path_traversal(tmp_path) -> None:
    root = tmp_path / "store"
    init_store(root)
    manifest_path = root / "manifest.json"
    manifest_json = json.loads(manifest_path.read_text())
    manifest_json["value_logs"][0]["path"] = "../outside.mnv"
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="relative store path"):
        open_store(root)


def test_manifest_rejects_commitment_file_path_traversal(tmp_path) -> None:
    root = tmp_path / "store"
    init_store(root)
    manifest_path = root / "manifest.json"
    manifest_json = json.loads(manifest_path.read_text())
    manifest_json["commitment"]["files"] = ["/tmp/root.sig"]
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="relative store path"):
        open_store(root)


@pytest.mark.parametrize(
    ("commitment", "match"),
    (
        (
            {
                "enabled": True,
                "backend": None,
                "root": None,
                "files": [],
            },
            "commitment backend must be set when enabled",
        ),
        (
            {
                "enabled": True,
                "backend": "mmr-v1",
                "root": "00",
                "files": ["receipts/commitment-mmr-v1.json"],
            },
            "commitment root must be 32 bytes",
        ),
        (
            {
                "enabled": True,
                "backend": "mmr-v1",
                "root": "zz" * 32,
                "files": ["receipts/commitment-mmr-v1.json"],
            },
            "commitment root must be hex bytes",
        ),
        (
            {
                "enabled": True,
                "backend": "mmr-v1",
                "root": "00" * 32,
                "files": [],
            },
            "commitment files must not be empty when enabled",
        ),
        (
            {
                "enabled": True,
                "backend": "sha256-tree-v1",
                "root": "00" * 32,
                "files": ["receipts/commitment-mmr-v1.json"],
            },
            "commitment backend is unsupported",
        ),
        (
            {
                "enabled": False,
                "backend": "mmr-v1",
                "root": None,
                "files": [],
            },
            "disabled commitment must not set backend, root, or files",
        ),
    ),
)
def test_manifest_rejects_incoherent_commitment_state(
    tmp_path,
    commitment: dict[str, object],
    match: str,
) -> None:
    root = tmp_path / "store"
    init_store(root)
    manifest_path = root / "manifest.json"
    manifest_json = json.loads(manifest_path.read_text())
    manifest_json["commitment"] = commitment
    manifest_path.write_text(json.dumps(manifest_json), encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match=match):
        open_store(root)


def test_init_store_refuses_existing_manifest_without_exist_ok(tmp_path) -> None:
    root = tmp_path / "store"
    init_store(root)

    with pytest.raises(StoreError, match="already exists"):
        init_store(root)
