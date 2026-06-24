from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mneme.adapter import AdapterCheckpointMetadata
from mneme.eval import DatasetRef, validate_report_json
from mneme.remote import ErrorMessage, QueryRequest
from mneme.store import INDEX_DATA_SCHEMA, StoreManifest

SNAPSHOT = Path("tests/fixtures/compat/public_api_snapshot.json")
SCHEMA_PAYLOADS = Path("tests/fixtures/compat/schema_payloads.json")
COMPAT_DOC = Path("docs/release/API_COMPATIBILITY.md")


def test_public_exports_signatures_and_schema_versions_match_snapshot() -> None:
    snapshot = _load_mapping(SNAPSHOT)

    assert snapshot["schema_version"] == "mneme.public_api_snapshot.v1"
    assert _live_modules(snapshot["modules"]) == snapshot["modules"]
    assert _live_signatures(snapshot["signatures"]) == snapshot["signatures"]
    assert (
        _live_schema_versions(snapshot["schema_versions"])
        == snapshot["schema_versions"]
    )


def test_documented_protocols_remain_public_protocols() -> None:
    snapshot = _load_mapping(SNAPSHOT)

    protocols = snapshot["protocols"]
    assert isinstance(protocols, list)
    for path in protocols:
        symbol = _resolve(str(path))
        assert getattr(symbol, "_is_protocol", False) is True


def test_schema_payload_fixtures_validate_current_persisted_objects() -> None:
    payloads = _load_mapping(SCHEMA_PAYLOADS)

    assert payloads["schema_version"] == "mneme.schema_payload_fixtures.v1"
    adapter = AdapterCheckpointMetadata.from_json(
        _mapping(payloads["adapter_checkpoint_metadata"])
    )
    dataset = DatasetRef.from_json(payloads["dataset_ref"])
    report = validate_report_json(payloads["eval_report"])
    error = ErrorMessage.from_json(payloads["remote_error_message"])
    query = QueryRequest.from_json(payloads["remote_query_request"])
    manifest = StoreManifest.from_json(payloads["store_manifest"])
    index_snapshot = _mapping(payloads["index_snapshot"])

    assert adapter.schema_version == "mneme.adapter_checkpoint.v1"
    assert dataset.schema_version == "mneme.dataset_ref.v1"
    assert report.schema_version == "mneme.eval_report.v1"
    assert error.schema_version == "mneme.error.v1"
    assert query.schema_version == "mneme.query.request.v1"
    assert manifest.schema_version == "mneme.store_manifest.v1"
    assert index_snapshot["schema_version"] == INDEX_DATA_SCHEMA
    assert index_snapshot["item_count"] == len(index_snapshot["items"])


def test_compatibility_snapshot_digest_is_documented() -> None:
    digest = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    text = COMPAT_DOC.read_text(encoding="utf-8")

    assert f"Snapshot digest: `{digest}`" in text
    assert "migration note or changelog entry" in text
    assert "deprecation policy" in text.lower()


def _live_modules(expected: object) -> dict[str, list[str]]:
    modules = _mapping(expected)
    return {
        module_name: list(importlib.import_module(module_name).__all__)
        for module_name in modules
    }


def _live_signatures(expected: object) -> dict[str, str]:
    signatures = _mapping(expected)
    return {
        symbol_path: str(inspect.signature(_resolve(symbol_path)))
        for symbol_path in signatures
    }


def _live_schema_versions(expected: object) -> dict[str, str]:
    constants = _mapping(expected)
    return {symbol_path: str(_resolve(symbol_path)) for symbol_path in constants}


def _resolve(path: str) -> Any:
    module_name, _, name = path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, name)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")))


def _mapping(value: object) -> Mapping[str, Any]:
    assert isinstance(value, Mapping)
    return value
