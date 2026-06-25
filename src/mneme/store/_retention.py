"""Retention policy application for local stores."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from mneme.core import Cid, MemoryItem, StoreCorruptionError
from mneme.core._ids import cid_from_hex
from mneme.store._manifest import StoreManifest


def visible_cids(
    manifest: StoreManifest,
    items: Mapping[Cid, MemoryItem],
) -> set[Cid]:
    """Return content ids visible after applying manifest tombstones."""

    return set(items) - tombstoned_cids(manifest)


def tombstoned_cids(manifest: StoreManifest) -> set[Cid]:
    """Return manifest tombstones as content-id bytes."""

    return set(_retention_tombstones(manifest.retention_policy))


def apply_retention_policy(
    policy: Mapping[str, Any],
    all_items: Mapping[Cid, MemoryItem],
    *,
    transaction_id: str,
) -> dict[str, Any]:
    """Apply a normalized manifest retention policy to the current item set."""

    policy_name = str(policy.get("policy", "none"))
    tombstones = _retention_tombstones(policy)
    live = {cid: item for cid, item in all_items.items() if cid not in tombstones}
    if policy_name == "count":
        max_items = _retention_int(policy, "max_items")
        evicted = _count_evictions(live, max_items)
        tombstones.update(
            _new_tombstones(evicted, reason="count", transaction_id=transaction_id)
        )
        return {
            "policy": "count",
            "max_items": max_items,
            "tombstones": _sorted_tombstones(tombstones),
        }
    if policy_name == "age":
        max_age_seconds = _retention_int(policy, "max_age_seconds")
        evicted = _age_evictions(live, max_age_seconds)
        tombstones.update(
            _new_tombstones(evicted, reason="age", transaction_id=transaction_id)
        )
        return {
            "policy": "age",
            "max_age_seconds": max_age_seconds,
            "tombstones": _sorted_tombstones(tombstones),
        }
    return {"policy": "none", "tombstones": _sorted_tombstones(tombstones)}


def _retention_tombstones(policy: Mapping[str, Any]) -> dict[Cid, dict[str, Any]]:
    raw_tombstones = policy.get("tombstones", [])
    tombstones: dict[Cid, dict[str, Any]] = {}
    if not isinstance(raw_tombstones, list):
        raise StoreCorruptionError("retention_policy tombstones must be a list")
    for index, raw in enumerate(raw_tombstones):
        if not isinstance(raw, Mapping):
            raise StoreCorruptionError(
                f"retention_policy tombstones[{index}] must be an object"
            )
        cid = cid_from_hex(
            raw.get("content_id"),
            "tombstone content_id",
            error_type=StoreCorruptionError,
        )
        tombstones[cid] = dict(raw)
    return tombstones


def _retention_int(policy: Mapping[str, Any], field_name: str) -> int:
    value = policy.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreCorruptionError(f"retention_policy {field_name} must be an integer")
    if value < 0:
        raise StoreCorruptionError(f"retention_policy {field_name} must be >= 0")
    return value


def _count_evictions(items: Mapping[Cid, MemoryItem], max_items: int) -> list[Cid]:
    ordered = sorted(
        items.items(),
        key=lambda entry: (entry[1].value.t, entry[0]),
        reverse=True,
    )
    return [cid for cid, _ in ordered[max_items:]]


def _age_evictions(items: Mapping[Cid, MemoryItem], max_age_seconds: int) -> list[Cid]:
    if not items:
        return []
    newest = max(item.value.t for item in items.values())
    return [
        cid for cid, item in items.items() if newest - item.value.t > max_age_seconds
    ]


def _new_tombstones(
    cids: list[Cid],
    *,
    reason: str,
    transaction_id: str,
) -> dict[Cid, dict[str, Any]]:
    created_at = _utc_now()
    return {
        cid: {
            "content_id": cid.hex(),
            "reason": reason,
            "created_at": created_at,
            "transaction_id": transaction_id,
        }
        for cid in cids
    }


def _sorted_tombstones(
    tombstones: Mapping[Cid, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(tombstones[cid]) for cid in sorted(tombstones)]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "apply_retention_policy",
    "tombstoned_cids",
    "visible_cids",
]
