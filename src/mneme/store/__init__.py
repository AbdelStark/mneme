"""Local memory-store layout and manifest API."""

from mneme.store._local import (
    LocalStore,
    StoreRecoveryEvent,
    StoreStats,
    init_store,
    load_manifest,
    open_store,
)
from mneme.store._manifest import (
    STORE_MANIFEST_SCHEMA,
    CommitmentState,
    IndexConfig,
    StoreManifest,
    ValueLogRef,
    age_retention,
    count_retention,
)
from mneme.store._verify import (
    INDEX_DATA_SCHEMA,
    INDEX_REBUILD_SCHEMA,
    STORE_VERIFICATION_SCHEMA,
    IndexRebuildReport,
    StoreVerificationReport,
    rebuild_index,
    verify_store,
)

__all__ = [
    "INDEX_DATA_SCHEMA",
    "INDEX_REBUILD_SCHEMA",
    "STORE_MANIFEST_SCHEMA",
    "STORE_VERIFICATION_SCHEMA",
    "CommitmentState",
    "IndexConfig",
    "IndexRebuildReport",
    "LocalStore",
    "StoreRecoveryEvent",
    "StoreManifest",
    "StoreStats",
    "StoreVerificationReport",
    "ValueLogRef",
    "age_retention",
    "count_retention",
    "init_store",
    "load_manifest",
    "open_store",
    "rebuild_index",
    "verify_store",
]
