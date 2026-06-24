"""Optional trained memory adapter modules."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "ADAPTER_CHECKPOINT_METADATA_FILE":
        from mneme.adapter._checkpoint import ADAPTER_CHECKPOINT_METADATA_FILE

        return ADAPTER_CHECKPOINT_METADATA_FILE
    if name == "ADAPTER_CHECKPOINT_SCHEMA":
        from mneme.adapter._checkpoint import ADAPTER_CHECKPOINT_SCHEMA

        return ADAPTER_CHECKPOINT_SCHEMA
    if name == "DEFAULT_ADAPTER_WEIGHTS_FILE":
        from mneme.adapter._checkpoint import DEFAULT_ADAPTER_WEIGHTS_FILE

        return DEFAULT_ADAPTER_WEIGHTS_FILE
    if name == "AdapterCheckpoint":
        from mneme.adapter._checkpoint import AdapterCheckpoint

        return AdapterCheckpoint
    if name == "AdapterCheckpointMetadata":
        from mneme.adapter._checkpoint import AdapterCheckpointMetadata

        return AdapterCheckpointMetadata
    if name == "CrossAttnAdapter":
        from mneme.adapter._cross_attention import CrossAttnAdapter

        return CrossAttnAdapter
    if name == "AdapterTrainingBatch":
        from mneme.adapter._training import AdapterTrainingBatch

        return AdapterTrainingBatch
    if name == "train_frozen_base_adapter":
        from mneme.adapter._training import train_frozen_base_adapter

        return train_frozen_base_adapter
    if name == "load_adapter_checkpoint":
        from mneme.adapter._checkpoint import load_adapter_checkpoint

        return load_adapter_checkpoint
    if name == "load_adapter_checkpoint_metadata":
        from mneme.adapter._checkpoint import load_adapter_checkpoint_metadata

        return load_adapter_checkpoint_metadata
    if name == "save_adapter_checkpoint_metadata":
        from mneme.adapter._checkpoint import save_adapter_checkpoint_metadata

        return save_adapter_checkpoint_metadata
    raise AttributeError(f"module 'mneme.adapter' has no attribute {name!r}")


__all__ = [
    "ADAPTER_CHECKPOINT_METADATA_FILE",
    "ADAPTER_CHECKPOINT_SCHEMA",
    "DEFAULT_ADAPTER_WEIGHTS_FILE",
    "AdapterCheckpoint",
    "AdapterCheckpointMetadata",
    "AdapterTrainingBatch",
    "CrossAttnAdapter",
    "load_adapter_checkpoint",
    "load_adapter_checkpoint_metadata",
    "save_adapter_checkpoint_metadata",
    "train_frozen_base_adapter",
]
