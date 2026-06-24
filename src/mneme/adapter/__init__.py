"""Optional trained memory adapter modules."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "CrossAttnAdapter":
        from mneme.adapter._cross_attention import CrossAttnAdapter

        return CrossAttnAdapter
    if name == "AdapterTrainingBatch":
        from mneme.adapter._training import AdapterTrainingBatch

        return AdapterTrainingBatch
    if name == "train_frozen_base_adapter":
        from mneme.adapter._training import train_frozen_base_adapter

        return train_frozen_base_adapter
    raise AttributeError(f"module 'mneme.adapter' has no attribute {name!r}")


__all__ = ["AdapterTrainingBatch", "CrossAttnAdapter", "train_frozen_base_adapter"]
