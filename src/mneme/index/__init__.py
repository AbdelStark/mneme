"""Index backend public API."""

from mneme.index._flat import FlatIndex
from mneme.index._protocols import Index
from mneme.index._query import (
    FilterPredicate,
    apply_temporal_decay,
    deduplicate_results,
    planned_search_k,
    search_index,
)

__all__ = [
    "FilterPredicate",
    "FlatIndex",
    "Index",
    "apply_temporal_decay",
    "deduplicate_results",
    "planned_search_k",
    "search_index",
]
