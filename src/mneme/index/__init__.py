"""Index backend public API."""

from mneme.index._faiss_hnsw import FaissHnswIndex, create_index_backend
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
    "FaissHnswIndex",
    "FilterPredicate",
    "FlatIndex",
    "Index",
    "apply_temporal_decay",
    "create_index_backend",
    "deduplicate_results",
    "planned_search_k",
    "search_index",
]
