"""Index backend public API."""

from mneme.index._flat import FlatIndex
from mneme.index._protocols import Index

__all__ = ["FlatIndex", "Index"]
