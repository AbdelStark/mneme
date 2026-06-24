"""Index backend protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mneme.core import Cid, Metric, SummaryVec


@runtime_checkable
class Index(Protocol):
    """Backend-neutral index contract."""

    def add(self, cid: Cid, key: SummaryVec) -> None:
        """Add or replace one key by content id."""
        ...

    def add_batch(self, items: Sequence[tuple[Cid, SummaryVec]]) -> None:
        """Add or replace multiple keys."""
        ...

    def search(
        self,
        q: SummaryVec,
        k: int,
        *,
        metric: Metric,
        ef: int | None = None,
    ) -> list[tuple[Cid, float]]:
        """Return ids and distances or scores in stable order."""
        ...

    def __len__(self) -> int:
        """Return number of indexed ids."""
        ...


__all__ = ["Index"]
