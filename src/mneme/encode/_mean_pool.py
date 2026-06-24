"""Deterministic mean-pool summary-key generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mneme.core import (
    DTypeError,
    Latent,
    ShapeError,
    SummaryVec,
    UnsupportedOperationError,
    ValidationError,
)


@dataclass(frozen=True)
class MeanPoolSummarizer:
    """Mean-pool latent values into one-dimensional summary keys."""

    normalize: bool = True
    output_dim: int | None = None
    id: str = "meanpool-v1"

    def __post_init__(self) -> None:
        if self.output_dim is not None:
            raise UnsupportedOperationError(
                "deterministic projection is deferred until v0.2"
            )
        if not isinstance(self.normalize, bool):
            raise TypeError("normalize must be a bool")
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")

    def summarize(self, z: Latent) -> SummaryVec:
        """Return a contiguous finite float32 summary vector."""

        array = _as_numpy_array(z)
        if not np.issubdtype(array.dtype, np.number):
            raise DTypeError("latent must have a numeric dtype")
        if array.ndim == 0:
            raise ShapeError("latent must have at least one dimension")
        if any(int(dim) <= 0 for dim in array.shape):
            raise ShapeError("latent shape dimensions must be positive")
        if not bool(np.isfinite(array).all()):
            raise ValidationError("latent must contain only finite values")

        if array.ndim == 1:
            pooled = array
        else:
            pooled = array.reshape(-1, array.shape[-1]).mean(axis=0)
        summary = np.ascontiguousarray(pooled, dtype=np.float32)

        if self.normalize:
            norm = float(np.linalg.norm(summary))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValidationError("cannot normalize a zero or non-finite summary")
            summary = np.ascontiguousarray(summary / norm, dtype=np.float32)

        if summary.ndim != 1 or summary.shape[0] <= 0:
            raise ShapeError("summary must be one-dimensional and non-empty")
        if not bool(np.isfinite(summary).all()):
            raise ValidationError("summary must contain only finite values")
        return summary


def _as_numpy_array(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
        if isinstance(converted, np.ndarray):
            return converted
    raise DTypeError("latent must be a numpy array or tensor-like object")


__all__ = ["MeanPoolSummarizer"]
