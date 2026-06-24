"""Training-free kNN conditioner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from mneme.condition._protocols import CondCtx
from mneme.core import (
    DTypeError,
    Latent,
    Retrieval,
    ShapeError,
    Transition,
    ValidationError,
)

KnnMode = Literal["delta", "absolute"]


@dataclass(frozen=True)
class KnnCorrector:
    """Distance-weighted nonparametric corrector for retrieved transitions."""

    tau: float = 0.1
    lambda_max: float = 0.5
    alpha: float = 10.0
    delta0: float = 0.2
    mode: KnnMode = "delta"

    def __post_init__(self) -> None:
        _require_positive_finite(self.tau, "tau")
        _require_probability(self.lambda_max, "lambda_max")
        _require_non_negative_finite(self.alpha, "alpha")
        _require_finite(self.delta0, "delta0")
        if self.mode not in {"delta", "absolute"}:
            raise ValidationError("mode must be 'delta' or 'absolute'")

    def condition(
        self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx
    ) -> Latent:
        """Blend a parametric prediction with retrieved transition evidence."""

        if not retrieval.items:
            return parametric
        parametric_array = _require_array(parametric, "parametric")
        distances = _require_distances(retrieval)
        weights = _softmax(-distances / self.tau)
        if self.mode == "delta":
            z_knn = self._delta_prediction(parametric_array, retrieval, ctx, weights)
        else:
            z_knn = self._absolute_prediction(parametric_array, retrieval, weights)
        gate = self.gate(float(distances.min()))
        blended = (1.0 - gate) * parametric_array + gate * z_knn
        return np.ascontiguousarray(blended, dtype=parametric_array.dtype)

    def gate(self, nearest_distance: float) -> float:
        """Return the memory interpolation weight for a nearest-neighbor distance."""

        distance = _require_non_negative_finite(nearest_distance, "nearest_distance")
        return self.lambda_max * _sigmoid(self.alpha * (self.delta0 - distance))

    def _delta_prediction(
        self,
        parametric: np.ndarray,
        retrieval: Retrieval,
        ctx: CondCtx,
        weights: np.ndarray,
    ) -> np.ndarray:
        if ctx.current_latent is None:
            raise ValidationError("delta mode requires CondCtx.current_latent")
        current = _require_array(ctx.current_latent, "ctx.current_latent")
        _require_same_shape(current, parametric, "ctx.current_latent")
        deltas = [
            _transition_array(item.value, "delta", parametric)
            for item in retrieval.items
        ]
        return cast(np.ndarray, current + _weighted_sum(deltas, weights))

    def _absolute_prediction(
        self,
        parametric: np.ndarray,
        retrieval: Retrieval,
        weights: np.ndarray,
    ) -> np.ndarray:
        successors = [
            _transition_array(item.value, "z_next", parametric)
            for item in retrieval.items
        ]
        return _weighted_sum(successors, weights)


def _transition_array(
    value: object,
    field_name: Literal["delta", "z_next"],
    parametric: np.ndarray,
) -> np.ndarray:
    if not isinstance(value, Transition):
        raise ValidationError("retrieval values must be Transition instances")
    array = _require_array(getattr(value, field_name), f"transition.{field_name}")
    _require_same_shape(array, parametric, f"transition.{field_name}")
    return array


def _require_array(value: object, field_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise DTypeError(f"{field_name} must be a numpy.ndarray")
    if not np.issubdtype(value.dtype, np.floating):
        raise DTypeError(f"{field_name} must have a floating dtype")
    if value.shape == ():
        raise ShapeError(f"{field_name} must have at least one dimension")
    if any(dim <= 0 for dim in value.shape):
        raise ShapeError(f"{field_name} dimensions must be positive")
    if not bool(np.isfinite(value).all()):
        raise ValidationError(f"{field_name} must contain only finite values")
    return np.asarray(value)


def _require_same_shape(
    value: np.ndarray, expected: np.ndarray, field_name: str
) -> None:
    if value.shape != expected.shape:
        raise ShapeError(
            f"{field_name} shape {value.shape} does not match parametric shape "
            f"{expected.shape}"
        )


def _require_distances(retrieval: Retrieval) -> np.ndarray:
    distances = np.asarray(retrieval.distances, dtype=np.float64)
    if distances.shape != (len(retrieval.items),):
        raise ShapeError("retrieval distances must match retrieval items")
    if not bool(np.isfinite(distances).all()):
        raise ValidationError("retrieval distances must be finite")
    return distances


def _weighted_sum(values: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(values, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    total = weights.sum()
    if not math.isfinite(float(total)) or float(total) <= 0.0:
        raise ValidationError("distance weights are not finite")
    return cast(np.ndarray, weights / total)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _require_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field_name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValidationError(f"{field_name} must be a finite number")
    return converted


def _require_positive_finite(value: object, field_name: str) -> None:
    converted = _require_finite(value, field_name)
    if converted <= 0.0:
        raise ValidationError(f"{field_name} must be positive")


def _require_non_negative_finite(value: object, field_name: str) -> float:
    converted = _require_finite(value, field_name)
    if converted < 0.0:
        raise ValidationError(f"{field_name} must be non-negative")
    return converted


def _require_probability(value: object, field_name: str) -> None:
    converted = _require_finite(value, field_name)
    if converted < 0.0 or converted > 1.0:
        raise ValidationError(f"{field_name} must be between 0 and 1")


__all__ = ["KnnCorrector", "KnnMode"]
