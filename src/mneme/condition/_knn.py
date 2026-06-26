"""Training-free kNN conditioner."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, cast

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
from mneme.observability import (
    ObservabilityConfig,
    distance_mean,
    distance_min,
    emit_event,
    start_event_timer,
)

KnnMode = Literal["delta", "absolute"]
LatentBackend = Literal["numpy", "torch"]


@dataclass(frozen=True)
class _LatentView:
    array: np.ndarray
    original: object
    backend: LatentBackend
    dtype: object
    device: object | None


@dataclass(frozen=True)
class KnnCorrector:
    """Distance-weighted nonparametric corrector for retrieved transitions."""

    tau: float = 0.1
    lambda_max: float = 0.5
    alpha: float = 10.0
    delta0: float = 0.2
    mode: KnnMode = "delta"
    observability: ObservabilityConfig | None = None

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

        started = start_event_timer(self.observability)
        try:
            if not retrieval.items:
                if started is not None:
                    emit_event(
                        self.observability,
                        event="mneme.condition.apply",
                        operation="condition.apply",
                        status="ok",
                        started=started,
                        mode=self.mode,
                        empty_retrieval=True,
                        hit_count=0,
                        gate_lambda=0.0,
                        output_finite=_latent_is_finite(parametric),
                    )
                return parametric
            parametric_view = _require_latent(parametric, "parametric")
            with _inference_mode(parametric_view):
                parametric_array = parametric_view.array
                distances = _require_distances(retrieval)
                weights = _softmax(-distances / self.tau)
                if self.mode == "delta":
                    z_knn = self._delta_prediction(
                        parametric_view, retrieval, ctx, weights
                    )
                else:
                    z_knn = self._absolute_prediction(
                        parametric_view, retrieval, weights
                    )
                gate = self.gate(float(distances.min()))
                blended = (1.0 - gate) * parametric_array + gate * z_knn
                result_array = np.ascontiguousarray(
                    blended, dtype=parametric_array.dtype
                )
                result = _restore_backend(result_array, parametric_view)
        except Exception as exc:
            if started is not None:
                emit_event(
                    self.observability,
                    event="mneme.condition.apply",
                    operation="condition.apply",
                    status="error",
                    started=started,
                    error=exc,
                    mode=self.mode,
                    empty_retrieval=_safe_empty_retrieval(retrieval),
                    hit_count=_safe_retrieval_len(retrieval),
                )
            raise
        if started is not None:
            distance_values = tuple(float(item) for item in distances)
            emit_event(
                self.observability,
                event="mneme.condition.apply",
                operation="condition.apply",
                status="ok",
                started=started,
                mode=self.mode,
                empty_retrieval=False,
                hit_count=len(retrieval.items),
                distance_min=distance_min(distance_values),
                distance_mean=distance_mean(distance_values),
                gate_lambda=gate,
                output_finite=_latent_is_finite(result),
            )
        return result

    def gate(self, nearest_distance: float) -> float:
        """Return the memory interpolation weight for a nearest-neighbor distance."""

        distance = _require_non_negative_finite(nearest_distance, "nearest_distance")
        return self.lambda_max * _sigmoid(self.alpha * (self.delta0 - distance))

    def _delta_prediction(
        self,
        parametric: _LatentView,
        retrieval: Retrieval,
        ctx: CondCtx,
        weights: np.ndarray,
    ) -> np.ndarray:
        if ctx.current_latent is None:
            raise ValidationError("delta mode requires CondCtx.current_latent")
        current = _require_latent(ctx.current_latent, "ctx.current_latent").array
        _require_same_shape(current, parametric.array, "ctx.current_latent")
        deltas = [
            _transition_array(item.value, "delta", parametric.array)
            for item in retrieval.items
        ]
        return cast(np.ndarray, current + _weighted_sum(deltas, weights))

    def _absolute_prediction(
        self,
        parametric: _LatentView,
        retrieval: Retrieval,
        weights: np.ndarray,
    ) -> np.ndarray:
        successors = [
            _transition_array(item.value, "z_next", parametric.array)
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
    array = _require_latent(
        getattr(value, field_name), f"transition.{field_name}"
    ).array
    _require_same_shape(array, parametric, f"transition.{field_name}")
    return array


def _require_latent(value: object, field_name: str) -> _LatentView:
    if isinstance(value, np.ndarray):
        array = _require_numpy_array(value, field_name)
        return _LatentView(
            array=np.asarray(array),
            original=value,
            backend="numpy",
            dtype=value.dtype,
            device=None,
        )
    if _is_torch_tensor(value):
        array = _tensor_to_numpy(value, field_name)
        return _LatentView(
            array=array,
            original=value,
            backend="torch",
            dtype=getattr(value, "dtype", None),
            device=getattr(value, "device", None),
        )
    raise DTypeError(f"{field_name} must be a numpy.ndarray or torch.Tensor")


def _require_numpy_array(value: np.ndarray, field_name: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.floating):
        raise DTypeError(f"{field_name} must have a floating dtype")
    if value.shape == ():
        raise ShapeError(f"{field_name} must have at least one dimension")
    if any(dim <= 0 for dim in value.shape):
        raise ShapeError(f"{field_name} dimensions must be positive")
    if not bool(np.isfinite(value).all()):
        raise ValidationError(f"{field_name} must contain only finite values")
    return np.asarray(value)


def _is_torch_tensor(value: object) -> bool:
    if type(value).__module__.split(".", 1)[0] != "torch":
        return False
    return all(
        hasattr(value, attr)
        for attr in ("detach", "cpu", "numpy", "shape", "dtype", "to")
    )


def _tensor_to_numpy(value: object, field_name: str) -> np.ndarray:
    dtype = str(getattr(value, "dtype", ""))
    if "float" not in dtype:
        raise DTypeError(f"{field_name} must have a floating dtype")
    current = cast(Any, value)
    current = current.detach()
    current = current.cpu()
    converted = current.numpy()
    if not isinstance(converted, np.ndarray):
        raise DTypeError(f"{field_name} tensor did not convert to numpy.ndarray")
    return np.ascontiguousarray(_require_numpy_array(converted, field_name))


def _restore_backend(array: np.ndarray, template: _LatentView) -> Latent:
    if template.backend == "numpy":
        return np.ascontiguousarray(array, dtype=cast(np.dtype[Any], template.dtype))
    new_tensor = getattr(template.original, "new_tensor", None)
    if callable(new_tensor):
        result = new_tensor(array)
    else:
        torch = _import_torch()
        result = torch.as_tensor(array)
    to = getattr(result, "to", None)
    if callable(to):
        return to(dtype=template.dtype, device=template.device)
    return result


def _inference_mode(template: _LatentView) -> Any:
    if template.backend != "torch":
        return nullcontext()
    inference_mode = getattr(_import_torch(), "inference_mode", None)
    if callable(inference_mode):
        return inference_mode()
    return nullcontext()


def _import_torch() -> Any:
    import importlib

    return importlib.import_module("torch")


def _require_same_shape(
    value: np.ndarray, expected: np.ndarray, field_name: str
) -> None:
    if value.shape != expected.shape:
        raise ShapeError(
            f"{field_name} shape {value.shape} does not match parametric shape "
            f"{expected.shape}"
        )


def _require_distances(retrieval: Retrieval) -> np.ndarray:
    try:
        if any(
            isinstance(distance, (bool, np.bool_)) for distance in retrieval.distances
        ):
            raise ValidationError("retrieval distances must be finite numbers")
        distances = np.asarray(retrieval.distances, dtype=np.float64)
    except ValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ValidationError("retrieval distances must be finite numbers") from exc
    if distances.shape != (len(retrieval.items),):
        raise ShapeError("retrieval distances must match retrieval items")
    if not bool(np.isfinite(distances).all()):
        raise ValidationError("retrieval distances must be finite")
    if bool((distances < 0.0).any()):
        raise ValidationError("retrieval distances must be non-negative")
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


def _latent_is_finite(value: object) -> bool | None:
    try:
        view = _require_latent(value, "latent")
    except Exception:
        return None
    return bool(np.isfinite(view.array).all())


def _safe_empty_retrieval(retrieval: object) -> bool | None:
    items = getattr(retrieval, "items", None)
    if items is None:
        return None
    return not bool(items)


def _safe_retrieval_len(retrieval: object) -> int | None:
    items = getattr(retrieval, "items", None)
    if items is None:
        return None
    try:
        return len(items)
    except TypeError:
        return None


__all__ = ["KnnCorrector", "KnnMode"]
