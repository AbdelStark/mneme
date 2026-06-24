"""PyTorch cross-attention memory adapter."""

from __future__ import annotations

import importlib
from typing import Any

from mneme.core import DTypeError, OptionalDependencyError, ShapeError, ValidationError

_TORCH = None


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ImportError as exc:
            raise OptionalDependencyError(
                "CrossAttnAdapter requires the 'ml' extra",
                extra="ml",
                package="torch",
            ) from exc
    return _TORCH


_NN = _torch().nn


class CrossAttnAdapter(_NN.Module):  # type: ignore[misc, name-defined]
    """Stacked cross-attention adapter over retrieved latent values.

    The adapter is independent from any external predictor: callers provide the
    predictor hidden states and retrieved value latents directly.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _require_positive_int(latent_dim, "latent_dim")
        _require_positive_int(hidden_dim, "hidden_dim")
        _require_positive_int(num_heads, "num_heads")
        _require_positive_int(num_layers, "num_layers")
        _require_dropout(dropout)
        if hidden_dim % num_heads != 0:
            raise ValidationError("hidden_dim must be divisible by num_heads")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = float(dropout)
        self.value_projection = _NN.Linear(latent_dim, hidden_dim)
        self.layers = _NN.ModuleList(
            [_CrossAttnBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_norm = _NN.LayerNorm(hidden_dim)

    def forward(
        self,
        predictor_hidden: Any,
        retrieved_values: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        """Return hidden states updated by cross-attending to retrieved values."""

        hidden_shape = _require_rank3_tensor(predictor_hidden, "predictor_hidden")
        value_shape = _require_rank3_tensor(retrieved_values, "retrieved_values")
        if hidden_shape[0] != value_shape[0]:
            raise ShapeError(
                "retrieved_values batch size must match predictor_hidden batch size"
            )
        if hidden_shape[2] != self.hidden_dim:
            raise ShapeError(
                "predictor_hidden last dimension must match adapter hidden_dim"
            )
        if value_shape[2] != self.latent_dim:
            raise ShapeError(
                "retrieved_values last dimension must match adapter latent_dim"
            )

        device = getattr(predictor_hidden, "device", None)
        dtype = getattr(predictor_hidden, "dtype", None)
        retrieved = _move_tensor(
            retrieved_values,
            field_name="retrieved_values",
            dtype=dtype,
            device=device,
        )
        memory = self.value_projection(retrieved)
        key_padding_mask = _key_padding_mask(
            attention_mask,
            batch_size=hidden_shape[0],
            value_count=value_shape[1],
            device=device,
        )
        hidden = predictor_hidden
        for layer in self.layers:
            hidden = layer(hidden, memory, key_padding_mask)
        return self.output_norm(hidden)


class _CrossAttnBlock(_NN.Module):  # type: ignore[misc, name-defined]
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = _NN.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = _NN.Dropout(dropout)
        self.attention_norm = _NN.LayerNorm(hidden_dim)
        self.feed_forward = _NN.Sequential(
            _NN.Linear(hidden_dim, hidden_dim * 4),
            _NN.GELU(),
            _NN.Dropout(dropout),
            _NN.Linear(hidden_dim * 4, hidden_dim),
        )
        self.feed_forward_norm = _NN.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Any,
        memory: Any,
        key_padding_mask: Any | None,
    ) -> Any:
        attended, _ = self.attention(
            query=hidden,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + self.dropout(attended))
        updated = self.feed_forward(hidden)
        return self.feed_forward_norm(hidden + self.dropout(updated))


def _require_rank3_tensor(value: object, field_name: str) -> tuple[int, int, int]:
    if not _is_torch_tensor(value):
        raise DTypeError(f"{field_name} must be a torch.Tensor")
    shape = _shape_tuple(value, field_name)
    if len(shape) != 3:
        raise ShapeError(f"{field_name} must have shape (batch, tokens, dim)")
    if any(dim <= 0 for dim in shape):
        raise ShapeError(f"{field_name} dimensions must be positive")
    return shape


def _is_torch_tensor(value: object) -> bool:
    is_tensor = getattr(_torch(), "is_tensor", None)
    if callable(is_tensor):
        return bool(is_tensor(value))
    return type(value).__module__.split(".", 1)[0] == "torch"


def _shape_tuple(value: object, field_name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ShapeError(f"{field_name} must expose shape")
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise ShapeError(f"{field_name} shape must be an integer sequence") from exc


def _move_tensor(
    value: object,
    *,
    field_name: str,
    dtype: object | None,
    device: object | None,
) -> Any:
    to = getattr(value, "to", None)
    if not callable(to):
        raise DTypeError(f"{field_name} must support torch.Tensor.to")
    kwargs: dict[str, object] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    if not kwargs:
        return value
    return to(**kwargs)


def _key_padding_mask(
    attention_mask: object | None,
    *,
    batch_size: int,
    value_count: int,
    device: object | None,
) -> Any | None:
    if attention_mask is None:
        return None
    if not _is_torch_tensor(attention_mask):
        raise DTypeError("attention_mask must be a torch.Tensor")
    if _shape_tuple(attention_mask, "attention_mask") != (batch_size, value_count):
        raise ShapeError("attention_mask must have shape (batch, retrieved)")
    valid_mask = _move_tensor(
        attention_mask,
        field_name="attention_mask",
        dtype=getattr(_torch(), "bool", None),
        device=device,
    )
    return ~valid_mask


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{field_name} must be a positive integer")


def _require_dropout(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError("dropout must be a number between 0 and 1")
    converted = float(value)
    if converted < 0.0 or converted > 1.0:
        raise ValidationError("dropout must be between 0 and 1")


__all__ = ["CrossAttnAdapter"]
