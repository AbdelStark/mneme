"""In-context retrieved-token conditioning baseline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast, runtime_checkable

import numpy as np

from mneme.condition._knn import (
    _latent_is_finite,
    _require_latent,
    _require_same_shape,
    _safe_empty_retrieval,
    _safe_retrieval_len,
)
from mneme.condition._protocols import CondCtx
from mneme.core import Latent, Retrieval, Transition, ValidationError
from mneme.observability import (
    ObservabilityConfig,
    emit_event,
    start_event_timer,
)


@runtime_checkable
class InContextPredictor(Protocol):
    """Predictor wrapper compatible with the in-context baseline."""

    def predict_with_context(
        self,
        parametric: Latent,
        retrieved_tokens: Sequence[Latent],
        ctx: CondCtx,
    ) -> Latent:
        """Return a prediction after appending retrieved value tokens."""
        ...


InContextCallable: TypeAlias = Callable[[Latent, Sequence[Latent], CondCtx], Latent]
InContextPredictorLike: TypeAlias = InContextPredictor | InContextCallable


@dataclass(frozen=True)
class InContextConditioner:
    """Append retrieved successor latents as context tokens for a predictor.

    This is a comparison baseline for predictor wrappers that can accept extra
    retrieved tokens. It does not train parameters or become the default
    conditioner because self-attention cost grows with the number of retrieved
    tokens.
    """

    predictor: InContextPredictorLike
    max_tokens: int | None = None
    observability: ObservabilityConfig | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValidationError("max_tokens must be None or a positive integer")
        if not _can_predict_with_context(self.predictor):
            raise ValidationError(
                "predictor must expose predict_with_context(...) or be callable"
            )

    def condition(
        self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx
    ) -> Latent:
        """Return the predictor output with retrieved value tokens appended."""

        started = start_event_timer(self.observability)
        token_count = 0
        try:
            if not retrieval.items:
                if started is not None:
                    emit_event(
                        self.observability,
                        event="mneme.condition.apply",
                        operation="condition.apply",
                        status="ok",
                        started=started,
                        mode="in_context",
                        empty_retrieval=True,
                        hit_count=0,
                        retrieved_context_count=0,
                        output_finite=_latent_is_finite(parametric),
                    )
                return parametric
            parametric_view = _require_latent(parametric, "parametric")
            tokens = _retrieved_value_tokens(
                retrieval,
                parametric_view.array,
                max_tokens=self.max_tokens,
            )
            token_count = len(tokens)
            result = _call_predictor(self.predictor, parametric, tokens, ctx)
            result_view = _require_latent(result, "predictor result")
            _require_same_shape(
                result_view.array,
                parametric_view.array,
                "predictor result",
            )
        except Exception as exc:
            if started is not None:
                emit_event(
                    self.observability,
                    event="mneme.condition.apply",
                    operation="condition.apply",
                    status="error",
                    started=started,
                    error=exc,
                    mode="in_context",
                    empty_retrieval=_safe_empty_retrieval(retrieval),
                    hit_count=_safe_retrieval_len(retrieval),
                    retrieved_context_count=token_count,
                )
            raise
        if started is not None:
            emit_event(
                self.observability,
                event="mneme.condition.apply",
                operation="condition.apply",
                status="ok",
                started=started,
                mode="in_context",
                empty_retrieval=False,
                hit_count=len(retrieval.items),
                retrieved_context_count=token_count,
                output_finite=_latent_is_finite(result),
            )
        return result


def _retrieved_value_tokens(
    retrieval: Retrieval,
    parametric: np.ndarray,
    *,
    max_tokens: int | None,
) -> tuple[Latent, ...]:
    selected = retrieval.items if max_tokens is None else retrieval.items[:max_tokens]
    tokens: list[Latent] = []
    for index, item in enumerate(selected):
        value = item.value
        if not isinstance(value, Transition):
            raise ValidationError("retrieval values must be Transition instances")
        token = value.z_next
        token_view = _require_latent(
            token,
            f"retrieval.items[{index}].value.z_next",
        )
        _require_same_shape(
            token_view.array,
            parametric,
            f"retrieval.items[{index}].value.z_next",
        )
        tokens.append(token)
    return tuple(tokens)


def _can_predict_with_context(predictor: object) -> bool:
    method = getattr(predictor, "predict_with_context", None)
    return callable(method) or callable(predictor)


def _call_predictor(
    predictor: InContextPredictorLike,
    parametric: Latent,
    retrieved_tokens: Sequence[Latent],
    ctx: CondCtx,
) -> Latent:
    method = getattr(predictor, "predict_with_context", None)
    if callable(method):
        return method(parametric, retrieved_tokens, ctx)
    call = cast(InContextCallable, predictor)
    return call(parametric, retrieved_tokens, ctx)


__all__ = ["InContextConditioner", "InContextPredictor"]
