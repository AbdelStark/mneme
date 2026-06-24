from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import numpy as np
import pytest

from mneme.condition import (
    CondCtx,
    Conditioner,
    InContextConditioner,
    InContextPredictor,
)
from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Retrieval,
    ShapeError,
    Transition,
    ValidationError,
)
from mneme.observability import ObservabilityConfig


class RecordingPredictor:
    def __init__(self, result: np.ndarray | None = None) -> None:
        self.result = result
        self.calls: list[tuple[object, tuple[object, ...], CondCtx]] = []

    def predict_with_context(
        self,
        parametric: object,
        retrieved_tokens: Sequence[object],
        ctx: CondCtx,
    ) -> object:
        tokens = tuple(retrieved_tokens)
        self.calls.append((parametric, tokens, ctx))
        if self.result is not None:
            return self.result
        return np.asarray(parametric, dtype=np.float32) + np.asarray(
            tokens[0], dtype=np.float32
        )


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, event: object) -> None:
        assert isinstance(event, dict)
        self.events.append(event)


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _item(z_next: np.ndarray, *, key_value: float = 1.0) -> MemoryItem:
    z_src = np.zeros_like(z_next)
    return MemoryItem(
        content_id=None,
        key=np.array([key_value, 0.0], dtype=np.float32),
        value=Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_next,
            delta=z_next - z_src,
            t=1,
            episode_id=uuid4(),
        ),
        meta={},
        encoder_fp=_fingerprint(),
    )


def _retrieval(
    first: np.ndarray | None = None,
    second: np.ndarray | None = None,
) -> Retrieval:
    z_next_0 = np.array([4.0, 0.0], dtype=np.float32) if first is None else first
    z_next_1 = np.array([0.0, 4.0], dtype=np.float32) if second is None else second
    return Retrieval(
        items=(
            _item(z_next_0),
            _item(z_next_1, key_value=2.0),
        ),
        distances=(0.0, 0.25),
    )


def test_in_context_conditioner_calls_fixture_predictor_with_z_next_tokens() -> None:
    z_next_0 = np.array([4.0, 0.0], dtype=np.float32)
    z_next_1 = np.array([0.0, 4.0], dtype=np.float32)
    retrieval = _retrieval(z_next_0, z_next_1)
    parametric = np.array([10.0, 10.0], dtype=np.float32)
    ctx = CondCtx(current_latent=np.array([1.0, 1.0], dtype=np.float32))
    predictor = RecordingPredictor()
    conditioner = InContextConditioner(predictor)

    result = conditioner.condition(parametric, retrieval, ctx)

    assert isinstance(conditioner, Conditioner)
    assert isinstance(predictor, InContextPredictor)
    np.testing.assert_allclose(result, np.array([14.0, 10.0], dtype=np.float32))
    assert len(predictor.calls) == 1
    called_parametric, tokens, called_ctx = predictor.calls[0]
    assert called_parametric is parametric
    assert called_ctx is ctx
    assert tokens[0] is z_next_0
    assert tokens[1] is z_next_1


def test_empty_retrieval_returns_parametric_without_calling_predictor() -> None:
    parametric = np.array([3.0, 4.0], dtype=np.float32)
    predictor = RecordingPredictor()

    result = InContextConditioner(predictor).condition(
        parametric,
        Retrieval((), ()),
        CondCtx(None),
    )

    assert result is parametric
    assert predictor.calls == []


def test_max_tokens_limits_appended_retrieved_context() -> None:
    parametric = np.array([10.0, 10.0], dtype=np.float32)
    predictor = RecordingPredictor()

    InContextConditioner(predictor, max_tokens=1).condition(
        parametric,
        _retrieval(),
        CondCtx(None),
    )

    assert len(predictor.calls[0][1]) == 1


def test_callable_predictor_wrapper_is_supported() -> None:
    calls: list[tuple[object, tuple[object, ...], CondCtx]] = []

    def predictor(
        parametric: object,
        retrieved_tokens: Sequence[object],
        ctx: CondCtx,
    ) -> object:
        tokens = tuple(retrieved_tokens)
        calls.append((parametric, tokens, ctx))
        return np.asarray(parametric, dtype=np.float32)

    parametric = np.array([1.0, 2.0], dtype=np.float32)

    result = InContextConditioner(predictor).condition(
        parametric,
        _retrieval(),
        CondCtx(None),
    )

    np.testing.assert_array_equal(result, parametric)
    assert len(calls[0][1]) == 2


def test_retrieved_token_shape_must_match_parametric_prediction() -> None:
    conditioner = InContextConditioner(RecordingPredictor())

    with pytest.raises(ShapeError, match="value.z_next shape"):
        conditioner.condition(
            np.array([1.0, 0.0], dtype=np.float32),
            Retrieval(
                items=(_item(np.array([1.0, 0.0, 0.0], dtype=np.float32)),),
                distances=(0.0,),
            ),
            CondCtx(None),
        )


def test_predictor_result_shape_must_match_parametric_prediction() -> None:
    conditioner = InContextConditioner(
        RecordingPredictor(result=np.zeros((3,), dtype=np.float32))
    )

    with pytest.raises(ShapeError, match="predictor result shape"):
        conditioner.condition(
            np.array([1.0, 0.0], dtype=np.float32),
            _retrieval(),
            CondCtx(None),
        )


def test_invalid_configuration_and_incompatible_predictor_raise_typed_errors() -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        InContextConditioner(RecordingPredictor(), max_tokens=0)
    with pytest.raises(ValidationError, match="predictor"):
        InContextConditioner(object())  # type: ignore[arg-type]


def test_in_context_observability_records_baseline_context_size() -> None:
    sink = RecordingSink()
    conditioner = InContextConditioner(
        RecordingPredictor(),
        observability=ObservabilityConfig(event_sink=sink),
    )

    conditioner.condition(
        np.array([1.0, 2.0], dtype=np.float32),
        _retrieval(),
        CondCtx(None),
    )

    assert sink.events[-1]["event"] == "mneme.condition.apply"
    assert sink.events[-1]["mode"] == "in_context"
    assert sink.events[-1]["empty_retrieval"] is False
    assert sink.events[-1]["hit_count"] == 2
    assert sink.events[-1]["retrieved_context_count"] == 2
