from __future__ import annotations

from math import log
from uuid import uuid4

import numpy as np
import pytest

from mneme.condition import CondCtx, Conditioner, KnnCorrector
from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Retrieval,
    ShapeError,
    Transition,
    ValidationError,
)


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _item(
    delta: np.ndarray,
    z_next: np.ndarray,
    *,
    key_value: float = 1.0,
) -> MemoryItem:
    z_src = np.zeros_like(delta)
    return MemoryItem(
        content_id=None,
        key=np.array([key_value, 0.0], dtype=np.float32),
        value=Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_next,
            delta=delta,
            t=1,
            episode_id=uuid4(),
        ),
        meta={},
        encoder_fp=_fingerprint(),
    )


def _retrieval() -> Retrieval:
    return Retrieval(
        items=(
            _item(
                np.array([2.0, 0.0], dtype=np.float32),
                np.array([4.0, 0.0], dtype=np.float32),
            ),
            _item(
                np.array([0.0, 2.0], dtype=np.float32),
                np.array([0.0, 4.0], dtype=np.float32),
                key_value=2.0,
            ),
        ),
        distances=(0.0, log(3.0)),
    )


def test_empty_retrieval_returns_parametric_exactly() -> None:
    corrector = KnnCorrector()
    parametric = np.array([3.0, 4.0], dtype=np.float32)

    result = corrector.condition(parametric, Retrieval((), ()), CondCtx(None))

    assert result is parametric


def test_delta_mode_matches_hand_computed_fixture() -> None:
    corrector = KnnCorrector(tau=1.0, lambda_max=1.0, alpha=0.0, mode="delta")
    parametric = np.array([10.0, 10.0], dtype=np.float32)
    ctx = CondCtx(current_latent=np.array([1.0, 1.0], dtype=np.float32))

    result = corrector.condition(parametric, _retrieval(), ctx)

    np.testing.assert_allclose(
        result,
        np.array([6.25, 5.75], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_absolute_mode_matches_hand_computed_fixture() -> None:
    corrector = KnnCorrector(tau=1.0, lambda_max=1.0, alpha=0.0, mode="absolute")
    parametric = np.array([10.0, 10.0], dtype=np.float32)

    result = corrector.condition(parametric, _retrieval(), CondCtx(None))

    np.testing.assert_allclose(
        result,
        np.array([6.5, 5.5], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_knn_corrector_satisfies_conditioner_protocol() -> None:
    assert isinstance(KnnCorrector(), Conditioner)


def test_invalid_retrieval_distances_raise_typed_error() -> None:
    corrector = KnnCorrector()
    retrieval = object.__new__(Retrieval)
    object.__setattr__(retrieval, "items", _retrieval().items)
    object.__setattr__(retrieval, "distances", (float("nan"), 0.0))
    object.__setattr__(retrieval, "receipt", None)
    object.__setattr__(retrieval, "schema_version", "mneme.retrieval.v1")

    with pytest.raises(ValidationError, match="distances"):
        corrector.condition(
            np.array([1.0, 0.0], dtype=np.float32),
            retrieval,
            CondCtx(np.array([1.0, 0.0], dtype=np.float32)),
        )


def test_invalid_retrieval_value_shape_raises_typed_error() -> None:
    corrector = KnnCorrector()
    retrieval = Retrieval(
        items=(
            _item(
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([2.0, 0.0, 0.0], dtype=np.float32),
            ),
        ),
        distances=(0.0,),
    )

    with pytest.raises(ShapeError, match="transition.delta shape"):
        corrector.condition(
            np.array([1.0, 0.0], dtype=np.float32),
            retrieval,
            CondCtx(np.array([1.0, 0.0], dtype=np.float32)),
        )


def test_delta_mode_requires_current_latent() -> None:
    corrector = KnnCorrector(mode="delta")

    with pytest.raises(ValidationError, match="current_latent"):
        corrector.condition(
            np.array([1.0, 0.0], dtype=np.float32),
            _retrieval(),
            CondCtx(None),
        )


def test_invalid_parameters_raise_typed_errors() -> None:
    with pytest.raises(ValidationError, match="tau"):
        KnnCorrector(tau=0.0)
    with pytest.raises(ValidationError, match="lambda_max"):
        KnnCorrector(lambda_max=1.5)
    with pytest.raises(ValidationError, match="mode"):
        KnnCorrector(mode="bad")  # type: ignore[arg-type]
