from __future__ import annotations

import json
from math import log
from pathlib import Path
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

_FIXTURE_PATH = Path("tests/fixtures/condition/gate_behavior_cases.json")


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


def test_distance_gate_near_and_far_cases() -> None:
    corrector = KnnCorrector()

    near_gate = corrector.gate(0.0)
    far_gate = corrector.gate(10.0)

    assert near_gate > 0.1
    assert near_gate <= corrector.lambda_max
    assert far_gate < 1e-6


def test_gate_rejects_non_finite_distances() -> None:
    corrector = KnnCorrector()

    with pytest.raises(ValidationError, match="nearest_distance"):
        corrector.gate(float("nan"))
    with pytest.raises(ValidationError, match="nearest_distance"):
        corrector.gate(float("inf"))


def test_non_finite_parametric_and_transition_values_raise_validation_error() -> None:
    corrector = KnnCorrector()

    with pytest.raises(ValidationError, match="parametric"):
        corrector.condition(
            np.array([float("nan"), 0.0], dtype=np.float32),
            _retrieval(),
            CondCtx(np.array([1.0, 0.0], dtype=np.float32)),
        )

    retrieval = Retrieval(
        items=(
            _item(
                np.array([float("nan"), 0.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            ),
        ),
        distances=(0.0,),
    )
    with pytest.raises(ValidationError, match="transition.delta"):
        corrector.condition(
            np.array([1.0, 0.0], dtype=np.float32),
            retrieval,
            CondCtx(np.array([1.0, 0.0], dtype=np.float32)),
        )


def test_gate_behavior_fixture_cases_are_consumable() -> None:
    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    corrector = KnnCorrector(**fixture["corrector"])

    assert fixture["schema_version"] == "mneme.gate_behavior_fixture.v1"
    for case in fixture["cases"]:
        gate = corrector.gate(case["nearest_distance"])
        assert case["expected_gate_min"] <= gate <= case["expected_gate_max"]


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
