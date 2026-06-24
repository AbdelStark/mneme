from __future__ import annotations

import inspect
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest

from mneme.condition import COND_CTX_SCHEMA, CondCtx, Conditioner, InContextConditioner
from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Retrieval,
    SchemaVersionError,
    Transition,
    ValidationError,
)


class EmptyFallbackConditioner:
    def condition(
        self,
        parametric: object,
        retrieval: Retrieval,
        ctx: CondCtx,
    ) -> object:
        if not retrieval.items:
            return parametric
        return ctx.current_latent if ctx.current_latent is not None else parametric


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _retrieval() -> Retrieval:
    z_src = np.array([1.0, 0.0], dtype=np.float32)
    z_next = np.array([1.5, 0.0], dtype=np.float32)
    item = MemoryItem(
        content_id=None,
        key=np.array([1.0, 0.0], dtype=np.float32),
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
    return Retrieval(items=(item,), distances=(0.0,))


def test_conditioner_protocol_is_importable_from_public_api() -> None:
    assert Conditioner.__name__ == "Conditioner"
    assert CondCtx.__name__ == "CondCtx"
    assert InContextConditioner.__name__ == "InContextConditioner"
    assert COND_CTX_SCHEMA == "mneme.cond_ctx.v1"


def test_condctx_records_current_goal_step_and_metadata() -> None:
    current = np.array([1.0, 0.0], dtype=np.float32)
    goal = np.array([2.0, 0.0], dtype=np.float32)

    ctx = CondCtx(
        current_latent=current,
        goal_latent=goal,
        step=4,
        metadata={"split": "fixture", "tags": ["near", "safe"]},
    )

    assert ctx.current_latent is current
    assert ctx.goal_latent is goal
    assert ctx.current is current
    assert ctx.goal is goal
    assert ctx.step == 4
    assert ctx.metadata is not None
    assert ctx.metadata["split"] == "fixture"
    assert ctx.metadata["tags"] == ("near", "safe")


def test_condctx_rejects_bad_schema_step_and_metadata() -> None:
    with pytest.raises(SchemaVersionError, match="unsupported CondCtx schema"):
        CondCtx(current_latent=None, schema_version="mneme.cond_ctx.v2")
    with pytest.raises(ValidationError, match="non-negative"):
        CondCtx(current_latent=None, step=-1)
    with pytest.raises(ValidationError, match="metadata keys"):
        CondCtx(current_latent=None, metadata={"": "bad"})
    with pytest.raises(ValidationError, match="metadata.score"):
        CondCtx(current_latent=None, metadata={"score": float("nan")})


def test_fixture_conditioner_satisfies_runtime_protocol_and_empty_fallback() -> None:
    conditioner = EmptyFallbackConditioner()
    parametric = np.array([3.0, 0.0], dtype=np.float32)

    assert isinstance(conditioner, Conditioner)
    assert (
        conditioner.condition(parametric, Retrieval((), ()), CondCtx(None))
        is parametric
    )
    np.testing.assert_array_equal(
        conditioner.condition(parametric, _retrieval(), CondCtx(parametric)),
        parametric,
    )


def test_conditioner_contract_does_not_require_base_model_or_gradients() -> None:
    signature = inspect.signature(Conditioner.condition)

    assert list(signature.parameters) == ["self", "parametric", "retrieval", "ctx"]
    assert "model" not in signature.parameters
    assert "grad" not in signature.parameters


def test_condition_import_does_not_load_optional_ml_backends() -> None:
    script = (
        "import sys; "
        "import mneme.condition; "
        "blocked = {'torch', 'faiss', 'cryptography', 'pydantic'}; "
        "loaded = sorted(blocked & set(sys.modules)); "
        "print(','.join(loaded)); "
        "raise SystemExit(1 if loaded else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
