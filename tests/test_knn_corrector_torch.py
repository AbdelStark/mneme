from __future__ import annotations

import sys
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from mneme.condition import CondCtx, KnnCorrector
from mneme.core import (
    DTypeError,
    EncoderFingerprint,
    MemoryItem,
    Retrieval,
    Transition,
)


class FakeTensor:
    __module__ = "torch"

    def __init__(
        self,
        value: object,
        *,
        dtype: object = "torch.float32",
        device: object = "cpu",
        calls: list[str] | None = None,
        name: str = "tensor",
    ) -> None:
        self._array = np.asarray(value, dtype=np.float32)
        self.dtype = dtype
        self.device = device
        self.calls = calls if calls is not None else []
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    def detach(self) -> FakeTensor:
        self.calls.append(f"{self.name}:detach:{self.device}")
        return self

    def cpu(self) -> FakeTensor:
        self.calls.append(f"{self.name}:cpu:{self.device}")
        return FakeTensor(
            self._array,
            dtype=self.dtype,
            device="cpu",
            calls=self.calls,
            name=self.name,
        )

    def numpy(self) -> np.ndarray:
        self.calls.append(f"{self.name}:numpy:{self.device}")
        return np.array(self._array, copy=True)

    def new_tensor(self, value: object) -> FakeTensor:
        self.calls.append(f"{self.name}:new_tensor:{self.device}")
        return FakeTensor(
            value,
            dtype=self.dtype,
            device=self.device,
            calls=self.calls,
            name="result",
        )

    def to(
        self, *, dtype: object | None = None, device: object | None = None
    ) -> FakeTensor:
        self.calls.append(f"{self.name}:to:{dtype}:{device}")
        return FakeTensor(
            self._array,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            calls=self.calls,
            name=self.name,
        )


class FakeInferenceMode:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def __enter__(self) -> None:
        self._calls.append("inference:enter")

    def __exit__(self, *_exc_info: object) -> bool:
        self._calls.append("inference:exit")
        return False


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    fake_torch = SimpleNamespace(
        as_tensor=lambda value: FakeTensor(value, calls=calls, name="as_tensor"),
        inference_mode=lambda: FakeInferenceMode(calls),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _numpy_item(
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


def _torch_item(
    delta: object,
    z_next: object,
    *,
    key_value: float = 1.0,
) -> MemoryItem:
    z_src = FakeTensor(np.zeros((2,), dtype=np.float32), name="z_src")
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


def _numpy_retrieval() -> Retrieval:
    return Retrieval(
        items=(
            _numpy_item(
                np.array([2.0, 0.0], dtype=np.float32),
                np.array([4.0, 0.0], dtype=np.float32),
            ),
            _numpy_item(
                np.array([0.0, 2.0], dtype=np.float32),
                np.array([0.0, 4.0], dtype=np.float32),
                key_value=2.0,
            ),
        ),
        distances=(0.0, np.log(3.0)),
    )


def _torch_retrieval(calls: list[str]) -> Retrieval:
    return Retrieval(
        items=(
            _torch_item(
                FakeTensor([2.0, 0.0], device="cpu", calls=calls, name="delta_0"),
                FakeTensor([4.0, 0.0], device="cpu", calls=calls, name="next_0"),
            ),
            _torch_item(
                FakeTensor([0.0, 2.0], device="cpu", calls=calls, name="delta_1"),
                FakeTensor([0.0, 4.0], device="cpu", calls=calls, name="next_1"),
                key_value=2.0,
            ),
        ),
        distances=(0.0, np.log(3.0)),
    )


def test_delta_mode_torch_matches_numpy_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_fake_torch(monkeypatch, calls)
    corrector = KnnCorrector(tau=1.0, lambda_max=1.0, alpha=0.0, mode="delta")
    numpy_result = corrector.condition(
        np.array([10.0, 10.0], dtype=np.float32),
        _numpy_retrieval(),
        CondCtx(np.array([1.0, 1.0], dtype=np.float32)),
    )

    torch_result = corrector.condition(
        FakeTensor(
            [10.0, 10.0],
            dtype="torch.float32",
            device="cuda:0",
            calls=calls,
            name="parametric",
        ),
        _torch_retrieval(calls),
        CondCtx(
            FakeTensor(
                [1.0, 1.0],
                dtype="torch.float32",
                device="cuda:0",
                calls=calls,
                name="current",
            )
        ),
    )

    assert isinstance(torch_result, FakeTensor)
    assert torch_result.dtype == "torch.float32"
    assert torch_result.device == "cuda:0"
    np.testing.assert_allclose(
        torch_result.numpy(),
        numpy_result,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        torch_result.numpy(),
        np.array([6.25, 5.75], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_torch_path_uses_inference_mode_and_cpu_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_fake_torch(monkeypatch, calls)
    corrector = KnnCorrector(tau=1.0, lambda_max=1.0, alpha=0.0, mode="absolute")

    result = corrector.condition(
        FakeTensor(
            [10.0, 10.0],
            dtype="torch.float64",
            device="mps:0",
            calls=calls,
            name="parametric",
        ),
        _torch_retrieval(calls),
        CondCtx(None),
    )

    assert isinstance(result, FakeTensor)
    assert result.dtype == "torch.float64"
    assert result.device == "mps:0"
    assert "inference:enter" in calls
    assert "inference:exit" in calls
    assert calls.index("inference:enter") < calls.index("next_0:cpu:cpu")
    assert calls.index("next_1:numpy:cpu") < calls.index("inference:exit")
    np.testing.assert_allclose(
        result.numpy(),
        np.array([6.5, 5.5], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_torch_non_floating_parametric_dtype_raises_typed_error() -> None:
    corrector = KnnCorrector(tau=1.0, lambda_max=1.0, alpha=0.0, mode="delta")

    with pytest.raises(DTypeError, match="parametric"):
        corrector.condition(
            FakeTensor([10.0, 10.0], dtype="torch.int64", device="cuda:0"),
            _numpy_retrieval(),
            CondCtx(np.array([1.0, 1.0], dtype=np.float32)),
        )
