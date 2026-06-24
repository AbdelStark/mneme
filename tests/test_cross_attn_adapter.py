from __future__ import annotations

import importlib
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mneme.core import OptionalDependencyError, ShapeError, ValidationError


class FakeTensor:
    __module__ = "torch"

    def __init__(
        self,
        value: object,
        *,
        dtype: object = "torch.float32",
        device: object = "cpu",
    ) -> None:
        self.array = np.asarray(value)
        self.dtype = dtype
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def to(
        self,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> FakeTensor:
        return FakeTensor(
            self.array,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
        )

    def with_shape(self, shape: tuple[int, ...]) -> FakeTensor:
        return FakeTensor(np.zeros(shape), dtype=self.dtype, device=self.device)

    def __add__(self, other: object) -> FakeTensor:
        if not isinstance(other, FakeTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise AssertionError(f"shape mismatch: {self.shape} != {other.shape}")
        return FakeTensor(self.array, dtype=self.dtype, device=self.device)

    def __invert__(self) -> FakeTensor:
        return FakeTensor(
            np.logical_not(self.array.astype(bool)),
            dtype="torch.bool",
            device=self.device,
        )


class FakeModule:
    def __init__(self) -> None:
        self.training = True

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.forward(*args, **kwargs)

    def forward(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def train(self, mode: bool = True) -> FakeModule:
        self.training = mode
        return self

    def eval(self) -> FakeModule:
        return self.train(False)


class FakeModuleList(list[FakeModule]):
    pass


class FakeLinear(FakeModule):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, value: object) -> FakeTensor:
        if not isinstance(value, FakeTensor):
            raise AssertionError("Linear expected FakeTensor")
        if value.shape[-1] != self.in_features:
            raise AssertionError("Linear input feature mismatch")
        return value.with_shape((*value.shape[:-1], self.out_features))


class FakeDropout(FakeModule):
    def __init__(self, _dropout: float) -> None:
        super().__init__()

    def forward(self, value: object) -> object:
        return value


class FakeLayerNorm(FakeModule):
    def __init__(self, _normalized_shape: int) -> None:
        super().__init__()

    def forward(self, value: object) -> object:
        return value


class FakeGELU(FakeModule):
    def forward(self, value: object) -> object:
        return value


class FakeSequential(FakeModule):
    def __init__(self, *modules: FakeModule) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, value: object) -> object:
        for module in self.modules:
            value = module(value)
        return value


class FakeInferenceMode:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls

    def __enter__(self) -> None:
        self.calls.append({"event": "inference_enter"})

    def __exit__(self, *_exc_info: object) -> bool:
        self.calls.append({"event": "inference_exit"})
        return False


def test_adapter_package_import_does_not_load_torch() -> None:
    script = (
        "import sys; "
        "import mneme.adapter; "
        "raise SystemExit(1 if 'torch' in sys.modules else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_cross_attn_adapter_missing_torch_raises_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_adapter_modules()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    original_import = importlib.import_module

    def missing_torch(name: str, package: str | None = None) -> Any:
        if name == "torch":
            raise ImportError("missing torch")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_torch)

    adapter = importlib.import_module("mneme.adapter")
    with pytest.raises(OptionalDependencyError) as raised:
        adapter.__getattr__("CrossAttnAdapter")

    assert raised.value.extra == "ml"
    assert raised.value.package == "torch"


def test_cross_attn_adapter_preserves_hidden_shape_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_class, calls = _load_adapter_with_fake_torch(monkeypatch)
    adapter = adapter_class(latent_dim=3, hidden_dim=4, num_heads=2, num_layers=2)
    hidden = FakeTensor(
        np.zeros((2, 5, 4)),
        dtype="torch.float16",
        device="cuda:0",
    )
    retrieved = FakeTensor(
        np.zeros((2, 7, 3)),
        dtype="torch.float32",
        device="cpu",
    )

    output = adapter(hidden, retrieved)

    assert isinstance(output, FakeTensor)
    assert output.shape == (2, 5, 4)
    assert output.dtype == "torch.float16"
    assert output.device == "cuda:0"
    attention_calls = [call for call in calls if call["event"] == "attention"]
    assert len(attention_calls) == 2
    assert attention_calls[0]["key"].shape == (2, 7, 4)
    assert attention_calls[0]["key"].dtype == "torch.float16"
    assert attention_calls[0]["key"].device == "cuda:0"


def test_cross_attn_adapter_converts_valid_mask_to_key_padding_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_class, calls = _load_adapter_with_fake_torch(monkeypatch)
    adapter = adapter_class(latent_dim=3, hidden_dim=4, num_heads=2, num_layers=1)
    hidden = FakeTensor(np.zeros((2, 5, 4)), device="mps:0")
    retrieved = FakeTensor(np.zeros((2, 3, 3)), device="cpu")
    mask = FakeTensor(
        np.array([[True, False, True], [True, True, False]]),
        dtype="torch.bool",
        device="cpu",
    )

    adapter(hidden, retrieved, mask)

    attention_call = next(call for call in calls if call["event"] == "attention")
    key_padding_mask = attention_call["key_padding_mask"]
    assert isinstance(key_padding_mask, FakeTensor)
    assert key_padding_mask.device == "mps:0"
    assert key_padding_mask.dtype == "torch.bool"
    np.testing.assert_array_equal(
        key_padding_mask.array,
        np.array([[False, True, False], [False, False, True]]),
    )


def test_cross_attn_adapter_runs_in_train_and_inference_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_class, calls = _load_adapter_with_fake_torch(monkeypatch)
    torch = sys.modules["torch"]
    adapter = adapter_class(latent_dim=3, hidden_dim=4, num_heads=2, num_layers=1)
    hidden = FakeTensor(np.zeros((1, 2, 4)))
    retrieved = FakeTensor(np.zeros((1, 3, 3)))

    adapter.train()
    train_output = adapter(hidden, retrieved)
    adapter.eval()
    with torch.inference_mode():
        eval_output = adapter(hidden, retrieved)

    assert isinstance(train_output, FakeTensor)
    assert isinstance(eval_output, FakeTensor)
    assert adapter.training is False
    assert {"event": "inference_enter"} in calls
    assert {"event": "inference_exit"} in calls


def test_cross_attn_adapter_rejects_invalid_shapes_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter_class, _calls = _load_adapter_with_fake_torch(monkeypatch)
    with pytest.raises(ValidationError, match="divisible"):
        adapter_class(latent_dim=3, hidden_dim=5, num_heads=2, num_layers=1)

    adapter = adapter_class(latent_dim=3, hidden_dim=4, num_heads=2, num_layers=1)
    with pytest.raises(ShapeError, match="batch size"):
        adapter(FakeTensor(np.zeros((2, 5, 4))), FakeTensor(np.zeros((1, 3, 3))))
    with pytest.raises(ShapeError, match="attention_mask"):
        adapter(
            FakeTensor(np.zeros((2, 5, 4))),
            FakeTensor(np.zeros((2, 3, 3))),
            FakeTensor(np.ones((2, 4), dtype=bool), dtype="torch.bool"),
        )


def _load_adapter_with_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[type[object], list[dict[str, object]]]:
    _clear_adapter_modules()
    calls: list[dict[str, object]] = []

    class FakeMultiheadAttention(FakeModule):
        def __init__(
            self,
            *,
            embed_dim: int,
            num_heads: int,
            dropout: float,
            batch_first: bool,
        ) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.dropout = dropout
            self.batch_first = batch_first

        def forward(
            self,
            *,
            query: object,
            key: object,
            value: object,
            key_padding_mask: object | None,
            need_weights: bool,
        ) -> tuple[FakeTensor, None]:
            if not isinstance(query, FakeTensor):
                raise AssertionError("attention query must be FakeTensor")
            calls.append(
                {
                    "event": "attention",
                    "query": query,
                    "key": key,
                    "value": value,
                    "key_padding_mask": key_padding_mask,
                    "need_weights": need_weights,
                }
            )
            return query.with_shape(query.shape), None

    fake_torch = SimpleNamespace(
        bool="torch.bool",
        inference_mode=lambda: FakeInferenceMode(calls),
        is_tensor=lambda value: isinstance(value, FakeTensor),
        nn=SimpleNamespace(
            Module=FakeModule,
            Linear=FakeLinear,
            ModuleList=FakeModuleList,
            MultiheadAttention=FakeMultiheadAttention,
            Dropout=FakeDropout,
            LayerNorm=FakeLayerNorm,
            Sequential=FakeSequential,
            GELU=FakeGELU,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    module = importlib.import_module("mneme.adapter._cross_attention")
    return module.CrossAttnAdapter, calls


def _clear_adapter_modules() -> None:
    sys.modules.pop("mneme.adapter._cross_attention", None)
    adapter = sys.modules.get("mneme.adapter")
    if adapter is not None:
        adapter.__dict__.pop("CrossAttnAdapter", None)
