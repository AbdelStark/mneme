from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from mneme.adapter import AdapterTrainingBatch, train_frozen_base_adapter
from mneme.core import EvaluationError, OptionalDependencyError, ValidationError
from mneme.eval import validate_report_json


class FakeTensor:
    def __init__(self, value: float) -> None:
        self.value = float(value)


class FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True
        self.grad: FakeTensor | None = None


class FakeBaseModel:
    def __init__(self) -> None:
        self.parameter = FakeParameter()
        self.training = True
        self.calls = 0

    def parameters(self) -> tuple[FakeParameter, ...]:
        return (self.parameter,)

    def eval(self) -> None:
        self.training = False

    def __call__(self, predictor_input: FakeTensor) -> FakeTensor:
        self.calls += 1
        return FakeTensor(predictor_input.value + 1.0)


class FakeAdapter:
    def __init__(self) -> None:
        self.parameter = FakeParameter()
        self.training = False
        self.calls: list[tuple[FakeTensor, FakeTensor, object | None]] = []

    def parameters(self) -> tuple[FakeParameter, ...]:
        return (self.parameter,)

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def __call__(
        self,
        predictor_hidden: FakeTensor,
        retrieved_values: FakeTensor,
        attention_mask: object | None = None,
    ) -> FakeTensor:
        self.calls.append((predictor_hidden, retrieved_values, attention_mask))
        return FakeTensor(predictor_hidden.value + retrieved_values.value)


class FakeLoss:
    def __init__(
        self,
        value: float,
        *,
        adapter_parameters: tuple[FakeParameter, ...],
        base_parameters: tuple[FakeParameter, ...],
        leak_base_gradient: bool,
    ) -> None:
        self.value = value
        self.adapter_parameters = adapter_parameters
        self.base_parameters = base_parameters
        self.leak_base_gradient = leak_base_gradient

    def backward(self) -> None:
        for parameter in self.adapter_parameters:
            parameter.grad = FakeTensor(1.0)
        if self.leak_base_gradient:
            self.base_parameters[0].grad = FakeTensor(1.0)

    def detach(self) -> FakeLoss:
        return self

    def cpu(self) -> FakeLoss:
        return self

    def item(self) -> float:
        return self.value


class FakeLossFn:
    def __init__(
        self,
        *,
        adapter_parameters: tuple[FakeParameter, ...],
        base_parameters: tuple[FakeParameter, ...],
        leak_base_gradient: bool = False,
    ) -> None:
        self.adapter_parameters = adapter_parameters
        self.base_parameters = base_parameters
        self.leak_base_gradient = leak_base_gradient

    def __call__(self, prediction: FakeTensor, target: FakeTensor) -> FakeLoss:
        return FakeLoss(
            abs(prediction.value - target.value),
            adapter_parameters=self.adapter_parameters,
            base_parameters=self.base_parameters,
            leak_base_gradient=self.leak_base_gradient,
        )


class FakeNoGrad:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __enter__(self) -> None:
        self.calls.append("no_grad_enter")

    def __exit__(self, *_exc_info: object) -> bool:
        self.calls.append("no_grad_exit")
        return False


class FakeAdamW:
    instances: list[FakeAdamW] = []

    def __init__(self, parameters: object, *, lr: float) -> None:
        self.parameters = tuple(parameters)
        self.lr = lr
        self.zero_grad_calls = 0
        self.step_calls = 0
        FakeAdamW.instances.append(self)

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        self.step_calls += 1


def test_frozen_base_training_report_records_splits_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_torch(monkeypatch)
    base = FakeBaseModel()
    adapter = FakeAdapter()
    batches = _batches()

    report = train_frozen_base_adapter(
        base_model=base,
        adapter=adapter,
        batches=batches,
        epochs=2,
        learning_rate=0.05,
        seed=123,
        loss_fn=FakeLossFn(
            adapter_parameters=adapter.parameters(),
            base_parameters=base.parameters(),
        ),
        command=("mneme", "adapter", "train-fixture"),
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    decoded = validate_report_json(report.to_json())

    assert decoded == report
    assert base.parameter.requires_grad is False
    assert base.parameter.grad is None
    assert FakeAdamW.instances[0].parameters == adapter.parameters()
    assert FakeAdamW.instances[0].lr == 0.05
    assert FakeAdamW.instances[0].step_calls == 4
    assert report.metrics["base_gradients_absent"] == 1
    assert report.metrics["optimizer_step_count"] == 4
    assert report.metrics["train_batch_count"] == 2
    assert report.metrics["calibration_batch_count"] == 1
    assert report.metrics["validation_batch_count"] == 1
    assert report.metrics["last_train_loss"] >= 0.0
    assert report.dataset.metadata["train_batch_count"] == 2
    assert report.seed == 123
    assert report.caveats
    assert "manual_seed:123" in calls
    assert calls.count("no_grad_enter") >= 3


def test_frozen_base_training_fails_when_base_receives_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)
    base = FakeBaseModel()
    adapter = FakeAdapter()

    with pytest.raises(ValidationError, match="base parameters received gradients"):
        train_frozen_base_adapter(
            base_model=base,
            adapter=adapter,
            batches=_batches(),
            loss_fn=FakeLossFn(
                adapter_parameters=adapter.parameters(),
                base_parameters=base.parameters(),
                leak_base_gradient=True,
            ),
        )


def test_frozen_base_training_requires_all_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch(monkeypatch)

    with pytest.raises(EvaluationError, match="validation split"):
        train_frozen_base_adapter(
            base_model=FakeBaseModel(),
            adapter=FakeAdapter(),
            batches={
                "train": (
                    AdapterTrainingBatch(
                        FakeTensor(0.0), FakeTensor(0.0), FakeTensor(0.0)
                    ),
                ),
                "calibration": (
                    AdapterTrainingBatch(
                        FakeTensor(0.0), FakeTensor(0.0), FakeTensor(0.0)
                    ),
                ),
            },
        )


def test_frozen_base_training_missing_torch_raises_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    original_import = importlib.import_module

    def missing_torch(name: str, package: str | None = None) -> Any:
        if name == "torch":
            raise ImportError("missing torch")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", missing_torch)

    with pytest.raises(OptionalDependencyError) as raised:
        train_frozen_base_adapter(
            base_model=FakeBaseModel(),
            adapter=FakeAdapter(),
            batches=_batches(),
        )

    assert raised.value.extra == "ml"
    assert raised.value.package == "torch"


def _batches() -> dict[str, tuple[AdapterTrainingBatch, ...]]:
    return {
        "train": (
            AdapterTrainingBatch(FakeTensor(0.0), FakeTensor(1.0), FakeTensor(2.0)),
            AdapterTrainingBatch(FakeTensor(1.0), FakeTensor(1.0), FakeTensor(3.0)),
        ),
        "calibration": (
            AdapterTrainingBatch(FakeTensor(2.0), FakeTensor(1.0), FakeTensor(4.0)),
        ),
        "validation": (
            AdapterTrainingBatch(FakeTensor(3.0), FakeTensor(1.0), FakeTensor(5.0)),
        ),
    }


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    FakeAdamW.instances = []
    calls: list[str] = []
    fake_torch = SimpleNamespace(
        manual_seed=lambda seed: calls.append(f"manual_seed:{seed}"),
        no_grad=lambda: FakeNoGrad(calls),
        optim=SimpleNamespace(AdamW=FakeAdamW),
        nn=SimpleNamespace(MSELoss=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return calls
