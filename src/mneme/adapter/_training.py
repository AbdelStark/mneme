"""Frozen-base adapter training harness."""

from __future__ import annotations

import importlib
import math
import platform as platform_module
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, cast

from mneme._version import __version__
from mneme.core import EvaluationError, OptionalDependencyError, ValidationError
from mneme.eval import DatasetRef, EvalMetric, EvalReport

_REQUIRED_SPLITS: Final = ("train", "calibration", "validation")
_TRAINING_CAVEAT: Final = (
    "Synthetic adapter training fixtures validate the frozen-base training "
    "contract but cannot prove external task success or adapter superiority."
)


@dataclass(frozen=True)
class AdapterTrainingBatch:
    """One offline adapter-training batch.

    `base_model(predictor_input)` must produce predictor hidden states accepted
    by the adapter. `target_hidden` is compared with the adapter output by the
    configured loss function.
    """

    predictor_input: object
    retrieved_values: object
    target_hidden: object
    attention_mask: object | None = None


def train_frozen_base_adapter(
    *,
    base_model: object,
    adapter: object,
    batches: Mapping[str, Sequence[AdapterTrainingBatch]],
    epochs: int = 1,
    learning_rate: float = 1e-3,
    seed: int = 0,
    loss_fn: object | None = None,
    optimizer: object | None = None,
    command: Sequence[str] = ("mneme", "adapter", "train-fixture"),
    created_at: str | None = None,
    git_commit: str | None = None,
) -> EvalReport:
    """Train adapter parameters while proving the base model stays frozen."""

    torch = _torch()
    _require_positive_int(epochs, "epochs")
    _require_positive_float(learning_rate, "learning_rate")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationError("seed must be an integer")
    split_batches = _require_splits(batches)
    _manual_seed(torch, seed)

    base_parameters = _parameters(base_model, "base_model")
    adapter_parameters = _parameters(adapter, "adapter")
    if not adapter_parameters:
        raise EvaluationError("adapter must expose at least one parameter")
    _freeze_base_parameters(base_parameters)
    optimizer_obj = (
        optimizer
        if optimizer is not None
        else torch.optim.AdamW(adapter_parameters, lr=learning_rate)
    )
    loss_callable = loss_fn if loss_fn is not None else torch.nn.MSELoss()

    train_losses: list[float] = []
    _call_optional(base_model, "eval")
    _call_optional(adapter, "train")
    for _epoch in range(epochs):
        for batch in split_batches["train"]:
            _call_required(optimizer_obj, "zero_grad")
            with torch.no_grad():
                predictor_hidden = _call_module(base_model, batch.predictor_input)
            prediction = _call_module(
                adapter,
                predictor_hidden,
                batch.retrieved_values,
                batch.attention_mask,
            )
            loss = _call_module(loss_callable, prediction, batch.target_hidden)
            _call_required(loss, "backward")
            _assert_no_base_gradients(base_parameters)
            _call_required(optimizer_obj, "step")
            train_losses.append(_loss_to_float(loss))

    _assert_no_base_gradients(base_parameters)
    split_losses = {
        split: _evaluate_split(
            torch,
            base_model=base_model,
            adapter=adapter,
            batches=split_batches[split],
            loss_fn=loss_callable,
        )
        for split in _REQUIRED_SPLITS
    }
    metrics: dict[str, EvalMetric] = {
        "epoch_count": epochs,
        "train_batch_count": len(split_batches["train"]),
        "calibration_batch_count": len(split_batches["calibration"]),
        "validation_batch_count": len(split_batches["validation"]),
        "optimizer_step_count": epochs * len(split_batches["train"]),
        "adapter_parameter_count": len(adapter_parameters),
        "base_parameter_count": len(base_parameters),
        "base_gradients_absent": 1,
        "last_train_loss": train_losses[-1],
        "train_loss": split_losses["train"],
        "calibration_loss": split_losses["calibration"],
        "validation_loss": split_losses["validation"],
    }
    return EvalReport(
        report_id="mneme-adapter-frozen-base-training-v1",
        command=tuple(command),
        package_version=__version__,
        git_commit=_detect_git_commit() if git_commit is None else git_commit,
        created_at=_utc_now() if created_at is None else created_at,
        platform=_platform_summary(),
        seed=seed,
        dataset=DatasetRef(
            dataset_id="adapter-training-fixture",
            kind="fixture",
            split="train-calibration-validation",
            version="v1",
            metadata={
                "fixture_scale": True,
                "synthetic": True,
                "train_batch_count": len(split_batches["train"]),
                "calibration_batch_count": len(split_batches["calibration"]),
                "validation_batch_count": len(split_batches["validation"]),
            },
        ),
        metrics=metrics,
        artifacts={
            "report_kind": "adapter-training-fixture",
            "base_model": type(base_model).__name__,
            "adapter": type(adapter).__name__,
        },
        caveats=(_TRAINING_CAVEAT,),
        passed=metrics["base_gradients_absent"] == 1
        and math.isfinite(float(metrics["validation_loss"])),
    )


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Adapter training requires the 'ml' extra",
            extra="ml",
            package="torch",
        ) from exc


def _require_splits(
    batches: Mapping[str, Sequence[AdapterTrainingBatch]],
) -> dict[str, tuple[AdapterTrainingBatch, ...]]:
    split_batches: dict[str, tuple[AdapterTrainingBatch, ...]] = {}
    for split in _REQUIRED_SPLITS:
        values = tuple(batches.get(split, ()))
        if not values:
            raise EvaluationError(f"{split} split must contain at least one batch")
        for batch in values:
            if not isinstance(batch, AdapterTrainingBatch):
                raise EvaluationError(f"{split} split contains a non-training batch")
        split_batches[split] = values
    return split_batches


def _parameters(module: object, field_name: str) -> tuple[Any, ...]:
    try:
        parameters = cast(Any, module).parameters
    except AttributeError as exc:
        raise EvaluationError(f"{field_name} must expose parameters()") from exc
    if not callable(parameters):
        raise EvaluationError(f"{field_name}.parameters must be callable")
    return tuple(parameters())


def _freeze_base_parameters(parameters: Sequence[object]) -> None:
    for parameter in parameters:
        parameter_obj = cast(Any, parameter)
        parameter_obj.requires_grad = False
        parameter_obj.grad = None


def _assert_no_base_gradients(parameters: Sequence[object]) -> None:
    leaked = [
        index for index, parameter in enumerate(parameters) if _has_grad(parameter)
    ]
    if leaked:
        raise ValidationError(
            "base parameters received gradients while training adapter: "
            + ", ".join(str(index) for index in leaked)
        )


def _has_grad(parameter: object) -> bool:
    try:
        grad = cast(Any, parameter).grad
    except AttributeError:
        return False
    return grad is not None


def _manual_seed(torch: Any, seed: int) -> None:
    manual_seed = getattr(torch, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)


def _evaluate_split(
    torch: Any,
    *,
    base_model: object,
    adapter: object,
    batches: Sequence[AdapterTrainingBatch],
    loss_fn: object,
) -> float:
    _call_optional(adapter, "eval")
    losses: list[float] = []
    with torch.no_grad():
        for batch in batches:
            predictor_hidden = _call_module(base_model, batch.predictor_input)
            prediction = _call_module(
                adapter,
                predictor_hidden,
                batch.retrieved_values,
                batch.attention_mask,
            )
            loss = _call_module(loss_fn, prediction, batch.target_hidden)
            losses.append(_loss_to_float(loss))
    return float(sum(losses) / len(losses))


def _loss_to_float(loss: object) -> float:
    value = loss
    if _has_callable(value, "detach"):
        value = cast(Any, value).detach()
    if _has_callable(value, "cpu"):
        value = cast(Any, value).cpu()
    if _has_callable(value, "item"):
        value = cast(Any, value).item()
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise EvaluationError("loss must be convertible to float") from exc
    if not math.isfinite(converted):
        raise EvaluationError("loss must be finite")
    return converted


def _call_module(module: object, *args: object) -> Any:
    if not callable(module):
        raise EvaluationError(f"{type(module).__name__} must be callable")
    return module(*args)


def _call_required(target: object, method_name: str) -> Any:
    try:
        method = getattr(cast(Any, target), method_name)
    except AttributeError as exc:
        raise EvaluationError(
            f"{type(target).__name__}.{method_name} is required"
        ) from exc
    if not callable(method):
        raise EvaluationError(f"{type(target).__name__}.{method_name} must be callable")
    return method()


def _call_optional(target: object, method_name: str) -> None:
    try:
        method = getattr(cast(Any, target), method_name)
    except AttributeError:
        return
    if callable(method):
        method()


def _has_callable(target: object, method_name: str) -> bool:
    try:
        method = getattr(cast(Any, target), method_name)
    except AttributeError:
        return False
    return callable(method)


def _require_positive_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvaluationError(f"{field_name} must be a positive integer")


def _require_positive_float(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0.0:
        raise EvaluationError(f"{field_name} must be a positive number")
    if not math.isfinite(float(value)):
        raise EvaluationError(f"{field_name} must be finite")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _platform_summary() -> dict[str, str]:
    return {
        "machine": platform_module.machine() or "unknown",
        "python": platform_module.python_version(),
        "system": platform_module.system() or "unknown",
    }


def _detect_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


__all__ = ["AdapterTrainingBatch", "train_frozen_base_adapter"]
