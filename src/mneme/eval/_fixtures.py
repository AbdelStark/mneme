"""Deterministic fixture-scale evaluation reports."""

from __future__ import annotations

import platform as platform_module
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

import numpy as np

from mneme._version import __version__
from mneme.condition import CondCtx, KnnCorrector
from mneme.core import EncoderFingerprint, MemoryItem, Retrieval, Transition
from mneme.eval._reports import DatasetRef, EvalReport

_REPORT_SCHEMA_CAVEAT = (
    "Synthetic fixture evidence cannot prove external task success or broad "
    "benchmark improvement."
)


def run_fixture_evaluation(
    *,
    seed: int = 0,
    command: Sequence[str] = ("mneme", "eval", "fixtures"),
    created_at: str | None = None,
    git_commit: str | None = None,
) -> EvalReport:
    """Build the deterministic v0.1 fixture drift and gate report."""

    corrector = KnnCorrector(tau=1.0, lambda_max=1.0, alpha=0.0, mode="delta")
    current = np.array([1.0, 1.0], dtype=np.float32)
    parametric = np.array([10.0, 10.0], dtype=np.float32)
    true_next = np.array([2.5, 1.5], dtype=np.float32)
    retrieval = _drift_retrieval()
    corrected = corrector.condition(parametric, retrieval, CondCtx(current))

    no_memory_error = _l2(parametric, true_next)
    corrector_error = _l2(_as_array(corrected), true_next)
    near_gate = KnnCorrector().gate(0.0)
    far_gate = KnnCorrector().gate(10.0)

    return EvalReport(
        report_id="mneme-fixture-drift-gate-v1",
        command=tuple(command),
        package_version=__version__,
        git_commit=_detect_git_commit() if git_commit is None else git_commit,
        created_at=_utc_now() if created_at is None else created_at,
        platform=_platform_summary(),
        seed=seed,
        dataset=DatasetRef(
            dataset_id="synthetic-drift-gate-fixture",
            kind="fixture",
            split="deterministic",
            version="v1",
            metadata={
                "synthetic": True,
                "fixture_scale": True,
                "case_count": 3,
            },
        ),
        metrics={
            "no_memory_l2": no_memory_error,
            "corrector_l2": corrector_error,
            "corrector_improves_fixture": int(corrector_error < no_memory_error),
            "gate_in_distribution_near": near_gate,
            "gate_out_of_distribution_far": far_gate,
            "gate_in_distribution_case_count": 1,
            "gate_out_of_distribution_case_count": 1,
        },
        artifacts={
            "gate_cases": "tests/fixtures/condition/gate_behavior_cases.json",
            "report_kind": "fixture-scale",
        },
        caveats=(_REPORT_SCHEMA_CAVEAT,),
        passed=corrector_error < no_memory_error and far_gate < 1e-6,
    )


def _drift_retrieval() -> Retrieval:
    fingerprint = EncoderFingerprint(
        encoder_id="fixture.encoder",
        summarizer_id="fixture.summary",
        weights_digest=None,
        config_digest="blake3:fixture",
    )
    episode_id = UUID("12345678-1234-5678-1234-567812345678")
    first_delta = np.array([2.0, 0.0], dtype=np.float32)
    second_delta = np.array([0.0, 2.0], dtype=np.float32)
    return Retrieval(
        items=(
            _item(first_delta, fingerprint, episode_id, step=1),
            _item(second_delta, fingerprint, episode_id, step=2),
        ),
        distances=(0.0, float(np.log(3.0))),
    )


def _item(
    delta: np.ndarray,
    fingerprint: EncoderFingerprint,
    episode_id: UUID,
    *,
    step: int,
) -> MemoryItem:
    z_src = np.zeros_like(delta)
    return MemoryItem(
        content_id=None,
        key=np.array([float(step), 0.0], dtype=np.float32),
        value=Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_src + delta,
            delta=delta,
            t=step,
            episode_id=episode_id,
        ),
        meta={"fixture": "drift-gate", "step": step},
        encoder_fp=fingerprint,
    )


def _l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _as_array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("fixture corrector returned a non-NumPy value")
    return value


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


__all__ = ["run_fixture_evaluation"]
