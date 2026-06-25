from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mneme.core import (
    FingerprintMismatchError,
    MemoryItem,
    Transition,
    ValidationError,
    content_id,
)
from mneme.encode import (
    MeanPoolSummarizer,
    build_encoder_fingerprint,
    digest_config,
    digest_weights,
    ensure_fingerprint_match,
    fingerprints_match,
    format_fingerprint,
)


def _transition() -> Transition:
    z_src = np.array([1.0, 2.0], dtype=np.float32)
    z_next = np.array([1.5, 2.5], dtype=np.float32)
    return Transition(
        z_src=z_src,
        action=np.array([0.1], dtype=np.float32),
        z_next=z_next,
        delta=z_next - z_src,
        t=0,
        episode_id=uuid4(),
    )


def test_config_digest_is_stable_and_order_independent() -> None:
    left = digest_config({"b": [2, True], "a": {"x": 1}})
    right = digest_config({"a": {"x": 1}, "b": [2, True]})

    assert left == right
    assert left.startswith("blake3:")


def test_fingerprint_changes_when_summarizer_config_changes() -> None:
    normalized = build_encoder_fingerprint(
        "encoder",
        MeanPoolSummarizer(normalize=True),
        encoder_config={"preprocess": "v1"},
        unweighted=True,
    )
    unnormalized = build_encoder_fingerprint(
        "encoder",
        MeanPoolSummarizer(normalize=False),
        encoder_config={"preprocess": "v1"},
        unweighted=True,
    )

    assert normalized.summarizer_id == "meanpool-v1"
    assert normalized.config_digest != unnormalized.config_digest


def test_weight_digest_policy_requires_explicit_unweighted_or_digest() -> None:
    with pytest.raises(ValidationError, match="weights_digest is required"):
        build_encoder_fingerprint("encoder", "summary")

    unweighted = build_encoder_fingerprint("encoder", "summary", unweighted=True)

    assert unweighted.weights_digest is None


def test_weight_digest_can_come_from_bytes_or_file(tmp_path: Path) -> None:
    weights = b"weights"
    path = tmp_path / "weights.bin"
    path.write_bytes(weights)

    assert digest_weights(weights) == digest_weights(path)
    assert build_encoder_fingerprint(
        "encoder", "summary", weights=weights
    ).weights_digest == digest_weights(weights)


def test_weight_digest_wraps_missing_weight_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="weights file not found"):
        digest_weights(tmp_path / "missing.bin")


def test_weight_digest_wraps_unreadable_weight_path(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.mkdir()

    with pytest.raises(ValidationError, match="weights file could not be read"):
        digest_weights(path)


def test_mismatched_fingerprints_fail_closed() -> None:
    expected = build_encoder_fingerprint("encoder", "summary", unweighted=True)
    actual = build_encoder_fingerprint("encoder", "other", unweighted=True)

    assert fingerprints_match(expected, expected)
    assert not fingerprints_match(expected, actual)
    with pytest.raises(FingerprintMismatchError, match="fingerprint mismatch"):
        ensure_fingerprint_match(expected, actual)


def test_format_fingerprint_includes_display_fields() -> None:
    fingerprint = build_encoder_fingerprint(
        "encoder",
        "summary",
        weights_digest="blake3:weights",
    )

    display = format_fingerprint(fingerprint)

    assert "encoder/summary" in display
    assert "config=blake3:" in display
    assert "weights=blake3:weights" in display


def test_fingerprint_is_included_in_content_id_input() -> None:
    transition = _transition()
    key = np.array([0.6, 0.8], dtype=np.float32)
    left_fp = build_encoder_fingerprint("encoder", "summary", unweighted=True)
    right_fp = build_encoder_fingerprint("encoder", "other", unweighted=True)

    left = MemoryItem(None, key, transition, {"source": "fixture"}, left_fp)
    right = MemoryItem(None, key, transition, {"source": "fixture"}, right_fp)

    assert content_id(left) != content_id(right)


def test_digest_config_rejects_unsupported_values() -> None:
    with pytest.raises(ValidationError, match="unsupported configuration"):
        digest_config({"raw": b"bytes"})
