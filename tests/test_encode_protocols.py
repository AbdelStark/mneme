from __future__ import annotations

import subprocess
import sys

import numpy as np

from mneme.core import EncoderFingerprint
from mneme.encode import Encoder, Summarizer


class FixtureEncoder:
    def encode(self, obs: object) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def fingerprint(self) -> EncoderFingerprint:
        return EncoderFingerprint(
            encoder_id="fixture.encoder",
            summarizer_id="fixture.summary",
            weights_digest=None,
            config_digest="sha256:fixture",
        )


class FixtureSummarizer:
    @property
    def id(self) -> str:
        return "fixture.summary"

    def summarize(self, z: object) -> np.ndarray:
        array = np.asarray(z, dtype=np.float32)
        return np.ascontiguousarray(array.reshape(-1).mean(keepdims=True))


def test_protocols_are_importable_from_public_api() -> None:
    assert Encoder.__name__ == "Encoder"
    assert Summarizer.__name__ == "Summarizer"


def test_fixture_adapter_satisfies_runtime_protocols() -> None:
    encoder = FixtureEncoder()
    summarizer = FixtureSummarizer()

    assert isinstance(encoder, Encoder)
    assert isinstance(summarizer, Summarizer)
    assert encoder.fingerprint().summarizer_id == summarizer.id
    np.testing.assert_array_equal(
        summarizer.summarize(encoder.encode([1.0, 3.0])),
        np.array([2.0], dtype=np.float32),
    )


def test_encode_import_does_not_load_optional_ml_backends() -> None:
    script = (
        "import sys; "
        "import mneme.encode; "
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
