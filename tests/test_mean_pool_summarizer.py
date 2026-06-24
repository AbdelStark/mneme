from __future__ import annotations

import numpy as np
import pytest

from mneme.core import (
    DTypeError,
    ShapeError,
    UnsupportedOperationError,
    ValidationError,
)
from mneme.encode import MeanPoolSummarizer, Summarizer


class TensorLike:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value
        self.detached = False
        self.on_cpu = False

    def detach(self) -> TensorLike:
        self.detached = True
        return self

    def cpu(self) -> TensorLike:
        self.on_cpu = True
        return self

    def numpy(self) -> np.ndarray:
        return self.value


def test_mean_pool_summarizer_outputs_contiguous_float32_vector() -> None:
    latent = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float64,
    )
    summarizer = MeanPoolSummarizer(normalize=False)

    summary = summarizer.summarize(latent)

    assert isinstance(summarizer, Summarizer)
    assert summary.dtype == np.float32
    assert summary.shape == (2,)
    assert summary.flags.c_contiguous
    np.testing.assert_allclose(summary, np.array([4.0, 5.0], dtype=np.float32))


def test_mean_pool_summarizer_normalizes_cosine_keys() -> None:
    latent = np.array([[3.0, 4.0], [3.0, 4.0]], dtype=np.float32)

    summary = MeanPoolSummarizer().summarize(latent)

    assert summary.dtype == np.float32
    assert summary.shape == (2,)
    np.testing.assert_allclose(np.linalg.norm(summary), 1.0, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(summary, np.array([0.6, 0.8], dtype=np.float32))


def test_mean_pool_summarizer_supports_tensor_like_torch_path() -> None:
    latent = TensorLike(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    summary = MeanPoolSummarizer().summarize(latent)

    assert latent.detached
    assert latent.on_cpu
    np.testing.assert_allclose(summary, np.array([0.70710677, 0.70710677]))


@pytest.mark.parametrize(
    ("latent", "error_type", "match"),
    [
        (np.array(1.0, dtype=np.float32), ShapeError, "at least one dimension"),
        (np.array(["bad"]), DTypeError, "numeric dtype"),
        (np.array([np.nan], dtype=np.float32), ValidationError, "finite"),
        (np.array([0.0, 0.0], dtype=np.float32), ValidationError, "zero"),
        (object(), DTypeError, "numpy array or tensor-like"),
    ],
)
def test_mean_pool_summarizer_rejects_invalid_latents(
    latent: object, error_type: type[Exception], match: str
) -> None:
    with pytest.raises(error_type, match=match):
        MeanPoolSummarizer().summarize(latent)


def test_projection_is_explicitly_deferred() -> None:
    with pytest.raises(UnsupportedOperationError, match="deferred"):
        MeanPoolSummarizer(output_dim=4)
