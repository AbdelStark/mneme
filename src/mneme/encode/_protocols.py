"""Protocols for model adapters and summary-key generators."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mneme.core import EncoderFingerprint, Latent, SummaryVec


@runtime_checkable
class Encoder(Protocol):
    """Adapter contract for turning observations into latent values."""

    def encode(self, obs: object) -> Latent:
        """Encode an observation into a latent backend object."""
        ...

    def fingerprint(self) -> EncoderFingerprint:
        """Return the fingerprint for produced latents and summary keys."""
        ...


@runtime_checkable
class Summarizer(Protocol):
    """Contract for producing compact index keys from latents."""

    @property
    def id(self) -> str:
        """Stable summarizer identifier included in fingerprints."""
        ...

    def summarize(self, z: Latent) -> SummaryVec:
        """Summarize a latent into a one-dimensional float32 key."""
        ...


__all__ = ["Encoder", "Summarizer"]
