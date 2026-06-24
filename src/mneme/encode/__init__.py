"""Encoder and summarizer public contracts."""

from mneme.encode._fingerprint import (
    build_encoder_fingerprint,
    digest_config,
    digest_weights,
    ensure_fingerprint_match,
    fingerprints_match,
    format_fingerprint,
)
from mneme.encode._mean_pool import MeanPoolSummarizer
from mneme.encode._protocols import Encoder, Summarizer

__all__ = [
    "Encoder",
    "MeanPoolSummarizer",
    "Summarizer",
    "build_encoder_fingerprint",
    "digest_config",
    "digest_weights",
    "ensure_fingerprint_match",
    "fingerprints_match",
    "format_fingerprint",
]
