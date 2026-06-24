"""Encoder and summarizer public contracts."""

from mneme.encode._mean_pool import MeanPoolSummarizer
from mneme.encode._protocols import Encoder, Summarizer

__all__ = ["Encoder", "MeanPoolSummarizer", "Summarizer"]
