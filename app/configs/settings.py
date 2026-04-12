"""
Global configuration for Hallucination Hunter.
All thresholds, model names, and runtime settings live here.

NOTE: torch is lazy-imported so the app can start and bind its port
      even if PyTorch is slow to initialise or absent.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Sentence-Transformer (baseline similarity)
    similarity_model: str = "sentence-transformers/all-mpnet-base-v2"

    # Cross-encoder NLI (core engine)
    nli_model: str = "cross-encoder/nli-deberta-v3-small"

    # spaCy pipeline for claim extraction
    spacy_model: str = "en_core_web_sm"


@dataclass
class ThresholdConfig:
    # Similarity baseline
    similarity_threshold: float = 0.65

    # NLI aggregation
    entailment_fraction_threshold: float = 0.60

    # NLI label indices for cross-encoder/nli-deberta-v3-small
    # Labels: 0 = contradiction, 1 = entailment, 2 = neutral
    contradiction_idx: int = 0
    entailment_idx: int = 1
    neutral_idx: int = 2


@dataclass
class InferenceConfig:
    nli_batch_size: int = 32
    similarity_batch_size: int = 64
    max_claims: int = 50
    device: str = ""  # resolved lazily

    def __post_init__(self):
        if not self.device:
            self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@dataclass
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# Singleton
settings = Settings()
