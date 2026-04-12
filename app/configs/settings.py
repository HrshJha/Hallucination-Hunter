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
    # all-MiniLM-L6-v2: 80 MB — fits Render free tier (512 MB)
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Cross-encoder NLI (core engine)
    # nli-MiniLM2-L6-H768: 80 MB, same 3-class labels (contradiction/entailment/neutral)
    nli_model: str = "cross-encoder/nli-MiniLM2-L6-H768"

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


class InferenceConfig:
    """Inference settings with truly lazy device detection.

    ``device`` is only resolved the first time it is read, so
    ``import torch`` never runs during module import / app startup.
    """

    def __init__(
        self,
        nli_batch_size: int = 32,
        similarity_batch_size: int = 64,
        max_claims: int = 50,
        device: str = "",
    ):
        self.nli_batch_size = nli_batch_size
        self.similarity_batch_size = similarity_batch_size
        self.max_claims = max_claims
        self._device = device  # store raw value; empty = detect later

    @property
    def device(self) -> str:
        if not self._device:
            self._device = self._detect_device()
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value

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
