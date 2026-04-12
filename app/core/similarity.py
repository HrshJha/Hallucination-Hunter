"""
Baseline Similarity Model
--------------------------
Uses sentence-transformers (all-mpnet-base-v2) to compute cosine
similarity between source and response embeddings.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from app.configs.settings import settings

# ── Lazy singleton ───────────────────────────────────────────────────

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            settings.model.similarity_model,
            device=settings.inference.device,
        )
    return _model


# ── Core functions ───────────────────────────────────────────────────

def embed(texts: List[str]) -> np.ndarray:
    """Return L2-normalised embeddings (N × D)."""
    model = _get_model()
    return model.encode(
        texts,
        batch_size=settings.inference.similarity_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def cosine_matrix(source_sents: List[str], claims: List[str]) -> np.ndarray:
    """
    Compute cosine similarity matrix.

    Returns
    -------
    np.ndarray of shape (len(claims), len(source_sents))
        ``matrix[i][j]`` = cosine similarity between claim *i* and source sentence *j*.
    """
    src_emb = embed(source_sents)   # (S, D)
    clm_emb = embed(claims)         # (C, D)
    return clm_emb @ src_emb.T      # (C, S)


def similarity_scores(
    source_sents: List[str],
    claims: List[str],
) -> Tuple[float, float, np.ndarray]:
    """
    Compute aggregate similarity scores.

    Returns
    -------
    avg_sim : float
        Mean of per-claim max similarities.
    min_sim : float
        Min of per-claim max similarities.
    matrix : np.ndarray
        Full cosine similarity matrix (C × S).
    """
    matrix = cosine_matrix(source_sents, claims)
    per_claim_max = matrix.max(axis=1)  # best source match per claim
    avg_sim = float(np.mean(per_claim_max))
    min_sim = float(np.min(per_claim_max))
    return avg_sim, min_sim, matrix


def baseline_predict(
    source_sents: List[str],
    claims: List[str],
    threshold: float | None = None,
) -> Tuple[str, float]:
    """
    Threshold-based classifier on average similarity.

    Returns
    -------
    label : str
        FAITHFUL or HALLUCINATED
    confidence : float
    """
    if threshold is None:
        threshold = settings.threshold.similarity_threshold

    avg_sim, _min_sim, _matrix = similarity_scores(source_sents, claims)

    if avg_sim >= threshold:
        return "FAITHFUL", avg_sim
    else:
        return "HALLUCINATED", 1.0 - avg_sim
