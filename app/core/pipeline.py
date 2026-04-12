"""
Full Detection Pipeline
-----------------------
Orchestrates:
  1. extract_claims(response)
  2. extract_source_sentences(source)
  3. run_nli(claims, source)          — via nli.classify_claims_against_source
  4. compute_similarity(claims, src)  — via similarity.cosine_matrix
  5. aggregate(nli_results, matrix)   — classify + aggregate + confidence + summary
"""

from __future__ import annotations

from typing import Dict, List

from app.claims.extractor import extract_claims, extract_source_sentences
from app.core.nli import classify_claims_against_source
from app.core.similarity import cosine_matrix
from app.core.aggregator import aggregate
from app.models.schemas import ClaimResult, DetectionResponse, DetectionMetrics
from configs.settings import settings


def detect(source: str, response: str) -> DetectionResponse:
    """
    Run the full hallucination detection pipeline.

    Parameters
    ----------
    source : str
        Original reference passage (ground truth).
    response : str
        AI-generated answer to verify.

    Returns
    -------
    DetectionResponse
        Complete JSON-serialisable result with verdict, confidence,
        summary, per-claim breakdown, alignment matrix, and metrics.
    """

    # ── Step 1: Extract claims from response ──
    claims: List[str] = extract_claims(
        response,
        max_claims=settings.inference.max_claims,
    )

    if not claims:
        return DetectionResponse(
            verdict="FAITHFUL",
            confidence=0.95,
            summary="No claims extracted → response is faithful",
            claims=[],
            alignment_matrix=[],
            metrics=DetectionMetrics(
                entailed_fraction=1.0,
                unsupported_fraction=0.0,
                contradiction_count=0,
                avg_similarity=1.0,
            ),
        )

    # ── Step 2: Extract source sentences for alignment matrix ──
    source_sents: List[str] = extract_source_sentences(source)
    if not source_sents:
        source_sents = [source]

    # ── Step 3: NLI classification (claim vs source) ──
    nli_results: List[Dict] = classify_claims_against_source(source, claims)

    # ── Step 4: Semantic similarity matrix ──
    try:
        matrix = cosine_matrix(source_sents, claims)
        alignment = matrix.tolist()
    except Exception:
        # Fallback: empty matrix (aggregator handles missing rows)
        alignment = []

    # ── Step 5: Aggregate → verdict + confidence + summary ──
    verdict, confidence, summary, processed_claims, metrics = aggregate(
        nli_results, alignment
    )

    # ── Step 6: Build typed response ──
    claim_objs = [ClaimResult(**c) for c in processed_claims]

    return DetectionResponse(
        verdict=verdict,
        confidence=confidence,
        summary=summary,
        claims=claim_objs,
        alignment_matrix=alignment,
        metrics=DetectionMetrics(**metrics),
    )
