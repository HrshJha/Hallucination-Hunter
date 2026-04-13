"""
Full Detection Pipeline (SAFE VERSION)
"""

from __future__ import annotations

from typing import Dict, List

from app.claims.extractor import extract_claims, extract_source_sentences
from app.core.nli import classify_claims_against_source
from app.core.similarity import cosine_matrix
from app.core.aggregator import aggregate
from app.models.schemas import ClaimResult, DetectionResponse, DetectionMetrics
from app.configs.settings import settings


def detect(source: str, response: str) -> DetectionResponse:
    try:
        # ── Step 0: Input validation ──
        if not source or not response:
            raise ValueError("Source or response is empty")

        # ── Step 1: Extract claims ──
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

        # ── Step 2: Source sentences ──
        source_sents: List[str] = extract_source_sentences(source)
        if not source_sents:
            source_sents = [source]

        # ── Step 3: NLI ──
        try:
            nli_results: List[Dict] = classify_claims_against_source(source, claims)
        except Exception as e:
            print("🔥 NLI ERROR:", str(e))
            nli_results = []

        # ── Step 4: Similarity ──
        try:
            matrix = cosine_matrix(source_sents, claims)
            alignment = matrix.tolist()
        except Exception as e:
            print("🔥 SIMILARITY ERROR:", str(e))
            alignment = []

        # ── Step 5: Aggregation ──
        try:
            verdict, confidence, summary, processed_claims, metrics = aggregate(
                nli_results, alignment
            )
        except Exception as e:
            print("🔥 AGGREGATION ERROR:", str(e))

            return DetectionResponse(
                verdict="ERROR",
                confidence=0.0,
                summary=f"Aggregation failed: {str(e)}",
                claims=[],
                alignment_matrix=alignment if 'alignment' in locals() else [],
                metrics=DetectionMetrics(
                    entailed_fraction=0.0,
                    unsupported_fraction=0.0,
                    contradiction_count=0,
                    avg_similarity=0.0,
                ),
            )

        # ── Step 6: Build response ──
        try:
            claim_objs = [ClaimResult(**c) for c in processed_claims]
        except Exception as e:
            print("🔥 CLAIM PARSE ERROR:", str(e))
            claim_objs = []

        return DetectionResponse(
            verdict=verdict,
            confidence=float(confidence),
            summary=summary,
            claims=claim_objs,
            alignment_matrix=alignment,
            metrics=DetectionMetrics(**metrics),
        )

    except Exception as e:
        print("🔥 PIPELINE FATAL ERROR:", str(e))

        return DetectionResponse(
            verdict="ERROR",
            confidence=0.0,
            summary=f"Pipeline failed: {str(e)}",
            claims=[],
            alignment_matrix=[],
            metrics=DetectionMetrics(
                entailed_fraction=0.0,
                unsupported_fraction=0.0,
                contradiction_count=0,
                avg_similarity=0.0,
            ),
        )