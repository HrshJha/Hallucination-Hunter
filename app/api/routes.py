"""
API Routes – hallucination detection endpoint.
"""

from __future__ import annotations

import base64
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from models.schemas import DetectionRequest, DetectionResponse
from app.core.pipeline import detect
from app.claims.extractor import extract_claims, extract_source_sentences
from app.utils.visualization import plot_alignment_matrix

router = APIRouter()


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect hallucinations in AI-generated responses",
    tags=["Detection"],
)
async def detect_hallucination(req: DetectionRequest) -> DetectionResponse:
    """
    **Input:**
    - `source`: original reference passage
    - `response`: AI-generated answer

    **Output:**
    - `label`: FAITHFUL or HALLUCINATED
    - `confidence`: 0–1 float
    - `claims`: per-claim NLI breakdown
    - `alignment_matrix`: cosine similarity (claims × source sentences)
    """
    try:
        result = detect(req.source, req.response)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/detect/visualize",
    summary="Detect + return alignment heatmap as base64 PNG",
    tags=["Detection"],
)
async def detect_and_visualize(req: DetectionRequest):
    """Same as /detect but also returns a base64-encoded heatmap image."""
    try:
        result = detect(req.source, req.response)

        # Build heatmap
        claims_text = [c.text for c in result.claims]
        source_sents = extract_source_sentences(req.source) or [req.source]

        img_bytes = plot_alignment_matrix(
            result.alignment_matrix,
            claim_labels=claims_text,
            source_labels=source_sents,
        )

        return {
            **result.model_dump(),
            "heatmap_png_base64": base64.b64encode(img_bytes).decode() if img_bytes else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}
