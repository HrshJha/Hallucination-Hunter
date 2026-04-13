"""
API Routes – hallucination detection endpoint.
"""

from __future__ import annotations

import base64
from fastapi import APIRouter, HTTPException

from app.models.schemas import DetectionRequest, DetectionResponse
from app.core.pipeline import detect
from app.claims.extractor import extract_source_sentences
from app.utils.visualization import plot_alignment_matrix

router = APIRouter()


def _check_warmup():
    """Raise 503 if ML models haven't finished loading yet."""
    # Import here to avoid circular imports
    from app.main import _warmup_state
    if not _warmup_state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="ML models are still loading. Please wait and retry in a few seconds.",
        )


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
    _check_warmup()
    try:
        # Input validation
        if not req.source or not req.response:
            raise ValueError("Source or response cannot be empty")

        result = detect(req.source, req.response)

        # Safety check
        if result is None:
            raise ValueError("Detection returned None")

        return result

    except HTTPException:
        raise  # re-raise 503 from _check_warmup
    except Exception as e:
        print("🔥 ERROR IN /detect:", str(e))

        # Never crash → always return structured response
        return DetectionResponse(
            verdict="ERROR",
            confidence=0.0,
            summary=f"Internal error: {str(e)}",
            claims=[],
            metrics={
                "entailed_fraction": 0.0,
                "unsupported_fraction": 0.0,
                "contradiction_count": 0,
                "avg_similarity": 0.0,
            },
            alignment_matrix=[],
        )


@router.post(
    "/detect/visualize",
    summary="Detect + return alignment heatmap as base64 PNG",
    tags=["Detection"],
)
async def detect_and_visualize(req: DetectionRequest):
    """Same as /detect but also returns a base64-encoded heatmap image."""
    _check_warmup()
    try:
        if not req.source or not req.response:
            raise ValueError("Source or response cannot be empty")

        result = detect(req.source, req.response)

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

    except HTTPException:
        raise
    except Exception as e:
        print("🔥 ERROR IN /detect/visualize:", str(e))

        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "summary": str(e),
            "claims": [],
            "metrics": {},
            "alignment_matrix": [],
            "heatmap_png_base64": None,
        }


@router.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}
