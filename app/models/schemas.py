"""
Pydantic schemas for API request / response validation.

Output format:

{
  "verdict": "FAITHFUL | HALLUCINATED",
  "confidence": float,           # [0.50, 0.95], never 1.00
  "summary": "...",               # human-readable 1-liner
  "claims": [
    {
      "text": "...",
      "nli_label": "...",         # entailed | contradiction | unsupported
      "similarity": float,        # never 0.00
      "final_label": "ENTAILED | CONTRADICTION | UNSUPPORTED"
    }
  ],
  "metrics": {
    "entailed_fraction": float,
    "unsupported_fraction": float,
    "contradiction_count": int,
    "avg_similarity": float
  },
  "alignment_matrix": [[float]]
}
"""

from pydantic import BaseModel, Field
from typing import List


# ── Request ──────────────────────────────────────────────────────────

class DetectionRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Original reference passage")
    response: str = Field(..., min_length=1, description="AI-generated answer to verify")


# ── Claim-level result ──────────────────────────────────────────────

class ClaimResult(BaseModel):
    text: str
    nli_label: str                                      # entailed | contradiction | unsupported
    similarity: float = Field(..., ge=0.0, le=1.0)      # never 0.00
    final_label: str                                    # ENTAILED | CONTRADICTION | UNSUPPORTED


# ── Metrics ─────────────────────────────────────────────────────────

class DetectionMetrics(BaseModel):
    entailed_fraction: float
    unsupported_fraction: float
    contradiction_count: int
    avg_similarity: float


# ── Full response ───────────────────────────────────────────────────

class DetectionResponse(BaseModel):
    verdict: str                                        # FAITHFUL | HALLUCINATED
    confidence: float = Field(..., ge=0.0, le=1.0)      # [0.50, 0.95]
    summary: str                                        # human-readable 1-liner
    claims: List[ClaimResult]
    metrics: DetectionMetrics
    alignment_matrix: List[List[float]]
