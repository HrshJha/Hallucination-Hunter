"""
Aggregation & Classification Logic
------------------------------------
Clean, refactored pipeline functions:

1. classify_claim(nli_label, similarity)  → per-claim label
2. aggregate_results(claims)              → verdict + metrics
3. compute_confidence(metrics)            → calibrated float
4. generate_summary(metrics)              → human-readable string
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────
# 1. LABEL MAPPING — convert raw NLI string to final label
# ─────────────────────────────────────────────────────────────────────

_NLI_MAP = {
    "entailment":    "ENTAILED",
    "contradiction": "CONTRADICTION",
    "neutral":       "UNSUPPORTED",
}


def _map_nli_label(raw: str) -> str:
    """Map a raw NLI label to a clean final label. Default: UNSUPPORTED."""
    return _NLI_MAP.get(raw.lower().strip(), "UNSUPPORTED") if raw else "UNSUPPORTED"


# ─────────────────────────────────────────────────────────────────────
# 2. SIMILARITY SAFETY — never return 0.00
# ─────────────────────────────────────────────────────────────────────

def _safe_similarity(raw_value: float) -> float:
    """Clamp similarity: fallback to 0.3 if zero/missing, round to 2dp."""
    if raw_value is None or raw_value <= 0.0:
        return 0.3
    return round(float(raw_value), 2)


# ─────────────────────────────────────────────────────────────────────
# 3. CLASSIFY CLAIM — hybrid NLI + similarity
# ─────────────────────────────────────────────────────────────────────

def classify_claim(nli_label: str, similarity: float) -> str:
    """
    Hybrid NLI + similarity classification.

    Priority order:
      1. Contradiction always wins (NLI says conflict → CONTRADICTION)
      2. High similarity (≥ 0.75) overrides neutral → ENTAILED (safety override)
      3. Medium-high similarity (≥ 0.70) → ENTAILED
      4. NLI entailment → ENTAILED
      5. Everything else → UNSUPPORTED

    Key: High similarity MUST override NLI neutral, but never override contradiction.
    """
    mapped = _map_nli_label(nli_label)

    # Step 1: Contradiction always wins — never masked by similarity
    if mapped == "CONTRADICTION":
        return "CONTRADICTION"

    # Step 2: Safety override — high similarity forces ENTAILED
    #         (overrides neutral or weak NLI)
    if similarity >= 0.75:
        return "ENTAILED"

    # Step 3: Medium-high similarity → ENTAILED
    if similarity >= 0.70:
        return "ENTAILED"

    # Step 4: NLI entailment → ENTAILED
    if mapped == "ENTAILED":
        return "ENTAILED"

    # Step 5: Everything else (neutral + low sim) → UNSUPPORTED
    return "UNSUPPORTED"


# ─────────────────────────────────────────────────────────────────────
# 4. AGGREGATE RESULTS — verdict from processed claims
# ─────────────────────────────────────────────────────────────────────

def aggregate_results(
    processed_claims: List[Dict],
) -> Dict:
    """
    Count labels and decide strict final verdict.

    Rules:
    - ANY claim == CONTRADICTION → HALLUCINATED
    - ANY claim == UNSUPPORTED → HALLUCINATED
    - Else → FAITHFUL
    """
    n = len(processed_claims)
    if n == 0:
        return {
            "verdict": "FAITHFUL",
            "entailed_count": 0,
            "unsupported_count": 0,
            "contradiction_count": 0,
            "entailed_fraction": 1.0,
            "unsupported_fraction": 0.0,
            "avg_similarity": 1.0,
        }

    entailed_count = 0
    unsupported_count = 0
    contradiction_count = 0
    total_sim = 0.0

    for c in processed_claims:
        label = c["final_label"]
        total_sim += c["similarity"]
        if label == "ENTAILED":
            entailed_count += 1
        elif label == "CONTRADICTION":
            contradiction_count += 1
        else:
            unsupported_count += 1

    avg_sim = total_sim / n

    if contradiction_count > 0:
        verdict = "HALLUCINATED"
    elif unsupported_count > 0:
        verdict = "HALLUCINATED"
    else:
        verdict = "FAITHFUL"

    return {
        "verdict": verdict,
        "entailed_count": entailed_count,
        "unsupported_count": unsupported_count,
        "contradiction_count": contradiction_count,
        "entailed_fraction": round(entailed_count / n, 4),
        "unsupported_fraction": round(unsupported_count / n, 4),
        "avg_similarity": round(avg_sim, 4),
    }


# ─────────────────────────────────────────────────────────────────────
# 5. COMPUTE CONFIDENCE — calibrated, never 1.00
# ─────────────────────────────────────────────────────────────────────

def compute_confidence(metrics: Dict) -> float:
    """
    Fix: Reduce NLI dominance. Use strict weights:
    confidence = 0.6 * avg_sim + 0.4 * entailed_fraction
    Clamped to [0.50, 0.95], rounded to 2dp. NEVER returns 1.00.
    """
    avg_sim = metrics.get("avg_similarity", 0.0)
    ent_frac = metrics.get("entailed_fraction", 0.0)

    raw = 0.6 * avg_sim + 0.4 * ent_frac

    return round(min(max(raw, 0.50), 0.95), 2)


# ─────────────────────────────────────────────────────────────────────
# 6. GENERATE SUMMARY — human-readable 1-liner
# ─────────────────────────────────────────────────────────────────────

def generate_summary(metrics: Dict) -> str:
    """Produce a strict summary sentence."""
    unsup = metrics.get("unsupported_count", 0)
    contr = metrics.get("contradiction_count", 0)

    if contr == 0 and unsup == 0:
        return "All claims are supported by the source → response is faithful"

    return (
        f"{unsup} unsupported and {contr} contradictory "
        f"claims detected → response is hallucinated"
    )


# ─────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY-POINT — called by pipeline.py
# ─────────────────────────────────────────────────────────────────────

def aggregate(
    claim_nli_results: List[Dict],
    alignment_matrix: List[List[float]],
) -> Tuple[str, float, str, List[Dict], Dict]:
    """
    Full aggregation pipeline:

    Returns: (verdict, confidence, summary, processed_claims, metrics_dict)
    """
    if not claim_nli_results:
        return (
            "FAITHFUL",
            0.95,
            "No claims extracted → response is faithful",
            [],
            {
                "entailed_fraction": 1.0,
                "unsupported_fraction": 0.0,
                "contradiction_count": 0,
                "avg_similarity": 1.0,
            },
        )

    processed: List[Dict] = []

    for i, claim_res in enumerate(claim_nli_results):
        nli_label = claim_res.get("label", "")

        # Get max cosine similarity from alignment matrix row
        if (
            alignment_matrix
            and i < len(alignment_matrix)
            and alignment_matrix[i]
        ):
            raw_sim = max(alignment_matrix[i])
        else:
            raw_sim = 0.0

        similarity = _safe_similarity(raw_sim)
        final_label = classify_claim(nli_label, similarity)

        processed.append({
            "text": claim_res.get("text", ""),
            "nli_label": _map_nli_label(nli_label).lower(),
            "similarity": similarity,
            "final_label": final_label,
        })

    metrics = aggregate_results(processed)
    confidence = compute_confidence(metrics)
    summary = generate_summary(metrics)
    verdict = metrics["verdict"]

    metrics_out = {
        "entailed_fraction": metrics["entailed_fraction"],
        "unsupported_fraction": metrics["unsupported_fraction"],
        "contradiction_count": metrics["contradiction_count"],
        "avg_similarity": metrics["avg_similarity"],
    }

    return verdict, confidence, summary, processed, metrics_out
