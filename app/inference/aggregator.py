from __future__ import annotations
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────
# 1. NLI LABEL MAPPING
# ─────────────────────────────────────────────

_NLI_MAP = {
    "entailment": "ENTAILED",
    "contradiction": "CONTRADICTION",
    "neutral": "UNSUPPORTED",
}


def _map_nli_label(raw: str) -> str:
    if not raw:
        return "UNSUPPORTED"
    return _NLI_MAP.get(raw.lower().strip(), "UNSUPPORTED")


# ─────────────────────────────────────────────
# 2. SAFE SIMILARITY (NO ZERO)
# ─────────────────────────────────────────────

def _safe_similarity(raw_value: float) -> float:
    if raw_value is None or raw_value <= 0.0:
        return 0.3
    return round(float(raw_value), 2)


# ─────────────────────────────────────────────
# 3. CLASSIFICATION (FINAL FIX)
# ─────────────────────────────────────────────

def classify_claim(nli_label: str, similarity: float) -> str:
    mapped = _map_nli_label(nli_label)

    # 1. Hard contradiction
    if mapped == "CONTRADICTION":
        return "CONTRADICTION"

    # 2. Strong similarity override (MOST IMPORTANT FIX)
    if similarity >= 0.75:
        return "ENTAILED"

    # 3. Medium similarity
    if similarity >= 0.70:
        return "ENTAILED"

    # 4. NLI entailment
    if mapped == "ENTAILED":
        return "ENTAILED"

    # 5. Default
    return "UNSUPPORTED"


# ─────────────────────────────────────────────
# 4. AGGREGATION (STRICT)
# ─────────────────────────────────────────────

def aggregate_results(processed_claims: List[Dict]) -> Dict:
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

    entailed = unsupported = contradiction = 0
    total_sim = 0.0

    for c in processed_claims:
        label = c["final_label"]
        sim = c["similarity"]

        total_sim += sim

        if label == "ENTAILED":
            entailed += 1
        elif label == "CONTRADICTION":
            contradiction += 1
        else:
            unsupported += 1

    avg_sim = total_sim / n

    # STRICT VERDICT (NEVER LEAK CLAIM LABEL)
    if contradiction > 0:
        verdict = "HALLUCINATED"
    elif unsupported > 0:
        verdict = "HALLUCINATED"
    else:
        verdict = "FAITHFUL"

    return {
        "verdict": verdict,
        "entailed_count": entailed,
        "unsupported_count": unsupported,
        "contradiction_count": contradiction,
        "entailed_fraction": round(entailed / n, 4),
        "unsupported_fraction": round(unsupported / n, 4),
        "avg_similarity": round(avg_sim, 4),
    }


# ─────────────────────────────────────────────
# 5. CONFIDENCE (NO 1.00 EVER)
# ─────────────────────────────────────────────

def compute_confidence(metrics: Dict) -> float:
    avg_sim = metrics.get("avg_similarity", 0.0)
    ent_frac = metrics.get("entailed_fraction", 0.0)

    raw = 0.6 * avg_sim + 0.4 * ent_frac

    return round(min(max(raw, 0.50), 0.95), 2)


# ─────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────

def generate_summary(metrics: Dict) -> str:
    unsup = metrics.get("unsupported_count", 0)
    contr = metrics.get("contradiction_count", 0)

    if contr == 0 and unsup == 0:
        return "All claims are supported by the source → response is faithful"

    return f"{unsup} unsupported and {contr} contradictory claims detected → response is hallucinated"


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def aggregate(
    claim_nli_results: List[Dict],
    alignment_matrix: List[List[float]],
) -> Tuple[str, float, str, List[Dict], Dict]:

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

    processed = []

    for i, claim in enumerate(claim_nli_results):
        nli_label = claim.get("label", "")

        # SAFE SIMILARITY EXTRACTION
        if alignment_matrix and i < len(alignment_matrix) and alignment_matrix[i]:
            raw_sim = max(alignment_matrix[i])
        else:
            raw_sim = 0.0

        similarity = _safe_similarity(raw_sim)

        final_label = classify_claim(nli_label, similarity)

        processed.append({
            "text": claim.get("text", ""),
            "nli_label": _map_nli_label(nli_label),  # FIXED (no .lower())
            "similarity": similarity,
            "final_label": final_label,
        })

    metrics = aggregate_results(processed)
    confidence = compute_confidence(metrics)
    summary = generate_summary(metrics)

    return (
        metrics["verdict"],
        confidence,
        summary,
        processed,
        {
            "entailed_fraction": metrics["entailed_fraction"],
            "unsupported_fraction": metrics["unsupported_fraction"],
            "contradiction_count": metrics["contradiction_count"],
            "avg_similarity": metrics["avg_similarity"],
        },
    )