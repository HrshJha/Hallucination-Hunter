"""
Smoke test: verify the refactored pipeline produces correct results
for all 4 mandatory test scenarios.
"""

import sys
import os
import json

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.inference.pipeline import detect


def run(name, source, response, expected_verdict):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Expected: {expected_verdict}")
    print(f"{'='*60}")

    result = detect(source, response)
    d = result.model_dump()

    # Assertions
    ok = True

    # 1. Verdict correct
    if d["verdict"] != expected_verdict:
        print(f"  ❌ VERDICT: got {d['verdict']}, expected {expected_verdict}")
        ok = False

    # 2. No UNKNOWN anywhere
    for c in d["claims"]:
        if "UNKNOWN" in c["final_label"]:
            print(f"  ❌ UNKNOWN label found: {c}")
            ok = False

    # 3. No 0.00 similarity
    for c in d["claims"]:
        if c["similarity"] <= 0.0:
            print(f"  ❌ Similarity is 0.00: {c}")
            ok = False

    # 4. Confidence never 1.00
    if d["confidence"] >= 1.0:
        print(f"  ❌ Confidence is {d['confidence']} (>=1.0)")
        ok = False

    # 5. Summary exists
    if not d["summary"]:
        print(f"  ❌ Summary is empty")
        ok = False

    if ok:
        print(f"  ✅ PASSED")

    # Print compact output
    print(f"  Verdict:    {d['verdict']}")
    print(f"  Confidence: {d['confidence']}")
    print(f"  Summary:    {d['summary']}")
    for c in d["claims"]:
        print(f"    [{c['final_label']:15s}] sim={c['similarity']:.2f}  nli={c['nli_label']:14s}  → {c['text'][:60]}")

    return ok


if __name__ == "__main__":
    results = []

    results.append(run(
        "1. Pure paraphrase → FAITHFUL",
        source="The quick brown fox jumps over the lazy dog.",
        response="A fast brown-colored fox leaps across a sleeping dog.",
        expected_verdict="FAITHFUL",
    ))

    results.append(run(
        "2. Direct contradiction → HALLUCINATED",
        source="Python was released in 1991 by Guido van Rossum.",
        response="Python was released in 2005 by Linus Torvalds.",
        expected_verdict="HALLUCINATED",
    ))

    results.append(run(
        "3. Mixed claims → HALLUCINATED",
        source="The Eiffel Tower is located in Paris, France. It is made of iron.",
        response="The Eiffel Tower is located in Paris, France, and it is made of solid gold.",
        expected_verdict="HALLUCINATED",
    ))

    results.append(run(
        "4. Unsupported extra info → HALLUCINATED",
        source="The sky is blue today due to Rayleigh scattering.",
        response="The sky is blue today. Additionally, tomatoes are fruits.",
        expected_verdict="HALLUCINATED",
    ))

    print(f"\n{'='*60}")
    print(f"RESULTS: {sum(results)}/{len(results)} passed")
    print(f"{'='*60}")
