"""
Colab-friendly standalone script.
Run this as a single cell in Google Colab to test the full pipeline
without starting the FastAPI server.
"""

# ── 1. Install ──────────────────────────────────────────────────────
# !pip install -q torch transformers sentence-transformers spacy scikit-learn pandas datasets matplotlib seaborn tqdm
# !python -m spacy download en_core_web_sm

import sys
import os

# If running from within hallucination_hunter/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── 2. Run detection ────────────────────────────────────────────────

from app.inference.pipeline import detect
from app.utils.visualization import plot_alignment_matrix
from app.claims.extractor import extract_source_sentences

# Example 1: Faithful response
source_1 = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair. "
    "The tower is 330 metres tall and was the tallest man-made structure in the world until 1930."
)

response_1 = (
    "The Eiffel Tower is located in Paris, France on the Champ de Mars. "
    "It was built between 1887 and 1889 for the 1889 World's Fair. "
    "The tower stands at 330 metres in height."
)

print("=" * 60)
print("EXAMPLE 1: Expected FAITHFUL")
print("=" * 60)
result_1 = detect(source_1, response_1)
print(f"Label:      {result_1.label}")
print(f"Confidence: {result_1.confidence}")
print(f"Claims:")
for c in result_1.claims:
    print(f"  [{c.label}] {c.text} (score={c.score:.3f})")

# Example 2: Hallucinated response
source_2 = (
    "Python is a high-level programming language created by Guido van Rossum. "
    "It was first released in 1991. Python emphasizes code readability "
    "and supports multiple programming paradigms."
)

response_2 = (
    "Python was created by Linus Torvalds in 2005. "
    "It is primarily used for operating system development. "
    "Python only supports object-oriented programming."
)

print("\n" + "=" * 60)
print("EXAMPLE 2: Expected HALLUCINATED")
print("=" * 60)
result_2 = detect(source_2, response_2)
print(f"Label:      {result_2.label}")
print(f"Confidence: {result_2.confidence}")
print(f"Claims:")
for c in result_2.claims:
    print(f"  [{c.label}] {c.text} (score={c.score:.3f})")

# ── 3. Visualization ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ALIGNMENT MATRIX VISUALIZATION")
print("=" * 60)

claims_text = [c.text for c in result_2.claims]
source_sents = extract_source_sentences(source_2)

img_bytes = plot_alignment_matrix(
    result_2.alignment_matrix,
    claim_labels=claims_text,
    source_labels=source_sents,
    save_path="alignment_heatmap.png",
)
print("Heatmap saved to alignment_heatmap.png")

# In Colab, display inline:
# from IPython.display import Image, display
# display(Image("alignment_heatmap.png"))

print("\n✅ All tests passed!")
