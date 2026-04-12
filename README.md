# 🔍 Hallucination Hunter

**Production-ready factual consistency & hallucination detection for AI-generated responses.**

Hallucination Hunter takes a source passage and an AI-generated answer, then returns a **FAITHFUL / HALLUCINATED** verdict with per-claim NLI breakdown and an alignment heatmap.

---

## 🏗 Architecture

```
hallucination_hunter/
│
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── api/
│   │   └── routes.py            # POST /detect, /detect/visualize, /health
│   ├── inference/
│   │   ├── similarity.py        # Baseline: sentence-transformers cosine similarity
│   │   ├── nli.py               # Core: cross-encoder NLI (DeBERTa-v3-small)
│   │   ├── aggregator.py        # Aggregation rules → final verdict
│   │   ├── pipeline.py          # Full orchestration pipeline
│   │   └── ensemble.py          # Optional ensemble (LR / XGBoost)
│   ├── models/
│   │   └── schemas.py           # Pydantic request / response models
│   ├── claims/
│   │   └── extractor.py         # spaCy claim extraction + filtering
│   └── utils/
│       ├── helpers.py           # Timer, truncation helpers
│       └── visualization.py     # Alignment matrix heatmap (matplotlib/seaborn)
│
├── training/
│   ├── data_pipeline.py         # HaluEval loader + preprocessing + splitting
│   └── train_ensemble.py        # Train ensemble classifier
│
├── evaluation/
│   └── evaluate.py              # Balanced accuracy, precision, recall, F1
│
├── configs/
│   └── settings.py              # All configuration in one place
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd hallucination_hunter
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the API

```bash
# From hallucination_hunter/ directory
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Open Interactive Docs

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Usage

### POST `/detect`

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "source": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World Fair.",
    "response": "The Eiffel Tower is located in Berlin, Germany. It was built in 1920 for the Olympic Games."
  }'
```

**Response:**

```json
{
  "label": "HALLUCINATED",
  "confidence": 0.92,
  "claims": [
    {
      "text": "The Eiffel Tower is located in Berlin, Germany.",
      "label": "contradiction",
      "score": 0.97
    },
    {
      "text": "It was built in 1920 for the Olympic Games.",
      "label": "contradiction",
      "score": 0.94
    }
  ],
  "alignment_matrix": [[0.45, 0.12], [0.08, 0.31]]
}
```

### POST `/detect/visualize`

Same as `/detect` but also returns `heatmap_png_base64` — a base64-encoded PNG alignment heatmap.

### GET `/health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## 🧠 Models

| Model | Purpose | HuggingFace ID |
|-------|---------|----------------|
| Similarity Baseline | Cosine similarity between embeddings | `sentence-transformers/all-mpnet-base-v2` |
| NLI Engine (Core) | Per-claim entailment/contradiction/neutral | `cross-encoder/nli-deberta-v3-small` |
| Ensemble (Optional) | Combines similarity + NLI features | Logistic Regression / XGBoost |

---

## 📊 Pipeline Flow

```
AI Response
    │
    ▼
┌─────────────────────┐
│  Claim Extraction   │  (spaCy sentence segmentation + filtering)
│  (extractor.py)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│  NLI Classification │     │  Similarity Matrix   │
│  (nli.py)           │     │  (similarity.py)     │
└─────────┬───────────┘     └─────────┬───────────┘
          │                           │
          ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Aggregation Rules  │     │  Alignment Heatmap   │
│  (aggregator.py)    │     │  (visualization.py)  │
└─────────┬───────────┘     └─────────────────────┘
          │
          ▼
   FAITHFUL / HALLUCINATED
```

---

## 📦 Data Pipeline

Supports two major hallucination benchmarks:

| Benchmark | Source | Samples | Tasks |
|-----------|--------|---------|-------|
| **HaluEval** | [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval) | 35K | QA, Dialogue, Summarization, General |
| **TRUE** | [google-research/true](https://github.com/google-research/true) | 11 datasets | FRANK, QAGS, BEGIN, Q², DialFact, VitaminC, PAWS |

### HaluEval only

```bash
# Single split (qa / dialogue / summarization / general)
python -m training.data_pipeline --source halueval --halueval-split qa --max-samples 1000

# All HaluEval splits combined
python -m training.data_pipeline --source halueval --halueval-split all
```

### TRUE benchmark only

```bash
# Default lightweight TRUE datasets (FRANK, QAGS, BEGIN, Q², DialFact)
python -m training.data_pipeline --source true

# Specific TRUE datasets
python -m training.data_pipeline --source true --true-datasets frank,qags_cnndm,vitc,begin
```

### Combined (HaluEval + TRUE)

```bash
python -m training.data_pipeline --source combined --halueval-split all --true-datasets frank,qags_cnndm,begin
```

Outputs: `data/train.parquet`, `data/val.parquet`, `data/test.parquet`

---

## 🏋️ Training (Optional Ensemble)

```bash
# Train on HaluEval QA
python -m training.train_ensemble --source halueval --halueval-split qa --max-samples 200

# Train on combined HaluEval + TRUE for stronger generalization
python -m training.train_ensemble \
  --source combined \
  --halueval-split all \
  --true-datasets frank,qags_cnndm,begin \
  --max-samples 300 \
  --model-type xgboost \
  --output models/ensemble.pkl
```

---

## 📈 Evaluation

```bash
# Evaluate on cached test split
python -m evaluation.evaluate --data-dir data --split test --max-samples 100

# Evaluate directly on HaluEval QA
python -m evaluation.evaluate --source halueval --halueval-split qa --max-samples 50

# Evaluate on TRUE benchmark
python -m evaluation.evaluate --source true --true-datasets frank,qags_cnndm --max-samples 50

# Evaluate on combined data with per-dataset breakdown
python -m evaluation.evaluate --source combined --max-samples 100 --output evaluation/metrics.json
```

Outputs: balanced accuracy, precision, recall, F1, ROC AUC, confusion matrix, per-dataset breakdown.

---

## 🐳 Docker

### Build

```bash
docker build -t hallucination-hunter .
```

### Run

```bash
docker run -p 8000:8000 hallucination-hunter
```

### Test

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"source": "Python was created by Guido van Rossum.", "response": "Python was created by Linus Torvalds."}'
```

---

## 🧪 Google Colab

```python
# Install dependencies
!pip install fastapi uvicorn torch transformers sentence-transformers spacy scikit-learn pandas datasets matplotlib seaborn tqdm xgboost
!python -m spacy download en_core_web_sm

# Clone or upload the hallucination_hunter/ folder, then:
import sys
sys.path.insert(0, '/content/hallucination_hunter')

from app.inference.pipeline import detect

result = detect(
    source="The Great Wall of China is over 13,000 miles long.",
    response="The Great Wall of China is 500 miles long and located in Japan."
)

print(result.label)       # HALLUCINATED
print(result.confidence)  # ~0.9
for c in result.claims:
    print(f"  {c.label}: {c.text} ({c.score:.2f})")
```

### Run API in Colab with ngrok

```python
!pip install pyngrok
from pyngrok import ngrok
import nest_asyncio
nest_asyncio.apply()

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Run server
!uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## ⚙️ Configuration

All settings in `configs/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `similarity_threshold` | 0.65 | Baseline similarity cutoff |
| `entailment_fraction_threshold` | 0.60 | Min entailed fraction for FAITHFUL |
| `nli_batch_size` | 32 | NLI inference batch size |
| `max_claims` | 50 | Max claims extracted per response |

---

## 📜 License

MIT
