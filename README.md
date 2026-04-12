<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Hallucination%20Hunter&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35&desc=AI%20Fact-Checking%2C%20Engineered%20for%20Truth&descAlignY=55&descSize=18" width="100%"/>

<br/>

<!-- BADGES -->
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/harsh/hallucination-hunter?style=for-the-badge&logo=github&color=yellow)](https://github.com/harsh/hallucination-hunter/stargazers)

<br/>

> **"Don't trust. Verify."** — Built to catch what LLMs get wrong.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

</div>

<br/>

## 📌 The Problem

Large Language Models (LLMs) **hallucinate**. They confidently generate text that sounds factually correct — but isn't. Developers, researchers, and businesses integrating AI into production systems have no reliable, automated way to catch these fabrications before they reach end users.

> **Example:** Ask an LLM about the Eiffel Tower, and it might tell you it's in Berlin. With high confidence. And a smile.

There was no lightweight, self-hostable, API-first tool for **claim-level** hallucination detection — until now.

<br/>

## 💡 The Solution

**Hallucination Hunter** is a production-ready REST API that takes a **source passage** and an **AI-generated response**, breaks the response into individual claims, and verifies each one using a combination of:

- 🔬 **Neural NLI** (Natural Language Inference) via DeBERTa-v3
- 📐 **Semantic Similarity** via Sentence Transformers
- 🧩 **Claim Extraction** via spaCy
- 🔁 **Optional Ensemble** via Logistic Regression / XGBoost

Every claim gets a verdict. The response gets a verdict. You get the truth.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🚀 Features

| Feature | Description |
|---|---|
| 🎯 **Claim-Level Detection** | Breaks AI responses into atomic claims and verifies each one independently |
| ⚡ **REST API** | FastAPI-powered endpoints ready for production integration |
| 🧠 **State-of-the-Art NLI** | Uses `cross-encoder/nli-deberta-v3-small` for deep entailment reasoning |
| 📊 **Alignment Heatmap** | Visual matrix showing semantic similarity between claims and source sentences |
| 🔢 **Confidence Scores** | Every verdict comes with a calibrated confidence value |
| 🧩 **Explainable Output** | Know *which* claims are wrong, not just that something is wrong |
| 🐳 **Docker Ready** | One command to containerize and ship |
| 📦 **Benchmark Support** | Integrates HaluEval (35K samples) and TRUE (11 datasets) for training & eval |
| 🔧 **Fully Configurable** | Tweak thresholds, models, and batch sizes from a single config file |
| 🌐 **Interactive Docs** | Swagger UI auto-generated at `/docs` |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🛠 Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat-square)](https://huggingface.co)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white)](https://spacy.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)](https://matplotlib.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Uvicorn](https://img.shields.io/badge/Uvicorn-4051B5?style=flat-square)](https://www.uvicorn.org)

</div>

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI + Uvicorn | High-performance async REST API |
| **NLI Model** | DeBERTa-v3-small (cross-encoder) | Per-claim entailment classification |
| **Embeddings** | all-mpnet-base-v2 | Semantic similarity scoring |
| **Claim Extraction** | spaCy (en_core_web_sm) | Sentence segmentation + filtering |
| **Ensemble** | scikit-learn / XGBoost | Optional stacking classifier |
| **Visualization** | Matplotlib + Seaborn | Alignment heatmap generation |
| **Data** | HaluEval + TRUE | Training & evaluation benchmarks |
| **Containerization** | Docker | One-command deployment |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 📂 Project Structure

```
hallucination_hunter/
│
├── 📁 app/                          # Core application
│   ├── 🐍 main.py                   # FastAPI entry point
│   ├── 📁 api/
│   │   └── 🐍 routes.py             # POST /detect, /detect/visualize, /health
│   ├── 📁 inference/
│   │   ├── 🐍 similarity.py         # Sentence-Transformers cosine similarity
│   │   ├── 🐍 nli.py                # DeBERTa-v3 NLI cross-encoder
│   │   ├── 🐍 aggregator.py         # Aggregation logic → final verdict
│   │   ├── 🐍 pipeline.py           # Full orchestration pipeline
│   │   └── 🐍 ensemble.py           # Optional LR / XGBoost ensemble
│   ├── 📁 models/
│   │   └── 🐍 schemas.py            # Pydantic request / response schemas
│   ├── 📁 claims/
│   │   └── 🐍 extractor.py          # spaCy claim extraction + filtering
│   └── 📁 utils/
│       ├── 🐍 helpers.py            # Timer, truncation utilities
│       └── 🐍 visualization.py      # Alignment heatmap (matplotlib/seaborn)
│
├── 📁 training/
│   ├── 🐍 data_pipeline.py          # HaluEval + TRUE loader, preprocessing
│   └── 🐍 train_ensemble.py         # Train the ensemble classifier
│
├── 📁 evaluation/
│   └── 🐍 evaluate.py               # Balanced accuracy, P/R/F1, ROC AUC
│
├── 📁 configs/
│   └── 🐍 settings.py               # All configuration in one place
│
├── 📁 data/                          # Auto-generated after pipeline runs
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
│
├── 📁 models/                        # Saved ensemble weights
│   └── ensemble.pkl
│
├── 🐳 Dockerfile
├── 📋 requirements.txt
└── 📄 README.md
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## ⚙️ Installation & Setup

### ✅ Prerequisites

Before you start, make sure you have the following installed:

| Tool | Version | Download |
|---|---|---|
| Python | 3.9+ | [python.org](https://www.python.org/downloads/) |
| Git | Any | [git-scm.com](https://git-scm.com/downloads) |
| Docker *(optional)* | Any | [docker.com](https://docker.com) |

Verify your setup:

```bash
python --version   # Should output Python 3.9+
git --version      # Should output git version x.x.x
```

---

### 📥 Step 1 — Clone the Repository

```bash
git clone https://github.com/harsh/hallucination-hunter.git
cd hallucination-hunter
```

---

### 📦 Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

Then download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

### 🚀 Step 3 — Start the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

### 🌐 Step 4 — Open Interactive Docs

Open your browser and visit:

```
http://localhost:8000/docs
```

You'll see a fully interactive Swagger UI to test every endpoint — no code needed.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 📡 API Reference

### `POST /detect`

Detects hallucinations in an AI-generated response given a source passage.

**Request Body:**

```json
{
  "source": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889.",
  "response": "The Eiffel Tower is located in Berlin, Germany. It was built in 1920 for the Olympic Games."
}
```

**Terminal Command:**

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "source": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889.",
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

---

### `POST /detect/visualize`

Same as `/detect` but additionally returns a **base64-encoded PNG heatmap** of the claim-to-source semantic alignment matrix.

```bash
curl -X POST http://localhost:8000/detect/visualize \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Python was created by Guido van Rossum in 1991.",
    "response": "Python was invented by Linus Torvalds in 2005."
  }'
```

**Additional field in response:**

```json
{
  "label": "HALLUCINATED",
  "confidence": 0.95,
  "heatmap_png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok" }
```

---

### 📊 Claim Labels Reference

| Label | Meaning |
|---|---|
| `entailment` | Claim is supported by the source ✅ |
| `neutral` | Claim is neither supported nor contradicted 🟡 |
| `contradiction` | Claim directly conflicts with the source ❌ |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🧠 How It Works — Pipeline Deep Dive

```
┌─────────────────────────────────────────────────────┐
│                   AI Response Text                   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────┐
           │     Claim Extraction     │
           │  spaCy sentence splits   │
           │  + short-claim filter    │
           └────────────┬────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
 ┌─────────────────────┐ ┌─────────────────────────┐
 │   NLI Classification │ │  Semantic Similarity     │
 │  DeBERTa-v3-small    │ │  all-mpnet-base-v2       │
 │  (cross-encoder)     │ │  cosine similarity matrix│
 └──────────┬──────────┘ └────────────┬────────────┘
            │                         │
            ▼                         ▼
 ┌─────────────────────┐ ┌─────────────────────────┐
 │   Aggregation Rules  │ │   Alignment Heatmap      │
 │  entailment fraction │ │   matplotlib/seaborn     │
 │  + confidence score  │ │   base64 PNG export      │
 └──────────┬──────────┘ └─────────────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ FAITHFUL  ╱╲    │
   │ HALLUCINATED    │
   └─────────────────┘
```

### Aggregation Logic

| Condition | Verdict |
|---|---|
| ≥ 60% of claims are `entailment` | ✅ `FAITHFUL` |
| Any claim is `contradiction` with score > threshold | ❌ `HALLUCINATED` |
| Similarity score < 0.65 baseline | ❌ `HALLUCINATED` |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 📦 Data Pipeline

Supports two major hallucination detection benchmarks out of the box:

| Benchmark | Source | Samples | Tasks |
|---|---|---|---|
| **HaluEval** | RUCAIBox/HaluEval | 35,000 | QA, Dialogue, Summarization, General |
| **TRUE** | google-research/true | 11 datasets | FRANK, QAGS, BEGIN, Q², DialFact, VitaminC, PAWS |

### Run HaluEval Only

```bash
# Single task split
python -m training.data_pipeline \
  --source halueval \
  --halueval-split qa \
  --max-samples 1000

# All HaluEval splits combined
python -m training.data_pipeline \
  --source halueval \
  --halueval-split all
```

### Run TRUE Benchmark Only

```bash
# Lightweight default datasets
python -m training.data_pipeline --source true

# Specific datasets
python -m training.data_pipeline \
  --source true \
  --true-datasets frank,qags_cnndm,vitc,begin
```

### Run Combined (Strongest Generalization)

```bash
python -m training.data_pipeline \
  --source combined \
  --halueval-split all \
  --true-datasets frank,qags_cnndm,begin
```

**Output files created in `data/`:**

```
data/
├── train.parquet
├── val.parquet
└── test.parquet
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🏋️ Training — Optional Ensemble Classifier

The optional ensemble stacks NLI and similarity features into a meta-classifier for higher accuracy.

### Train on HaluEval QA

```bash
python -m training.train_ensemble \
  --source halueval \
  --halueval-split qa \
  --max-samples 200
```

### Train on Combined Data with XGBoost

```bash
python -m training.train_ensemble \
  --source combined \
  --halueval-split all \
  --true-datasets frank,qags_cnndm,begin \
  --max-samples 300 \
  --model-type xgboost \
  --output models/ensemble.pkl
```

Saved weights land in `models/ensemble.pkl` and are automatically loaded by the pipeline.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 📈 Evaluation

### Evaluate on Cached Test Split

```bash
python -m evaluation.evaluate \
  --data-dir data \
  --split test \
  --max-samples 100
```

### Evaluate on HaluEval QA Directly

```bash
python -m evaluation.evaluate \
  --source halueval \
  --halueval-split qa \
  --max-samples 50
```

### Evaluate on TRUE Benchmark

```bash
python -m evaluation.evaluate \
  --source true \
  --true-datasets frank,qags_cnndm \
  --max-samples 50
```

### Full Evaluation with Per-Dataset Breakdown

```bash
python -m evaluation.evaluate \
  --source combined \
  --max-samples 100 \
  --output evaluation/metrics.json
```

**Metrics reported:**

```
✔ Balanced Accuracy
✔ Precision / Recall / F1
✔ ROC AUC
✔ Confusion Matrix
✔ Per-dataset Breakdown
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t hallucination-hunter .
```

### Run the Container

```bash
docker run -p 8000:8000 hallucination-hunter
```

### Quick Test After Launch

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Python was created by Guido van Rossum.",
    "response": "Python was created by Linus Torvalds."
  }'
```

**Expected:**

```json
{
  "label": "HALLUCINATED",
  "confidence": 0.96
}
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🧪 Google Colab — Zero Setup Testing

Want to try it instantly without any local setup? Run it in Colab:

```python
# Step 1 — Install everything
!pip install fastapi uvicorn torch transformers sentence-transformers \
             spacy scikit-learn pandas datasets matplotlib seaborn tqdm xgboost
!python -m spacy download en_core_web_sm

# Step 2 — Set path
import sys
sys.path.insert(0, '/content/hallucination_hunter')

# Step 3 — Run detection
from app.inference.pipeline import detect

result = detect(
    source="The Great Wall of China is over 13,000 miles long.",
    response="The Great Wall of China is 500 miles long and located in Japan."
)

print(result.label)         # HALLUCINATED
print(result.confidence)    # ~0.9

for claim in result.claims:
    print(f"  [{claim.label.upper()}] {claim.text} (score: {claim.score:.2f})")
```

### Run Full API in Colab via ngrok

```python
!pip install pyngrok
from pyngrok import ngrok
import nest_asyncio
nest_asyncio.apply()

# Create a public tunnel
public_url = ngrok.connect(8000)
print(f"🌐 Public URL: {public_url}")

# Launch server
!uvicorn app.main:app --host 0.0.0.0 --port 8000
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## ⚙️ Configuration

All settings live in one file: `configs/settings.py`

| Setting | Default | Description |
|---|---|---|
| `similarity_threshold` | `0.65` | Minimum cosine similarity to pass baseline check |
| `entailment_fraction_threshold` | `0.60` | Min fraction of claims needing entailment for FAITHFUL |
| `nli_batch_size` | `32` | Batch size for NLI model inference |
| `max_claims` | `50` | Maximum claims extracted per response |
| `nli_model_id` | `cross-encoder/nli-deberta-v3-small` | HuggingFace model ID for NLI |
| `similarity_model_id` | `all-mpnet-base-v2` | HuggingFace model ID for embeddings |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🗺️ Roadmap

Here's what's coming next:

- [ ] 🖼️ **Web UI** — Drag-and-drop interface for non-technical users
- [ ] 🌍 **Multilingual Support** — Detect hallucinations in French, German, Hindi, and more
- [ ] 📎 **PDF / Document Input** — Verify responses grounded in uploaded documents
- [ ] 🔌 **LangChain Integration** — Plug-in hallucination guard for LangChain pipelines
- [ ] 🧪 **GPT-4 / Claude Comparison Mode** — Side-by-side LLM hallucination rate benchmarks
- [ ] 📬 **Webhook Support** — Fire detection results to Slack, Discord, or any webhook URL
- [ ] 📦 **PyPI Package** — `pip install hallucination-hunter`

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🤝 Contributing

Contributions are what make open source incredible. Every PR, issue, and idea is welcome.

### How to Contribute

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/hallucination-hunter.git
cd hallucination-hunter

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes and commit
git add .
git commit -m "feat: describe your change clearly"

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Open a Pull Request on GitHub 🎉
```

### Contribution Guidelines

- 🧹 Keep code clean and well-commented
- ✅ Add tests for new features where possible
- 📝 Update the README if your change affects usage
- 🐛 Use GitHub Issues for bug reports and feature requests
- 💬 Be kind — this is a welcoming space

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 📜 License

```
MIT License

Copyright (c) 2026 Harsh Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

## 🌟 Support the Project

If Hallucination Hunter helped you catch an AI lie, consider:

<div align="center">

⭐ **Starring the repository** — it helps others discover this tool

🍴 **Forking it** — build something great on top of it

🐛 **Reporting issues** — help make it better

📢 **Sharing it** — tweet it, blog it, or just tell a friend

</div>

<br/>

<div align="center">

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>

**Made with ❤️ by [Harsh Kumar](https://github.com/harsh)**

*"The truth is out there. Now you have a tool to find it."*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-harsh-181717?style=flat-square&logo=github)](https://github.com/harsh)

</div>
