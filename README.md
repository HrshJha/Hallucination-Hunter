<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=220&section=header&text=🔍%20Hallucination%20Hunter&fontSize=52&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI%20doesn't%20always%20tell%20the%20truth.%20Now%20you%20can%20prove%20it.&descAlignY=58&descSize=16&descColor=ccccff" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/hrshjha/hallucination-hunter?style=for-the-badge&logo=github&color=f59e0b&labelColor=1a1a2e)](https://github.com/hrshjha/hallucination-hunter)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-22c55e?style=for-the-badge&logo=git&logoColor=white)](CONTRIBUTING.md)

<br/>

<table>
<tr>
<td align="center">🎯<br/><b>Claim-Level</b><br/>Verification</td>
<td align="center">⚡<br/><b>REST API</b><br/>Ready</td>
<td align="center">🧠<br/><b>NLI-Powered</b><br/>Reasoning</td>
<td align="center">📊<br/><b>Explainable</b><br/>Output</td>
<td align="center">🔌<br/><b>RAG Pipeline</b><br/>Compatible</td>
</tr>
</table>

<br/>

> *"Don't trust. Verify. Automatically."*

<br/>

**Made with ❤️ by [Harsh Kumar](https://github.com/hrshjha)**

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

</div>

<br/>

## 📌 The Problem

Every day, millions of users receive AI-generated answers they **can't verify**. Language models are extraordinarily fluent — but fluency is not the same as accuracy.

```
User asks  →  LLM answers confidently  →  User trusts it  →  User is wrong
```

The failure modes are subtle and dangerous:

| Type | Example |
|---|---|
| 🔴 **Direct Contradiction** | Source says "Paris", model says "Berlin" |
| 🟡 **Unsupported Claim** | Source never mentions the year, model invents one |
| 🟠 **Partial Hallucination** | 80% correct, one quietly fabricated detail |

> Existing tools either flag the entire response or do nothing at all. Neither is useful.

**Hallucination Hunter solves this at the claim level** — every sentence gets its own verdict.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 💡 The Solution

Instead of treating an AI response as a single block of text, Hallucination Hunter **decomposes it into individual claims** and verifies each one independently against your source.

<div align="center">

```
"The Eiffel Tower is in Berlin and was built in 1920."
                        ↓ decompose
     ┌──────────────────────────────────────┐
     │  Claim 1: Eiffel Tower is in Berlin  │  ← ❌ CONTRADICTION
     │  Claim 2: It was built in 1920       │  ← ❌ UNSUPPORTED
     └──────────────────────────────────────┘
                        ↓ aggregate
                   HALLUCINATED (0.78)
```

</div>

This matters because **partial hallucinations are the hardest to catch** — a response can be 90% accurate and still dangerously wrong.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 🚀 Features

- 🔬 &nbsp;**Claim-Level Granularity** — Each sentence in the AI response gets its own verdict, not a single blurry score for the whole response
- 🧠 &nbsp;**Neural NLI Reasoning** — DeBERTa-v3 cross-encoder classifies every claim as `ENTAILED`, `CONTRADICTION`, or `NEUTRAL`
- 📐 &nbsp;**Semantic Similarity Grounding** — Sentence-Transformers cosine similarity anchors claims to the source semantically
- ⚖️ &nbsp;**Conservative Classification** — Neutral predictions are treated as `UNSUPPORTED` to minimize false positives
- 📊 &nbsp;**Alignment Heatmap** — Visual matrix showing exactly how each claim maps to each source sentence
- 📝 &nbsp;**Human-Readable Summary** — Plain-English explanation of *why* the verdict was reached
- ⚡ &nbsp;**FastAPI REST Interface** — Production-ready endpoints, Swagger UI included
- 🔌 &nbsp;**RAG Pipeline Ready** — Drop-in verification layer for any Retrieval-Augmented Generation system
- 🧩 &nbsp;**Extensible Architecture** — Swap NLI models, add extractors, or plug in custom aggregation logic
- 🐳 &nbsp;**Docker Ready** — One command to containerize and deploy anywhere

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 🛠 Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat-square)](https://huggingface.co)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat-square)](https://spacy.io)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat-square)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-4B8BBE?style=flat-square)](https://seaborn.pydata.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)

</div>

<br/>

| Component | Technology | Why |
|---|---|---|
| **API Layer** | FastAPI + Uvicorn | Async, fast, auto-documented |
| **NLI Engine** | `cross-encoder/nli-deberta-v3-small` | Best-in-class entailment model |
| **Embeddings** | `all-mpnet-base-v2` | High-quality semantic similarity |
| **Claim Extractor** | spaCy `en_core_web_sm` | Lightweight, accurate sentence splits |
| **Ensemble** | scikit-learn / XGBoost | Optional meta-classifier stacking |
| **Visualization** | Matplotlib + Seaborn | Alignment heatmap as base64 PNG |
| **Data** | HaluEval + TRUE benchmark | 35K+ annotated hallucination examples |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 📂 Project Structure

```
hallucination_hunter/
│
├── 📁 app/                              # Core backend (FastAPI + ML pipeline)
│   ├── 🐍 main.py                       # App entry point (FastAPI instance)
│   │
│   ├── 📁 api/                          # API layer (routing only)
│   │   └── 🐍 routes.py                 # /detect · /detect/visualize · /health
│   │
│   ├── 📁 core/                         # Core logic (ML pipeline)
│   │   ├── 🐍 pipeline.py               # Orchestrates full flow
│   │   ├── 🐍 aggregator.py             # Final verdict logic
│   │   ├── 🐍 nli.py                    # DeBERTa-v3 NLI cross-encoder
│   │   ├── 🐍 similarity.py             # Semantic similarity scoring
│   │   └── 🐍 ensemble.py               # Optional LR / XGBoost stacking
│   │
│   ├── 📁 claims/                       # Input preprocessing
│   │   └── 🐍 extractor.py              # spaCy segmentation + claim filtering
│   │
│   ├── 📁 schemas/                      # Pydantic request & response models
│   │   └── 🐍 schemas.py
│   │
│   └── 📁 utils/
│       ├── 🐍 helpers.py                # Timers, truncation utilities
│       └── 🐍 visualization.py          # Heatmap → base64 PNG
│
├── 📁 configs/
│   └── 🐍 settings.py                   # Single source of truth for all config
│
├── 📁 data/                             # Auto-generated after pipeline runs
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
│
├── 📁 models/                           # Saved ensemble weights
│   └── ensemble.pkl
│
├── 📁 training/
│   ├── 🐍 data_pipeline.py              # HaluEval + TRUE loader & preprocessor
│   └── 🐍 train_ensemble.py             # Ensemble training script
│
├── 📁 evaluation/
│   └── 🐍 evaluate.py                   # Balanced accuracy, F1, ROC AUC
│
├── 📁 ui/                               # Frontend (browser interface)
│   └── 🌐 index.html
│
├── 📁 scripts/                          # All execution scripts in one place
│   ├── 🐍 run_demo.py
│   ├── 🐍 serve_ui.py
│   └── 🔧 start.sh
│
├── 🐳 Dockerfile
├── 📋 requirements.txt
├── 📄 README.md
└── 🚫 .gitignore
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## ⚙️ Installation & Setup

### Prerequisites

| Requirement | Version | Link |
|---|---|---|
| Python | 3.9 or higher | [python.org](https://www.python.org/downloads/) |
| Git | Any | [git-scm.com](https://git-scm.com/downloads) |
| Docker *(optional)* | Any | [docker.com](https://docker.com) |

Confirm your environment before starting:

```bash
python --version
# Python 3.9.x or higher

git --version
# git version 2.x.x
```

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/hrshjha/hallucination-hunter.git
cd hallucination-hunter
```

---

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

Then download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

### Step 3 — Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

When the server is ready, you'll see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

### Step 4 — Explore the Interactive Docs

```
http://localhost:8000/docs
```

Swagger UI is auto-generated — test every endpoint directly in the browser, no extra tools needed.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 📸 Screenshots & Demo

<div align="center">

| Swagger UI | Heatmap Output | Claim Breakdown |
|:---:|:---:|:---:|
| <a href="https://hallucination-hunter-io2j.onrender.com/docs"><img src="https://raw.githubusercontent.com/HrshJha/Hallucination-Hunter/main/assets/swagger.png" width="280"/></a> | <img src="https://via.placeholder.com/280x180/302b63/ffffff?text=Heatmap+Preview" width="280"/> | <img src="https://via.placeholder.com/280x180/24243e/ffffff?text=Claims+Preview" width="280"/> |
| *Interactive API docs* | *Semantic alignment matrix* | *Per-claim NLI analysis* |

</div>

---

### 🚀 Live Demo

* 🌐 App: https://hallucination-hunter-io2j.onrender.com
* 📡 API Docs: https://hallucination-hunter-io2j.onrender.com/docs

---

### ⚡ Example Request

```bash
curl -X POST https://hallucination-hunter-io2j.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{
    "source": "The Sun rises in the east.",
    "response": "The Sun rises in the east."
  }'
```

---

### 📊 Example Response

```json
{
  "verdict": "FAITHFUL",
  "confidence": 0.97,
  "summary": "All claims are supported by the source.",
  "claims": [],
  "alignment_matrix": [],
  "metrics": {
    "entailed_fraction": 1.0,
    "unsupported_fraction": 0.0,
    "contradiction_count": 0,
    "avg_similarity": 1.0
  }
}
```


## 🧪 Usage

### API Endpoints at a Glance

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/detect` | Returns verdict + per-claim analysis |
| `POST` | `/detect/visualize` | Same as above + base64 alignment heatmap |
| `GET` | `/health` | Server health check |

---

### Detect Hallucinations

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "source": "The Eiffel Tower is in Paris and was built between 1887 and 1889.",
    "response": "The Eiffel Tower is in Berlin and was built in 1920."
  }'
```

**Response:**

```json
{
  "verdict": "HALLUCINATED",
  "confidence": 0.78,
  "summary": "2 unsupported and 1 contradictory claims detected → response is hallucinated",
  "claims": [
    {
      "text": "The Eiffel Tower is located in Berlin.",
      "final_label": "CONTRADICTION",
      "similarity": 0.42
    },
    {
      "text": "It was built in 1920.",
      "final_label": "UNSUPPORTED",
      "similarity": 0.35
    }
  ]
}
```

---

### Get Verdict with Heatmap

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
  "verdict": "HALLUCINATED",
  "confidence": 0.95,
  "heatmap_png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

Decode the base64 string to get a PNG image of the semantic alignment matrix.

---

### Health Check

```bash
curl http://localhost:8000/health
# { "status": "ok" }
```

---

### Claim Label Reference

| Label | Meaning | Effect on Verdict |
|---|---|---|
| ✅ `ENTAILED` | Claim is directly supported by the source | Counts toward FAITHFUL |
| ❌ `CONTRADICTION` | Claim directly conflicts with the source | Triggers HALLUCINATED |
| 🟡 `UNSUPPORTED` | Claim is not grounded in source (neutral → unsupported) | Triggers HALLUCINATED |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 🔬 Pipeline — Deep Dive

```
┌──────────────────────────────────────────────────────────────┐
│                      AI Response Text                         │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────┐
              │      Claim Extraction       │
              │  spaCy sentence splitter    │
              │  + short-claim filter       │
              │  + deduplication            │
              └───────────────┬────────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
  ┌────────────────────────┐   ┌──────────────────────────┐
  │    NLI Classification   │   │   Semantic Similarity     │
  │  DeBERTa-v3 cross-enc  │   │   all-mpnet-base-v2       │
  │  → ENTAILED             │   │   cosine similarity       │
  │  → CONTRADICTION        │   │   claim × source matrix   │
  │  → NEUTRAL              │   │   → alignment heatmap     │
  └────────────┬───────────┘   └──────────────────────────┘
               │
               ▼
  ┌────────────────────────┐
  │    Aggregation Engine   │
  │  entailment fraction   │
  │  contradiction flag    │
  │  similarity baseline   │
  └────────────┬───────────┘
               │
               ▼
  ┌──────────────────────────────────┐
  │   FAITHFUL  ━━━━━━  HALLUCINATED │
  │   + confidence score             │
  │   + plain-English summary        │
  └──────────────────────────────────┘
```

### Aggregation Rules

| Condition | Verdict |
|---|---|
| ≥ 60% of claims are `ENTAILED` | ✅ `FAITHFUL` |
| Any claim is `CONTRADICTION` above score threshold | ❌ `HALLUCINATED` |
| Cosine similarity falls below 0.65 baseline | ❌ `HALLUCINATED` |
| Claim is `NEUTRAL` (unsupported) | Treated as ❌ `HALLUCINATED` |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

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

### Verify It's Working

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Python was created by Guido van Rossum.",
    "response": "Python was created by Linus Torvalds."
  }'
```

Expected output:

```json
{
  "verdict": "HALLUCINATED",
  "confidence": 0.96,
  "summary": "1 contradictory claim detected → response is hallucinated"
}
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## ⚙️ Configuration

All settings live in `configs/settings.py` — one file, no scattered environment variables.

```python
# configs/settings.py

similarity_threshold           = 0.65   # Minimum cosine similarity to pass baseline
entailment_fraction_threshold  = 0.60   # Min entailed fraction for FAITHFUL verdict
nli_batch_size                 = 32     # NLI model inference batch size
max_claims                     = 50     # Max claims extracted per response
nli_model_id                   = "cross-encoder/nli-deberta-v3-small"
similarity_model_id            = "sentence-transformers/all-mpnet-base-v2"
```

| Setting | Default | Effect |
|---|---|---|
| `similarity_threshold` | `0.65` | Lower = more permissive baseline check |
| `entailment_fraction_threshold` | `0.60` | Lower = easier to pass as FAITHFUL |
| `nli_batch_size` | `32` | Higher = faster inference, more VRAM |
| `max_claims` | `50` | Caps extraction on very long responses |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 📦 Scope & Limitations

This system verifies claims **only against the provided source passage**. It is a **grounding checker**, not a world-knowledge fact-checker.

**It does NOT:**
- Access external knowledge or the internet
- Validate facts beyond the given source document
- Replace domain-expert review for high-stakes decisions

**Performance depends on:**
- Quality and completeness of the source passage
- Clarity and atomicity of claims in the AI response
- NLI model confidence thresholds (all tunable in config)

> **Ideal use case:** RAG (Retrieval-Augmented Generation) pipelines where a retrieved source document is always available. Hallucination Hunter acts as the final verification gate before responses reach users.

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 📈 Future Roadmap

| Status | Feature |
|---|---|
| 🔜 | **Web UI** — Browser-based drag-and-drop interface for non-technical users |
| 🔜 | **Multilingual Support** — Hindi, French, German, Spanish |
| 🔜 | **PDF / Document Input** — Upload source as a PDF, not just raw text |
| 🔜 | **LangChain Integration** — Plug-in hallucination guard for LangChain pipelines |
| 🔜 | **Streaming Verdicts** — Stream per-claim results as they are computed |
| 🔜 | **Webhook Support** — Fire results to Slack, Discord, or any endpoint |
| 🔜 | **Batch API** — Verify hundreds of response-source pairs in one request |
| 🔜 | **PyPI Package** — `pip install hallucination-hunter` |
| 🔜 | **Confidence Calibration** — Platt scaling for better probability estimates |

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 🤝 Contributing

All contributions are welcome — from bug fixes to new features to documentation improvements.

### How to Contribute

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork locally
git clone https://github.com/YOUR-USERNAME/hallucination-hunter.git
cd hallucination-hunter

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes, then stage and commit
git add .
git commit -m "feat: clear description of what you changed"

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Open a Pull Request on GitHub 🎉
```

### Contribution Standards

- ✅ Keep code readable and well-commented
- ✅ Write tests for new logic where possible
- ✅ Update the README if your change affects usage or architecture
- ✅ Use descriptive commit messages (`feat:`, `fix:`, `docs:`, `refactor:`)
- ❤️ Be respectful — everyone is learning

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 📜 License

```
MIT License — Copyright (c) 2026 Harsh Kumar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software — subject to the condition that the above copyright
notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

<br/>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">

<br/>

## 🌟 Support the Project

If Hallucination Hunter helped you catch something an LLM got wrong, here's how to give back:

<div align="center">

| Action | Impact |
|:---:|:---:|
| ⭐ **Star the repo** | Helps others discover this tool |
| 🍴 **Fork it** | Build your own verification layer on top |
| 🐛 **Open an issue** | Help improve accuracy and reliability |
| 📢 **Share it** | Tweet it, write about it, tell your team |
| 💬 **Give feedback** | Even a comment goes a long way |

<br/>

[![Star this repo](https://img.shields.io/badge/⭐%20Star%20this%20repo-f59e0b?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hrshjha/hallucination-hunter)
[![Fork it](https://img.shields.io/badge/🍴%20Fork%20it-6366f1?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hrshjha/hallucination-hunter/fork)
[![Open an Issue](https://img.shields.io/badge/🐛%20Open%20an%20Issue-22c55e?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hrshjha/hallucination-hunter/issues)

</div>

<br/>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=130&section=footer&animation=fadeIn" width="100%"/>

**Made with ❤️ by [Harsh Kumar](https://github.com/hrshjha)**

*"The truth is out there — now you have a tool to find it."*

<br/>

[![GitHub followers](https://img.shields.io/github/followers/hrshjha?style=social)](https://github.com/hrshjha)
&nbsp;&nbsp;
[![Twitter Follow](https://img.shields.io/twitter/follow/m_eharsh?style=social)](https://twitter.com/m_eharsh)

</div>
