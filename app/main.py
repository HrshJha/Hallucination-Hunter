"""
Hallucination Hunter – FastAPI Application
"""

from __future__ import annotations

import sys
import os

# ── Path Setup ──────────────────────────────────────────────────────
# PROJECT_ROOT = parent of `app/` = the repo root (hallucination_hunter/)
# We add it to sys.path so bare imports like `from configs.settings ...`,
# `from core.pipeline ...`, `from models.schemas ...` resolve correctly.
# This is also reinforced via PYTHONPATH in the start command / Dockerfile.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also add app/ itself so sibling packages resolve from within app/
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.routes import router

app = FastAPI(
    title="Hallucination Hunter",
    description=(
        "Factual consistency & hallucination detection for AI-generated responses. "
        "Submit a source passage and an AI response to get a FAITHFUL/HALLUCINATED verdict "
        "with per-claim NLI breakdown and alignment heatmap."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS – allow everything for development / Colab
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the Web UI at root
INDEX_HTML = os.path.join(PROJECT_ROOT, "ui", "index.html")


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the Hallucination Hunter web interface."""
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    return {"message": "Hallucination Hunter API is running. Visit /docs for API documentation."}


@app.on_event("startup")
async def startup_event():
    """Warm up models on startup for faster first request."""
    port = os.environ.get("PORT", "8001")
    print("[Hallucination Hunter] Models will be lazy-loaded on first request.")
    print("[Hallucination Hunter] ────────────────────────────────────────────")
    print(f"[Hallucination Hunter] 🌐 Web UI  → http://localhost:{port}")
    print(f"[Hallucination Hunter] 📡 API     → http://localhost:{port}/detect")
    print(f"[Hallucination Hunter] 📚 Docs    → http://localhost:{port}/docs")
    print("[Hallucination Hunter] ────────────────────────────────────────────")



if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
    )
