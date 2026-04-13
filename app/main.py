"""
Hallucination Hunter – FastAPI Application
"""

from __future__ import annotations


import os
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.routes import router

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

# ── Model warmup state ─────────────────────────────────────────────
_warmup_state = {"ready": False, "error": None, "started_at": None}


def _warmup_models():
    """Pre-load ML models in a background thread so first /detect isn't slow."""
    global _warmup_state
    _warmup_state["started_at"] = time.time()
    try:
        print("[Warmup] Loading NLI model...")
        from app.core.nli import _load as load_nli
        load_nli()
        print("[Warmup] NLI model loaded.")

        print("[Warmup] Loading similarity model...")
        from app.core.similarity import _get_model as load_sim
        load_sim()
        print("[Warmup] Similarity model loaded.")

        print("[Warmup] Loading spaCy model...")
        from app.claims.extractor import _get_nlp
        _get_nlp()
        print("[Warmup] spaCy model loaded.")

        _warmup_state["ready"] = True
        elapsed = time.time() - _warmup_state["started_at"]
        print(f"[Warmup] ✅ All models ready in {elapsed:.1f}s")
    except Exception as e:
        _warmup_state["error"] = str(e)
        print(f"[Warmup] ❌ Failed: {e}")


@app.get("/warmup-status", tags=["System"])
async def warmup_status():
    """Check if ML models are loaded and ready."""
    elapsed = None
    if _warmup_state["started_at"]:
        elapsed = round(time.time() - _warmup_state["started_at"], 1)
    return JSONResponse({
        "ready": _warmup_state["ready"],
        "error": _warmup_state["error"],
        "elapsed_seconds": elapsed,
    })


# Serve the Web UI at root
INDEX_HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "index.html")


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the Hallucination Hunter web interface."""
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    return {"message": "Hallucination Hunter API is running. Visit /docs for API documentation."}


@app.on_event("startup")
async def startup_event():
    """Start background model warmup so /detect doesn't timeout on first call."""
    port = os.environ.get("PORT", "10000")
    print("[Hallucination Hunter] ────────────────────────────────────────────")
    print(f"[Hallucination Hunter] 🌐 Web UI  → http://0.0.0.0:{port}")
    print(f"[Hallucination Hunter] 📡 API     → http://0.0.0.0:{port}/detect")
    print(f"[Hallucination Hunter] 📚 Docs    → http://0.0.0.0:{port}/docs")
    print("[Hallucination Hunter] ────────────────────────────────────────────")
    print("[Hallucination Hunter] Starting background model warmup...")
    thread = threading.Thread(target=_warmup_models, daemon=True)
    thread.start()


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
