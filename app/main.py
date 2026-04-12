"""
Hallucination Hunter – FastAPI Application
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
