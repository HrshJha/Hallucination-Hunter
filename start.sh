#!/bin/bash
# Move to the script's directory (hallucination_hunter)
cd "$(dirname "$0")"

echo "=========================================="
echo "🚀 Starting Hallucination Hunter API server"
echo "=========================================="

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚙️ Virtual environment not found. Building it now..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
else
    echo "✅ Found virtual environment (.venv)."
    source .venv/bin/activate
fi

# We use port 8001 since 8000 might be in use
echo "🌐 Starting FastAPI on http://localhost:8001"
# Use exec to replace the shell so signals pass through properly
exec uvicorn app.main:app --host 0.0.0.0 --port 8001
