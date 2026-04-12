#!/bin/bash
# Move to the project root (hallucination_hunter/)
cd "$(dirname "$0")/.."

echo "=========================================="
echo "🚀 Starting Hallucination Hunter API server"
echo "=========================================="

# Set PYTHONPATH so bare imports resolve
export PYTHONPATH="$(pwd):$(pwd)/app"

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚙️ Virtual environment not found. Building it now..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    echo "✅ Found virtual environment (.venv)."
    source .venv/bin/activate
fi

# Use PORT env var with fallback to 8001
PORT="${PORT:-8001}"
echo "🌐 Starting FastAPI on http://localhost:${PORT}"
# Use exec to replace the shell so signals pass through properly
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
