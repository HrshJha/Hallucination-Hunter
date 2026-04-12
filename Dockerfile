FROM python:3.11-slim

WORKDIR /opt/render/project/src

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps — install CPU-only torch first, then everything else
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy source
COPY . .

# Set PYTHONPATH so app-prefixed imports resolve from project root
ENV PYTHONPATH="/opt/render/project/src"

# Expose port (Render overrides this via PORT env var)
EXPOSE 10000

# Use shell form so $PORT is expanded at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
