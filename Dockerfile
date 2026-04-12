FROM python:3.11-slim

WORKDIR /hallucination_hunter

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps — install CPU-only torch first, then everything else
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Set PYTHONPATH so bare imports (from configs.X, from core.X, etc.) resolve
ENV PYTHONPATH="/hallucination_hunter:/hallucination_hunter/app"

# Expose port (Render overrides this via PORT env var)
EXPOSE 10000

# Use shell form so $PORT is expanded at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
