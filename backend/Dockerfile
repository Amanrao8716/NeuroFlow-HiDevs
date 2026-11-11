FROM python:3.11-slim AS deps

WORKDIR /build
COPY backend/requirements.txt ./requirements.txt

# Use CPU-only PyTorch; the GPU distribution adds hundreds of megabytes of
# CUDA libraries that are not used by this service.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r neuroflow \
    && useradd -r -g neuroflow -d /home/neuroflow -m neuroflow

WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY backend/ /app/
COPY pipelines/ /app/pipelines/
COPY evaluation/ /app/evaluation/

# Several modules support both `api.*` and `backend.*` imports.
RUN ln -s /app /app/backend \
    && chown -R neuroflow:neuroflow /app /home/neuroflow

USER neuroflow
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
