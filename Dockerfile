# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt \
    --break-system-packages 2>/dev/null || \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

ENV PYTHONPATH=/app/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
