# ── Stage 1: Base image ───────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="Aman Chaudhary <amanchaudharyy01@gmail.com>"
LABEL description="SentimentIQ — Real-Time NLP Sentiment Analyser"
LABEL version="1.0"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python deps first (layer caching) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Train model at build time so container starts instantly ───────────────────
RUN python model/train.py

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 5000

# ── Environment ───────────────────────────────────────────────────────────────
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
