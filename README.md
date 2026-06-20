# SentimentIQ — Real-Time NLP Sentiment Analyser

**Aman Chaudhary**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange) ![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)

**🔗 [Live Demo](https://sentiment-analysis-nlp-rrme.onrender.com/)**

---

## What It Does
A production-ready web application that classifies any text as **Positive**, **Neutral**, or **Negative** with confidence scores — served via a Flask REST API, containerised with Docker, and documented with an OpenAPI spec. Includes a transformer benchmark comparing TF-IDF against DistilBERT.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Logistic Regression (Scikit-Learn) |
| Features | TF-IDF with bigrams, stop-word removal |
| Transformer Benchmark | DistilBERT (Hugging Face Transformers) |
| Backend API | Flask REST API |
| Containerisation | Docker |
| API Spec | OpenAPI 3.0 (Postman ready) |
| Testing | 600-input benchmark with latency profiling |

## Key Results

- **93.3% accuracy** on 3-class classification across 600 tested inputs
- **0.03ms per input** inference latency (sub-2ms P95)
- **259x faster** than DistilBERT transformer at inference time
- DistilBERT achieves 97.5% on binary (Positive/Negative) — identified as accuracy upgrade path

## Model Benchmark: TF-IDF vs DistilBERT

| Metric | TF-IDF + LR | DistilBERT |
|--------|-------------|------------|
| Accuracy | 93.3% | 97.5% (binary) |
| Latency per input | 0.03ms | 8.81ms |
| Model size | ~10KB | ~260MB |
| Requires GPU | No | Recommended |
| Production ready | ✅ | ⚠️ |

**Conclusion:** TF-IDF + LR is optimal for sub-2ms production serving. DistilBERT is the recommended upgrade path when accuracy > latency and GPU is available.

---

## Run with Docker (Recommended)

```bash
docker build -t sentimentiq .
docker run -p 5000:5000 sentimentiq
# Open http://localhost:5000
```

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Run Transformer Benchmark

```bash
pip install transformers torch
python benchmark_transformer.py
# Compares TF-IDF vs DistilBERT across 600 inputs
```

---

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'

# Batch prediction
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Amazing!", "Terrible.", "It was okay."]}'

# Health check
curl http://localhost:5000/api/health
```

---

## Resume Bullet Points

```
- Built 3-class NLP Sentiment Analyser (TF-IDF + Logistic Regression);
  93.3% accuracy across 600 tested inputs; served via Flask REST API
  at sub-2ms P95 latency, containerised with Docker

- Benchmarked TF-IDF against DistilBERT transformer (Hugging Face);
  TF-IDF proved 259x faster (0.03ms vs 8.81ms/input) — optimal for
  production serving; DistilBERT identified as accuracy upgrade path

- Containerised with Docker; documented OpenAPI 3.0 spec;
  tested end-to-end with Postman
```
