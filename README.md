# SentimentIQ — Real-Time NLP Sentiment Analyser

Aman Chaudhary 

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange)

**🔗 [Live Demo](https://sentiment-analysis-nlp-rrme.onrender.com/)**
---

## What It Does
A production-ready web application that classifies any text as **Positive**, **Neutral**, or **Negative** with confidence scores — served via a Flask REST API, containerised with Docker, and documented with an OpenAPI spec.

## Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | Logistic Regression (Scikit-Learn) |
| Features | TF-IDF with bigrams, stop-word removal |
| Backend API | Flask REST API |
| Containerisation | Docker |
| API Spec | OpenAPI 3.0 (Postman ready) |
| Testing | Custom benchmark — 500+ inputs, latency profiling |

## Key Results
- **Sub-2ms** inference latency (P95) across 500+ tested inputs
- **78% accuracy** on 3-class classification
- Full end-to-end pipeline: text → TF-IDF → model → JSON response

---

## Run with Docker (Recommended)

```bash
# 1. Build image
docker build -t sentimentiq .

# 2. Run container
docker run -p 5000:5000 sentimentiq

# 3. Open in browser
http://localhost:5000
```

## Run Locally

```bash
pip install -r requirements.txt
python app.py
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

## Import into Postman
1. Open Postman → **Import**
2. Select `openapi.yaml` from this repo
3. All 3 endpoints load automatically with example payloads

## Run Benchmark (500+ inputs)
```bash
# With server running:
python test_api.py
```

---

## Resume Bullet Points
```
• Built 3-class NLP Sentiment Analyser using TF-IDF + Logistic Regression;
  benchmarked against baseline achieving 78% accuracy across 500+ test inputs

• Served via Flask REST API (/predict, /batch); serialised Scikit-Learn pipeline
  at sub-2ms P95 inference latency across 500+ tested inputs

• Containerised full stack with Docker; documented OpenAPI 3.0 spec;
  tested end-to-end with Postman collection
```
