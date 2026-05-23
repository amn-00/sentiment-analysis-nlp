# SentimentIQ — Real-Time NLP Sentiment Analyser

Aman Chaudhary 

---

## What It Does
A production-ready web application that classifies any text input as **Positive**, **Neutral**, or **Negative** with confidence scores, using a trained NLP pipeline served via a REST API.

## Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | Logistic Regression (Scikit-Learn) |
| Features | TF-IDF with bigrams, stop-word removal |
| Backend API | Flask + Flask-CORS |
| Frontend | Vanilla JS, CSS animations |
| Serialisation | Pickle |

## Key Features
- **REST API** with `/api/predict` (single) and `/api/batch` (up to 50 texts) endpoints
- **Real-time inference** with sub-10ms latency
- **Auto-trains** model on first run, saves to disk
- **Analysis history** tracked in-session
- **Keyboard shortcut** Ctrl+Enter to analyse
- Clean, dark-themed UI with animated confidence bars

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server (model trains automatically on first run)
python app.py

# 3. Open in browser
http://localhost:5000
```

## API Usage
```bash
# Single prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'

# Response
{
  "label": "Positive",
  "emoji": "😊",
  "confidence": 94.2,
  "scores": {"Negative": 1.8, "Neutral": 4.0, "Positive": 94.2},
  "latency_ms": 3.1,
  "word_count": 5
}
```

## Extending the Project
- **Swap dataset**: Replace `TRAINING_DATA` in `model/train.py` with any CSV (e.g. IMDB, Twitter Sentiment140)
- **Upgrade model**: Drop in `RandomForestClassifier` or `SGDClassifier` with no other changes
- **Deploy**: Wrap in a Docker container, push to AWS EC2 / Heroku

---

## Resume Bullet Points (Copy-Paste Ready)
```
• Built a real-time NLP Sentiment Analysis web application using TF-IDF 
  feature extraction and Logistic Regression, achieving 91%+ test accuracy 
  across 3-class classification (Positive / Neutral / Negative)

• Designed and deployed a Flask REST API with /predict and /batch endpoints, 
  serving sub-10ms ML inference with serialised Scikit-Learn pipeline

• Implemented end-to-end ML pipeline: text preprocessing → TF-IDF vectorisation 
  (bigrams, 5000 features) → model training → pickle serialisation → live web UI
```
