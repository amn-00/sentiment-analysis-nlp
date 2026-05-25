"""
Sentiment Analysis API — AI/ML Engineer Portfolio Project
Author: Aman Chaudhary
Stack: Flask · Scikit-Learn · NLTK · TF-IDF · Logistic Regression
"""

from flask import Flask, request, jsonify, render_template

import pickle, os, re, time
from model.train import train_and_save, MODEL_PATH, VECTORIZER_PATH

app = Flask(__name__)


# ── Load or train model on startup ──────────────────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("⚙  Model not found — training now...")
    train_and_save()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("✅  Model loaded successfully")


# ── Helper ───────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJI_MAP = {0: "😞", 1: "😐", 2: "😊"}
COLOR_MAP = {0: "#e24b4a", 1: "#ba7517", 2: "#1d9e75"}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 chars)"}), 400

    t0 = time.perf_counter()
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    pred = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0].tolist()
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return jsonify({
        "label":      LABEL_MAP[pred],
        "emoji":      EMOJI_MAP[pred],
        "color":      COLOR_MAP[pred],
        "confidence": round(max(proba) * 100, 1),
        "scores": {
            "Negative": round(proba[0] * 100, 1),
            "Neutral":  round(proba[1] * 100, 1),
            "Positive": round(proba[2] * 100, 1),
        },
        "latency_ms": latency_ms,
        "word_count": len(text.split()),
    })


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    """Analyse multiple texts at once — useful for CSV uploads."""
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    if not texts or len(texts) > 50:
        return jsonify({"error": "Provide 1–50 texts"}), 400

    results = []
    for t in texts:
        clean = preprocess(t)
        vec = vectorizer.transform([clean])
        pred = int(model.predict(vec)[0])
        proba = model.predict_proba(vec)[0].tolist()
        results.append({
            "text":       t[:80] + "…" if len(t) > 80 else t,
            "label":      LABEL_MAP[pred],
            "confidence": round(max(proba) * 100, 1),
        })
    return jsonify({"results": results, "count": len(results)})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "TF-IDF + Logistic Regression"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
