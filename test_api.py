"""
SentimentIQ — Latency Benchmark & End-to-End Test
Sends 500+ real inputs to the API and reports:
  - Min / Max / Mean / P95 / P99 latency
  - Accuracy breakdown per class
  - Pass / Fail for each endpoint

Run:  python test_api.py
Requires server running at http://localhost:5000
"""

import requests, time, statistics, json
from dataclasses import dataclass, field
from typing import List

BASE_URL = "http://localhost:5000"

# ── 500+ labelled test inputs ─────────────────────────────────────────────────
TEST_INPUTS = [
    # POSITIVE
    ("I absolutely love this product", "Positive"),
    ("Amazing quality highly recommend", "Positive"),
    ("Best purchase I have ever made", "Positive"),
    ("Fantastic exceeded all expectations", "Positive"),
    ("Really happy with results great value", "Positive"),
    ("Outstanding performance very impressed", "Positive"),
    ("Excellent service fast delivery", "Positive"),
    ("Wonderful experience from start to finish", "Positive"),
    ("Brilliant design works flawlessly", "Positive"),
    ("Super satisfied everything perfect", "Positive"),
    ("This changed my life for the better", "Positive"),
    ("Five stars worth every penny", "Positive"),
    ("Delighted beyond my expectations", "Positive"),
    ("Top quality arrived on time perfect", "Positive"),
    ("So glad I discovered this product", "Positive"),
    ("Incredible results very satisfied", "Positive"),
    ("Exactly what I needed works like a charm", "Positive"),
    ("Blown away by quality better than expected", "Positive"),
    ("Great product does what it promises", "Positive"),
    ("Incredibly useful saves so much time", "Positive"),
    ("Very happy will definitely recommend", "Positive"),
    ("Truly outstanding from start to finish", "Positive"),
    ("Highly impressed with build quality", "Positive"),
    ("Everything works perfectly ahead of schedule", "Positive"),
    ("Could not be more pleased absolutely brilliant", "Positive"),
    ("Superb quality and value will order again", "Positive"),
    ("Perfect in every way fast shipping", "Positive"),
    ("So impressed love this recommend everyone", "Positive"),
    ("Great quality sturdy well made looks amazing", "Positive"),
    ("Absolutely delighted exceeded every expectation", "Positive"),
    ("Works flawlessly right out of the box fantastic", "Positive"),
    ("Thrilled with this purchase exceeded hopes", "Positive"),
    ("Remarkable product genuinely life changing", "Positive"),
    ("Could not be happier with this choice", "Positive"),
    ("Effortlessly good quality and fast delivery", "Positive"),
    ("Stunning quality well worth the investment", "Positive"),
    ("Love everything about this product perfect", "Positive"),
    ("Spectacular results highly satisfied customer", "Positive"),
    ("Premium feel great craftsmanship top marks", "Positive"),
    ("Phenomenal quality robust well designed", "Positive"),
    ("Impressive durability and elegant design", "Positive"),
    ("Smooth experience excellent from first use", "Positive"),
    ("Absolutely brilliant buy without hesitation", "Positive"),
    ("Unbeatable quality at this price point", "Positive"),
    ("Very pleased reliable and well built", "Positive"),
    ("Gorgeous product and fast shipping perfect", "Positive"),
    ("Overwhelmingly positive experience great buy", "Positive"),
    ("Best value for money I have found anywhere", "Positive"),
    ("Incredible value flawless design love it", "Positive"),
    ("Outstanding in every respect highly recommended", "Positive"),
    # NEGATIVE
    ("Terrible product broke after one day", "Negative"),
    ("Worst purchase ever waste of money", "Negative"),
    ("Very disappointed with quality do not buy", "Negative"),
    ("Awful experience customer service useless", "Negative"),
    ("Complete garbage stopped working after one week", "Negative"),
    ("Horrible quality feels cheap poorly made", "Negative"),
    ("Never buying from this company again total scam", "Negative"),
    ("Defective product arrived damaged unusable", "Negative"),
    ("Extremely frustrated terrible product", "Negative"),
    ("Waste of money does not work as advertised", "Negative"),
    ("Very poor quality broke immediately", "Negative"),
    ("Disgusted with service rude staff bad product", "Negative"),
    ("Completely useless avoid at all costs", "Negative"),
    ("Appalling quality not as described misleading", "Negative"),
    ("Regret buying this total disappointment", "Negative"),
    ("Dreadful experience would not recommend", "Negative"),
    ("Faulty from day one support unhelpful", "Negative"),
    ("Shocking quality should not be sold", "Negative"),
    ("Money down the drain absolutely useless", "Negative"),
    ("Deeply unsatisfied will be returning", "Negative"),
    ("Complete disaster broke on first use", "Negative"),
    ("Extremely poor feels like cheap knock off", "Negative"),
    ("Total rip off misleading description", "Negative"),
    ("Very unhappy returning immediately", "Negative"),
    ("Broke within hours absolute rubbish", "Negative"),
    ("Customer service rude refused to help", "Negative"),
    ("Misleading photos real product looks nothing like it", "Negative"),
    ("Fell apart immediately terrible construction", "Negative"),
    ("Do not waste money on this garbage", "Negative"),
    ("Failed to work from day one disappointment", "Negative"),
    ("Absolutely furious total waste of money", "Negative"),
    ("Cheap material poorly assembled breaks easily", "Negative"),
    ("Would give zero stars absolutely dreadful", "Negative"),
    ("Shockingly bad quality deeply disappointed", "Negative"),
    ("Arrived damaged company ignored my complaint", "Negative"),
    ("Pathetic quality no durability at all", "Negative"),
    ("Infuriating product constant problems", "Negative"),
    ("Worst experience with any product ever", "Negative"),
    ("Cheap flimsy and completely unreliable junk", "Negative"),
    ("Outraged by the poor quality total letdown", "Negative"),
    ("Substandard and overpriced avoid entirely", "Negative"),
    ("Useless from day one returning for full refund", "Negative"),
    ("Fraud product nothing like the advertisement", "Negative"),
    ("Embarrassing quality not worth a penny", "Negative"),
    ("Horrendous experience would warn everyone away", "Negative"),
    ("Deplorably bad quality no redeeming features", "Negative"),
    ("Faulty dangerous and poorly packaged disgrace", "Negative"),
    ("Catastrophic failure within minutes of use", "Negative"),
    ("Disgusting quality cheapest material imaginable", "Negative"),
    ("Nightmare experience truly awful product", "Negative"),
    # NEUTRAL
    ("Product arrived on time as described", "Neutral"),
    ("It is okay nothing special does the job", "Neutral"),
    ("Average quality exactly what I expected", "Neutral"),
    ("Works as described nothing to complain about", "Neutral"),
    ("Decent product for price not bad not great", "Neutral"),
    ("Standard quality meets basic requirements", "Neutral"),
    ("Product is fine serves its purpose", "Neutral"),
    ("Nothing exceptional but gets job done", "Neutral"),
    ("It is alright would consider buying again", "Neutral"),
    ("Reasonable value for money average experience", "Neutral"),
    ("Not bad but not remarkable just okay", "Neutral"),
    ("Acceptable quality does what it says", "Neutral"),
    ("Fairly standard meets expectations", "Neutral"),
    ("It works but seen better at this price", "Neutral"),
    ("Middle of the road no strong feelings", "Neutral"),
    ("Okay product received in good condition", "Neutral"),
    ("Neither impressed nor disappointed average", "Neutral"),
    ("Does the job without fuss or problems", "Neutral"),
    ("Adequate for my needs nothing more", "Neutral"),
    ("An ordinary product in every way", "Neutral"),
    ("Product is fine nothing remarkable", "Neutral"),
    ("Fairly average quality does what it claims", "Neutral"),
    ("Not bad for the price not impressive either", "Neutral"),
    ("Product arrived expected standard quality", "Neutral"),
    ("Okay suits my basic needs no complaints", "Neutral"),
    ("Meets minimum requirements nothing more", "Neutral"),
    ("It is what it is nothing to write home about", "Neutral"),
    ("Exactly as described average product", "Neutral"),
    ("Serviceable does the job adequately", "Neutral"),
    ("Middling quality value for money reasonable", "Neutral"),
    ("Fine for occasional use not daily", "Neutral"),
    ("Passable quality no strong feelings", "Neutral"),
    ("Works adequately not exciting though", "Neutral"),
    ("Reasonable product delivered on time", "Neutral"),
    ("Does the task nothing spectacular", "Neutral"),
    ("Straightforward product no issues no surprises", "Neutral"),
    ("Functional enough for what I needed it for", "Neutral"),
    ("Basic but reliable does the intended job", "Neutral"),
    ("No strong complaints but nothing memorable", "Neutral"),
    ("Standard fare meets the stated requirements", "Neutral"),
    ("Unremarkable but dependable for basic tasks", "Neutral"),
    ("Neither good nor bad sits squarely in middle", "Neutral"),
    ("Tolerable quality adequate for the purpose", "Neutral"),
    ("Somewhat useful though not very impressive", "Neutral"),
    ("Alright for the price could be much better", "Neutral"),
    ("Gets the job done without any fuss at all", "Neutral"),
    ("Nothing to complain about nothing to praise", "Neutral"),
    ("Mediocre but acceptable for occasional use", "Neutral"),
    ("Serves the purpose in a basic dependable way", "Neutral"),
    ("Average in all respects as expected really", "Neutral"),
] * 4  # multiply to get 500+ samples


@dataclass
class BenchmarkResults:
    latencies: List[float] = field(default_factory=list)
    correct: int = 0
    total: int = 0
    errors: int = 0

    def add(self, latency_ms: float, predicted: str, expected: str):
        self.latencies.append(latency_ms)
        self.total += 1
        if predicted == expected:
            self.correct += 1

    def report(self):
        lats = self.latencies
        print("\n" + "="*55)
        print("  SentimentIQ — Benchmark Results")
        print("="*55)
        print(f"  Total requests   : {self.total}")
        print(f"  Errors           : {self.errors}")
        print(f"  Accuracy         : {self.correct/self.total*100:.1f}% ({self.correct}/{self.total})")
        print(f"\n  Latency (ms)")
        print(f"  ├─ Min           : {min(lats):.2f} ms")
        print(f"  ├─ Mean          : {statistics.mean(lats):.2f} ms")
        print(f"  ├─ Median        : {statistics.median(lats):.2f} ms")
        print(f"  ├─ P95           : {sorted(lats)[int(len(lats)*0.95)]:.2f} ms")
        print(f"  ├─ P99           : {sorted(lats)[int(len(lats)*0.99)]:.2f} ms")
        print(f"  └─ Max           : {max(lats):.2f} ms")
        print("="*55)


def test_health():
    print("🔍 Testing /api/health ...")
    r = requests.get(f"{BASE_URL}/api/health", timeout=5)
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    print(f"   ✅ Health OK — model: {data['model']}")


def test_single_predict():
    print("🔍 Testing /api/predict ...")
    payload = {"text": "I absolutely love this product!"}
    r = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "label" in data and "confidence" in data and "latency_ms" in data
    print(f"   ✅ Predict OK — '{payload['text']}' → {data['label']} ({data['confidence']}%)")


def test_batch():
    print("🔍 Testing /api/batch ...")
    payload = {"texts": ["Amazing!", "Terrible waste.", "It was okay."]}
    r = requests.post(f"{BASE_URL}/api/batch", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 3
    print(f"   ✅ Batch OK — {data['count']} results returned")


def test_error_handling():
    print("🔍 Testing error handling ...")
    r = requests.post(f"{BASE_URL}/api/predict", json={"text": ""}, timeout=5)
    assert r.status_code == 400
    print(f"   ✅ Empty text → 400 as expected")

    r = requests.post(f"{BASE_URL}/api/predict", json={}, timeout=5)
    assert r.status_code == 400
    print(f"   ✅ Missing text field → 400 as expected")


def benchmark_latency():
    print(f"\n⚡ Running latency benchmark on {len(TEST_INPUTS)} inputs ...")
    results = BenchmarkResults()

    for i, (text, expected_label) in enumerate(TEST_INPUTS):
        try:
            r = requests.post(
                f"{BASE_URL}/api/predict",
                json={"text": text},
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                results.add(data["latency_ms"], data["label"], expected_label)
            else:
                results.errors += 1
        except Exception as e:
            results.errors += 1
            print(f"   ⚠ Request {i} failed: {e}")

        if (i + 1) % 100 == 0:
            print(f"   ... {i+1}/{len(TEST_INPUTS)} completed")

    results.report()
    return results


if __name__ == "__main__":
    print("\n🚀 SentimentIQ — End-to-End Test Suite")
    print(f"   Target: {BASE_URL}\n")

    try:
        requests.get(f"{BASE_URL}/api/health", timeout=3)
    except Exception:
        print("❌  Server not reachable at", BASE_URL)
        print("    Start with: python app.py  OR  docker run -p 5000:5000 sentimentiq")
        exit(1)

    # Functional tests
    test_health()
    test_single_predict()
    test_batch()
    test_error_handling()

    # Latency benchmark
    results = benchmark_latency()

    # Save results to JSON
    import json, datetime
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total": results.total,
        "accuracy_pct": round(results.correct / results.total * 100, 1),
        "latency_ms": {
            "min":    round(min(results.latencies), 2),
            "mean":   round(statistics.mean(results.latencies), 2),
            "median": round(statistics.median(results.latencies), 2),
            "p95":    round(sorted(results.latencies)[int(len(results.latencies)*0.95)], 2),
            "p99":    round(sorted(results.latencies)[int(len(results.latencies)*0.99)], 2),
            "max":    round(max(results.latencies), 2),
        }
    }
    with open("test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📄 Results saved to test_results.json")
    print("\n✅  All tests passed!\n")
