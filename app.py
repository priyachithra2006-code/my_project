"""JobGuard – Flask server: serves the website and the prediction API."""
import json
import os
import sqlite3
from datetime import datetime

import joblib
from flask import Flask, jsonify, request, send_from_directory

import analysis

BASE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(BASE, "website")
MODEL_PATH = os.path.join(BASE, "model", "jobguard_model.joblib")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")
DB_PATH = os.environ.get("JOBGUARD_DB", os.path.join(BASE, "jobguard.db"))

app = Flask(__name__, static_folder=SITE, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 1_000_000

# ---------- load trained models (server still works in rules-only mode without them) ----------
bundle, metrics, load_error = None, None, None
if os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except Exception as e:                      # e.g. scikit-learn version mismatch
        bundle, metrics, load_error = None, None, str(e)
        print("Could not load model:", e)
KEY_OF = {v: k for k, v in analysis.MODEL_NAMES.items()}
BEST_KEY = KEY_OF[metrics["best_model"]] if metrics else "lr"


# ---------- database ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


with db() as _c:
    _c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, title TEXT,
        mode TEXT, score REAL, label TEXT, category TEXT, flags TEXT)""")


# ---------- pages ----------
@app.route("/")
def home():
    return send_from_directory(SITE, "index.html")


# ---------- API ----------
@app.route("/api/health")
def health():
    return jsonify(
        ok=True,
        model_ready=bundle is not None,
        models=[{"key": KEY_OF[n], "name": n} for n in bundle["models"]] if bundle else [],
        best=BEST_KEY,
        metrics=metrics["results"] if metrics else None,
        best_model=metrics["best_model"] if metrics else None,
        test_rows=metrics["test_rows"] if metrics else None,
        ml_weight=analysis.ML_WEIGHT,
        error=load_error,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    if not (str(data.get("title", "")).strip() or str(data.get("description", "")).strip()):
        return jsonify(error="Please enter at least a job title or a description."), 400

    key = data.get("model") or BEST_KEY
    if bundle is not None and key not in analysis.MODEL_NAMES:
        return jsonify(error="Unknown model."), 400

    out = analysis.analyze(data, bundle, key)
    with db() as conn:
        conn.execute(
            "INSERT INTO predictions (created_at,title,mode,score,label,category,flags) VALUES (?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), str(data.get("title", ""))[:80] or "(no title)",
             out["mode"], out["score"], out["label"], out["category"], json.dumps(out["flags"])))
    return jsonify(out)


@app.route("/api/history", methods=["GET", "DELETE"])
def history():
    with db() as conn:
        if request.method == "DELETE":
            conn.execute("DELETE FROM predictions")
            return jsonify(ok=True)
        rows = [{"time": r["created_at"], "title": r["title"], "mode": r["mode"], "score": r["score"],
                 "label": r["label"], "category": r["category"]}
                for r in conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 200")]
    return jsonify(rows=rows)


if __name__ == "__main__":
    app.run(debug=True)
