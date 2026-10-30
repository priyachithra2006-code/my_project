"""Train 3 models on the Kaggle 'Real or Fake Job Posting' dataset.

Usage:   python train.py data/fake_job_postings.csv
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from features import build_text, meta_features

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/fake_job_postings.csv"
if not os.path.exists(csv_path):
    print("\nERROR: dataset file not found:", csv_path)
    print("Download 'fake_job_postings.csv' from Kaggle (Real or Fake Job Posting Prediction)")
    print("and put it inside the 'data' folder. Then run this again.")
    sys.exit(1)

print("Reading", csv_path)
df = pd.read_csv(csv_path).fillna("")
rows = df.to_dict("records")
texts = [build_text(r) for r in rows]
meta = np.array([meta_features(r) for r in rows])
y = df["fraudulent"].astype(int).values
print(f"Rows: {len(df)} | fake: {y.sum()} | real: {len(y) - y.sum()}")

idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(stop_words="english", max_features=10000,
                             ngram_range=(1, 2), min_df=2)
X_train = hstack([vectorizer.fit_transform([texts[i] for i in idx_train]),
                  csr_matrix(meta[idx_train])]).tocsr()
X_test = hstack([vectorizer.transform([texts[i] for i in idx_test]),
                 csr_matrix(meta[idx_test])]).tocsr()
y_train, y_test = y[idx_train], y[idx_test]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Naive Bayes": MultinomialNB(),
    # 100 trees + min_samples_leaf keeps the saved file small enough to upload to GitHub
    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                            class_weight="balanced", n_jobs=-1, random_state=42),
}

results = {}
for name, model in models.items():
    print("Training", name, "...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "precision": round(precision_score(y_test, pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, pred, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print("  ", {k: v for k, v in results[name].items() if k != "confusion_matrix"})

best = max(results, key=lambda n: results[n]["f1"])
print("Best model (by F1 on fake class):", best)

os.makedirs("model", exist_ok=True)
joblib.dump({"vectorizer": vectorizer, "models": models}, "model/jobguard_model.joblib", compress=3)
with open("model/metrics.json", "w") as f:
    json.dump({"best_model": best, "train_rows": int(len(idx_train)),
               "test_rows": int(len(idx_test)), "results": results}, f, indent=2)

size_mb = os.path.getsize("model/jobguard_model.joblib") / 1e6
print(f"Saved model/jobguard_model.joblib ({size_mb:.1f} MB) and model/metrics.json")
if size_mb > 90:
    print("WARNING: file is over 90 MB. GitHub blocks files over 100 MB. Lower n_estimators in train.py.")
print(f"\nBefore deploying, make sure requirements.txt has the line:  scikit-learn=={sklearn.__version__}")
