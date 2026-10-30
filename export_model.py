"""Copies the trained Logistic Regression + Naive Bayes into website/model.js
so the plain HTML website can use your REAL trained model (no Flask needed).

Run AFTER train.py:   python export_model.py
"""
import json
import os
import sys

import joblib

if not os.path.exists("model/jobguard_model.joblib"):
    print("ERROR: train the model first:  python train.py data/fake_job_postings.csv")
    sys.exit(1)

bundle = joblib.load("model/jobguard_model.joblib")
vec = bundle["vectorizer"]
lr = bundle["models"]["Logistic Regression"]
nb = bundle["models"]["Naive Bayes"]
metrics = json.load(open("model/metrics.json"))

n = len(vec.vocabulary_)
terms = [None] * n
for term, i in vec.vocabulary_.items():
    terms[i] = term

r = lambda arr: [round(float(x), 6) for x in arr]
coef = lr.coef_[0]
diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]

model = {
    "stop": sorted(vec.get_stop_words()),
    "terms": terms,
    "idf": r(vec.idf_),
    "lr": {"coef": r(coef[:n]), "meta": r(coef[n:]), "intercept": round(float(lr.intercept_[0]), 6)},
    "nb": {"diff": r(diff), "prior": round(float(nb.class_log_prior_[1] - nb.class_log_prior_[0]), 6)},
    "metrics": {k: v for k, v in metrics["results"].items()},
    "best_model": metrics["best_model"],
    "train_rows": metrics["train_rows"],
    "test_rows": metrics["test_rows"],
}
with open("website/model.js", "w", encoding="utf-8") as f:
    f.write("window.JOBGUARD_MODEL = " + json.dumps(model, ensure_ascii=False) + ";\n")
print("Saved website/model.js  (terms:", n, ")")
