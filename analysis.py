"""JobGuard detection logic: rule checks + ML probability -> risk score.
This is the Python twin of website/engine.js (same rules, same weights)."""
import math
import re

from scipy.sparse import csr_matrix, hstack

from features import build_text, meta_features, _s, _flag

MODEL_NAMES = {"lr": "Logistic Regression", "nb": "Naive Bayes", "rf": "Random Forest"}

SCAM_PHRASES = [
    "wire transfer", "registration fee", "security deposit", "processing fee", "training fee",
    "no interview", "without interview", "no experience needed", "no experience required",
    "earn money fast", "easy money", "work from home and earn", "limited seats", "pay to",
    "send money", "bank account details", "western union", "urgent hiring", "immediate joining",
    "100% placement", "guaranteed job", "whatsapp", "telegram", "click the link", "refundable",
]
PAYMENT_PHRASES = [
    "registration fee", "security deposit", "processing fee", "training fee", "pay to",
    "send money", "wire transfer", "bank account details", "western union",
]
FREE_DOMAINS = {"gmail.com", "yahoo.com", "yahoo.in", "outlook.com", "hotmail.com",
                "rediffmail.com", "live.com", "aol.com", "protonmail.com", "mail.com"}
WEIGHTS = {"keywords": 0.35, "salary": 0.20, "email": 0.25, "profile": 0.20}
ML_WEIGHT = 0.6


def _round1(x):
    return math.floor(x * 10 + 0.5) / 10


def _inr(n):
    """150000 -> '1,50,000' (Indian digit grouping)."""
    s = str(int(math.floor(n + 0.5)))
    if len(s) <= 3:
        return s
    head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def _clamp(n, lo, hi):
    return max(lo, min(hi, n))


# ---------------- rule checks (each 0-100) ----------------
def keyword_score(text):
    t = text.lower()
    found = [p for p in SCAM_PHRASES if p in t]
    pay = [p for p in PAYMENT_PHRASES if p in t]
    return {"score": _clamp(len(found) * 25, 0, 100), "found": found, "pay": pay}


def salary_score(d):
    raw = _s(d.get("salary_range")).strip()
    if not raw:
        return {"score": 0, "note": "No salary given – nothing to check"}
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", raw.replace(",", ""))]
    if not nums:
        return {"score": 0, "note": "Salary has no numbers – nothing to check"}
    low, high = nums[0], (nums[1] if len(nums) > 1 else nums[0])
    if re.search(r"lakh|lpa", raw, re.I):
        low, high = low * 100000, high * 100000
    if high == 0:
        return {"score": 0, "note": "Salary is 0 – nothing to check"}
    if low > high:
        return {"score": 60, "note": "Minimum salary is higher than maximum"}
    yearly = d.get("salary_period") == "yearly" or re.search(r"lpa|per annum|per year|yearly|annual", raw, re.I)
    monthly = high / 12 if yearly else high
    entry = d.get("experience") == "entry"
    if monthly >= 100000 and entry:
        return {"score": 90, "note": f"Very high salary (about ₹{_inr(monthly)}/month) for an entry-level role"}
    if monthly >= 100000:
        return {"score": 40, "note": f"High salary (about ₹{_inr(monthly)}/month) – verify it"}
    if low > 0 and high / low >= 5:
        return {"score": 70, "note": "Salary range is unusually wide"}
    return {"score": 0, "note": "Salary looks normal"}


def email_score(d):
    email = _s(d.get("contact_email")).strip().lower()
    if not email:
        return {"score": 0, "note": "No recruiter email given"}
    m = re.match(r"^[^@\s]+@([^@\s]+\.[a-z]{2,})$", email)
    if not m:
        return {"score": 60, "note": "Email address looks invalid"}
    domain = m.group(1)
    if domain in FREE_DOMAINS:
        return {"score": 80, "note": f"Free email domain (@{domain}) instead of a company email"}
    skip = {"pvt", "ltd", "inc", "llc", "the", "private", "limited", "company", "technologies"}
    company = [w for w in re.sub(r"[^a-z0-9 ]", " ", _s(d.get("company_name")).lower()).split()
               if len(w) >= 3 and w not in skip]
    if company and not any(w in domain for w in company):
        return {"score": 40, "note": f"Email domain ({domain}) does not match the company name"}
    return {"score": 0, "note": "Email domain looks fine"}


def completeness_score(d):
    s, notes = 0, []
    if not _s(d.get("company_profile")).strip():
        s += 40; notes.append("Company profile is missing")
    if not _flag(d.get("has_company_logo")):
        s += 30; notes.append("No company logo")
    if not _s(d.get("requirements")).strip():
        s += 20; notes.append("Requirements are missing")
    if not _flag(d.get("has_questions")):
        s += 10; notes.append("No screening questions")
    return {"score": _clamp(s, 0, 100), "note": ", ".join(notes) if notes else "Posting looks complete"}


# ---------------- ML probability ----------------
def ml_probability(d, bundle, key):
    name = MODEL_NAMES[key]
    X = hstack([bundle["vectorizer"].transform([build_text(d)]),
                csr_matrix([meta_features(d)])]).tocsr()
    return float(bundle["models"][name].predict_proba(X)[0][1])


# ---------------- final analysis ----------------
def analyze(d, bundle=None, key="lr"):
    text = build_text(d)
    kw, sal, em, co = keyword_score(text), salary_score(d), email_score(d), completeness_score(d)

    rules = (kw["score"] * WEIGHTS["keywords"] + sal["score"] * WEIGHTS["salary"] +
             em["score"] * WEIGHTS["email"] + co["score"] * WEIGHTS["profile"])
    if kw["pay"]:
        rules = max(rules, 70)
    rules = _clamp(rules, 0, 100)

    p = ml_probability(d, bundle, key) if bundle is not None else None
    risk = rules if p is None else ML_WEIGHT * (p * 100) + (1 - ML_WEIGHT) * rules
    score = _round1(_clamp(risk, 0, 100))

    flags = []
    if kw["found"]:
        flags.append("Scam phrases found: " + ", ".join(kw["found"]))
    if kw["pay"]:
        flags.append("Asks for money or bank details: " + ", ".join(kw["pay"]))
    if sal["score"] >= 40:
        flags.append(sal["note"])
    if em["score"] >= 40:
        flags.append(em["note"])
    if co["score"] >= 40:
        flags.append(co["note"])

    return {
        "score": score,
        "label": "Fake" if score >= 50 else "Genuine",
        "category": "Low" if score < 30 else "Medium" if score < 60 else "High",
        "mode": "rules" if p is None else "ml-" + key,
        "mlPercent": None if p is None else math.floor(p * 1000 + 0.5) / 10,
        "rulesPercent": _round1(rules),
        "parts": [
            {"name": "Scam keywords", "score": kw["score"],
             "note": f"{len(kw['found'])} phrase(s) found" if kw["found"] else "No scam phrases found"},
            {"name": "Salary check", "score": sal["score"], "note": sal["note"]},
            {"name": "Email check", "score": em["score"], "note": em["note"]},
            {"name": "Profile completeness", "score": co["score"], "note": co["note"]},
        ],
        "scamPhrases": kw["found"],
        "flags": flags,
    }
