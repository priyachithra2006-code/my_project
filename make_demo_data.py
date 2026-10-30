"""Makes a SMALL FAKE demo CSV (same columns as Kaggle) ONLY to test that the code runs.
Do NOT use results from this file in your report or PPT."""
import random
import pandas as pd

random.seed(1)
real_desc = ["Develop and test web applications using Python and SQL with the engineering team.",
             "Analyze business data, build dashboards and present insights to managers.",
             "Support customers, resolve tickets and document solutions for the helpdesk.",
             "Design and maintain network infrastructure and monitor system performance."]
fake_desc = ["No experience needed earn money fast from home. Pay registration fee, no interview required.",
             "Urgent hiring immediate joining. Send bank account details and security deposit. WhatsApp us.",
             "Work from home and earn easy money. Limited seats, training fee refundable, wire transfer.",
             "100% placement guaranteed. No interview. Pay processing fee to confirm your job."]
rows = []
for i in range(600):
    fake = i % 6 == 0
    d = random.choice(fake_desc if fake else real_desc)
    rows.append({
        "job_id": i, "title": ("Urgent Data Entry " if fake else "Software Analyst ") + str(i % 7),
        "salary_range": "" if not fake and i % 2 else ("10000-90000" if fake else "400000-600000"),
        "company_profile": "" if fake else "Established company with offices in several cities and a long client list.",
        "description": d + " Role " + str(i % 11),
        "requirements": "" if fake else "Degree in computer science and good communication skills.",
        "benefits": "guaranteed job" if fake else "Health insurance and learning budget",
        "telecommuting": int(fake), "has_company_logo": int(not fake), "has_questions": int(not fake),
        "fraudulent": int(fake)})
pd.DataFrame(rows).to_csv("data/demo_sample.csv", index=False)
print("Wrote data/demo_sample.csv (FAKE demo data, rows:", len(rows), ")")
