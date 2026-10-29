JOBGUARD WEBSITE (plain HTML + CSS + JavaScript)

HOW TO OPEN
  Double-click index.html. That's all. No Flask, no Python, no internet needed.

FILES
  index.html  - the page
  style.css   - colours and design
  script.js   - buttons, result screen, history
  engine.js   - the detection logic (rules + optional ML model)
  model.js    - your trained ML model (empty until you export it)

RULE-BASED MODE (default)
  Without a trained model the score comes from rule checks only
  (scam keywords, salary, recruiter email, profile completeness).
  The page tells you this clearly at the top.

USE YOUR REAL TRAINED MODEL (optional)
  1. Put fake_job_postings.csv in the project's data folder
  2. python train.py data/fake_job_postings.csv
  3. python export_model.py        (this fills website/model.js)
  4. Refresh index.html. The top message turns green and the
     "How it works" tab shows YOUR real accuracy numbers.
  The website can run Logistic Regression and Naive Bayes.
  Random Forest is too big for a browser, so it stays in the Flask app.

Your history is saved only in your own browser (localStorage).
