/* JobGuard engine – pure logic (no DOM). Used by script.js in the browser. */
(function (root) {
  'use strict';

  const SCAM_PHRASES = [
    'wire transfer', 'registration fee', 'security deposit', 'processing fee', 'training fee',
    'no interview', 'without interview', 'no experience needed', 'no experience required',
    'earn money fast', 'easy money', 'work from home and earn', 'limited seats', 'pay to',
    'send money', 'bank account details', 'western union', 'urgent hiring', 'immediate joining',
    '100% placement', 'guaranteed job', 'whatsapp', 'telegram', 'click the link', 'refundable'
  ];
  // phrases that ask for money / bank details (strongest warning)
  const PAYMENT_PHRASES = [
    'registration fee', 'security deposit', 'processing fee', 'training fee', 'pay to',
    'send money', 'wire transfer', 'bank account details', 'western union'
  ];
  const FREE_DOMAINS = ['gmail.com', 'yahoo.com', 'yahoo.in', 'outlook.com', 'hotmail.com',
    'rediffmail.com', 'live.com', 'aol.com', 'protonmail.com', 'mail.com'];

  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  const str = (v) => (v === undefined || v === null) ? '' : String(v);

  /* ---------- text helpers (same steps as features.py build_text) ---------- */
  function buildText(d) {
    let t = [d.title, d.company_profile, d.description, d.requirements, d.benefits]
      .map(str).join(' ');
    t = t.replace(/<[^>]+>/g, ' ');
    t = t.replace(/http\S+|www\.\S+/g, ' url ');
    return t.replace(/\s+/g, ' ').trim().toLowerCase();
  }
  function metaFeatures(d) {
    const f = (v) => (v === 1 || v === true || v === '1' || v === 'on') ? 1 : 0;
    return [
      f(d.has_company_logo), f(d.telecommuting), f(d.has_questions),
      str(d.salary_range).trim() ? 1 : 0,
      str(d.company_profile).trim() ? 0 : 1,
      str(d.requirements).trim() ? 0 : 1
    ];
  }
  function findPhrases(text, list) {
    const t = text.toLowerCase();
    return list.filter((p) => t.indexOf(p) !== -1);
  }

  /* ---------- optional trained model (exported from scikit-learn) ---------- */
  let cache = null;
  function prepare(M) {
    if (cache && cache.M === M) return cache;
    const index = new Map();
    M.terms.forEach((t, i) => index.set(t, i));
    cache = { M, index, stop: new Set(M.stop) };
    return cache;
  }
  function tfidf(text, M) {
    const { index, stop } = prepare(M);
    const tokens = (text.match(/[\p{L}\p{N}_]{2,}/gu) || []).filter((w) => !stop.has(w));
    const grams = tokens.slice();
    for (let i = 0; i + 1 < tokens.length; i++) grams.push(tokens[i] + ' ' + tokens[i + 1]);
    const counts = new Map();
    grams.forEach((g) => { const k = index.get(g); if (k !== undefined) counts.set(k, (counts.get(k) || 0) + 1); });
    let norm = 0;
    const vec = [];
    counts.forEach((c, k) => { const v = c * M.idf[k]; vec.push([k, v]); norm += v * v; });
    norm = Math.sqrt(norm) || 1;
    return vec.map(([k, v]) => [k, v / norm]);
  }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  function mlProbability(d, which) {
    const M = root.JOBGUARD_MODEL;
    if (!M) return null;
    const x = tfidf(buildText(d), M);
    const meta = metaFeatures(d);
    const n = M.terms.length;
    if (which === 'nb' && M.nb) {
      let z = M.nb.prior;
      x.forEach(([k, v]) => { z += v * M.nb.diff[k]; });
      meta.forEach((m, j) => { z += m * M.nb.diff[n + j]; });
      return sigmoid(z);
    }
    let z = M.lr.intercept;
    x.forEach(([k, v]) => { z += v * M.lr.coef[k]; });
    meta.forEach((m, j) => { z += m * M.lr.meta[j]; });
    return sigmoid(z);
  }

  /* ---------- rule-based indicators (each gives 0-100) ---------- */
  function keywordScore(text) {
    const found = findPhrases(text, SCAM_PHRASES);
    const pay = findPhrases(text, PAYMENT_PHRASES);
    return { score: clamp(found.length * 25, 0, 100), found, pay };
  }

  function salaryScore(d) {
    const raw = str(d.salary_range).trim();
    if (!raw) return { score: 0, note: 'No salary given – nothing to check' };
    const nums = (raw.replace(/,/g, '').match(/\d+(\.\d+)?/g) || []).map(Number);
    if (!nums.length) return { score: 0, note: 'Salary has no numbers – nothing to check' };
    let low = nums[0], high = nums.length > 1 ? nums[1] : nums[0];
    if (/lakh|lpa/i.test(raw)) { low *= 100000; high *= 100000; }
    if (high === 0) return { score: 0, note: 'Salary is 0 – nothing to check' };
    if (low > high) return { score: 60, note: 'Minimum salary is higher than maximum' };
    const yearly = d.salary_period === 'yearly' || /lpa|per annum|per year|yearly|annual/i.test(raw);
    const monthly = yearly ? high / 12 : high;
    const entry = d.experience === 'entry';
    if (monthly >= 100000 && entry) return { score: 90, note: 'Very high salary (about ₹' + Math.round(monthly).toLocaleString('en-IN') + '/month) for an entry-level role' };
    if (monthly >= 100000) return { score: 40, note: 'High salary (about ₹' + Math.round(monthly).toLocaleString('en-IN') + '/month) – verify it' };
    if (low > 0 && high / low >= 5) return { score: 70, note: 'Salary range is unusually wide' };
    return { score: 0, note: 'Salary looks normal' };
  }

  function emailScore(d) {
    const email = str(d.contact_email).trim().toLowerCase();
    if (!email) return { score: 0, note: 'No recruiter email given' };
    const m = email.match(/^[^@\s]+@([^@\s]+\.[a-z]{2,})$/);
    if (!m) return { score: 60, note: 'Email address looks invalid' };
    const domain = m[1];
    if (FREE_DOMAINS.indexOf(domain) !== -1) return { score: 80, note: 'Free email domain (@' + domain + ') instead of a company email' };
    const company = str(d.company_name).toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(' ')
      .filter((w) => w.length >= 3 && ['pvt', 'ltd', 'inc', 'llc', 'the', 'private', 'limited', 'company', 'technologies'].indexOf(w) === -1);
    if (company.length && !company.some((w) => domain.indexOf(w) !== -1))
      return { score: 40, note: 'Email domain (' + domain + ') does not match the company name' };
    return { score: 0, note: 'Email domain looks fine' };
  }

  function completenessScore(d) {
    let s = 0; const notes = [];
    if (!str(d.company_profile).trim()) { s += 40; notes.push('Company profile is missing'); }
    if (!(d.has_company_logo === 1 || d.has_company_logo === true)) { s += 30; notes.push('No company logo'); }
    if (!str(d.requirements).trim()) { s += 20; notes.push('Requirements are missing'); }
    if (!(d.has_questions === 1 || d.has_questions === true)) { s += 10; notes.push('No screening questions'); }
    return { score: clamp(s, 0, 100), note: notes.length ? notes.join(', ') : 'Posting looks complete' };
  }

  /* ---------- final analysis ---------- */
  const WEIGHTS = { keywords: 0.35, salary: 0.20, email: 0.25, profile: 0.20 };
  const ML_WEIGHT = 0.6;

  function analyze(d, which) {
    const text = buildText(d);
    const kw = keywordScore(text);
    const sal = salaryScore(d);
    const em = emailScore(d);
    const co = completenessScore(d);

    let rules = kw.score * WEIGHTS.keywords + sal.score * WEIGHTS.salary +
                em.score * WEIGHTS.email + co.score * WEIGHTS.profile;
    if (kw.pay.length) rules = Math.max(rules, 70);           // asking for money is a strong sign
    rules = clamp(rules, 0, 100);

    const p = mlProbability(d, which);
    const risk = p === null ? rules : ML_WEIGHT * (p * 100) + (1 - ML_WEIGHT) * rules;
    const score = Math.round(clamp(risk, 0, 100) * 10) / 10;

    const flags = [];
    if (kw.found.length) flags.push('Scam phrases found: ' + kw.found.join(', '));
    if (kw.pay.length) flags.push('Asks for money or bank details: ' + kw.pay.join(', '));
    if (sal.score >= 40) flags.push(sal.note);
    if (em.score >= 40) flags.push(em.note);
    if (co.score >= 40) flags.push(co.note);

    return {
      score,
      label: score >= 50 ? 'Fake' : 'Genuine',
      category: score < 30 ? 'Low' : score < 60 ? 'Medium' : 'High',
      mode: p === null ? 'rules' : (which === 'nb' ? 'ml-nb' : 'ml-lr'),
      mlPercent: p === null ? null : Math.round(p * 1000) / 10,
      rulesPercent: Math.round(rules * 10) / 10,
      parts: [
        { name: 'Scam keywords', score: kw.score, note: kw.found.length ? kw.found.length + ' phrase(s) found' : 'No scam phrases found' },
        { name: 'Salary check', score: sal.score, note: sal.note },
        { name: 'Email check', score: em.score, note: em.note },
        { name: 'Profile completeness', score: co.score, note: co.note }
      ],
      scamPhrases: kw.found,
      flags
    };
  }

  const api = { analyze, buildText, metaFeatures, mlProbability, tfidf, SCAM_PHRASES, WEIGHTS, ML_WEIGHT };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.JobGuardEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
