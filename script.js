/* JobGuard website – user interface (needs engine.js and model.js) */
(function () {
  'use strict';
  const E = window.JobGuardEngine;
  const M = window.JOBGUARD_MODEL;
  const $ = (id) => document.getElementById(id);
  const COLORS = { low: '#10b981', medium: '#f59e0b', high: '#ef4444' };
  const ADVICE = {
    low: 'Looks fine, but still verify the company on its official website before sharing documents.',
    medium: 'Be careful. Check the company and recruiter details before you reply.',
    high: 'Very suspicious! Do NOT pay any fee or share personal or bank details.'
  };
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const colorFor = (s) => s < 30 ? COLORS.low : s < 60 ? COLORS.medium : COLORS.high;

  /* ---------- theme ---------- */
  const themeBtn = $('theme');
  const paintTheme = () => { themeBtn.textContent = document.documentElement.getAttribute('data-theme') === 'dark' ? '☀️' : '🌙'; };
  themeBtn.onclick = () => {
    const n = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', n);
    try { localStorage.setItem('jg-theme', n); } catch (e) {}
    paintTheme();
  };
  paintTheme();

  /* ---------- tabs ---------- */
  function openTab(name) {
    if (!$(name)) name = 'analyze';
    document.querySelectorAll('.page').forEach((p) => p.classList.toggle('hidden', p.id !== name));
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
    if (name === 'history') renderHistory();
    window.scrollTo(0, 0);
  }
  document.querySelectorAll('[data-tab]').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault(); openTab(a.dataset.tab); history.replaceState(null, '', '#' + a.dataset.tab);
  }));
  openTab((location.hash || '#analyze').slice(1));

  /* ---------- model status ---------- */
  const modeNote = $('modeNote');
  if (M) {
    modeNote.textContent = '✅ Trained ML model loaded (' + M.terms.length + ' text features) + rule checks';
    modeNote.classList.add('ml');
    $('modelPick').classList.remove('hidden');
  } else {
    modeNote.textContent = '⚠️ Rule-based mode: no trained model file found, so the score uses the rule checks only';
  }
  const W = E.WEIGHTS;
  $('formula').textContent = M
    ? 'Risk = ' + E.ML_WEIGHT + ' × ML probability + ' + (1 - E.ML_WEIGHT).toFixed(1) + ' × rule score. Rule score = ' + W.keywords + ' × keywords + ' + W.salary + ' × salary + ' + W.email + ' × email + ' + W.profile + ' × profile.'
    : 'Risk = rule score = ' + W.keywords + ' × keywords + ' + W.salary + ' × salary + ' + W.email + ' × email + ' + W.profile + ' × profile. (With a trained model: 0.6 × ML + 0.4 × rules.)';

  function renderMetrics() {
    const box = $('metricsCard');
    if (!M || !M.metrics) {
      box.innerHTML = '<h3>🤖 Model results</h3><p class="muted">No trained model is loaded, so no accuracy is shown here. Train the model (<code>train.py</code>) and run <code>export_model.py</code> to see your real results.</p>';
      return;
    }
    let rows = '';
    Object.keys(M.metrics).forEach((n) => {
      const r = M.metrics[n];
      rows += '<tr><td>' + esc(n) + (n === M.best_model ? ' ★' : '') + '</td><td>' + r.accuracy + '%</td><td>' + r.precision + '%</td><td>' + r.recall + '%</td><td>' + r.f1 + '%</td></tr>';
    });
    box.innerHTML = '<h3>🤖 Model results (your test set)</h3><div class="scroll"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>' + rows +
      '</tbody></table></div><p class="muted small">Precision, recall and F1 are for the fake class. Test rows: ' + M.test_rows + '. This website uses Logistic Regression or Naive Bayes (Random Forest is too big to run in a browser).</p>';
  }
  renderMetrics();

  /* ---------- analyze ---------- */
  const form = $('form');
  function readForm() {
    const d = {};
    new FormData(form).forEach((v, k) => { d[k] = v; });
    ['has_company_logo', 'telecommuting', 'has_questions'].forEach((k) => { d[k] = form[k].checked ? 1 : 0; });
    return d;
  }
  function countUp(el, target) {
    const t0 = performance.now();
    (function step(t) {
      const p = Math.min((t - t0) / 1100, 1);
      el.textContent = (target * (1 - Math.pow(1 - p, 3))).toFixed(1) + '%';
      if (p < 1) requestAnimationFrame(step);
    })(t0);
  }
  function highlight(text, phrases) {
    let html = esc(text);
    phrases.forEach((p) => {
      html = html.replace(new RegExp(p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), (m) => '<mark>' + m + '</mark>');
    });
    return html;
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const d = readForm();
    if (!(d.title || '').trim() && !(d.description || '').trim()) {
      alert('Please enter at least a job title or a description.');
      return;
    }
    const go = $('go');
    go.disabled = true; go.innerHTML = '<span class="spin"></span>Analyzing...';
    setTimeout(() => {                                   // small delay so the spinner is visible
      const out = E.analyze(d, d.model);
      show(out, d);
      saveHistory(d, out);
      go.disabled = false; go.innerHTML = '🔍 Analyze posting';
    }, 350);
  });

  function show(out, d) {
    const cat = out.category.toLowerCase();
    $('placeholder').classList.add('hidden');
    $('result').classList.remove('hidden');

    const fill = $('fill');
    fill.style.stroke = COLORS[cat];
    fill.style.strokeDashoffset = 502;
    void fill.getBoundingClientRect();
    fill.style.strokeDashoffset = 502 - 502 * out.score / 100;
    countUp($('score'), out.score);

    const badge = $('badge');
    badge.className = 'badge ' + cat;
    badge.textContent = (cat === 'low' ? '✅ ' : cat === 'medium' ? '⚠️ ' : '🚨 ') + out.label + ' · ' + out.category + ' risk';
    $('advice').textContent = ADVICE[cat];
    $('how').textContent = out.mode === 'rules'
      ? 'Score from rule checks only (rule score ' + out.rulesPercent + '%)'
      : (out.mode === 'ml-nb' ? 'Naive Bayes' : 'Logistic Regression') + ' says ' + out.mlPercent + '% fake · rule score ' + out.rulesPercent + '%';

    const parts = $('parts'); parts.innerHTML = '';
    const rows = out.mlPercent === null ? out.parts : [{ name: 'ML model', score: out.mlPercent, note: 'Probability that the text looks like a fake posting' }].concat(out.parts);
    rows.forEach((p) => {
      const div = document.createElement('div'); div.className = 'part';
      div.innerHTML = '<div class="top"><span>' + esc(p.name) + '</span><span>' + Math.round(p.score) + '/100</span></div>' +
        '<div class="meter"><i style="background:' + colorFor(p.score) + '"></i></div><div class="note">' + esc(p.note) + '</div>';
      parts.appendChild(div);
      requestAnimationFrame(() => requestAnimationFrame(() => { div.querySelector('i').style.width = p.score + '%'; }));
    });

    const ul = $('flags'); ul.innerHTML = '';
    if (!out.flags.length) { const li = document.createElement('li'); li.className = 'ok'; li.textContent = 'No strong warning signs found'; ul.appendChild(li); }
    out.flags.forEach((f, i) => { const li = document.createElement('li'); li.textContent = f; li.style.animationDelay = (i * 0.08) + 's'; ul.appendChild(li); });

    const fullText = [d.title, d.description, d.requirements, d.benefits].filter(Boolean).join('\n');
    $('hl').innerHTML = out.scamPhrases.length ? highlight(fullText, out.scamPhrases).replace(/\n/g, '<br>') : '<span class="muted">No scam phrases found in the text.</span>';

    if (window.innerWidth < 820) $('resultCard').scrollIntoView({ behavior: 'smooth' });
  }

  /* ---------- examples ---------- */
  function fillForm(o) {
    form.reset();
    Object.keys(o).forEach((k) => {
      const el = form[k]; if (!el) return;
      if (el.type === 'checkbox') el.checked = !!o[k]; else el.value = o[k];
    });
  }
  $('clear').onclick = () => { form.reset(); $('result').classList.add('hidden'); $('placeholder').classList.remove('hidden'); };
  $('ex-fake').onclick = () => fillForm({
    title: 'Urgent Hiring - Work From Home Data Entry', company_name: 'Global Jobs Hub',
    description: 'No experience needed. Earn money fast from home. No interview required, immediate joining. Pay a small registration fee to receive your training kit, refundable after first month. Contact us on WhatsApp.',
    benefits: '100% placement guaranteed', salary_range: '150000', salary_period: 'monthly', experience: 'entry',
    contact_email: 'hr.jobsample@gmail.com', telecommuting: 1 });
  $('ex-real').onclick = () => fillForm({
    title: 'Junior Software Engineer', company_name: 'Acme Technologies',
    company_profile: 'We are a software services company with offices in Chennai and Bengaluru, building web products for banking clients since 2008.',
    description: 'Join our engineering team to develop and test web applications using Python and JavaScript. You will work with senior engineers, take part in code reviews and learn our agile process.',
    requirements: 'B.E/B.Tech in Computer Science or related field. Good knowledge of Python, SQL and data structures.',
    benefits: 'Health insurance, learning budget, hybrid work', salary_range: '25000-40000', salary_period: 'monthly', experience: 'entry',
    contact_email: 'careers@acmetechnologies.com', has_company_logo: 1, has_questions: 1 });

  /* ---------- history (saved in this browser) ---------- */
  const KEY = 'jobguard-history';
  function loadHistory() { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return memory; } }
  let memory = [];
  function saveHistory(d, out) {
    const list = loadHistory();
    list.unshift({ time: new Date().toLocaleString(), title: (d.title || '(no title)').slice(0, 80), mode: out.mode, score: out.score, label: out.label, category: out.category });
    memory = list.slice(0, 100);
    try { localStorage.setItem(KEY, JSON.stringify(memory)); } catch (e) {}
  }
  const MODE = { rules: 'Rules only', 'ml-lr': 'ML (LogReg) + rules', 'ml-nb': 'ML (NB) + rules' };
  function renderHistory() {
    const list = loadHistory();
    const fake = list.filter((x) => x.label === 'Fake').length;
    $('st-total').textContent = list.length; $('st-fake').textContent = fake; $('st-real').textContent = list.length - fake;
    const pct = list.length ? Math.round(fake / list.length * 360) : 0;
    $('donut').style.background = list.length ? 'conic-gradient(#ef4444 0deg ' + pct + 'deg, #10b981 ' + pct + 'deg 360deg)' : 'var(--line)';
    const cats = { Low: 0, Medium: 0, High: 0 };
    list.forEach((x) => { cats[x.category]++; });
    const max = Math.max(1, cats.Low, cats.Medium, cats.High);
    $('bars').innerHTML = ['Low', 'Medium', 'High'].map((c) =>
      '<div><b>' + cats[c] + '</b><i style="height:' + Math.round(cats[c] / max * 120) + 'px;background:' + COLORS[c.toLowerCase()] + '"></i>' + c + '</div>').join('');
    $('histBody').innerHTML = list.length ? list.slice(0, 50).map((x) =>
      '<tr><td>' + esc(x.time) + '</td><td>' + esc(x.title) + '</td><td>' + esc(MODE[x.mode] || x.mode) + '</td><td>' + x.score + '% <span class="pill ' + x.category.toLowerCase() + '">' + x.category + '</span></td><td class="tag ' + x.label.toLowerCase() + '">' + x.label + '</td></tr>').join('')
      : '<tr><td colspan="5" class="nohist">No checks yet. Go to Analyze and check a posting.</td></tr>';
  }
  $('clearHist').onclick = () => {
    if (!confirm('Delete all history from this browser?')) return;
    memory = []; try { localStorage.removeItem(KEY); } catch (e) {}
    renderHistory();
  };
})();
