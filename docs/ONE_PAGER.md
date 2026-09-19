# ⚡ NeuroScope — Project One-Pager

> The 60-second version for reviewers, recruiters, and hiring managers.
> Details: [Playbook](PLAYBOOK.md) · [Model Card](MODEL_CARD.md) · [ADRs](ARCHITECTURE_DECISIONS.md) · [Resume Bullets](RESUME_BULLETS.md)

---

## The pitch

**NeuroScope turns a raw customer CSV into churn scores, customer segments, and
executive-ready AI playbooks — in under a minute, fully explainable, fully tested.**

It's what a Customer Success team wishes they had instead of a spreadsheet, built with
the engineering rigor of a production ML service.

---

## What it does

| Feature | Under the hood |
|---|---|
| 🔮 **Churn prediction** | 4 competing classifiers (LogReg, Random Forest, GBM, Extra Trees) on identical stratified splits; **0.80 AUC-ROC** |
| 🔍 **Explainability** | Global SHAP rankings + per-prediction waterfalls — every score shows its *why* |
| 🧩 **Segmentation** | K-Means, DBSCAN, Hierarchical compete on silhouette; 3D PCA scatter, persona cards, CSV export |
| 🤖 **AI executive copilot** | Groq LLaMA-3.3-70B streams grounded retention playbooks from real centroids & SHAP values — with offline fallback |
| 🧪 **Interactive what-if** | Adjust any customer attribute, get live churn probability + SHAP breakdown |
| 📥 **Bring your own data** | CSV upload with schema auto-detection and graceful degradation |

---

## Engineering highlights (what a senior reviewer will notice)

- **Clean architecture:** UI-free Python core (`src/`), pages are thin orchestration;
  only one module touches Streamlit caching.
- **SHAP version-tolerance layer:** normalizes 3 explainer output layouts across
  shap/sklearn versions, with tested fallback chains — the class of bug that breaks
  real deployments *months* later, handled up front.
- **4-layer test suite, 92% coverage (70% enforced):** data unit → ML unit (incl. all
  SHAP layouts) → chart builders → **AppTest E2E that boots pages and clicks buttons**.
  CI (GitHub Actions) enforces it across Python 3.10–3.12.
- **Graceful degradation everywhere:** no Groq key → mock mode; missing CSV columns →
  warnings; SHAP failure → importances fallback. The demo never dies on stage.
- **9 ADRs** documenting every significant decision *and the alternatives rejected*.

---

## Stack

Python 3.10–3.12 · scikit-learn · SHAP · Plotly · pandas · Streamlit · Groq SDK
(LLaMA-3.3-70B) · pytest + pytest-cov + AppTest · ruff · GitHub Actions · Docker

---

## By the numbers

| Metric | Value |
|---|---|
| Churn model AUC-ROC | **0.8045** (best of 4) |
| Test coverage | **92%** (54 tests, 4 layers) |
| Classifiers / clustering algos | 4 / 3 |
| Features engineered | 9 (+ binary target) |
| Dataset | 1,500 rows, seeded & reproducible |
| Boot to first insight | < 60 s |

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py          # → http://localhost:8501
# optional AI: set GROQ_API_KEY (free at console.groq.com) — otherwise mock mode
```

---

## Honest limitations (said out loud, like in the docs)

Metrics are on synthetic data — a demonstration of engineering, not market performance.
Recall at the default threshold is ~0.42 (fix: threshold tuning before operational use);
probabilities are uncalibrated; no drift monitoring. All documented in the
[Model Card](MODEL_CARD.md), because knowing what a system *can't* do is part of
building it.
