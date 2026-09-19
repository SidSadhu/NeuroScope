<div align="center">

# ⚡ NeuroScope

### AI-Powered Customer Intelligence & Predictive Analytics Platform

*Churn prediction you can explain. Segments you can act on. Strategies an LLM writes from your actual data — not its imagination.*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable-00d4ff?style=for-the-badge)](https://shap.readthedocs.io/)
[![Groq LLaMA-3.3](https://img.shields.io/badge/Groq-LLaMA--3.3--70B-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-87%20passing-10b981?style=for-the-badge&logo=pytest&logoColor=white)](#-testing--code-quality)
[![Coverage](https://img.shields.io/badge/Coverage-92%25-brightgreen?style=for-the-badge)](#-testing--code-quality)
[![Code Style](https://img.shields.io/badge/Code%20Style-ruff-261230?style=for-the-badge)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

[Key Features](#-what-it-delivers) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Deployment](#-deployment) • [Documentation](#-documentation) • [Testing](#-testing--code-quality)

</div>

---

## 🌟 Why NeuroScope Exists

Customer Success and Growth teams sit on raw telemetry but starve for insight: churn
risk lives in spreadsheets, segmentation is guesswork, and ML outputs are unreadable to
the people who need them.

**NeuroScope closes that gap** — one app that takes a customer CSV and returns:

| | Deliverable | How |
|---|---|---|
| 🔮 | **Churn risk scores** — with a *why* attached to every single prediction | 4 competing classifiers (Logistic Regression, Random Forest, Gradient Boosting, Extra Trees) on identical stratified splits, winner picked by AUC-ROC, explained via SHAP waterfalls |
| 🧩 | **Customer segments** — persona cards, radar profiles, 3D projections, CSV export | K-Means, DBSCAN & Hierarchical compete on silhouette; PCA 3D scatter; per-cluster behavioral profiles |
| 🤖 | **Executive playbooks** — retention strategies, A/B experiment ideas, grounded in the actual numbers | Groq LLaMA-3.3-70B streams narratives *from real cluster centroids and SHAP values*, with a labeled offline fallback |
| 🔬 | **A what-if sandbox** — adjust any customer attribute, watch churn probability move live | Interactive form + semi-circular risk gauge + per-prediction SHAP breakdown |

**Bring your own data** — upload any CSV; schema is auto-detected with graceful
degradation (missing target → classification disables, segmentation keeps working).
No data? A seeded 1,500-row synthetic dataset with realistic churn patterns is built in.

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (pages/)                  │
│    Overview · Churn Prediction · Segmentation · AI Insights      │
└────────────────────────────┬─────────────────────────────────────┘
                             │ imports only — UI is a thin layer
┌────────────────────────────▼─────────────────────────────────────┐
│                       Application Core (src/)                    │
│                                                                  │
│   src/data/            src/models/            src/ai/            │
│   ├── generator.py     ├── classifiers.py     └── insights.py    │
│   │   seeded synth     │   4-model arena         Groq LLM        │
│   └── loader.py        ├── clustering.py            (streaming)   │
│       cache·CSV·valid  │   3 algos + PCA                           │
│                        └── explainability.py                     │
│                            SHAP + fallbacks                      │
│                                                                  │
│        src/visualization/charts.py — 12 Plotly builders          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              Customer CSV upload or seeded generator
```

**The design rule that makes everything testable:** pages orchestrate, `src/` does the
work. Only one module touches Streamlit caching; the entire ML/AI core runs in CI with
no browser. Nine [Architecture Decision Records](docs/ARCHITECTURE_DECISIONS.md)
document every significant choice — and the alternatives that were rejected.

---

## ⚡ Quick Start

```bash
git clone https://github.com/SidSadhu/NeuroScope.git
cd NeuroScope

python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (macOS/Linux: source .venv/bin/activate)

pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py                # → http://localhost:8501
```

**Optional — live AI insights:** grab a free key at
[console.groq.com](https://console.groq.com/) and either paste it in the AI page's
sidebar, set `GROQ_API_KEY`, or add it to `.streamlit/secrets.toml`
(see [`secrets.toml.example`](.streamlit/secrets.toml.example)). Without a key the app
runs its clearly-labeled mock mode — everything else works identically.

---

## 📊 What the Models Actually Achieve

Trained/evaluated on the built-in dataset (1,500 rows, ~31% churn rate, stratified
80/20 split, seed 42 — fully reproducible):

| Model | Accuracy | Precision | Recall | F1 | **AUC-ROC** |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.7567 | 0.6842 | 0.4149 | 0.5166 | **0.8045** |
| Random Forest | 0.7533 | 0.6724 | 0.4149 | 0.5132 | 0.7926 |
| Gradient Boosting | 0.7533 | 0.6562 | 0.4468 | 0.5316 | 0.7871 |
| Extra Trees | 0.7300 | 0.6327 | 0.3298 | 0.4336 | 0.7758 |

**Top churn drivers (SHAP):** `support_calls_3m` → `login_frequency` →
`satisfaction_score` → `tenure_months` → `last_purchase_days`

> 📌 **Honesty section** (rare, we know): these are *synthetic-data* metrics — a
> demonstration of engineering, not market performance. Recall at the default 0.5
> threshold is ~0.42 (tune the threshold before operational use), probabilities are
> uncalibrated, and silhouettes are moderate (0.13–0.18). All of it is measured and
> documented in the [Model Card](docs/MODEL_CARD.md), because knowing what a system
> *can't* do is part of building it.

---

## 🧪 Engineering Highlights

The details a senior reviewer looks for:

- **SHAP version-tolerance layer** — SHAP changed its output contract (legacy list →
  2D → 3D per-class arrays). A central normalizer accepts all three layouts with
  tested fallback chains (`feature_importances_` → `coef_` → graceful degradation).
  This is the exact bug class that kills real deployments months later — handled
  up front, unit-tested per layout.
- **4-layer test suite — 87 tests, 92% coverage, 70% gate enforced** — data unit →
  ML unit (incl. every SHAP layout + fallback branch) → chart builders →
  **AppTest end-to-end tests that boot every page and click the buttons**. The E2E
  layer has already caught two bugs unit tests missed.
- **Graceful degradation everywhere** — no Groq key → mock mode; bad CSV → warnings +
  partial features; SHAP failure → fallback importances. The demo never dies on stage.
- **Streaming LLM contract** — insights are generator functions yielding chunks; the
  vendor call is 30 lines behind a swap-friendly seam, not a framework lock-in.
- **One CSS token file** — the whole dark design system lives in one `:root` block;
  rebranding is a one-file change.
- **CI matrix** — Python 3.10/3.11/3.12 × (ruff + pytest + coverage gate) on every push.

---

## 🚀 Deployment

| Path | Best for | Time |
|---|---|---|
| **Streamlit Community Cloud** | free public demo, auto-deploys from `main` | ~5 min |
| **Docker / Cloud Run / Fly / ECS** | portable self-hosting behind your own domain | ~10 min |

Full step-by-step runbook (including Streamlit Cloud secrets, health checks, and
per-cloud command sketches) in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**:

```bash
# Docker, the short version:
docker compose up --build -d
curl -f http://localhost:8501/_stcore/health
```

---

## 📚 Documentation

Documented the way real teams document — each doc has one job and one audience:

| Doc | Read it when you want to… |
|---|---|
| [**Developer Playbook**](docs/PLAYBOOK.md) | Onboard or extend the app — architecture, ML/AI internals, design system, extension recipes, troubleshooting |
| [**Model Card**](docs/MODEL_CARD.md) | Evaluate the ML honestly — data provenance, metrics, calibration, limitations, ethics |
| [**Architecture Decisions**](docs/ARCHITECTURE_DECISIONS.md) | Understand *why* — 9 ADRs with context, alternatives, consequences |
| [**Implementation Chronicle**](docs/IMPLEMENTATION_CHRONICLE.md) | See the full build log — every phase, decision, bug, and verification from empty folder to deployment |
| [**Deployment Runbook**](docs/DEPLOYMENT.md) | Ship it — Streamlit Cloud + Docker + cloud targets |
| [**Project One-Pager**](docs/ONE_PAGER.md) | Get the 60-second pitch |

---

## 🧪 Testing & Code Quality

```bash
pytest                                   # 87 tests: unit + charts + E2E smoke
pytest --cov=src --cov-report=term-missing
ruff check . && ruff format --check .    # CI-enforced lint + format
streamlit run app.py                     # manual pass: train → tabs → cluster → AI
```

GitHub Actions runs the full matrix (Python 3.10–3.12) on every push and PR:
lint → format → tests → coverage gate (70% floor, currently 92%).

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Streamlit 1.55, custom glassmorphism design tokens, `st.navigation` |
| **ML** | scikit-learn (LogReg, Random Forest, Gradient Boosting, Extra Trees, K-Means, DBSCAN, Agglomerative, PCA) |
| **Explainability** | SHAP (TreeExplainer / LinearExplainer) with version-tolerant normalization + fallbacks |
| **GenAI** | Groq SDK · `llama-3.3-70b-versatile` · streaming generators · offline mock mode |
| **Visualization** | Plotly Graph Objects — ROC, radar, 3D scatter, donut, gauge, heatmap |
| **Data** | pandas · NumPy · seeded synthetic generator · CSV validation |
| **Quality** | pytest + pytest-cov + Streamlit AppTest · ruff · GitHub Actions matrix · Docker |

---

## 🗺️ Roadmap

- [ ] Decision-threshold tuning & probability calibration (recall-first operations)
- [ ] Flag-gated cross-validation mode for the classifier arena
- [ ] Drift monitoring + model registry when wired to production data
- [ ] Subgroup fairness evaluation on real (non-synthetic) data
- [ ] Multi-LLM support behind the existing streaming contract

---

## 👤 Author

**Siddharth Sadhu** — [GitHub](https://github.com/SidSadhu)

## 📄 License

MIT — see [LICENSE](LICENSE).
