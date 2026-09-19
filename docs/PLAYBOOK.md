# 🛠️ NeuroScope Developer Playbook

> The complete handbook for developers building, extending, or maintaining NeuroScope.
> **Audience:** contributors and engineers onboarding to this codebase.
> **Status:** v2.1 · September 2026

---

## 1. What Is NeuroScope?

NeuroScope is an end-to-end **customer intelligence platform** that turns raw customer
data into three deliverables:

| Deliverable | How | Where |
|---|---|---|
| **Churn risk scores** | 4 competing classifiers + SHAP explainability | `pages/churn_prediction.py` |
| **Customer segments** | K-Means, DBSCAN, Hierarchical clustering + personas | `pages/segmentation.py` |
| **Executive narratives** | Groq LLaMA-3.3-70B streaming analysis of ML artifacts | `pages/ai_insights.py` |

It is a **Streamlit multi-page app** with a modular Python core — the UI is a thin layer
over importable, testable `src/` modules. The LLM adds narrative on top of ML outputs;
it never computes the numbers itself.

### Who it's for
- **Customer Success teams** — identify at-risk accounts before they churn
- **Growth/Marketing teams** — target segments with tailored campaigns
- **Data scientists** — a production-pattern reference app (pipelines, SHAP, caching, CI)

### What it is NOT
- Not a production scoring API (no REST endpoint, no auth, no model registry)
- Not a real-time system (batch scoring on page interaction)
- Not a replacement for feature stores / MLOps pipelines

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (pages/)                  │
│   overview · churn_prediction · segmentation · ai_insights      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ imports only
┌───────────────────────────▼─────────────────────────────────────┐
│                      Application Core (src/)                    │
│                                                                 │
│  src/data/            src/models/           src/ai/             │
│  ├── generator.py     ├── classifiers.py    └── insights.py     │
│  │   synthetic data   │   4 sklearn models      Groq LLM       │
│  └── loader.py        ├── clustering.py                          │
│      CSV/cache/valid  │   3 algos + PCA                          │
│                       └── explainability.py                      │
│                           SHAP + fallbacks                      │
│                                                                 │
│  src/visualization/charts.py — 12 Plotly chart builders         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              Customer CSV upload or synthetic generator
```

**Key rule:** `pages/` never trains models, computes SHAP, or calls Groq inline with
business logic mixed in — everything lives in `src/` and is imported. This is what
makes the core testable without a UI.

### Module dependency order
`generator → loader → classifiers/clustering/explainability → charts → pages → app`

Nothing in `src/` imports Streamlit except `loader.py` (for `@st.cache_data`) — and
that's the only acceptable place. Everything else is pure Python so pytest runs fast
and in CI without a browser.

### Data flow (churn page example)
1. `load_default_dataset()` — cached, 1,500 rows
2. `get_features_and_target(df)` → `X, y`
3. User clicks **Train All Models** → `train_all_models(X, y)`
4. Results dict stored in `st.session_state["churn_results"]`
5. Every tab reads from session state — never retrains on rerun
6. SHAP computed lazily on first tab visit, cached by model name

---

## 3. Project Layout

```text
predict-segment-app/
├── app.py                      # Entry: global CSS + st.navigation
├── pages/                      # One file per Streamlit page
│   ├── overview.py             # Hero, KPIs, donut, correlation heatmap
│   ├── churn_prediction.py     # 4-model arena, SHAP, what-if form
│   ├── segmentation.py         # Clustering, explorer, personas, export
│   └── ai_insights.py          # Groq streaming, 2 modes
├── src/
│   ├── data/
│   │   ├── generator.py        # Synthetic data (seeded, reproducible)
│   │   └── loader.py           # Cache, CSV validation, X/y split
│   ├── models/
│   │   ├── classifiers.py      # Registry, ModelResult, metrics
│   │   ├── clustering.py       # ClusteringResult, elbow, PCA
│   │   └── explainability.py   # SHAP (version-tolerant) + fallbacks
│   ├── ai/
│   │   └── insights.py         # Groq client, prompts, mock fallback
│   └── visualization/
│       └── charts.py           # All Plotly figures, dark theme
├── tests/                      # 54 tests: unit + charts + E2E smoke
│   ├── test_data.py            # Schema, ranges, reproducibility
│   ├── test_models.py          # Classifiers, clustering, SHAP, fallbacks
│   ├── test_insights.py        # Groq mocking, API key resolution
│   ├── test_charts.py          # Every chart builder
│   └── test_app_smoke.py       # AppTest E2E: boots, clicks, asserts
├── docs/                       # This playbook + companion docs
├── .github/workflows/ci.yml    # Ruff + pytest matrix (3.10–3.12)
├── Dockerfile / docker-compose.yml
└── pyproject.toml              # pytest, ruff, coverage config
```

---

## 4. Getting Started

```bash
git clone https://github.com/SidSadhu/predict-segment-app.git
cd predict-segment-app

python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1   macOS/Linux: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install ruff pytest-cov            # dev tools (CI installs these too)

streamlit run app.py                   # → http://localhost:8501
```

**Optional AI:** set `GROQ_API_KEY` (env var, `.streamlit/secrets.toml`, or the in-app
sidebar). Without a key, the AI page runs its built-in mock synthesizer — everything
else works identically.

**Docker:**
```bash
docker-compose up --build
```

---

## 5. The ML Layer — How It Actually Works

### 5.1 Churn classification (`src/models/classifiers.py`)

Four models compete on an identical 80/20 stratified split (`random_state=42`):

| Model | Config | Needs scaler? |
|---|---|---|
| Logistic Regression | C=1.0, max_iter=2000 | ✅ `StandardScaler` in-pipeline |
| Random Forest | 120 trees | ❌ |
| Gradient Boosting | 120 rounds, lr=0.1 | ❌ |
| Extra Trees | 120 trees | ❌ |

Each returns a `ModelResult` dataclass: fitted `pipeline`, `metrics`, `y_test/y_pred/y_prob`,
`X_test/X_train`, `feature_names`. Scaling lives **inside** the sklearn Pipeline, so
inference on raw rows is leakage-safe and the deployed artifact is one object.

**Metrics** (`_compute_metrics`): Accuracy, Precision, Recall, F1, AUC-ROC — all
rounded to 4 dp, `zero_division=0` guards.

**Model selection:** `get_best_model(results)` — highest AUC-ROC wins. Rationale:
AUC is threshold-independent and robust to the class imbalance (~31% churn).

**Reference metrics** (synthetic dataset, seed 42 — reproduce with `train_all_models`):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.7567 | 0.6842 | 0.4149 | 0.5166 | **0.8045** |
| Random Forest | 0.7533 | 0.6724 | 0.4149 | 0.5132 | 0.7926 |
| Gradient Boosting | 0.7533 | 0.6562 | 0.4468 | 0.5316 | 0.7871 |
| Extra Trees | 0.7300 | 0.6327 | 0.3298 | 0.4336 | 0.7758 |

> Linear wins on this synthetic set because the generator uses mostly additive
> signal. On real data expect tree ensembles to win — that's exactly why 4 models run.

### 5.2 SHAP explainability (`src/models/explainability.py`)

Two entry points:
- `compute_shap_feature_importance(pipeline, X, name)` — mean |SHAP| per feature, ranked
- `compute_shap_waterfall_values(pipeline, row, name)` — per-feature contributions for one prediction

**The version-tolerance layer (read this before touching SHAP code):**
`_select_positive_class()` normalizes three known SHAP output layouts to a clean
`(n_samples, n_features)` matrix + scalar base value:
1. Legacy list `[class0, class1]` (shap < 0.45)
2. 2D `(n_samples, n_features)` — single output
3. 3D `(n_samples, n_features, n_classes)` — shap ≥ 0.45 with sklearn ≥ 1.4

Every call is wrapped in `try/except` with a graceful fallback:
- Tree models → `feature_importances_`
- Linear models → `|coef_|`
- Neither → ones vector (UI degrades, never crashes)

**Verified compatible:** shap 0.49.1, sklearn 1.7.2.

### 5.3 Clustering (`src/models/clustering.py`)

| Algorithm | Behavior | Notes |
|---|---|---|
| K-Means | user-chosen K (2–8) | Elbow + silhouette helpers guide K |
| DBSCAN | auto-detects clusters | k-NN distance curve estimates eps; outliers labeled −1 |
| Hierarchical | Ward linkage, user K | |

All three return a `ClusteringResult` (`labels`, `n_clusters`, `silhouette`,
`noise_fraction`), scored head-to-head on silhouette; the winner gets the 🏆 banner
and feeds the AI Insights page. PCA (3 components) powers the 3D scatter — labeled
clearly as a projection, not original feature space.

**Reference silhouettes** (K=3): K-Means 0.169 · DBSCAN 0.184 (5.5% noise) · Hierarchical 0.131

### 5.4 Data layer

`generator.py` — seeded (`random_state`) synthetic dataset: 1,500 rows, 9 numeric
features + binary `churned`. Churn probability responds to satisfaction, support
calls, tenure, spend — so models learn **real, recoverable patterns**, and the churn
rate lands near 31.4%.

`loader.py` — `load_csv()` validates uploads: missing feature columns → warning +
graceful degradation; missing `churned` → classification disabled, clustering still
works. The only `@st.cache_data` in the codebase lives here.

---

## 6. The AI Layer (`src/ai/insights.py`)

**Key resolution order:** sidebar input → `.streamlit/secrets.toml` → `GROQ_API_KEY`
env var → none (mock mode).

**Streaming contract:** both `stream_segment_insights()` and `stream_churn_insights()`
are **generators yielding string chunks**. The UI renders them with `st.write_stream()`.
No client (no key / API error) → generator yields the mock narrative instead, clearly
labeled *sample output*. The page never blocks on network failures.

**Prompt design:** cluster centroids / model metrics + top SHAP features are embedded
into a structured prompt requesting: persona name, value tier, risk factors *with
actual data values*, 3 actionable recommendations, A/B experiment ideas. The prompt
insists on grounding in the provided numbers — the LLM narrates, it doesn't invent.

**Model:** `llama-3.3-70b-versatile` via Groq (LPU inference; sub-second first tokens).

---

## 7. Frontend & Design System

**Theme:** dark glassmorphism. All tokens are CSS custom properties in `app.py`
(`--bg #050b18`, `--cyan #00d4ff`, `--purple #7c3aed`…) — change the palette in ONE
place. `.streamlit/config.toml` mirrors the base colors for native widgets.

**Reusable CSS classes** (defined in `app.py`, used everywhere):
`hero-wrap` `hero-badge` `hero-title` `hero-sub` `stat-card` `feat-card` `step-wrap`
`tech-badge` `persona-card` `winner-banner` `page-head` `section-title`

**Conventions when editing pages:**
- Page headers use `page-head` + gradient `<h1>` (cyan for churn, purple for segmentation, amber/coral for AI)
- Section titles use `section-title` (auto accent bar) — not bare `###`
- Plotly charts: always `width="stretch"` (`use_container_width` is deprecated)
- Buttons: `type="primary"` + `width="stretch"` for primary actions
- Every long operation wrapped in `st.spinner` with a human description
- A11y: `prefers-reduced-motion` disables all animations

**Session-state contract** (keys are the API between pages):
| Key | Set by | Consumed by |
|---|---|---|
| `churn_results`, `churn_X`, `churn_y` | churn page | churn page tabs |
| `metrics_df`, `shap_importance_df` | churn SHAP tab | AI insights (churn mode) |
| `seg_results`, `seg_X`, `seg_df`, `seg_selected` | segmentation page | segmentation explorer |
| `seg_profiles`, `seg_best_algo` | segmentation page | AI insights (segment mode) |

If you add a cross-page handoff, document the key here.

---

## 8. Testing Strategy (54 tests, 4 layers)

| Layer | File | What it proves |
|---|---|---|
| Data unit | `test_data.py` | schema, feature ranges, seeded reproducibility |
| ML unit | `test_models.py` | metrics in [0,1], binary preds, AUC > 0.55, cluster labels, **SHAP across 3 output layouts**, all 3 fallback branches |
| Chart unit | `test_charts.py` | every builder returns the right trace types/counts; gauge risk thresholds |
| **E2E smoke** | `test_app_smoke.py` | **real AppTest boots each page, clicks Train/Run buttons, asserts session state** — catches UI breakage unit tests can't |

```bash
pytest                                  # all 54
pytest --cov=src --cov-report=term-missing
pytest tests/test_app_smoke.py -q       # E2E only (slower)
ruff check . && ruff format --check .   # lint + format (CI-enforced)
```

Coverage gate: **70% minimum** in `pyproject.toml` — currently **92%**.

**The E2E layer is not optional.** It already caught two bugs unit tests missed:
a Plotly `update_layout` conflict that crashed the Overview page, and a deprecated
Streamlit API. If your change touches a page or chart, run the smoke tests.

**Known untested areas** (acceptable, be aware): `insights.py` Groq network paths
(mocked in tests), `charts.py` line 518 (defensive clip branch), `loader.py` upload
paths (Streamlit file objects don't serialize in pytest).

---

## 9. CI/CD & Release Flow

`.github/workflows/ci.yml` — every push/PR to main runs:
1. Matrix **Python 3.10 / 3.11 / 3.12**
2. `ruff check .` + `ruff format --check .`
3. `pytest --cov=src` (70% gate) + coverage.xml artifact (py3.11)

Release checklist:
1. `pytest` green, `ruff check .` clean, `ruff format --check .` clean
2. `streamlit run app.py` boots with zero warnings in the log
3. Manual pass: train models → check all 4 tabs → run clustering → open AI page both modes
4. Bump version in `CHANGELOG.md` (Keep-a-Changelog style)

---

## 10. Extension Guide — "How do I add …?"

**…a new classifier?** Add to `CLASSIFIER_REGISTRY` in `classifiers.py`. If it needs
preprocessing, extend `_build_pipeline`. Add a color to `MODEL_COLORS` in both
`classifiers.py` and `charts.py`. Done — training, metrics, radar, ROC, SHAP, and the
AI page pick it up automatically. Add a unit test.

**…a new feature column?** `generator.py` (add to `NUMERIC_FEATURES` + `FEATURE_METADATA`
with min/max/step/default/desc) → the what-if form and loader adapt automatically.
If SHAP-relevant, no other change needed.

**…a new chart?** Build it in `charts.py` using `_layout()`/`_axes_style()` helpers.
⚠️ `_layout()` already sets `legend` and `margin` — **merge overrides into the dict
instead of passing them twice to `update_layout()`** (Plotly raises `TypeError`).
Then add a `test_charts.py` test.

**…a new page?** Create `pages/your_page.py`, register in `st.navigation` in `app.py`,
use the `page-head` header pattern, read/write documented session-state keys only.

**…a real dataset?** Skip the generator; your CSV needs the 9 numeric features
(+`churned` for classification). See `loader.load_csv()` for tolerance rules.
Keep the generator as the demo fallback.

**…a different LLM?** Only `insights.py` touches Groq — swap `_get_client()` and keep
the generator-yield contract. The mock fallback makes this a zero-risk change.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Per-column arrays must each be 1-dimensional` on SHAP | SHAP returned 3D per-class array (old code path) | Fixed in `_select_positive_class`; update if you vendored old code |
| `update_layout() got multiple values for 'legend'/'margin'` | passing keys that `_layout()` already sets | merge into the layout dict (see §10) |
| Plotly chart not filling container | deprecated `use_container_width` | use `width="stretch"` |
| `AttributeError: ... no attribute 'download_button'` in AppTest | not all widgets exposed in this Streamlit version | assert via `at.session_state` instead |
| Models retrain on every interaction | results not in `st.session_state` | check the training branch stores results (§7 table) |
| AI page shows "sample output" unexpectedly | no API key resolved, or Groq error | check sidebar → env var → secrets.toml order (§6) |
| `enableCORS` warning in boot log | config conflicts with XSRF protection | removed in v2.1; don't re-add |
| DBSCAN finds 1 cluster / all noise | eps heuristic mismatch for your feature scale | standardize features or tune in `clustering.py` |
| Port 8501 busy | previous server alive | `taskkill //F //PID <pid>` (Windows bash) or change `--server.port` |

---

## 12. Glossary

- **AUC-ROC** — probability a random churner scores higher than a random retained customer; threshold-free
- **SHAP** — Shapley-value attribution splitting a prediction among features fairly
- **Silhouette** — cluster cohesion vs separation, [-1, 1]; > 0.25 usable, > 0.5 strong
- **Elbow method** — pick K where inertia's marginal drop flattens
- **eps (DBSCAN)** — neighborhood radius; controls cluster granularity and noise
- **Stratified split** — train/test split preserving class ratios; mandatory at ~31% churn
- **Session state** — Streamlit's per-user server-side dict; the app's cross-page memory
- **Mock mode** — deterministic offline AI fallback so the demo never breaks

---

*Owned by the NeuroScope maintainers. Update this playbook whenever architecture,
session-state keys, or test strategy change — it's the source of truth.*
