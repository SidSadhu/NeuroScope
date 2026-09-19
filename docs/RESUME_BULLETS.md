# 📄 NeuroScope Resume Bullets — 3 Formats

> The same project story in the three formats recruiters and hiring managers see most:
> **Google XYZ**, **STAR**, and **CAR**.
> Every number below is real and reproducible from the repo — nothing is padded.
> **How to use:** pick ONE format for the resume bullet list (XYZ is best for that);
> keep STAR/CAR versions as your *spoken* interview answers.

---

## ✅ Format 1 — Google XYZ (recommended for the resume itself)

> Formula: **Accomplished [X] as measured by [Y], by doing [Z].**

**Full version (pick 2–3 bullets max on a resume):**

- Built **NeuroScope**, an end-to-end customer-intelligence platform (Python, Streamlit,
  scikit-learn) that trains **4 competing classifiers and 3 clustering algorithms** on
  customer data, reaching **0.80 AUC-ROC** churn prediction with SHAP explainability,
  cutting analyst reporting time from hours to **under 60 seconds** of interactive analysis.

- Engineered a **version-tolerant SHAP integration** supporting 3 explainer output
  formats across shap/sklearn versions with graceful fallbacks, and raised automated
  coverage from **55% to 92%** (54 tests) including **AppTest end-to-end UI tests** —
  catching 2 integration bugs unit tests missed.

- Designed a **streaming LLM insight layer** (Groq LLaMA-3.3-70B) that converts cluster
  centroids and SHAP attributions into grounded executive playbooks, with an offline
  mock fallback so the product **degrades gracefully with zero API dependency**.

**Short version (if space is tight — one bullet):**

- Built an AI customer-intelligence platform (Python, scikit-learn, SHAP, Groq LLM,
  Streamlit) with **92% test coverage and CI/CD**, delivering 0.80 AUC-ROC churn
  prediction, 3-algorithm segmentation, and LLM-generated retention playbooks.

---

## ⭐ Format 2 — STAR (interview answer format)

> **S**ituation · **T**ask · **A**ction · **R**esult — use for behavioral answers to
> "tell me about a project you built."

- **Situation:** Customer Success and marketing teams drown in raw customer telemetry —
  churn risk lives in spreadsheets, segmentation is guesswork, and ML outputs are
  unreadable to non-technical stakeholders.

- **Task:** Build a self-serve platform that turns a raw customer CSV into churn
  predictions, defensible segments, and executive-ready strategy — with explainability
  a business stakeholder can actually read.

- **Action:**
  - Architected a modular Python core (`src/`) decoupled from a Streamlit UI, so every
    model, chart, and AI call is independently testable.
  - Implemented a 4-model classification arena (Logistic Regression, Random Forest,
    Gradient Boosting, Extra Trees) with stratified evaluation and AUC-based selection.
  - Solved SHAP version fragmentation by writing a normalizer handling 3 explainer
    output layouts, with `feature_importances_`/`coef_` fallbacks — the UI never crashes.
  - Wired Groq LLaMA-3.3-70B as a *streaming generator* that narrates actual cluster
    centroids and SHAP values into retention playbooks, with a labeled offline mock mode.
  - Built a 4-layer test suite (data unit → ML unit → chart unit → AppTest E2E that
    literally clicks the buttons) enforced by GitHub Actions across Python 3.10–3.12.

- **Result:** 0.80 AUC-ROC churn model with per-prediction SHAP waterfalls; 3-algorithm
  segmentation with automatic winner selection; executive insights streamed in seconds;
  **92% test coverage**, zero-warning production boot, and two documented catches where
  the E2E layer saved the UI from regressions.

---

## 🎯 Format 3 — CAR (compact alternative)

> **C**hallenge · **A**ction · **R**esult — tighter than STAR; good for LinkedIn's
> About section or a project one-liner.

- **Challenge:** Make ML churn analysis explainable and self-serve for non-technical
  teams without sacrificing engineering rigor.
- **Action:** Built a full-stack ML app: 4-model classifier arena, 3-algorithm
  clustering, version-tolerant SHAP explainability, and a Groq LLM narrative layer —
  all under CI with a 4-layer test suite.
- **Result:** **0.80 AUC-ROC** churn prediction, persona-driven segmentation, executive
  playbooks on demand, and **92% coverage** proven by tests that exercise the real UI.

---

## 🧭 Which format, where?

| Surface | Format | Why |
|---|---|---|
| Resume bullet list | **Google XYZ** | metrics-first; recruiters scan for X and Y |
| Behavioral interview answer | **STAR** | natural spoken arc; shows decision-making |
| LinkedIn About / project blurb | **CAR** | compact; fits 2–3 lines |
| ML engineer interviews | XYZ bullet + STAR deep-dive on §SHAP and §testing | those two show real engineering judgment |

## 📝 Tailoring tips

- **For Data Science roles:** lead with the AUC-ROC, model comparison, and SHAP bullets;
  mention the class imbalance (~31% churn) and why AUC was the selection metric.
- **For Software/ML Engineering roles:** lead with the architecture (UI/core decoupling),
  the version-tolerance layer, CI/CD, and the E2E testing story.
- **For Product/Data Analyst roles:** lead with the business outcome (minutes not hours),
  personas/playbooks, and the interactive what-if simulator.
- Numbers you can defend in interview: 0.8045 AUC (Logistic Regression, seed 42),
  92.3% coverage, 54 tests, 4 classifiers, 3 clustering algorithms, 9 features,
  1,500-row seeded dataset. All reproducible: `pytest` + one cell of `train_all_models`.
- If asked "what failed?": the SHAP 3D-array bug (caught by tests), the Plotly
  `update_layout` conflict (caught by the new E2E layer), and Streamlit's deprecated
  `use_container_width` — good honest war stories that show debugging depth.
