# 🧠 NeuroScope Model Card

> Transparent, reproducible documentation of every model in the platform.
> **Audience:** senior ML/data-science engineers evaluating this work.
> Format inspired by Google's *Model Cards for Model Reporting* (Mitchell et al., 2019).

---

## 1. Model Overview

| | Churn Classification | Customer Segmentation |
|---|---|---|
| **Task** | Binary classification (`churned`: 0/1) | Unsupervised clustering |
| **Models** | Logistic Regression, Random Forest, Gradient Boosting, Extra Trees | K-Means, DBSCAN, Agglomerative (Ward) |
| **Selection rule** | Highest AUC-ROC on a stratified holdout | Highest silhouette score |
| **Explainability** | SHAP (global ranking + per-prediction waterfall) | Normalized radar profiles per cluster |
| **Owner** | Project maintainer | Project maintainer |
| **Version** | v2.1 (Sept 2026) | v2.1 (Sept 2026) |

---

## 2. Intended Use

- ✅ **In scope:** interactive analysis of customer telemetry; identifying at-risk
  customers; segmenting customers for targeted campaigns; demonstrating production-pattern
  ML engineering (pipelines, explainability, testing).
- ❌ **Out of scope:** automated consequential decisions about individuals (credit,
  employment, healthcare, or any regulated decisioning); real-time scoring; deployment
  without human review. Outputs are *decision support*, not decisions.

---

## 3. Training & Evaluation Data

| Property | Value |
|---|---|
| Source | Synthetic generator (`src/data/generator.py`), seeded & reproducible |
| Rows | 1,500 customers (by default) |
| Features | 9 numeric: `monthly_spend`, `satisfaction_score`, `tenure_months`, `login_frequency`, `support_calls_3m`, `last_purchase_days`, `avg_order_value`, `discount_usage_rate`, `is_premium` |
| Target | `churned` ∈ {0, 1}, base rate ≈ **31.4%** |
| Splits | 80/20 **stratified** split, `random_state=42` (identical for all 4 classifiers) |
| Preprocessing | `StandardScaler` inside the Logistic Regression pipeline only (tree models don't need it) — fits on train folds only, so no leakage |

**Known data caveats (read before trusting metrics):**
- Synthetic data encodes *additive, mostly linear* signal — which is why Logistic
  Regression wins here. This is a property of the demo data, not a claim about real markets.
- Generator ranges are realistic but bounded; the what-if simulator rejects inputs
  outside documented feature ranges.
- Swap in a real CSV anytime (schema in `loader.py`); metrics will differ and should be
  re-documented here.

---

## 4. Metrics & Performance

**Evaluation protocol:** held-out 20% stratified split; identical split across all
models for fair comparison. Metrics: Accuracy, Precision, Recall, F1, AUC-ROC.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.7567 | 0.6842 | 0.4149 | 0.5166 | **0.8045** |
| Random Forest | 0.7533 | 0.6724 | 0.4149 | 0.5132 | 0.7926 |
| Gradient Boosting | 0.7533 | 0.6562 | 0.4468 | 0.5316 | 0.7871 |
| Extra Trees | 0.7300 | 0.6327 | 0.3298 | 0.4336 | 0.7758 |

**Segmentation** (K=3, features: spend, satisfaction, tenure, login frequency, support calls):

| Algorithm | Silhouette | Clusters | Noise |
|---|---|---|---|
| DBSCAN | 0.1838 | 2 (+5.5% noise) | outliers labeled −1 |
| K-Means | 0.1689 | 3 | — |
| Hierarchical | 0.1309 | 3 | — |

**Reading the numbers honestly:**
- Recall ~0.42 at the default 0.5 threshold means the model misses more than half of
  churners — typical at this base rate without threshold tuning. For retention
  campaigns, lower the decision threshold (the UI scores probability, not hard labels).
- AUC (0.80) is the headline metric because it's threshold-free; the class split is
  ~31/69, so raw accuracy alone would be misleading.
- Silhouettes ~0.13–0.18 are *weak-to-moderate*: the synthetic feature space overlaps —
  realistic for behavioral data, and a good teaching point the radar charts make visible.
- All figures reproducible with seed 42: `train_all_models(X, y)` / `run_clustering(X, 3)`.

**Top SHAP drivers of churn** (global, best model):
`support_calls_3m` → `login_frequency` → `satisfaction_score` → `tenure_months` → `last_purchase_days`

---

## 5. Quantitative & Qualitative Analyses

- **Fairness across subgroups:** not computable on synthetic data (no protected
  attributes are generated). On real data, evaluate metrics per segment before deployment.
- **Calibration:** predicted probabilities are not calibrated (no isotonic/Platt step);
  treat the gauge as a ranking signal, not a literal probability.
- **Human-in-the-loop check:** every single prediction shows its SHAP waterfall so an
  analyst can sanity-check *why* before acting.

---

## 6. Limitations

1. **Synthetic training data** — patterns reflect the generator's assumptions, not real
   market behavior. Do not quote metrics as real-world performance.
2. **Recall-limited at default threshold** — raise sensitivity before operational use.
3. **No drift monitoring** — a batch scoring app, not a monitored deployment.
4. **No hyperparameter search** — sensible fixed defaults; the architecture (registry +
   pipelines) is built so a search can slot in without UI changes.
5. **PCA projection ≠ feature space** — 3D scatter axes are principal components,
   labeled as such in the UI.
6. **DBSCAN heuristic** — k-NN-curve eps estimation can under- or over-segment on
   differently scaled real features; standardize before importing.

---

## 7. Ethical Considerations

- Churn models optimize retention spend; on real data, audit for disparate impact
  across customer demographics before targeting.
- LLM-generated insights can hallucinate; the prompt grounds outputs in actual
  centroids/SHAP values, and the UI labels AI output as AI output with a review reminder.
- No PII is required; the demo dataset is fully synthetic. Uploaded CSVs never leave
  the app process (except optional Groq calls, which receive only aggregate profiles).
- Explainability is a first-class feature precisely so predictions are auditable.

---

## 8. Reproducibility

```python
from src.data.loader import load_default_dataset, get_features_and_target
from src.models.classifiers import train_all_models

df = load_default_dataset()  # seed 42 baked in
X, y = get_features_and_target(df)
results = train_all_models(X, y)  # → table above, byte-identical
```

Environment: Python 3.10–3.12 · scikit-learn 1.7.2 · shap 0.49.1 · pandas 2.3.3
(pinned ranges in `requirements.txt`; CI verifies the matrix).
