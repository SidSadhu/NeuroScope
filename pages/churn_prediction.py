"""NeuroScope — Churn Prediction page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.loader import (
    FEATURE_METADATA,
    NUMERIC_FEATURES,
    get_features_and_target,
    load_csv,
    load_default_dataset,
)
from src.models.classifiers import (
    get_best_model,
    get_classification_report,
    metrics_to_dataframe,
    train_all_models,
)
from src.models.explainability import (
    compute_shap_feature_importance,
    compute_shap_waterfall_values,
)
from src.visualization.charts import (
    plot_confusion_matrix,
    plot_model_radar,
    plot_risk_gauge,
    plot_roc_curves,
    plot_shap_importance,
    plot_shap_waterfall,
)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-head">
    <div class="hero-badge">📈 Supervised Learning</div>
    <h1 style="background:linear-gradient(135deg,#fff,#00d4ff);">Churn Prediction</h1>
    <p>
        Train 4 classifiers simultaneously · Compare with SHAP explainability ·
        Predict churn probability for any customer profile.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Data Source")
    data_source = st.radio(
        "Source",
        ["📊 Synthetic Dataset (1,500 rows)", "📁 Upload CSV"],
        label_visibility="collapsed",
    )
    uploaded = None
    if "Upload" in data_source:
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    st.divider()
    st.markdown("### 🧪 Classifiers")
    st.markdown(
        """
    <div style="font-size:.8rem;color:#64748b;line-height:2;">
        ✦ Logistic Regression<br>
        ✦ Random Forest<br>
        ✦ Gradient Boosting<br>
        ✦ Extra Trees
    </div>""",
        unsafe_allow_html=True,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
if "Upload" in data_source:
    if not uploaded:
        st.info("👆 Upload a CSV, or switch to the built-in synthetic dataset.")
        st.stop()
    df, warnings = load_csv(uploaded)
    for w in warnings:
        st.warning(w)
else:
    with st.spinner("Loading dataset…"):
        df = load_default_dataset()

X, y = get_features_and_target(df)
if y is None:
    st.error(
        "No 'churned' column found. Cannot train classifiers — use the Segmentation page instead."
    )
    st.stop()

# ── Dataset overview ──────────────────────────────────────────────────────────
st.markdown("#### 📋 Dataset Overview")
total, churned_n = len(df), int(y.sum())
churn_rate = churned_n / total
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers", f"{total:,}")
c2.metric("Churned", f"{churned_n:,}", delta=f"{churn_rate:.1%} rate", delta_color="inverse")
c3.metric("Retained", f"{total - churned_n:,}")
c4.metric("Features", len(X.columns))
c5.metric("Class Split", f"{churn_rate:.0%} / {1 - churn_rate:.0%}")

with st.expander("👁️ Preview data", expanded=False):
    st.dataframe(df.head(20), width="stretch")

st.divider()

# ── Train button ──────────────────────────────────────────────────────────────
col_btn, col_hint = st.columns([1, 3])
with col_btn:
    train_clicked = st.button("🚀 Train All Models", type="primary", width="stretch")
with col_hint:
    if "churn_results" not in st.session_state:
        st.info(
            "Click **Train All Models** to train all 4 classifiers on an 80/20 stratified split."
        )

if train_clicked:
    with st.spinner("Training 4 models… (~15 seconds)"):
        results = train_all_models(X, y)
    st.session_state["churn_results"] = results
    st.session_state["churn_X"] = X
    st.session_state["churn_y"] = y
    st.success("✅ All models trained! Explore the tabs below.")

if "churn_results" not in st.session_state:
    st.stop()

results = st.session_state["churn_results"]
X_ref = st.session_state["churn_X"]
best = get_best_model(results)
metrics_df = metrics_to_dataframe(results)

# ── Best model banner ─────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="winner-banner">
    <span style="font-size:2rem;">🏆</span>
    <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;color:#10b981;font-size:1.05rem;">
            Best Model: {best.name}
        </div>
        <div style="color:#64748b;font-size:.84rem;">
            AUC-ROC: {best.metrics["AUC-ROC"]:.4f} &nbsp;·&nbsp;
            F1: {best.metrics["F1"]:.4f} &nbsp;·&nbsp;
            Accuracy: {best.metrics["Accuracy"]:.4f} &nbsp;·&nbsp;
            Recall: {best.metrics["Recall"]:.4f}
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Results tabs ──────────────────────────────────────────────────────────────
tab_cmp, tab_eval, tab_shap, tab_pred = st.tabs(
    [
        "📊 Model Comparison",
        "📈 Evaluation",
        "🔍 SHAP Explainability",
        "🔮 Single Prediction",
    ]
)

# ── Tab 1: Comparison ─────────────────────────────────────────────────────────
with tab_cmp:
    st.markdown("#### Leaderboard — ranked by AUC-ROC")
    st.dataframe(
        metrics_df.style.format("{:.4f}")
        .highlight_max(axis=0, color="#0f2d1e")
        .set_properties(**{"text-align": "center"}),
        width="stretch",
    )
    st.plotly_chart(plot_model_radar(metrics_df), width="stretch")

# ── Tab 2: Evaluation ────────────────────────────────────────────────────────
with tab_eval:
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.plotly_chart(plot_confusion_matrix(best.y_test, best.y_pred, best.name), width="stretch")
    with ecol2:
        st.plotly_chart(plot_roc_curves(results), width="stretch")

    st.markdown(f"#### Classification Report — {best.name}")
    st.dataframe(get_classification_report(best), width="stretch")

# ── Tab 3: SHAP ──────────────────────────────────────────────────────────────
with tab_shap:
    st.markdown(f"#### Feature Importance (Mean |SHAP|) — {best.name}")
    st.caption(
        "SHAP (SHapley Additive exPlanations) quantifies each feature's average contribution to "
        "predictions. Higher bar = more influential feature."
    )
    key = f"shap_imp_{best.name}"
    if key not in st.session_state:
        with st.spinner("Computing SHAP values… (first run only)"):
            imp_df = compute_shap_feature_importance(best.pipeline, best.X_test, best.name)
            st.session_state[key] = imp_df
    st.plotly_chart(plot_shap_importance(st.session_state[key]), width="stretch")
    st.info(
        "💡 **Interpretation**: 🟥 Coral bars increase churn probability · 🟩 Green bars decrease it. "
        "Values represent the *mean absolute* SHAP impact across the test set."
    )
    # Store for AI Insights page
    st.session_state["shap_importance_df"] = st.session_state[key]
    st.session_state["metrics_df"] = metrics_df

# ── Tab 4: Single Prediction ─────────────────────────────────────────────────
with tab_pred:
    st.markdown("#### 🔮 Single Customer Churn Risk")
    st.caption(f"Model: **{best.name}** · AUC-ROC: {best.metrics['AUC-ROC']:.4f}")

    feature_cols = [f for f in NUMERIC_FEATURES if f in X_ref.columns]
    with st.form("churn_form"):
        st.markdown("Enter customer attributes:")
        cols = st.columns(3)
        values: dict[str, float] = {}
        for i, feat in enumerate(feature_cols):
            meta = FEATURE_METADATA[feat]
            with cols[i % 3]:
                values[feat] = st.number_input(
                    feat.replace("_", " ").title(),
                    min_value=float(meta["min"]),
                    max_value=float(meta["max"]),
                    value=float(meta["default"]),
                    step=float(meta["step"]),
                    help=meta["desc"],
                    key=f"inp_{feat}",
                )
        submitted = st.form_submit_button("⚡ Assess Churn Risk", type="primary")

    if submitted:
        row_df = pd.DataFrame([values])[feature_cols]
        prob = float(best.pipeline.predict_proba(row_df)[0, 1])
        st.session_state["last_pred_prob"] = prob
        st.session_state["last_pred_row"] = values

    if "last_pred_prob" in st.session_state:
        prob = st.session_state["last_pred_prob"]

        risk_label = (
            "🔴 HIGH RISK" if prob > 0.65 else ("🟡 MEDIUM RISK" if prob > 0.35 else "🟢 LOW RISK")
        )
        color = "#ff6b6b" if prob > 0.65 else ("#f59e0b" if prob > 0.35 else "#10b981")

        gc1, gc2 = st.columns([3, 2])
        with gc1:
            st.markdown(
                f"""
            <div style="background:rgba(13,22,38,0.85);border:1px solid {color}3a;
                        border-radius:18px;padding:2.2rem;text-align:center;margin:1rem 0;">
                <div style="font-size:.8rem;color:#64748b;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;">
                    Churn Risk Assessment
                </div>
                <div style="font-size:1.4rem;font-weight:700;color:{color};margin-bottom:.3rem;">
                    {risk_label}
                </div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:3.4rem;font-weight:800;color:{color};">
                    {prob:.1%}
                </div>
                <div style="color:#64748b;font-size:.88rem;margin-top:.3rem;">
                    Estimated churn probability · Model: {best.name}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with gc2:
            st.plotly_chart(plot_risk_gauge(prob), width="stretch")

        wf_key = "last_pred_row" in st.session_state
        if wf_key:
            row_df = pd.DataFrame([st.session_state["last_pred_row"]])[feature_cols]
            with st.spinner("Computing prediction breakdown…"):
                wf = compute_shap_waterfall_values(best.pipeline, row_df, best.name)
            if wf:
                st.plotly_chart(plot_shap_waterfall(wf, prob), width="stretch")
            else:
                st.caption("SHAP waterfall not available for this model.")
