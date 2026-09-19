"""NeuroScope — Overview / Landing page."""

import streamlit as st

st.markdown(
    """
<div class="hero-wrap">
    <div class="hero-badge">⚡ AI-Powered Customer Intelligence Platform</div>
    <h1 class="hero-title">NeuroScope</h1>
    <p class="hero-sub">
        Transform raw customer data into strategic intelligence.
        Multi-model ML analysis, 3-algorithm clustering, SHAP explainability,
        and LLM-powered business insights — all in one platform.
    </p>
    <div style="display:flex;gap:.6rem;flex-wrap:wrap;">
        <span class="tech-badge">🐍 Python 3.12</span>
        <span class="tech-badge">🔴 Streamlit</span>
        <span class="tech-badge">🤖 scikit-learn</span>
        <span class="tech-badge">⚡ Groq LLaMA-3.3</span>
        <span class="tech-badge">📊 Plotly</span>
        <span class="tech-badge">🔍 SHAP</span>
        <span class="tech-badge">🐳 Docker</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Stats row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        """
    <div class="stat-card">
        <span class="stat-num">4</span>
        <div class="stat-lbl">ML Classifiers</div>
    </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
    <div class="stat-card">
        <span class="stat-num">3</span>
        <div class="stat-lbl">Clustering Algorithms</div>
    </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
    <div class="stat-card">
        <span class="stat-num">LLM</span>
        <div class="stat-lbl">Groq AI Insights</div>
    </div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        """
    <div class="stat-card">
        <span class="stat-num">SHAP</span>
        <div class="stat-lbl">Explainability</div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🚀 Platform Capabilities</div>', unsafe_allow_html=True)
fc1, fc2, fc3, fc4 = st.columns(4)

cards = [
    (
        "🔮",
        "Churn Prediction",
        "Train 4 classifiers simultaneously. Compare Accuracy, Precision, Recall, F1 & AUC-ROC. "
        "Get SHAP explanations for every single prediction.",
    ),
    (
        "🧩",
        "Customer Segmentation",
        "K-Means, DBSCAN, and Hierarchical clustering with automatic algorithm comparison, "
        "3D scatter visualisation, and cluster persona cards.",
    ),
    (
        "🤖",
        "AI Insights (Groq)",
        "LLaMA-3.3-70B generates business personas, churn risk narratives, "
        "retention strategies, and A/B experiment ideas — streamed live.",
    ),
    (
        "📥",
        "Bring Your Data",
        "Upload your own CSV or explore our 1,500-row synthetic customer dataset. "
        "Schema auto-detection with graceful degradation.",
    ),
]
for col, (icon, title, desc) in zip([fc1, fc2, fc3, fc4], cards):
    with col:
        st.markdown(
            f"""
        <div class="feat-card">
            <div class="feat-icon">{icon}</div>
            <div class="feat-title">{title}</div>
            <div class="feat-desc">{desc}</div>
        </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 How It Works</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
steps = [
    (
        "1",
        "Load Data",
        "Use the built-in 1,500-row synthetic customer dataset or upload your own CSV.",
    ),
    (
        "2",
        "Train & Analyse",
        "Click one button to train all 4 models. Compare metrics. Drill into SHAP explanations.",
    ),
    (
        "3",
        "Get AI Insights",
        "Groq streams live LLM-generated business personas and retention strategies per segment.",
    ),
]
for col, (num, title, desc) in zip([s1, s2, s3], steps):
    with col:
        st.markdown(
            f"""
        <div class="step-wrap">
            <div class="step-num">{num}</div>
            <div>
                <strong style="color:#e2e8f0;font-family:'Space Grotesk',sans-serif;">{title}</strong><br>
                <span style="color:#64748b;font-size:.83rem;">{desc}</span>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

st.divider()

# ── Dataset snapshot ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Dataset Snapshot</div>', unsafe_allow_html=True)
st.caption("Built-in synthetic customer dataset — 1,500 rows, 9 features, realistic churn patterns")

from src.data.loader import load_default_dataset  # noqa: E402
from src.visualization.charts import plot_churn_donut, plot_correlation_heatmap  # noqa: E402

with st.spinner("Loading dataset…"):
    df = load_default_dataset()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Features", "9")
c3.metric("Churn Rate", f"{df['churned'].mean():.1%}")
c4.metric("Premium Users", f"{df['is_premium'].mean():.1%}")

# ── Live dataset charts ───────────────────────────────────────────────────────
dc1, dc2 = st.columns([2, 3])
with dc1:
    st.plotly_chart(plot_churn_donut(df["churned"]), width="stretch")
with dc2:
    st.plotly_chart(plot_correlation_heatmap(df), width="stretch")

with st.expander("👁️ Preview the dataset", expanded=False):
    st.dataframe(df.head(12), width="stretch")

st.markdown(
    """
<br>
<div style="text-align:center;color:#475569;font-size:.8rem;">
    Built by <strong style="color:#00d4ff;">Siddharth Sadhu</strong> &nbsp;·&nbsp;
    <a href="https://github.com/SidSadhu" style="color:#00d4ff;text-decoration:none;">GitHub</a>
    &nbsp;·&nbsp; MIT License
</div>
""",
    unsafe_allow_html=True,
)
