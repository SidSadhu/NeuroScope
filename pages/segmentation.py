"""NeuroScope — Customer Segmentation page."""

from __future__ import annotations

import streamlit as st

from src.data.loader import (
    CLUSTERING_DEFAULT_FEATURES,
    get_features_and_target,
    load_csv,
    load_default_dataset,
)
from src.models.clustering import (
    compute_elbow_data,
    get_cluster_profiles,
    reduce_to_3d,
    run_clustering,
)
from src.visualization.charts import (
    plot_algorithm_comparison,
    plot_cluster_2d,
    plot_cluster_3d,
    plot_cluster_profiles,
    plot_elbow_silhouette,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-head">
    <div class="hero-badge">🧩 Unsupervised Learning</div>
    <h1 style="background:linear-gradient(135deg,#fff,#7c3aed);">Customer Segmentation</h1>
    <p>
        K-Means · DBSCAN · Hierarchical — automatic comparison, 3D visualisation,
        persona cards, and AI-ready cluster profiles.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Data Source")
    data_src = st.radio(
        "Source", ["📊 Synthetic Dataset", "📁 Upload CSV"], label_visibility="collapsed"
    )
    uploaded = None
    if "Upload" in data_src:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.markdown("### 🎛️ Clustering")
    n_clusters = st.slider("Target clusters (K)", 2, 8, 3)
    st.caption("K is used by K-Means and Hierarchical. DBSCAN auto-detects its own clusters.")

# ── Load data ─────────────────────────────────────────────────────────────────
if "Upload" in data_src:
    if not uploaded:
        st.info("👆 Upload a CSV, or switch to the built-in synthetic dataset.")
        st.stop()
    df, warnings = load_csv(uploaded)
    for w in warnings:
        st.warning(w)
else:
    with st.spinner("Loading dataset…"):
        df = load_default_dataset()

X_all, _ = get_features_and_target(df)
available = list(X_all.columns)

# ── Feature selection ─────────────────────────────────────────────────────────
defaults = [f for f in CLUSTERING_DEFAULT_FEATURES if f in available]
selected = st.multiselect(
    "Select features for clustering",
    options=available,
    default=defaults if defaults else available[:5],
    help="Choose 2–9 numeric features. More features = richer segments but slower DBSCAN.",
)

if len(selected) < 2:
    st.warning("Select at least 2 features to continue.")
    st.stop()

X = X_all[selected]

# ── Elbow analysis ────────────────────────────────────────────────────────────
st.markdown("#### 📉 Elbow & Silhouette Analysis")
st.caption("Use this to select the best K before running clustering.")
with st.spinner("Computing elbow curve…"):
    elbow_df = compute_elbow_data(X, max_k=8)
st.plotly_chart(plot_elbow_silhouette(elbow_df), width="stretch")

st.divider()

# ── Run clustering ────────────────────────────────────────────────────────────
col_btn, col_hint = st.columns([1, 3])
with col_btn:
    run_clicked = st.button("🧩 Run All Algorithms", type="primary", width="stretch")
with col_hint:
    if "seg_results" not in st.session_state:
        st.info(
            "Click **Run All Algorithms** to compare K-Means, DBSCAN, and Hierarchical clustering."
        )

if run_clicked:
    with st.spinner("Running 3 clustering algorithms…"):
        results = run_clustering(X, n_clusters=n_clusters)
    st.session_state["seg_results"] = results
    st.session_state["seg_X"] = X
    st.session_state["seg_df"] = df
    st.session_state["seg_selected"] = selected
    st.success("✅ Clustering complete!")

if "seg_results" not in st.session_state:
    st.stop()

results = st.session_state["seg_results"]
X_seg = st.session_state["seg_X"]
df_seg = st.session_state["seg_df"]
feat_sel = st.session_state["seg_selected"]

# ── Algorithm comparison ──────────────────────────────────────────────────────
st.markdown("#### 🏆 Algorithm Comparison")
c1, c2, c3 = st.columns(3)
for col, (algo, res) in zip([c1, c2, c3], results.items()):
    noise_str = f" · {res.noise_fraction:.0%} noise" if algo == "DBSCAN" else ""
    col.metric(
        algo,
        f"Silhouette: {res.silhouette:.3f}",
        f"{res.n_clusters} clusters{noise_str}",
        delta_color="normal",
    )
st.plotly_chart(plot_algorithm_comparison(results), width="stretch")

# Pick best algorithm
best_algo = max(results, key=lambda k: results[k].silhouette)
best_labels = results[best_algo].labels

st.markdown(
    f"""
<div class="winner-banner">
    <span style="font-size:2rem;">🏆</span>
    <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;color:#10b981;font-size:1.05rem;">
            Best Algorithm: {best_algo}
        </div>
        <div style="color:#64748b;font-size:.84rem;">
            Silhouette: {results[best_algo].silhouette:.4f} &nbsp;·&nbsp;
            {results[best_algo].n_clusters} clusters &nbsp;·&nbsp;
            Noise: {results[best_algo].noise_fraction:.1%}
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Store best result for AI Insights (always the top-silhouette algorithm)
profiles_df = get_cluster_profiles(df_seg, best_labels, feat_sel)
st.session_state["seg_profiles"] = profiles_df
st.session_state["seg_best_algo"] = best_algo

st.divider()

# ── Algorithm explorer ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔬 Explore an Algorithm</div>', unsafe_allow_html=True)
st.caption("Compare any algorithm's clusters below — the winner is pre-selected for you.")
viz_algo = st.radio(
    "Algorithm",
    list(results.keys()),
    index=list(results.keys()).index(best_algo),
    horizontal=True,
    label_visibility="collapsed",
)
sel_res = results[viz_algo]
sel_labels = sel_res.labels
sel_profiles = get_cluster_profiles(df_seg, sel_labels, feat_sel)

# ── Visualisation tabs ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Cluster Visualisation</div>', unsafe_allow_html=True)
vtab_3d, vtab_2d, vtab_profile = st.tabs(["🌐 3D Scatter", "🔵 2D Scatter", "📡 Feature Profiles"])

with vtab_3d:
    X_3d = reduce_to_3d(X_seg)
    st.plotly_chart(
        plot_cluster_3d(X_3d, sel_labels, f"{viz_algo} — 3D PCA Projection"),
        width="stretch",
    )
    st.caption(
        "3D projection via PCA — axes represent principal components, not original features."
    )

with vtab_2d:
    ax1_opt, ax2_opt = st.columns(2)
    with ax1_opt:
        ax1 = st.selectbox("X-axis feature", feat_sel, index=0)
    with ax2_opt:
        ax2 = st.selectbox("Y-axis feature", feat_sel, index=min(1, len(feat_sel) - 1))
    if ax1 != ax2:
        st.plotly_chart(
            plot_cluster_2d(X_seg[ax1].values, X_seg[ax2].values, sel_labels, ax1, ax2),
            width="stretch",
        )
    else:
        st.warning("Select two different features.")

with vtab_profile:
    if not sel_profiles.empty:
        st.plotly_chart(plot_cluster_profiles(sel_profiles), width="stretch")
        st.caption(
            "Normalised mean feature values per cluster — shows relative strengths of each segment."
        )

st.divider()

# ── Persona cards ─────────────────────────────────────────────────────────────
PERSONA_EMOJIS = ["💎", "⚠️", "🌱", "🚀", "🎯", "🔥", "🌟"]

st.markdown('<div class="section-title">👥 Customer Personas</div>', unsafe_allow_html=True)
st.caption(f"Personas for the currently selected algorithm: **{viz_algo}**")
if not sel_profiles.empty:
    unique_clusters = sorted([c for c in sel_profiles.index if c >= 0])
    n_cols = min(len(unique_clusters), 4)
    pers_cols = st.columns(n_cols)
    for i, cluster_id in enumerate(unique_clusters):
        mask = sel_labels == cluster_id
        count = int(mask.sum())
        pct = count / len(sel_labels) * 100
        with pers_cols[i % n_cols]:
            st.markdown(
                f"""
            <div class="persona-card">
                <span class="persona-emoji">{PERSONA_EMOJIS[i % len(PERSONA_EMOJIS)]}</span>
                <div class="persona-name">Segment {cluster_id}</div>
                <div class="persona-count">{count:,} customers ({pct:.1f}%)</div>
            </div>""",
                unsafe_allow_html=True,
            )

st.divider()

# ── Cluster summary table ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Cluster Summary</div>', unsafe_allow_html=True)
summary = sel_profiles.copy()
counts = {c: int((sel_labels == c).sum()) for c in summary.index}
summary.insert(0, "Count", [counts[c] for c in summary.index])
st.dataframe(summary, width="stretch")

# ── Download ──────────────────────────────────────────────────────────────────
result_df = df_seg.copy()
result_df["cluster"] = sel_labels
result_df["algorithm"] = viz_algo

col_dl, col_hint2 = st.columns([1, 2])
with col_dl:
    st.download_button(
        "⬇️ Download Clustered CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name=f"neuroscope_clustered_{viz_algo.lower().replace('-', '_')}.csv",
        mime="text/csv",
        width="stretch",
    )
with col_hint2:
    st.caption(f"Exports the {len(result_df):,} rows with **{viz_algo}** cluster assignments.")
