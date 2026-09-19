"""NeuroScope — AI Insights page (powered by Groq LLaMA-3.3)."""

from __future__ import annotations

import streamlit as st

from src.ai.insights import resolve_api_key, stream_churn_insights, stream_segment_insights

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="page-head">
    <div class="hero-badge">🤖 Powered by Groq · LLaMA-3.3-70B-Versatile</div>
    <h1 style="background:linear-gradient(135deg,#fff,#f59e0b,#ff6b6b);">AI Insights</h1>
    <p>
        Groq streams live LLM-generated business narratives from your ML results —
        customer personas, churn drivers, retention strategies, and A/B experiment ideas.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── API key ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Groq API Key")
    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="gsk_...",
        help=(
            "Get a free key at console.groq.com. "
            "Alternatively set GROQ_API_KEY in .streamlit/secrets.toml or as an env variable."
        ),
        label_visibility="collapsed",
    )
    resolved_key = resolve_api_key(api_key_input or None)
    if resolved_key:
        st.success("✅ API key detected")
    else:
        st.info(
            "No API key found.\n\n"
            "App will show **sample output** that demonstrates exactly what live output looks like."
        )
    st.divider()
    st.markdown("### 📖 Key Sources (in priority order)")
    st.markdown(
        """
    <div style="font-size:.78rem;color:#64748b;line-height:2;">
        1. Sidebar input above<br>
        2. <code>.streamlit/secrets.toml</code><br>
        3. <code>GROQ_API_KEY</code> env var
    </div>""",
        unsafe_allow_html=True,
    )

# ── Mode selector ─────────────────────────────────────────────────────────────
st.markdown("#### 🎯 Select Analysis Mode")
mode = st.radio(
    "Mode",
    ["🧩 Segment Personas", "🔮 Churn Model Report"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Segment Personas
# ══════════════════════════════════════════════════════════════════════════════
if "Segment" in mode:
    st.markdown("### 🧩 Customer Segment Personas")

    has_seg = "seg_profiles" in st.session_state and st.session_state["seg_profiles"] is not None

    if not has_seg:
        st.warning(
            "⚠️ No segmentation results found. "
            "Run clustering on the **Customer Segmentation** page first, then come back here."
        )
        st.markdown(
            """
        <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.18);
                    border-radius:12px;padding:1.5rem;margin-top:1rem;">
            <strong style="color:#e2e8f0;">Here's what you'll get after running segmentation:</strong>
            <ul style="color:#64748b;margin-top:.6rem;font-size:.9rem;">
                <li>A creative persona name for each cluster (e.g., "💎 High-Value Loyalists")</li>
                <li>Business value assessment (LTV tier + rationale)</li>
                <li>Top 2 churn/risk factors with actual data values</li>
                <li>3 specific, actionable retention recommendations per segment</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        profiles_df = st.session_state["seg_profiles"]
        best_algo = st.session_state.get("seg_best_algo", "K-Means")

        st.markdown(
            f"**Input:** {len(profiles_df)} clusters from **{best_algo}** · Sending to LLaMA-3.3-70B"
        )
        with st.expander("📋 Cluster profiles being analysed", expanded=False):
            st.dataframe(profiles_df, width="stretch")

        if st.button("⚡ Generate Segment Personas", type="primary", key="gen_seg"):
            st.markdown(
                """
            <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);
                        border-radius:12px;padding:1.5rem;margin:1rem 0;">
            """,
                unsafe_allow_html=True,
            )
            with st.spinner("Groq is generating insights…"):
                output = st.write_stream(
                    stream_segment_insights(profiles_df, best_algo, resolved_key)
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.session_state["seg_insight_output"] = output

        if "seg_insight_output" in st.session_state:
            st.divider()
            st.download_button(
                "📥 Download Insights (Markdown)",
                st.session_state["seg_insight_output"],
                file_name="neuroscope_segment_insights.md",
                mime="text/markdown",
            )

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Churn Model Report
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### 🔮 Churn Model Analysis Report")

    has_churn = (
        "metrics_df" in st.session_state
        and st.session_state["metrics_df"] is not None
        and "shap_importance_df" in st.session_state
    )

    if not has_churn:
        st.warning(
            "⚠️ No churn model results found. "
            "Train models on the **Churn Prediction** page first, then visit the SHAP tab to generate "
            "feature importance."
        )
        st.markdown(
            """
        <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.18);
                    border-radius:12px;padding:1.5rem;margin-top:1rem;">
            <strong style="color:#e2e8f0;">After training models, you'll get:</strong>
            <ul style="color:#64748b;margin-top:.6rem;font-size:.9rem;">
                <li>Executive summary of model readiness for production</li>
                <li>Model selection recommendation with metric justification</li>
                <li>Business interpretation of each key churn driver (SHAP features)</li>
                <li>A production deployment playbook (threshold, monitoring, retraining)</li>
                <li>2 hypothesis-driven A/B experiment ideas</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        metrics_df = st.session_state["metrics_df"]
        shap_df = st.session_state["shap_importance_df"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model metrics (input to LLM)**")
            st.dataframe(
                metrics_df.style.format("{:.4f}").highlight_max(axis=0, color="#0f2d1e"),
                width="stretch",
            )
        with col2:
            st.markdown("**Top SHAP features (input to LLM)**")
            st.dataframe(shap_df.head(6).reset_index(drop=True), width="stretch")

        if st.button("⚡ Generate Churn Report", type="primary", key="gen_churn"):
            st.markdown(
                """
            <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.18);
                        border-radius:12px;padding:1.5rem;margin:1rem 0;">
            """,
                unsafe_allow_html=True,
            )
            with st.spinner("Groq is generating the churn report…"):
                output = st.write_stream(stream_churn_insights(metrics_df, shap_df, resolved_key))
            st.markdown("</div>", unsafe_allow_html=True)
            st.session_state["churn_insight_output"] = output

        if "churn_insight_output" in st.session_state:
            st.divider()
            st.download_button(
                "📥 Download Report (Markdown)",
                st.session_state["churn_insight_output"],
                file_name="neuroscope_churn_report.md",
                mime="text/markdown",
            )

# ── Footer note ───────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
<div style="text-align:center;color:#475569;font-size:.78rem;padding:.5rem 0;">
    AI output is generated by <strong style="color:#f59e0b;">LLaMA-3.3-70B-Versatile</strong> via
    <strong style="color:#f59e0b;">Groq</strong> inference.
    Always review AI recommendations before business decisions.
</div>
""",
    unsafe_allow_html=True,
)
