"""Groq-powered AI insight generation for NeuroScope."""

from __future__ import annotations

import os
from collections.abc import Generator

import pandas as pd
import streamlit as st

_MODEL_ID = "llama-3.3-70b-versatile"
_MOCK_FOOTER = "\n\n---\n*⚠️ Sample output — add your Groq API key to generate live AI insights.*"


# ── Key resolution ────────────────────────────────────────────────────────────


def resolve_api_key(user_input: str | None = None) -> str | None:
    """
    Resolve Groq API key from (in priority order):
    1. Explicit user_input argument
    2. st.secrets["GROQ_API_KEY"]
    3. GROQ_API_KEY environment variable
    """
    if user_input and user_input.strip():
        return user_input.strip()
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def _get_client(api_key: str | None):
    try:
        from groq import Groq  # noqa: PLC0415

        key = resolve_api_key(api_key)
        return Groq(api_key=key) if key else None
    except ImportError:
        return None


# ── Segment insights ──────────────────────────────────────────────────────────


def _segment_prompt(profiles_df: pd.DataFrame, algorithm: str) -> str:
    return f"""You are a senior customer analytics consultant.

A company ran **{algorithm}** clustering on {len(profiles_df)} customer segments.
Here are the mean feature values per cluster:

{profiles_df.to_string()}

Feature legend:
- age: Customer age (years)
- tenure_months: How long they've been a customer
- monthly_spend: Average monthly spend (USD)
- num_products: Active products/subscriptions
- login_frequency: Logins per month (engagement proxy)
- support_calls_3m: Support calls in last 3 months (frustration proxy)
- satisfaction_score: CSAT score 1–5 (higher = happier)
- is_premium: Premium subscriber? (1=yes, 0=no)
- last_purchase_days: Days since last purchase (recency)

For EACH cluster provide exactly this structure:

## Cluster [N] — [Creative Persona Name with Emoji]

**Profile**: 2–3 sentences describing who these customers are and what makes them distinct.

**Business Value**: LTV tier (🟢 High / 🟡 Medium / 🔴 Low) and a one-sentence rationale.

**Top Risk Factors**: The 2 biggest churn/risk signals with the specific numbers.

**Recommended Actions**:
1. [Specific, actionable recommendation with expected outcome]
2. [Specific, actionable recommendation with expected outcome]
3. [Specific, actionable recommendation with expected outcome]

Use the actual numbers from the data. Be sharp and business-focused.
"""


def stream_segment_insights(
    profiles_df: pd.DataFrame,
    algorithm: str,
    api_key: str | None = None,
) -> Generator[str, None, None]:
    """Yield streamed tokens of AI-generated customer segment personas."""
    client = _get_client(api_key)
    if not client:
        yield _mock_segment_insights(profiles_df)
        return

    try:
        stream = client.chat.completions.create(
            model=_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert customer analytics and marketing strategy consultant. "
                        "Provide precise, data-driven, and actionable insights formatted in markdown."
                    ),
                },
                {"role": "user", "content": _segment_prompt(profiles_df, algorithm)},
            ],
            stream=True,
            temperature=0.65,
            max_tokens=2000,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as exc:
        yield f"\n\n❌ **Groq API error:** `{exc}`\n\nShowing sample output instead:\n\n"
        yield _mock_segment_insights(profiles_df)


# ── Churn insights ────────────────────────────────────────────────────────────


def _churn_prompt(metrics_df: pd.DataFrame, importance_df: pd.DataFrame) -> str:
    return f"""You are a senior ML engineer and customer retention strategist.

## Model Performance (test set)
{metrics_df.to_string()}

## Top Churn Predictors (SHAP importance, descending)
{importance_df.head(6).to_string(index=False)}

Provide the following analysis:

## Executive Summary
2–3 sentences on model readiness for production.

## Model Selection Recommendation
Which model goes to production and why? Reference the specific metric numbers.

## Key Churn Drivers
For each of the top 3 predictors: what is the business implication, and what intervention reduces that risk?

## Deployment Playbook
3 concrete steps for taking this model to production (threshold tuning, monitoring, retraining cadence).

## A/B Experiment Ideas
2 hypothesis-driven experiments to validate retention impact of model-driven interventions.

Reference the actual metric values and feature names throughout.
"""


def stream_churn_insights(
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    api_key: str | None = None,
) -> Generator[str, None, None]:
    """Yield streamed tokens of AI-generated churn model analysis."""
    client = _get_client(api_key)
    if not client:
        yield _mock_churn_insights()
        return

    try:
        stream = client.chat.completions.create(
            model=_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert ML engineer specialising in production churn prediction systems. "
                        "Give strategic, numbers-backed recommendations in clean markdown."
                    ),
                },
                {"role": "user", "content": _churn_prompt(metrics_df, importance_df)},
            ],
            stream=True,
            temperature=0.6,
            max_tokens=2000,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as exc:
        yield f"\n\n❌ **Groq API error:** `{exc}`\n\nShowing sample output instead:\n\n"
        yield _mock_churn_insights()


# ── Mock fallbacks ────────────────────────────────────────────────────────────


def _mock_segment_insights(profiles_df: pd.DataFrame) -> str:
    n = len(profiles_df)
    return (
        f"""## Customer Segment Analysis — {n} Segments Identified

## Cluster 0 — 💎 High-Value Loyalists
**Profile**: Long-tenure customers (avg 60+ months) with above-average spend and high satisfaction. They log in frequently and rarely contact support — this is your core revenue engine.

**Business Value**: 🟢 **High LTV** — likely generating 40–50% of total revenue from ~20% of the base.

**Top Risk Factors**: Low current risk, but watch for satisfaction score drops below 4.0 and sudden login frequency decline.

**Recommended Actions**:
1. Enrol in a VIP loyalty tier with exclusive early-access features to reinforce belonging
2. Assign dedicated Customer Success managers for accounts with monthly_spend > $500
3. Conduct quarterly strategic reviews to surface premium upsell opportunities

---

## Cluster 1 — ⚠️ At-Risk Mid-Tier
**Profile**: Mid-tenure customers with moderate spend but clear frustration signals — elevated support call volume and declining satisfaction scores suggest unresolved friction.

**Business Value**: 🟡 **Medium LTV** — significant volume but at acute churn risk without intervention.

**Top Risk Factors**: support_calls_3m above cluster average (frustration) and satisfaction_score trending below 3.0.

**Recommended Actions**:
1. Trigger a proactive CS outreach within 7 days for any account matching this segment profile
2. Offer a complimentary "product health check" session to identify and resolve pain points
3. A/B test a 15% annual renewal discount targeted exclusively at this cohort

---

## Cluster 2 — 🌱 Growth-Stage Newcomers
**Profile**: Short tenure (<12 months), modest spend, but surprisingly high satisfaction scores. They're still discovering value — early-stage customers with significant upside.

**Business Value**: 🟡→🟢 **Medium LTV with high growth potential** if onboarded successfully.

**Top Risk Factors**: Time-to-value risk — if they don't reach their "aha moment" within 60 days, churn probability spikes non-linearly.

**Recommended Actions**:
1. Deploy a 60-day personalised onboarding sequence with milestone-triggered emails
2. Offer a complimentary 30-day premium trial to accelerate feature discovery
3. Monitor weekly feature adoption milestones and trigger in-app celebrations on completion
"""
        + _MOCK_FOOTER
    )


def _mock_churn_insights() -> str:
    return (
        """## Churn Model Analysis — Executive Report

## Executive Summary
The ensemble models (Gradient Boosting, Random Forest) achieve AUC-ROC above 0.85, indicating strong readiness for production deployment. The model reliably surfaces the top 20% highest-risk churners, enabling targeted and cost-efficient retention spend.

## Model Selection Recommendation
**Gradient Boosting** is recommended for production: it delivers the strongest AUC-ROC / F1 balance and is fully SHAP-compatible for regulatory explainability. For teams with strict interpretability requirements, **Logistic Regression** provides near-equivalent accuracy with transparent coefficient-level reasoning.

## Key Churn Drivers

**1. support_calls_3m** — Each additional support call raises churn probability by ~12%. *Intervention*: Implement a proactive resolution SLA; flag accounts after 2 calls for CS escalation.

**2. satisfaction_score** — Non-linear risk spike below 3.0. *Intervention*: Automated alert at 3.5; trigger a personalised outreach from CS within 48 hours.

**3. last_purchase_days** — Customers inactive for >45 days show 3× baseline churn risk. *Intervention*: Day-30 win-back email with personalised product recommendation.

## Deployment Playbook
1. **Threshold tuning**: Lower decision threshold to 0.35 (vs default 0.50) — in a retention economics context, recall is more valuable than precision.
2. **Monthly retraining**: Roll a 12-month training window; use PSI scores to detect feature drift before retraining.
3. **Shadow mode first**: Run the model in shadow mode for 30 days alongside existing rules-based system before going live.

## A/B Experiment Ideas
1. **Retention offer test**: Randomly assign 50% of top-quintile risk customers to a "Save Offer" (20% renewal discount). Measure 90-day retention lift and revenue impact.
2. **Proactive outreach test**: For model-flagged accounts, trigger a CS call vs. control group. Measure NPS delta and 6-month LTV difference.
"""
        + _MOCK_FOOTER
    )
