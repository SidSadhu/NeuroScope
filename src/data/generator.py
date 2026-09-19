"""Synthetic customer dataset generator for NeuroScope."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Metadata for UI rendering (min, max, default, step)
FEATURE_METADATA: dict[str, dict] = {
    "age": {"min": 18, "max": 75, "default": 35.0, "step": 1.0, "desc": "Age (years)"},
    "tenure_months": {
        "min": 1,
        "max": 84,
        "default": 24.0,
        "step": 1.0,
        "desc": "Account tenure (months)",
    },
    "monthly_spend": {
        "min": 10.0,
        "max": 2000,
        "default": 120.0,
        "step": 10.0,
        "desc": "Avg monthly spend (USD)",
    },
    "num_products": {
        "min": 1,
        "max": 5,
        "default": 2.0,
        "step": 1.0,
        "desc": "Active products/subscriptions",
    },
    "login_frequency": {
        "min": 0,
        "max": 30,
        "default": 10.0,
        "step": 1.0,
        "desc": "Logins per month",
    },
    "support_calls_3m": {
        "min": 0,
        "max": 10,
        "default": 1.0,
        "step": 1.0,
        "desc": "Support calls (last 3 months)",
    },
    "satisfaction_score": {
        "min": 1.0,
        "max": 5.0,
        "default": 3.5,
        "step": 0.1,
        "desc": "CSAT score (1–5)",
    },
    "is_premium": {
        "min": 0,
        "max": 1,
        "default": 0.0,
        "step": 1.0,
        "desc": "Premium subscriber (0/1)",
    },
    "last_purchase_days": {
        "min": 0,
        "max": 180,
        "default": 30.0,
        "step": 1.0,
        "desc": "Days since last purchase",
    },
}

NUMERIC_FEATURES: list[str] = list(FEATURE_METADATA.keys())
TARGET_COL = "churned"


def generate_customer_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic customer dataset with natural churn patterns.

    Churn label is derived from a logistic model of business-relevant signals:
    - More support calls  → higher churn probability
    - Lower satisfaction  → higher churn probability
    - Longer inactivity   → higher churn probability
    - Premium subscribers → lower churn probability
    - Longer tenure       → lower churn probability

    Average churn rate: ~22%
    """
    rng = np.random.default_rng(random_state)

    # ── Demographics ─────────────────────────────────────────────────────
    age = rng.integers(18, 75, n_samples).astype(float)
    tenure_months = rng.integers(1, 84, n_samples).astype(float)

    # ── Engagement ───────────────────────────────────────────────────────
    monthly_spend = np.clip(
        np.round(rng.lognormal(mean=3.5, sigma=0.8, size=n_samples), 2),
        10.0,
        2000.0,
    )
    num_products = rng.integers(1, 6, n_samples).astype(float)
    login_frequency = rng.integers(0, 31, n_samples).astype(float)

    # ── Health signals ────────────────────────────────────────────────────
    support_calls_3m = rng.integers(0, 11, n_samples).astype(float)
    satisfaction_score = np.round(rng.uniform(1.0, 5.0, n_samples), 1)
    is_premium = rng.choice([0.0, 1.0], n_samples, p=[0.70, 0.30])
    last_purchase_days = rng.integers(0, 181, n_samples).astype(float)

    # ── Region (categorical, useful for segmentation context) ─────────────
    region = rng.choice(["North", "South", "East", "West", "Central"], n_samples)

    # ── Churn label via logistic model ────────────────────────────────────
    logit = (
        0.0
        - 0.025 * tenure_months
        + 0.35 * support_calls_3m
        - 0.45 * satisfaction_score
        + 0.004 * last_purchase_days
        - 0.08 * login_frequency
        - 0.35 * is_premium
        + 0.50
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.uniform(size=n_samples) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend,
            "num_products": num_products,
            "login_frequency": login_frequency,
            "support_calls_3m": support_calls_3m,
            "satisfaction_score": satisfaction_score,
            "is_premium": is_premium,
            "last_purchase_days": last_purchase_days,
            "region": region,
            TARGET_COL: churned,
        }
    )
