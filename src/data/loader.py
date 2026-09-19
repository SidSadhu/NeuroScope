"""Data loading and preprocessing utilities for NeuroScope."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.generator import (
    FEATURE_METADATA,
    NUMERIC_FEATURES,
    TARGET_COL,
    generate_customer_dataset,
)

# Features used for clustering when no specific selection is made
CLUSTERING_DEFAULT_FEATURES = [
    "monthly_spend",
    "satisfaction_score",
    "tenure_months",
    "login_frequency",
    "support_calls_3m",
]

__all__ = [
    "FEATURE_METADATA",
    "NUMERIC_FEATURES",
    "TARGET_COL",
    "CLUSTERING_DEFAULT_FEATURES",
    "load_default_dataset",
    "load_csv",
    "get_features_and_target",
]


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    """Return the built-in 1,500-row synthetic customer dataset (cached)."""
    return generate_customer_dataset()


def load_csv(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and lightly validate a user-uploaded CSV.

    Returns
    -------
    df : pd.DataFrame
    warnings : list[str]  — user-facing warnings about missing columns
    """
    warnings: list[str] = []
    df = pd.read_csv(uploaded_file)

    missing_features = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing_features:
        warnings.append(
            f"Missing expected feature columns: {missing_features}. "
            "Only available columns will be used."
        )

    if TARGET_COL not in df.columns:
        warnings.append(
            f"Target column '{TARGET_COL}' not found. "
            "Classification is disabled; segmentation is still available."
        )

    return df, warnings


def get_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Extract feature matrix X and target vector y from a dataframe."""
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[available]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y
