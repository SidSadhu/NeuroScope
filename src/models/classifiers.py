"""Multi-model classifier training and comparison for NeuroScope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Registry ─────────────────────────────────────────────────────────────────

CLASSIFIER_REGISTRY: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=120, random_state=42, learning_rate=0.1
    ),
    "Extra Trees": ExtraTreesClassifier(n_estimators=120, random_state=42, n_jobs=-1),
}

# Visual palette used in charts (must match CLASSIFIER_REGISTRY keys)
MODEL_COLORS: dict[str, str] = {
    "Logistic Regression": "#00d4ff",
    "Random Forest": "#7c3aed",
    "Gradient Boosting": "#ff6b6b",
    "Extra Trees": "#10b981",
}

METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass
class ModelResult:
    name: str
    pipeline: Any  # fitted sklearn Pipeline
    metrics: dict[str, float]
    y_test: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    X_test: pd.DataFrame
    X_train: pd.DataFrame
    y_train: np.ndarray
    feature_names: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_pipeline(name: str, clf: Any) -> Pipeline:
    """Logistic Regression needs a StandardScaler; tree models do not."""
    if name == "Logistic Regression":
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return Pipeline([("clf", clf)])


def _compute_metrics(
    y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "AUC-ROC": round(float(roc_auc_score(y_test, y_prob)), 4),
    }


# ── Public API ────────────────────────────────────────────────────────────────


def train_all_models(X: pd.DataFrame, y: pd.Series) -> dict[str, ModelResult]:
    """
    Train all registered classifiers on a stratified 80/20 split.

    Returns a dict mapping model name → ModelResult.
    Not cached here — callers manage caching via st.session_state.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    results: dict[str, ModelResult] = {}

    for name, clf in CLASSIFIER_REGISTRY.items():
        pipeline = _build_pipeline(name, clf)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(
            y_test.values if hasattr(y_test, "values") else y_test,
            y_pred,
            y_prob,
        )

        results[name] = ModelResult(
            name=name,
            pipeline=pipeline,
            metrics=metrics,
            y_test=y_test.values if hasattr(y_test, "values") else np.array(y_test),
            y_pred=y_pred,
            y_prob=y_prob,
            X_test=X_test.reset_index(drop=True),
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.values if hasattr(y_train, "values") else np.array(y_train),
            feature_names=list(X.columns),
        )

    return results


def get_best_model(results: dict[str, ModelResult], metric: str = "AUC-ROC") -> ModelResult:
    """Return the ModelResult with the highest value for *metric*."""
    return max(results.values(), key=lambda r: r.metrics[metric])


def metrics_to_dataframe(results: dict[str, ModelResult]) -> pd.DataFrame:
    """Build a summary DataFrame indexed by model name."""
    rows = [{"Model": name} | r.metrics for name, r in results.items()]
    return pd.DataFrame(rows).set_index("Model")


def get_classification_report(result: ModelResult) -> pd.DataFrame:
    report = classification_report(
        result.y_test,
        result.y_pred,
        target_names=["Retained", "Churned"],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().round(3)
