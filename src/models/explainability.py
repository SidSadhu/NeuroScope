"""SHAP-based model explainability for NeuroScope."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _select_positive_class(shap_values, expected_value) -> tuple[np.ndarray, float]:
    """
    Normalize SHAP outputs across shap versions to the positive-class slice.

    Handles the three layouts seen in the wild:
      * ``[class0, class1]`` list          — legacy shap API
      * ndarray ``(n_samples, n_features)`` — single-output / binary class-1 only
      * ndarray ``(n_samples, n_features, n_classes)`` — shap >= 0.45 with sklearn >= 1.4

    Returns
    -------
    (shap_matrix, base_value) : 2D array of shape (n_samples, n_features) and
    the scalar expected value for the positive class.
    """
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1])
        ev = np.asarray(expected_value).reshape(-1)
        return arr, float(ev[1])

    arr = np.asarray(shap_values)
    ev = np.asarray(expected_value)

    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        idx = 1 if arr.shape[2] > 1 else 0
        arr = arr[:, :, idx]
    else:
        idx = 1 if ev.size > 1 else 0

    flat_ev = ev.reshape(-1)
    base = float(flat_ev[idx]) if flat_ev.size > 1 else float(flat_ev[0])
    return arr, base


def compute_shap_feature_importance(
    pipeline,
    X: pd.DataFrame,
    model_name: str,
    sample_size: int = 200,
) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values as a feature importance ranking.

    Falls back to the model's native ``feature_importances_`` / ``coef_``
    if SHAP raises an error.

    Returns
    -------
    pd.DataFrame with columns ['feature', 'importance'], sorted descending.
    """
    X_sample = X.iloc[:sample_size]
    try:
        import shap  # noqa: PLC0415

        if model_name == "Logistic Regression":
            scaler = pipeline.named_steps["scaler"]
            clf = pipeline.named_steps["clf"]
            X_scaled = scaler.transform(X_sample)
            explainer = shap.LinearExplainer(clf, X_scaled, feature_names=list(X.columns))
            shap_values = explainer.shap_values(X_scaled)
        else:
            clf = pipeline.named_steps["clf"]
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample.values)

        # RandomForest / ExtraTrees return a list [class0, class1] on legacy
        # shap, or a (n_samples, n_features, n_classes) array on newer shap.
        shap_matrix, _ = _select_positive_class(shap_values, explainer.expected_value)
        mean_abs = np.abs(shap_matrix).mean(axis=0)
    except Exception:
        mean_abs = _fallback_importances(pipeline, X_sample, model_name)

    return (
        pd.DataFrame({"feature": list(X.columns), "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_shap_waterfall_values(
    pipeline,
    X_row: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Compute per-feature SHAP values for a single prediction row.

    Returns a dict with keys:
        base_value, shap_values, feature_names, feature_values
    or an empty dict on failure.
    """
    try:
        import shap  # noqa: PLC0415

        if model_name == "Logistic Regression":
            scaler = pipeline.named_steps["scaler"]
            clf = pipeline.named_steps["clf"]
            X_scaled = scaler.transform(X_row)
            explainer = shap.LinearExplainer(clf, X_scaled)
            shap_matrix, base_value = _select_positive_class(
                explainer.shap_values(X_scaled), explainer.expected_value
            )
        else:
            clf = pipeline.named_steps["clf"]
            explainer = shap.TreeExplainer(clf)
            shap_matrix, base_value = _select_positive_class(
                explainer.shap_values(X_row.values), explainer.expected_value
            )

        shap_vals = np.asarray(shap_matrix)[0]

        return {
            "base_value": base_value,
            "shap_values": shap_vals,
            "feature_names": list(X_row.columns),
            "feature_values": X_row.iloc[0].values,
        }
    except Exception:
        return {}


# ── Fallback ──────────────────────────────────────────────────────────────────


def _fallback_importances(pipeline, X: pd.DataFrame, model_name: str) -> np.ndarray:
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    if hasattr(clf, "coef_"):
        return np.abs(clf.coef_[0])
    return np.ones(X.shape[1])
