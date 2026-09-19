"""Tests for ML model training and clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_customer_dataset
from src.models.classifiers import (
    METRIC_LABELS,
    get_best_model,
    metrics_to_dataframe,
    train_all_models,
)
from src.models.clustering import (
    compute_elbow_data,
    get_cluster_profiles,
    reduce_to_3d,
    run_clustering,
)
from src.models.explainability import (
    _fallback_importances,
    _select_positive_class,
    compute_shap_feature_importance,
    compute_shap_waterfall_values,
)


@pytest.fixture(scope="module")
def customer_data():
    df = generate_customer_dataset(n_samples=300, random_state=42)
    return df


@pytest.fixture(scope="module")
def X_y(customer_data):
    from src.data.loader import get_features_and_target

    return get_features_and_target(customer_data)


@pytest.fixture(scope="module")
def model_results(X_y):
    X, y = X_y
    return train_all_models(X, y)


# ── Classifier tests ──────────────────────────────────────────────────────────


class TestClassifiers:
    def test_all_models_trained(self, model_results):
        assert len(model_results) == 4

    def test_model_names(self, model_results):
        expected = {"Logistic Regression", "Random Forest", "Gradient Boosting", "Extra Trees"}
        assert set(model_results.keys()) == expected

    def test_metrics_in_valid_range(self, model_results):
        for name, result in model_results.items():
            for metric, val in result.metrics.items():
                assert 0.0 <= val <= 1.0, f"{name} {metric}={val:.4f} out of [0,1]"

    def test_predictions_are_binary(self, model_results):
        for name, result in model_results.items():
            assert set(np.unique(result.y_pred)).issubset({0, 1}), f"{name}: non-binary predictions"

    def test_probabilities_in_range(self, model_results):
        for name, result in model_results.items():
            assert result.y_prob.min() >= 0.0
            assert result.y_prob.max() <= 1.0

    def test_auc_above_random(self, model_results):
        for name, result in model_results.items():
            assert result.metrics["AUC-ROC"] > 0.55, f"{name} AUC-ROC too low"

    def test_get_best_model(self, model_results):
        best = get_best_model(model_results)
        assert best.name in model_results

    def test_metrics_dataframe_shape(self, model_results):
        df = metrics_to_dataframe(model_results)
        assert df.shape == (4, len(METRIC_LABELS))

    def test_feature_names_populated(self, model_results):
        for name, result in model_results.items():
            assert len(result.feature_names) > 0


# ── Clustering tests ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def X_cluster(customer_data):
    cols = ["monthly_spend", "satisfaction_score", "tenure_months", "login_frequency"]
    return customer_data[cols]


class TestClustering:
    def test_run_clustering_returns_three_algos(self, X_cluster):
        results = run_clustering(X_cluster, n_clusters=3)
        assert set(results.keys()) == {"K-Means", "DBSCAN", "Hierarchical"}

    def test_label_length_matches_input(self, X_cluster):
        results = run_clustering(X_cluster, n_clusters=3)
        for algo, res in results.items():
            assert len(res.labels) == len(X_cluster), f"{algo} labels length mismatch"

    def test_kmeans_cluster_count(self, X_cluster):
        results = run_clustering(X_cluster, n_clusters=4)
        assert results["K-Means"].n_clusters == 4

    def test_silhouette_in_valid_range(self, X_cluster):
        results = run_clustering(X_cluster, n_clusters=3)
        for algo, res in results.items():
            assert -1.0 <= res.silhouette <= 1.0, f"{algo} silhouette out of range"

    def test_elbow_data_shape(self, X_cluster):
        df = compute_elbow_data(X_cluster, max_k=5)
        assert list(df.columns) == ["k", "inertia", "silhouette"]
        assert len(df) == 4  # k=2..5

    def test_reduce_to_3d(self, X_cluster):
        arr = reduce_to_3d(X_cluster)
        assert arr.shape[1] == 3

    def test_cluster_profiles(self, X_cluster):
        results = run_clustering(X_cluster, n_clusters=3)
        labels = results["K-Means"].labels
        profiles = get_cluster_profiles(X_cluster, labels, list(X_cluster.columns))
        assert len(profiles) >= 1


# ── Explainability tests ──────────────────────────────────────────────────────


class TestExplainability:
    def test_shap_importance_columns(self, model_results, X_y):
        X, _ = X_y
        best = get_best_model(model_results)
        df = compute_shap_feature_importance(best.pipeline, best.X_test, best.name, sample_size=50)
        assert "feature" in df.columns
        assert "importance" in df.columns

    def test_shap_importance_length(self, model_results, X_y):
        X, _ = X_y
        best = get_best_model(model_results)
        df = compute_shap_feature_importance(best.pipeline, best.X_test, best.name, sample_size=50)
        assert len(df) == X.shape[1]

    def test_shap_importance_non_negative(self, model_results, X_y):
        X, _ = X_y
        best = get_best_model(model_results)
        df = compute_shap_feature_importance(best.pipeline, best.X_test, best.name, sample_size=50)
        assert (df["importance"] >= 0).all()

    def test_waterfall_returns_expected_keys(self, model_results, X_y):
        X, _ = X_y
        best = get_best_model(model_results)
        row = best.X_test.iloc[[0]]
        data = compute_shap_waterfall_values(best.pipeline, row, best.name)
        if not data:
            pytest.skip("waterfall computation unavailable for this explainer")
        assert set(data.keys()) == {
            "base_value",
            "shap_values",
            "feature_names",
            "feature_values",
        }
        assert len(data["shap_values"]) == X.shape[1]
        assert len(data["feature_names"]) == X.shape[1]
        assert np.isfinite(data["base_value"])

    def test_fallback_importances_tree_model(self, model_results, X_y):
        X, _ = X_y
        rf = model_results["Random Forest"]
        sample = rf.X_test.iloc[:10]
        imp = _fallback_importances(rf.pipeline, sample, "Random Forest")
        assert len(imp) == X.shape[1]
        assert (imp >= 0).all()

    def test_fallback_importances_linear_model(self, model_results, X_y):
        X, _ = X_y
        lr = model_results["Logistic Regression"]
        sample = lr.X_test.iloc[:10]
        imp = _fallback_importances(lr.pipeline, sample, "Logistic Regression")
        assert len(imp) == X.shape[1]
        assert (imp >= 0).all()

    def test_fallback_importances_unknown_model(self):
        """clf with neither feature_importances_ nor coef_ → ones vector."""
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([("clf", DummyClassifier())])
        sample = pd.DataFrame(np.zeros((5, 3)), columns=["a", "b", "c"])
        imp = _fallback_importances(pipeline, sample, "Unknown Model")
        assert len(imp) == 3
        assert np.allclose(imp, 1.0)


class TestSelectPositiveClass:
    """Tests for the SHAP version-compatibility normalizer."""

    def test_legacy_list_api(self):
        class0 = np.array([[1.0, -1.0], [2.0, -2.0]])
        class1 = np.array([[0.5, 0.5], [-0.5, -0.5]])
        arr, base = _select_positive_class([class0, class1], np.array([0.6, 0.4]))
        assert arr.shape == (2, 2)
        np.testing.assert_allclose(arr, class1)
        assert base == pytest.approx(0.4)

    def test_3d_array_new_shap(self):
        arr_in = np.zeros((10, 4, 2))
        arr_in[:, :, 1] = 1.0
        out, base = _select_positive_class(arr_in, np.array([0.3, 0.7]))
        assert out.shape == (10, 4)
        assert (out == 1.0).all()
        assert base == pytest.approx(0.7)

    def test_3d_single_output(self):
        arr_in = np.full((5, 3, 1), 2.0)
        out, base = _select_positive_class(arr_in, np.array(0.9))
        assert out.shape == (5, 3)
        assert base == pytest.approx(0.9)

    def test_2d_single_output(self):
        arr_in = np.full((6, 3), 1.5)
        out, base = _select_positive_class(arr_in, np.array(0.2))
        assert out.shape == (6, 3)
        assert base == pytest.approx(0.2)

    def test_2d_with_class_vector(self):
        arr_in = np.zeros((4, 3))
        out, base = _select_positive_class(arr_in, np.array([0.55, 0.45]))
        assert out.shape == (4, 3)
        assert base == pytest.approx(0.45)
