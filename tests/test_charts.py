"""Tests for the Plotly visualization suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure

from src.data.generator import generate_customer_dataset
from src.models.classifiers import get_best_model, metrics_to_dataframe, train_all_models
from src.models.clustering import get_cluster_profiles, reduce_to_3d, run_clustering
from src.models.explainability import compute_shap_feature_importance
from src.visualization.charts import (
    MODEL_COLORS,
    plot_algorithm_comparison,
    plot_churn_donut,
    plot_cluster_2d,
    plot_cluster_3d,
    plot_cluster_profiles,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_elbow_silhouette,
    plot_model_radar,
    plot_risk_gauge,
    plot_roc_curves,
    plot_shap_importance,
    plot_shap_waterfall,
)


@pytest.fixture(scope="module")
def customer_data():
    return generate_customer_dataset(n_samples=300, random_state=42)


@pytest.fixture(scope="module")
def model_results(customer_data):
    from src.data.loader import get_features_and_target

    X, y = get_features_and_target(customer_data)
    return train_all_models(X, y)


@pytest.fixture(scope="module")
def best(model_results):
    return get_best_model(model_results)


@pytest.fixture(scope="module")
def shap_importance(best):
    return compute_shap_feature_importance(best.pipeline, best.X_test, best.name, sample_size=50)


@pytest.fixture(scope="module")
def waterfall_data(best):
    from src.models.explainability import compute_shap_waterfall_values

    row = best.X_test.iloc[[0]]
    return compute_shap_waterfall_values(best.pipeline, row, best.name)


@pytest.fixture(scope="module")
def cluster_results(customer_data):
    cols = ["monthly_spend", "satisfaction_score", "tenure_months", "login_frequency"]
    X = customer_data[cols]
    return X, run_clustering(X, n_clusters=3)


@pytest.fixture(scope="module")
def elbow_df(cluster_results):
    from src.models.clustering import compute_elbow_data

    X, _ = cluster_results
    return compute_elbow_data(X, max_k=4)


class TestRocCurves:
    def test_returns_figure(self, model_results):
        fig = plot_roc_curves(model_results)
        assert isinstance(fig, Figure)
        # 1 baseline trace + one per model
        assert len(fig.data) == 1 + len(model_results)

    def test_traces_named_with_auc(self, model_results):
        fig = plot_roc_curves(model_results)
        names = [t.name for t in fig.data if t.name]
        for name in model_results:
            assert any(name in n for n in names)


class TestConfusionMatrix:
    def test_returns_heatmap_figure(self, best):
        fig = plot_confusion_matrix(best.y_test, best.y_pred, best.name)
        assert isinstance(fig, Figure)
        assert fig.data[0].type == "heatmap"
        assert best.name in fig.layout.title.text

    def test_default_title_without_name(self, best):
        fig = plot_confusion_matrix(best.y_test, best.y_pred)
        assert "Confusion Matrix" in fig.layout.title.text


class TestShapImportanceChart:
    def test_returns_bar_figure(self, shap_importance):
        fig = plot_shap_importance(shap_importance)
        assert isinstance(fig, Figure)
        assert fig.data[0].type == "bar"
        assert len(fig.data[0].x) == len(shap_importance)

    def test_handles_zero_importance(self):
        df = pd.DataFrame({"feature": ["a", "b"], "importance": [0.0, 0.0]})
        fig = plot_shap_importance(df)
        assert isinstance(fig, Figure)


class TestShapWaterfallChart:
    def test_empty_data_returns_empty_figure(self):
        fig = plot_shap_waterfall({}, predicted_prob=0.5)
        assert isinstance(fig, Figure)
        assert len(fig.data) == 0

    def test_valid_data_returns_bar_figure(self, waterfall_data, best):
        if not waterfall_data:
            pytest.skip("waterfall computation unavailable for this model")
        prob = float(best.y_prob[0])
        fig = plot_shap_waterfall(waterfall_data, predicted_prob=prob)
        assert isinstance(fig, Figure)
        assert fig.data[0].type == "bar"
        assert len(fig.data[0].x) == len(waterfall_data["feature_names"])


class TestModelRadar:
    def test_returns_scatterpolar_per_model(self, model_results):
        fig = plot_model_radar(metrics_to_dataframe(model_results))
        assert isinstance(fig, Figure)
        assert len(fig.data) == len(model_results)
        assert all(t.type == "scatterpolar" for t in fig.data)

    def test_custom_metrics_subset(self, model_results):
        df = metrics_to_dataframe(model_results)[["Accuracy", "F1"]]
        fig = plot_model_radar(df)
        assert len(fig.data) == len(model_results)


class TestCluster3d:
    def test_shape_matches_clusters(self, cluster_results):
        X, results = cluster_results
        labels = results["K-Means"].labels
        X_3d = reduce_to_3d(X)
        fig = plot_cluster_3d(X_3d, labels)
        assert isinstance(fig, Figure)
        n_clusters = len(set(labels))
        assert len(fig.data) == n_clusters

    def test_noise_label_named(self, cluster_results):
        X, _ = cluster_results
        labels = np.full(len(X), -1)
        fig = plot_cluster_3d(reduce_to_3d(X), labels)
        assert fig.data[0].name == "Noise"


class TestCluster2d:
    def test_returns_scatter_figure(self, cluster_results):
        X, results = cluster_results
        labels = results["K-Means"].labels
        fig = plot_cluster_2d(X.iloc[:, 0], X.iloc[:, 1], labels, "spend", "satisfaction")
        assert isinstance(fig, Figure)
        assert len(fig.data) == len(set(labels))


class TestElbowSilhouette:
    def test_returns_dual_axis_figure(self, elbow_df):
        fig = plot_elbow_silhouette(elbow_df)
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2


class TestAlgorithmComparison:
    def test_returns_bar_figure(self, cluster_results):
        _, results = cluster_results
        fig = plot_algorithm_comparison(results)
        assert isinstance(fig, Figure)
        assert fig.data[0].type == "bar"
        assert list(fig.data[0].x) == list(results.keys())


class TestClusterProfiles:
    def test_returns_scatterpolar_figure(self, cluster_results):
        X, results = cluster_results
        labels = results["K-Means"].labels
        profiles = get_cluster_profiles(X, labels, list(X.columns))
        fig = plot_cluster_profiles(profiles)
        assert isinstance(fig, Figure)
        assert len(fig.data) == len(profiles)
        assert all(t.type == "scatterpolar" for t in fig.data)


class TestPalette:
    def test_model_colors_match_registry(self):
        from src.models.classifiers import CLASSIFIER_REGISTRY

        assert set(MODEL_COLORS.keys()) == set(CLASSIFIER_REGISTRY.keys())


class TestChurnDonut:
    def test_returns_pie_with_center_annotation(self, customer_data):
        fig = plot_churn_donut(customer_data["churned"])
        assert isinstance(fig, Figure)
        assert fig.data[0].type == "pie"
        assert fig.data[0].hole == 0.62
        assert len(fig.layout.annotations) == 1
        assert "churn rate" in fig.layout.annotations[0].text

    def test_values_match_input(self, customer_data):
        y = customer_data["churned"]
        fig = plot_churn_donut(y)
        assert list(fig.data[0].values) == [int((y == 0).sum()), int((y == 1).sum())]


class TestCorrelationHeatmap:
    def test_returns_square_heatmap(self, customer_data):
        fig = plot_correlation_heatmap(customer_data)
        assert isinstance(fig, Figure)
        assert fig.data[0].type == "heatmap"
        n = customer_data.select_dtypes(include=[np.number]).shape[1]
        assert fig.data[0].z.shape == (n, n)
        assert fig.data[0].zmax == 1
        assert fig.data[0].zmin == -1


class TestRiskGauge:
    def test_low_risk_green(self):
        fig = plot_risk_gauge(0.10)
        assert "LOW RISK" in fig.layout.title.text

    def test_medium_risk_amber(self):
        fig = plot_risk_gauge(0.50)
        assert "MEDIUM RISK" in fig.layout.title.text

    def test_high_risk_coral(self):
        fig = plot_risk_gauge(0.90)
        assert "HIGH RISK" in fig.layout.title.text

    def test_clips_out_of_range(self):
        fig = plot_risk_gauge(7.0)
        assert fig.data[0].value == 100.0
