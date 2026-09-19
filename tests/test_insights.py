"""Tests for AI insights module (Groq integration with mocking)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ai.insights import (
    _mock_churn_insights,
    _mock_segment_insights,
    resolve_api_key,
    stream_churn_insights,
    stream_segment_insights,
)


@pytest.fixture
def sample_profiles():
    return pd.DataFrame(
        {
            "monthly_spend": [450.0, 90.0, 230.0],
            "satisfaction_score": [4.5, 2.1, 3.8],
            "tenure_months": [62.0, 8.0, 30.0],
            "login_frequency": [22.0, 3.0, 12.0],
            "support_calls_3m": [0.5, 4.2, 1.8],
        },
        index=[0, 1, 2],
    )


@pytest.fixture
def sample_metrics():
    return pd.DataFrame(
        {
            "Accuracy": [0.78, 0.83, 0.85, 0.82],
            "AUC-ROC": [0.84, 0.89, 0.91, 0.87],
            "F1": [0.72, 0.78, 0.80, 0.76],
        },
        index=["Logistic Regression", "Random Forest", "Gradient Boosting", "Extra Trees"],
    )


@pytest.fixture
def sample_importance():
    return pd.DataFrame(
        {
            "feature": ["support_calls_3m", "satisfaction_score", "last_purchase_days"],
            "importance": [0.35, 0.28, 0.18],
        }
    )


class TestResolveApiKey:
    def test_returns_explicit_input(self):
        assert resolve_api_key("gsk_test123") == "gsk_test123"

    def test_strips_whitespace(self):
        assert resolve_api_key("  gsk_abc  ") == "gsk_abc"

    def test_returns_none_on_empty_string(self):
        with patch.dict("os.environ", {}, clear=True):
            result = resolve_api_key("")
            # May return env or None but not empty string
            assert result != ""

    def test_reads_env_variable(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "gsk_from_env"}):
            assert resolve_api_key() == "gsk_from_env"


class TestMockFallbacks:
    def test_mock_segment_insights_returns_string(self, sample_profiles):
        result = _mock_segment_insights(sample_profiles)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_mock_segment_insights_contains_cluster_count(self, sample_profiles):
        result = _mock_segment_insights(sample_profiles)
        assert "3" in result  # 3 clusters

    def test_mock_churn_insights_returns_string(self):
        result = _mock_churn_insights()
        assert isinstance(result, str)
        assert "Model" in result or "Churn" in result

    def test_mock_footer_present(self, sample_profiles):
        result = _mock_segment_insights(sample_profiles)
        assert "Sample output" in result or "sample output" in result.lower()


class TestStreamSegmentInsights:
    def test_yields_mock_when_no_client(self, sample_profiles):
        with patch("src.ai.insights._get_client", return_value=None):
            chunks = list(stream_segment_insights(sample_profiles, "K-Means", api_key=None))
        assert len(chunks) >= 1
        full = "".join(chunks)
        assert len(full) > 100

    def test_yields_streamed_content_with_client(self, sample_profiles):
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Test segment insight chunk"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        with patch("src.ai.insights._get_client", return_value=mock_client):
            chunks = list(stream_segment_insights(sample_profiles, "K-Means", api_key="gsk_fake"))
        assert "Test segment insight chunk" in chunks


class TestStreamChurnInsights:
    def test_yields_mock_when_no_client(self, sample_metrics, sample_importance):
        with patch("src.ai.insights._get_client", return_value=None):
            chunks = list(stream_churn_insights(sample_metrics, sample_importance, api_key=None))
        full = "".join(chunks)
        assert len(full) > 100

    def test_yields_streamed_content_with_client(self, sample_metrics, sample_importance):
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = "Churn insight streamed"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        with patch("src.ai.insights._get_client", return_value=mock_client):
            chunks = list(
                stream_churn_insights(sample_metrics, sample_importance, api_key="gsk_fake")
            )
        assert "Churn insight streamed" in chunks
