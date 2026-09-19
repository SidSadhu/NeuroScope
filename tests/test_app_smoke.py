"""End-to-end Streamlit smoke tests using the official AppTest framework."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

at_pages = [
    "pages/overview.py",
    "pages/churn_prediction.py",
    "pages/segmentation.py",
    "pages/ai_insights.py",
]


@pytest.mark.parametrize("page_path", at_pages)
def test_page_renders_without_exception(page_path):
    """Every page must boot and reach completion with no uncaught exception."""
    at = AppTest.from_file(page_path, default_timeout=60)
    at.run()
    assert not at.exception, f"{page_path} raised: {at.exception}"


class TestOverviewPage:
    def test_metrics_rendered(self):
        at = AppTest.from_file("pages/overview.py", default_timeout=60)
        at.run()
        assert len(at.metric) >= 4

    def test_markdown_blocks_present(self):
        at = AppTest.from_file("pages/overview.py", default_timeout=60)
        at.run()
        html = " ".join(md.value for md in at.markdown)
        assert "NeuroScope" in html
        assert "Dataset Snapshot" in html


class TestChurnPage:
    def test_pre_train_state(self):
        at = AppTest.from_file("pages/churn_prediction.py", default_timeout=60)
        at.run()
        assert len(at.metric) >= 4  # dataset overview metrics
        assert len(at.button) >= 1  # train button exists

    def test_train_button_populates_results(self):
        at = AppTest.from_file("pages/churn_prediction.py", default_timeout=120)
        at.run()
        at.button[0].click()
        at.run()
        assert not at.exception
        # Tabs (4 results tabs) and winner banner appear after training
        assert len(at.tabs) >= 4

    def test_training_persists_in_session_state(self):
        at = AppTest.from_file("pages/churn_prediction.py", default_timeout=120)
        at.run()
        at.button[0].click()
        at.run()
        assert "churn_results" in at.session_state["churn_results"] or (
            at.session_state["churn_results"] is not None
        )


class TestSegmentationPage:
    def test_pre_run_state(self):
        at = AppTest.from_file("pages/segmentation.py", default_timeout=90)
        at.run()
        assert len(at.multiselect) >= 1  # feature selector
        assert len(at.button) >= 1  # run button

    def test_run_button_populates_results(self):
        at = AppTest.from_file("pages/segmentation.py", default_timeout=180)
        at.run()
        at.button[0].click()
        at.run()
        assert not at.exception
        # algo explorer radio + persona cards rendered
        assert len(at.radio) >= 1

    def test_session_state_keys_after_run(self):
        at = AppTest.from_file("pages/segmentation.py", default_timeout=180)
        at.run()
        at.button[0].click()
        at.run()
        assert at.session_state["seg_results"] is not None
        assert at.session_state["seg_profiles"] is not None


class TestAiInsightsPage:
    def test_mode_selector_renders(self):
        at = AppTest.from_file("pages/ai_insights.py", default_timeout=60)
        at.run()
        assert len(at.radio) >= 1

    def test_segment_mode_warning_without_data(self):
        at = AppTest.from_file("pages/ai_insights.py", default_timeout=60)
        at.run()
        # Fresh session → no seg_profiles → warning path
        assert len(at.warning) >= 1 or len(at.info) >= 1

    def test_churn_mode_switch(self):
        at = AppTest.from_file("pages/ai_insights.py", default_timeout=60)
        at.run()
        at.radio[0].set_value("🔮 Churn Model Report").run()
        assert not at.exception
