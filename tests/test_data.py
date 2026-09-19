"""Tests for data loading and generation."""

from __future__ import annotations

import pandas as pd

from src.data.generator import (
    FEATURE_METADATA,
    NUMERIC_FEATURES,
    TARGET_COL,
    generate_customer_dataset,
)


class TestGenerateCustomerDataset:
    def test_default_shape(self):
        df = generate_customer_dataset()
        assert df.shape == (1500, 11), "Expected 1500 rows and 11 columns"

    def test_custom_size(self):
        df = generate_customer_dataset(n_samples=200)
        assert len(df) == 200

    def test_all_feature_columns_present(self):
        df = generate_customer_dataset(n_samples=50)
        for col in NUMERIC_FEATURES:
            assert col in df.columns, f"Missing feature column: {col}"
        assert TARGET_COL in df.columns

    def test_target_is_binary(self):
        df = generate_customer_dataset(n_samples=200)
        assert set(df[TARGET_COL].unique()).issubset({0, 1})

    def test_churn_rate_realistic(self):
        df = generate_customer_dataset(n_samples=1000, random_state=0)
        rate = df[TARGET_COL].mean()
        assert 0.10 <= rate <= 0.45, f"Churn rate {rate:.2%} outside expected range"

    def test_feature_ranges(self):
        df = generate_customer_dataset(n_samples=200)
        for feat, meta in FEATURE_METADATA.items():
            assert df[feat].min() >= meta["min"], f"{feat} below min"
            assert df[feat].max() <= meta["max"], f"{feat} above max"

    def test_reproducible(self):
        df1 = generate_customer_dataset(random_state=7)
        df2 = generate_customer_dataset(random_state=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_customer_dataset(random_state=1)
        df2 = generate_customer_dataset(random_state=2)
        assert not df1.equals(df2)
