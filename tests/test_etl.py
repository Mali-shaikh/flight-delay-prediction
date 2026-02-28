"""
Unit Tests for ETL Pipeline
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.extract import create_sample_data
from etl.transform import clean_data, engineer_features, select_features


@pytest.fixture
def raw_df():
    """Generate a small sample DataFrame for testing."""
    return create_sample_data(n=500)


class TestExtract:
    def test_sample_data_shape(self, raw_df):
        assert len(raw_df) == 500
        assert "ARR_DELAY" in raw_df.columns
        assert "DEP_DELAY" in raw_df.columns

    def test_sample_data_columns(self, raw_df):
        required_cols = ["YEAR", "MONTH", "DAY_OF_WEEK", "OP_CARRIER",
                         "ORIGIN", "DEST", "ARR_DELAY", "CANCELLED", "DISTANCE"]
        for col in required_cols:
            assert col in raw_df.columns, f"Missing column: {col}"


class TestTransform:
    def test_clean_removes_cancelled(self, raw_df):
        cleaned = clean_data(raw_df)
        assert cleaned["CANCELLED"].sum() == 0

    def test_clean_no_missing_arr_delay(self, raw_df):
        cleaned = clean_data(raw_df)
        assert cleaned["ARR_DELAY"].isna().sum() == 0

    def test_engineer_creates_delayed_column(self, raw_df):
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)
        assert "DELAYED" in engineered.columns
        assert set(engineered["DELAYED"].unique()).issubset({0, 1})

    def test_engineer_dep_hour_range(self, raw_df):
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)
        assert engineered["DEP_HOUR"].between(0, 23).all()

    def test_engineer_is_weekend_binary(self, raw_df):
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)
        assert set(engineered["IS_WEEKEND"].unique()).issubset({0, 1})

    def test_select_features_has_target(self, raw_df):
        cleaned = clean_data(raw_df)
        engineered = engineer_features(cleaned)
        selected = select_features(engineered)
        assert "DELAYED" in selected.columns

    def test_no_inf_values(self, raw_df):
        from etl.transform import transform
        processed = transform(raw_df)
        assert not np.isinf(processed.select_dtypes(include=np.number).values).any()
