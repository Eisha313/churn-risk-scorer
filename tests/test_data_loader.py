"""Tests for the data loader module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from churn_risk_scorer.data_loader import DataLoader, load_customer_data


@pytest.fixture
def sample_csv_path():
    """Create a temporary CSV file with sample data."""
    data = """customer_id,tenure,monthly_charges,total_charges,churn
1001,12,50.0,600.0,No
1002,24,75.5,1812.0,Yes
1003,6,30.0,180.0,No
1004,36,100.0,3600.0,Yes
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(data)
        return f.name


@pytest.fixture
def sample_csv_missing_values():
    """Create a CSV with missing values."""
    data = """customer_id,tenure,monthly_charges,total_charges,churn
1001,12,50.0,,No
1002,24,75.5,1812.0,Yes
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(data)
        return f.name


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_load_valid_csv(self, sample_csv_path):
        """Test loading a valid CSV file."""
        loader = DataLoader(sample_csv_path)
        df = loader.load()

        assert len(df) == 4
        assert "customer_id" in df.columns
        assert "tenure" in df.columns

    def test_churn_conversion(self, sample_csv_path):
        """Test that churn column is converted to binary."""
        loader = DataLoader(sample_csv_path)
        df = loader.load()

        assert df["churn"].dtype == int
        assert set(df["churn"].unique()) == {0, 1}

    def test_missing_file_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        loader = DataLoader("/nonexistent/path.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        data = """customer_id,tenure\n1001,12"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(data)
            path = f.name

        loader = DataLoader(path)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load()

    def test_fill_missing_total_charges(self, sample_csv_missing_values):
        """Test that missing total_charges are calculated."""
        loader = DataLoader(sample_csv_missing_values)
        df = loader.load()

        # First row: 12 * 50.0 = 600.0
        assert df.loc[0, "total_charges"] == 600.0

    def test_get_feature_target_split(self, sample_csv_path):
        """Test feature/target splitting."""
        loader = DataLoader(sample_csv_path)
        loader.load()

        features, target = loader.get_feature_target_split()

        assert "churn" not in features.columns
        assert "customer_id" not in features.columns
        assert len(target) == 4
        assert target.name == "churn"

    def test_get_summary(self, sample_csv_path):
        """Test summary generation."""
        loader = DataLoader(sample_csv_path)
        loader.load()

        summary = loader.get_summary()

        assert summary["total_records"] == 4
        assert "churn_rate" in summary
        assert 0 <= summary["churn_rate"] <= 1


def test_convenience_function(sample_csv_path):
    """Test the load_customer_data convenience function."""
    df = load_customer_data(sample_csv_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
