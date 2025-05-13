"""Pytest fixtures for churn risk scorer tests."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_customer_data() -> pd.DataFrame:
    """Generate sample customer data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples).round(2),
        'total_charges': np.random.uniform(100, 8000, n_samples).round(2),
        'contract_type': np.random.choice(['month-to-month', 'one_year', 'two_year'], n_samples),
        'payment_method': np.random.choice(['credit_card', 'bank_transfer', 'electronic_check'], n_samples),
        'num_support_tickets': np.random.randint(0, 10, n_samples),
        'churned': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })


@pytest.fixture
def sample_csv_file(sample_customer_data, tmp_path) -> Path:
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "test_customers.csv"
    sample_customer_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def empty_csv_file(tmp_path) -> Path:
    """Create an empty CSV file."""
    csv_path = tmp_path / "empty.csv"
    csv_path.touch()
    return csv_path


@pytest.fixture
def malformed_csv_file(tmp_path) -> Path:
    """Create a malformed CSV file."""
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text("col1,col2,col3\n1,2\n3,4,5,6\n")
    return csv_path


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Generate sample feature data for model testing."""
    np.random.seed(42)
    n_samples = 50
    
    return pd.DataFrame({
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'num_support_tickets': np.random.randint(0, 10, n_samples),
        'contract_type_one_year': np.random.choice([0, 1], n_samples),
        'contract_type_two_year': np.random.choice([0, 1], n_samples)
    })


@pytest.fixture
def sample_labels() -> pd.Series:
    """Generate sample labels for model testing."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], 50, p=[0.7, 0.3]), name='churned')


@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Create a temporary directory for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def mock_config():
    """Return mock configuration for testing."""
    return {
        'random_state': 42,
        'test_size': 0.2,
        'feature_columns': [
            'tenure_months',
            'monthly_charges',
            'total_charges',
            'num_support_tickets'
        ],
        'target_column': 'churned',
        'risk_thresholds': {
            'low': 0.3,
            'medium': 0.6,
            'high': 1.0
        }
    }
