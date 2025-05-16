"""Tests for the data preprocessing module."""

import pytest
import numpy as np
import pandas as pd

from churn_risk_scorer.preprocessor import DataPreprocessor
from churn_risk_scorer.exceptions import DataValidationError, PreprocessingError


@pytest.fixture
def sample_customer_data():
    """Create sample customer data for testing."""
    return pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'tenure': [12, 24, 6, 36, 18],
        'monthly_charges': [50.0, 75.0, 30.0, 100.0, 60.0],
        'total_charges': [600.0, 1800.0, 180.0, 3600.0, 1080.0],
        'contract_type': ['monthly', 'annual', 'monthly', 'annual', 'monthly'],
        'payment_method': ['credit_card', 'bank_transfer', 'credit_card', 'credit_card', 'bank_transfer'],
        'churn': [1, 0, 1, 0, 0]
    })


@pytest.fixture
def sample_data_with_nulls():
    """Create sample data with missing values."""
    return pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004'],
        'tenure': [12, None, 6, 36],
        'monthly_charges': [50.0, 75.0, None, 100.0],
        'contract_type': ['monthly', 'annual', None, 'annual'],
        'churn': [1, 0, 1, 0]
    })


@pytest.fixture
def preprocessor():
    """Create a preprocessor with test configuration."""
    return DataPreprocessor(
        numeric_columns=['tenure', 'monthly_charges', 'total_charges'],
        categorical_columns=['contract_type', 'payment_method'],
        target_column='churn'
    )


class TestDataPreprocessorInit:
    """Tests for DataPreprocessor initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        preprocessor = DataPreprocessor()
        
        assert preprocessor.numeric_columns == DataPreprocessor.DEFAULT_NUMERIC_COLUMNS
        assert preprocessor.categorical_columns == DataPreprocessor.DEFAULT_CATEGORICAL_COLUMNS
        assert preprocessor.target_column == 'churn'
        assert not preprocessor.is_fitted
    
    def test_init_with_custom_columns(self):
        """Test initialization with custom column lists."""
        numeric = ['col1', 'col2']
        categorical = ['cat1', 'cat2']
        
        preprocessor = DataPreprocessor(
            numeric_columns=numeric,
            categorical_columns=categorical,
            target_column='target'
        )
        
        assert preprocessor.numeric_columns == numeric
        assert preprocessor.categorical_columns == categorical
        assert preprocessor.target_column == 'target'


class TestDataPreprocessorValidation:
    """Tests for data validation."""
    
    def test_validate_empty_dataframe(self, preprocessor):
        """Test validation fails for empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="empty"):
            preprocessor.fit_transform(empty_df)
    
    def test_validate_missing_all_columns(self, preprocessor):
        """Test validation fails when no expected columns exist."""
        df = pd.DataFrame({'unknown_col': [1, 2, 3]})
        
        with pytest.raises(DataValidationError, match="No configured"):
            preprocessor.fit_transform(df)


class TestDataPreprocessorFitTransform:
    """Tests for fit_transform functionality."""
    
    def test_fit_transform_basic(self, preprocessor, sample_customer_data):
        """Test basic fit_transform operation."""
        X, y = preprocessor.fit_transform(sample_customer_data)
        
        assert preprocessor.is_fitted
        assert len(X) == len(sample_customer_data)
        assert y is not None
        assert len(y) == len(sample_customer_data)
        assert 'churn' not in X.columns
    
    def test_fit_transform_scales_numeric(self, preprocessor, sample_customer_data):
        """Test that numeric columns are scaled."""
        X, _ = preprocessor.fit_transform(sample_customer_data)
        
        # After standard scaling, mean should be ~0 and std ~1
        for col in ['tenure', 'monthly_charges', 'total_charges']:
            if col in X.columns:
                assert abs(X[col].mean()) < 0.1
                assert abs(X[col].std() - 1.0) < 0.5
    
    def test_fit_transform_encodes_categorical(self, preprocessor, sample_customer_data):
        """Test that categorical columns are encoded as integers."""
        X, _ = preprocessor.fit_transform(sample_customer_data)
        
        for col in ['contract_type', 'payment_method']:
            if col in X.columns:
                assert X[col].dtype in [np.int32, np.int64]
    
    def test_fit_transform_without_target(self, preprocessor, sample_customer_data):
        """Test fit_transform when excluding target."""
        X, y = preprocessor.fit_transform(sample_customer_data, include_target=False)
        
        assert y is None
        # Original target column may still be in X if not extracted
    
    def test_fit_transform_handles_missing_values(self, preprocessor, sample_data_with_nulls):
        """Test that missing values are handled."""
        X, _ = preprocessor.fit_transform(sample_data_with_nulls)
        
        # Check no null values remain in processed columns
        for col in X.columns:
            if col in preprocessor.numeric_columns or col in preprocessor.categorical_columns:
                assert not X[col].isnull().any()


class TestDataPreprocessorTransform:
    """Tests for transform functionality."""
    
    def test_transform_without_fit_raises_error(self, preprocessor, sample_customer_data):
        """Test that transform without fit raises error."""
        with pytest.raises(PreprocessingError, match="not been fitted"):
            preprocessor.transform(sample_customer_data)
    
    def test_transform_after_fit(self, preprocessor, sample_customer_data):
        """Test transform works after fitting."""
        # First fit
        preprocessor.fit_transform(sample_customer_data)
        
        # Create new data
        new_data = pd.DataFrame({
            'customer_id': ['C006'],
            'tenure': [24],
            'monthly_charges': [80.0],
            'total_charges': [1920.0],
            'contract_type': ['annual'],
            'payment_method': ['credit_card']
        })
        
        X, _ = preprocessor.transform(new_data)
        
        assert len(X) == 1
        assert 'tenure' in X.columns


class TestDataPreprocessorFeatureNames:
    """Tests for feature name extraction."""
    
    def test_get_feature_names(self, preprocessor, sample_customer_data):
        """Test getting feature names."""
        features = preprocessor.get_feature_names(sample_customer_data)
        
        assert 'tenure' in features
        assert 'monthly_charges' in features
        assert 'contract_type' in features
        assert 'churn' not in features  # target should not be in features
