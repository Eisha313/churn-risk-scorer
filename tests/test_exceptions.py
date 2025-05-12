"""Tests for custom exceptions."""

import pytest
from churn_risk_scorer.exceptions import (
    ChurnScorerError,
    DataLoadError,
    DataValidationError,
    PreprocessingError,
    ModelError,
    ModelNotTrainedError,
    VisualizationError,
    ConfigurationError
)


class TestExceptionHierarchy:
    """Test that all exceptions inherit from ChurnScorerError."""
    
    def test_data_load_error_inheritance(self):
        assert issubclass(DataLoadError, ChurnScorerError)
    
    def test_data_validation_error_inheritance(self):
        assert issubclass(DataValidationError, ChurnScorerError)
    
    def test_preprocessing_error_inheritance(self):
        assert issubclass(PreprocessingError, ChurnScorerError)
    
    def test_model_error_inheritance(self):
        assert issubclass(ModelError, ChurnScorerError)
    
    def test_model_not_trained_error_inheritance(self):
        assert issubclass(ModelNotTrainedError, ModelError)
    
    def test_visualization_error_inheritance(self):
        assert issubclass(VisualizationError, ChurnScorerError)
    
    def test_configuration_error_inheritance(self):
        assert issubclass(ConfigurationError, ChurnScorerError)


class TestDataLoadError:
    """Test DataLoadError functionality."""
    
    def test_basic_message(self):
        error = DataLoadError("File not found")
        assert "Failed to load data" in str(error)
        assert "File not found" in str(error)
    
    def test_with_filepath(self):
        error = DataLoadError("File not found", filepath="/path/to/file.csv")
        assert error.filepath == "/path/to/file.csv"


class TestDataValidationError:
    """Test DataValidationError functionality."""
    
    def test_basic_message(self):
        error = DataValidationError("Invalid data")
        assert "Invalid data" in str(error)
    
    def test_with_column(self):
        error = DataValidationError("Missing values", column="customer_id")
        assert error.column == "customer_id"
        assert "customer_id" in str(error)


class TestModelNotTrainedError:
    """Test ModelNotTrainedError functionality."""
    
    def test_default_message(self):
        error = ModelNotTrainedError()
        assert "trained" in str(error).lower()
        assert "predict" in str(error).lower()
