"""Tests for model persistence module."""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from churn_risk_scorer.model_persistence import (
    ModelPersistence,
    ModelPersistenceError,
)


class TestModelPersistence:
    """Tests for ModelPersistence class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def persistence(self, temp_dir):
        """Create a ModelPersistence instance."""
        return ModelPersistence(base_path=temp_dir)
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.predict = MagicMock(return_value=[0, 1])
        model.predict_proba = MagicMock(return_value=[[0.8, 0.2], [0.3, 0.7]])
        return model
    
    def test_init_creates_directory(self, temp_dir):
        """Test that initialization creates the base directory."""
        path = Path(temp_dir) / "new_models"
        assert not path.exists()
        
        ModelPersistence(base_path=str(path))
        
        assert path.exists()
    
    def test_save_model(self, persistence, mock_model):
        """Test saving a model."""
        feature_names = ["feature1", "feature2", "feature3"]
        metrics = {"accuracy": 0.85, "auc": 0.92}
        
        model_dir = persistence.save_model(
            model=mock_model,
            model_name="test_model",
            feature_names=feature_names,
            metrics=metrics
        )
        
        assert model_dir.exists()
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "metadata.json").exists()
    
    def test_save_model_metadata_content(self, persistence, mock_model):
        """Test that saved metadata contains expected fields."""
        feature_names = ["f1", "f2"]
        metrics = {"accuracy": 0.9}
        extra = {"custom_field": "value"}
        
        model_dir = persistence.save_model(
            model=mock_model,
            model_name="test_model",
            feature_names=feature_names,
            metrics=metrics,
            extra_metadata=extra
        )
        
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        assert metadata["model_name"] == "test_model"
        assert metadata["feature_names"] == feature_names
        assert metadata["num_features"] == 2
        assert metadata["metrics"] == metrics
        assert metadata["extra"] == extra
        assert "timestamp" in metadata
        assert "created_at" in metadata
    
    def test_load_model(self, persistence, mock_model):
        """Test loading a saved model."""
        feature_names = ["f1", "f2"]
        
        persistence.save_model(
            model=mock_model,
            model_name="load_test",
            feature_names=feature_names
        )
        
        loaded_model, metadata = persistence.load_model("load_test")
        
        assert loaded_model is not None
        assert metadata["model_name"] == "load_test"
        assert metadata["feature_names"] == feature_names
    
    def test_load_model_not_found(self, persistence):
        """Test loading a non-existent model raises error."""
        with pytest.raises(ModelPersistenceError, match="not found"):
            persistence.load_model("nonexistent_model")
    
    def test_load_specific_version(self, persistence, mock_model):
        """Test loading a specific model version."""
        # Save first version
        model_dir1 = persistence.save_model(
            model=mock_model,
            model_name="versioned",
            feature_names=["f1"]
        )
        version1 = model_dir1.name.split("_")[-2] + "_" + model_dir1.name.split("_")[-1]
        
        # Load specific version
        _, metadata = persistence.load_model("versioned", version=version1)
        assert metadata["model_name"] == "versioned"
    
    def test_list_models_empty(self, persistence):
        """Test listing models when none exist."""
        models = persistence.list_models()
        assert models == []
    
    def test_list_models(self, persistence, mock_model):
        """Test listing saved models."""
        persistence.save_model(mock_model, "model_a", ["f1"])
        persistence.save_model(mock_model, "model_b", ["f2"])
        persistence.save_model(mock_model, "model_a", ["f1", "f2"])
        
        all_models = persistence.list_models()
        assert len(all_models) == 3
        
        model_a_only = persistence.list_models(model_name="model_a")
        assert len(model_a_only) == 2
        assert all(m["name"] == "model_a" for m in model_a_only)
    
    def test_delete_model(self, persistence, mock_model):
        """Test deleting a model."""
        model_dir = persistence.save_model(
            mock_model, "delete_test", ["f1"]
        )
        version = model_dir.name.replace("delete_test_", "")
        
        assert model_dir.exists()
        
        result = persistence.delete_model("delete_test", version)
        
        assert result is True
        assert not model_dir.exists()
    
    def test_delete_nonexistent_model(self, persistence):
        """Test deleting a non-existent model returns False."""
        result = persistence.delete_model("nonexistent", "20240101_000000")
        assert result is False
    
    def test_latest_model_loading(self, persistence, mock_model):
        """Test that loading without version gets the latest model."""
        import time
        
        # Save two versions
        persistence.save_model(
            mock_model, "latest_test", ["f1"],
            metrics={"accuracy": 0.8}
        )
        
        time.sleep(0.1)  # Ensure different timestamp
        
        persistence.save_model(
            mock_model, "latest_test", ["f1", "f2"],
            metrics={"accuracy": 0.9}
        )
        
        # Load latest (should be second one)
        _, metadata = persistence.load_model("latest_test")
        
        assert metadata["num_features"] == 2
        assert metadata["metrics"]["accuracy"] == 0.9


class TestModelPersistenceError:
    """Tests for ModelPersistenceError."""
    
    def test_error_inheritance(self):
        """Test that ModelPersistenceError inherits from ChurnRiskScorerError."""
        from churn_risk_scorer.exceptions import ChurnRiskScorerError
        
        error = ModelPersistenceError("test error")
        assert isinstance(error, ChurnRiskScorerError)
    
    def test_error_message(self):
        """Test error message is preserved."""
        error = ModelPersistenceError("custom message")
        assert str(error) == "custom message"
