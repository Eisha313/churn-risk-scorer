"""Tests for the ChurnScorer class."""

import pytest
import numpy as np
import pandas as pd

from churn_risk_scorer.scorer import ChurnScorer
from churn_risk_scorer.exceptions import ModelNotTrainedError, ScoringError


class TestChurnScorerInit:
    """Tests for ChurnScorer initialization."""
    
    def test_default_initialization(self):
        """Test scorer initializes with default parameters."""
        scorer = ChurnScorer()
        assert scorer.is_trained is False
        assert scorer.feature_names is None
        assert scorer.training_metrics is None
    
    def test_custom_initialization(self):
        """Test scorer initializes with custom parameters."""
        scorer = ChurnScorer(random_state=123, max_iter=500, C=0.5)
        assert scorer.is_trained is False


class TestChurnScorerTraining:
    """Tests for ChurnScorer training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        # Create target correlated with features
        prob = 1 / (1 + np.exp(-(X['feature1'] + X['feature2'])))
        y = pd.Series((prob > 0.5).astype(int))
        
        return X, y
    
    def test_train_with_validation(self, sample_data):
        """Test training with validation split."""
        X, y = sample_data
        scorer = ChurnScorer(random_state=42)
        
        metrics = scorer.train(X, y, test_size=0.2, validate=True)
        
        assert scorer.is_trained is True
        assert scorer.feature_names == ['feature1', 'feature2', 'feature3']
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['val_accuracy'] <= 1
    
    def test_train_without_validation(self, sample_data):
        """Test training without validation."""
        X, y = sample_data
        scorer = ChurnScorer(random_state=42)
        
        metrics = scorer.train(X, y, validate=False)
        
        assert scorer.is_trained is True
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' not in metrics
    
    def test_train_stores_metrics(self, sample_data):
        """Test that training metrics are stored."""
        X, y = sample_data
        scorer = ChurnScorer(random_state=42)
        
        metrics = scorer.train(X, y)
        
        assert scorer.training_metrics == metrics


class TestChurnScorerPrediction:
    """Tests for ChurnScorer prediction."""
    
    @pytest.fixture
    def trained_scorer(self):
        """Create a trained scorer."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        scorer = ChurnScorer(random_state=42)
        scorer.train(X, y, validate=False)
        
        return scorer
    
    def test_predict_proba_untrained(self):
        """Test that predict_proba raises error when untrained."""
        scorer = ChurnScorer()
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ModelNotTrainedError):
            scorer.predict_proba(X)
    
    def test_predict_proba_trained(self, trained_scorer):
        """Test predict_proba with trained model."""
        X = pd.DataFrame({
            'feature1': [0.5, -0.5, 1.0],
            'feature2': [0.3, 0.7, -0.2]
        })
        
        proba = trained_scorer.predict_proba(X)
        
        assert len(proba) == 3
        assert all(0 <= p <= 1 for p in proba)
    
    def test_predict_binary(self, trained_scorer):
        """Test binary prediction."""
        X = pd.DataFrame({
            'feature1': [0.5, -0.5, 1.0],
            'feature2': [0.3, 0.7, -0.2]
        })
        
        predictions = trained_scorer.predict(X)
        
        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_custom_threshold(self, trained_scorer):
        """Test prediction with custom threshold."""
        X = pd.DataFrame({
            'feature1': [0.5],
            'feature2': [0.3]
        })
        
        pred_low = trained_scorer.predict(X, threshold=0.1)
        pred_high = trained_scorer.predict(X, threshold=0.9)
        
        # With low threshold, more likely to predict churn
        # With high threshold, less likely to predict churn
        assert pred_low[0] >= pred_high[0] or pred_low[0] == pred_high[0]


class TestChurnScorerScoring:
    """Tests for ChurnScorer customer scoring."""
    
    @pytest.fixture
    def trained_scorer(self):
        """Create a trained scorer."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        scorer = ChurnScorer(random_state=42)
        scorer.train(X, y, validate=False)
        
        return scorer
    
    def test_score_customers(self, trained_scorer):
        """Test customer scoring."""
        X = pd.DataFrame({
            'feature1': [0.5, -0.5, 1.0],
            'feature2': [0.3, 0.7, -0.2]
        })
        
        result = trained_scorer.score_customers(X)
        
        assert len(result) == 3
        assert 'customer_id' in result.columns
        assert 'churn_probability' in result.columns
        assert 'risk_level' in result.columns
        assert all(level in ['Low', 'Medium', 'High'] for level in result['risk_level'])
    
    def test_score_customers_with_ids(self, trained_scorer):
        """Test customer scoring with custom IDs."""
        X = pd.DataFrame({
            'feature1': [0.5, -0.5, 1.0],
            'feature2': [0.3, 0.7, -0.2]
        })
        customer_ids = pd.Series(['A001', 'A002', 'A003'])
        
        result = trained_scorer.score_customers(X, customer_ids=customer_ids)
        
        assert list(result['customer_id']) == ['A001', 'A002', 'A003']


class TestChurnScorerFeatureImportance:
    """Tests for feature importance."""
    
    @pytest.fixture
    def trained_scorer(self):
        """Create a trained scorer."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        scorer = ChurnScorer(random_state=42)
        scorer.train(X, y, validate=False)
        
        return scorer
    
    def test_feature_importance_untrained(self):
        """Test that get_feature_importance raises error when untrained."""
        scorer = ChurnScorer()
        
        with pytest.raises(ModelNotTrainedError):
            scorer.get_feature_importance()
    
    def test_feature_importance_trained(self, trained_scorer):
        """Test feature importance with trained model."""
        importance = trained_scorer.get_feature_importance()
        
        assert len(importance) == 2
        assert 'feature' in importance.columns
        assert 'coefficient' in importance.columns
        assert 'abs_importance' in importance.columns
        # Should be sorted by absolute importance
        assert importance['abs_importance'].is_monotonic_decreasing
