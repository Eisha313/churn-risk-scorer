"""Tests for the visualizer module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from churn_risk_scorer.visualizer import ChurnVisualizer
from churn_risk_scorer.exceptions import VisualizationError


class TestChurnVisualizer:
    """Test cases for ChurnVisualizer class."""
    
    @pytest.fixture
    def visualizer(self):
        """Create a ChurnVisualizer instance."""
        return ChurnVisualizer()
    
    @pytest.fixture
    def sample_scores(self):
        """Create sample churn scores."""
        return pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'churn_probability': [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.15, 0.3, 0.6, 0.9],
            'risk_category': ['low', 'low', 'medium', 'medium', 'high', 'high', 'low', 'medium', 'high', 'high']
        })
    
    @pytest.fixture
    def sample_feature_importance(self):
        """Create sample feature importance data."""
        return pd.DataFrame({
            'feature': ['tenure', 'monthly_charges', 'contract_type', 'payment_method', 'total_charges'],
            'importance': [0.35, 0.25, 0.20, 0.12, 0.08]
        })
    
    @pytest.fixture
    def sample_customer_data(self):
        """Create sample customer data for analysis."""
        return pd.DataFrame({
            'customer_id': range(1, 101),
            'tenure': np.random.randint(1, 72, 100),
            'monthly_charges': np.random.uniform(20, 100, 100),
            'total_charges': np.random.uniform(100, 5000, 100),
            'contract_type': np.random.choice(['month-to-month', 'one_year', 'two_year'], 100),
            'churn': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        })
    
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initializes correctly."""
        assert visualizer is not None
        assert hasattr(visualizer, 'default_colors')
    
    @patch('churn_risk_scorer.visualizer.go.Figure')
    def test_plot_risk_distribution(self, mock_figure, visualizer, sample_scores):
        """Test risk distribution plot creation."""
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        result = visualizer.plot_risk_distribution(sample_scores)
        
        assert result is not None
        mock_figure.assert_called()
    
    @patch('churn_risk_scorer.visualizer.go.Figure')
    def test_plot_churn_probability_histogram(self, mock_figure, visualizer, sample_scores):
        """Test churn probability histogram creation."""
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        result = visualizer.plot_churn_probability_histogram(sample_scores)
        
        assert result is not None
        mock_figure.assert_called()
    
    @patch('churn_risk_scorer.visualizer.go.Figure')
    def test_plot_feature_importance(self, mock_figure, visualizer, sample_feature_importance):
        """Test feature importance plot creation."""
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        result = visualizer.plot_feature_importance(sample_feature_importance)
        
        assert result is not None
        mock_figure.assert_called()
    
    def test_plot_risk_distribution_empty_data(self, visualizer):
        """Test risk distribution with empty data raises error."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(VisualizationError, match="empty"):
            visualizer.plot_risk_distribution(empty_df)
    
    def test_plot_risk_distribution_missing_column(self, visualizer):
        """Test risk distribution with missing required column."""
        invalid_df = pd.DataFrame({'customer_id': [1, 2, 3]})
        
        with pytest.raises(VisualizationError, match="Missing required column"):
            visualizer.plot_risk_distribution(invalid_df)
    
    def test_plot_feature_importance_missing_column(self, visualizer):
        """Test feature importance with missing required column."""
        invalid_df = pd.DataFrame({'feature': ['a', 'b']})
        
        with pytest.raises(VisualizationError, match="Missing required column"):
            visualizer.plot_feature_importance(invalid_df)
    
    @patch('churn_risk_scorer.visualizer.go.Figure')
    def test_plot_risk_by_segment(self, mock_figure, visualizer, sample_customer_data, sample_scores):
        """Test risk by segment plot creation."""
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        # Add risk category to customer data
        sample_customer_data['risk_category'] = np.random.choice(
            ['low', 'medium', 'high'], len(sample_customer_data)
        )
        
        result = visualizer.plot_risk_by_segment(
            sample_customer_data, 
            segment_column='contract_type'
        )
        
        assert result is not None
    
    def test_get_risk_summary_statistics(self, visualizer, sample_scores):
        """Test risk summary statistics calculation."""
        summary = visualizer.get_risk_summary_statistics(sample_scores)
        
        assert 'total_customers' in summary
        assert 'mean_probability' in summary
        assert 'risk_counts' in summary
        assert summary['total_customers'] == 10
    
    def test_get_risk_summary_statistics_empty_data(self, visualizer):
        """Test risk summary with empty data raises error."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(VisualizationError, match="empty"):
            visualizer.get_risk_summary_statistics(empty_df)
    
    @patch('churn_risk_scorer.visualizer.go.Figure')
    def test_create_dashboard(self, mock_figure, visualizer, sample_scores, sample_feature_importance):
        """Test dashboard creation with multiple plots."""
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        result = visualizer.create_dashboard(
            scores_df=sample_scores,
            feature_importance_df=sample_feature_importance
        )
        
        assert result is not None
    
    @patch('churn_risk_scorer.visualizer.go.Figure')
    def test_save_figure_html(self, mock_figure, visualizer, sample_scores, tmp_path):
        """Test saving figure to HTML file."""
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        output_path = tmp_path / "test_plot.html"
        
        fig = visualizer.plot_risk_distribution(sample_scores)
        visualizer.save_figure(fig, str(output_path), format='html')
        
        mock_fig_instance.write_html.assert_called_once()
    
    @patch('churn_risk_scorer.visualizer.go.Figure')
    def test_save_figure_png(self, mock_figure, visualizer, sample_scores, tmp_path):
        """Test saving figure to PNG file."""
        mock_fig_instance = MagicMock()
        mock_figure.return_value = mock_fig_instance
        
        output_path = tmp_path / "test_plot.png"
        
        fig = visualizer.plot_risk_distribution(sample_scores)
        visualizer.save_figure(fig, str(output_path), format='png')
        
        mock_fig_instance.write_image.assert_called_once()
    
    def test_validate_scores_dataframe(self, visualizer, sample_scores):
        """Test dataframe validation method."""
        # Should not raise any exception
        visualizer._validate_scores_dataframe(sample_scores)
    
    def test_validate_scores_dataframe_invalid(self, visualizer):
        """Test dataframe validation with invalid input."""
        with pytest.raises(VisualizationError):
            visualizer._validate_scores_dataframe("not a dataframe")
    
    def test_color_mapping_for_risk_categories(self, visualizer):
        """Test that risk categories have proper color mapping."""
        colors = visualizer.get_risk_colors()
        
        assert 'low' in colors
        assert 'medium' in colors
        assert 'high' in colors
