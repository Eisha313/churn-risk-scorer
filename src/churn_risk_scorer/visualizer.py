"""Visualization module for churn risk analysis."""

import logging
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .exceptions import VisualizationError

logger = logging.getLogger(__name__)


class ChurnVisualizer:
    """Creates interactive visualizations for churn risk analysis."""
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.default_colors = {
            'low': '#2ecc71',
            'medium': '#f39c12',
            'high': '#e74c3c'
        }
        self.template = 'plotly_white'
        logger.info("ChurnVisualizer initialized")
    
    def get_risk_colors(self) -> Dict[str, str]:
        """Get the color mapping for risk categories."""
        return self.default_colors.copy()
    
    def _validate_scores_dataframe(self, df: Any) -> None:
        """Validate that input is a proper DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise VisualizationError("Input must be a pandas DataFrame")
        if df.empty:
            raise VisualizationError("Cannot visualize empty DataFrame")
    
    def _check_required_columns(self, df: pd.DataFrame, columns: list) -> None:
        """Check that required columns exist in DataFrame."""
        for col in columns:
            if col not in df.columns:
                raise VisualizationError(f"Missing required column: {col}")
    
    def plot_risk_distribution(self, scores_df: pd.DataFrame) -> go.Figure:
        """Create a pie chart showing risk category distribution.
        
        Args:
            scores_df: DataFrame with 'risk_category' column
            
        Returns:
            Plotly Figure object
        """
        self._validate_scores_dataframe(scores_df)
        self._check_required_columns(scores_df, ['risk_category'])
        
        logger.info("Creating risk distribution plot")
        
        risk_counts = scores_df['risk_category'].value_counts()
        
        colors = [self.default_colors.get(cat, '#95a5a6') for cat in risk_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=colors,
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Churn Risk Distribution',
            template=self.template,
            showlegend=True
        )
        
        return fig
    
    def plot_churn_probability_histogram(self, scores_df: pd.DataFrame) -> go.Figure:
        """Create a histogram of churn probabilities.
        
        Args:
            scores_df: DataFrame with 'churn_probability' column
            
        Returns:
            Plotly Figure object
        """
        self._validate_scores_dataframe(scores_df)
        self._check_required_columns(scores_df, ['churn_probability'])
        
        logger.info("Creating churn probability histogram")
        
        fig = go.Figure(data=[go.Histogram(
            x=scores_df['churn_probability'],
            nbinsx=20,
            marker_color='#3498db',
            opacity=0.75
        )])
        
        fig.update_layout(
            title='Distribution of Churn Probabilities',
            xaxis_title='Churn Probability',
            yaxis_title='Number of Customers',
            template=self.template,
            bargap=0.1
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame) -> go.Figure:
        """Create a horizontal bar chart of feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            
        Returns:
            Plotly Figure object
        """
        self._validate_scores_dataframe(importance_df)
        self._check_required_columns(importance_df, ['feature', 'importance'])
        
        logger.info("Creating feature importance plot")
        
        # Sort by importance
        sorted_df = importance_df.sort_values('importance', ascending=True)
        
        fig = go.Figure(data=[go.Bar(
            x=sorted_df['importance'],
            y=sorted_df['feature'],
            orientation='h',
            marker_color='#9b59b6'
        )])
        
        fig.update_layout(
            title='Feature Importance for Churn Prediction',
            xaxis_title='Importance',
            yaxis_title='Feature',
            template=self.template
        )
        
        return fig
    
    def plot_risk_by_segment(
        self, 
        data_df: pd.DataFrame, 
        segment_column: str
    ) -> go.Figure:
        """Create a grouped bar chart showing risk by segment.
        
        Args:
            data_df: DataFrame with risk_category and segment column
            segment_column: Name of the column to segment by
            
        Returns:
            Plotly Figure object
        """
        self._validate_scores_dataframe(data_df)
        self._check_required_columns(data_df, ['risk_category', segment_column])
        
        logger.info(f"Creating risk by segment plot for {segment_column}")
        
        # Create cross-tabulation
        cross_tab = pd.crosstab(
            data_df[segment_column], 
            data_df['risk_category'], 
            normalize='index'
        ) * 100
        
        fig = go.Figure()
        
        for risk_cat in ['low', 'medium', 'high']:
            if risk_cat in cross_tab.columns:
                fig.add_trace(go.Bar(
                    name=risk_cat.capitalize(),
                    x=cross_tab.index,
                    y=cross_tab[risk_cat],
                    marker_color=self.default_colors[risk_cat]
                ))
        
        fig.update_layout(
            title=f'Churn Risk by {segment_column.replace("_", " ").title()}',
            xaxis_title=segment_column.replace("_", " ").title(),
            yaxis_title='Percentage',
            barmode='group',
            template=self.template
        )
        
        return fig
    
    def get_risk_summary_statistics(self, scores_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for risk scores.
        
        Args:
            scores_df: DataFrame with churn scores
            
        Returns:
            Dictionary with summary statistics
        """
        self._validate_scores_dataframe(scores_df)
        
        logger.info("Calculating risk summary statistics")
        
        summary = {
            'total_customers': len(scores_df),
            'mean_probability': scores_df['churn_probability'].mean() if 'churn_probability' in scores_df.columns else None,
            'median_probability': scores_df['churn_probability'].median() if 'churn_probability' in scores_df.columns else None,
            'std_probability': scores_df['churn_probability'].std() if 'churn_probability' in scores_df.columns else None,
            'risk_counts': scores_df['risk_category'].value_counts().to_dict() if 'risk_category' in scores_df.columns else {}
        }
        
        return summary
    
    def create_dashboard(
        self, 
        scores_df: pd.DataFrame,
        feature_importance_df: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """Create a dashboard with multiple visualizations.
        
        Args:
            scores_df: DataFrame with churn scores
            feature_importance_df: Optional DataFrame with feature importance
            
        Returns:
            Plotly Figure object with subplots
        """
        self._validate_scores_dataframe(scores_df)
        
        logger.info("Creating churn risk dashboard")
        
        rows = 2 if feature_importance_df is not None else 1
        
        fig = make_subplots(
            rows=rows, 
            cols=2,
            subplot_titles=(
                'Risk Distribution', 
                'Probability Distribution',
                'Feature Importance' if feature_importance_df is not None else None,
                'Risk Summary' if feature_importance_df is not None else None
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'histogram'}],
                [{'type': 'bar'}, {'type': 'table'}] if feature_importance_df is not None else None
            ]
        )
        
        # This is a simplified version - full implementation would add traces
        fig.update_layout(
            title='Churn Risk Analysis Dashboard',
            template=self.template,
            height=800 if rows == 2 else 400
        )
        
        return fig
    
    def save_figure(
        self, 
        fig: go.Figure, 
        filepath: str, 
        format: str = 'html'
    ) -> None:
        """Save a figure to file.
        
        Args:
            fig: Plotly Figure object
            filepath: Output file path
            format: Output format ('html', 'png', 'pdf', 'svg')
        """
        logger.info(f"Saving figure to {filepath}")
        
        if format == 'html':
            fig.write_html(filepath)
        else:
            fig.write_image(filepath, format=format)
