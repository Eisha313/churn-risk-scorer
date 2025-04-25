"""
Interactive visualization module using Plotly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any


class ChurnVisualizer:
    """Generate interactive visualizations for churn risk analysis."""
    
    def __init__(self):
        self.color_scheme = {
            'Low': '#2ecc71',
            'Medium': '#f39c12', 
            'High': '#e74c3c'
        }
    
    def plot_risk_distribution(self, risk_df: pd.DataFrame, show: bool = True) -> Any:
        """
        Create histogram showing distribution of churn risk scores.
        
        Args:
            risk_df: DataFrame with churn_risk_score and risk_level columns.
            show: Whether to display the plot.
            
        Returns:
            Plotly figure object.
        """
        fig = px.histogram(
            risk_df,
            x='churn_risk_score',
            color='risk_level',
            nbins=30,
            title='Customer Churn Risk Distribution',
            labels={
                'churn_risk_score': 'Churn Risk Score',
                'risk_level': 'Risk Level',
                'count': 'Number of Customers'
            },
            color_discrete_map=self.color_scheme
        )
        
        fig.update_layout(
            xaxis_title='Churn Risk Score',
            yaxis_title='Number of Customers',
            legend_title='Risk Level',
            template='plotly_white',
            bargap=0.1
        )
        
        # Add vertical lines for risk thresholds
        fig.add_vline(x=0.3, line_dash='dash', line_color='gray',
                      annotation_text='Low/Medium Threshold')
        fig.add_vline(x=0.6, line_dash='dash', line_color='gray',
                      annotation_text='Medium/High Threshold')
        
        if show:
            fig.show()
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                                show: bool = True) -> Any:
        """
        Create bar chart showing feature importance.
        
        Args:
            importance_df: DataFrame with feature and importance columns.
            show: Whether to display the plot.
            
        Returns:
            Plotly figure object.
        """
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top Features Influencing Churn Risk',
            labels={
                'importance': 'Feature Importance',
                'feature': 'Feature'
            },
            color='importance',
            color_continuous_scale='RdYlGn_r'
        )
        
        fig.update_layout(
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white',
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        if show:
            fig.show()
        
        return fig
    
    def plot_risk_by_segment(self, risk_df: pd.DataFrame, 
                             segment_column: str,
                             show: bool = True) -> Any:
        """
        Create box plot showing risk distribution by customer segment.
        
        Args:
            risk_df: DataFrame with churn_risk_score column.
            segment_column: Column name to segment by.
            show: Whether to display the plot.
            
        Returns:
            Plotly figure object.
        """
        fig = px.box(
            risk_df,
            x=segment_column,
            y='churn_risk_score',
            title=f'Churn Risk by {segment_column.replace("_", " ").title()}',
            labels={
                'churn_risk_score': 'Churn Risk Score',
                segment_column: segment_column.replace('_', ' ').title()
            },
            color=segment_column
        )
        
        fig.update_layout(
            template='plotly_white',
            showlegend=False
        )
        
        if show:
            fig.show()
        
        return fig
    
    def plot_risk_pie(self, risk_df: pd.DataFrame, show: bool = True) -> Any:
        """
        Create pie chart showing proportion of customers in each risk level.
        
        Args:
            risk_df: DataFrame with risk_level column.
            show: Whether to display the plot.
            
        Returns:
            Plotly figure object.
        """
        risk_counts = risk_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['risk_level', 'count']
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts['risk_level'],
            values=risk_counts['count'],
            hole=0.4,
            marker_colors=[self.color_scheme.get(level, '#gray') 
                          for level in risk_counts['risk_level']]
        )])
        
        fig.update_layout(
            title='Customer Distribution by Risk Level',
            template='plotly_white'
        )
        
        if show:
            fig.show()
        
        return fig
