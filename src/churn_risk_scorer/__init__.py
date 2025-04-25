"""
Churn Risk Scorer - Customer churn prediction utility.

A machine learning tool for identifying customers at risk of churning
by analyzing behavior data and generating risk scores.
"""

from .scorer import ChurnRiskScorer
from .preprocessor import DataPreprocessor
from .visualizer import ChurnVisualizer

__version__ = "0.1.0"
__all__ = ["ChurnRiskScorer", "DataPreprocessor", "ChurnVisualizer"]
