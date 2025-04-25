"""
Main churn risk scoring module with logistic regression model.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .preprocessor import DataPreprocessor
from .visualizer import ChurnVisualizer


class ChurnRiskScorer:
    """
    Customer churn risk scorer using logistic regression.
    
    Provides methods to load data, train a model, predict churn risk,
    and generate visualizations.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the churn risk scorer.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.preprocessor = DataPreprocessor()
        self.visualizer = ChurnVisualizer()
        self.model: Optional[LogisticRegression] = None
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self._is_trained = False
    
    def load_data(self, filepath: str) -> 'ChurnRiskScorer':
        """
        Load customer data from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            
        Returns:
            Self for method chaining.
        """
        self.data = self.preprocessor.load_csv(filepath)
        return self
    
    def train(self, test_size: float = 0.2, target_column: str = 'churn') -> Dict[str, float]:
        """
        Train the logistic regression model.
        
        Args:
            test_size: Fraction of data to use for testing.
            target_column: Name of the target column.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Preprocess the data
        X, y = self.preprocessor.preprocess(self.data, target_column=target_column, fit=True)
        
        if y is None:
            raise ValueError(f"Target column '{target_column}' not found in data.")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Train logistic regression model
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        self.model.fit(self.X_train, self.y_train)
        self._is_trained = True
        
        # Evaluate model
        metrics = self._evaluate()
        print(f"Model trained successfully!")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
        
        return metrics
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test data."""
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(self.y_test, y_prob)
        }
    
    def predict_risk(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict churn risk scores for customers.
        
        Args:
            data: Optional DataFrame with customer data. Uses loaded data if None.
            
        Returns:
            DataFrame with customer IDs and risk scores.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if data is None:
            data = self.data
        
        if data is None:
            raise ValueError("No data provided.")
        
        # Preprocess without fitting
        X, _ = self.preprocessor.preprocess(data, fit=False)
        
        # Get probability of churn (class 1)
        risk_scores = self.model.predict_proba(X)[:, 1]
        
        # Create result DataFrame
        result = pd.DataFrame({
            'customer_id': data.get('customer_id', range(len(data))),
            'churn_risk_score': risk_scores,
            'risk_level': pd.cut(risk_scores, 
                                bins=[0, 0.3, 0.6, 1.0],
                                labels=['Low', 'Medium', 'High'])
        })
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with features and their importance scores.
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        importance = pd.DataFrame({
            'feature': self.preprocessor.feature_columns,
            'importance': np.abs(self.model.coef_[0])
        })
        
        return importance.sort_values('importance', ascending=False)
    
    def plot_risk_distribution(self, show: bool = True) -> Any:
        """
        Generate interactive visualization of risk score distribution.
        
        Args:
            show: Whether to display the plot.
            
        Returns:
            Plotly figure object.
        """
        risk_df = self.predict_risk()
        return self.visualizer.plot_risk_distribution(risk_df, show=show)
    
    def plot_feature_importance(self, top_n: int = 10, show: bool = True) -> Any:
        """
        Generate interactive visualization of feature importance.
        
        Args:
            top_n: Number of top features to display.
            show: Whether to display the plot.
            
        Returns:
            Plotly figure object.
        """
        importance_df = self.get_feature_importance().head(top_n)
        return self.visualizer.plot_feature_importance(importance_df, show=show)
