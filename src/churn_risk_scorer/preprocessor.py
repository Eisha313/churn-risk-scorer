"""
Data preprocessing utilities for customer churn data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


class DataPreprocessor:
    """Handles loading and preprocessing of customer data for churn prediction."""
    
    def __init__(self):
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self._fitted = False
        self._column_means: Optional[pd.Series] = None
        self._column_stds: Optional[pd.Series] = None
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load customer data from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            
        Returns:
            DataFrame containing the loaded data.
        """
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    
    def preprocess(self, df: pd.DataFrame, target_column: str = 'churn',
                   fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess the data for model training or prediction.
        
        Args:
            df: Input DataFrame with customer data.
            target_column: Name of the target variable column.
            fit: Whether to fit the preprocessor (True for training).
            
        Returns:
            Tuple of (features array, target array or None).
        """
        df = df.copy()
        
        # Identify column types
        if fit:
            self._identify_columns(df, target_column)
        
        # Handle missing values
        df = self._handle_missing(df)
        
        # Encode categorical variables
        df = self._encode_categoricals(df)
        
        # Extract target if present
        y = None
        if target_column in df.columns:
            y = df[target_column].values
            df = df.drop(columns=[target_column])
        
        # Select feature columns (exclude IDs)
        feature_cols = [c for c in df.columns 
                       if c not in ['customer_id', 'id'] and df[c].dtype in ['int64', 'float64']]
        
        if fit:
            self.feature_columns = feature_cols
        
        X = df[self.feature_columns].values.astype(np.float64)
        
        # Normalize numeric features
        X = self._normalize(X, fit=fit)
        
        if fit:
            self._fitted = True
        
        return X, y
    
    def _identify_columns(self, df: pd.DataFrame, target_column: str) -> None:
        """Identify categorical and numeric columns."""
        exclude = ['customer_id', 'id', target_column]
        
        self.categorical_columns = [
            c for c in df.select_dtypes(include=['object', 'category']).columns
            if c not in exclude
        ]
        self.numeric_columns = [
            c for c in df.select_dtypes(include=['int64', 'float64']).columns
            if c not in exclude
        ]
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill numeric columns with median
        for col in self.numeric_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'unknown')
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables."""
        for col in self.categorical_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        return df
    
    def _normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features using z-score normalization."""
        if fit:
            self._column_means = np.mean(X, axis=0)
            self._column_stds = np.std(X, axis=0)
            # Avoid division by zero
            self._column_stds[self._column_stds == 0] = 1
        
        return (X - self._column_means) / self._column_stds
