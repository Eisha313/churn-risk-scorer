"""Data preprocessing module for customer churn prediction."""

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from churn_risk_scorer.exceptions import DataValidationError, PreprocessingError

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses customer data for churn prediction model."""
    
    # Default numeric columns to scale
    DEFAULT_NUMERIC_COLUMNS = [
        'tenure', 'monthly_charges', 'total_charges',
        'num_products', 'account_age', 'support_tickets'
    ]
    
    # Default categorical columns to encode
    DEFAULT_CATEGORICAL_COLUMNS = [
        'contract_type', 'payment_method', 'gender', 'region'
    ]
    
    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        target_column: str = 'churn'
    ):
        """Initialize the preprocessor.
        
        Args:
            numeric_columns: List of numeric column names to scale.
            categorical_columns: List of categorical column names to encode.
            target_column: Name of the target column.
        """
        self.numeric_columns = numeric_columns or self.DEFAULT_NUMERIC_COLUMNS
        self.categorical_columns = categorical_columns or self.DEFAULT_CATEGORICAL_COLUMNS
        self.target_column = target_column
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self._is_fitted = False
        
        logger.info(
            f"DataPreprocessor initialized with {len(self.numeric_columns)} numeric "
            f"and {len(self.categorical_columns)} categorical columns"
        )
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the dataframe has required structure.
        
        Args:
            df: DataFrame to validate.
            
        Raises:
            DataValidationError: If validation fails.
        """
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        # Check for available columns (not all need to be present)
        available_numeric = [col for col in self.numeric_columns if col in df.columns]
        available_categorical = [col for col in self.categorical_columns if col in df.columns]
        
        if not available_numeric and not available_categorical:
            raise DataValidationError(
                "No configured numeric or categorical columns found in DataFrame. "
                f"Expected some of: {self.numeric_columns + self.categorical_columns}"
            )
        
        logger.debug(
            f"Validation passed. Found {len(available_numeric)} numeric "
            f"and {len(available_categorical)} categorical columns"
        )
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe.
        
        Args:
            df: DataFrame with potential missing values.
            
        Returns:
            DataFrame with missing values handled.
        """
        df = df.copy()
        initial_nulls = df.isnull().sum().sum()
        
        # Handle numeric columns - fill with median
        for col in self.numeric_columns:
            if col in df.columns and df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.debug(f"Filled {col} nulls with median: {median_value}")
        
        # Handle categorical columns - fill with mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isnull().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col] = df[col].fillna(mode_value)
                logger.debug(f"Filled {col} nulls with mode: {mode_value}")
        
        final_nulls = df.isnull().sum().sum()
        logger.info(f"Handled missing values: {initial_nulls} -> {final_nulls}")
        
        return df
    
    def _scale_numeric_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features using StandardScaler.
        
        Args:
            df: DataFrame with numeric columns.
            fit: Whether to fit the scaler or use existing fit.
            
        Returns:
            DataFrame with scaled numeric columns.
        """
        df = df.copy()
        available_numeric = [col for col in self.numeric_columns if col in df.columns]
        
        if not available_numeric:
            return df
        
        try:
            if fit:
                df[available_numeric] = self.scaler.fit_transform(df[available_numeric])
                logger.debug(f"Fitted and transformed {len(available_numeric)} numeric columns")
            else:
                df[available_numeric] = self.scaler.transform(df[available_numeric])
                logger.debug(f"Transformed {len(available_numeric)} numeric columns")
        except Exception as e:
            raise PreprocessingError(f"Failed to scale numeric features: {e}")
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder.
        
        Args:
            df: DataFrame with categorical columns.
            fit: Whether to fit the encoders or use existing fit.
            
        Returns:
            DataFrame with encoded categorical columns.
        """
        df = df.copy()
        available_categorical = [col for col in self.categorical_columns if col in df.columns]
        
        for col in available_categorical:
            try:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col not in self.label_encoders:
                        raise PreprocessingError(
                            f"No encoder found for column '{col}'. Call fit_transform first."
                        )
                    # Handle unseen labels
                    df[col] = df[col].astype(str)
                    known_labels = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_labels else self.label_encoders[col].classes_[0]
                    )
                    df[col] = self.label_encoders[col].transform(df[col])
            except Exception as e:
                raise PreprocessingError(f"Failed to encode column '{col}': {e}")
        
        logger.debug(f"Encoded {len(available_categorical)} categorical columns")
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame.
            include_target: Whether to extract and return target column.
            
        Returns:
            Tuple of (processed features DataFrame, target Series or None).
        """
        logger.info(f"Starting fit_transform on DataFrame with shape {df.shape}")
        
        self._validate_dataframe(df)
        
        # Extract target if present and requested
        target = None
        if include_target and self.target_column in df.columns:
            target = df[self.target_column].copy()
            df = df.drop(columns=[self.target_column])
            logger.debug(f"Extracted target column '{self.target_column}'")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Scale numeric features
        df = self._scale_numeric_features(df, fit=True)
        
        # Encode categorical features
        df = self._encode_categorical_features(df, fit=True)
        
        self._is_fitted = True
        logger.info(f"fit_transform complete. Output shape: {df.shape}")
        
        return df, target
    
    def transform(
        self,
        df: pd.DataFrame,
        include_target: bool = False
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform data using fitted preprocessor.
        
        Args:
            df: Input DataFrame.
            include_target: Whether to extract and return target column.
            
        Returns:
            Tuple of (processed features DataFrame, target Series or None).
            
        Raises:
            PreprocessingError: If preprocessor hasn't been fitted.
        """
        if not self._is_fitted:
            raise PreprocessingError(
                "Preprocessor has not been fitted. Call fit_transform first."
            )
        
        logger.info(f"Starting transform on DataFrame with shape {df.shape}")
        
        self._validate_dataframe(df)
        
        # Extract target if present and requested
        target = None
        if include_target and self.target_column in df.columns:
            target = df[self.target_column].copy()
            df = df.drop(columns=[self.target_column])
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Scale numeric features
        df = self._scale_numeric_features(df, fit=False)
        
        # Encode categorical features
        df = self._encode_categorical_features(df, fit=False)
        
        logger.info(f"transform complete. Output shape: {df.shape}")
        
        return df, target
    
    @property
    def is_fitted(self) -> bool:
        """Check if the preprocessor has been fitted."""
        return self._is_fitted
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get the feature names after preprocessing.
        
        Args:
            df: DataFrame to get feature names from.
            
        Returns:
            List of feature column names.
        """
        feature_cols = []
        
        for col in self.numeric_columns:
            if col in df.columns:
                feature_cols.append(col)
        
        for col in self.categorical_columns:
            if col in df.columns:
                feature_cols.append(col)
        
        return feature_cols
