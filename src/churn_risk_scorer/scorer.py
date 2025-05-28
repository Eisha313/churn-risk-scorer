"""Churn risk scoring using logistic regression."""

import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .exceptions import ModelNotTrainedError, ScoringError
from .config import settings

logger = logging.getLogger(__name__)


class ChurnScorer:
    """A churn risk scorer using logistic regression.
    
    This class provides functionality to train a logistic regression model
    on customer data and generate churn risk scores.
    
    Attributes:
        model: The trained logistic regression model.
        feature_names: List of feature names used in training.
        is_trained: Whether the model has been trained.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        max_iter: int = 1000,
        C: float = 1.0
    ):
        """Initialize the ChurnScorer.
        
        Args:
            random_state: Random seed for reproducibility.
            max_iter: Maximum iterations for logistic regression.
            C: Inverse regularization strength.
        """
        self._random_state = random_state or settings.RANDOM_STATE
        self._max_iter = max_iter
        self._C = C
        self._model: Optional[LogisticRegression] = None
        self._feature_names: Optional[list] = None
        self._is_trained: bool = False
        self._training_metrics: Optional[Dict[str, float]] = None
        
        logger.info(
            "ChurnScorer initialized with random_state=%d, max_iter=%d, C=%.2f",
            self._random_state, self._max_iter, self._C
        )
    
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._is_trained
    
    @property
    def feature_names(self) -> Optional[list]:
        """Get the feature names used in training."""
        return self._feature_names
    
    @property
    def training_metrics(self) -> Optional[Dict[str, float]]:
        """Get the training metrics."""
        return self._training_metrics
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        validate: bool = True
    ) -> Dict[str, float]:
        """Train the logistic regression model.
        
        Args:
            X: Feature DataFrame.
            y: Target Series (binary churn labels).
            test_size: Proportion of data to use for validation.
            validate: Whether to perform train/test split and validation.
            
        Returns:
            Dictionary containing training and validation metrics.
            
        Raises:
            ScoringError: If training fails.
        """
        try:
            logger.info("Starting model training with %d samples and %d features",
                       len(X), len(X.columns))
            
            self._feature_names = list(X.columns)
            
            # Initialize the model
            self._model = LogisticRegression(
                random_state=self._random_state,
                max_iter=self._max_iter,
                C=self._C,
                solver='lbfgs'
            )
            
            metrics = {}
            
            if validate and len(X) >= 10:
                # Split data for validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=self._random_state,
                    stratify=y if y.nunique() > 1 else None
                )
                
                # Train the model
                self._model.fit(X_train, y_train)
                
                # Calculate training metrics
                y_train_pred = self._model.predict(X_train)
                y_train_proba = self._model.predict_proba(X_train)[:, 1]
                
                metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
                metrics['train_precision'] = precision_score(y_train, y_train_pred, zero_division=0)
                metrics['train_recall'] = recall_score(y_train, y_train_pred, zero_division=0)
                metrics['train_f1'] = f1_score(y_train, y_train_pred, zero_division=0)
                
                if y_train.nunique() > 1:
                    metrics['train_auc'] = roc_auc_score(y_train, y_train_proba)
                
                # Calculate validation metrics
                y_val_pred = self._model.predict(X_val)
                y_val_proba = self._model.predict_proba(X_val)[:, 1]
                
                metrics['val_accuracy'] = accuracy_score(y_val, y_val_pred)
                metrics['val_precision'] = precision_score(y_val, y_val_pred, zero_division=0)
                metrics['val_recall'] = recall_score(y_val, y_val_pred, zero_division=0)
                metrics['val_f1'] = f1_score(y_val, y_val_pred, zero_division=0)
                
                if y_val.nunique() > 1:
                    metrics['val_auc'] = roc_auc_score(y_val, y_val_proba)
                
                logger.info("Validation accuracy: %.4f, F1: %.4f",
                           metrics['val_accuracy'], metrics['val_f1'])
            else:
                # Train on all data without validation
                self._model.fit(X, y)
                
                y_pred = self._model.predict(X)
                metrics['train_accuracy'] = accuracy_score(y, y_pred)
                metrics['train_f1'] = f1_score(y, y_pred, zero_division=0)
            
            self._is_trained = True
            self._training_metrics = metrics
            
            logger.info("Model training completed successfully")
            return metrics
            
        except Exception as e:
            logger.error("Model training failed: %s", str(e))
            raise ScoringError(f"Failed to train model: {str(e)}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probability for given features.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            Array of churn probabilities.
            
        Raises:
            ModelNotTrainedError: If the model hasn't been trained.
            ScoringError: If prediction fails.
        """
        if not self._is_trained or self._model is None:
            raise ModelNotTrainedError("Model must be trained before making predictions")
        
        try:
            # Ensure columns match training features
            X_aligned = self._align_features(X)
            probabilities = self._model.predict_proba(X_aligned)[:, 1]
            
            logger.debug("Generated predictions for %d samples", len(X))
            return probabilities
            
        except ModelNotTrainedError:
            raise
        except Exception as e:
            logger.error("Prediction failed: %s", str(e))
            raise ScoringError(f"Failed to generate predictions: {str(e)}")
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict churn labels for given features.
        
        Args:
            X: Feature DataFrame.
            threshold: Probability threshold for churn classification.
            
        Returns:
            Array of binary churn predictions.
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score_customers(
        self,
        X: pd.DataFrame,
        customer_ids: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Score customers and return a DataFrame with risk levels.
        
        Args:
            X: Feature DataFrame.
            customer_ids: Optional Series of customer IDs.
            
        Returns:
            DataFrame with customer IDs, churn probabilities, and risk levels.
        """
        probabilities = self.predict_proba(X)
        
        result = pd.DataFrame({
            'churn_probability': probabilities,
            'risk_level': pd.cut(
                probabilities,
                bins=[0, 0.3, 0.6, 1.0],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        })
        
        if customer_ids is not None:
            result.insert(0, 'customer_id', customer_ids.values)
        else:
            result.insert(0, 'customer_id', range(len(X)))
        
        logger.info(
            "Scored %d customers: %d High, %d Medium, %d Low risk",
            len(result),
            (result['risk_level'] == 'High').sum(),
            (result['risk_level'] == 'Medium').sum(),
            (result['risk_level'] == 'Low').sum()
        )
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on model coefficients.
        
        Returns:
            DataFrame with feature names and their importance scores.
            
        Raises:
            ModelNotTrainedError: If the model hasn't been trained.
        """
        if not self._is_trained or self._model is None:
            raise ModelNotTrainedError("Model must be trained before getting feature importance")
        
        coefficients = self._model.coef_[0]
        importance = pd.DataFrame({
            'feature': self._feature_names,
            'coefficient': coefficients,
            'abs_importance': np.abs(coefficients)
        })
        
        importance = importance.sort_values('abs_importance', ascending=False)
        importance = importance.reset_index(drop=True)
        
        return importance
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align input features with training features.
        
        Args:
            X: Input feature DataFrame.
            
        Returns:
            DataFrame with aligned features.
        """
        if self._feature_names is None:
            return X
        
        # Add missing columns with zeros
        for col in self._feature_names:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0
                logger.warning("Missing feature '%s' filled with zeros", col)
        
        # Select and order columns to match training
        return X[self._feature_names]
