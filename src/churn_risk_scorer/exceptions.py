"""Custom exceptions for the churn risk scorer package."""


class ChurnScorerError(Exception):
    """Base exception for churn risk scorer."""
    pass


class DataLoadError(ChurnScorerError):
    """Raised when data loading fails."""
    
    def __init__(self, message: str, filepath: str = None):
        self.filepath = filepath
        super().__init__(f"Failed to load data: {message}")


class DataValidationError(ChurnScorerError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, column: str = None):
        self.column = column
        if column:
            message = f"Validation error in column '{column}': {message}"
        super().__init__(message)


class PreprocessingError(ChurnScorerError):
    """Raised when data preprocessing fails."""
    pass


class ModelError(ChurnScorerError):
    """Raised when model training or prediction fails."""
    pass


class ModelNotTrainedError(ModelError):
    """Raised when trying to predict with an untrained model."""
    
    def __init__(self):
        super().__init__("Model must be trained before making predictions")


class VisualizationError(ChurnScorerError):
    """Raised when visualization generation fails."""
    pass


class ConfigurationError(ChurnScorerError):
    """Raised when configuration is invalid."""
    pass
