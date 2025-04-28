"""Configuration settings for the churn risk scorer."""

import os
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Settings:
    """Application settings with environment variable support."""
    
    def __init__(self):
        self.debug: bool = self._get_bool_env("CHURN_DEBUG", False)
        self.log_level: str = os.getenv("CHURN_LOG_LEVEL", "INFO")
        self.model_path: Path = MODELS_DIR / os.getenv("CHURN_MODEL_NAME", "churn_model.pkl")
        self.random_seed: int = int(os.getenv("CHURN_RANDOM_SEED", "42"))
        self.test_size: float = float(os.getenv("CHURN_TEST_SIZE", "0.2"))
        self.max_iter: int = int(os.getenv("CHURN_MAX_ITER", "1000"))
    
    @staticmethod
    def _get_bool_env(key: str, default: bool) -> bool:
        """Parse boolean environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes")


# Default feature columns for churn prediction
DEFAULT_FEATURE_COLUMNS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "payment_method",
    "num_support_tickets",
    "usage_frequency",
]

# Target column name
TARGET_COLUMN = "churned"

# Risk thresholds for classification
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0,
}

# Global settings instance
settings = Settings()
