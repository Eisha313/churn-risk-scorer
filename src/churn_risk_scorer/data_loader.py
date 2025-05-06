"""Data loading utilities for customer churn data."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import settings

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate customer data from CSV files."""

    REQUIRED_COLUMNS = [
        "customer_id",
        "tenure",
        "monthly_charges",
        "total_charges",
    ]

    OPTIONAL_COLUMNS = [
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "phone_service",
        "internet_service",
        "contract",
        "payment_method",
        "churn",
    ]

    def __init__(self, data_path: Optional[str] = None):
        """Initialize the data loader.

        Args:
            data_path: Path to the CSV file. If None, uses path from settings.
        """
        self.data_path = Path(data_path) if data_path else Path(settings.data_path)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load data from CSV file.

        Returns:
            DataFrame containing customer data.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
            ValueError: If required columns are missing.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        logger.info(f"Loading data from {self.data_path}")
        self._df = pd.read_csv(self.data_path)

        self._validate_columns()
        self._clean_data()

        logger.info(f"Loaded {len(self._df)} records")
        return self._df

    def _validate_columns(self) -> None:
        """Validate that required columns are present."""
        if self._df is None:
            raise ValueError("No data loaded")

        missing_columns = set(self.REQUIRED_COLUMNS) - set(self._df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.debug(f"Found columns: {list(self._df.columns)}")

    def _clean_data(self) -> None:
        """Clean and prepare the loaded data."""
        if self._df is None:
            return

        # Convert total_charges to numeric, handling empty strings
        if "total_charges" in self._df.columns:
            self._df["total_charges"] = pd.to_numeric(
                self._df["total_charges"], errors="coerce"
            )

        # Fill missing total_charges with monthly_charges * tenure
        mask = self._df["total_charges"].isna()
        if mask.any():
            self._df.loc[mask, "total_charges"] = (
                self._df.loc[mask, "monthly_charges"] * self._df.loc[mask, "tenure"]
            )
            logger.debug(f"Filled {mask.sum()} missing total_charges values")

        # Ensure customer_id is string
        self._df["customer_id"] = self._df["customer_id"].astype(str)

        # Convert churn to binary if present
        if "churn" in self._df.columns:
            self._df["churn"] = self._df["churn"].map(
                {"Yes": 1, "No": 0, 1: 1, 0: 0, True: 1, False: 0}
            ).fillna(0).astype(int)

    def get_feature_target_split(
        self, target_column: str = "churn"
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Split data into features and target.

        Args:
            target_column: Name of the target column.

        Returns:
            Tuple of (features DataFrame, target Series or None).
        """
        if self._df is None:
            self.load()

        df = self._df.copy()

        # Separate target if present
        target = None
        if target_column in df.columns:
            target = df[target_column]
            df = df.drop(columns=[target_column])

        # Remove customer_id from features
        if "customer_id" in df.columns:
            df = df.drop(columns=["customer_id"])

        return df, target

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the loaded data."""
        return self._df

    def get_summary(self) -> dict:
        """Get summary statistics of loaded data."""
        if self._df is None:
            return {"status": "no data loaded"}

        summary = {
            "total_records": len(self._df),
            "columns": list(self._df.columns),
            "missing_values": self._df.isnull().sum().to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in self._df.dtypes.items()},
        }

        if "churn" in self._df.columns:
            summary["churn_rate"] = float(self._df["churn"].mean())

        return summary


def load_customer_data(path: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to load customer data.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame with customer data.
    """
    loader = DataLoader(path)
    return loader.load()
