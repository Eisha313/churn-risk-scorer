"""Data loading utilities for customer churn data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .logging_config import get_logger
from .exceptions import DataLoadError, DataValidationError

logger = get_logger(__name__)


class DataLoader:
    """Load and validate customer data from CSV files."""
    
    REQUIRED_COLUMNS = ['customer_id']
    OPTIONAL_COLUMNS = ['tenure', 'monthly_charges', 'total_charges', 
                        'contract_type', 'payment_method', 'churn']
    
    def __init__(self, filepath: Union[str, Path]):
        """Initialize the data loader.
        
        Args:
            filepath: Path to the CSV file containing customer data.
        """
        self.filepath = Path(filepath)
        self._data: Optional[pd.DataFrame] = None
        logger.info(f"DataLoader initialized with filepath: {self.filepath}")
    
    def load(self) -> pd.DataFrame:
        """Load data from the CSV file.
        
        Returns:
            DataFrame containing the loaded data.
            
        Raises:
            DataLoadError: If the file cannot be loaded.
        """
        if not self.filepath.exists():
            raise DataLoadError(f"File not found: {self.filepath}", str(self.filepath))
        
        try:
            logger.info(f"Loading data from {self.filepath}")
            self._data = pd.read_csv(self.filepath)
            logger.info(f"Loaded {len(self._data)} rows and {len(self._data.columns)} columns")
            return self._data
        except pd.errors.EmptyDataError:
            raise DataLoadError("File is empty", str(self.filepath))
        except pd.errors.ParserError as e:
            raise DataLoadError(f"Failed to parse CSV: {e}", str(self.filepath))
        except Exception as e:
            raise DataLoadError(str(e), str(self.filepath))
    
    def validate(self, data: Optional[pd.DataFrame] = None) -> bool:
        """Validate the loaded data.
        
        Args:
            data: DataFrame to validate. Uses loaded data if not provided.
            
        Returns:
            True if validation passes.
            
        Raises:
            DataValidationError: If validation fails.
        """
        df = data if data is not None else self._data
        
        if df is None:
            raise DataValidationError("No data to validate. Call load() first.")
        
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        # Check for required columns
        missing_required = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_required:
            raise DataValidationError(
                f"Missing required columns: {missing_required}"
            )
        
        # Check for duplicate customer IDs
        if 'customer_id' in df.columns and df['customer_id'].duplicated().any():
            raise DataValidationError(
                "Duplicate customer IDs found",
                column='customer_id'
            )
        
        logger.info("Data validation passed")
        return True
    
    def load_and_validate(self) -> pd.DataFrame:
        """Load and validate data in one step.
        
        Returns:
            Validated DataFrame.
        """
        data = self.load()
        self.validate(data)
        return data
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get the loaded data."""
        return self._data
    
    def get_summary(self) -> dict:
        """Get a summary of the loaded data.
        
        Returns:
            Dictionary containing data summary statistics.
        """
        if self._data is None:
            return {}
        
        return {
            'rows': len(self._data),
            'columns': len(self._data.columns),
            'column_names': list(self._data.columns),
            'missing_values': self._data.isnull().sum().to_dict(),
            'dtypes': self._data.dtypes.astype(str).to_dict()
        }
