"""Batch processing module for handling large datasets efficiently."""

import logging
from pathlib import Path
from typing import Generator, Optional, Callable, Any
import pandas as pd
import numpy as np

from .exceptions import DataLoadError, ScoringError
from .data_loader import DataLoader
from .preprocessor import Preprocessor
from .scorer import ChurnScorer

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process large datasets in batches for memory efficiency."""
    
    def __init__(
        self,
        batch_size: int = 1000,
        preprocessor: Optional[Preprocessor] = None,
        scorer: Optional[ChurnScorer] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of records to process per batch
            preprocessor: Optional preprocessor instance
            scorer: Optional scorer instance
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        self.batch_size = batch_size
        self.preprocessor = preprocessor or Preprocessor()
        self.scorer = scorer or ChurnScorer()
        self._processed_count = 0
        self._error_count = 0
    
    @property
    def processed_count(self) -> int:
        """Return total number of processed records."""
        return self._processed_count
    
    @property
    def error_count(self) -> int:
        """Return total number of errors encountered."""
        return self._error_count
    
    def reset_counts(self) -> None:
        """Reset processing counters."""
        self._processed_count = 0
        self._error_count = 0
    
    def read_batches(
        self,
        filepath: str,
        **read_kwargs
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Read a CSV file in batches.
        
        Args:
            filepath: Path to the CSV file
            **read_kwargs: Additional arguments for pd.read_csv
            
        Yields:
            DataFrame batches
        """
        path = Path(filepath)
        if not path.exists():
            raise DataLoadError(f"File not found: {filepath}")
        
        try:
            reader = pd.read_csv(
                filepath,
                chunksize=self.batch_size,
                **read_kwargs
            )
            
            for batch_num, chunk in enumerate(reader, 1):
                logger.debug(f"Reading batch {batch_num} with {len(chunk)} records")
                yield chunk
                
        except Exception as e:
            raise DataLoadError(f"Error reading file in batches: {e}")
    
    def process_batch(
        self,
        batch: pd.DataFrame,
        preprocess: bool = True
    ) -> pd.DataFrame:
        """
        Process a single batch of data.
        
        Args:
            batch: DataFrame batch to process
            preprocess: Whether to apply preprocessing
            
        Returns:
            Processed batch with churn scores
        """
        try:
            if preprocess:
                processed = self.preprocessor.preprocess(batch)
            else:
                processed = batch.copy()
            
            # Get predictions if model is trained
            if self.scorer.is_trained:
                scores = self.scorer.predict_proba(processed)
                processed['churn_score'] = scores
                processed['risk_level'] = processed['churn_score'].apply(
                    self._classify_risk
                )
            
            self._processed_count += len(batch)
            return processed
            
        except Exception as e:
            self._error_count += len(batch)
            logger.error(f"Error processing batch: {e}")
            raise ScoringError(f"Batch processing failed: {e}")
    
    def _classify_risk(self, score: float) -> str:
        """Classify risk level based on churn score."""
        if score >= 0.7:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        return 'low'
    
    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        preprocess: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """
        Process an entire file in batches.
        
        Args:
            input_path: Path to input CSV file
            output_path: Optional path to save results
            preprocess: Whether to apply preprocessing
            progress_callback: Optional callback(processed, total) for progress updates
            
        Returns:
            Combined DataFrame with all processed results
        """
        self.reset_counts()
        results = []
        
        # Get total row count for progress tracking
        total_rows = sum(1 for _ in open(input_path)) - 1  # Subtract header
        
        logger.info(f"Processing {total_rows} records in batches of {self.batch_size}")
        
        for batch in self.read_batches(input_path):
            try:
                processed_batch = self.process_batch(batch, preprocess)
                results.append(processed_batch)
                
                if progress_callback:
                    progress_callback(self._processed_count, total_rows)
                    
            except ScoringError as e:
                logger.warning(f"Skipping batch due to error: {e}")
                continue
        
        if not results:
            raise ScoringError("No batches were successfully processed")
        
        combined = pd.concat(results, ignore_index=True)
        
        if output_path:
            combined.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        logger.info(
            f"Processing complete: {self._processed_count} processed, "
            f"{self._error_count} errors"
        )
        
        return combined
    
    def aggregate_results(
        self,
        results: pd.DataFrame,
        group_by: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Aggregate batch processing results.
        
        Args:
            results: Processed DataFrame
            group_by: Optional column to group aggregations by
            
        Returns:
            Dictionary containing aggregated statistics
        """
        stats = {
            'total_records': len(results),
            'processed_count': self._processed_count,
            'error_count': self._error_count
        }
        
        if 'churn_score' in results.columns:
            stats['score_stats'] = {
                'mean': float(results['churn_score'].mean()),
                'median': float(results['churn_score'].median()),
                'std': float(results['churn_score'].std()),
                'min': float(results['churn_score'].min()),
                'max': float(results['churn_score'].max())
            }
        
        if 'risk_level' in results.columns:
            risk_counts = results['risk_level'].value_counts().to_dict()
            stats['risk_distribution'] = risk_counts
        
        if group_by and group_by in results.columns:
            grouped_stats = {}
            for name, group in results.groupby(group_by):
                grouped_stats[str(name)] = {
                    'count': len(group),
                    'mean_score': float(group['churn_score'].mean()) if 'churn_score' in group.columns else None
                }
            stats['grouped'] = grouped_stats
        
        return stats
