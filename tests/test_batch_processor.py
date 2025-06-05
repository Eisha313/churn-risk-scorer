"""Tests for batch processing functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from churn_risk_scorer.batch_processor import BatchProcessor
from churn_risk_scorer.exceptions import DataLoadError, ScoringError
from churn_risk_scorer.scorer import ChurnScorer


class TestBatchProcessor:
    """Tests for BatchProcessor class."""
    
    def test_init_default(self):
        """Test default initialization."""
        processor = BatchProcessor()
        assert processor.batch_size == 1000
        assert processor.processed_count == 0
        assert processor.error_count == 0
    
    def test_init_custom_batch_size(self):
        """Test initialization with custom batch size."""
        processor = BatchProcessor(batch_size=500)
        assert processor.batch_size == 500
    
    def test_init_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            BatchProcessor(batch_size=0)
        
        with pytest.raises(ValueError, match="Batch size must be positive"):
            BatchProcessor(batch_size=-10)
    
    def test_reset_counts(self):
        """Test resetting processing counters."""
        processor = BatchProcessor()
        processor._processed_count = 100
        processor._error_count = 5
        
        processor.reset_counts()
        
        assert processor.processed_count == 0
        assert processor.error_count == 0
    
    def test_read_batches(self, tmp_path):
        """Test reading file in batches."""
        # Create test file
        df = pd.DataFrame({
            'customer_id': range(100),
            'value': np.random.randn(100)
        })
        filepath = tmp_path / "test_data.csv"
        df.to_csv(filepath, index=False)
        
        processor = BatchProcessor(batch_size=30)
        batches = list(processor.read_batches(str(filepath)))
        
        assert len(batches) == 4  # 100 records / 30 batch size = 4 batches
        assert len(batches[0]) == 30
        assert len(batches[-1]) == 10  # Last batch has remaining records
    
    def test_read_batches_file_not_found(self):
        """Test that reading non-existent file raises error."""
        processor = BatchProcessor()
        
        with pytest.raises(DataLoadError, match="File not found"):
            list(processor.read_batches("/nonexistent/file.csv"))
    
    def test_classify_risk(self):
        """Test risk classification."""
        processor = BatchProcessor()
        
        assert processor._classify_risk(0.8) == 'high'
        assert processor._classify_risk(0.7) == 'high'
        assert processor._classify_risk(0.5) == 'medium'
        assert processor._classify_risk(0.4) == 'medium'
        assert processor._classify_risk(0.3) == 'low'
        assert processor._classify_risk(0.0) == 'low'
    
    def test_process_batch_no_scoring(self, sample_customer_data):
        """Test batch processing without trained scorer."""
        processor = BatchProcessor()
        result = processor.process_batch(sample_customer_data, preprocess=False)
        
        assert len(result) == len(sample_customer_data)
        assert 'churn_score' not in result.columns  # Scorer not trained
    
    def test_aggregate_results(self):
        """Test result aggregation."""
        processor = BatchProcessor()
        processor._processed_count = 100
        processor._error_count = 2
        
        results = pd.DataFrame({
            'customer_id': range(100),
            'churn_score': np.random.uniform(0, 1, 100),
            'risk_level': np.random.choice(['low', 'medium', 'high'], 100)
        })
        
        stats = processor.aggregate_results(results)
        
        assert stats['total_records'] == 100
        assert stats['processed_count'] == 100
        assert stats['error_count'] == 2
        assert 'score_stats' in stats
        assert 'risk_distribution' in stats
    
    def test_aggregate_results_grouped(self):
        """Test grouped result aggregation."""
        processor = BatchProcessor()
        
        results = pd.DataFrame({
            'segment': ['A', 'A', 'B', 'B', 'B'],
            'churn_score': [0.1, 0.2, 0.7, 0.8, 0.9],
            'risk_level': ['low', 'low', 'high', 'high', 'high']
        })
        
        stats = processor.aggregate_results(results, group_by='segment')
        
        assert 'grouped' in stats
        assert 'A' in stats['grouped']
        assert 'B' in stats['grouped']
        assert stats['grouped']['A']['count'] == 2
        assert stats['grouped']['B']['count'] == 3
    
    def test_process_file_with_output(self, tmp_path):
        """Test processing file with output saving."""
        # Create test input file
        df = pd.DataFrame({
            'customer_id': range(50),
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        df.to_csv(input_path, index=False)
        
        processor = BatchProcessor(batch_size=20)
        result = processor.process_file(
            str(input_path),
            output_path=str(output_path),
            preprocess=False
        )
        
        assert len(result) == 50
        assert output_path.exists()
        assert processor.processed_count == 50


class TestBatchProcessorIntegration:
    """Integration tests for batch processing."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete batch processing pipeline."""
        # Create realistic test data
        np.random.seed(42)
        n_records = 200
        
        df = pd.DataFrame({
            'customer_id': range(n_records),
            'tenure_months': np.random.randint(1, 60, n_records),
            'monthly_charges': np.random.uniform(20, 100, n_records),
            'total_charges': np.random.uniform(100, 5000, n_records),
            'num_support_tickets': np.random.randint(0, 10, n_records),
            'contract_type': np.random.choice(['month-to-month', 'one_year', 'two_year'], n_records)
        })
        
        input_path = tmp_path / "customers.csv"
        df.to_csv(input_path, index=False)
        
        processor = BatchProcessor(batch_size=50)
        
        # Track progress
        progress_updates = []
        def track_progress(processed, total):
            progress_updates.append((processed, total))
        
        result = processor.process_file(
            str(input_path),
            preprocess=False,
            progress_callback=track_progress
        )
        
        assert len(result) == n_records
        assert len(progress_updates) == 4  # 200 records / 50 batch size
        assert processor.error_count == 0
