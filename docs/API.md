# Churn Risk Scorer API Documentation

This document provides detailed API documentation for all modules in the churn-risk-scorer package.

## Table of Contents

- [Data Loader](#data-loader)
- [Preprocessor](#preprocessor)
- [Scorer](#scorer)
- [Visualizer](#visualizer)
- [Batch Processor](#batch-processor)
- [Model Persistence](#model-persistence)
- [Exceptions](#exceptions)

---

## Data Loader

### `churn_risk_scorer.data_loader`

Handles loading customer data from CSV files.

#### Classes

##### `DataLoader`

Main class for loading and validating customer data.

```python
from churn_risk_scorer.data_loader import DataLoader

loader = DataLoader()
df = loader.load_csv("customers.csv")
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `load_csv(filepath)` | `filepath: str` | `pd.DataFrame` | Load data from CSV file |
| `validate_data(df)` | `df: pd.DataFrame` | `bool` | Validate required columns exist |

---

## Preprocessor

### `churn_risk_scorer.preprocessor`

Data preprocessing and feature engineering.

#### Classes

##### `DataPreprocessor`

Preprocesses customer data for model training.

```python
from churn_risk_scorer.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df)
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `fit(df)` | `df: pd.DataFrame` | `self` | Fit preprocessor to data |
| `transform(df)` | `df: pd.DataFrame` | `np.ndarray` | Transform data |
| `fit_transform(df)` | `df: pd.DataFrame` | `Tuple[np.ndarray, np.ndarray]` | Fit and transform in one step |
| `get_feature_names()` | None | `List[str]` | Get list of feature names |

**Attributes:**

- `scaler`: StandardScaler instance
- `feature_columns`: List of feature column names
- `target_column`: Name of target column (default: 'churned')

---

## Scorer

### `churn_risk_scorer.scorer`

Churn risk scoring using logistic regression.

#### Classes

##### `ChurnScorer`

Main scoring class that trains and predicts churn risk.

```python
from churn_risk_scorer.scorer import ChurnScorer

scorer = ChurnScorer()
scorer.fit(X_train, y_train)
probabilities = scorer.predict_proba(X_test)
risk_levels = scorer.get_risk_levels(X_test)
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `fit(X, y)` | `X: np.ndarray, y: np.ndarray` | `self` | Train the model |
| `predict(X)` | `X: np.ndarray` | `np.ndarray` | Predict churn (0/1) |
| `predict_proba(X)` | `X: np.ndarray` | `np.ndarray` | Get churn probabilities |
| `get_risk_levels(X)` | `X: np.ndarray` | `List[str]` | Get risk categories |
| `get_feature_importance()` | None | `Dict[str, float]` | Get feature importances |

**Risk Levels:**

- `LOW`: probability < 0.3
- `MEDIUM`: 0.3 <= probability < 0.7
- `HIGH`: probability >= 0.7

---

## Visualizer

### `churn_risk_scorer.visualizer`

Interactive Plotly visualizations.

#### Classes

##### `ChurnVisualizer`

Creates interactive charts for churn analysis.

```python
from churn_risk_scorer.visualizer import ChurnVisualizer

visualizer = ChurnVisualizer()
fig = visualizer.plot_risk_distribution(probabilities)
fig.show()
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `plot_risk_distribution(probs)` | `probs: np.ndarray` | `plotly.Figure` | Histogram of risk scores |
| `plot_feature_importance(importance)` | `importance: Dict` | `plotly.Figure` | Bar chart of features |
| `plot_risk_by_segment(df, probs)` | `df: pd.DataFrame, probs: np.ndarray` | `plotly.Figure` | Risk by customer segment |
| `save_figure(fig, path)` | `fig: Figure, path: str` | None | Save figure to HTML |

---

## Batch Processor

### `churn_risk_scorer.batch_processor`

Process large datasets in batches.

#### Classes

##### `BatchProcessor`

Handles batch processing for large datasets.

```python
from churn_risk_scorer.batch_processor import BatchProcessor

processor = BatchProcessor(batch_size=1000)
results = processor.process_file("large_data.csv", scorer)
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `process_file(filepath, scorer)` | `filepath: str, scorer: ChurnScorer` | `pd.DataFrame` | Process entire file |
| `process_batch(batch, scorer)` | `batch: pd.DataFrame, scorer: ChurnScorer` | `pd.DataFrame` | Process single batch |
| `get_batch_iterator(filepath)` | `filepath: str` | `Iterator` | Get iterator over batches |

**Parameters:**

- `batch_size`: Number of rows per batch (default: 1000)
- `n_jobs`: Number of parallel jobs (default: 1)

---

## Model Persistence

### `churn_risk_scorer.model_persistence`

Save and load trained models.

#### Functions

```python
from churn_risk_scorer.model_persistence import save_model, load_model

save_model(scorer, "model.pkl")
loaded_scorer = load_model("model.pkl")
```

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `save_model(model, path)` | `model: ChurnScorer, path: str` | None | Save model to file |
| `load_model(path)` | `path: str` | `ChurnScorer` | Load model from file |
| `save_model_with_metadata(model, path, metadata)` | `model, path, metadata: Dict` | None | Save with metadata |

---

## Exceptions

### `churn_risk_scorer.exceptions`

Custom exceptions for error handling.

#### Exception Classes

| Exception | Description |
|-----------|-------------|
| `ChurnScorerError` | Base exception for all package errors |
| `DataLoadError` | Error loading data files |
| `DataValidationError` | Data validation failed |
| `ModelNotFittedError` | Model used before training |
| `InvalidConfigError` | Configuration error |

```python
from churn_risk_scorer.exceptions import DataLoadError, ModelNotFittedError

try:
    loader.load_csv("missing.csv")
except DataLoadError as e:
    print(f"Failed to load: {e}")
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHURN_SCORER_LOG_LEVEL` | `INFO` | Logging level |
| `CHURN_SCORER_MODEL_DIR` | `./models` | Model storage directory |
| `CHURN_SCORER_BATCH_SIZE` | `1000` | Default batch size |

### Config File

Create a `config.yaml` for custom settings:

```yaml
model:
  regularization: 1.0
  max_iter: 1000
  
preprocessor:
  scale_features: true
  handle_missing: mean
  
risk_thresholds:
  low: 0.3
  high: 0.7
```
