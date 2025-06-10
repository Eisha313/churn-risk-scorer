# Churn Risk Scorer Usage Guide

This guide provides practical examples for using the churn-risk-scorer package.

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from churn_risk_scorer import ChurnScorer, DataPreprocessor, DataLoader

# Load data
loader = DataLoader()
df = loader.load_csv("customer_data.csv")

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df)

# Train and score
scorer = ChurnScorer()
scorer.fit(X, y)

# Get predictions
risk_scores = scorer.predict_proba(X)
risk_levels = scorer.get_risk_levels(X)
```

## Command Line Interface

### Train a Model

```bash
churn-scorer train --input data/customers.csv --output models/churn_model.pkl
```

### Score Customers

```bash
churn-scorer score --model models/churn_model.pkl --input new_customers.csv --output scores.csv
```

### Generate Visualizations

```bash
churn-scorer visualize --model models/churn_model.pkl --input data/customers.csv --output report.html
```

## Data Format

### Required Columns

Your CSV file should contain customer features. Example:

| customer_id | tenure | monthly_charges | total_charges | contract_type | churned |
|-------------|--------|-----------------|---------------|---------------|----------|
| 1001 | 12 | 65.5 | 786.0 | monthly | 0 |
| 1002 | 36 | 89.0 | 3204.0 | yearly | 0 |
| 1003 | 3 | 45.0 | 135.0 | monthly | 1 |

### Supported Feature Types

- **Numeric**: Automatically scaled
- **Categorical**: One-hot encoded
- **Missing values**: Imputed with mean/mode

## Advanced Usage

### Custom Risk Thresholds

```python
from churn_risk_scorer.scorer import ChurnScorer

scorer = ChurnScorer(
    low_risk_threshold=0.25,
    high_risk_threshold=0.75
)
```

### Batch Processing Large Files

```python
from churn_risk_scorer.batch_processor import BatchProcessor
from churn_risk_scorer.model_persistence import load_model

scorer = load_model("models/churn_model.pkl")
processor = BatchProcessor(batch_size=5000)

# Process large file efficiently
results = processor.process_file("large_customer_data.csv", scorer)
results.to_csv("scored_customers.csv", index=False)
```

### Creating Visualizations

```python
from churn_risk_scorer.visualizer import ChurnVisualizer

visualizer = ChurnVisualizer()

# Risk distribution histogram
fig1 = visualizer.plot_risk_distribution(risk_scores)
fig1.write_html("risk_distribution.html")

# Feature importance
importance = scorer.get_feature_importance()
fig2 = visualizer.plot_feature_importance(importance)
fig2.write_html("feature_importance.html")
```

### Model Persistence

```python
from churn_risk_scorer.model_persistence import save_model, load_model

# Save trained model
save_model(scorer, "models/churn_model.pkl")

# Load for later use
loaded_scorer = load_model("models/churn_model.pkl")
```

## Integration Examples

### Flask API

```python
from flask import Flask, request, jsonify
from churn_risk_scorer.model_persistence import load_model
from churn_risk_scorer.preprocessor import DataPreprocessor
import pandas as pd

app = Flask(__name__)
scorer = load_model("models/churn_model.pkl")
preprocessor = DataPreprocessor()

@app.route("/score", methods=["POST"])
def score_customer():
    data = request.json
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)
    prob = scorer.predict_proba(X)[0]
    risk = scorer.get_risk_levels(X)[0]
    return jsonify({"probability": prob, "risk_level": risk})
```

### Scheduled Batch Job

```python
import schedule
import time
from churn_risk_scorer.batch_processor import BatchProcessor
from churn_risk_scorer.model_persistence import load_model

def daily_scoring():
    scorer = load_model("models/churn_model.pkl")
    processor = BatchProcessor(batch_size=10000)
    results = processor.process_file("daily_export.csv", scorer)
    results.to_csv(f"scores_{time.strftime('%Y%m%d')}.csv")
    
schedule.every().day.at("06:00").do(daily_scoring)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Common Issues

**ModelNotFittedError**: Call `scorer.fit()` before prediction.

```python
scorer = ChurnScorer()
scorer.fit(X_train, y_train)  # Don't forget this!
```

**DataValidationError**: Check your CSV has required columns.

```python
loader = DataLoader()
if loader.validate_data(df):
    # proceed
```

**Memory Issues with Large Files**: Use batch processing.

```python
processor = BatchProcessor(batch_size=1000)  # Reduce batch size
```

## Best Practices

1. **Always validate data** before processing
2. **Use batch processing** for files > 100k rows
3. **Save models** after training for reproducibility
4. **Monitor feature importance** to understand predictions
5. **Set appropriate risk thresholds** for your business context
