# Churn Risk Scorer

A simple customer churn prediction utility that scores customer risk levels using machine learning.

## Features

- 📊 Load and preprocess customer data from CSV files using pandas and numpy
- 🤖 Train a simple logistic regression model to predict churn probability
- 📈 Generate interactive Plotly visualizations showing churn risk distribution and key factors
- 🚀 Batch processing for large datasets
- 💾 Model persistence for production deployment

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-risk-scorer.git
cd churn-risk-scorer

# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from churn_risk_scorer import ChurnScorer, DataPreprocessor, DataLoader

# Load and preprocess data
loader = DataLoader()
df = loader.load_csv("customer_data.csv")

preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df)

# Train model
scorer = ChurnScorer()
scorer.fit(X, y)

# Get risk scores
risk_levels = scorer.get_risk_levels(X)
print(risk_levels)  # ['LOW', 'HIGH', 'MEDIUM', ...]
```

## Command Line Usage

```bash
# Train a model
churn-scorer train --input data.csv --output model.pkl

# Score customers
churn-scorer score --model model.pkl --input customers.csv --output results.csv

# Generate visualizations
churn-scorer visualize --model model.pkl --input data.csv --output report.html
```

## Documentation

- [API Documentation](docs/API.md) - Detailed API reference
- [Usage Guide](docs/USAGE.md) - Practical examples and tutorials

## Project Structure

```
churn-risk-scorer/
├── src/churn_risk_scorer/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── data_loader.py      # CSV data loading
│   ├── preprocessor.py     # Data preprocessing
│   ├── scorer.py           # Churn prediction model
│   ├── visualizer.py       # Plotly visualizations
│   ├── batch_processor.py  # Large file processing
│   ├── model_persistence.py # Save/load models
│   ├── exceptions.py       # Custom exceptions
│   └── logging_config.py   # Logging setup
├── tests/                  # Test suite
├── docs/                   # Documentation
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=churn_risk_scorer

# Run linting
flake8 src/
```

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Environment variables:
- `CHURN_SCORER_LOG_LEVEL` - Logging level (default: INFO)
- `CHURN_SCORER_MODEL_DIR` - Model storage directory
- `CHURN_SCORER_BATCH_SIZE` - Default batch size for processing

## License

MIT License - see LICENSE for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.
