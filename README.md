# Churn Risk Scorer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A simple, powerful customer churn prediction utility that scores customer risk levels using machine learning.

## Features

- 📊 **Data Processing**: Load and preprocess customer data from CSV files using pandas and numpy
- 🤖 **ML Prediction**: Train logistic regression models to predict churn probability
- 📈 **Visualizations**: Generate interactive Plotly visualizations for churn risk analysis
- 🔄 **Batch Processing**: Process large datasets efficiently with progress tracking
- 💾 **Model Persistence**: Save and load trained models for production use
- 🖥️ **CLI Interface**: Easy-to-use command-line interface for all operations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/churn-risk-scorer.git
cd churn-risk-scorer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

#### Python API

```python
from churn_risk_scorer import DataLoader, Preprocessor, ChurnScorer, ChurnVisualizer

# Load and preprocess data
loader = DataLoader()
data = loader.load_csv("customers.csv")

preprocessor = Preprocessor()
processed_data, feature_names = preprocessor.fit_transform(data, target_column="churned")

# Train the model
scorer = ChurnScorer()
scorer.fit(processed_data, data["churned"])

# Get predictions
scores = scorer.predict_proba(processed_data)
risk_levels = scorer.get_risk_levels(scores)

# Visualize results
visualizer = ChurnVisualizer()
fig = visualizer.plot_risk_distribution(scores, risk_levels)
fig.show()
```

#### Command Line

```bash
# Train a model
churn-risk-scorer train --data customers.csv --target churned --output model.joblib

# Score new customers
churn-risk-scorer score --model model.joblib --data new_customers.csv --output scores.csv

# Generate visualization
churn-risk-scorer visualize --data scores.csv --output report.html

# Batch process with progress
churn-risk-scorer batch --model model.joblib --input-dir ./data --output-dir ./results
```

## Example Output

### Risk Score Distribution

The scorer categorizes customers into risk levels:

| Risk Level | Probability Range | Recommended Action |
|------------|-------------------|--------------------|
| Low        | 0.0 - 0.3        | Monitor |
| Medium     | 0.3 - 0.6        | Engage |
| High       | 0.6 - 0.8        | Intervene |
| Critical   | 0.8 - 1.0        | Immediate action |

### Sample Predictions

```python
results = scorer.score_customers(customer_data)
print(results.head())
```

```
   customer_id  churn_probability risk_level  confidence
0         1001              0.12        Low        0.88
1         1002              0.67       High        0.67
2         1003              0.45     Medium        0.55
3         1004              0.89   Critical        0.89
4         1005              0.23        Low        0.77
```

## Configuration

Configure the scorer using environment variables or a `.env` file:

```bash
# .env
CHURN_LOG_LEVEL=INFO
CHURN_MODEL_PATH=./models
CHURN_DEFAULT_THRESHOLD=0.5
CHURN_BATCH_SIZE=1000
```

See [.env.example](.env.example) for all available options.

## Project Structure

```
churn-risk-scorer/
├── src/churn_risk_scorer/   # Main package
│   ├── cli.py               # Command-line interface
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessor.py      # Feature preprocessing
│   ├── scorer.py            # ML model and scoring
│   ├── visualizer.py        # Plotly visualizations
│   ├── batch_processor.py   # Batch operations
│   └── model_persistence.py # Model save/load
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── API.md              # API reference
│   └── USAGE.md            # Usage guide
└── requirements.txt         # Dependencies
```

## Documentation

- [API Reference](docs/API.md) - Detailed API documentation
- [Usage Guide](docs/USAGE.md) - Comprehensive usage examples
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=churn_risk_scorer

# Format code
black src/ tests/

# Type checking
mypy src/
```

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- plotly >= 5.0.0
- click >= 8.0.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on the process for submitting pull requests.

## Changelog

### v1.0.0
- Initial release
- Basic churn prediction with logistic regression
- Interactive Plotly visualizations
- CLI interface
- Batch processing support
- Model persistence

## Support

If you encounter any issues or have questions:

1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/yourusername/churn-risk-scorer/issues)
3. Open a new issue with details about your problem

---

Made with ❤️ for customer retention teams everywhere.
