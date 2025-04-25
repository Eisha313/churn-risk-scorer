# Churn Risk Scorer

A customer churn prediction utility that scores customer risk levels using logistic regression and generates interactive visualizations to help businesses identify at-risk customers.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from churn_risk_scorer import ChurnRiskScorer

# Initialize the scorer
scorer = ChurnRiskScorer()

# Load and preprocess data
scorer.load_data('customer_data.csv')

# Train the model
scorer.train()

# Get risk scores for customers
scores = scorer.predict_risk()

# Generate visualizations
scorer.plot_risk_distribution()
scorer.plot_feature_importance()
```

### Command Line

```bash
# Train and score from CSV
python -m churn_risk_scorer --data customer_data.csv --output results.csv

# Generate visualizations
python -m churn_risk_scorer --data customer_data.csv --visualize
```

### Expected CSV Format

Your customer data CSV should include columns like:
- `customer_id`: Unique identifier
- `tenure`: Months as customer
- `monthly_charges`: Monthly billing amount
- `total_charges`: Total amount billed
- `contract_type`: Month-to-month, One year, Two year
- `churn`: Target variable (1 = churned, 0 = retained)

## Example

```python
import pandas as pd
from churn_risk_scorer import ChurnRiskScorer

# Create sample data
data = pd.DataFrame({
    'customer_id': range(100),
    'tenure': [12, 24, 6, 36, 3] * 20,
    'monthly_charges': [50, 75, 90, 45, 100] * 20,
    'total_charges': [600, 1800, 540, 1620, 300] * 20,
    'contract_type': ['month-to-month', 'one_year', 'month-to-month', 'two_year', 'month-to-month'] * 20,
    'churn': [1, 0, 1, 0, 1] * 20
})
data.to_csv('sample_data.csv', index=False)

scorer = ChurnRiskScorer()
scorer.load_data('sample_data.csv')
scorer.train()
print(scorer.predict_risk())
```

## License

MIT License
