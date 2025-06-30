# Contributing to Churn Risk Scorer

Thank you for your interest in contributing to Churn Risk Scorer! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/churn-risk-scorer.git
   cd churn-risk-scorer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install the package in editable mode**
   ```bash
   pip install -e .
   ```

## Development Workflow

### Branching Strategy

- `main` - stable release branch
- `develop` - integration branch for features
- `feature/*` - new features
- `bugfix/*` - bug fixes
- `docs/*` - documentation updates

### Making Changes

1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Write or update tests as needed

4. Run the test suite:
   ```bash
   pytest
   ```

5. Commit your changes with a descriptive message:
   ```bash
   git commit -m "feat: add new feature description"
   ```

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - new feature
- `fix:` - bug fix
- `docs:` - documentation changes
- `test:` - adding or updating tests
- `refactor:` - code refactoring
- `chore:` - maintenance tasks

### Pull Request Process

1. Push your branch to your fork
2. Open a Pull Request against the `develop` branch
3. Fill out the PR template with relevant information
4. Wait for code review
5. Address any feedback
6. Once approved, your PR will be merged

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings for all public functions and classes

### Example Function

```python
from typing import Optional
import pandas as pd


def process_data(
    data: pd.DataFrame,
    column: str,
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    Process data by filtering based on threshold.
    
    Args:
        data: Input DataFrame to process.
        column: Name of column to filter on.
        threshold: Optional threshold value. Defaults to None.
    
    Returns:
        Filtered DataFrame.
    
    Raises:
        ValueError: If column does not exist in data.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    if threshold is not None:
        return data[data[column] >= threshold]
    return data
```

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest fixtures for common test data
- Test both success and error cases

### Documentation

- Update docstrings for any changed functions
- Update README.md if adding new features
- Update API.md for new public interfaces
- Add usage examples for complex features

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. Python version and OS
2. Package version
3. Minimal code to reproduce the issue
4. Expected vs actual behavior
5. Full error traceback if applicable

### Feature Requests

For feature requests, please describe:

1. The problem you're trying to solve
2. Your proposed solution
3. Alternative solutions you've considered
4. Any relevant examples or references

## Project Structure

```
churn-risk-scorer/
├── src/churn_risk_scorer/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # CLI entry point
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration management
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessor.py      # Data preprocessing
│   ├── scorer.py            # Churn scoring model
│   ├── visualizer.py        # Visualization functions
│   ├── batch_processor.py   # Batch processing
│   ├── model_persistence.py # Model save/load
│   ├── exceptions.py        # Custom exceptions
│   └── logging_config.py    # Logging setup
├── tests/                   # Test suite
├── docs/                    # Documentation
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── pyproject.toml          # Package configuration
└── README.md               # Project overview
```

## Questions?

Feel free to open an issue for any questions about contributing. We're happy to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
