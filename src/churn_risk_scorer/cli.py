"""Command-line interface for churn risk scorer."""

import click
import sys
from pathlib import Path

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .scorer import ChurnScorer
from .visualizer import ChurnVisualizer
from .config import Config
from .logging_config import setup_logging, get_logger
from .exceptions import ChurnScorerError, DataLoadError, ModelError


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.pass_context
def cli(ctx, verbose, config):
    """Churn Risk Scorer - Predict customer churn probability."""
    ctx.ensure_object(dict)
    
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level=log_level)
    
    ctx.obj['logger'] = get_logger(__name__)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = Config()


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for scores')
@click.option('--model', '-m', type=click.Path(exists=True), help='Pre-trained model file')
@click.pass_context
def score(ctx, input_file, output, model):
    """Score customers from INPUT_FILE for churn risk."""
    logger = ctx.obj['logger']
    
    try:
        logger.info(f"Loading data from {input_file}")
        loader = DataLoader(input_file)
        data = loader.load()
        
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(data)
        
        scorer = ChurnScorer()
        
        if model:
            logger.info(f"Loading model from {model}")
            scorer.load_model(model)
        else:
            logger.info("Training new model")
            # For scoring, we need a trained model
            if 'churn' in processed_data.columns:
                X = processed_data.drop('churn', axis=1)
                y = processed_data['churn']
                scorer.fit(X, y)
            else:
                click.echo("Error: No 'churn' column found and no model provided.", err=True)
                sys.exit(1)
        
        logger.info("Generating scores")
        if 'churn' in processed_data.columns:
            X = processed_data.drop('churn', axis=1)
        else:
            X = processed_data
            
        scores = scorer.predict_proba(X)
        risk_levels = scorer.get_risk_levels(scores)
        
        # Create output dataframe
        import pandas as pd
        results = pd.DataFrame({
            'customer_id': range(len(scores)),
            'churn_probability': scores,
            'risk_level': risk_levels
        })
        
        if output:
            results.to_csv(output, index=False)
            click.echo(f"Scores saved to {output}")
        else:
            click.echo(results.to_string())
            
        # Summary statistics
        click.echo(f"\nSummary:")
        click.echo(f"  Total customers: {len(scores)}")
        click.echo(f"  High risk: {sum(r == 'high' for r in risk_levels)}")
        click.echo(f"  Medium risk: {sum(r == 'medium' for r in risk_levels)}")
        click.echo(f"  Low risk: {sum(r == 'low' for r in risk_levels)}")
        
    except ChurnScorerError as e:
        logger.error(f"Scoring failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='model.joblib', help='Output model file')
@click.option('--test-size', type=float, default=0.2, help='Test set size (0-1)')
@click.pass_context
def train(ctx, input_file, output, test_size):
    """Train a churn prediction model on INPUT_FILE."""
    logger = ctx.obj['logger']
    
    try:
        logger.info(f"Loading training data from {input_file}")
        loader = DataLoader(input_file)
        data = loader.load()
        
        if 'churn' not in data.columns:
            click.echo("Error: Training data must contain 'churn' column.", err=True)
            sys.exit(1)
        
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(data)
        
        X = processed_data.drop('churn', axis=1)
        y = processed_data['churn']
        
        logger.info("Training model")
        scorer = ChurnScorer()
        metrics = scorer.fit(X, y, test_size=test_size)
        
        logger.info(f"Saving model to {output}")
        scorer.save_model(output)
        
        click.echo(f"Model trained and saved to {output}")
        click.echo(f"\nTraining Metrics:")
        for key, value in metrics.items():
            click.echo(f"  {key}: {value:.4f}")
            
    except ChurnScorerError as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='./visualizations', help='Output directory')
@click.option('--model', '-m', type=click.Path(exists=True), help='Pre-trained model file')
@click.option('--show', is_flag=True, help='Show plots in browser')
@click.pass_context
def visualize(ctx, input_file, output_dir, model, show):
    """Generate visualizations for churn analysis."""
    logger = ctx.obj['logger']
    
    try:
        logger.info(f"Loading data from {input_file}")
        loader = DataLoader(input_file)
        data = loader.load()
        
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.fit_transform(data)
        
        scorer = ChurnScorer()
        
        if model:
            logger.info(f"Loading model from {model}")
            scorer.load_model(model)
        elif 'churn' in processed_data.columns:
            logger.info("Training model for visualization")
            X = processed_data.drop('churn', axis=1)
            y = processed_data['churn']
            scorer.fit(X, y)
        else:
            click.echo("Error: No model provided and no 'churn' column for training.", err=True)
            sys.exit(1)
        
        # Generate scores
        if 'churn' in processed_data.columns:
            X = processed_data.drop('churn', axis=1)
        else:
            X = processed_data
            
        scores = scorer.predict_proba(X)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating visualizations")
        visualizer = ChurnVisualizer()
        
        # Risk distribution
        fig_dist = visualizer.plot_risk_distribution(scores)
        dist_path = output_path / 'risk_distribution.html'
        fig_dist.write_html(str(dist_path))
        click.echo(f"Risk distribution saved to {dist_path}")
        
        # Feature importance
        if hasattr(scorer, 'model') and scorer.model is not None:
            feature_names = list(X.columns)
            fig_importance = visualizer.plot_feature_importance(scorer.model, feature_names)
            importance_path = output_path / 'feature_importance.html'
            fig_importance.write_html(str(importance_path))
            click.echo(f"Feature importance saved to {importance_path}")
        
        if show:
            fig_dist.show()
            
    except ChurnScorerError as e:
        logger.error(f"Visualization failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.pass_context
def validate(ctx, input_file):
    """Validate input data format and quality."""
    logger = ctx.obj['logger']
    
    try:
        logger.info(f"Validating {input_file}")
        loader = DataLoader(input_file)
        data = loader.load()
        
        click.echo(f"File: {input_file}")
        click.echo(f"Rows: {len(data)}")
        click.echo(f"Columns: {len(data.columns)}")
        click.echo(f"\nColumn names: {list(data.columns)}")
        click.echo(f"\nMissing values:")
        
        missing = data.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                click.echo(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
        
        if missing.sum() == 0:
            click.echo("  None")
        
        click.echo(f"\nData types:")
        for col, dtype in data.dtypes.items():
            click.echo(f"  {col}: {dtype}")
            
        if 'churn' in data.columns:
            click.echo(f"\nChurn distribution:")
            churn_counts = data['churn'].value_counts()
            for value, count in churn_counts.items():
                click.echo(f"  {value}: {count} ({count/len(data)*100:.1f}%)")
        else:
            click.echo("\nWarning: No 'churn' column found.")
            
        click.echo("\nValidation complete.")
        
    except DataLoadError as e:
        logger.error(f"Validation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()
