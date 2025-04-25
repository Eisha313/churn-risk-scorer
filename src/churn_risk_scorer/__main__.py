"""
Command-line interface for the churn risk scorer.
"""

import argparse
import sys
from .scorer import ChurnRiskScorer


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Customer Churn Risk Scorer - Predict customer churn probability'
    )
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to customer data CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Path to save risk scores CSV (optional)'
    )
    parser.add_argument(
        '--target', '-t',
        default='churn',
        help='Name of target column (default: churn)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate and display visualizations'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize scorer
        scorer = ChurnRiskScorer()
        
        # Load data
        print(f"Loading data from {args.data}...")
        scorer.load_data(args.data)
        
        # Train model
        print("\nTraining model...")
        metrics = scorer.train(test_size=args.test_size, target_column=args.target)
        
        # Generate predictions
        print("\nGenerating risk scores...")
        risk_scores = scorer.predict_risk()
        
        # Display summary
        print("\nRisk Level Summary:")
        print(risk_scores['risk_level'].value_counts().to_string())
        
        # Save output if specified
        if args.output:
            risk_scores.to_csv(args.output, index=False)
            print(f"\nRisk scores saved to {args.output}")
        
        # Generate visualizations if requested
        if args.visualize:
            print("\nGenerating visualizations...")
            scorer.plot_risk_distribution()
            scorer.plot_feature_importance()
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: File not found: {args.data}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
