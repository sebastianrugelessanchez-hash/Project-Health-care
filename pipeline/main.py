"""
Main Entry Point
Starts the ML pipeline with example configuration
"""

import logging
import sys
from pathlib import Path

from ejecutador import PipelineOrchestrator
from config import PIPELINE_PARAMS, DATA_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PIPELINE_PARAMS.get('log_file', 'pipeline.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the pipeline

    Example usage:
        python main.py --data data.csv --target target_column
    """

    import argparse

    parser = argparse.ArgumentParser(
        description='Machine Learning Pipeline for Healthcare Project'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='/Users/sebas.12/Desktop/Proyectos/Project Healthcare/Data base/Reporte de prueba.xlsx',
        help='Path to input data file (default: Reporte de prueba.xlsx)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=['2025-9'],
        nargs='+',
        help='Name of target column(s) - default: 2025-9'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        default=True,
        help='Perform hyperparameter tuning (default: True)'
    )
    parser.add_argument(
        '--no-tune',
        action='store_false',
        dest='tune',
        help='Disable hyperparameter tuning'
    )

    args = parser.parse_args()

    # Validate input file exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    # Handle single or multiple target columns
    target_column = args.target[0] if len(args.target) == 1 else args.target

    # Create and run pipeline
    logger.info("Initializing ML Pipeline")
    pipeline = PipelineOrchestrator(output_dir=args.output)

    try:
        results = pipeline.run_complete_pipeline(
            data_path=str(data_path),
            target_column=target_column,
            tune_models=args.tune
        )

        logger.info("Pipeline execution completed successfully!")
        logger.info(f"Results saved to: {args.output}")

        # Print summary with primary metrics (MAE, RMSE, R2)
        best_metrics = results['evaluation_results'].iloc[0]
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Input Data: {args.data}")
        print(f"Target Column(s): {target_column}")
        print(f"Output Directory: {args.output}")
        print(f"\nBest Model: {results['best_model_name']}")
        print("-" * 40)
        print("PRIMARY METRICS:")
        print(f"  MAE:  {best_metrics['MAE']:.4f}")
        print(f"  RMSE: {best_metrics['RMSE']:.4f}")
        print(f"  R²:   {best_metrics['R2-Score']:.4f}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
