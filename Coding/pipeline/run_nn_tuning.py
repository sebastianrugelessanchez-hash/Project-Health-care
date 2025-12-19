"""
Neural Network Tuning Runner
Execute hyperparameter tuning for Deep Neural Networks

Supports:
- TensorFlow/Keras with Metal GPU (Apple Silicon)
- Scikit-learn MLPRegressor fallback (CPU)

Usage:
    python run_nn_tuning.py                    # Auto-detect TensorFlow/CPU
    python run_nn_tuning.py --force-cpu        # Force sklearn CPU mode
    python run_nn_tuning.py --force-tensorflow # Force TensorFlow mode
"""

import argparse
import logging
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nn_tuning.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_and_prepare_data(logger):
    """
    Load and prepare data for tuning with proper train/val/test split.

    Split ratios:
    - Train (60%): Used for training neural networks
    - Validation (20%): Used internally by tuner for architecture selection
    - Test (20%): Hold-out for final evaluation only
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    from io_modulo import DataIOHandler
    from processing import DataProcessor
    from config import DATA_PARAMS, USABLE_COLUMNS, PIPELINE_PARAMS, DATABASE_PATHS

    logger.info("Loading data...")

    io_handler = DataIOHandler()
    data = io_handler.load_data(DATABASE_PATHS['healthcare_data'])
    logger.info(f"Raw data shape: {data.shape}")

    # Prepare features and target
    target_column = '2025-9'
    y = data[target_column].copy()
    X = data.drop(columns=[target_column])

    # Clean data
    processor = DataProcessor()

    # Remove NaN targets
    valid_rows = ~y.isnull()
    X = X[valid_rows].reset_index(drop=True)
    y = y[valid_rows].reset_index(drop=True)
    logger.info(f"After removing NaN targets: {X.shape}")

    # Clip negative values
    y = y.clip(lower=0)

    # Split data into train/val/test (60/20/20)
    # First split: separate test set (hold-out for final evaluation)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=PIPELINE_PARAMS['random_state']
    )

    # Second split: separate validation from training
    # val_size = 0.2 / 0.8 = 0.25 of remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=PIPELINE_PARAMS['random_state']
    )

    logger.info(f"Data split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Process features (fit on train only)
    X_train = processor.process_pipeline(
        X_train, DATA_PARAMS, fit=True,
        select_usable=True, usable_columns=USABLE_COLUMNS
    )
    X_val = processor.process_pipeline(
        X_val, DATA_PARAMS, fit=False,
        select_usable=True, usable_columns=USABLE_COLUMNS
    )
    X_test = processor.process_pipeline(
        X_test, DATA_PARAMS, fit=False,
        select_usable=True, usable_columns=USABLE_COLUMNS
    )
    logger.info(f"After processing - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(
        description='Neural Network Hyperparameter Tuning (GPU/CPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deep Network Architectures (14 configurations):
  Priority 1 (MAE + AdamW):
    - Deep_MAE_AdamW_Swish      [512, 256, 128, 64, 32]
    - VeryDeep_MAE_AdamW_ReLU   [1024, 512, 256, 128, 64, 32]
    - Wide_MAE_AdamW_Swish      [1024, 512, 256]
    - Pyramid_Swish_MAE         [1024, 512, 256, 128, 64, 32, 16]
    - UltraDeep_Swish_MAE       [512, 512, 256, 256, 128, 128, 64, 64, 32]

  Priority 2 (Mixed):
    - Deep_ReLU_Adam, VeryDeep_ReLU_Adam, Deep_ReLU_Huber
    - Deep_Swish_Adam, VeryDeep_Swish_RMSprop
    - Bottleneck_Linear_AdamW

  Priority 3 (Experimental):
    - ResidualStyle_Deep, Wide_Shallow_ReLU, Aggressive_Dropout_ReLU

Current Baseline (Neural Network):
    MAE:  25.79
    RMSE: 208.92
    R²:   0.8346
        """
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force scikit-learn CPU mode (ignore TensorFlow)'
    )
    parser.add_argument(
        '--force-tensorflow',
        action='store_true',
        help='Force TensorFlow mode (error if not available)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    logger = setup_logging()

    # Import tuner after logging setup
    from nn_tuning import NeuralNetworkTuner, TF_AVAILABLE, DEVICE
    from config import PIPELINE_PARAMS

    print("\n" + "=" * 60)
    print("DEEP NEURAL NETWORK HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"TensorFlow available: {TF_AVAILABLE}")
    print(f"Device: {DEVICE}")
    print(f"Output: {args.output}")

    if args.force_cpu:
        print("Mode: FORCED CPU (scikit-learn)")
        use_tf = False
    elif args.force_tensorflow:
        if not TF_AVAILABLE:
            print("ERROR: TensorFlow not available!")
            sys.exit(1)
        print("Mode: FORCED TensorFlow")
        use_tf = True
    else:
        use_tf = TF_AVAILABLE
        print(f"Mode: AUTO ({'TensorFlow/Keras' if use_tf else 'scikit-learn CPU'})")

    print("=" * 60 + "\n")

    # Load data with proper train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(logger)

    # Current baseline metrics
    baseline_metrics = {
        'MAE': 25.79,
        'RMSE': 208.92,
        'R2': 0.8346
    }

    # Initialize tuner
    tuner = NeuralNetworkTuner(
        random_state=PIPELINE_PARAMS['random_state'],
        n_jobs=PIPELINE_PARAMS['n_jobs'],
        output_dir=args.output
    )

    # Run tuning (use validation set for architecture selection)
    best_model, results = tuner.run_full_tuning(
        X_train, y_train,
        X_val, y_val,  # Validation set for model selection
        baseline_metrics=baseline_metrics,
        use_tensorflow=use_tf
    )

    # Final evaluation on test set (hold-out, used only once)
    from evaluation import RegressionEvaluator
    evaluator = RegressionEvaluator()
    test_metrics = None

    if best_model is not None:
        y_pred_test = best_model.predict(X_test)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred_test)
        logger.info(f"Final test set metrics: MAE={test_metrics['MAE']:.4f}, RMSE={test_metrics['RMSE']:.4f}, R2={test_metrics['R2']:.4f}")

        # Save NN final results to CSV for comparison
        import pandas as pd
        nn_results_df = pd.DataFrame([{
            'Model': results['params'].get('name', 'Neural Network') if results['params'] else 'Neural Network',
            'Architecture': str(results['params'].get('layers', [])) if results['params'] else '[]',
            'MAE': test_metrics['MAE'],
            'RMSE': test_metrics['RMSE'],
            'R2': test_metrics['R2']
        }])
        os.makedirs(args.output, exist_ok=True)
        nn_results_path = os.path.join(args.output, 'nn_final_results.csv')
        nn_results_df.to_csv(nn_results_path, index=False)
        logger.info(f"NN final results saved to: {nn_results_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("TUNING COMPLETE - FINAL RESULTS")
    print("=" * 60)

    if results['params']:
        print(f"\nBest Architecture: {results['params'].get('name', 'N/A')}")
        if isinstance(results['params'], dict):
            if 'layers' in results['params']:
                print(f"Layers: {results['params']['layers']}")
            if 'activation' in results['params']:
                print(f"Activation: {results['params']['activation']}")
            if 'optimizer' in results['params']:
                print(f"Optimizer: {results['params']['optimizer']}")
            if 'loss' in results['params']:
                print(f"Loss: {results['params']['loss']}")

    # Show validation metrics (used for model selection)
    print("\n" + "-" * 40)
    print("VALIDATION SET METRICS (Model Selection)")
    print("-" * 40)
    print(f"{'Metric':<10} {'Baseline':<12} {'Tuned':<12} {'Change':<12}")
    print("-" * 40)

    for metric in ['MAE', 'RMSE', 'R2']:
        baseline = baseline_metrics[metric]
        tuned = results['metrics'][metric]

        if metric in ['MAE', 'RMSE']:
            change = ((baseline - tuned) / baseline) * 100
            change_str = f"{change:+.2f}%"
        else:
            change = tuned - baseline
            change_str = f"{change:+.4f}"

        print(f"{metric:<10} {baseline:<12.4f} {tuned:<12.4f} {change_str:<12}")

    # Show test metrics (final evaluation)
    if best_model is not None and test_metrics:
        print("\n" + "-" * 40)
        print("TEST SET METRICS (Final Evaluation)")
        print("-" * 40)
        print(f"{'Metric':<10} {'Value':<12}")
        print("-" * 40)
        for metric in ['MAE', 'RMSE', 'R2']:
            print(f"{metric:<10} {test_metrics[metric]:<12.4f}")

    print("=" * 60 + "\n")

    return best_model, results, test_metrics if best_model is not None else None


if __name__ == "__main__":
    main()
