"""
Neural Network Hyperparameter Tuning Module
Optimizes Deep Neural Networks using TensorFlow/Keras with Metal GPU support
Falls back to scikit-learn MLPRegressor for CPU if TensorFlow unavailable

Priorities:
- Deep networks with ReLU, Linear, Swish activations
- MAE loss function with AdamW optimizer
- Metal GPU acceleration on Apple Silicon
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, TYPE_CHECKING
import logging
import json
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Help static analyzers see these names even if TF is optional
if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers, Model  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
    from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop  # type: ignore
    from tensorflow.keras.optimizers import AdamW  # type: ignore

# Try to import TensorFlow (runtime)
TF_AVAILABLE = False
tf = keras = layers = Model = EarlyStopping = ReduceLROnPlateau = Adam = AdamW = SGD = RMSprop = None
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Use legacy optimizers for better Metal GPU compatibility on Apple Silicon
    try:
        from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop
        logger.info("Using legacy optimizers for Metal GPU compatibility")
    except ImportError:
        from tensorflow.keras.optimizers import Adam, SGD, RMSprop
        logger.info("Using standard optimizers (legacy not available)")

    from tensorflow.keras.optimizers import AdamW  # AdamW not in legacy

    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"TensorFlow GPU detected: {gpus}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        TF_AVAILABLE = True
        DEVICE = 'GPU'
    else:
        TF_AVAILABLE = True
        DEVICE = 'CPU'
    logger.info(f"TensorFlow {tf.__version__} loaded, using {DEVICE}")
except ImportError as e:
    logger.warning(f"TensorFlow not available: {e}")
    logger.info("Falling back to scikit-learn MLPRegressor (CPU only)")
    DEVICE = 'CPU'

# Fallback dummy classes to keep names defined even when TensorFlow is missing.
# They should not be instantiated because TensorFlow workflows are skipped in that case.
if not TF_AVAILABLE:
    class _MissingTF:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow not available. Install tensorflow to use NeuralNetworkTuner with Keras.")

    Adam = Adam or _MissingTF
    AdamW = AdamW or _MissingTF
    SGD = SGD or _MissingTF
    RMSprop = RMSprop or _MissingTF

# Scikit-learn fallback
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class NeuralNetworkTuner:
    """
    Hyperparameter tuning for Deep Neural Networks
    Uses TensorFlow/Keras with Metal GPU when available
    """

    def __init__(self, random_state: int = 42, n_jobs: int = -1, output_dir: str = 'results'):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.best_model = None
        self.best_params = None
        self.tuning_history = []
        self.use_gpu = TF_AVAILABLE and DEVICE == 'GPU'

        # Set seeds for reproducibility
        np.random.seed(random_state)
        if TF_AVAILABLE:
            tf.random.set_seed(random_state)

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"NeuralNetworkTuner initialized")
        logger.info(f"  Device: {DEVICE}")
        logger.info(f"  TensorFlow available: {TF_AVAILABLE}")
        logger.info(f"  Output directory: {output_dir}")

    def get_deep_architectures(self) -> List[Dict]:
        """
        Define deep network architectures to explore
        Prioritizes deeper networks with ReLU, Linear, Swish activations
        """
        architectures = [
            # ========== MAE + AdamW (Priority Configuration) ==========
            {
                'name': 'Deep_MAE_AdamW_Swish',
                'layers': [512, 256, 128, 64, 32],
                'activation': 'swish',
                'output_activation': 'linear',
                'optimizer': 'adamw',
                'loss': 'mae',
                'learning_rate': 0.001,
                'dropout': 0.2,
                'batch_norm': True,
                'priority': 1
            },
            {
                'name': 'VeryDeep_MAE_AdamW_ReLU',
                'layers': [1024, 512, 256, 128, 64, 32],
                'activation': 'relu',
                'output_activation': 'linear',
                'optimizer': 'adamw',
                'loss': 'mae',
                'learning_rate': 0.0005,
                'dropout': 0.3,
                'batch_norm': True,
                'priority': 1
            },
            {
                'name': 'Wide_MAE_AdamW_Swish',
                'layers': [1024, 512, 256],
                'activation': 'swish',
                'output_activation': 'linear',
                'optimizer': 'adamw',
                'loss': 'mae',
                'learning_rate': 0.001,
                'dropout': 0.2,
                'batch_norm': True,
                'priority': 1
            },

            # ========== Deep ReLU Networks ==========
            {
                'name': 'Deep_ReLU_Adam',
                'layers': [512, 256, 128, 64],
                'activation': 'relu',
                'output_activation': 'linear',
                'optimizer': 'adam',
                'loss': 'mse',
                'learning_rate': 0.001,
                'dropout': 0.2,
                'batch_norm': True,
                'priority': 2
            },
            {
                'name': 'VeryDeep_ReLU_Adam',
                'layers': [768, 512, 384, 256, 128, 64],
                'activation': 'relu',
                'output_activation': 'linear',
                'optimizer': 'adam',
                'loss': 'mse',
                'learning_rate': 0.0005,
                'dropout': 0.3,
                'batch_norm': True,
                'priority': 2
            },
            {
                'name': 'Deep_ReLU_Huber',
                'layers': [512, 256, 128, 64, 32],
                'activation': 'relu',
                'output_activation': 'linear',
                'optimizer': 'adam',
                'loss': 'huber',
                'learning_rate': 0.001,
                'dropout': 0.2,
                'batch_norm': True,
                'priority': 2
            },

            # ========== Swish Networks ==========
            {
                'name': 'Deep_Swish_Adam',
                'layers': [512, 256, 128, 64],
                'activation': 'swish',
                'output_activation': 'linear',
                'optimizer': 'adam',
                'loss': 'mse',
                'learning_rate': 0.001,
                'dropout': 0.15,
                'batch_norm': True,
                'priority': 2
            },
            {
                'name': 'VeryDeep_Swish_RMSprop',
                'layers': [640, 512, 384, 256, 128, 64],
                'activation': 'swish',
                'output_activation': 'linear',
                'optimizer': 'rmsprop',
                'loss': 'mse',
                'learning_rate': 0.001,
                'dropout': 0.25,
                'batch_norm': True,
                'priority': 2
            },

            # ========== Linear (No Activation) Bottleneck ==========
            {
                'name': 'Bottleneck_Linear_AdamW',
                'layers': [512, 256, 64, 256, 128],  # Bottleneck architecture
                'activation': 'relu',
                'bottleneck_activation': 'linear',  # Linear at bottleneck
                'output_activation': 'linear',
                'optimizer': 'adamw',
                'loss': 'mae',
                'learning_rate': 0.001,
                'dropout': 0.2,
                'batch_norm': True,
                'priority': 2
            },

            # ========== Mixed Architectures ==========
            {
                'name': 'ResidualStyle_Deep',
                'layers': [256, 256, 256, 256, 128, 64],
                'activation': 'relu',
                'output_activation': 'linear',
                'optimizer': 'adamw',
                'loss': 'mae',
                'learning_rate': 0.0005,
                'dropout': 0.2,
                'batch_norm': True,
                'priority': 3
            },
            {
                'name': 'Pyramid_Swish_MAE',
                'layers': [1024, 512, 256, 128, 64, 32, 16],
                'activation': 'swish',
                'output_activation': 'linear',
                'optimizer': 'adamw',
                'loss': 'mae',
                'learning_rate': 0.0003,
                'dropout': 0.3,
                'batch_norm': True,
                'priority': 1
            },
            {
                'name': 'Wide_Shallow_ReLU',
                'layers': [2048, 1024, 512],
                'activation': 'relu',
                'output_activation': 'linear',
                'optimizer': 'adam',
                'loss': 'mse',
                'learning_rate': 0.001,
                'dropout': 0.4,
                'batch_norm': True,
                'priority': 3
            },

            # ========== Experimental ==========
            {
                'name': 'UltraDeep_Swish_MAE',
                'layers': [512, 512, 256, 256, 128, 128, 64, 64, 32],
                'activation': 'swish',
                'output_activation': 'linear',
                'optimizer': 'adamw',
                'loss': 'mae',
                'learning_rate': 0.0001,
                'dropout': 0.3,
                'batch_norm': True,
                'priority': 1
            },
            {
                'name': 'Aggressive_Dropout_ReLU',
                'layers': [768, 512, 256, 128, 64],
                'activation': 'relu',
                'output_activation': 'linear',
                'optimizer': 'adamw',
                'loss': 'mae',
                'learning_rate': 0.001,
                'dropout': 0.5,
                'batch_norm': True,
                'priority': 3
            },
        ]

        # Sort by priority (lower = higher priority)
        return sorted(architectures, key=lambda x: x['priority'])

    def build_keras_model(self, input_dim: int, config: Dict) -> Any:
        """
        Build a Keras model from configuration

        Args:
            input_dim: Number of input features
            config: Architecture configuration dictionary

        Returns:
            Compiled Keras model
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")

        # Local imports - use legacy optimizers for Metal GPU compatibility
        try:
            from tensorflow.keras.optimizers.legacy import Adam as KAdam, SGD as KSGD, RMSprop as KRMSprop  # type: ignore
        except ImportError:
            from tensorflow.keras.optimizers import Adam as KAdam, SGD as KSGD, RMSprop as KRMSprop  # type: ignore
        from tensorflow.keras.optimizers import AdamW as KAdamW  # type: ignore

        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        # Build hidden layers
        for i, units in enumerate(config['layers']):
            x = layers.Dense(units, kernel_initializer='he_normal')(x)

            # Batch normalization before activation
            if config.get('batch_norm', False):
                x = layers.BatchNormalization()(x)

            # Activation
            activation = config.get('bottleneck_activation') if (
                config.get('bottleneck_activation') and
                i == len(config['layers']) // 2
            ) else config['activation']

            if activation == 'swish':
                x = layers.Activation('swish')(x)
            elif activation == 'relu':
                x = layers.ReLU()(x)
            elif activation == 'linear' or activation is None:
                pass  # No activation (linear)
            else:
                x = layers.Activation(activation)(x)

            # Dropout
            if config.get('dropout', 0) > 0:
                x = layers.Dropout(config['dropout'])(x)

        # Output layer
        outputs = layers.Dense(1, activation=config.get('output_activation', 'linear'))(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Get optimizer
        lr = config.get('learning_rate', 0.001)
        optimizer_name = config.get('optimizer', 'adam').lower()

        if optimizer_name == 'adamw':
            optimizer = KAdamW(learning_rate=lr, weight_decay=0.01)
        elif optimizer_name == 'adam':
            optimizer = KAdam(learning_rate=lr)
        elif optimizer_name == 'sgd':
            optimizer = KSGD(learning_rate=lr, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            optimizer = KRMSprop(learning_rate=lr)
        else:
            optimizer = KAdam(learning_rate=lr)

        # Get loss function
        loss = config.get('loss', 'mse').lower()
        if loss == 'mae':
            loss_fn = 'mean_absolute_error'
        elif loss == 'huber':
            loss_fn = keras.losses.Huber()
        else:
            loss_fn = 'mean_squared_error'

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae', 'mse']
        )

        return model

    def train_keras_model(self, model: Any, X_train: np.ndarray,
                          y_train: np.ndarray, X_val: np.ndarray,
                          y_val: np.ndarray, config: Dict) -> Dict:
        """
        Train a Keras model with callbacks

        Args:
            model: Compiled Keras model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            config: Training configuration

        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(
                monitor='val_mae',
                patience=20,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_mae',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=0
            )
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.get('epochs', 200),
            batch_size=config.get('batch_size', 64),
            callbacks=callbacks,
            verbose=0
        )

        return history.history

    def tune_tensorflow(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        architectures: List[Dict] = None) -> Tuple[Any, Dict]:
        """
        Run hyperparameter tuning using TensorFlow/Keras

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            architectures: List of architecture configs (uses defaults if None)

        Returns:
            Tuple of (best_model, results_dict)
        """
        if architectures is None:
            architectures = self.get_deep_architectures()

        logger.info("=" * 60)
        logger.info("TENSORFLOW/KERAS NEURAL NETWORK TUNING")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Architectures to evaluate: {len(architectures)}")
        logger.info("=" * 60)

        # Split training data for validation
        val_split = 0.15
        val_size = int(len(X_train) * val_split)
        indices = np.random.permutation(len(X_train))

        X_val = X_train[indices[:val_size]]
        y_val = y_train[indices[:val_size]]
        X_train_sub = X_train[indices[val_size:]]
        y_train_sub = y_train[indices[val_size:]]

        results = []
        best_val_mae = float('inf')
        best_model = None
        best_config = None

        for i, config in enumerate(architectures):
            logger.info(f"\n[{i+1}/{len(architectures)}] Training: {config['name']}")
            logger.info(f"  Layers: {config['layers']}")
            logger.info(f"  Activation: {config['activation']}, Loss: {config['loss']}, Optimizer: {config['optimizer']}")

            try:
                # Build and train model
                model = self.build_keras_model(X_train.shape[1], config)

                training_config = {
                    'epochs': 200,
                    'batch_size': 64 if len(X_train) > 10000 else 32
                }

                history = self.train_keras_model(
                    model, X_train_sub, y_train_sub,
                    X_val, y_val, training_config
                )

                # Use validation performance (not test) to rank architectures
                val_mae = float(np.min(history['val_mae']))
                train_mae = float(np.min(history['mae']))

                result = {
                    'name': config['name'],
                    'config': config,
                    'val_mae': val_mae,
                    'train_mae': train_mae,
                    'epochs_trained': len(history['loss']),
                    'final_train_mae': history['mae'][-1],
                    'final_val_mae': history['val_mae'][-1]
                }
                results.append(result)

                logger.info(f"  Validation - MAE: {val_mae:.4f} (train MAE: {train_mae:.4f})")

                # Track best model
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_config = config
                    logger.info("  *** New best model by validation MAE ***")

            except Exception as e:
                logger.error(f"  Error training {config['name']}: {str(e)}")
                continue

        # Retrain best config on full training data (still only using validation for callbacks)
        if best_config is not None:
            logger.info("\nRetraining best architecture on full training data before final test evaluation")
            best_model = self.build_keras_model(X_train.shape[1], best_config)
            training_config = {
                'epochs': 200,
                'batch_size': 64 if len(X_train) > 10000 else 32
            }
            self.train_keras_model(
                best_model, X_train, y_train,
                X_val, y_val, training_config
            )

        # Sort results by validation MAE
        results_df = pd.DataFrame(results).sort_values('val_mae')

        self.best_model = best_model
        self.best_params = best_config

        return best_model, {
            'results': results_df,
            'best_config': best_config,
            'best_val_mae': best_val_mae,
            'all_results': results
        }

    def tune_sklearn(self, X_train: np.ndarray, y_train: np.ndarray,
                     scoring: str = 'neg_mean_absolute_error',
                     n_iter: int = 50, cv: int = 5) -> Tuple[Any, Dict]:
        """
        Fallback: Tune using scikit-learn MLPRegressor (CPU)
        Optimized for parallel execution
        """
        logger.info("=" * 60)
        logger.info("SCIKIT-LEARN MLP TUNING (CPU - Parallel)")
        logger.info("=" * 60)

        # Deep architectures for sklearn
        param_distributions = {
            'hidden_layer_sizes': [
                (512, 256, 128, 64),
                (512, 256, 128, 64, 32),
                (768, 512, 256, 128, 64),
                (1024, 512, 256, 128),
                (512, 512, 256, 256, 128),
                (640, 512, 384, 256, 128, 64),
                (1024, 512, 256, 128, 64, 32),
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam'],
            'alpha': [0.00001, 0.0001, 0.001, 0.01],
            'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005],
            'learning_rate': ['adaptive'],
            'max_iter': [500, 1000],
            'early_stopping': [True],
            'validation_fraction': [0.15],
            'n_iter_no_change': [20],
            'batch_size': [32, 64, 128]
        }

        model = MLPRegressor(random_state=self.random_state, verbose=False)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=self.n_jobs,  # Parallel execution
            verbose=2,
            random_state=self.random_state,
            return_train_score=True
        )

        search.fit(X_train, y_train)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_

        return search.best_estimator_, {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }

    def evaluate_model(self, model: Any, X_test: np.ndarray,
                       y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set"""
        if TF_AVAILABLE and hasattr(model, 'predict') and hasattr(model, 'layers'):
            y_pred = model.predict(X_test, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)

        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred)
        }

        logger.info("\n" + "-" * 40)
        logger.info("TEST SET EVALUATION")
        logger.info("-" * 40)
        logger.info(f"MAE:  {metrics['MAE']:.4f}")
        logger.info(f"RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"R²:   {metrics['R2']:.4f}")

        return metrics

    def compare_with_baseline(self, baseline_metrics: Dict, new_metrics: Dict) -> Dict:
        """Compare tuned model with baseline"""
        improvements = {}

        for metric in ['MAE', 'RMSE']:
            if metric in baseline_metrics and metric in new_metrics:
                baseline = baseline_metrics[metric]
                new = new_metrics[metric]
                improvement = ((baseline - new) / baseline) * 100
                improvements[f'{metric}_improvement_pct'] = improvement

        if 'R2' in baseline_metrics and 'R2' in new_metrics:
            improvements['R2_improvement'] = new_metrics['R2'] - baseline_metrics['R2']

        logger.info("\n" + "-" * 40)
        logger.info("IMPROVEMENT vs BASELINE")
        logger.info("-" * 40)
        for key, value in improvements.items():
            if 'R2' in key:
                logger.info(f"{key}: {value:+.4f}")
            else:
                logger.info(f"{key}: {value:+.2f}%")

        return improvements

    def save_results(self, results: Dict, metrics: Dict, filename: str = None):
        """Save tuning results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'nn_tuning_results_{timestamp}.json'

        filepath = os.path.join(self.output_dir, filename)

        # Prepare serializable data
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'device': DEVICE,
            'tensorflow_available': TF_AVAILABLE,
            'test_metrics': {k: float(v) for k, v in metrics.items()}
        }

        if 'best_config' in results:
            save_data['best_config'] = {
                k: v for k, v in results['best_config'].items()
                if not callable(v)
            }

        if 'results' in results and isinstance(results['results'], pd.DataFrame):
            results_csv = os.path.join(self.output_dir, 'nn_architecture_comparison.csv')
            results['results'].to_csv(results_csv, index=False)
            logger.info(f"Architecture comparison saved to: {results_csv}")

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        logger.info(f"Results saved to: {filepath}")

    def run_full_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        baseline_metrics: Dict = None,
                        use_tensorflow: bool = True) -> Tuple[Any, Dict]:
        """
        Run complete tuning pipeline

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            baseline_metrics: Baseline model metrics for comparison
            use_tensorflow: Whether to use TensorFlow (if available)

        Returns:
            Tuple of (best_model, complete_results)
        """
        # Default baseline (current Neural Network results)
        if baseline_metrics is None:
            baseline_metrics = {
                'MAE': 25.79,
                'RMSE': 208.92,
                'R2': 0.8346
            }

        # Choose tuning method
        if use_tensorflow and TF_AVAILABLE:
            best_model, results = self.tune_tensorflow(
                X_train, y_train, X_test, y_test
            )
        else:
            best_model, results = self.tune_sklearn(
                X_train, y_train
            )

        # Evaluate on test set
        metrics = self.evaluate_model(best_model, X_test, y_test)

        # Compare with baseline
        improvements = self.compare_with_baseline(baseline_metrics, metrics)

        # Save results
        self.save_results(results, metrics)

        complete_results = {
            'model': best_model,
            'params': self.best_params,
            'metrics': metrics,
            'improvements': improvements,
            'tuning_results': results
        }

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TUNING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Device used: {DEVICE}")
        logger.info(f"Best MAE: {metrics['MAE']:.4f} (baseline: {baseline_metrics['MAE']:.2f})")
        logger.info(f"Best RMSE: {metrics['RMSE']:.4f} (baseline: {baseline_metrics['RMSE']:.2f})")
        logger.info(f"Best R²: {metrics['R2']:.4f} (baseline: {baseline_metrics['R2']:.4f})")

        return best_model, complete_results


# Standalone execution
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from io_modulo import DataIOHandler
    from processing import DataProcessor
    from config import DATA_PARAMS, USABLE_COLUMNS, PIPELINE_PARAMS, DATABASE_PATHS
    from sklearn.model_selection import train_test_split

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('nn_tuning.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    print("\n" + "=" * 60)
    print("NEURAL NETWORK HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"TensorFlow available: {TF_AVAILABLE}")
    print(f"Device: {DEVICE}")
    print("=" * 60 + "\n")

    logger.info("Loading data...")

    # Load and prepare data
    io_handler = DataIOHandler()
    data = io_handler.load_data(DATABASE_PATHS['healthcare_data'])

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

    # Clip negative values
    y = y.clip(lower=0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=PIPELINE_PARAMS['random_state']
    )

    # Process features
    X_train = processor.process_pipeline(
        X_train, DATA_PARAMS, fit=True,
        select_usable=True, usable_columns=USABLE_COLUMNS
    )
    X_test = processor.process_pipeline(
        X_test, DATA_PARAMS, fit=False,
        select_usable=True, usable_columns=USABLE_COLUMNS
    )

    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")

    # Run tuning
    tuner = NeuralNetworkTuner(
        random_state=PIPELINE_PARAMS['random_state'],
        n_jobs=PIPELINE_PARAMS['n_jobs']
    )

    best_model, results = tuner.run_full_tuning(
        X_train, y_train,
        X_test, y_test,
        use_tensorflow=TF_AVAILABLE
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if results['params']:
        print(f"Best Architecture: {results['params'].get('name', 'N/A')}")
    print(f"MAE: {results['metrics']['MAE']:.4f}")
    print(f"RMSE: {results['metrics']['RMSE']:.4f}")
    print(f"R²: {results['metrics']['R2']:.4f}")
    print("=" * 60)
