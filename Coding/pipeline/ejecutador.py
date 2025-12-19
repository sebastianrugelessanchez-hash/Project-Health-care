"""
Pipeline Orchestrator (Ejecutador)
Orchestrates the entire ML pipeline execution
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import DATA_PARAMS, PIPELINE_PARAMS, CV_PARAMS, ML_HYPERPARAMETERS, USABLE_COLUMNS, VALUE_FILTERING
from io_modulo import DataIOHandler
from processing import DataProcessor
from ml_classic import MLClassicModels
from ensemble_models import EnsembleModels
from crossvalidation import CrossValidator
from evaluation import ModelEvaluator
from reporting import PipelineReporter

# Configure logging
logging.basicConfig(
    level=PIPELINE_PARAMS.get('verbose', logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Main pipeline orchestrator"""

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        self.io_handler = DataIOHandler(output_dir)
        self.processor = DataProcessor()
        self.ml_models = MLClassicModels(
            random_state=PIPELINE_PARAMS['random_state'],
            n_jobs=PIPELINE_PARAMS['n_jobs']
        )
        self.ensemble = EnsembleModels(
            random_state=PIPELINE_PARAMS['random_state'],
            n_jobs=PIPELINE_PARAMS['n_jobs']
        )
        self.cv_validator = CrossValidator(
            n_splits=CV_PARAMS['n_splits'],
            shuffle=CV_PARAMS['shuffle'],
            random_state=CV_PARAMS['random_state'],
            stratified=CV_PARAMS['stratified']
        )
        self.evaluator = ModelEvaluator()
        self.reporter = PipelineReporter(output_dir)

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X = None
        self.y = None
        self.data = None
        self.pipeline_info = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from file"""
        logger.info(f"Loading data from {filepath}")
        self.data = self.io_handler.load_data(filepath)
        logger.info(f"Data loaded successfully. Shape: {self.data.shape}")

        # Save data description
        description = self.io_handler.describe_data(self.data)
        self.io_handler.save_description(description)

        self.pipeline_info['data_info'] = {
            'shape': str(self.data.shape),
            'columns': list(self.data.columns),
            'features': len(self.data.columns) - 1
        }

        return self.data

    def prepare_data(self, target_column) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data by separating features and target.
        NOTE: This method only separates X and y. Row cleaning happens in clean_rows().

        Args:
            target_column: Name of target column or list of target columns

        Returns:
            Tuple of (features, target)
        """
        # Handle multiple target columns (for multivariate regression)
        if isinstance(target_column, list):
            logger.info(f"Preparing data with target columns: {target_column}")
            target_cols = target_column
            # Create combined target (mean of the columns)
            self.y = self.data[target_cols].mean(axis=1)
            self.X = self.data.drop(columns=target_cols)
        else:
            logger.info(f"Preparing data with target column: {target_column}")
            self.y = self.data[target_column].copy()
            self.X = self.data.drop(columns=[target_column])

        logger.info(f"Features shape: {self.X.shape}, Target shape: {self.y.shape}")

        return self.X, self.y

    def clean_rows(self, config: Dict = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Clean rows BEFORE train/test split.
        Combines X and y, cleans rows, then separates them again.
        This ensures X and y stay aligned when rows are removed.

        Args:
            config: Configuration dictionary for value filtering

        Returns:
            Tuple of (cleaned_X, cleaned_y)
        """
        if config is None:
            config = DATA_PARAMS

        logger.info("Starting row cleaning (pre-split)")

        # Combine X and y into a single DataFrame for cleaning
        combined_df = self.X.copy()
        combined_df['__target__'] = self.y.copy()

        # Apply clean_rows from processor
        cleaned_combined, clean_stats = self.processor.clean_rows(combined_df, config)

        logger.info(f"Cleaning stats: {clean_stats}")

        # Separate X and y again
        self.X = cleaned_combined.drop(columns=['__target__']).reset_index(drop=True)
        self.y = cleaned_combined['__target__'].reset_index(drop=True)

        # Remove rows where target is NaN (AFTER processor cleaning)
        nan_count_before = self.y.isnull().sum()
        if nan_count_before > 0:
            valid_rows = ~self.y.isnull()
            self.X = self.X[valid_rows].reset_index(drop=True)
            self.y = self.y[valid_rows].reset_index(drop=True)
            logger.info(f"Removed {nan_count_before} rows with NaN target values")

        # Handle negative target values
        negative_count = (self.y < 0).sum()
        if negative_count > 0:
            logger.info(f"Converting {negative_count} negative values to 0 in target")
            self.y = self.y.clip(lower=0)

        rows_removed = combined_df.shape[0] - len(self.X)
        logger.info(f"Total rows removed during cleaning: {rows_removed}")
        logger.info(f"Features shape: {self.X.shape}, Target shape: {self.y.shape}")

        self.pipeline_info['cleaning_stats'] = clean_stats

        return self.X, self.y

    def split_data(self, val_size: float = 0.2, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.

        Flow:
        - Train (60%): Used for model training
        - Validation (20%): Used for model selection (comparing models)
        - Test (20%): Used ONLY for final evaluation (never seen during training/selection)

        Args:
            val_size: Validation set size (default 0.2)
            test_size: Test set size (default 0.2)

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data: train={1-val_size-test_size:.0%}, val={val_size:.0%}, test={test_size:.0%}")

        # First split: separate test set (hold-out for final evaluation)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=PIPELINE_PARAMS['random_state']
        )

        # Second split: separate validation from training
        # Adjust val_size relative to remaining data
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=PIPELINE_PARAMS['random_state']
        )

        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Validation set: {self.X_val.shape}")
        logger.info(f"Test set: {self.X_test.shape} (hold-out for final evaluation)")

        self.pipeline_info['data_info']['train_size'] = self.X_train.shape[0]
        self.pipeline_info['data_info']['val_size'] = self.X_val.shape[0]
        self.pipeline_info['data_info']['test_size'] = self.X_test.shape[0]

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def process_data(self, processing_config: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process data according to configuration

        Args:
            processing_config: Custom processing configuration

        Returns:
            Tuple of (processed_X_train, processed_X_val, processed_X_test)
        """
        if processing_config is None:
            processing_config = DATA_PARAMS

        logger.info("Starting data processing")

        # Prepare full config with value filtering
        full_config = processing_config.copy()
        full_config['value_filtering'] = VALUE_FILTERING

        # Process training data (fit transformers)
        self.X_train = self.processor.process_pipeline(
            self.X_train, full_config, fit=True,
            select_usable=True,
            usable_columns=USABLE_COLUMNS,
            filter_values_flag=False
        )
        logger.info(f"Processed X_train shape: {self.X_train.shape}")

        # Process validation data (using fitted transformers)
        self.X_val = self.processor.process_pipeline(
            self.X_val, full_config, fit=False,
            select_usable=True,
            usable_columns=USABLE_COLUMNS,
            filter_values_flag=False
        )
        logger.info(f"Processed X_val shape: {self.X_val.shape}")

        # Process test data (using fitted transformers)
        self.X_test = self.processor.process_pipeline(
            self.X_test, full_config, fit=False,
            select_usable=True,
            usable_columns=USABLE_COLUMNS,
            filter_values_flag=False
        )
        logger.info(f"Processed X_test shape: {self.X_test.shape}")

        self.pipeline_info['processing_info'] = full_config

        return self.X_train, self.X_val, self.X_test

    def train_classic_models(self, tune: bool = True) -> Dict[str, Any]:
        """
        Train classical ML models

        Args:
            tune: Whether to perform hyperparameter tuning

        Returns:
            Dictionary of trained models
        """
        logger.info("Training classical ML models")

        models = {}
        hyperparams = ML_HYPERPARAMETERS if tune else {}

        # Logistic Regression
        model, info = self.ml_models.logistic_regression(
            self.X_train, self.y_train, self.X_test,
            params=hyperparams.get('logistic_regression'),
            tune=tune
        )
        models['Logistic Regression'] = model

        # Random Forest
        model, info = self.ml_models.random_forest(
            self.X_train, self.y_train, self.X_test,
            params=hyperparams.get('random_forest'),
            tune=tune
        )
        models['Random Forest'] = model

        # SVM
        model, info = self.ml_models.support_vector_machine(
            self.X_train, self.y_train, self.X_test,
            params=hyperparams.get('svm'),
            tune=tune
        )
        models['SVM'] = model

        # Gradient Boosting
        model, info = self.ml_models.gradient_boosting(
            self.X_train, self.y_train, self.X_test,
            params=hyperparams.get('gradient_boosting'),
            tune=tune
        )
        models['Gradient Boosting'] = model

        # Neural Network
        model, info = self.ml_models.neural_network(
            self.X_train, self.y_train, self.X_test,
            params=hyperparams.get('neural_network'),
            tune=tune
        )
        models['Neural Network'] = model

        # MLP Regressor
        model, info = self.ml_models.mlp_regressor(
            self.X_train, self.y_train, self.X_test,
            params=hyperparams.get('mlp_regressor'),
            tune=tune
        )
        models['MLP Regressor'] = model

        # Train ensemble models
        try:
            logger.info("Training ensemble models")

            # Get trained base models for ensemble
            base_models = [
                ('linear_reg', self.ml_models.get_model('logistic_regression')),
                ('rf', self.ml_models.get_model('random_forest')),
                ('svm', self.ml_models.get_model('svm')),
                ('gb', self.ml_models.get_model('gradient_boosting'))
            ]

            # Voting Regressor
            voting_model = self.ensemble.voting_regressor(base_models)
            voting_model.fit(self.X_train, self.y_train)
            models['Voting Regressor'] = voting_model
            logger.info("Voting Regressor trained successfully")

            # Stacking Regressor
            stacking_model = self.ensemble.stacking_regressor(base_models)
            stacking_model.fit(self.X_train, self.y_train)
            models['Stacking Regressor'] = stacking_model
            logger.info("Stacking Regressor trained successfully")

            # AdaBoost Regressor
            ada_model = self.ensemble.ada_boost(self.X_train, self.y_train)
            models['AdaBoost Regressor'] = ada_model
            logger.info("AdaBoost Regressor trained successfully")

            # Bagging Regressor
            bagging_model = self.ensemble.bagging_regressor(self.X_train, self.y_train)
            models['Bagging Regressor'] = bagging_model
            logger.info("Bagging Regressor trained successfully")

        except Exception as e:
            logger.warning(f"Ensemble training failed: {str(e)}. Continuing with classical models only.")

        logger.info(f"Trained {len(models)} total models (classical + ensemble)")
        self.pipeline_info['models_info'] = list(models.keys())

        return models

    def evaluate_models_on_validation(self, models: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate all models on VALIDATION set for model selection.
        This is used to compare models and select the best one.

        Args:
            models: Dictionary of trained models

        Returns:
            DataFrame with evaluation results
        """
        logger.info("Evaluating models on VALIDATION set (for model selection)")

        results = self.evaluator.evaluate_multiple_models(models, self.X_val, self.y_val)

        logger.info("\nModel Comparison on Validation Set:")
        logger.info(results.to_string())

        return results

    def final_evaluation(self, best_model: Any, best_model_name: str) -> pd.DataFrame:
        """
        Final evaluation on TEST set (hold-out).
        This should only be called ONCE with the selected best model.

        Args:
            best_model: The best model selected from validation
            best_model_name: Name of the best model

        Returns:
            DataFrame with final test evaluation results
        """
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION ON TEST SET (Hold-out)")
        logger.info("=" * 60)

        # Create a new evaluator instance for final test results
        test_evaluator = ModelEvaluator()
        metrics = test_evaluator.evaluate_model(best_model, self.X_test, self.y_test, best_model_name)

        results = pd.DataFrame([{
            'Model': best_model_name,
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse'],
            'R2-Score': metrics['r2'],
            'MSE': metrics['mse'],
            'MedAE': metrics['medae'],
            'EVS': metrics['evs'],
            'MAPE': metrics['mape']
        }])

        # Save final test results
        self.reporter.save_model_comparison(results, filename='final_test_results.csv')

        return results

    def perform_cross_validation(self, model: Any, model_name: str = 'Model') -> Dict[str, Any]:
        """
        Perform cross-validation on a model

        Args:
            model: Model to evaluate
            model_name: Name of the model

        Returns:
            Cross-validation results
        """
        logger.info(f"Performing cross-validation on {model_name}")

        results = self.cv_validator.cross_validate_model(model, self.X_train, self.y_train)

        logger.info(f"Cross-validation completed for {model_name}")

        return results

    def run_complete_pipeline(self, data_path: str, target_column: str,
                             tune_models: bool = True,
                             custom_config: Dict = None) -> Dict[str, Any]:
        """
        Execute complete ML pipeline with proper train/val/test split.

        Flow:
        1. Load and clean data
        2. Split into train (60%) / validation (20%) / test (20%)
        3. Train models on train set
        4. Select best model using VALIDATION set
        5. Final evaluation on TEST set (only once)

        Args:
            data_path: Path to input data
            target_column: Name of target column
            tune_models: Whether to tune hyperparameters
            custom_config: Custom configuration (optional)

        Returns:
            Dictionary with pipeline results
        """
        logger.info("=" * 80)
        logger.info("STARTING ML PIPELINE EXECUTION")
        logger.info("=" * 80)

        try:
            # Load and prepare data
            self.load_data(data_path)
            self.prepare_data(target_column)

            # IMPORTANT: Clean rows BEFORE split to avoid data leakage
            processing_config = custom_config if custom_config else DATA_PARAMS
            self.clean_rows(processing_config)

            # Split cleaned data into train/val/test
            self.split_data()

            # Process data (post-split, no row removal)
            self.process_data(processing_config)

            # Train models on TRAIN set
            models = self.train_classic_models(tune=tune_models)

            # Evaluate models on VALIDATION set (for model selection)
            validation_results = self.evaluate_models_on_validation(models)
            self.reporter.save_model_comparison(validation_results, filename='validation_comparison.csv')

            # Select best model based on validation performance
            best_model_name = validation_results.iloc[0]['Model']
            best_model = models[best_model_name]
            logger.info(f"\nBest model selected (by validation MAE): {best_model_name}")

            # Cross-validation on training data (for robustness check)
            cv_results = self.perform_cross_validation(best_model, best_model_name)

            # FINAL EVALUATION on TEST set (hold-out, only once)
            final_test_results = self.final_evaluation(best_model, best_model_name)

            # Generate reports with primary metrics from FINAL TEST
            final_metrics = final_test_results.iloc[0]
            val_metrics = validation_results.iloc[0]
            pipeline_summary = self.reporter.create_pipeline_summary({
                'total_samples': len(self.data),
                'features': self.X.shape[1],
                'target_classes': len(np.unique(self.y)),
                'processing_config': DATA_PARAMS,
                'models': list(models.keys()),
                'best_model': best_model_name,
                'best_mae': final_metrics['MAE'],
                'best_rmse': final_metrics['RMSE'],
                'best_r2': final_metrics['R2-Score']
            })

            logger.info(pipeline_summary)

            # Save validation results
            self.reporter.save_evaluation_results(self.evaluator.results)

            logger.info("=" * 80)
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            return {
                'models': models,
                'validation_results': validation_results,
                'final_test_results': final_test_results,
                'cv_results': cv_results,
                'best_model': best_model,
                'best_model_name': best_model_name,
                'evaluation_results': final_test_results  # For backwards compatibility
            }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    pipeline = PipelineOrchestrator(output_dir='results')

    # Example: pipeline.run_complete_pipeline('data.csv', 'target_column')
    logger.info("Pipeline module loaded successfully")