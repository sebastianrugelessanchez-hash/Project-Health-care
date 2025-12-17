"""
Model Evaluation Module
Evaluates regression models using various metrics and test sets
"""

from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            mean_absolute_percentage_error, median_absolute_error,
                            explained_variance_score)
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation utilities for regression"""

    def __init__(self):
        self.results = {}

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = 'Model') -> Dict[str, Any]:
        """
        Comprehensive model evaluation for regression

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)

        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)

        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except (ValueError, ZeroDivisionError):
            mape = None

        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'medae': medae,
            'r2': r2,
            'evs': evs,
            'mape': mape
        }

        self.results[model_name] = metrics

        # Log results
        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*50}")
        logger.info(f"MSE:  {metrics['mse']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE:  {metrics['mae']:.4f}")
        logger.info(f"MedAE: {metrics['medae']:.4f}")
        logger.info(f"R²:   {metrics['r2']:.4f}")
        logger.info(f"EVS:  {metrics['evs']:.4f}")
        if mape is not None:
            logger.info(f"MAPE: {metrics['mape']:.4f}")

        return metrics

    def evaluate_multiple_models(self, models: Dict[str, Any], X_test: np.ndarray,
                                y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate multiple models and compare results

        Args:
            models: Dictionary of model names and models
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with comparison results
        """
        results_list = []

        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            results_list.append({
                'Model': name,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MedAE': metrics['medae'],
                'R2-Score': metrics['r2'],
                'EVS': metrics['evs'],
                'MAPE': metrics['mape']
            })

        return pd.DataFrame(results_list).sort_values('R2-Score', ascending=False)

    def compare_results(self) -> pd.DataFrame:
        """Compare all evaluated regression models"""
        if not self.results:
            return pd.DataFrame()

        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MedAE': metrics['medae'],
                'R2-Score': metrics['r2'],
                'EVS': metrics['evs'],
                'MAPE': metrics['mape']
            })

        return pd.DataFrame(comparison_data).sort_values('R2-Score', ascending=False)

    def get_best_model(self, metric: str = 'r2') -> str:
        """
        Get the name of the best performing model

        Args:
            metric: Metric to use for comparison ('r2', 'mse', 'rmse', 'mae')

        Returns:
            Name of the best model
        """
        if not self.results:
            return None

        if metric in ['mse', 'rmse', 'mae', 'medae', 'mape']:
            # Lower is better
            best_model = min(self.results.items(),
                           key=lambda x: x[1][metric] if x[1][metric] is not None else float('inf'))
        else:
            # Higher is better (r2, evs)
            best_model = max(self.results.items(),
                           key=lambda x: x[1][metric] if x[1][metric] is not None else float('-inf'))

        return best_model[0]

    def get_prediction_errors(self, model: Any, X_test: np.ndarray,
                             y_test: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed prediction error analysis

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with error statistics
        """
        y_pred = model.predict(X_test)
        errors = y_test - y_pred

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'median_error': np.median(errors),
            'percentile_5': np.percentile(errors, 5),
            'percentile_95': np.percentile(errors, 95)
        }
