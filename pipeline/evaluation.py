"""
Model Evaluation Module
Evaluates models using various metrics and test sets
"""

from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            mean_absolute_percentage_error)
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation utilities"""

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

        try:
            mape = mean_absolute_percentage_error(y_test, y_pred)
        except:
            mape = None

        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
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
        logger.info(f"R²:   {metrics['r2']:.4f}")
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
                'R2-Score': metrics['r2'],
                'MAPE': metrics['mape']
            })

        return pd.DataFrame(results_list).sort_values('R2-Score', ascending=False)

    def get_detailed_report(self, model: Any, X_test: np.ndarray,
                           y_test: np.ndarray, model_name: str = 'Model') -> str:
        """Get detailed classification report"""
        y_pred = model.predict(X_test)
        return classification_report(y_test, y_pred)

    def get_confusion_matrix(self, model: Any, X_test: np.ndarray,
                            y_test: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def get_roc_curve_data(self, model: Any, X_test: np.ndarray,
                          y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get ROC curve data (only for binary classification)

        Returns:
            Tuple of (fpr, tpr, auc_score)
        """
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability prediction")
            return None, None, None

        y_proba = model.predict_proba(X_test)

        if len(np.unique(y_test)) != 2:
            logger.warning("ROC curve only supported for binary classification")
            return None, None, None

        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        auc_score = auc(fpr, tpr)

        return fpr, tpr, auc_score

    def get_precision_recall_curve_data(self, model: Any, X_test: np.ndarray,
                                       y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get precision-recall curve data (only for binary classification)"""
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model does not support probability prediction")
            return None, None, None

        y_proba = model.predict_proba(X_test)

        if len(np.unique(y_test)) != 2:
            logger.warning("Precision-recall curve only supported for binary classification")
            return None, None, None

        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        avg_precision = auc(recall, precision)

        return precision, recall, avg_precision

    def compare_results(self) -> pd.DataFrame:
        """Compare all evaluated models"""
        if not self.results:
            return pd.DataFrame()

        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'Kappa': metrics['cohen_kappa']
            })

        return pd.DataFrame(comparison_data).sort_values('F1', ascending=False)
