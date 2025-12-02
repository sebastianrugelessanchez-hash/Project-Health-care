"""
Cross-Validation Module
Implements cross-validation strategies and model evaluation
"""

from sklearn.model_selection import (StratifiedKFold, KFold, cross_val_score,
                                     cross_validate, cross_val_predict)
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CrossValidator:
    """Cross-validation utilities"""

    def __init__(self, n_splits: int = 5, shuffle: bool = True,
                 random_state: int = 42, stratified: bool = True):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified

    def get_cv_splitter(self):
        """Get cross-validation splitter"""
        if self.stratified:
            return StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                                  random_state=self.random_state)
        else:
            return KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                        random_state=self.random_state)

    def cross_validate_model(self, model: Any, X_train: np.ndarray,
                            y_train: np.ndarray, scoring: Dict = None) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation on model

        Args:
            model: Sklearn model
            X_train: Training features
            y_train: Training labels
            scoring: Dictionary of scoring functions

        Returns:
            Dictionary with cross-validation results
        """
        if scoring is None:
            scoring = {
                'r2': 'r2',
                'neg_mean_squared_error': 'neg_mean_squared_error',
                'neg_mean_absolute_error': 'neg_mean_absolute_error'
            }

        cv = self.get_cv_splitter()
        results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                               return_train_score=True, n_jobs=-1)

        # Log results
        for metric, scores in results.items():
            if 'test' in metric:
                metric_name = metric.replace('test_', '')
                logger.info(f"{metric_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return results

    def cross_val_predict_model(self, model: Any, X_train: np.ndarray,
                               y_train: np.ndarray) -> np.ndarray:
        """
        Get cross-validated predictions

        Args:
            model: Sklearn model
            X_train: Training features
            y_train: Training labels

        Returns:
            Cross-validated predictions
        """
        cv = self.get_cv_splitter()
        predictions = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
        return predictions

    def get_cv_scores(self, model: Any, X_train: np.ndarray,
                     y_train: np.ndarray, metric: str = 'accuracy') -> Tuple[np.ndarray, float, float]:
        """
        Get cross-validation scores for a specific metric

        Args:
            model: Sklearn model
            X_train: Training features
            y_train: Training labels
            metric: Scoring metric

        Returns:
            Tuple of (scores, mean, std)
        """
        cv = self.get_cv_splitter()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1)
        return scores, scores.mean(), scores.std()

    def compare_models(self, models: Dict[str, Any], X_train: np.ndarray,
                      y_train: np.ndarray, metric: str = 'accuracy') -> pd.DataFrame:
        """
        Compare multiple models using cross-validation

        Args:
            models: Dictionary of model names and models
            X_train: Training features
            y_train: Training labels
            metric: Scoring metric

        Returns:
            DataFrame with comparison results
        """
        results = []

        for name, model in models.items():
            scores, mean, std = self.get_cv_scores(model, X_train, y_train, metric)
            results.append({
                'Model': name,
                'Mean': mean,
                'Std': std,
                'Min': scores.min(),
                'Max': scores.max()
            })

            logger.info(f"{name}: {mean:.4f} (+/- {std:.4f})")

        return pd.DataFrame(results).sort_values('Mean', ascending=False)

    def evaluate_cv_folds(self, model: Any, X_train: np.ndarray,
                         y_train: np.ndarray, metrics: List[str] = None) -> pd.DataFrame:
        """
        Evaluate model across all CV folds

        Args:
            model: Sklearn model
            X_train: Training features
            y_train: Training labels
            metrics: List of metrics to evaluate

        Returns:
            DataFrame with per-fold results
        """
        if metrics is None:
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        cv = self.get_cv_splitter()
        scoring = {metric: metric for metric in metrics}

        results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                               return_train_score=True)

        fold_results = []
        for fold in range(self.n_splits):
            fold_data = {'Fold': fold + 1}
            for metric in metrics:
                fold_data[f'test_{metric}'] = results[f'test_{metric}'][fold]
                fold_data[f'train_{metric}'] = results[f'train_{metric}'][fold]
            fold_results.append(fold_data)

        return pd.DataFrame(fold_results)
