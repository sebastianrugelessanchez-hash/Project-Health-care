"""
Ensemble Models
Combines multiple models for improved predictions (Regression)
"""

from sklearn.ensemble import VotingRegressor, StackingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class EnsembleModels:
    """Ensemble learning methods"""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.ensemble_models = {}

    def voting_regressor(self, estimators: List[Tuple[str, Any]]) -> Any:
        """
        Create Voting Regressor

        Args:
            estimators: List of (name, estimator) tuples

        Returns:
            Fitted VotingRegressor
        """
        model = VotingRegressor(estimators=estimators, n_jobs=self.n_jobs)
        self.ensemble_models['voting_regressor'] = model
        logger.info(f"Voting Regressor created with {len(estimators)} estimators")
        return model

    def stacking_regressor(self, estimators: List[Tuple[str, Any]],
                           final_estimator: Any = None) -> Any:
        """
        Create Stacking Regressor

        Args:
            estimators: List of (name, estimator) tuples
            final_estimator: Meta-learner (default: LinearRegression)

        Returns:
            Fitted StackingRegressor
        """
        if final_estimator is None:
            final_estimator = LinearRegression()

        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5
        )
        self.ensemble_models['stacking_regressor'] = model
        logger.info(f"Stacking Regressor created with {len(estimators)} estimators")
        return model

    def ada_boost(self, X_train, y_train, n_estimators: int = 50,
                  learning_rate: float = 1.0) -> Any:
        """
        AdaBoost Regressor

        Args:
            X_train: Training features
            y_train: Training targets
            n_estimators: Number of estimators
            learning_rate: Learning rate shrinking

        Returns:
            Fitted AdaBoostRegressor
        """
        model = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.ensemble_models['ada_boost'] = model
        logger.info("AdaBoost Regressor trained successfully")
        return model

    def bagging_regressor(self, X_train, y_train, n_estimators: int = 10,
                          max_samples: int = 1.0) -> Any:
        """
        Bagging Regressor

        Args:
            X_train: Training features
            y_train: Training targets
            n_estimators: Number of estimators
            max_samples: Proportion of samples for each estimator

        Returns:
            Fitted BaggingRegressor
        """
        model = BaggingRegressor(
            n_estimators=n_estimators,
            max_samples=max_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.ensemble_models['bagging_regressor'] = model
        logger.info("Bagging Regressor trained successfully")
        return model

    def get_ensemble_model(self, name: str) -> Any:
        """Get ensemble model by name"""
        return self.ensemble_models.get(name)

    def get_all_ensemble_models(self) -> Dict[str, Any]:
        """Get all ensemble models"""
        return self.ensemble_models
