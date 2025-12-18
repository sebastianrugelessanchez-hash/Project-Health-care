"""
Machine Learning Classic Models
Implements various classical ML algorithms
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class MLClassicModels:
    """Wrapper class for classical ML models"""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.best_params = {}

    def _train_with_chunks(self, model, X_train, y_train, chunk_size: int = 2000) -> Any:
        """
        Train model with chunked data to reduce memory usage.
        Uses partial_fit for models that support it (SGDRegressor, MLPRegressor, etc.)

        Args:
            model: Model with partial_fit method
            X_train: Training features
            y_train: Training labels
            chunk_size: Size of each chunk

        Returns:
            Trained model
        """
        logger.info(f"Training {model.__class__.__name__} with chunks (size={chunk_size})")

        n_chunks = int(np.ceil(len(X_train) / chunk_size))
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(X_train))

            X_chunk = X_train[start_idx:end_idx]
            y_chunk = y_train[start_idx:end_idx]

            if i == 0:
                # First chunk: fit
                model.fit(X_chunk, y_chunk)
            else:
                # Subsequent chunks: partial_fit
                model.partial_fit(X_chunk, y_chunk)

            logger.info(f"  Chunk {i+1}/{n_chunks} processed ({len(X_chunk)} samples)")

        return model

    def logistic_regression(self, X_train, y_train, X_test, params: Dict = None,
                           tune: bool = True) -> Tuple[Any, Dict]:
        """
        Linear Regression for regression tasks

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            params: Hyperparameter grid for tuning
            tune: Whether to perform grid search

        Returns:
            Trained model and results
        """
        if params is None:
            params = {}

        if tune:
            model = LinearRegression()
            grid = GridSearchCV(model, params, cv=5, n_jobs=self.n_jobs, verbose=1) if params else model
            if params:
                grid.fit(X_train, y_train)
                self.best_params['logistic_regression'] = grid.best_params_
                self.models['logistic_regression'] = grid.best_estimator_
                logger.info(f"Linear Regression best params: {grid.best_params_}")
                return grid.best_estimator_, {'best_score': grid.best_score_}
            else:
                model.fit(X_train, y_train)
                self.models['logistic_regression'] = model
                return model, {}
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models['logistic_regression'] = model
            return model, {}

    def random_forest(self, X_train, y_train, X_test, params: Dict = None,
                     tune: bool = True) -> Tuple[Any, Dict]:
        """Random Forest Regressor with optional hyperparameter tuning"""
        if params is None:
            params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

        if tune:
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
            grid = GridSearchCV(model, params, cv=5, n_jobs=self.n_jobs, verbose=1)
            grid.fit(X_train, y_train)
            self.best_params['random_forest'] = grid.best_params_
            self.models['random_forest'] = grid.best_estimator_
            logger.info(f"Random Forest best params: {grid.best_params_}")
            return grid.best_estimator_, {'best_score': grid.best_score_}
        else:
            model = RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
            model.fit(X_train, y_train)
            self.models['random_forest'] = model
            return model, {}

    def support_vector_machine(self, X_train, y_train, X_test, params: Dict = None,
                              tune: bool = True) -> Tuple[Any, Dict]:
        """Support Vector Regressor with optional hyperparameter tuning"""
        if params is None:
            params = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }

        if tune:
            model = SVR()
            grid = GridSearchCV(model, params, cv=5, n_jobs=self.n_jobs, verbose=1)
            grid.fit(X_train, y_train)
            self.best_params['svm'] = grid.best_params_
            self.models['svm'] = grid.best_estimator_
            logger.info(f"SVR best params: {grid.best_params_}")
            return grid.best_estimator_, {'best_score': grid.best_score_}
        else:
            model = SVR()
            model.fit(X_train, y_train)
            self.models['svm'] = model
            return model, {}

    def gradient_boosting(self, X_train, y_train, X_test, params: Dict = None,
                         tune: bool = True) -> Tuple[Any, Dict]:
        """
        Gradient Boosting Regressor with memory-efficient chunked training.

        Hybrid approach:
        - If tune=True: GridSearch on 20% sample for hyperparameter tuning
        - Train best model on 100% of data using SGDRegressor (supports partial_fit)

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (not used for training)
            params: Hyperparameter grid for tuning
            tune: Whether to perform hyperparameter tuning

        Returns:
            Trained model and results dictionary
        """
        if params is None:
            params = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            }

        logger.info(f"Gradient Boosting (tune={tune}, dataset_size={len(X_train)})")

        if tune and params:
            model = GradientBoostingRegressor(random_state=self.random_state)
            grid = GridSearchCV(model, params, cv=3, n_jobs=self.n_jobs, verbose=1)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            self.best_params['gradient_boosting'] = grid.best_params_
            logger.info(f"  Best params found: {grid.best_params_}")
        else:
            best_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state
            )
            best_model.fit(X_train, y_train)
            logger.info("  Using default GradientBoostingRegressor parameters")

        self.models['gradient_boosting'] = best_model
        logger.info("Gradient Boosting training complete")

        return best_model, {'best_score': getattr(grid, 'best_score_', None) if tune else None}

    def neural_network(self, X_train, y_train, X_test, params: Dict = None,
                      tune: bool = True) -> Tuple[Any, Dict]:
        """Neural Network (MLP) Regressor with optional hyperparameter tuning"""
        if params is None:
            params = {
                'hidden_layer_sizes': [(300,), (400, 100), (100, 50, 25)],
                'activation': ['relu', 'tanh', 'logistic'],
                'learning_rate_init': [0.001, 0.01],
                'alpha': [0.0001, 0.001],
                'max_iter': [1000]
            }

        if tune:
            model = MLPRegressor(random_state=self.random_state)
            grid = GridSearchCV(model, params, cv=5, n_jobs=self.n_jobs, verbose=1)
            grid.fit(X_train, y_train)
            self.best_params['neural_network'] = grid.best_params_
            self.models['neural_network'] = grid.best_estimator_
            logger.info(f"Neural Network best params: {grid.best_params_}")
            return grid.best_estimator_, {'best_score': grid.best_score_}
        else:
            model = MLPRegressor(random_state=self.random_state, max_iter=1000)
            model.fit(X_train, y_train)
            self.models['neural_network'] = model
            return model, {}

    def mlp_regressor(self, X_train, y_train, X_test, params: Dict = None,
                     tune: bool = True) -> Tuple[Any, Dict]:
        """
        MLP Regressor with memory-efficient chunked training.

        Hybrid approach:
        - If tune=True: GridSearch on 20% sample for hyperparameter tuning
        - Train best model on 100% of data using chunked partial_fit

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (not used for training)
            params: Hyperparameter grid for tuning
            tune: Whether to perform hyperparameter tuning

        Returns:
            Trained model and results dictionary
        """
        if params is None:
            params = {
                'hidden_layer_sizes': [(100,), (200,), (300,)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01],
                'alpha': [0.0001, 0.001]
            }

        logger.info(f"MLP Regressor (tune={tune}, dataset_size={len(X_train)})")

        if tune and params:
            # GridSearch on 20% sample for efficiency
            sample_size = max(len(X_train) // 5, 1000)  # At least 1000 samples
            X_sample = X_train[:sample_size]
            y_sample = y_train[:sample_size]

            logger.info(f"  GridSearch on {len(X_sample)} sample (20% of data)")
            model = MLPRegressor(random_state=self.random_state, max_iter=500, warm_start=True)
            grid = GridSearchCV(model, params, cv=3, n_jobs=self.n_jobs, verbose=1)
            grid.fit(X_sample, y_sample)

            best_model = grid.best_estimator_
            self.best_params['mlp_regressor'] = grid.best_params_
            logger.info(f"  Best params found: {grid.best_params_}")
        else:
            # Use default MLPRegressor with warm_start for chunking
            best_model = MLPRegressor(
                random_state=self.random_state,
                max_iter=500,
                warm_start=True,
                learning_rate_init=0.001,
                activation='relu'
            )
            logger.info("  Using default MLPRegressor parameters")

        # Train on 100% of data with chunking
        logger.info(f"  Training on {len(X_train)} full dataset with chunks")
        best_model = self._train_with_chunks(best_model, X_train, y_train, chunk_size=2000)

        self.models['mlp_regressor'] = best_model
        logger.info("MLP Regressor training complete")

        return best_model, {'best_score': getattr(grid, 'best_score_', None) if tune else None}

    def get_model(self, name: str):
        """Get trained model by name"""
        return self.models.get(name)

    def get_all_models(self) -> Dict:
        """Get all trained models"""
        return self.models
