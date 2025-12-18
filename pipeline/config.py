"""
Configuration Module
Defines core rules, evaluation metrics, and global parameters for the pipeline
"""

# ============ EVALUATION METRICS ============
# Primary metrics for regression evaluation (in order of importance)
PRIMARY_METRICS = ['MAE', 'RMSE', 'R2-Score']

METRICS = {
    'mae': True,       # Mean Absolute Error - primary
    'rmse': True,      # Root Mean Squared Error - primary
    'r2': True,        # R-squared - primary
    'mse': True,       # Mean Squared Error
    'medae': True,     # Median Absolute Error
    'evs': True,       # Explained Variance Score
    'mape': True       # Mean Absolute Percentage Error
}

# ============ CROSS-VALIDATION PARAMETERS ============
CV_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42,
    'stratified': False  # Regression does not use stratified CV
}

# ============ MODEL HYPERPARAMETERS ============
# Hyperparameter grids for GridSearchCV tuning
ML_HYPERPARAMETERS = {
    'logistic_regression': {
        # LinearRegression doesn't have hyperparameters to tune
        # Using empty dict to skip tuning for this model
    },
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    },
    'neural_network': {
        'hidden_layer_sizes': [(300,), (400, 100), (100, 50, 25)],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001],
        'activation': ['relu', 'tanh']
    },
    'mlp_regressor': {
        'hidden_layer_sizes': [(100,), (200,), (300,), (100, 50), (200, 100)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.0001, 0.001, 0.01]
    }
}

# ============ DATA PROCESSING PARAMETERS ============
DATA_PARAMS = {
    'train_test_split': 0.8,
    'handle_missing_values': 'mean',  # 'mean', 'median', 'drop', 'forward_fill'
    'outlier_method': None,  # Disabled: 'iqr', 'zscore', 'isolation_forest'
    'feature_scaling': 'standardscaler'  # 'standardscaler', 'minmaxscaler', 'robust'
}

# ============ USABLE COLUMNS & VALUE FILTERING ============
# Features for regression model (2024-1 through 2025-8, plus 2025-7 and 2025-8 as features)
# Target is 2025-9
USABLE_COLUMNS = [
    '2024-1', '2024-2', '2024-3', '2024-4', '2024-5', '2024-6',
    '2024-7', '2024-8', '2024-9', '2024-10', '2024-11', '2024-12',
    '2025-1', '2025-2', '2025-3', '2025-4', '2025-5', '2025-6',
    '2025-7', '2025-8'
]

VALUE_FILTERING = {
    'remove_empty': False,          # Don't remove rows with NaN (will be handled by imputation)
    'remove_negative': False,       # Don't remove negative values (keep all data for regression)
    'value_range': None,            # No value range filtering
    'normalize_column_names': True  # Normalize column names to match USABLE_COLUMNS
}

# ============ PIPELINE EXECUTION PARAMETERS ============
PIPELINE_PARAMS = {
    'random_state': 42,
    'n_jobs': -1,  # Use all available processors
    'verbose': 2,
    'save_models': True,
    'output_dir': 'results'
}

# ============ FILE PATHS & DATABASE CONFIGURATION ============
FILE_PATHS = {
    'data_dir': 'data',                           # Input data directory
    'results_dir': 'results',                     # Output results directory
    'models_dir': 'models',                       # Trained models directory
    'logs_dir': 'logs',                           # Logs directory
}

DATABASE_PATHS = {
    'healthcare_data': '/Users/sebas.12/Desktop/Proyectos/Project Healthcare/Data base/Reporte de prueba.xlsx',  # Main healthcare dataset
    'training_data': 'data/training_data.csv',                  # Training dataset
    'validation_data': 'data/validation_data.csv',              # Validation dataset
    'test_data': 'data/test_data.csv',                          # Test dataset
}

# ============ LOGGING CONFIGURATION ============
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'pipeline.log'
}
