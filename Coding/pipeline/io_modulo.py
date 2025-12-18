"""
I/O Module
Handles data loading, saving, and statistical description
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import json
import os
import logging

logger = logging.getLogger(__name__)

# ============ DATABASE FILE PATHS ============
DATABASE_FILE_PATHS = {
    'healthcare_data': r"C:\Users\Sebas\OneDrive\Desktop\Project Healthcare\Data base\Reporte de prueba.xlsx",
    'backup_path': 'data/healthcare_data_backup.csv'
}


class DataIOHandler:
    """Handles input/output operations for data"""

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        self._create_output_dir()

    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from various formats

        Args:
            filepath: Path to data file (.csv, .xlsx, .json, .parquet)

        Returns:
            Loaded DataFrame
        """
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        elif filepath.endswith('.parquet'):
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def save_data(self, df: pd.DataFrame, filepath: str, format: str = 'csv'):
        """
        Save DataFrame to file

        Args:
            df: DataFrame to save
            filepath: Output file path
            format: File format ('csv', 'xlsx', 'json', 'parquet')
        """
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'xlsx':
            df.to_excel(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records')
        elif format == 'parquet':
            df.to_parquet(filepath, index=False)

    def describe_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistical description of data

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with statistical information
        """
        description = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_stats': df.describe().to_dict(),
            'correlation_matrix': df.corr(numeric_only=True).to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_stats': {col: df[col].nunique() for col in df.select_dtypes(include=['object']).columns}
        }
        return description

    def save_description(self, description: Dict, filename: str = 'data_description.json'):
        """Save data description to file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(description, f, indent=2, default=str)

    def save_metrics(self, metrics: Dict, filename: str = 'metrics.json'):
        """Save metrics to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def log_pipeline_info(self, info: Dict, filename: str = 'pipeline_info.json'):
        """Save pipeline information"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2, default=str)

    def load_healthcare_database(self) -> pd.DataFrame:
        """
        Load the healthcare database from the configured path

        Returns:
            DataFrame with healthcare data

        Raises:
            FileNotFoundError: If database file is not found
            ValueError: If file format is not supported
        """
        database_path = DATABASE_FILE_PATHS['healthcare_data']

        # Check if file exists
        if not os.path.exists(database_path):
            logger.error(f"Database file not found: {database_path}")
            raise FileNotFoundError(f"Database file not found at: {database_path}")

        logger.info(f"Loading healthcare database from: {database_path}")

        try:
            # Load data using the generic load_data method
            df = self.load_data(database_path)
            logger.info(f"Healthcare database loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading healthcare database: {str(e)}")
            raise
