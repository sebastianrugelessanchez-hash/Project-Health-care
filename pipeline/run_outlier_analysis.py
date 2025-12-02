"""
Run Outlier Analysis
Script to analyze outliers in your healthcare dataset
"""

import sys
import logging
from pathlib import Path
import pandas as pd

from outlier_analysis import OutlierAnalyzer
from processing import DataProcessor
from config import DATABASE_PATHS, USABLE_COLUMNS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from Excel or CSV file"""
    if filepath.endswith('.xlsx'):
        return pd.read_excel(filepath)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def main():
    """Main entry point for outlier analysis"""

    # Load data
    data_path = DATABASE_PATHS['healthcare_data']
    logger.info(f"Loading data from: {data_path}")

    try:
        df = load_data(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        sys.exit(1)

    # Clean column names
    processor = DataProcessor()
    df = processor.normalize_column_names(df)

    # Select only usable columns
    df_analysis = df[[col for col in df.columns if any(
        col.lower().replace(' ', '').replace('-', '').replace('_', '') ==
        uc.lower().replace(' ', '').replace('-', '').replace('_', '')
        for uc in USABLE_COLUMNS
    )]]

    if df_analysis.shape[1] == 0:
        logger.warning("No usable columns found. Using all numeric columns instead.")
        df_analysis = df.select_dtypes(include=['number'])

    logger.info(f"Analyzing {df_analysis.shape[1]} columns: {list(df_analysis.columns)}")

    # Create analyzer and generate report
    analyzer = OutlierAnalyzer(output_dir='outlier_analysis')

    print("\n" + "="*100)
    print("STARTING OUTLIER ANALYSIS")
    print("="*100)
    print(f"Dataset: {data_path}")
    print(f"Rows: {len(df_analysis)}, Columns: {df_analysis.shape[1]}")
    print("="*100 + "\n")

    # Generate comprehensive report
    analyzer.generate_report(df_analysis)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print("Check the 'outlier_analysis' folder for detailed reports and visualizations.")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
