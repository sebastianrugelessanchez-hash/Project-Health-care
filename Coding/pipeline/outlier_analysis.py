"""
Outlier Analysis Module
Analyzes data distribution and visualizes outliers using statistical methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List
import logging
import os

logger = logging.getLogger(__name__)


class OutlierAnalyzer:
    """Analyze and visualize outliers in datasets"""

    def __init__(self, output_dir: str = 'outlier_analysis'):
        self.output_dir = output_dir
        self._create_output_dir()
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)

    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def get_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate descriptive statistics for all numeric columns

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats_data = []
        for col in numeric_cols:
            data = df[col].dropna()

            stats_data.append({
                'Column': col,
                'Mean': data.mean(),
                'Std Dev': data.std(),
                'Min': data.min(),
                'Q1 (25%)': data.quantile(0.25),
                'Median (50%)': data.quantile(0.50),
                'Q3 (75%)': data.quantile(0.75),
                'Max': data.max(),
                'IQR': data.quantile(0.75) - data.quantile(0.25),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data)
            })

        return pd.DataFrame(stats_data)

    def detect_outliers_iqr(self, df: pd.DataFrame, threshold: float = 1.5) -> Dict[str, np.ndarray]:
        """
        Detect outliers using Interquartile Range (IQR) method

        Args:
            df: Input DataFrame
            threshold: IQR multiplier (default 1.5 is standard)

        Returns:
            Dictionary with outlier masks for each column
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = {
                'mask': outlier_mask,
                'count': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

        return outliers

    def detect_outliers_zscore(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, np.ndarray]:
        """
        Detect outliers using Z-score method

        Args:
            df: Input DataFrame
            threshold: Z-score threshold (default 3.0 = 99.7% of data)

        Returns:
            Dictionary with outlier masks for each column
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}

        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_mask = np.abs(stats.zscore(df[col])) > threshold

            outliers[col] = {
                'mask': outlier_mask,
                'count': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(df[col].dropna())) * 100,
                'threshold': threshold
            }

        return outliers

    def plot_distribution_with_gaussian(self, df: pd.DataFrame, columns: List[str] = None,
                                       save: bool = True) -> None:
        """
        Plot distribution with Gaussian curve and quartile lines

        Args:
            df: Input DataFrame
            columns: Specific columns to plot (all numeric if None)
            save: Whether to save the plot
        """
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        if columns is not None and len(columns) > 0:
            numeric_cols = [col for col in columns if col in numeric_cols]

        # Create subplots
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            data = df[col].dropna()

            # Histogram
            ax.hist(data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')

            # Gaussian curve
            mu, sigma = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Gaussian Curve')

            # Quartile lines
            Q1 = data.quantile(0.25)
            Q2 = data.quantile(0.50)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            ax.axvline(Q1, color='green', linestyle='--', linewidth=2, label=f'Q1 ({Q1:.2f})')
            ax.axvline(Q2, color='orange', linestyle='--', linewidth=2, label=f'Median ({Q2:.2f})')
            ax.axvline(Q3, color='purple', linestyle='--', linewidth=2, label=f'Q3 ({Q3:.2f})')

            # Outlier bounds (IQR method)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            ax.axvline(lower_bound, color='red', linestyle=':', linewidth=2, label=f'IQR Lower ({lower_bound:.2f})')
            ax.axvline(upper_bound, color='red', linestyle=':', linewidth=2, label=f'IQR Upper ({upper_bound:.2f})')

            ax.set_title(f'{col}\n(Mean={mu:.2f}, Std={sigma:.2f}, IQR={IQR:.2f})', fontsize=11, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'distribution_with_gaussian.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {filepath}")

        plt.show()

    def plot_outliers_comparison(self, df: pd.DataFrame, columns: List[str] = None,
                                 save: bool = True) -> None:
        """
        Compare outliers detected by IQR vs Z-score methods

        Args:
            df: Input DataFrame
            columns: Specific columns to plot
            save: Whether to save the plot
        """
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        if columns is not None and len(columns) > 0:
            numeric_cols = [col for col in columns if col in numeric_cols]

        outliers_iqr = self.detect_outliers_iqr(df)
        outliers_zscore = self.detect_outliers_zscore(df, threshold=3.0)

        # Create comparison DataFrame
        comparison_data = []
        for col in numeric_cols:
            comparison_data.append({
                'Column': col,
                'IQR Count': outliers_iqr[col]['count'],
                'IQR %': outliers_iqr[col]['percentage'],
                'Z-score Count': outliers_zscore[col]['count'],
                'Z-score %': outliers_zscore[col]['percentage']
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Count comparison
        x = np.arange(len(numeric_cols))
        width = 0.35
        axes[0].bar(x - width/2, comparison_df['IQR Count'], width, label='IQR', color='skyblue')
        axes[0].bar(x + width/2, comparison_df['Z-score Count'], width, label='Z-score', color='lightcoral')
        axes[0].set_xlabel('Columns')
        axes[0].set_ylabel('Number of Outliers')
        axes[0].set_title('Outlier Count Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(numeric_cols, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Percentage comparison
        axes[1].bar(x - width/2, comparison_df['IQR %'], width, label='IQR', color='skyblue')
        axes[1].bar(x + width/2, comparison_df['Z-score %'], width, label='Z-score', color='lightcoral')
        axes[1].set_xlabel('Columns')
        axes[1].set_ylabel('Percentage of Outliers (%)')
        axes[1].set_title('Outlier Percentage Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(numeric_cols, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'outliers_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {filepath}")

        plt.show()

        # Print comparison table
        print("\n" + "="*80)
        print("OUTLIER DETECTION COMPARISON (IQR vs Z-score)")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80 + "\n")

    def plot_boxplots(self, df: pd.DataFrame, columns: List[str] = None,
                     save: bool = True) -> None:
        """
        Create boxplots to visualize quartiles and outliers

        Args:
            df: Input DataFrame
            columns: Specific columns to plot
            save: Whether to save the plot
        """
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        if columns is not None and len(columns) > 0:
            numeric_cols = [col for col in columns if col in numeric_cols]

        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4*len(numeric_cols), 6))
        axes = axes if len(numeric_cols) > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            axes[idx].boxplot(df[col].dropna(), vert=True)
            axes[idx].set_title(f'{col}', fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'boxplots.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Boxplot saved to {filepath}")

        plt.show()

    def generate_report(self, df: pd.DataFrame, columns: List[str] = None) -> None:
        """
        Generate comprehensive outlier analysis report

        Args:
            df: Input DataFrame
            columns: Specific columns to analyze
        """
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        if columns is not None and len(columns) > 0:
            numeric_cols = [col for col in columns if col in numeric_cols]

        # Get statistics
        stats_df = self.get_statistics(df[numeric_cols])

        # Save statistics
        stats_filepath = os.path.join(self.output_dir, 'statistics.csv')
        stats_df.to_csv(stats_filepath, index=False)
        logger.info(f"Statistics saved to {stats_filepath}")

        # Print statistics
        print("\n" + "="*100)
        print("DESCRIPTIVE STATISTICS")
        print("="*100)
        print(stats_df.to_string(index=False))
        print("="*100 + "\n")

        # Generate plots
        print("Generating visualization plots...")
        self.plot_distribution_with_gaussian(df, columns=numeric_cols)
        self.plot_boxplots(df, columns=numeric_cols)
        self.plot_outliers_comparison(df, columns=numeric_cols)

        print(f"\nAll analyses saved to: {self.output_dir}")
