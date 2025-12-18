"""
Reporting Module
Generates comprehensive reports and visualizations
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PipelineReporter:
    """Generate reports from pipeline results"""

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        self._create_output_dir()
        self.reports = {}

    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_metrics_summary(self, metrics_data: Dict[str, Any], filename: str = 'metrics_summary.csv'):
        """
        Save metrics summary to CSV

        Args:
            metrics_data: Dictionary of metrics
            filename: Output filename
        """
        df = pd.DataFrame([metrics_data])
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Metrics summary saved to {filepath}")

    def save_model_comparison(self, comparison_df: pd.DataFrame, filename: str = 'model_comparison.csv'):
        """Save model comparison results"""
        filepath = os.path.join(self.output_dir, filename)
        comparison_df.to_csv(filepath, index=False)
        logger.info(f"Model comparison saved to {filepath}")

    def save_evaluation_results(self, results: Dict[str, Any], filename: str = 'evaluation_results.json'):
        """Save detailed evaluation results"""
        filepath = os.path.join(self.output_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = self._make_serializable(value)
            else:
                serializable_results[key] = self._make_serializable(value)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {filepath}")

    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj

    def generate_text_report(self, pipeline_info: Dict[str, Any],
                            filename: str = 'pipeline_report.txt'):
        """
        Generate comprehensive text report

        Args:
            pipeline_info: Dictionary with pipeline information
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MACHINE LEARNING PIPELINE REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Data Information
            if 'data_info' in pipeline_info:
                f.write("DATA INFORMATION\n")
                f.write("-" * 80 + "\n")
                data_info = pipeline_info['data_info']
                f.write(f"Dataset Shape: {data_info.get('shape', 'N/A')}\n")
                f.write(f"Features: {data_info.get('features', 'N/A')}\n")
                f.write(f"Training Set Size: {data_info.get('train_size', 'N/A')}\n")
                f.write(f"Test Set Size: {data_info.get('test_size', 'N/A')}\n\n")

            # Processing Information
            if 'processing_info' in pipeline_info:
                f.write("DATA PROCESSING\n")
                f.write("-" * 80 + "\n")
                proc_info = pipeline_info['processing_info']
                f.write(f"Missing Values Handling: {proc_info.get('missing_values', 'N/A')}\n")
                f.write(f"Outlier Detection: {proc_info.get('outlier_method', 'N/A')}\n")
                f.write(f"Feature Scaling: {proc_info.get('feature_scaling', 'N/A')}\n")
                f.write(f"Categorical Encoding: {proc_info.get('categorical_encoding', 'N/A')}\n\n")

            # Model Information
            if 'models_info' in pipeline_info:
                f.write("MODELS TRAINED\n")
                f.write("-" * 80 + "\n")
                models = pipeline_info['models_info']
                for model_name in models:
                    f.write(f"  - {model_name}\n")
                f.write("\n")

            # Results
            if 'results' in pipeline_info:
                f.write("MODEL EVALUATION RESULTS\n")
                f.write("-" * 80 + "\n")
                results = pipeline_info['results']
                for metric, value in results.items():
                    f.write(f"{metric}: {value}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Text report saved to {filepath}")

    def generate_summary_report(self, data: Dict[str, Any],
                               filename: str = 'summary_report.txt'):
        """
        Generate summary report with all key information

        Args:
            data: Dictionary with pipeline data and results
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write("PIPELINE EXECUTION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for section, content in data.items():
                f.write(f"{section}\n")
                f.write("-" * 80 + "\n")

                if isinstance(content, dict):
                    for key, value in content.items():
                        if isinstance(value, (list, dict)):
                            f.write(f"{key}:\n{json.dumps(value, indent=2, default=str)}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                elif isinstance(content, pd.DataFrame):
                    f.write(content.to_string())
                else:
                    f.write(str(content))

                f.write("\n\n")

        logger.info(f"Summary report saved to {filepath}")

    def export_results_to_csv(self, results_df: pd.DataFrame, prefix: str = 'results'):
        """
        Export results to CSV files

        Args:
            results_df: DataFrame with results
            prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        results_df.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")

        return filepath

    def create_pipeline_summary(self, pipeline_data: Dict[str, Any]) -> str:
        """
        Create a comprehensive pipeline summary

        Args:
            pipeline_data: Complete pipeline execution data

        Returns:
            Summary string
        """
        summary = f"""
PIPELINE EXECUTION SUMMARY
{'='*80}

Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Data Statistics:
{'-'*80}
- Total Samples: {pipeline_data.get('total_samples', 'N/A')}
- Features: {pipeline_data.get('features', 'N/A')}
- Target Classes: {pipeline_data.get('target_classes', 'N/A')}

Processing Configuration:
{'-'*80}
{self._format_config(pipeline_data.get('processing_config', {}))}

Models Evaluated:
{'-'*80}
{self._format_models(pipeline_data.get('models', []))}

Best Model: {pipeline_data.get('best_model', 'N/A')}

Primary Metrics:
{'-'*80}
  MAE:  {pipeline_data.get('best_mae', 'N/A')}
  RMSE: {pipeline_data.get('best_rmse', 'N/A')}
  R²:   {pipeline_data.get('best_r2', 'N/A')}

{'='*80}
        """
        return summary

    def _format_config(self, config: Dict) -> str:
        """Format configuration dictionary"""
        lines = []
        for key, value in config.items():
            lines.append(f"  - {key}: {value}")
        return "\n".join(lines) if lines else "  No configuration"

    def _format_models(self, models: list) -> str:
        """Format models list"""
        lines = []
        for model in models:
            lines.append(f"  - {model}")
        return "\n".join(lines) if lines else "  No models"
