"""
Final Model Comparison Script

Compares the best models from:
1. Classic/Ensemble pipeline (ejecutador.py)
2. Neural Network tuning (run_nn_tuning.py)

Generates a consolidated report with the overall best model.

Usage:
    python compare_final_models.py
    python compare_final_models.py --classic-results results/final_test_results.csv
    python compare_final_models.py --nn-results results/nn_final_results.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_classic_results(filepath: str) -> dict:
    """Load results from classic/ensemble pipeline"""
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    if df.empty:
        return None

    row = df.iloc[0]
    return {
        'model_name': row.get('Model', 'Classic Model'),
        'MAE': row.get('MAE', np.nan),
        'RMSE': row.get('RMSE', np.nan),
        'R2': row.get('R2-Score', np.nan),
        'source': 'Classic/Ensemble Pipeline'
    }


def load_nn_results(filepath: str) -> dict:
    """Load results from neural network tuning"""
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    if df.empty:
        return None

    row = df.iloc[0]
    return {
        'model_name': row.get('Model', row.get('Architecture', 'Neural Network')),
        'MAE': row.get('MAE', np.nan),
        'RMSE': row.get('RMSE', np.nan),
        'R2': row.get('R2', row.get('R2-Score', np.nan)),
        'source': 'Neural Network Tuning'
    }


def compare_models(classic_results: dict, nn_results: dict) -> dict:
    """
    Compare two models and determine the winner.

    Primary metric: MAE (lower is better)
    Secondary metrics: RMSE, R2
    """
    if classic_results is None and nn_results is None:
        return None

    if classic_results is None:
        return {
            'winner': nn_results,
            'loser': None,
            'comparison': 'Only NN results available'
        }

    if nn_results is None:
        return {
            'winner': classic_results,
            'loser': None,
            'comparison': 'Only Classic results available'
        }

    # Compare by MAE (lower is better)
    classic_mae = classic_results['MAE']
    nn_mae = nn_results['MAE']

    if nn_mae < classic_mae:
        winner = nn_results
        loser = classic_results
        improvement = ((classic_mae - nn_mae) / classic_mae) * 100
    else:
        winner = classic_results
        loser = nn_results
        improvement = ((nn_mae - classic_mae) / nn_mae) * 100 if nn_mae > 0 else 0

    return {
        'winner': winner,
        'loser': loser,
        'improvement_mae_pct': improvement,
        'classic_results': classic_results,
        'nn_results': nn_results
    }


def generate_report(comparison: dict, output_dir: str) -> str:
    """Generate comparison report"""

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FINAL MODEL COMPARISON REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)

    if comparison is None:
        report_lines.append("\nNo results available for comparison.")
        return "\n".join(report_lines)

    winner = comparison['winner']
    loser = comparison.get('loser')

    # Winner section
    report_lines.append("\n" + "=" * 70)
    report_lines.append("OVERALL BEST MODEL")
    report_lines.append("=" * 70)
    report_lines.append(f"\nModel: {winner['model_name']}")
    report_lines.append(f"Source: {winner['source']}")
    report_lines.append("\nTest Set Metrics:")
    report_lines.append(f"  MAE:  {winner['MAE']:.4f}")
    report_lines.append(f"  RMSE: {winner['RMSE']:.4f}")
    report_lines.append(f"  R²:   {winner['R2']:.4f}")

    # Comparison section
    if loser and 'classic_results' in comparison:
        report_lines.append("\n" + "-" * 70)
        report_lines.append("DETAILED COMPARISON")
        report_lines.append("-" * 70)

        classic = comparison['classic_results']
        nn = comparison['nn_results']

        report_lines.append(f"\n{'Metric':<12} {'Classic/Ensemble':<20} {'Neural Network':<20} {'Winner':<15}")
        report_lines.append("-" * 70)

        # MAE comparison
        mae_winner = "NN" if nn['MAE'] < classic['MAE'] else "Classic"
        report_lines.append(f"{'MAE':<12} {classic['MAE']:<20.4f} {nn['MAE']:<20.4f} {mae_winner:<15}")

        # RMSE comparison
        rmse_winner = "NN" if nn['RMSE'] < classic['RMSE'] else "Classic"
        report_lines.append(f"{'RMSE':<12} {classic['RMSE']:<20.4f} {nn['RMSE']:<20.4f} {rmse_winner:<15}")

        # R2 comparison
        r2_winner = "NN" if nn['R2'] > classic['R2'] else "Classic"
        report_lines.append(f"{'R²':<12} {classic['R2']:<20.4f} {nn['R2']:<20.4f} {r2_winner:<15}")

        # Improvement summary
        report_lines.append("\n" + "-" * 70)
        report_lines.append("IMPROVEMENT SUMMARY")
        report_lines.append("-" * 70)

        mae_diff = classic['MAE'] - nn['MAE']
        mae_pct = (mae_diff / classic['MAE']) * 100 if classic['MAE'] > 0 else 0

        rmse_diff = classic['RMSE'] - nn['RMSE']
        rmse_pct = (rmse_diff / classic['RMSE']) * 100 if classic['RMSE'] > 0 else 0

        r2_diff = nn['R2'] - classic['R2']

        report_lines.append(f"\nNeural Network vs Classic/Ensemble:")
        report_lines.append(f"  MAE:  {mae_diff:+.4f} ({mae_pct:+.2f}%) {'(better)' if mae_diff > 0 else '(worse)'}")
        report_lines.append(f"  RMSE: {rmse_diff:+.4f} ({rmse_pct:+.2f}%) {'(better)' if rmse_diff > 0 else '(worse)'}")
        report_lines.append(f"  R²:   {r2_diff:+.4f} {'(better)' if r2_diff > 0 else '(worse)'}")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("RECOMMENDATION")
    report_lines.append("=" * 70)
    report_lines.append(f"\nUse '{winner['model_name']}' from {winner['source']}")
    report_lines.append(f"for production deployment based on TEST set performance.")
    report_lines.append("=" * 70)

    report = "\n".join(report_lines)

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'final_model_comparison.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    # Save as CSV for easy parsing
    csv_path = os.path.join(output_dir, 'final_model_comparison.csv')
    if comparison.get('classic_results') and comparison.get('nn_results'):
        df = pd.DataFrame([
            {
                'Model': comparison['classic_results']['model_name'],
                'Source': 'Classic/Ensemble',
                'MAE': comparison['classic_results']['MAE'],
                'RMSE': comparison['classic_results']['RMSE'],
                'R2': comparison['classic_results']['R2']
            },
            {
                'Model': comparison['nn_results']['model_name'],
                'Source': 'Neural Network',
                'MAE': comparison['nn_results']['MAE'],
                'RMSE': comparison['nn_results']['RMSE'],
                'R2': comparison['nn_results']['R2']
            }
        ])
        df = df.sort_values('MAE')
        df.to_csv(csv_path, index=False)

    # Save as JSON
    json_path = os.path.join(output_dir, 'final_model_comparison.json')
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'winner': winner,
        'classic_results': comparison.get('classic_results'),
        'nn_results': comparison.get('nn_results'),
        'improvement_mae_pct': comparison.get('improvement_mae_pct', 0)
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Compare final models from Classic and Neural Network pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--classic-results',
        type=str,
        default='results/final_test_results.csv',
        help='Path to classic pipeline final test results'
    )
    parser.add_argument(
        '--nn-results',
        type=str,
        default='results/nn_final_results.csv',
        help='Path to neural network final test results'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for comparison report'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    # Load results
    print(f"\nLoading Classic/Ensemble results from: {args.classic_results}")
    classic_results = load_classic_results(args.classic_results)
    if classic_results:
        print(f"  Found: {classic_results['model_name']} (MAE: {classic_results['MAE']:.4f})")
    else:
        print("  No results found")

    print(f"\nLoading Neural Network results from: {args.nn_results}")
    nn_results = load_nn_results(args.nn_results)
    if nn_results:
        print(f"  Found: {nn_results['model_name']} (MAE: {nn_results['MAE']:.4f})")
    else:
        print("  No results found")

    # Compare
    comparison = compare_models(classic_results, nn_results)

    # Generate report
    report = generate_report(comparison, args.output)
    print("\n" + report)

    print(f"\nReports saved to: {args.output}/")
    print("  - final_model_comparison.txt")
    print("  - final_model_comparison.csv")
    print("  - final_model_comparison.json")

    return comparison


if __name__ == "__main__":
    main()
