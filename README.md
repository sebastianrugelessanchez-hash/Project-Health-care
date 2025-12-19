# Project Healthcare – ML Pipeline

## Overview
- Goal: build regression models to forecast next-month medication volume (inventory rotation) for a healthcare company.
- Tech stack: Python, scikit-learn for classical/ensemble models, TensorFlow/Keras for deep nets (optional), with modular pipeline components and logging.
- Primary metrics: MAE (units of medication), RMSE, R²; MAE is emphasized because it communicates unit error directly.

## Repository Structure
- `Coding/pipeline/main.py` – entry point for the full pipeline (ETL → training → evaluation → reporting).
- `Coding/pipeline/ejecutador.py` – orchestrator that wires data IO, processing, model training, evaluation, and reporting.
- `Coding/pipeline/processing.py` – preprocessing/feature pipeline and cleaning utilities.
- `Coding/pipeline/io_modulo.py` – data loading/saving and dataset descriptions.
- `Coding/pipeline/config.py` – global settings (metrics, hyperparams, paths, usable columns, randomness).
- `Coding/pipeline/ml_classic.py` – classical models (Linear Regression, RF, SVR, GB, MLP) with optional tuning.
- `Coding/pipeline/ensemble_models.py` – voting/stacking/bagging/AdaBoost helpers.
- `Coding/pipeline/crossvalidation.py` – CV utilities.
- `Coding/pipeline/evaluation.py` – metric computation and model comparison.
- `Coding/pipeline/reporting.py` – summaries and CSV outputs.
- `Coding/pipeline/nn_tuning.py` – deep NN architecture search with Keras; sklearn MLP fallback.
- `Coding/pipeline/run_nn_tuning.py` – CLI runner for NN tuning.
- `Coding/pipeline/outlier_analysis.py`, `run_outlier_analysis.py` – exploratory outlier analysis and plots.
- `results/` – saved tuning runs and comparisons (e.g., `nn_tuning_results_*.json`, CSVs).
- `Data base/` – expected location for the main Excel dataset (`Reporte de prueba.xlsx` by default).

## Problem & Approach
- Problem: Predict medication inventory volume for the next month (single-column regression target, e.g., `2025-9`) from historical monthly features (`2024-1` … `2025-8`).
- Pipeline logic:
  1) ETL: load Excel/CSV, describe data, basic cleaning (missing targets removed, negatives clipped to 0 by default).
  2) Split: train/test split (default 80/20; configurable).
  3) Processing: select usable columns, scale features, apply configured cleaning steps.
  4) Modeling: train multiple classical models + ensembles; optional hyperparameter tuning.
  5) Deep nets: optional Keras tuning over predefined architectures; select by validation MAE, then evaluate once on test.
  6) Evaluation: compute MAE, RMSE, R² (plus secondary metrics), compare models, save reports/CSV.
  7) Reporting: persist metrics, configs, and (optionally) models.

## Running the Pipeline
### Full pipeline (classical + ensembles)
```bash
python Coding/pipeline/main.py \
  --data "/Users/sebas.12/Desktop/Proyectos/Project Healthcare/Data base/Reporte de prueba.xlsx" \
  --target 2025-9 \
  --output results
```
- Flags: `--no-tune` to skip hyperparameter search; adjust `--data`/`--target`/`--output` as needed.

### Deep neural network tuning
```bash
python Coding/pipeline/run_nn_tuning.py        # auto-detect TF/CPU
python Coding/pipeline/run_nn_tuning.py --force-cpu        # sklearn MLP fallback
python Coding/pipeline/run_nn_tuning.py --force-tensorflow # require TF
```
- Outputs JSON/CSV in `results/` with best config and metrics.

## Configuration
- Centralized in `Coding/pipeline/config.py`:
  - Metrics toggles, CV params, model hyperparameter grids.
  - Data processing: split ratio, missing-value handling, scaling, usable columns.
  - Paths: data, results, logs; adjust to your environment.

## Data Expectations
- Tabular regression dataset with numeric monthly features.
- Default target column: `2025-9`; features include months `2024-1` … `2025-8`.
- Missing target rows are dropped; negative targets clipped to zero (adjust in `config.py` if needed).

## Outputs
- Logs: `pipeline.log`, `nn_tuning.log`.
- Results: CSV comparisons (`nn_architecture_comparison.csv`, model comparison tables), JSON with best configs (`results/nn_tuning_results_*.json`).
- Reports: summaries via `reporting.py` (best model, metrics).

## Notes & Tips
- Keep the test set untouched until final evaluation; selection happens via validation/CV in the training set.
- If TensorFlow is unavailable, the tuner falls back to sklearn MLP.
- For deployment, persist the preprocessing pipeline alongside the chosen model to ensure consistent inference. 'Please add to this the Tensor Flow Metal to increase the power computation'

- 🚀 Hardware Acceleration with TensorFlow Metal (Apple Silicon)

This project supports GPU acceleration on Apple Silicon (M-series) using TensorFlow Metal, enabling faster training and experimentation for deep learning models.

When running on macOS with an M-series chip, TensorFlow automatically leverages the Apple GPU through the Metal backend, significantly improving computational performance during neural network training and architecture tuning. This is particularly beneficial for large batch sizes, deep architectures, and repeated hyperparameter searches.

GPU usage is automatically detected at runtime—no code changes are required once the environment is correctly configured.

Recommended Environment (Apple Silicon)
	•	Python: 3.11
	•	TensorFlow: 2.15.0
	•	TensorFlow Metal: 1.1.0
