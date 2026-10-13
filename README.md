# SAUP — Shift-Aware Adaptive Uncertainty Prediction

**An Adaptive Uncertainty-Aware ML Framework for Reliable Prediction Under Distribution Shift**

## Overview

SAUP integrates six components into a unified framework for reliable prediction under distribution shift:

1. **Conventional ML Classification** — Logistic Regression, Random Forest, MLP, XGBoost
2. **Distribution-Shift Detection** — KS test, PSI, MMD with calibrated thresholds
3. **Probability Calibration** — Temperature scaling
4. **Conformal Prediction** — Split conformal and importance-weighted variants
5. **Label-Shift Estimation** — Black Box Shift Estimation (BBSE)
6. **Adaptive Selective Abstention** — Accept / Return set / Abstain decision policy

Evaluated on the UCI Adult/Census Income dataset under controlled covariate, label, mixed, and severe distribution shifts.

## Quick Start

### 1. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_data.py
```

If automatic download fails, manually download from [UCI](https://archive.ics.uci.edu/static/public/2/adult.zip) and run:

```bash
python scripts/download_data.py --zip-path path/to/adult.zip
```

### 3. Run Full Pipeline

```bash
python scripts/run_all.py
```

### 4. Run Individual Steps

```bash
python scripts/run_baselines.py           # Train and evaluate models
python scripts/run_shift_experiments.py    # Distribution shift experiments
python scripts/run_ablation.py             # Ablation study
```

### 5. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
SAUP/
├── configs/           # YAML configuration files
├── data/raw/          # Downloaded dataset (auto-created)
├── data/processed/    # Preprocessing metadata
├── models/            # Saved model artifacts
├── src/saup/          # Core Python package
│   ├── config.py              # Configuration management
│   ├── data_download.py       # Dataset download & validation
│   ├── data_loader.py         # Data loading & cleaning
│   ├── preprocessing.py       # sklearn pipelines
│   ├── splitting.py           # Stratified data splitting
│   ├── models.py              # Model factory (LR, RF, MLP, XGB)
│   ├── metrics.py             # All evaluation metrics
│   ├── calibration.py         # Temperature scaling
│   ├── conformal.py           # Split & weighted conformal prediction
│   ├── shift_generation.py    # Controlled shift scenarios
│   ├── shift_detection.py     # KS, PSI, MMD detectors
│   ├── label_shift.py         # BBSE estimator
│   ├── decision_engine.py     # SAUP adaptive decision framework
│   ├── experiment_runner.py   # Pipeline orchestration
│   ├── visualization.py       # Publication-quality plots
│   └── reporting.py           # Markdown report generation
├── scripts/           # CLI entry points
├── tests/             # Unit and integration tests
├── app/               # Streamlit dashboard
└── reports/           # Generated results, figures, reports
```

## Configuration

All parameters are configurable via `configs/default.yaml`:
- Random seeds, split ratios, model hyperparameters
- Calibration, conformal, shift detection settings
- Decision policy thresholds

## Key Results

Results are saved to `reports/results/` as CSV files:
- `baseline_results.csv` — Model comparison on source data
- `shift_experiment_results.csv` — All shift scenarios × severities × seeds
- `saup_results.csv` — SAUP decision engine results
- `ablation_results.csv` — Component ablation study

Figures are saved to `reports/figures/` as PNG files.

## Ethical Notice

The UCI Adult dataset contains demographic attributes (race, sex, native country). This framework is a **research prototype** and must NOT be used for real-world decisions about individuals. Predictions reflect statistical patterns in historical census data, not causal relationships.

## Limitations

- All distribution shifts are synthetically generated from the same dataset
- The severe/OOD scenario uses numerical perturbation (explicitly synthetic)
- BBSE assumes label shift (P(X|Y) preserved), which may not hold under all shifts
- Importance-weighted conformal uses estimated density ratios; exact guarantees require known ratios
- See `reports/limitations.md` for the full list

## Requirements

- Python ≥ 3.9
- See `requirements.txt` for all dependencies

## License

Research use only. See references in `reports/references.md`.
