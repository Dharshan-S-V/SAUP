# SAUP — Shift-Aware Adaptive Uncertainty Prediction

**An Adaptive Uncertainty-Aware Machine Learning Framework for Reliable Prediction Under Distribution Shift**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 100 Passed](https://img.shields.io/badge/tests-100%20passed-brightgreen.svg)](tests/)

---

## 1. Overview & Framework Architecture

Machine learning models deployed in open-world settings face inevitable distribution shifts between training and production environments, leading to overconfident misclassifications and silent catastrophic failures. 

**SAUP (Shift-Aware Adaptive Uncertainty Prediction)** provides an end-to-end framework that unifies:

1. **Standard Supervised Classifiers:** Logistic Regression, Random Forest, Multi-Layer Perceptron (MLP), and XGBoost.
2. **Multi-Detector Shift Sensing:** Benjamini-Hochberg FDR-controlled Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI), and source-standardized Maximum Mean Discrepancy (MMD) with bootstrap thresholding.
3. **Probability Calibration:** Temperature Scaling via validation negative log-likelihood (NLL) optimization.
4. **Conformal Prediction:** Split conformal prediction and importance-weighted conformal inference guaranteeing finite-sample marginal coverage $1 - \alpha$.
5. **Label Shift Adaptation:** Black Box Shift Estimation (BBSE) for target prior reconstruction.
6. **Adaptive 3-Way Selective Decision Policy:**
   - **`ACCEPT`**: High confidence point prediction ($|C(x)| \le 1$ and confidence $\ge \tau$).
   - **`SET_PREDICTION`**: Return conformal prediction set $C(x)$ to hedge under moderate epistemic uncertainty.
   - **`ABSTAIN`**: Refuse automated prediction under severe distribution shift, low effective sample size (ESS), or empty prediction sets.

---

## 2. Multi-Dataset Benchmark

SAUP is evaluated across three diverse UCI empirical domains:

| Dataset | Domain | Instances | Features | Class Imbalance | Design Decision / Safeguard |
|---|---|---|---|---|---|
| **UCI Adult** | Demographics & Census Income | 48,842 (32,561 dev / 16,281 test) | 14 (6 num, 8 cat) | ~24% positive (>50K) | Research prototype; sensitive demographic attributes analyzed with strict isolation. |
| **UCI Bank Marketing** | Telemarketing Deposit Subscription | 45,211 (36,168 dev / 9,043 test) | 15 (6 num, 9 cat) | ~11.7% positive (`yes`) | **Target leakage prevention:** Call duration (`duration`) is strictly dropped. |
| **UCI Wine Quality** | Physicochemical Quality Scoring | 6,497 (5,197 dev / 1,300 test) | 12 (11 num, 1 cat) | ~19.7% positive ($\ge 7$) | Combined Red & White variants; binary threshold $\ge 7$ identifies expert-certified premium wine. |

---

## 3. Comprehensive Codebase Audit & Corrections

Following a systematic statistical and software QA audit, critical issues were identified and resolved:

| Component | Audit Issue Identified | Mathematical / Implementation Fix Applied |
|---|---|---|
| **Weighted Conformal** | Double-loop variable overwrite in `conformal.py`; p-value missing normalized test weight. | Implemented mathematically exact weighted p-value with test point weight $w_{n+1}$ in numerator and denominator: $p(y) = \frac{\sum_{i=1}^n w_i \mathbf{1}\{V_i \ge V_{n+1}^y\} + w_{n+1}}{\sum_{i=1}^n w_i + w_{n+1}}$. |
| **MMD Shift Detector** | Vanishing MMD (values reported as `0.000000`) caused by scale disparity (e.g. `fnlwgt` variance $\sim 10^{10}$ washing out `age` $\sim 100$). | Embedded a source-only `StandardScaler` inside `ShiftDetector.fit()`. MMD now accurately scales with shift severity across all scenarios. |
| **Test Set Isolation** | Shift scenarios previously resampled from `splits["train"]`, resulting in model evaluation on resampled training records. | Strictly isolated evaluation: all shift scenarios are generated exclusively from held-out `test_df`. |
| **Decision Engine Dispatch** | BBSE unconditionally overrode covariate adaptation even when $P(X \mid Y)$ invariance was violated. Mutually exclusive rates collided. | Conditioned BBSE on label-dominant shifts. Enforced mutual exclusivity: `accept_rate + set_rate + abstain_rate == 1.0`. |
| **Metrics Disambiguation** | Conformal empirical coverage and selective prediction coverage were conflated. | Disambiguated `conformal_coverage` ($P(Y \in C(X))$) from `accepted_coverage` ($N_{\text{accept}} / N$). Selective risk returns `NaN` when $N_{\text{accept}} = 0$. |
| **Ablation Validity** | Ablation previously simulated thresholding on confidence rather than calling the engine. | Rewired `run_ablation()` to authentically execute `SAUPDecisionEngine` for `full_saup`. |

> **Audit Provenance Note:** Historical baseline results are permanently preserved in `reports/results/` and `reports/figures/`. All audit-corrected results, figures, and research reports are generated into `reports/audit_corrected/`.

---

## 4. Quick Start & Reproduction

### Prerequisites

```bash
python -m pip install -r requirements.txt
```

### 1. Run Complete Test Suite (100 Tests)

```bash
python -m pytest tests/ -v
```

### 2. Run Audit-Corrected Experiments & Generate Reports

To reproduce all audit-corrected results across Adult, Bank, and Wine Quality:

```bash
python scripts/run_audit_corrections.py
```

Outputs will be generated in:
- Results: `reports/audit_corrected/results/` (`unified_results.csv`, individual baseline, shift, saup, and ablation CSVs)
- Figures: `reports/audit_corrected/figures/` (92 publication-ready plots)
- Reports: `reports/audit_corrected/` (`cross_dataset_comparison.md`, `adult_report.md`, `bank_report.md`, `wine_report.md`, etc.)

### 3. Launch Interactive Streamlit Research Dashboard

```bash
streamlit run app/streamlit_app.py
```

Features:
- Toggle between **Audit-Corrected (Recommended)** and **Historical Baseline** result suites.
- Dataset selector across **UCI Adult**, **UCI Bank Marketing**, and **UCI Wine Quality**.
- Interactive pages for Cross-Dataset Benchmarking, Distribution Shift Lab, Conformal Uncertainty, and SAUP Decision Explorer.

---

## 5. Repository Structure

```
SAUP/
├── app/
│   └── streamlit_app.py         # Multi-dataset interactive research dashboard
├── configs/
│   ├── default.yaml             # Default configuration (UCI Adult)
│   ├── bank.yaml                # UCI Bank Marketing configuration
│   ├── wine.yaml                # UCI Wine Quality configuration
│   └── experiments.yaml         # Shift scenario configurations
├── data/
│   ├── raw/                     # Raw downloaded datasets (Adult, Bank, Wine)
│   └── processed/               # Data split metadata & preprocessor specifications
├── models/                      # Serialized preprocessors and trained model artifacts
├── reports/
│   ├── audit_corrected/         # AUDIT-CORRECTED RESULTS & REPORTS
│   │   ├── figures/             # 92 corrected PNG figures
│   │   ├── results/             # 17 corrected CSV result tables & unified_results.csv
│   │   ├── adult_report.md
│   │   ├── bank_report.md
│   │   ├── wine_report.md
│   │   ├── cross_dataset_comparison.md
│   │   ├── audit_findings.md    # Detailed audit defect log and evidence
│   │   ├── methodology.md
│   │   ├── limitations.md
│   │   └── references.md
│   ├── figures/                 # Historical baseline figures (immutable)
│   └── results/                 # Historical baseline CSV results (immutable)
├── scripts/
│   ├── run_audit_corrections.py # Master entry point for audit revalidation
│   ├── run_all.py               # Historical pipeline runner
│   ├── download_data.py         # Dataset fetcher and validator
│   ├── run_baselines.py
│   ├── run_shift_experiments.py
│   └── run_ablation.py
├── src/saup/
│   ├── calibration.py           # Temperature scaling
│   ├── conformal.py             # Split and weighted conformal inference
│   ├── decision_engine.py       # SAUP 3-way decision engine
│   ├── experiment_runner.py     # Orchestration pipeline
│   ├── label_shift.py           # Black Box Shift Estimation (BBSE)
│   ├── metrics.py               # Classification, calibration, conformal, & shift metrics
│   ├── models.py                # Model factory (LR, RF, MLP, XGB)
│   ├── preprocessing.py         # Leakage-safe transformers
│   ├── reporting.py             # Markdown and CSV report compilation
│   ├── shift_detection.py       # Standardized KS, PSI, and MMD detectors
│   ├── shift_generation.py      # Covariate, label, mixed, and severe shifts
│   └── visualization.py         # Figure generators
└── tests/                       # 100 passing unit and integration tests
```

---

## 6. Citation & References

For complete bibliographic citations and methodology literature, see [`reports/audit_corrected/references.md`](reports/audit_corrected/references.md).
