# SAUP Experimental Report — UCI Adult / Census Income

**Generated:** 2026-10-14 00:20:33
**Dataset:** UCI Adult / Census Income

## Abstract

This report presents empirical evaluation of the Shift-Aware Adaptive Uncertainty Prediction (SAUP) framework on the **UCI Adult / Census Income** dataset. The study covers standard baselines, temperature-scaling probability calibration, split conformal prediction, distribution-shift detection (KS, PSI, MMD), label-shift estimation (BBSE), and selective decision policies.

## 1. Dataset Overview

- **Source:** UCI Machine Learning Repository (Adult / Census Income)
- **Task:** Binary classification (income >50K vs ≤50K)
- **Instances:** 32,561 development, 16,281 official test
- **Features:** 6 numerical, 8 categorical
- **Ethical Note:** Census survey data containing demographic attributes; research prototype only.

## 2. Baseline Model Performance

Evaluated on the held-out test set:

| Model | Accuracy | Balanced Acc | F1 | ROC-AUC | Brier Score | ECE | Train Time (s) |
|-------|----------|--------------|----|---------|-------------|-----|----------------|
| logistic_regression | 0.8523 | 0.7636 | 0.6558 | 0.9052 | 0.1019 | 0.0073 | 2.78 |
| random_forest | 0.8622 | 0.7654 | 0.6662 | 0.9141 | 0.0976 | 0.0294 | 1.16 |
| mlp | 0.8565 | 0.7809 | 0.6773 | 0.9093 | 0.0997 | 0.0133 | 9.79 |
| xgboost | 0.8719 | 0.7957 | 0.7061 | 0.9257 | 0.0892 | 0.0105 | 0.73 |

## 3. Distribution Shift Experiments

### 3.1 Performance and Uncertainty Under Shift

| Scenario | Severity | Model | Accuracy | F1 | Coverage | Set Size | Abstention |
|----------|----------|-------|----------|----|----------|----------|------------|
| covariate | mild | xgboost | 0.8652 | 0.7247 | 0.8967 | 1.073 | 0.199 |
| covariate | moderate | xgboost | 0.8488 | 0.7193 | 0.8876 | 1.087 | 0.216 |
| covariate | severe | xgboost | 0.8217 | 0.6966 | 0.8625 | 1.091 | 0.238 |
| label | mild | xgboost | 0.8445 | 0.7310 | 0.8795 | 1.075 | 0.197 |
| label | moderate | xgboost | 0.8179 | 0.7470 | 0.8563 | 1.083 | 0.219 |
| label | severe | xgboost | 0.7834 | 0.7625 | 0.8254 | 1.093 | 0.249 |
| mixed | mild | xgboost | 0.8321 | 0.7367 | 0.8703 | 1.083 | 0.214 |
| mixed | moderate | xgboost | 0.7937 | 0.7605 | 0.8392 | 1.102 | 0.251 |
| mixed | severe | xgboost | 0.7472 | 0.7710 | 0.7975 | 1.116 | 0.285 |
| severe | mild | xgboost | 0.8600 | 0.5545 | 0.8734 | 1.032 | 0.092 |
| severe | moderate | xgboost | 0.8675 | 0.4717 | 0.8795 | 1.023 | 0.067 |
| severe | severe | xgboost | 0.8397 | 0.4008 | 0.8498 | 1.024 | 0.066 |
| source | none | xgboost | 0.8719 | 0.7061 | 0.9016 | 1.068 | 0.181 |

### 3.2 Shift Detection Statistics

| Scenario | Severity | Detected | Severity Est. | MMD Value | Max PSI |
|----------|----------|----------|---------------|-----------|---------|
| covariate | mild | True | moderate | 0.001902 | 0.0281 |
| covariate | moderate | True | severe | 0.013017 | 0.1130 |
| covariate | severe | True | severe | 0.036329 | 0.2851 |
| label | mild | True | moderate | 0.001205 | 0.0155 |
| label | moderate | True | moderate | 0.002563 | 0.0384 |
| label | severe | True | severe | 0.006719 | 0.1092 |
| mixed | mild | True | moderate | 0.005000 | 0.0532 |
| mixed | moderate | True | severe | 0.019789 | 0.1854 |
| mixed | severe | True | severe | 0.056241 | 0.5178 |
| severe | mild | True | severe | 0.027581 | 0.6985 |
| severe | moderate | True | severe | 0.174513 | 2.0742 |
| severe | severe | True | severe | 0.253511 | 3.1205 |
| source | none | False | none | -0.000152 | 0.0048 |

## 4. SAUP Adaptive Decision Framework

| Scenario | Severity | Accuracy | Accept Rate | Set Rate | Abstain Rate | Selective Acc | Coverage |
|----------|----------|----------|-------------|----------|--------------|---------------|----------|
| source | none | 0.8719 | 0.819 | 0.181 | 0.000 | 0.9302 | 0.9016 |
| covariate | mild | 0.8652 | 0.801 | 0.199 | 0.000 | 0.9259 | 0.8967 |
| covariate | moderate | 0.8488 | 0.000 | 0.000 | 1.000 | N/A | 0.8876 |
| covariate | severe | 0.8217 | 0.000 | 0.000 | 1.000 | N/A | 0.8625 |
| label | mild | 0.8445 | 0.803 | 0.197 | 0.000 | 0.9115 | 0.8795 |
| label | moderate | 0.8179 | 0.781 | 0.219 | 0.000 | 0.8884 | 0.8563 |
| label | severe | 0.7834 | 0.000 | 0.000 | 1.000 | N/A | 0.8254 |
| mixed | mild | 0.8321 | 0.786 | 0.214 | 0.000 | 0.9022 | 0.8703 |
| mixed | moderate | 0.7937 | 0.000 | 0.000 | 1.000 | N/A | 0.8392 |
| mixed | severe | 0.7472 | 0.000 | 0.000 | 1.000 | N/A | 0.7975 |
| severe | mild | 0.8600 | 0.000 | 0.000 | 1.000 | N/A | 0.8734 |
| severe | moderate | 0.8675 | 0.000 | 0.000 | 1.000 | N/A | 0.8795 |
| severe | severe | 0.8397 | 0.000 | 0.000 | 1.000 | N/A | 0.8498 |

## 5. Ablation Study

| Variant | Scenario | Accuracy | F1 | ECE | Coverage |
|---------|----------|----------|----|-----|----------|
| uncalibrated | covariate | 0.8488 | 0.7193 | 0.0142 | N/A |
| calibrated_only | covariate | 0.8488 | 0.7193 | 0.0136 | N/A |
| conformal_only | covariate | 0.8488 | 0.7193 | 0.0136 | 0.8876 |
| full_saup | covariate | 0.8488 | 0.7193 | 0.0136 | 0.8876 |
| uncalibrated | label | 0.8179 | 0.7470 | 0.0874 | N/A |
| calibrated_only | label | 0.8179 | 0.7470 | 0.0871 | N/A |
| conformal_only | label | 0.8179 | 0.7470 | 0.0871 | 0.8563 |
| full_saup | label | 0.8179 | 0.7470 | 0.0871 | 0.8563 |
| uncalibrated | mixed | 0.7937 | 0.7605 | 0.1035 | N/A |
| calibrated_only | mixed | 0.7937 | 0.7605 | 0.1032 | N/A |
| conformal_only | mixed | 0.7937 | 0.7605 | 0.1032 | 0.8392 |
| full_saup | mixed | 0.7937 | 0.7605 | 0.1032 | 0.8392 |
