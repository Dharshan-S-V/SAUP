# SAUP Experimental Report — UCI Bank Marketing (Deposit Subscription)

**Generated:** 2026-10-12 10:35:58
**Dataset:** UCI Bank Marketing (Deposit Subscription)

## Abstract

This report presents empirical evaluation of the Shift-Aware Adaptive Uncertainty Prediction (SAUP) framework on the **UCI Bank Marketing (Deposit Subscription)** dataset. The study covers standard baselines, temperature-scaling probability calibration, split conformal prediction, distribution-shift detection (KS, PSI, MMD), label-shift estimation (BBSE), and selective decision policies.

## 1. Dataset Overview

- **Source:** UCI Machine Learning Repository (Bank Marketing)
- **Task:** Binary classification (term deposit subscription: yes vs no)
- **Instances:** 45,211 total (stratified 80% dev, 20% held-out test)
- **Features:** 6 numerical, 9 categorical
- **Leakage Prevention:** Call duration (`duration`) is strictly excluded. Call duration is post-outcome information unavailable before a marketing call is placed, and highly correlates with the outcome. Dropping duration ensures realistic prediction-time validity.

## 2. Baseline Model Performance

Evaluated on the held-out test set:

| Model | Accuracy | Balanced Acc | F1 | ROC-AUC | Brier Score | ECE | Train Time (s) |
|-------|----------|--------------|----|---------|-------------|-----|----------------|
| logistic_regression | 0.8931 | 0.5820 | 0.2778 | 0.7709 | 0.0852 | 0.0140 | 3.01 |
| random_forest | 0.8940 | 0.5816 | 0.2773 | 0.8014 | 0.0816 | 0.0142 | 1.37 |
| mlp | 0.8945 | 0.5971 | 0.3166 | 0.7833 | 0.0839 | 0.0138 | 8.35 |
| xgboost | 0.8957 | 0.6204 | 0.3692 | 0.7952 | 0.0821 | 0.0154 | 1.23 |

## 3. Distribution Shift Experiments

### 3.1 Performance and Uncertainty Under Shift

| Scenario | Severity | Model | Accuracy | F1 | Coverage | Set Size | Abstention |
|----------|----------|-------|----------|----|----------|----------|------------|
| covariate | mild | xgboost | 0.9310 | 0.6464 | 0.9337 | 1.006 | 0.082 |
| covariate | moderate | xgboost | 0.9298 | 0.6835 | 0.9329 | 1.007 | 0.087 |
| covariate | severe | xgboost | 0.9318 | 0.7290 | 0.9343 | 1.006 | 0.100 |
| label | mild | xgboost | 0.8801 | 0.6024 | 0.8836 | 1.007 | 0.102 |
| label | moderate | xgboost | 0.8320 | 0.6115 | 0.8371 | 1.011 | 0.133 |
| label | severe | xgboost | 0.7635 | 0.6078 | 0.7708 | 1.015 | 0.174 |
| mixed | mild | xgboost | 0.8816 | 0.6336 | 0.8861 | 1.009 | 0.117 |
| mixed | moderate | xgboost | 0.8492 | 0.7219 | 0.8538 | 1.011 | 0.155 |
| mixed | severe | xgboost | 0.7782 | 0.7150 | 0.7869 | 1.018 | 0.244 |
| severe | mild | xgboost | 0.9229 | 0.8061 | 0.9243 | 1.003 | 0.124 |
| severe | moderate | xgboost | 0.9046 | 0.6926 | 0.9150 | 1.015 | 0.142 |
| severe | severe | xgboost | 0.8904 | 0.6669 | 0.8974 | 1.014 | 0.185 |
| source | none | xgboost | 0.8957 | 0.3692 | 0.8988 | 1.007 | 0.082 |

### 3.2 Shift Detection Statistics

| Scenario | Severity | Detected | Severity Est. | MMD Value | Max PSI |
|----------|----------|----------|---------------|-----------|---------|
| covariate | mild | True | moderate | 0.004441 | 0.0365 |
| covariate | moderate | True | severe | 0.029924 | 0.2295 |
| covariate | severe | True | severe | 0.116291 | 0.7247 |
| label | mild | True | moderate | 0.000730 | 0.0048 |
| label | moderate | True | moderate | 0.005563 | 0.0205 |
| label | severe | True | moderate | 0.005944 | 0.0517 |
| mixed | mild | True | moderate | 0.006810 | 0.0448 |
| mixed | moderate | True | severe | 0.041945 | 0.2924 |
| mixed | severe | True | severe | 0.131202 | 0.7789 |
| severe | mild | True | severe | 1.081474 | 5.1499 |
| severe | moderate | True | severe | 1.068206 | 5.2558 |
| severe | severe | True | severe | 0.781876 | 5.6943 |
| source | none | False | none | 0.000000 | 0.0022 |

## 4. SAUP Adaptive Decision Framework

| Scenario | Severity | Accuracy | Accept Rate | Set Rate | Abstain Rate | Selective Acc | Coverage |
|----------|----------|----------|-------------|----------|--------------|---------------|----------|
| source | none | 0.8957 | 0.918 | 0.082 | 0.000 | 0.9211 | 0.8988 |
| covariate | mild | 0.9290 | 0.880 | 0.120 | 0.000 | 0.9645 | 0.9314 |
| covariate | moderate | 0.9298 | 0.000 | 0.000 | 1.000 | N/A | 0.9329 |
| covariate | severe | 0.9318 | 0.000 | 0.000 | 1.000 | N/A | 0.9343 |
| label | mild | 0.8907 | 0.774 | 0.226 | 0.000 | 0.9537 | 0.8988 |
| label | moderate | 0.8424 | 0.562 | 0.438 | 0.000 | 0.9341 | 0.8588 |
| label | severe | 0.6853 | 0.526 | 0.474 | 0.000 | 0.8217 | 0.7048 |
| mixed | mild | 0.8873 | 0.719 | 0.281 | 0.000 | 0.9547 | 0.8980 |
| mixed | moderate | 0.8492 | 0.000 | 0.000 | 1.000 | N/A | 0.8538 |
| mixed | severe | 0.7782 | 0.000 | 0.000 | 1.000 | N/A | 0.7869 |
| severe | mild | 0.9229 | 0.000 | 0.000 | 1.000 | N/A | 0.9243 |
| severe | moderate | 0.9046 | 0.000 | 0.000 | 1.000 | N/A | 0.9150 |
| severe | severe | 0.8904 | 0.000 | 0.000 | 1.000 | N/A | 0.8974 |

## 5. Ablation Study

| Variant | Scenario | Accuracy | F1 | ECE | Coverage |
|---------|----------|----------|----|-----|----------|
| uncalibrated | covariate | 0.9298 | 0.6835 | 0.0427 | N/A |
| calibrated_only | covariate | 0.9298 | 0.6835 | 0.0495 | N/A |
| conformal_only | covariate | 0.9298 | 0.6835 | 0.0495 | 0.9329 |
| full_saup | covariate | 0.9298 | 0.6835 | 0.0495 | 0.9329 |
| uncalibrated | label | 0.8320 | 0.6115 | 0.1171 | N/A |
| calibrated_only | label | 0.8320 | 0.6115 | 0.1205 | N/A |
| conformal_only | label | 0.8320 | 0.6115 | 0.1205 | 0.8371 |
| full_saup | label | 0.8320 | 0.6115 | 0.1205 | 0.8371 |
| uncalibrated | mixed | 0.8492 | 0.7219 | 0.1208 | N/A |
| calibrated_only | mixed | 0.8492 | 0.7219 | 0.1207 | N/A |
| conformal_only | mixed | 0.8492 | 0.7219 | 0.1207 | 0.8538 |
| full_saup | mixed | 0.8492 | 0.7219 | 0.1207 | 0.8538 |
