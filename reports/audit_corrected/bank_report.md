# SAUP Experimental Report — UCI Bank Marketing (Deposit Subscription)

**Generated:** 2026-10-14 00:20:33
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
| logistic_regression | 0.8931 | 0.5820 | 0.2778 | 0.7709 | 0.0852 | 0.0140 | 1.75 |
| random_forest | 0.8940 | 0.5816 | 0.2773 | 0.8014 | 0.0816 | 0.0142 | 1.29 |
| mlp | 0.8945 | 0.5971 | 0.3166 | 0.7833 | 0.0839 | 0.0138 | 8.35 |
| xgboost | 0.8957 | 0.6204 | 0.3692 | 0.7952 | 0.0821 | 0.0154 | 0.98 |

## 3. Distribution Shift Experiments

### 3.1 Performance and Uncertainty Under Shift

| Scenario | Severity | Model | Accuracy | F1 | Coverage | Set Size | Abstention |
|----------|----------|-------|----------|----|----------|----------|------------|
| covariate | mild | xgboost | 0.8921 | 0.3388 | 0.8945 | 1.006 | 0.080 |
| covariate | moderate | xgboost | 0.8921 | 0.3854 | 0.8936 | 1.005 | 0.102 |
| covariate | severe | xgboost | 0.8881 | 0.4157 | 0.8907 | 1.006 | 0.139 |
| label | mild | xgboost | 0.8334 | 0.3990 | 0.8363 | 1.006 | 0.102 |
| label | moderate | xgboost | 0.7728 | 0.4234 | 0.7766 | 1.008 | 0.124 |
| label | severe | xgboost | 0.6879 | 0.4262 | 0.6939 | 1.012 | 0.156 |
| mixed | mild | xgboost | 0.8184 | 0.4230 | 0.8213 | 1.007 | 0.117 |
| mixed | moderate | xgboost | 0.7622 | 0.4733 | 0.7660 | 1.008 | 0.174 |
| mixed | severe | xgboost | 0.6516 | 0.4652 | 0.6549 | 1.007 | 0.270 |
| severe | mild | xgboost | 0.9453 | 0.3320 | 0.9522 | 1.010 | 0.057 |
| severe | moderate | xgboost | 0.8816 | 0.3352 | 0.8898 | 1.013 | 0.194 |
| severe | severe | xgboost | 0.8202 | 0.4283 | 0.8298 | 1.018 | 0.223 |
| source | none | xgboost | 0.8957 | 0.3692 | 0.8988 | 1.007 | 0.082 |

### 3.2 Shift Detection Statistics

| Scenario | Severity | Detected | Severity Est. | MMD Value | Max PSI |
|----------|----------|----------|---------------|-----------|---------|
| covariate | mild | True | moderate | 0.003853 | 0.0471 |
| covariate | moderate | True | severe | 0.021851 | 0.1967 |
| covariate | severe | True | severe | 0.097760 | 0.7184 |
| label | mild | True | moderate | 0.000352 | 0.0100 |
| label | moderate | True | moderate | 0.001548 | 0.0299 |
| label | severe | True | moderate | 0.003165 | 0.0707 |
| mixed | mild | True | moderate | 0.005642 | 0.0442 |
| mixed | moderate | True | severe | 0.027084 | 0.2138 |
| mixed | severe | True | severe | 0.111382 | 0.7956 |
| severe | mild | True | severe | 0.589461 | 3.7676 |
| severe | moderate | True | severe | 0.847376 | 6.7183 |
| severe | severe | True | severe | 0.534524 | 5.5117 |
| source | none | False | none | -0.000112 | 0.0022 |

## 4. SAUP Adaptive Decision Framework

| Scenario | Severity | Accuracy | Accept Rate | Set Rate | Abstain Rate | Selective Acc | Coverage |
|----------|----------|----------|-------------|----------|--------------|---------------|----------|
| source | none | 0.8957 | 0.918 | 0.082 | 0.000 | 0.9211 | 0.8988 |
| covariate | mild | 0.8921 | 0.920 | 0.080 | 0.000 | 0.9216 | 0.8945 |
| covariate | moderate | 0.8921 | 0.000 | 0.000 | 1.000 | N/A | 0.8936 |
| covariate | severe | 0.8881 | 0.000 | 0.000 | 1.000 | N/A | 0.8907 |
| label | mild | 0.8334 | 0.898 | 0.102 | 0.000 | 0.8599 | 0.8363 |
| label | moderate | 0.7728 | 0.876 | 0.124 | 0.000 | 0.8014 | 0.7766 |
| label | severe | 0.6879 | 0.844 | 0.156 | 0.000 | 0.7195 | 0.6939 |
| mixed | mild | 0.8184 | 0.883 | 0.117 | 0.000 | 0.8442 | 0.8213 |
| mixed | moderate | 0.7622 | 0.000 | 0.000 | 1.000 | N/A | 0.7660 |
| mixed | severe | 0.6516 | 0.000 | 0.000 | 1.000 | N/A | 0.6549 |
| severe | mild | 0.9453 | 0.000 | 0.000 | 1.000 | N/A | 0.9522 |
| severe | moderate | 0.8816 | 0.000 | 0.000 | 1.000 | N/A | 0.8898 |
| severe | severe | 0.8202 | 0.000 | 0.000 | 1.000 | N/A | 0.8298 |

## 5. Ablation Study

| Variant | Scenario | Accuracy | F1 | ECE | Coverage |
|---------|----------|----------|----|-----|----------|
| uncalibrated | covariate | 0.8921 | 0.3854 | 0.0234 | N/A |
| calibrated_only | covariate | 0.8921 | 0.3854 | 0.0214 | N/A |
| conformal_only | covariate | 0.8921 | 0.3854 | 0.0214 | 0.8936 |
| full_saup | covariate | 0.8921 | 0.3854 | 0.0214 | 0.8936 |
| uncalibrated | label | 0.7728 | 0.4234 | 0.1410 | N/A |
| calibrated_only | label | 0.7728 | 0.4234 | 0.1306 | N/A |
| conformal_only | label | 0.7728 | 0.4234 | 0.1306 | 0.7766 |
| full_saup | label | 0.7728 | 0.4234 | 0.1306 | 0.7766 |
| uncalibrated | mixed | 0.7622 | 0.4733 | 0.1413 | N/A |
| calibrated_only | mixed | 0.7622 | 0.4733 | 0.1313 | N/A |
| conformal_only | mixed | 0.7622 | 0.4733 | 0.1313 | 0.7660 |
| full_saup | mixed | 0.7622 | 0.4733 | 0.1313 | 0.7660 |
