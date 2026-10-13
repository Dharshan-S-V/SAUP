# SAUP Experimental Report — UCI Adult / Census Income

**Generated:** 2026-10-12 10:35:58
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
| logistic_regression | 0.8523 | 0.7636 | 0.6558 | 0.9052 | 0.1019 | 0.0073 | 3.57 |
| random_forest | 0.8622 | 0.7654 | 0.6662 | 0.9141 | 0.0976 | 0.0294 | 1.60 |
| mlp | 0.8565 | 0.7809 | 0.6773 | 0.9093 | 0.0997 | 0.0133 | 16.44 |
| xgboost | 0.8719 | 0.7957 | 0.7061 | 0.9257 | 0.0892 | 0.0105 | 3.80 |

## 3. Distribution Shift Experiments

### 3.1 Performance and Uncertainty Under Shift

| Scenario | Severity | Model | Accuracy | F1 | Coverage | Set Size | Abstention |
|----------|----------|-------|----------|----|----------|----------|------------|
| covariate | mild | xgboost | 0.8910 | 0.7813 | 0.9248 | 1.072 | 0.192 |
| covariate | moderate | xgboost | 0.8908 | 0.7992 | 0.9258 | 1.077 | 0.203 |
| covariate | severe | xgboost | 0.8845 | 0.8138 | 0.9210 | 1.083 | 0.229 |
| label | mild | xgboost | 0.8843 | 0.8083 | 0.9191 | 1.077 | 0.197 |
| label | moderate | xgboost | 0.8620 | 0.8167 | 0.9008 | 1.086 | 0.218 |
| label | severe | xgboost | 0.8323 | 0.8240 | 0.8795 | 1.101 | 0.249 |
| mixed | mild | xgboost | 0.8765 | 0.8151 | 0.9162 | 1.090 | 0.215 |
| mixed | moderate | xgboost | 0.8518 | 0.8328 | 0.8987 | 1.102 | 0.242 |
| mixed | severe | xgboost | 0.8279 | 0.8518 | 0.8807 | 1.112 | 0.278 |
| severe | mild | xgboost | 0.8778 | 0.6329 | 0.8948 | 1.033 | 0.088 |
| severe | moderate | xgboost | 0.9011 | 0.6415 | 0.9110 | 1.023 | 0.069 |
| severe | severe | xgboost | 0.8907 | 0.5722 | 0.9009 | 1.021 | 0.058 |
| source | none | xgboost | 0.8719 | 0.7061 | 0.9016 | 1.068 | 0.181 |

### 3.2 Shift Detection Statistics

| Scenario | Severity | Detected | Severity Est. | MMD Value | Max PSI |
|----------|----------|----------|---------------|-----------|---------|
| covariate | mild | True | mild | 0.000000 | 0.0260 |
| covariate | moderate | True | moderate | 0.000000 | 0.1105 |
| covariate | severe | True | moderate | 0.000316 | 0.2746 |
| label | mild | True | mild | 0.000000 | 0.0099 |
| label | moderate | True | mild | 0.000000 | 0.0368 |
| label | severe | True | moderate | 0.000000 | 0.1031 |
| mixed | mild | True | mild | 0.000000 | 0.0541 |
| mixed | moderate | True | moderate | 0.000578 | 0.1988 |
| mixed | severe | True | severe | 0.000900 | 0.5332 |
| severe | mild | True | severe | 0.000000 | 0.7176 |
| severe | moderate | True | severe | 0.004217 | 1.9596 |
| severe | severe | True | severe | 0.003763 | 3.1024 |
| source | none | False | none | 0.000000 | 0.0048 |

## 4. SAUP Adaptive Decision Framework

| Scenario | Severity | Accuracy | Accept Rate | Set Rate | Abstain Rate | Selective Acc | Coverage |
|----------|----------|----------|-------------|----------|--------------|---------------|----------|
| source | none | 0.8719 | 0.819 | 0.181 | 0.000 | 0.9302 | 0.9016 |
| covariate | mild | 0.8891 | 0.804 | 0.196 | 0.000 | 0.9566 | 0.9267 |
| covariate | moderate | 0.8848 | 0.781 | 0.219 | 0.000 | 0.9556 | 0.9259 |
| covariate | severe | 0.8619 | 0.753 | 0.247 | 0.000 | 0.9504 | 0.9165 |
| label | mild | 0.8879 | 0.797 | 0.203 | 0.000 | 0.9583 | 0.9254 |
| label | moderate | 0.8824 | 0.797 | 0.203 | 0.000 | 0.9497 | 0.9182 |
| label | severe | 0.8853 | 0.834 | 0.166 | 0.000 | 0.9384 | 0.9117 |
| mixed | mild | 0.8752 | 0.784 | 0.216 | 0.000 | 0.9506 | 0.9137 |
| mixed | moderate | 0.8633 | 0.812 | 0.188 | 0.000 | 0.9219 | 0.8948 |
| mixed | severe | 0.8279 | 0.000 | 0.000 | 1.000 | N/A | 0.8807 |
| severe | mild | 0.8778 | 0.000 | 0.000 | 1.000 | N/A | 0.8948 |
| severe | moderate | 0.9011 | 0.000 | 0.000 | 1.000 | N/A | 0.9110 |
| severe | severe | 0.8907 | 0.000 | 0.000 | 1.000 | N/A | 0.9009 |

## 5. Ablation Study

| Variant | Scenario | Accuracy | F1 | ECE | Coverage |
|---------|----------|----------|----|-----|----------|
| uncalibrated | covariate | 0.8908 | 0.7992 | 0.0310 | N/A |
| calibrated_only | covariate | 0.8908 | 0.7992 | 0.0318 | N/A |
| conformal_only | covariate | 0.8908 | 0.7992 | 0.0318 | 0.9258 |
| full_saup | covariate | 0.8908 | 0.7992 | 0.0318 | 0.9258 |
| uncalibrated | label | 0.8620 | 0.8167 | 0.0773 | N/A |
| calibrated_only | label | 0.8620 | 0.8167 | 0.0773 | N/A |
| conformal_only | label | 0.8620 | 0.8167 | 0.0773 | 0.9008 |
| full_saup | label | 0.8620 | 0.8167 | 0.0773 | 0.9008 |
| uncalibrated | mixed | 0.8518 | 0.8328 | 0.0848 | N/A |
| calibrated_only | mixed | 0.8518 | 0.8328 | 0.0849 | N/A |
| conformal_only | mixed | 0.8518 | 0.8328 | 0.0849 | 0.8987 |
| full_saup | mixed | 0.8518 | 0.8328 | 0.0849 | 0.8987 |
