# SAUP Experimental Report — UCI Wine Quality (Physicochemical Quality >= 7)

**Generated:** 2026-10-12 10:35:58
**Dataset:** UCI Wine Quality (Physicochemical Quality >= 7)

## Abstract

This report presents empirical evaluation of the Shift-Aware Adaptive Uncertainty Prediction (SAUP) framework on the **UCI Wine Quality (Physicochemical Quality >= 7)** dataset. The study covers standard baselines, temperature-scaling probability calibration, split conformal prediction, distribution-shift detection (KS, PSI, MMD), label-shift estimation (BBSE), and selective decision policies.

## 1. Dataset Overview

- **Source:** UCI Machine Learning Repository (Wine Quality)
- **Task:** Binary classification (Quality ≥ 7: high quality vs < 7: standard/low quality)
- **Instances:** 6,497 total (1,599 red, 4,898 white; stratified 80% dev, 20% held-out test)
- **Features:** 11 numerical physicochemical measurements + 1 categorical feature (wine_type)
- **Design Decision:** Sensory quality scores range 3-9 with mode 5-6. The threshold quality ≥ 7 distinguishes premium wines recognized by experts (~20% minority class) from standard table wines.

## 2. Baseline Model Performance

Evaluated on the held-out test set:

| Model | Accuracy | Balanced Acc | F1 | ROC-AUC | Brier Score | ECE | Train Time (s) |
|-------|----------|--------------|----|---------|-------------|-----|----------------|
| logistic_regression | 0.8215 | 0.6088 | 0.3626 | 0.8039 | 0.1252 | 0.0216 | 2.67 |
| random_forest | 0.8608 | 0.6966 | 0.5464 | 0.8734 | 0.1038 | 0.0286 | 0.75 |
| mlp | 0.8246 | 0.6520 | 0.4519 | 0.8350 | 0.1184 | 0.0308 | 2.40 |
| xgboost | 0.8662 | 0.7427 | 0.6133 | 0.8733 | 0.1029 | 0.0571 | 1.60 |

## 3. Distribution Shift Experiments

### 3.1 Performance and Uncertainty Under Shift

| Scenario | Severity | Model | Accuracy | F1 | Coverage | Set Size | Abstention |
|----------|----------|-------|----------|----|----------|----------|------------|
| covariate | mild | xgboost | 0.9978 | 0.9950 | 1.0000 | 1.004 | 0.036 |
| covariate | moderate | xgboost | 0.9994 | 0.9988 | 0.9994 | 1.004 | 0.027 |
| covariate | severe | xgboost | 0.9994 | 0.9988 | 1.0000 | 1.003 | 0.024 |
| label | mild | xgboost | 0.9983 | 0.9971 | 1.0000 | 1.012 | 0.054 |
| label | moderate | xgboost | 0.9983 | 0.9978 | 1.0000 | 1.013 | 0.067 |
| label | severe | xgboost | 0.9983 | 0.9983 | 1.0000 | 1.019 | 0.080 |
| mixed | mild | xgboost | 0.9994 | 0.9991 | 1.0000 | 1.012 | 0.058 |
| mixed | moderate | xgboost | 0.9989 | 0.9987 | 1.0000 | 1.007 | 0.054 |
| mixed | severe | xgboost | 0.9994 | 0.9995 | 1.0000 | 1.011 | 0.058 |
| severe | mild | xgboost | 0.9703 | 0.9227 | 0.9868 | 1.059 | 0.101 |
| severe | moderate | xgboost | 0.9587 | 0.8288 | 0.9818 | 1.050 | 0.081 |
| severe | severe | xgboost | 0.9571 | 0.7886 | 0.9703 | 1.043 | 0.082 |
| source | none | xgboost | 0.8662 | 0.6133 | 0.9008 | 1.082 | 0.135 |

### 3.2 Shift Detection Statistics

| Scenario | Severity | Detected | Severity Est. | MMD Value | Max PSI |
|----------|----------|----------|---------------|-----------|---------|
| covariate | mild | True | moderate | 0.002317 | 0.0434 |
| covariate | moderate | True | moderate | 0.012063 | 0.0971 |
| covariate | severe | True | severe | 0.033184 | 0.2817 |
| label | mild | False | none | 0.000284 | 0.0132 |
| label | moderate | True | moderate | 0.001009 | 0.0378 |
| label | severe | True | moderate | 0.002783 | 0.0971 |
| mixed | mild | True | moderate | 0.004389 | 0.0655 |
| mixed | moderate | True | severe | 0.012355 | 0.2035 |
| mixed | severe | True | severe | 0.042022 | 0.5580 |
| severe | mild | True | severe | 0.019283 | 0.2438 |
| severe | moderate | True | severe | 0.119892 | 1.4994 |
| severe | severe | True | severe | 0.176642 | 2.7893 |
| source | none | False | none | 0.000000 | 0.0151 |

## 4. SAUP Adaptive Decision Framework

| Scenario | Severity | Accuracy | Accept Rate | Set Rate | Abstain Rate | Selective Acc | Coverage |
|----------|----------|----------|-------------|----------|--------------|---------------|----------|
| source | none | 0.8662 | 0.865 | 0.135 | 0.000 | 0.9048 | 0.9008 |
| covariate | mild | 0.9972 | 0.929 | 0.071 | 0.000 | 1.0000 | 1.0000 |
| covariate | moderate | 0.9956 | 0.920 | 0.080 | 0.000 | 1.0000 | 1.0000 |
| covariate | severe | 0.9994 | 0.000 | 0.000 | 1.000 | N/A | 1.0000 |
| label | mild | 0.9983 | 0.946 | 0.054 | 0.000 | 1.0000 | 1.0000 |
| label | moderate | 0.8850 | 0.755 | 0.245 | 0.000 | 0.9847 | 0.9626 |
| label | severe | 0.7475 | 0.816 | 0.184 | 0.000 | 0.8120 | 0.7981 |
| mixed | mild | 0.9626 | 0.811 | 0.189 | 0.000 | 1.0000 | 0.9972 |
| mixed | moderate | 0.9989 | 0.000 | 0.000 | 1.000 | N/A | 1.0000 |
| mixed | severe | 0.9994 | 0.000 | 0.000 | 1.000 | N/A | 1.0000 |
| severe | mild | 0.9703 | 0.000 | 0.000 | 1.000 | N/A | 0.9868 |
| severe | moderate | 0.9587 | 0.000 | 0.000 | 1.000 | N/A | 0.9818 |
| severe | severe | 0.9571 | 0.000 | 0.000 | 1.000 | N/A | 0.9703 |

## 5. Ablation Study

| Variant | Scenario | Accuracy | F1 | ECE | Coverage |
|---------|----------|----------|----|-----|----------|
| uncalibrated | covariate | 0.9994 | 0.9988 | 0.0486 | N/A |
| calibrated_only | covariate | 0.9994 | 0.9988 | 0.0912 | N/A |
| conformal_only | covariate | 0.9994 | 0.9988 | 0.0912 | 0.9994 |
| full_saup | covariate | 0.9994 | 0.9988 | 0.0912 | 0.9994 |
| uncalibrated | label | 0.9983 | 0.9978 | 0.0671 | N/A |
| calibrated_only | label | 0.9983 | 0.9978 | 0.1153 | N/A |
| conformal_only | label | 0.9983 | 0.9978 | 0.1153 | 1.0000 |
| full_saup | label | 0.9983 | 0.9978 | 0.1153 | 1.0000 |
| uncalibrated | mixed | 0.9989 | 0.9987 | 0.0656 | N/A |
| calibrated_only | mixed | 0.9989 | 0.9987 | 0.1150 | N/A |
| conformal_only | mixed | 0.9989 | 0.9987 | 0.1150 | 1.0000 |
| full_saup | mixed | 0.9989 | 0.9987 | 0.1150 | 1.0000 |
