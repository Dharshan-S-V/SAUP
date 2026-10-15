# SAUP Experimental Report — UCI Wine Quality (Physicochemical Quality >= 7)

**Generated:** 2026-10-14 00:20:33
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
| logistic_regression | 0.8215 | 0.6088 | 0.3626 | 0.8039 | 0.1252 | 0.0216 | 1.49 |
| random_forest | 0.8608 | 0.6966 | 0.5464 | 0.8734 | 0.1038 | 0.0286 | 0.63 |
| mlp | 0.8246 | 0.6520 | 0.4519 | 0.8350 | 0.1184 | 0.0308 | 2.03 |
| xgboost | 0.8662 | 0.7427 | 0.6133 | 0.8733 | 0.1029 | 0.0571 | 0.29 |

## 3. Distribution Shift Experiments

### 3.1 Performance and Uncertainty Under Shift

| Scenario | Severity | Model | Accuracy | F1 | Coverage | Set Size | Abstention |
|----------|----------|-------|----------|----|----------|----------|------------|
| covariate | mild | xgboost | 0.8662 | 0.6298 | 0.8954 | 1.086 | 0.131 |
| covariate | moderate | xgboost | 0.8446 | 0.6354 | 0.8800 | 1.094 | 0.160 |
| covariate | severe | xgboost | 0.8215 | 0.6329 | 0.8569 | 1.100 | 0.157 |
| label | mild | xgboost | 0.8092 | 0.5974 | 0.8462 | 1.072 | 0.134 |
| label | moderate | xgboost | 0.7769 | 0.6329 | 0.8108 | 1.082 | 0.148 |
| label | severe | xgboost | 0.7400 | 0.6718 | 0.7785 | 1.109 | 0.175 |
| mixed | mild | xgboost | 0.7969 | 0.6048 | 0.8400 | 1.077 | 0.155 |
| mixed | moderate | xgboost | 0.7523 | 0.6625 | 0.8000 | 1.112 | 0.203 |
| mixed | severe | xgboost | 0.7585 | 0.7241 | 0.7892 | 1.115 | 0.191 |
| severe | mild | xgboost | 0.8476 | 0.5823 | 0.8845 | 1.076 | 0.145 |
| severe | moderate | xgboost | 0.7968 | 0.4943 | 0.8291 | 1.072 | 0.118 |
| severe | severe | xgboost | 0.7529 | 0.3515 | 0.7783 | 1.046 | 0.095 |
| source | none | xgboost | 0.8662 | 0.6133 | 0.9008 | 1.082 | 0.135 |

### 3.2 Shift Detection Statistics

| Scenario | Severity | Detected | Severity Est. | MMD Value | Max PSI |
|----------|----------|----------|---------------|-----------|---------|
| covariate | mild | True | moderate | 0.003706 | 0.0497 |
| covariate | moderate | True | severe | 0.017389 | 0.1496 |
| covariate | severe | True | severe | 0.041742 | 0.3284 |
| label | mild | False | none | 0.000618 | 0.0408 |
| label | moderate | True | moderate | 0.002249 | 0.0442 |
| label | severe | True | moderate | 0.007597 | 0.0994 |
| mixed | mild | True | moderate | 0.004084 | 0.0765 |
| mixed | moderate | True | severe | 0.022699 | 0.2672 |
| mixed | severe | True | severe | 0.056180 | 0.5544 |
| severe | mild | True | severe | 0.012823 | 0.1659 |
| severe | moderate | True | severe | 0.094144 | 1.1437 |
| severe | severe | True | severe | 0.257526 | 2.5359 |
| source | none | False | none | -0.000015 | 0.0151 |

## 4. SAUP Adaptive Decision Framework

| Scenario | Severity | Accuracy | Accept Rate | Set Rate | Abstain Rate | Selective Acc | Coverage |
|----------|----------|----------|-------------|----------|--------------|---------------|----------|
| source | none | 0.8662 | 0.865 | 0.135 | 0.000 | 0.9048 | 0.9008 |
| covariate | mild | 0.8662 | 0.869 | 0.131 | 0.000 | 0.8956 | 0.8954 |
| covariate | moderate | 0.8446 | 0.000 | 0.000 | 1.000 | N/A | 0.8800 |
| covariate | severe | 0.8215 | 0.000 | 0.000 | 1.000 | N/A | 0.8569 |
| label | mild | 0.8092 | 0.866 | 0.134 | 0.000 | 0.8455 | 0.8462 |
| label | moderate | 0.7769 | 0.852 | 0.148 | 0.000 | 0.8087 | 0.8108 |
| label | severe | 0.7400 | 0.825 | 0.175 | 0.000 | 0.7631 | 0.7785 |
| mixed | mild | 0.7969 | 0.845 | 0.155 | 0.000 | 0.8506 | 0.8400 |
| mixed | moderate | 0.7523 | 0.000 | 0.000 | 1.000 | N/A | 0.8000 |
| mixed | severe | 0.7585 | 0.000 | 0.000 | 1.000 | N/A | 0.7892 |
| severe | mild | 0.8476 | 0.000 | 0.000 | 1.000 | N/A | 0.8845 |
| severe | moderate | 0.7968 | 0.000 | 0.000 | 1.000 | N/A | 0.8291 |
| severe | severe | 0.7529 | 0.000 | 0.000 | 1.000 | N/A | 0.7783 |

## 5. Ablation Study

| Variant | Scenario | Accuracy | F1 | ECE | Coverage |
|---------|----------|----------|----|-----|----------|
| uncalibrated | covariate | 0.8446 | 0.6354 | 0.0746 | N/A |
| calibrated_only | covariate | 0.8446 | 0.6354 | 0.0480 | N/A |
| conformal_only | covariate | 0.8446 | 0.6354 | 0.0480 | 0.8800 |
| full_saup | covariate | 0.8446 | 0.6354 | 0.0480 | 0.8800 |
| uncalibrated | label | 0.7769 | 0.6329 | 0.1481 | N/A |
| calibrated_only | label | 0.7769 | 0.6329 | 0.1181 | N/A |
| conformal_only | label | 0.7769 | 0.6329 | 0.1181 | 0.8108 |
| full_saup | label | 0.7769 | 0.6329 | 0.1181 | 0.8108 |
| uncalibrated | mixed | 0.7523 | 0.6625 | 0.1789 | N/A |
| calibrated_only | mixed | 0.7523 | 0.6625 | 0.1572 | N/A |
| conformal_only | mixed | 0.7523 | 0.6625 | 0.1572 | 0.8000 |
| full_saup | mixed | 0.7523 | 0.6625 | 0.1572 | 0.8000 |
