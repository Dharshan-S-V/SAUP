# SAUP Experimental Report

**Generated:** 2026-10-12 09:16:27

## Abstract

This report presents the experimental results of the Shift-Aware Adaptive Uncertainty Prediction (SAUP) framework, evaluated on the UCI Adult/Census Income dataset. SAUP integrates distribution-shift detection, probability calibration, conformal prediction, and selective abstention to improve prediction reliability under controlled distribution shifts.

## 1. Dataset

- **Source:** UCI Adult / Census Income dataset
- **Task:** Binary classification (income ≤50K vs >50K)
- **Features:** 6 numerical, 8 categorical

## 2. Baseline Model Results

| Model | Accuracy | F1 | ROC-AUC | Brier | ECE |
|-------|----------|----|---------| ------|-----|
| logistic_regression | 0.8523 | 0.6558 | 0.9052 | 0.1019 | 0.0073 |
| random_forest | 0.8622 | 0.6662 | 0.9141 | 0.0976 | 0.0294 |
| mlp | 0.8565 | 0.6773 | 0.9093 | 0.0997 | 0.0133 |
| xgboost | 0.8719 | 0.7061 | 0.9257 | 0.0892 | 0.0105 |

## 3. Distribution Shift Experiments

### 3.1 Classification Performance Under Shift

| Scenario | Severity | Accuracy | F1 | ROC-AUC | Coverage | Set Size |
|----------|----------|----------|----|---------| ---------|----------|
| covariate | mild | 0.8910 | 0.7813 | 0.9550 | 0.9248 | 1.072 |
| covariate | moderate | 0.8908 | 0.7992 | 0.9541 | 0.9258 | 1.077 |
| covariate | severe | 0.8845 | 0.8138 | 0.9542 | 0.9210 | 1.083 |
| label | mild | 0.8843 | 0.8083 | 0.9603 | 0.9191 | 1.077 |
| label | moderate | 0.8620 | 0.8167 | 0.9599 | 0.9008 | 1.086 |
| label | severe | 0.8323 | 0.8240 | 0.9597 | 0.8795 | 1.101 |
| mixed | mild | 0.8765 | 0.8151 | 0.9570 | 0.9162 | 1.090 |
| mixed | moderate | 0.8518 | 0.8328 | 0.9561 | 0.8987 | 1.102 |
| mixed | severe | 0.8279 | 0.8518 | 0.9498 | 0.8807 | 1.112 |
| severe | mild | 0.8778 | 0.6329 | 0.9212 | 0.8948 | 1.033 |
| severe | moderate | 0.9011 | 0.6415 | 0.9126 | 0.9110 | 1.023 |
| severe | severe | 0.8907 | 0.5722 | 0.8994 | 0.9009 | 1.021 |
| source | none | 0.8719 | 0.7061 | 0.9257 | 0.9016 | 1.068 |

### 3.2 Shift Detection Results

| Scenario | Severity | Detected | Severity Est. | MMD | Max PSI |
|----------|----------|----------|---------------|-----|---------|
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

## 4. SAUP Adaptive Decision Results

| Scenario | Severity | Accuracy | Accept Rate | Abstain Rate | Sel. Accuracy | Coverage |
|----------|----------|----------|-------------|--------------|---------------|----------|
| source | none | 0.8719 | 0.819 | 0.000 | 0.9302 | 0.9016 |
| covariate | mild | 0.8891 | 0.804 | 0.000 | 0.9566 | 0.9267 |
| covariate | moderate | 0.8848 | 0.781 | 0.000 | 0.9556 | 0.9259 |
| covariate | severe | 0.8619 | 0.753 | 0.000 | 0.9504 | 0.9165 |
| label | mild | 0.8879 | 0.797 | 0.000 | 0.9583 | 0.9254 |
| label | moderate | 0.8824 | 0.797 | 0.000 | 0.9497 | 0.9182 |
| label | severe | 0.8853 | 0.834 | 0.000 | 0.9384 | 0.9117 |
| mixed | mild | 0.8752 | 0.784 | 0.000 | 0.9506 | 0.9137 |
| mixed | moderate | 0.8633 | 0.812 | 0.000 | 0.9219 | 0.8948 |
| mixed | severe | 0.8279 | 0.000 | 1.000 | nan | 0.8807 |
| severe | mild | 0.8778 | 0.000 | 1.000 | nan | 0.8948 |
| severe | moderate | 0.9011 | 0.000 | 1.000 | nan | 0.9110 |
| severe | severe | 0.8907 | 0.000 | 1.000 | nan | 0.9009 |

## 5. Ablation Study

| Variant | Scenario | Accuracy | F1 | ECE | Coverage |
|---------|----------|----------|----| ----|----------|
| uncalibrated | covariate | 0.8908 | 0.7992 | 0.0310 | nan |
| calibrated_only | covariate | 0.8908 | 0.7992 | 0.0318 | nan |
| conformal_only | covariate | 0.8908 | 0.7992 | 0.0318 | 0.9258 |
| full_saup | covariate | 0.8908 | 0.7992 | 0.0318 | 0.9258 |
| uncalibrated | label | 0.8620 | 0.8167 | 0.0773 | nan |
| calibrated_only | label | 0.8620 | 0.8167 | 0.0773 | nan |
| conformal_only | label | 0.8620 | 0.8167 | 0.0773 | 0.9008 |
| full_saup | label | 0.8620 | 0.8167 | 0.0773 | 0.9008 |
| uncalibrated | mixed | 0.8518 | 0.8328 | 0.0848 | nan |
| calibrated_only | mixed | 0.8518 | 0.8328 | 0.0849 | nan |
| conformal_only | mixed | 0.8518 | 0.8328 | 0.0849 | 0.8987 |
| full_saup | mixed | 0.8518 | 0.8328 | 0.0849 | 0.8987 |

## 6. Limitations

- All distribution shifts are synthetically generated from the same dataset.
- The severe/OOD scenario uses numerical perturbation and should not be interpreted as evidence of performance on a genuine external population.
- BBSE assumes label shift (P(X|Y) preserved), which may not hold under covariate or mixed shifts.
- Importance-weighted conformal prediction uses estimated density ratios; exact finite-sample guarantees require known ratios.
- Temperature scaling may not improve calibration for all models or shift types.
- The UCI Adult dataset has known representation biases.

## 7. Ethical Considerations

- The UCI Adult dataset contains demographic attributes (race, sex). This framework is a research prototype.
- It must NOT be used for real-world decisions about individuals.
- The predictions reflect statistical patterns in historical census data, not causal relationships.
