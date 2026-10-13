# SAUP Methodology

## Overview
SAUP (Shift-Aware Adaptive Uncertainty Prediction) is an integration framework that combines:
1. Conventional ML classification (Logistic Regression, Random Forest, MLP, XGBoost)
2. Distribution-shift detection (KS test, PSI, MMD)
3. Probability calibration (Temperature Scaling)
4. Conformal prediction (Split conformal, importance-weighted)
5. Label-shift estimation (BBSE)
6. Selective abstention (accept / return set / abstain policy)

## Supported Datasets
- **UCI Adult:** Demographics & Income (32,561 train, 16,281 test)
- **UCI Bank Marketing:** Direct Marketing Campaign (45,211 rows; duration dropped for leakage prevention)
- **UCI Wine Quality:** Physicochemical properties (6,497 rows; quality ≥ 7)

## Mathematical Formulation

### Distribution Shift Detection
- **KS Test:** Two-sample Kolmogorov-Smirnov statistic per numerical feature, with Benjamini-Hochberg FDR correction.
- **PSI:** Population Stability Index using source-defined decile bins: PSI = Σ (p_target - p_source) * ln(p_target / p_source).
- **MMD:** Maximum Mean Discrepancy with RBF kernel and median bandwidth heuristic, with baseline and threshold calibrated on source splits.

### Temperature Scaling
Logit scaling with learned scalar T > 0: p_i = softmax(z_i / T). T is fitted by minimizing NLL on a held-out calibration split.

### Conformal Prediction
Nonconformity score s(x, y) = 1 - p(y|x). Given nominal level 1 - α, compute quantile threshold q_k and construct prediction set C(x) = {y : s(x, y) ≤ q_k}.

### Black Box Shift Estimation (BBSE)
Estimates target class priors w = (C^T)^{-1} μ̂ where C is the source confusion matrix on a policy split and μ̂ are target predictions.

### Decision Policy
- ACCEPT: confidence ≥ threshold AND |prediction set| ≤ 1
- SET_PREDICTION: moderate uncertainty; return prediction set
- ABSTAIN: extreme shift, empty conformal set, low effective sample size, or unstable adaptation
