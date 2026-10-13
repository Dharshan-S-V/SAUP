# SAUP Source-Code and Experimental Design Audit Findings

**Project**: Shift-Aware Adaptive Uncertainty Prediction (SAUP)  
**Date**: October 2026  
**Auditor**: Senior Machine Learning Research Engineer & QA Architect  
**Repository Branch**: `audit-fixes`  
**Status**: Initial Inspection & Defect Identification Complete  

---

## Executive Summary

An exhaustive audit of the SAUP codebase (`src/saup/`, `scripts/`, `tests/`, `configs/`, and `reports/`) was performed across all three supported benchmark datasets:
1. **UCI Adult** (Demographic income classification, tabular, moderate imbalance ~76:24)
2. **UCI Bank Marketing** (Term deposit marketing, tabular, severe imbalance ~88:12, `duration` dropped)
3. **UCI Wine Quality** (Physicochemical quality prediction, binary threshold $\ge 7$, ~80:20)

While foundational pipelines, baseline model training, and basic tests executed successfully, critical defects in statistical implementation, experimental design, metric aggregation, and ablation logic were confirmed. These defects explain anomalous artifacts in the reported results (e.g., vanishing MMD values, conflicting abstention rates, and pseudo-ablation results).

Below is the structured audit matrix detailing confirmed defects, mathematical evidence, severity ratings, and concrete remediation plans.

---

## Detailed Audit Findings Matrix

### Defect 1: Weighted Conformal Prediction Mathematical & Implementation Defect
- **Location**: `src/saup/conformal.py` (`WeightedConformalClassifier.predict_sets()`)
- **Severity**: **Critical (High)**
- **Evidence**:
  1. **Dead Code & Overwrite Bug**: In `predict_sets()`, lines 185–203 compute `pred_set`, and then immediately at line 206 `pred_set = []` overwrites it completely with a second loop.
  2. **Violated Weight Inclusion**: The second loop computes `above_weight = self.weights[self.scores > test_score].sum()`. It uses strict inequality `>` (omitting tied scores `scores == test_score`) and fails to include the test-point weight $w_{n+1}$ in the numerator.
  3. **Mathematical Inconsistency**: Under the standard weighted conformal prediction formulation (Tibshirani et al., 2019):
     $$p(y) = \frac{\sum_{i=1}^n w_i \mathbf{1}\{V_i \ge V_{n+1}^y\} + w_{n+1}}{\sum_{j=1}^n w_j + w_{n+1}}$$
     The candidate label $y$ should be included in $C(X_{n+1})$ if $p(y) > \alpha$ (or $\ge \alpha$ by chosen convention). The current implementation does not calculate this valid p-value.
  4. **Boundary Mishandling**: When weights are zero or extreme, numerical instability or improper set inclusion occurs.
- **Planned Fix**:
  - Implement the exact weighted conformal p-value calculation with test point weight in numerator and denominator.
  - Implement clean tie handling ($\ge$ test score) and strict validation ($w \ge 0$, $\sum w > 0$, finite inputs).
  - Add independent unit tests verifying manually computed toy examples with known weights, tied nonconformity scores, and extreme weights.

---

### Defect 2: Unscaled Features Causing Vanishing MMD Shift Detection
- **Location**: `src/saup/shift_detection.py` (`ShiftDetector.fit()`, `detect()`), `src/saup/metrics.py` (`maximum_mean_discrepancy()`)
- **Severity**: **Critical (High)**
- **Evidence**:
  1. **Unscaled Distances**: `ShiftDetector` extracts `self.source_numerical = X_source[self.numerical_cols].values.astype(float)` directly from raw DataFrames without standardization.
  2. In UCI Adult, `fnlwgt` has values $\sim 10^5$ and variance $\sim 10^{10}$, while `age` has values $\sim 30-60$ and variance $\sim 100$.
  3. In `_pairwise_sq_dists()`, squared Euclidean distance is completely dominated by `fnlwgt`.
  4. The median heuristic bandwidth $\sigma = \sqrt{\text{median}(D^2) / 2}$ scales to $\sim 10^5$, setting kernel bandwidth $\gamma = 1 / (2\sigma^2) \sim 10^{-10}$.
  5. When covariate shift tilts `age` or `hours_per_week` (leaving `fnlwgt` unshifted), the squared distance change $\sim 10^2$ is negligible relative to $10^{10}$. Hence $K_{ss} \approx K_{tt} \approx K_{st}$, yielding computed MMD values of exact $0.000000$ across all Adult shift experiments, despite PSI detecting major shifts!
  6. **Threshold Bias**: In `maximum_mean_discrepancy()`, the unbiased estimator $MMD^2$ can legitimately be slightly negative for identical distributions, but clipping with `max(mmd2, 0.0)` biases the bootstrap reference distribution.
- **Planned Fix**:
  - Fit a source-only `StandardScaler` inside `ShiftDetector.fit()` and apply it to both source and target matrices before computing MMD.
  - Retain unclipped values for unbiased MMD estimation with guards for numerical stability and zero distances in median heuristic.
  - Recompute MMD and threshold across all scenarios.

---

### Defect 3: Evaluation Data Contamination via Resampling from Training Split
- **Location**: `src/saup/experiment_runner.py` (lines 537 & 680)
- **Severity**: **Critical (High)**
- **Evidence**:
  1. In `_run_single_experiment()`:
     ```python
     source_df = self.test_df if scenario == "source" else self.splits["train"]
     ```
  2. And in `run_saup_experiments()`:
     ```python
     source_df = self.test_df if scenario == "source" else self.splits["train"]
     ```
  3. For all shift scenarios (`covariate`, `label`, `mixed`, `severe`), the shift generator sampled rows with replacement from `self.splits["train"]`.
  4. Because models were trained on `self.splits["train"]`, the shifted test sets consisted of resampled training examples! This constitutes evaluation data contamination.
- **Planned Fix**:
  - Set `source_df = self.test_df` for ALL test-time scenarios (source and shifted).
  - Shifts must be induced on the held-out test distribution, ensuring the model is strictly evaluated on unseen records.
  - Implement diagnostics reporting duplicate counts, unique fraction, and verify 0% overlap between training split and shifted test sets.

---

### Defect 4: Metric Inconsistencies & Column Overwrite Collision in SAUP Outcomes
- **Location**: `src/saup/decision_engine.py`, `src/saup/metrics.py`, `src/saup/experiment_runner.py`
- **Severity**: **High**
- **Evidence**:
  1. **Discrepancy Explained**: In `shift_experiment_results.csv`, row 1 (Adult source) reported `abstention_rate = 0.180579`. In `saup_results.csv`, row 1 reported `abstention_rate = 0.0`.
     - *Cause*: Shift experiments used simple binary thresholding (`confidence >= 0.70`), yielding $18.06\%$ unaccepted (called `abstention_rate`). SAUP decision engine used a 3-way decision policy: Accept ($81.94\%$), Set Prediction ($18.06\%$), Abstain ($0.0\%$).
     - *Bug*: In `experiment_runner.py`:
       ```python
       **sel_m,  # contained "abstention_rate": 0.180579
       "abstention_rate": summary["abstain_rate"],  # overwrote with 0.0
       ```
  2. **Ambiguous Column Name**: The column `coverage` was used for selective point prediction coverage ($n_{\text{accept}} / n$), while conformal coverage was reported under `empirical_coverage`. This creates dangerous ambiguity.
  3. **Zero Acceptance Undefined Metrics**: When $n_{\text{accepted}} = 0$, selective accuracy and selective risk were recorded as missing or zero without explicit assertion.
- **Planned Fix**:
  - Standardize mutually exclusive decision fractions:
    `accepted_fraction + prediction_set_fraction + abstention_rate == 1.0`.
  - Disambiguate columns:
    - `accepted_coverage` (or `accepted_fraction`): fraction of test points with accepted point prediction.
    - `conformal_empirical_coverage`: fraction of test points where true $y \in C(x)$.
    - `selective_risk`: error rate among accepted predictions (explicit NaN when $n_{\text{accept}} = 0$).
  - Preserve clear distinctions between baseline confidence thresholding and SAUP adaptive decision policy.

---

### Defect 5: Indiscriminate Invocation of BBSE Under Covariate Shift
- **Location**: `src/saup/decision_engine.py` (lines 156–168)
- **Severity**: **Medium**
- **Evidence**:
  1. When any shift was detected, `predict_batch()` unconditionally attempted BBSE correction as long as BBSE was instantiated.
  2. In `saup_results.csv`, covariate shift experiments (mild, moderate, severe) all reported `adaptation_method: bbse_corrected`.
  3. Covariate shift modifies $P(X)$ while keeping $P(Y|X)$ invariant, meaning $P(X|Y)$ changes. Applying BBSE assumes $P(X|Y)$ is invariant, which is theoretically contradicted by covariate shift.
- **Planned Fix**:
  - Condition adaptation method selection on the detected shift characteristics:
    - When covariate shift is dominant: utilize density ratio estimation / importance weighting.
    - When label shift is dominant (and feature shifts are minor): invoke BBSE.
    - When mixed/severe shift occurs: apply cautious temperature scaling and wider conformal prediction without unvalidated prior distortion.

---

### Defect 6: Incomplete and Simulated Ablation Study
- **Location**: `src/saup/experiment_runner.py` (`run_ablation()`, lines 825–831)
- **Severity**: **High**
- **Evidence**:
  1. In `run_ablation()`, the `full_saup` variant did NOT call `SAUPDecisionEngine`.
  2. Instead, it ran simple confidence thresholding: `accepted = np.max(cal_probas, axis=1) >= thresh`.
  3. Thus, the ablation study never actually ablated the real SAUP engine against its subcomponents.
- **Planned Fix**:
  - Modify `run_ablation()` so that:
    - `uncalibrated`: raw model probabilities + argmax predictions.
    - `calibrated_only`: temperature scaled probabilities + argmax predictions.
    - `conformal_only`: temperature scaled probabilities + split conformal prediction sets (no abstention/selective engine).
    - `full_saup`: full `SAUPDecisionEngine` with shift detection, conformal prediction, and selective action.
  - Recompute and compare all metrics consistently.

---

### Defect 7: Multi-Dataset Reporting Gaps & Bank Class-1 Coverage Explanation
- **Location**: `reports/`, `README.md`
- **Severity**: **Medium**
- **Evidence**:
  1. `README.md` and initial reports focused on UCI Adult, despite full datasets and models existing for Bank Marketing and Wine Quality.
  2. In Bank Marketing, reported class-1 coverage was ~27.98% while overall coverage was ~89.88%.
     - *Cause*: Bank Marketing has severe class imbalance (~88% Class 0, ~12% Class 1). Standard split conformal prediction guarantees marginal coverage $P(Y \in C(X)) \ge 1 - \alpha = 0.90$, dominated by majority class. It does *not* guarantee class-conditional coverage.
- **Planned Fix**:
  - Generate comprehensive reports for all three datasets and cross-dataset synthesis in `reports/audit_corrected/`.
  - Provide rigorous statistical discussion of marginal vs. class-conditional conformal coverage under imbalance.
  - Keep historical outputs immutable in `reports/results/`.

---

## Action Plan and Execution Phases

1. **Phase 2**: Refactor `src/saup/conformal.py` (`WeightedConformalClassifier` with exact mathematical p-values, tie handling, validation).
2. **Phase 3**: Refactor `src/saup/shift_detection.py` and `src/saup/metrics.py` (StandardScaler integration on source, median heuristic stability, unbiased unclipped MMD).
3. **Phase 4**: Refactor `src/saup/shift_generation.py` and `src/saup/experiment_runner.py` (guarantee `test_df` as basis for shifted evaluations, eliminating training contamination).
4. **Phase 5**: Refactor `src/saup/decision_engine.py` (principled adaptation dispatch, exact mutually exclusive outcome fractions, distinct coverage names).
5. **Phase 6 & 7 & 8**: Verify split conformal quantiles, BBSE conditioning guards, and temperature scaling metrics.
6. **Phase 9**: Repair `run_ablation()` to execute real `SAUPDecisionEngine`.
7. **Phase 10 & 11**: Rerun affected experiments for Adult, Bank, and Wine into `reports/audit_corrected/results/`. Regenerate figures in `reports/audit_corrected/figures/`.
8. **Phase 12 & 13**: Add comprehensive unit tests for all identified defects. Verify test suite passes.
9. **Phase 14**: Generate full suite of audit reports and documentation.
