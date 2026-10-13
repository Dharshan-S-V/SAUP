"""Distribution shift detection using KS test, PSI, and MMD across multiple datasets."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

from saup.data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from saup.metrics import (
    ks_test,
    population_stability_index,
    categorical_psi,
    maximum_mean_discrepancy,
)

logger = logging.getLogger(__name__)


class ShiftDetector:
    """Multi-method distribution shift detector.
    
    Combines KS tests, PSI, and MMD to assess whether target data
    has shifted from the source distribution. Does NOT use target labels.
    
    Attributes:
        source_stats: Stored source statistics for comparison.
        ks_alpha: Significance level for KS tests.
        psi_threshold: PSI threshold for flagging shift.
        mmd_threshold: MMD threshold (estimated from source-vs-source).
    """
    
    def __init__(
        self,
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.20,
        mmd_subsample: int = 2000,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold
        self.mmd_subsample = mmd_subsample
        self.user_numerical_features = numerical_features
        self.user_categorical_features = categorical_features
        
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.source_numerical: Optional[np.ndarray] = None
        self.source_numerical_scaled: Optional[np.ndarray] = None
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_scale_: Optional[np.ndarray] = None
        self.source_categorical: Dict[str, np.ndarray] = {}
        self.mmd_baseline: float = 0.0
        self.mmd_threshold: float = 0.0
        self.fitted = False
    
    def fit(
        self,
        X_source: pd.DataFrame,
        n_mmd_bootstrap: int = 10,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> "ShiftDetector":
        """Store source distribution reference and calibrate MMD threshold.
        
        Learns source-only standardization parameters and estimates baseline MMD
        distribution from independent source-vs-source splits. Does NOT use target labels.
        
        Args:
            X_source: Source feature DataFrame (without target).
            n_mmd_bootstrap: Number of bootstrap splits for MMD baseline.
            numerical_features: Optional explicit numerical feature list.
            categorical_features: Optional explicit categorical feature list.
            
        Returns:
            Self.
        """
        # Determine numerical features
        num_candidates = numerical_features or self.user_numerical_features
        if num_candidates is not None:
            self.numerical_cols = [c for c in num_candidates if c in X_source.columns]
        else:
            adult_match = [c for c in NUMERICAL_FEATURES if c in X_source.columns]
            if len(adult_match) > 0 and set(adult_match).issubset(set(X_source.columns)):
                self.numerical_cols = adult_match
            else:
                self.numerical_cols = list(X_source.select_dtypes(include=[np.number]).columns)
        
        # Determine categorical features
        cat_candidates = categorical_features or self.user_categorical_features
        if cat_candidates is not None:
            self.categorical_cols = [c for c in cat_candidates if c in X_source.columns]
        else:
            adult_cat_match = [c for c in CATEGORICAL_FEATURES if c in X_source.columns]
            if len(adult_cat_match) > 0:
                self.categorical_cols = adult_cat_match
            else:
                self.categorical_cols = list(X_source.select_dtypes(include=["object", "category"]).columns)
        
        # Fit source-only standardization parameters to prevent large-scale features from dominating MMD
        if len(self.numerical_cols) > 0:
            raw_vals = X_source[self.numerical_cols].values.astype(float)
            self.source_numerical = raw_vals
            
            self.scaler_mean_ = np.nanmean(raw_vals, axis=0)
            self.scaler_scale_ = np.nanstd(raw_vals, axis=0)
            # Guard against zero or near-zero variance features
            self.scaler_scale_[self.scaler_scale_ < 1e-8] = 1.0
            
            scaled_source = (raw_vals - self.scaler_mean_) / self.scaler_scale_
            self.source_numerical_scaled = np.nan_to_num(scaled_source, nan=0.0)
        else:
            self.source_numerical = np.zeros((len(X_source), 0))
            self.source_numerical_scaled = np.zeros((len(X_source), 0))
            self.scaler_mean_ = np.array([])
            self.scaler_scale_ = np.array([])
        
        # Store categorical reference arrays
        self.source_categorical = {
            col: X_source[col].dropna().values for col in self.categorical_cols
        }
        
        # MMD baseline: source vs source bootstrap on standardized features
        if self.source_numerical_scaled.shape[1] > 0:
            rng = np.random.RandomState(42)
            mmd_values = []
            n = len(self.source_numerical_scaled)
            half = n // 2
            for b_idx in range(n_mmd_bootstrap):
                perm = rng.permutation(n)
                mmd_val = maximum_mean_discrepancy(
                    self.source_numerical_scaled[perm[:half]],
                    self.source_numerical_scaled[perm[half:2*half]],
                    subsample=self.mmd_subsample,
                    seed=42 + b_idx,
                )
                mmd_values.append(mmd_val)
            
            self.mmd_baseline = float(np.mean(mmd_values))
            p95 = float(np.percentile(mmd_values, 95))
            two_std = float(np.mean(mmd_values) + 2.0 * np.std(mmd_values))
            self.mmd_threshold = float(max(p95, two_std, 1e-5))
        else:
            self.mmd_baseline = 0.0
            self.mmd_threshold = 0.0
        
        self.fitted = True
        logger.info(
            "ShiftDetector fitted on %d samples (%d num, %d cat). "
            "MMD baseline=%.8f, threshold=%.8f",
            len(X_source), len(self.numerical_cols), len(self.categorical_cols),
            self.mmd_baseline, self.mmd_threshold
        )
        return self
    
    def detect(
        self,
        X_target: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Run shift detection on target data.
        
        Does NOT use target labels.
        
        Args:
            X_target: Target feature DataFrame (without target).
            
        Returns:
            Shift report dictionary.
        """
        if not self.fitted:
            raise RuntimeError("ShiftDetector not fitted")
        
        report = {
            "n_source": len(self.source_numerical),
            "n_target": len(X_target),
            "ks_results": {},
            "psi_results": {},
            "mmd": {},
            "overall": {},
        }
        
        # ── KS tests on numerical features ──
        num_cols = [c for c in self.numerical_cols if c in X_target.columns]
        ks_stats = []
        ks_pvals = []
        
        for i, col in enumerate(num_cols):
            source_vals = self.source_numerical[:, i]
            target_vals = X_target[col].values.astype(float)
            
            # Handle constant columns
            if np.std(source_vals) < 1e-10 and np.std(target_vals) < 1e-10:
                stat, pval = 0.0, 1.0
            else:
                stat, pval = ks_test(source_vals, target_vals)
            
            ks_stats.append(stat)
            ks_pvals.append(pval)
            report["ks_results"][col] = {
                "statistic": stat,
                "p_value": pval,
            }
        
        # Benjamini-Hochberg correction
        if len(ks_pvals) > 0:
            rejected, corrected_pvals, _, _ = multipletests(
                ks_pvals, alpha=self.ks_alpha, method="fdr_bh"
            )
            for i, col in enumerate(num_cols):
                report["ks_results"][col]["corrected_p_value"] = float(corrected_pvals[i])
                report["ks_results"][col]["rejected"] = bool(rejected[i])
            
            n_rejected = int(np.sum(rejected))
            report["ks_results"]["n_rejected"] = n_rejected
            report["ks_results"]["n_tested"] = len(num_cols)
        
        # ── PSI on numerical features ──
        for i, col in enumerate(num_cols):
            source_vals = self.source_numerical[:, i]
            target_vals = X_target[col].values.astype(float)
            
            if np.std(source_vals) < 1e-10:
                psi_val = 0.0
            else:
                psi_val = population_stability_index(source_vals, target_vals)
            
            report["psi_results"][col] = {
                "psi": psi_val,
                "flag": "major" if psi_val >= 0.25 else (
                    "moderate" if psi_val >= 0.1 else "minor"
                ),
            }
        
        # Categorical PSI
        cat_cols = [c for c in self.categorical_cols if c in X_target.columns]
        for col in cat_cols:
            if col in self.source_categorical:
                target_cat_vals = X_target[col].dropna().values
                psi_val = categorical_psi(
                    self.source_categorical[col],
                    target_cat_vals,
                )
                report["psi_results"][col] = {
                    "psi": psi_val,
                    "flag": "major" if psi_val >= 0.25 else (
                        "moderate" if psi_val >= 0.1 else "minor"
                    ),
                }
        
        # ── MMD on standardized numerical features ──
        if len(num_cols) > 0 and self.source_numerical_scaled.shape[1] > 0:
            target_num = X_target[num_cols].values.astype(float)
            target_scaled = (target_num - self.scaler_mean_) / self.scaler_scale_
            target_scaled = np.nan_to_num(target_scaled, nan=0.0)
            mmd_val = maximum_mean_discrepancy(
                self.source_numerical_scaled, target_scaled, subsample=self.mmd_subsample, seed=42
            )
        else:
            mmd_val = 0.0
            
        report["mmd"]["value"] = float(mmd_val)
        report["mmd"]["baseline"] = float(self.mmd_baseline)
        report["mmd"]["threshold"] = float(self.mmd_threshold)
        report["mmd"]["detected"] = bool(mmd_val > self.mmd_threshold)
        
        # ── Overall assessment ──
        ks_detected = report["ks_results"].get("n_rejected", 0) > 0
        psi_detected = any(
            v.get("flag") in ("moderate", "major")
            for v in report["psi_results"].values()
            if isinstance(v, dict)
        )
        mmd_detected = report["mmd"]["detected"]
        
        # Severity estimation
        detectors_triggered = sum([ks_detected, psi_detected, mmd_detected])
        if detectors_triggered == 0:
            severity_estimate = "none"
            severity_score = 0.0
        elif detectors_triggered == 1:
            severity_estimate = "mild"
            severity_score = 0.3
        elif detectors_triggered == 2:
            severity_estimate = "moderate"
            severity_score = 0.6
        else:
            severity_estimate = "severe"
            severity_score = 1.0
        
        # Refine with PSI magnitudes
        max_psi = max(
            (v.get("psi", 0) for v in report["psi_results"].values() if isinstance(v, dict)),
            default=0
        )
        if max_psi > 0.5:
            severity_score = max(severity_score, 0.8)
            severity_estimate = "severe"
        
        report["overall"] = {
            "shift_detected": detectors_triggered > 0,
            "ks_detected": ks_detected,
            "psi_detected": psi_detected,
            "mmd_detected": mmd_detected,
            "detectors_triggered": detectors_triggered,
            "severity_estimate": severity_estimate,
            "severity_score": severity_score,
            "max_psi": max_psi,
            "mmd_ratio": mmd_val / max(self.mmd_threshold, 1e-10),
        }
        
        logger.info(
            "Shift detection: detected=%s, severity=%s (score=%.2f), "
            "KS=%s, PSI=%s, MMD=%s",
            report["overall"]["shift_detected"],
            severity_estimate, severity_score,
            ks_detected, psi_detected, mmd_detected,
        )
        
        return report
