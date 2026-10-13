"""Comprehensive evaluation metrics for classification, calibration, conformal, selective prediction, and shift detection."""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


# ─────────────────────────── 6.1 Classification ────────────────────────────

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics.
    
    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Predicted probabilities, shape (n,2) or (n,) for positive class.
        
    Returns:
        Dictionary of metric names to values.
    """
    metrics = {}
    
    # Handle edge case: single class in y_true
    unique_classes = np.unique(y_true)
    
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0.0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0.0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0.0)
    
    if y_prob is not None:
        prob_pos = _get_positive_prob(y_prob)
        
        if len(unique_classes) >= 2:
            metrics["roc_auc"] = roc_auc_score(y_true, prob_pos)
            metrics["pr_auc"] = average_precision_score(y_true, prob_pos)
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        
    Returns:
        Confusion matrix array.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


# ─────────────────────────── 6.2 Calibration ───────────────────────────────

def _get_positive_prob(y_prob: np.ndarray) -> np.ndarray:
    """Extract positive-class probability.
    
    Uses column 1 if 2D, otherwise returns as-is.
    """
    if y_prob.ndim == 2:
        return y_prob[:, 1]
    return y_prob


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute probability calibration metrics.
    
    Convention: all metrics use the positive-class probability (column 1).
    
    Args:
        y_true: Ground-truth labels (0/1).
        y_prob: Predicted probabilities, shape (n,2) or (n,).
        
    Returns:
        Dictionary of calibration metric values.
    """
    prob_pos = _get_positive_prob(y_prob)
    prob_pos = np.clip(prob_pos, 1e-15, 1 - 1e-15)
    
    metrics = {}
    metrics["brier_score"] = brier_score_loss(y_true, prob_pos)
    metrics["log_loss"] = log_loss(y_true, prob_pos)
    metrics["ece"] = expected_calibration_error(y_true, prob_pos)
    metrics["mce"] = maximum_calibration_error(y_true, prob_pos)
    
    return metrics


def expected_calibration_error(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    Bins are defined uniformly over [0, 1]. ECE is the weighted average
    of |accuracy - confidence| per bin, weighted by bin count.
    
    Args:
        y_true: Ground-truth binary labels.
        prob_pos: Positive-class predicted probabilities.
        n_bins: Number of confidence bins.
        strategy: 'uniform' for equal-width bins.
        
    Returns:
        ECE value in [0, 1].
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    
    if total == 0:
        return 0.0
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (prob_pos >= lo) & (prob_pos <= hi)
        else:
            mask = (prob_pos >= lo) & (prob_pos < hi)
        
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        
        avg_confidence = prob_pos[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_bin / total) * abs(avg_accuracy - avg_confidence)
    
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error (MCE).
    
    Args:
        y_true: Ground-truth binary labels.
        prob_pos: Positive-class predicted probabilities.
        n_bins: Number of bins.
        
    Returns:
        MCE value.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (prob_pos >= lo) & (prob_pos <= hi)
        else:
            mask = (prob_pos >= lo) & (prob_pos < hi)
        
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        
        avg_confidence = prob_pos[mask].mean()
        avg_accuracy = y_true[mask].mean()
        mce = max(mce, abs(avg_accuracy - avg_confidence))
    
    return mce


def reliability_diagram_data(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute data for reliability diagrams.
    
    Args:
        y_true: Ground-truth labels.
        prob_pos: Positive-class probabilities.
        n_bins: Number of bins.
        
    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_counts).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    accuracies = []
    counts = []
    
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (prob_pos >= lo) & (prob_pos <= hi)
        else:
            mask = (prob_pos >= lo) & (prob_pos < hi)
        
        n_bin = mask.sum()
        centers.append((lo + hi) / 2)
        counts.append(n_bin)
        if n_bin > 0:
            accuracies.append(y_true[mask].mean())
        else:
            accuracies.append(np.nan)
    
    return np.array(centers), np.array(accuracies), np.array(counts)


# ─────────────────────────── 6.3 Conformal ─────────────────────────────────

def conformal_metrics(
    y_true: np.ndarray,
    prediction_sets: List[List[int]],
    alpha: float = 0.10,
) -> Dict[str, float]:
    """Compute conformal prediction metrics.
    
    Args:
        y_true: Ground-truth labels.
        prediction_sets: List of prediction sets (list of class labels per sample).
        alpha: Significance level (nominal coverage = 1 - alpha).
        
    Returns:
        Dictionary of conformal metrics.
    """
    n = len(y_true)
    nominal_coverage = 1 - alpha
    
    # Coverage: fraction of samples where true label is in the prediction set
    covered = sum(1 for i in range(n) if y_true[i] in prediction_sets[i])
    empirical_coverage = covered / n if n > 0 else 0.0
    
    # Set sizes
    set_sizes = [len(s) for s in prediction_sets]
    avg_set_size = np.mean(set_sizes) if n > 0 else 0.0
    
    # Empty and full sets
    empty_rate = sum(1 for s in set_sizes if s == 0) / n if n > 0 else 0.0
    full_rate = sum(1 for s in set_sizes if s == 2) / n if n > 0 else 0.0  # binary
    
    # Class-wise coverage
    class_coverage = {}
    for c in [0, 1]:
        mask = y_true == c
        if mask.sum() > 0:
            c_covered = sum(
                1 for i in range(n) if mask[i] and y_true[i] in prediction_sets[i]
            )
            class_coverage[c] = c_covered / mask.sum()
        else:
            class_coverage[c] = float("nan")
    
    return {
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": empirical_coverage,
        "conformal_empirical_coverage": empirical_coverage,
        "coverage_gap": empirical_coverage - nominal_coverage,
        "conformal_coverage_gap": empirical_coverage - nominal_coverage,
        "avg_set_size": avg_set_size,
        "avg_prediction_set_size": avg_set_size,
        "empty_set_rate": empty_rate,
        "full_set_rate": full_rate,
        "class_0_coverage": class_coverage[0],
        "class_1_coverage": class_coverage[1],
    }


# ─────────────────────────── 6.4 Selective Prediction ──────────────────────

def selective_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accepted_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute selective prediction metrics.
    
    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        accepted_mask: Boolean array indicating accepted predictions.
        
    Returns:
        Dictionary of selective prediction metrics.
    """
    n = len(y_true)
    n_accepted = int(accepted_mask.sum())
    
    if n == 0:
        return {
            "accepted_coverage": 0.0,
            "accepted_fraction": 0.0,
            "coverage": 0.0,
            "selective_risk": float("nan"),
            "abstention_rate": 1.0,
            "selective_accuracy": float("nan"),
            "accepted_prediction_accuracy": float("nan"),
        }
    
    coverage = n_accepted / n
    abstention_rate = 1.0 - coverage
    
    if n_accepted > 0:
        selective_accuracy = float(accuracy_score(y_true[accepted_mask], y_pred[accepted_mask]))
        selective_risk = float(1.0 - selective_accuracy)
    else:
        selective_accuracy = float("nan")
        selective_risk = float("nan")
    
    return {
        "accepted_coverage": coverage,
        "accepted_fraction": coverage,
        "coverage": coverage,
        "selective_risk": selective_risk,
        "abstention_rate": abstention_rate,
        "selective_accuracy": selective_accuracy,
        "accepted_prediction_accuracy": selective_accuracy,
    }


def risk_coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute risk-coverage curve by sweeping confidence thresholds.
    
    Higher confidence → accept. We sweep from low to high threshold.
    
    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        confidence: Confidence scores (higher = more confident).
        n_thresholds: Number of threshold levels.
        
    Returns:
        Tuple of (coverages, risks) arrays.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    coverages = []
    risks = []
    
    for t in thresholds:
        accepted = confidence >= t
        n_accepted = accepted.sum()
        coverage = n_accepted / len(y_true) if len(y_true) > 0 else 0.0
        coverages.append(coverage)
        
        if n_accepted > 0:
            risk = 1.0 - accuracy_score(y_true[accepted], y_pred[accepted])
        else:
            risk = 0.0
        risks.append(risk)
    
    return np.array(coverages), np.array(risks)


def area_under_risk_coverage(
    coverages: np.ndarray, risks: np.ndarray
) -> float:
    """Compute area under the risk-coverage curve.
    
    Lower AURC = better. We compute using trapezoidal rule.
    The curve is sorted by descending coverage (left to right).
    
    Args:
        coverages: Coverage values.
        risks: Risk values.
        
    Returns:
        AURC value.
    """
    # Sort by coverage ascending
    order = np.argsort(coverages)
    c_sorted = coverages[order]
    r_sorted = risks[order]
    return float(np.trapz(r_sorted, c_sorted))


# ─────────────────────────── 6.5 Shift Metrics ────────────────────────────

def ks_test(source: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.
    
    Args:
        source: Source distribution samples.
        target: Target distribution samples.
        
    Returns:
        Tuple of (KS statistic, p-value).
    """
    stat, pval = stats.ks_2samp(source, target)
    return float(stat), float(pval)


def population_stability_index(
    source: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index for numerical features.
    
    Bins are defined on the source distribution.
    
    Args:
        source: Source samples.
        target: Target samples.
        n_bins: Number of bins.
        eps: Smoothing constant to avoid division by zero.
        
    Returns:
        PSI value. Heuristic thresholds: <0.1 minor, <0.2 moderate, >=0.25 major.
    """
    # Define bins from source
    bin_edges = np.percentile(source, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Remove duplicate edges
    bin_edges = np.unique(bin_edges)
    
    source_counts = np.histogram(source, bins=bin_edges)[0].astype(float)
    target_counts = np.histogram(target, bins=bin_edges)[0].astype(float)
    
    # Proportions with smoothing
    source_pct = (source_counts + eps) / (source_counts.sum() + eps * len(source_counts))
    target_pct = (target_counts + eps) / (target_counts.sum() + eps * len(target_counts))
    
    psi = np.sum((target_pct - source_pct) * np.log(target_pct / source_pct))
    return float(psi)


def categorical_psi(
    source: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute PSI for categorical features.
    
    Args:
        source: Source category values.
        target: Target category values.
        eps: Smoothing constant.
        
    Returns:
        PSI value.
    """
    # Filter out NaN values (from '?' -> NaN conversion in categorical columns)
    source_clean = np.array([x for x in source if pd.notna(x)])
    target_clean = np.array([x for x in target if pd.notna(x)])
    
    if len(source_clean) == 0 or len(target_clean) == 0:
        return 0.0
    
    all_categories = set(np.unique(source_clean)) | set(np.unique(target_clean))
    
    source_counts = {}
    target_counts = {}
    for cat in all_categories:
        source_counts[cat] = np.sum(source_clean == cat)
        target_counts[cat] = np.sum(target_clean == cat)
    
    n_source = len(source_clean)
    n_target = len(target_clean)
    n_cats = len(all_categories)
    
    psi = 0.0
    for cat in all_categories:
        s_pct = (source_counts[cat] + eps) / (n_source + eps * n_cats)
        t_pct = (target_counts[cat] + eps) / (n_target + eps * n_cats)
        psi += (t_pct - s_pct) * np.log(t_pct / s_pct)
    
    return float(psi)


def maximum_mean_discrepancy(
    source: np.ndarray,
    target: np.ndarray,
    bandwidth: Optional[float] = None,
    subsample: int = 2000,
    unbiased: bool = True,
    seed: int = 42,
) -> float:
    """Compute Maximum Mean Discrepancy (MMD^2) with RBF kernel.
    
    Supports both unbiased (Gretton et al., 2012) and biased estimators.
    Uses median heuristic on pairwise distances for RBF kernel bandwidth sigma.
    
    Args:
        source: Source feature matrix, shape (n, d).
        target: Target feature matrix, shape (m, d).
        bandwidth: RBF kernel bandwidth (sigma). If None, uses median distance heuristic.
        subsample: Maximum number of samples from each distribution (deterministic via seed).
        unbiased: If True, uses unbiased MMD^2 estimator (can be slightly negative around 0);
                  if False, uses biased V-statistic estimator (always non-negative).
        seed: Random seed for reproducible subsampling.
        
    Returns:
        Computed MMD^2 value (unrounded float).
    """
    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    
    if source_arr.ndim == 1:
        source_arr = source_arr.reshape(-1, 1)
    if target_arr.ndim == 1:
        target_arr = target_arr.reshape(-1, 1)
        
    if len(source_arr) == 0 or len(target_arr) == 0:
        return 0.0
        
    rng = np.random.RandomState(seed)
    if len(source_arr) > subsample:
        idx_s = rng.choice(len(source_arr), subsample, replace=False)
        source_arr = source_arr[idx_s]
    if len(target_arr) > subsample:
        idx_t = rng.choice(len(target_arr), subsample, replace=False)
        target_arr = target_arr[idx_t]
        
    n = len(source_arr)
    m = len(target_arr)
    
    # Bandwidth selection via median heuristic
    if bandwidth is None:
        n_med_s = min(500, n)
        n_med_t = min(500, m)
        combined = np.vstack([source_arr[:n_med_s], target_arr[:n_med_t]])
        dists = _pairwise_sq_dists(combined, combined)
        # Filter out self-distances and duplicate point zero distances
        pos_dists = dists[dists > 1e-10]
        if len(pos_dists) > 0:
            median_sq = np.median(pos_dists)
            bandwidth = np.sqrt(median_sq / 2.0)
            if bandwidth < 1e-6:
                bandwidth = 1.0
        else:
            bandwidth = 1.0
            
    gamma = 1.0 / (2.0 * (bandwidth ** 2))
    
    K_ss = _rbf_kernel(source_arr, source_arr, gamma)
    K_tt = _rbf_kernel(target_arr, target_arr, gamma)
    K_st = _rbf_kernel(source_arr, target_arr, gamma)
    
    if unbiased:
        # Unbiased estimator: exclude diagonals (n*(n-1))
        if n < 2 or m < 2:
            # Fallback to biased for single points
            return float(K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean())
        np.fill_diagonal(K_ss, 0.0)
        np.fill_diagonal(K_tt, 0.0)
        mmd2 = (K_ss.sum() / (n * (n - 1)) +
                K_tt.sum() / (m * (m - 1)) -
                2.0 * K_st.sum() / (n * m))
    else:
        # Biased V-statistic estimator
        mmd2 = K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean()
        mmd2 = max(0.0, mmd2)
        
    return float(mmd2)


def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances with numerical non-negativity guard."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = XX + YY.T - 2.0 * np.dot(X, Y.T)
    return np.maximum(dists, 0.0)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel matrix."""
    dists = _pairwise_sq_dists(X, Y)
    return np.exp(-gamma * dists)


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute Kish's effective sample size from importance weights.
    
    ESS = (sum(w))^2 / sum(w^2)
    
    Args:
        weights: Non-negative importance weights.
        
    Returns:
        ESS value.
    """
    w = np.asarray(weights, dtype=float)
    w_sum = w.sum()
    if w_sum < 1e-15:
        return 0.0
    return float(w_sum ** 2 / np.sum(w ** 2))
