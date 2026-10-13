"""Split conformal prediction with standard and importance-weighted variants."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from saup.metrics import effective_sample_size

logger = logging.getLogger(__name__)


class SplitConformalClassifier:
    """Standard split conformal classification.
    
    Uses nonconformity score: s(x, y) = 1 - p(y | x)
    
    The finite-sample quantile is computed as:
        k = ceil((n+1) * (1 - alpha))
    and the threshold is the k-th smallest nonconformity score,
    capped at n (in which case threshold = infinity, all classes included).
    """
    
    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Significance level. Nominal coverage = 1 - alpha.
        """
        self.alpha = alpha
        self.threshold: Optional[float] = None
        self.n_calibration: int = 0
        self.fitted = False
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        probabilities_cal: np.ndarray,
    ) -> float:
        """Compute the conformal threshold from calibration data.
        
        Args:
            y_cal: Calibration labels.
            probabilities_cal: Calibration probabilities, shape (n, 2).
            
        Returns:
            The computed threshold.
        """
        n = len(y_cal)
        self.n_calibration = n
        
        # Nonconformity scores: s_i = 1 - p(y_true | x)
        scores = 1.0 - probabilities_cal[np.arange(n), y_cal.astype(int)]
        
        # Finite-sample quantile rank
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        
        sorted_scores = np.sort(scores)
        
        if k > n:
            # Threshold = infinity → include all classes
            self.threshold = float("inf")
            logger.warning(
                "Conformal rank k=%d > n=%d; threshold=inf (all classes included)",
                k, n
            )
        else:
            self.threshold = sorted_scores[k - 1]  # 0-indexed
        
        self.fitted = True
        logger.info(
            "Conformal calibrated: alpha=%.2f, n=%d, k=%d, threshold=%.4f",
            self.alpha, n, k, self.threshold
        )
        return self.threshold
    
    def predict_sets(
        self,
        probabilities: np.ndarray,
    ) -> List[List[int]]:
        """Generate prediction sets for new data.
        
        A class y is included if: 1 - p(y|x) <= threshold.
        
        Args:
            probabilities: Predicted probabilities, shape (n, 2).
            
        Returns:
            List of prediction sets (each a list of class indices).
        """
        if not self.fitted:
            raise RuntimeError("Conformal classifier not calibrated")
        
        n_classes = probabilities.shape[1]
        prediction_sets = []
        
        for i in range(len(probabilities)):
            pred_set = []
            for c in range(n_classes):
                score = 1.0 - probabilities[i, c]
                if score <= self.threshold:
                    pred_set.append(c)
            prediction_sets.append(pred_set)
        
        return prediction_sets
class WeightedConformalClassifier:
    """Importance-weighted conformal prediction for covariate shift.
    
    Implements the weighted conformal prediction framework (Tibshirani et al., 2019,
    "Conformal Prediction Under Covariate Shift").
    
    Mathematical Formulation:
    For nonconformity score s(x, y) = 1 - p(y | x), calibration observations
    (x_1, y_1), ..., (x_n, y_n) have nonconformity scores V_1, ..., V_n with
    importance weights w_1, ..., w_n where w(x) = p_target(x) / p_source(x).
    
    For a test point X_{n+1} with candidate class y and importance weight w_{n+1},
    the candidate nonconformity score is V_{n+1}^y = 1 - p(y | X_{n+1}).
    
    The conformal p-value for candidate label y is:
        p(y) = ( sum_{i=1}^n w_i * 1{V_i >= V_{n+1}^y} + w_{n+1} ) / ( sum_{i=1}^n w_i + w_{n+1} )
        
    Candidate class y is included in prediction set C(X_{n+1}) if:
        p(y) > alpha
        
    Tie Handling:
        Calibration scores V_i >= V_{n+1}^y are counted towards the p-value numerator.
        The test point itself contributes w_{n+1} to both numerator and denominator.
        
    Validity Note:
        Exact finite-sample coverage guarantees require known density ratios w(x).
        When density ratios are estimated (e.g., via domain classifier logistic regression)
        or clipped for stability, coverage holds approximately subject to estimation error.
    """
    
    def __init__(self, alpha: float = 0.10, weight_clip: float = 20.0):
        """
        Args:
            alpha: Significance level in (0, 1). Nominal coverage = 1 - alpha.
            weight_clip: Maximum importance weight cap for numerical stability.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        if weight_clip <= 0:
            raise ValueError(f"weight_clip must be positive, got {weight_clip}")
            
        self.alpha = float(alpha)
        self.weight_clip = float(weight_clip)
        self.scores: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.n_calibration: int = 0
        self.fitted = False
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        probabilities_cal: np.ndarray,
        weights_cal: np.ndarray,
    ) -> None:
        """Store calibration scores and importance weights with strict validation.
        
        Args:
            y_cal: Calibration labels, shape (n,).
            probabilities_cal: Calibration predicted probabilities, shape (n, K).
            weights_cal: Non-negative importance weights for calibration points, shape (n,).
        """
        y_arr = np.asarray(y_cal)
        probs_arr = np.asarray(probabilities_cal, dtype=float)
        weights_arr = np.asarray(weights_cal, dtype=float)
        
        n = len(y_arr)
        if n == 0:
            raise ValueError("Calibration set must not be empty.")
        if probs_arr.ndim != 2 or len(probs_arr) != n:
            raise ValueError(f"probabilities_cal must be shape (n, K), got {probs_arr.shape} for n={n}")
        if weights_arr.ndim != 1 or len(weights_arr) != n:
            raise ValueError(f"weights_cal must be 1D array of length {n}, got {weights_arr.shape}")
            
        if not np.all(np.isfinite(probs_arr)):
            raise ValueError("probabilities_cal contains non-finite values (NaN or Inf).")
        if not np.all(np.isfinite(weights_arr)):
            raise ValueError("weights_cal contains non-finite values (NaN or Inf).")
        if np.any(weights_arr < 0):
            raise ValueError("Importance weights must be non-negative.")
        if weights_arr.sum() <= 0:
            raise ValueError("Total calibration importance weight must be strictly positive.")
            
        # Nonconformity score s(x, y) = 1 - p(y | x)
        y_indices = y_arr.astype(int)
        if np.any(y_indices < 0) or np.any(y_indices >= probs_arr.shape[1]):
            raise ValueError(f"Calibration labels out of range [0, {probs_arr.shape[1] - 1}]")
            
        scores = 1.0 - probs_arr[np.arange(n), y_indices]
        clipped_weights = np.clip(weights_arr, 0.0, self.weight_clip)
        
        self.scores = scores
        self.weights = clipped_weights
        self.n_calibration = n
        self.fitted = True
        
        ess = effective_sample_size(self.weights)
        logger.info(
            "Weighted conformal calibrated: n=%d, ESS=%.1f, "
            "weight range=[%.3f, %.3f] (clipped at %.1f)",
            n, ess, self.weights.min(), self.weights.max(), self.weight_clip
        )
    
    def compute_p_values(
        self,
        probabilities: np.ndarray,
        weights_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute conformal p-values for all candidate classes.
        
        p(y) = ( sum_{i=1}^n w_i * 1{V_i >= V_{n+1}^y} + w_{n+1} ) / ( sum_{i=1}^n w_i + w_{n+1} )
        
        Args:
            probabilities: Test probabilities, shape (m, K).
            weights_test: Optional test-point importance weights, shape (m,).
            
        Returns:
            p_values: Conformal p-values, shape (m, K).
        """
        if not self.fitted:
            raise RuntimeError("WeightedConformalClassifier not calibrated.")
            
        probs_arr = np.asarray(probabilities, dtype=float)
        if probs_arr.ndim != 2:
            raise ValueError(f"probabilities must be 2D array, got shape {probs_arr.shape}")
        if not np.all(np.isfinite(probs_arr)):
            raise ValueError("probabilities contains non-finite values.")
            
        m, n_classes = probs_arr.shape
        if weights_test is None:
            w_test_arr = np.ones(m, dtype=float)
        else:
            w_test_arr = np.asarray(weights_test, dtype=float)
            if w_test_arr.ndim != 1 or len(w_test_arr) != m:
                raise ValueError(f"weights_test must be 1D array of length {m}")
            if not np.all(np.isfinite(w_test_arr)):
                raise ValueError("weights_test contains non-finite values.")
            if np.any(w_test_arr < 0):
                raise ValueError("Test weights must be non-negative.")
            w_test_arr = np.clip(w_test_arr, 0.0, self.weight_clip)
            
        cal_weight_sum = float(np.sum(self.weights))
        p_values = np.zeros((m, n_classes), dtype=float)
        
        for j in range(m):
            w_pt = float(w_test_arr[j])
            denom = cal_weight_sum + w_pt
            if denom <= 0:
                p_values[j, :] = 1.0
                continue
                
            test_scores = 1.0 - probs_arr[j]
            # mask: (n_cal, n_classes) indicating calibration scores >= test candidate score
            mask = self.scores[:, None] >= test_scores[None, :]
            # Weighted numerator: sum(w_i * 1{V_i >= test_score}) + w_{test}
            numerators = np.sum(self.weights[:, None] * mask, axis=0) + w_pt
            p_values[j] = numerators / denom
            
        return p_values

    def predict_sets(
        self,
        probabilities: np.ndarray,
        weights_test: Optional[np.ndarray] = None,
    ) -> List[List[int]]:
        """Generate weighted prediction sets.
        
        Candidate class c is included if conformal p-value p(c) > alpha.
        
        Args:
            probabilities: Test probabilities, shape (m, K).
            weights_test: Optional test-point weights, shape (m,).
            
        Returns:
            List of prediction sets (each a list of class indices).
        """
        p_values = self.compute_p_values(probabilities, weights_test)
        m, n_classes = p_values.shape
        
        prediction_sets = []
        for j in range(m):
            pred_set = [c for c in range(n_classes) if p_values[j, c] > self.alpha]
            prediction_sets.append(pred_set)
            
        return prediction_sets


def estimate_density_ratios(
    X_source: np.ndarray,
    X_target: np.ndarray,
    clip: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate density ratios using a domain classifier.
    
    Trains a logistic regression to distinguish source (label=0) from
    target (label=1) samples, then computes w(x) = p(target|x) / p(source|x).
    
    Args:
        X_source: Source feature matrix.
        X_target: Target feature matrix.
        clip: Maximum weight for clipping.
        
    Returns:
        Tuple of (source_weights, target_weights, ess).
        source_weights has shape (n_source,).
        target_weights has shape (n_target,).
    """
    n_s = len(X_source)
    n_t = len(X_target)
    
    X_combined = np.vstack([X_source, X_target])
    y_domain = np.concatenate([np.zeros(n_s), np.ones(n_t)])
    
    clf = LogisticRegression(
        max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
    )
    clf.fit(X_combined, y_domain)
    
    # Probabilities for source samples
    probs_source = clf.predict_proba(X_source)
    # w(x) = p(target|x) / p(source|x) * (n_source / n_target)
    p_target_given_x = np.clip(probs_source[:, 1], 1e-10, 1 - 1e-10)
    p_source_given_x = np.clip(probs_source[:, 0], 1e-10, 1 - 1e-10)
    
    weights_source = (p_target_given_x / p_source_given_x) * (n_s / n_t)
    weights_source = np.clip(weights_source, 0, clip)
    
    # Weights for target samples
    probs_target = clf.predict_proba(X_target)
    p_target_given_x_t = np.clip(probs_target[:, 1], 1e-10, 1 - 1e-10)
    p_source_given_x_t = np.clip(probs_target[:, 0], 1e-10, 1 - 1e-10)
    weights_target = (p_target_given_x_t / p_source_given_x_t) * (n_s / n_t)
    weights_target = np.clip(weights_target, 0, clip)
    
    ess = effective_sample_size(weights_source)
    
    logger.info(
        "Density ratios estimated: n_source=%d, n_target=%d, ESS=%.1f, "
        "weight range=[%.3f, %.3f]",
        n_s, n_t, ess, weights_source.min(), weights_source.max()
    )
    
    return weights_source, weights_target, ess
