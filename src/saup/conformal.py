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
    
    Uses estimated density ratios w(x) = p_target(x) / p_source(x)
    to compute a weighted conformal quantile.
    
    Limitations:
    - Requires estimated density ratios (typically from a domain classifier).
    - Does not automatically restore exact finite-sample guarantees
      if the density ratios are estimated rather than known.
    - Performance degrades with low effective sample size.
    """
    
    def __init__(self, alpha: float = 0.10, weight_clip: float = 20.0):
        """
        Args:
            alpha: Significance level.
            weight_clip: Maximum importance weight (for stability).
        """
        self.alpha = alpha
        self.weight_clip = weight_clip
        self.scores: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.fitted = False
    
    def calibrate(
        self,
        y_cal: np.ndarray,
        probabilities_cal: np.ndarray,
        weights_cal: np.ndarray,
    ) -> None:
        """Store calibration scores and weights.
        
        Args:
            y_cal: Calibration labels.
            probabilities_cal: Calibration probabilities, shape (n, 2).
            weights_cal: Importance weights for calibration points.
        """
        n = len(y_cal)
        self.scores = 1.0 - probabilities_cal[np.arange(n), y_cal.astype(int)]
        self.weights = np.clip(weights_cal, 0, self.weight_clip)
        self.fitted = True
        
        ess = effective_sample_size(self.weights)
        logger.info(
            "Weighted conformal calibrated: n=%d, ESS=%.1f, "
            "weight range=[%.3f, %.3f]",
            n, ess, self.weights.min(), self.weights.max()
        )
    
    def predict_sets(
        self,
        probabilities: np.ndarray,
        weights_test: Optional[np.ndarray] = None,
    ) -> List[List[int]]:
        """Generate weighted prediction sets.
        
        Uses the weighted quantile method including the test-point mass.
        
        Args:
            probabilities: Test probabilities, shape (n, 2).
            weights_test: Test-point weights (if available).
            
        Returns:
            List of prediction sets.
        """
        if not self.fitted:
            raise RuntimeError("Weighted conformal classifier not calibrated")
        
        n_classes = probabilities.shape[1]
        prediction_sets = []
        
        for i in range(len(probabilities)):
            pred_set = []
            w_test = weights_test[i] if weights_test is not None else 1.0
            w_test = min(w_test, self.weight_clip)
            
            for c in range(n_classes):
                test_score = 1.0 - probabilities[i, c]
                
                # Weighted quantile: include test point mass
                total_weight = self.weights.sum() + w_test
                
                # Weight of calibration points with score <= test_score
                below_weight = self.weights[self.scores <= test_score].sum()
                # Add test point weight
                below_weight += w_test
                
                # Include class if weighted fraction >= 1 - alpha
                # i.e., the test score is at or below the weighted quantile
                if below_weight / total_weight >= 1 - self.alpha:
                    # Score is high enough → not in prediction set
                    pass
                else:
                    pred_set.append(c)
                
            # Correction: include class if score is NOT too large
            # Rewrite: class c included if test_score <= weighted quantile
            pred_set = []
            for c in range(n_classes):
                test_score = 1.0 - probabilities[i, c]
                total_weight = self.weights.sum() + w_test
                
                # Weight of scores > test_score (calibration)
                above_weight = self.weights[self.scores > test_score].sum()
                
                # Proportion of weight at or below test_score
                at_or_below = total_weight - above_weight
                proportion_above = above_weight / total_weight
                
                # Include if: proportion of weight strictly above <= alpha
                if proportion_above <= self.alpha:
                    pass  # test_score is large, don't include
                else:
                    pred_set.append(c)
            
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
