"""Black Box Shift Estimation (BBSE) for label-shift correction."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BBSEEstimator:
    """Black Box Shift Estimation for label shift.
    
    Estimates target class priors using:
    1. Source confusion matrix (from labelled validation data)
    2. Target class-prediction frequencies (from unlabelled target data)
    
    Assumptions:
    - Label shift: P(X|Y) is preserved, only P(Y) changes.
    - The source classifier's confusion matrix is invertible.
    
    Limitations:
    - Assumes the classifier's error pattern is stable under shift.
    - Estimated priors may be inaccurate under covariate shift.
    - Ill-conditioned confusion matrices lead to unstable estimates.
    """
    
    def __init__(self, n_classes: int = 2):
        self.n_classes = n_classes
        self.confusion_matrix: Optional[np.ndarray] = None
        self.source_priors: Optional[np.ndarray] = None
        self.is_stable: bool = True
        self.condition_number: float = 0.0
        self.fitted = False
    
    def fit(
        self,
        y_source: np.ndarray,
        y_pred_source: np.ndarray,
    ) -> "BBSEEstimator":
        """Estimate the source confusion matrix.
        
        Args:
            y_source: True labels from labelled source validation data.
            y_pred_source: Predicted labels on the same data.
            
        Returns:
            Self.
        """
        # Confusion matrix: C[i,j] = P(Y_hat = j | Y = i)
        C = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            mask = y_source == i
            if mask.sum() > 0:
                for j in range(self.n_classes):
                    C[i, j] = np.sum(y_pred_source[mask] == j) / mask.sum()
            else:
                C[i, :] = 1.0 / self.n_classes  # Uniform if no samples
        
        self.confusion_matrix = C
        self.source_priors = np.array([
            (y_source == i).mean() for i in range(self.n_classes)
        ])
        
        # Check conditioning
        self.condition_number = float(np.linalg.cond(C))
        self.is_stable = self.condition_number < 100
        
        if not self.is_stable:
            logger.warning(
                "BBSE confusion matrix is ill-conditioned (cond=%.2f). "
                "Estimates may be unreliable.",
                self.condition_number
            )
        
        self.fitted = True
        logger.info(
            "BBSE fitted: confusion matrix cond=%.2f, source priors=%s",
            self.condition_number, self.source_priors
        )
        
        return self
    
    def estimate_target_priors(
        self,
        y_pred_target: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """Estimate target class priors from unlabelled predictions.
        
        Args:
            y_pred_target: Predicted labels on target data (no true labels used).
            
        Returns:
            Tuple of (estimated_priors, info_dict).
        """
        if not self.fitted:
            raise RuntimeError("BBSEEstimator not fitted")
        
        # Target prediction frequencies: mu_hat[j] = P(Y_hat = j) on target
        mu_hat = np.array([
            (y_pred_target == j).mean() for j in range(self.n_classes)
        ])
        
        # Solve: C^T * w = mu_hat
        # where w[i] = P_target(Y=i)
        try:
            C_T = self.confusion_matrix.T
            estimated_priors = np.linalg.solve(C_T, mu_hat)
        except np.linalg.LinAlgError:
            logger.error("BBSE: Singular confusion matrix, falling back to source priors")
            return self.source_priors.copy(), {
                "reliable": False,
                "reason": "singular_confusion_matrix",
            }
        
        # Clip and normalize (non-negative least squares approximation)
        estimated_priors = np.clip(estimated_priors, 0, None)
        prior_sum = estimated_priors.sum()
        if prior_sum < 1e-10:
            logger.warning("BBSE: Estimated priors sum to ~0, using source priors")
            return self.source_priors.copy(), {
                "reliable": False,
                "reason": "zero_sum_priors",
            }
        
        estimated_priors = estimated_priors / prior_sum
        
        info = {
            "reliable": self.is_stable,
            "condition_number": self.condition_number,
            "source_priors": self.source_priors.tolist(),
            "estimated_target_priors": estimated_priors.tolist(),
            "target_pred_frequencies": mu_hat.tolist(),
        }
        
        logger.info(
            "BBSE estimated target priors: %s (reliable=%s)",
            estimated_priors, info["reliable"]
        )
        
        return estimated_priors, info
    
    def correct_probabilities(
        self,
        probabilities: np.ndarray,
        target_priors: np.ndarray,
    ) -> np.ndarray:
        """Correct predicted probabilities using estimated target priors.
        
        Applies Bayes' rule: P_target(Y|X) ∝ P_source(Y|X) * w(Y)
        where w(Y) = P_target(Y) / P_source(Y).
        
        Args:
            probabilities: Source-calibrated probabilities, shape (n, K).
            target_priors: Estimated target class priors, shape (K,).
            
        Returns:
            Corrected probabilities, shape (n, K).
        """
        # Prior ratio weights
        eps = 1e-10
        w = target_priors / (self.source_priors + eps)
        w = np.clip(w, eps, 100)
        
        # Apply correction
        corrected = probabilities * w[np.newaxis, :]
        
        # Renormalize
        row_sums = corrected.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, eps, None)
        corrected = corrected / row_sums
        
        # Validate
        corrected = np.clip(corrected, eps, 1.0)
        corrected = corrected / corrected.sum(axis=1, keepdims=True)
        
        assert np.all(np.isfinite(corrected)), "Non-finite corrected probabilities"
        
        return corrected
