"""Probability calibration via temperature scaling."""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """Temperature scaling for probability calibration.
    
    Fits a single positive temperature parameter T by minimizing
    negative log-likelihood on calibration data. For models producing
    probabilities (not logits), we work in log-probability space.
    
    Attributes:
        temperature: The fitted temperature parameter.
        fitted: Whether the scaler has been fitted.
    """
    
    def __init__(self):
        self.temperature: float = 1.0
        self.fitted: bool = False
    
    def fit(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> "TemperatureScaler":
        """Fit temperature parameter on calibration data.
        
        Args:
            y_true: Ground-truth labels (0/1).
            probabilities: Model predicted probabilities, shape (n, 2).
            lr: Not used (using scipy optimizer instead).
            max_iter: Not used (using scipy optimizer instead).
            
        Returns:
            Self.
        """
        # Clip probabilities to avoid log(0)
        probs = np.clip(probabilities, 1e-10, 1 - 1e-10)
        logits = np.log(probs)
        
        def neg_log_likelihood(T):
            if T <= 0:
                return 1e10
            scaled_logits = logits / T
            # Softmax
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            softmax_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            softmax_probs = np.clip(softmax_probs, 1e-15, 1.0)
            
            # NLL
            nll = -np.mean(
                np.log(softmax_probs[np.arange(len(y_true)), y_true.astype(int)])
            )
            return nll
        
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(0.01, 10.0),
            method="bounded",
        )
        
        self.temperature = result.x
        self.fitted = True
        
        logger.info(
            "Temperature scaling fitted: T=%.4f (NLL=%.4f)",
            self.temperature, result.fun
        )
        
        return self
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities.
        
        Args:
            probabilities: Model predicted probabilities, shape (n, 2).
            
        Returns:
            Calibrated probabilities, shape (n, 2).
        """
        if not self.fitted:
            raise RuntimeError("TemperatureScaler not fitted")
        
        probs = np.clip(probabilities, 1e-10, 1 - 1e-10)
        logits = np.log(probs)
        
        scaled_logits = logits / self.temperature
        
        # Softmax
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Validate
        calibrated = np.clip(calibrated, 1e-15, 1.0)
        row_sums = calibrated.sum(axis=1)
        calibrated = calibrated / row_sums[:, np.newaxis]
        
        assert np.all(np.isfinite(calibrated)), "Non-finite calibrated probabilities"
        
        return calibrated
    
    def fit_transform(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Fit and then transform probabilities.
        
        Args:
            y_true: Ground-truth labels.
            probabilities: Model probabilities.
            
        Returns:
            Calibrated probabilities.
        """
        self.fit(y_true, probabilities, **kwargs)
        return self.transform(probabilities)
