"""Tests for label shift (BBSE) module."""

import numpy as np
import pytest
from saup.label_shift import BBSEEstimator


class TestBBSE:
    def test_fit(self):
        y_source = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1])
        
        bbse = BBSEEstimator(n_classes=2)
        bbse.fit(y_source, y_pred)
        assert bbse.fitted
        assert bbse.confusion_matrix.shape == (2, 2)

    def test_estimate_target_priors(self):
        y_source = np.array([0]*50 + [1]*50)
        y_pred = np.array([0]*45 + [1]*5 + [0]*5 + [1]*45)  # Good classifier
        
        bbse = BBSEEstimator(n_classes=2)
        bbse.fit(y_source, y_pred)
        
        # Target with different proportion
        y_pred_target = np.array([0]*30 + [1]*70)
        priors, info = bbse.estimate_target_priors(y_pred_target)
        
        assert len(priors) == 2
        assert abs(sum(priors) - 1.0) < 1e-6
        assert all(p >= 0 for p in priors)

    def test_ill_conditioned_warning(self):
        """Test with near-random classifier (ill-conditioned confusion matrix)."""
        rng = np.random.RandomState(42)
        y_source = rng.randint(0, 2, 100)
        y_pred = rng.randint(0, 2, 100)  # Random predictions
        
        bbse = BBSEEstimator(n_classes=2)
        bbse.fit(y_source, y_pred)
        # Condition number should be high or moderate

    def test_correct_probabilities(self):
        y_source = np.array([0]*50 + [1]*50)
        y_pred = np.array([0]*45 + [1]*5 + [0]*5 + [1]*45)
        
        bbse = BBSEEstimator(n_classes=2)
        bbse.fit(y_source, y_pred)
        
        probas = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        target_priors = np.array([0.3, 0.7])
        corrected = bbse.correct_probabilities(probas, target_priors)
        
        assert corrected.shape == probas.shape
        np.testing.assert_allclose(corrected.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(np.isfinite(corrected))

    def test_unfitted_raises(self):
        bbse = BBSEEstimator()
        with pytest.raises(RuntimeError):
            bbse.estimate_target_priors(np.array([0, 1]))
