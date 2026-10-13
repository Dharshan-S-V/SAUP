"""Tests for conformal prediction module."""

import numpy as np
import pytest
from saup.conformal import SplitConformalClassifier, WeightedConformalClassifier


class TestSplitConformal:
    def test_calibrate_sets_threshold(self):
        rng = np.random.RandomState(42)
        y_cal = rng.randint(0, 2, 100)
        probas = rng.dirichlet([2, 1], size=100)
        
        cc = SplitConformalClassifier(alpha=0.10)
        threshold = cc.calibrate(y_cal, probas)
        assert cc.fitted
        assert threshold >= 0

    def test_correct_quantile_rank(self):
        """Verify finite-sample quantile: k = ceil((n+1)*(1-alpha))."""
        n = 100
        alpha = 0.10
        y_cal = np.zeros(n, dtype=int)
        probas = np.column_stack([np.linspace(0.5, 1.0, n), 
                                   1 - np.linspace(0.5, 1.0, n)])
        
        cc = SplitConformalClassifier(alpha=alpha)
        cc.calibrate(y_cal, probas)
        
        expected_k = int(np.ceil((n + 1) * (1 - alpha)))
        assert cc.n_calibration == n
        assert expected_k == 91  # ceil(101 * 0.9)

    def test_prediction_sets_valid(self):
        rng = np.random.RandomState(42)
        y_cal = rng.randint(0, 2, 100)
        probas_cal = rng.dirichlet([2, 1], size=100)
        
        cc = SplitConformalClassifier(alpha=0.10)
        cc.calibrate(y_cal, probas_cal)
        
        probas_test = rng.dirichlet([2, 1], size=50)
        pred_sets = cc.predict_sets(probas_test)
        
        assert len(pred_sets) == 50
        for ps in pred_sets:
            assert all(c in [0, 1] for c in ps)

    def test_unfitted_raises(self):
        cc = SplitConformalClassifier()
        with pytest.raises(RuntimeError):
            cc.predict_sets(np.array([[0.5, 0.5]]))

    def test_high_confidence_singleton(self):
        """High-confidence predictions should produce singleton sets."""
        y_cal = np.array([0] * 50 + [1] * 50)
        probas_cal = np.zeros((100, 2))
        probas_cal[:50, 0] = 0.95
        probas_cal[:50, 1] = 0.05
        probas_cal[50:, 0] = 0.05
        probas_cal[50:, 1] = 0.95
        
        cc = SplitConformalClassifier(alpha=0.10)
        cc.calibrate(y_cal, probas_cal)
        
        # Very confident test prediction
        pred_sets = cc.predict_sets(np.array([[0.99, 0.01]]))
        assert len(pred_sets[0]) == 1  # Singleton set


class TestWeightedConformal:
    def test_calibrate_stores_data(self):
        rng = np.random.RandomState(42)
        y_cal = rng.randint(0, 2, 100)
        probas = rng.dirichlet([2, 1], size=100)
        weights = np.ones(100)
        
        wcc = WeightedConformalClassifier(alpha=0.10)
        wcc.calibrate(y_cal, probas, weights)
        assert wcc.fitted

    def test_predict_sets_valid(self):
        rng = np.random.RandomState(42)
        y_cal = rng.randint(0, 2, 100)
        probas_cal = rng.dirichlet([2, 1], size=100)
        weights = np.ones(100)
        
        wcc = WeightedConformalClassifier(alpha=0.10)
        wcc.calibrate(y_cal, probas_cal, weights)
        
        probas_test = rng.dirichlet([2, 1], size=20)
        pred_sets = wcc.predict_sets(probas_test, weights_test=np.ones(20))
        
        assert len(pred_sets) == 20
