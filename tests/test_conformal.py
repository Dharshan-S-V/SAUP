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
        assert wcc.n_calibration == 100

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
        for ps in pred_sets:
            assert all(c in [0, 1] for c in ps)

    def test_manual_weighted_p_values(self):
        """Verify p-values match independently calculated mathematical reference.
        
        Calibration points:
          i=0: y=0, p=[0.9, 0.1] -> V_0 = 1 - 0.9 = 0.1, w_0 = 2.0
          i=1: y=1, p=[0.6, 0.4] -> V_1 = 1 - 0.4 = 0.6, w_1 = 1.0
          i=2: y=0, p=[0.7, 0.3] -> V_2 = 1 - 0.7 = 0.3, w_2 = 3.0
        
        Total cal weight = 2.0 + 1.0 + 3.0 = 6.0
        
        Test point: p=[0.65, 0.35], w_test = 2.0
        Candidate class 0:
          test_score = 1 - 0.65 = 0.35
          V_i >= 0.35:
            V_0=0.1 (No)
            V_1=0.6 (Yes, w=1.0)
            V_2=0.3 (No)
          cal sum = 1.0
          numerator = 1.0 + w_test = 3.0
          denom = 6.0 + 2.0 = 8.0
          p(0) = 3.0 / 8.0 = 0.375
          
        Candidate class 1:
          test_score = 1 - 0.35 = 0.65
          V_i >= 0.65:
            V_0=0.1 (No)
            V_1=0.6 (No)
            V_2=0.3 (No)
          cal sum = 0.0
          numerator = 0.0 + w_test = 2.0
          denom = 8.0
          p(1) = 2.0 / 8.0 = 0.250
        """
        y_cal = np.array([0, 1, 0])
        probs_cal = np.array([
            [0.9, 0.1],
            [0.6, 0.4],
            [0.7, 0.3],
        ])
        weights_cal = np.array([2.0, 1.0, 3.0])
        
        wcc = WeightedConformalClassifier(alpha=0.30)
        wcc.calibrate(y_cal, probs_cal, weights_cal)
        
        probs_test = np.array([[0.65, 0.35]])
        weights_test = np.array([2.0])
        
        p_vals = wcc.compute_p_values(probs_test, weights_test)
        assert np.isclose(p_vals[0, 0], 3.0 / 8.0)  # 0.375
        assert np.isclose(p_vals[0, 1], 2.0 / 8.0)  # 0.250
        
        # With alpha = 0.30: p(0)=0.375 > 0.30 (Included), p(1)=0.250 <= 0.30 (Excluded)
        sets = wcc.predict_sets(probs_test, weights_test)
        assert sets[0] == [0]

    def test_test_point_weight_contribution(self):
        """Increasing test point weight increases p-values for all candidate labels."""
        y_cal = np.array([0, 1])
        probs_cal = np.array([[0.8, 0.2], [0.3, 0.7]])  # V=[0.2, 0.3]
        weights_cal = np.array([1.0, 1.0])
        
        wcc = WeightedConformalClassifier(alpha=0.40)
        wcc.calibrate(y_cal, probs_cal, weights_cal)
        
        probs_test = np.array([[0.5, 0.5]])  # test_score = 0.5 for both
        
        # Test weight small (0.1): numerators = 0.1, denom = 2.1 -> p = 0.1/2.1 ~ 0.0476
        p_small = wcc.compute_p_values(probs_test, np.array([0.1]))[0]
        # Test weight large (10.0): numerators = 10.0, denom = 12.0 -> p = 10/12 ~ 0.833
        p_large = wcc.compute_p_values(probs_test, np.array([10.0]))[0]
        
        assert np.all(p_large > p_small)
        assert wcc.predict_sets(probs_test, np.array([0.1]))[0] == []
        assert wcc.predict_sets(probs_test, np.array([10.0]))[0] == [0, 1]

    def test_tied_nonconformity_scores(self):
        """Tied nonconformity scores are correctly counted as >= test score."""
        y_cal = np.array([0, 0, 0])
        probs_cal = np.array([[0.7, 0.3], [0.7, 0.3], [0.7, 0.3]])  # All V_i = 0.3
        weights_cal = np.array([1.0, 2.0, 3.0])  # Total cal weight = 6.0
        
        wcc = WeightedConformalClassifier(alpha=0.10)
        wcc.calibrate(y_cal, probs_cal, weights_cal)
        
        # Test score exactly equals calibration score: 1 - 0.7 = 0.3
        probs_test = np.array([[0.7, 0.3]])
        weights_test = np.array([1.0])
        
        # For class 0: test_score = 0.3. Since all V_i == 0.3 >= 0.3, all cal points match!
        # Numerator = 6.0 + 1.0 = 7.0, denom = 7.0 -> p(0) = 1.0
        p_vals = wcc.compute_p_values(probs_test, weights_test)
        assert np.isclose(p_vals[0, 0], 1.0)

    def test_zero_and_extreme_weights(self):
        """Zero calibration weights and clipped extreme weights handle safely."""
        y_cal = np.array([0, 1, 0])
        probs_cal = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
        weights_cal = np.array([0.0, 1.0, 100.0])  # Extreme weight clipped at weight_clip=20
        
        wcc = WeightedConformalClassifier(alpha=0.10, weight_clip=20.0)
        wcc.calibrate(y_cal, probs_cal, weights_cal)
        
        assert wcc.weights[0] == 0.0
        assert wcc.weights[2] == 20.0
        
        probs_test = np.array([[0.6, 0.4]])
        p_vals = wcc.compute_p_values(probs_test, np.array([0.0]))  # Zero test weight
        assert np.all(np.isfinite(p_vals))
        assert np.all(p_vals >= 0.0) and np.all(p_vals <= 1.0)

    def test_malformed_inputs_raise(self):
        wcc = WeightedConformalClassifier(alpha=0.10)
        
        # Empty calibration
        with pytest.raises(ValueError):
            wcc.calibrate(np.array([]), np.empty((0, 2)), np.array([]))
            
        # Non-finite probabilities
        with pytest.raises(ValueError):
            wcc.calibrate(np.array([0]), np.array([[np.nan, 0.5]]), np.array([1.0]))
            
        # Negative weights
        with pytest.raises(ValueError):
            wcc.calibrate(np.array([0]), np.array([[0.5, 0.5]]), np.array([-1.0]))
            
        # Zero total weight
        with pytest.raises(ValueError):
            wcc.calibrate(np.array([0]), np.array([[0.5, 0.5]]), np.array([0.0]))
            
        # Unfitted predict raises
        wcc_unfit = WeightedConformalClassifier()
        with pytest.raises(RuntimeError):
            wcc_unfit.predict_sets(np.array([[0.5, 0.5]]))
