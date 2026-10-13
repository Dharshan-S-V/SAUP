"""Tests for calibration module."""

import numpy as np
import pytest
from saup.calibration import TemperatureScaler


class TestTemperatureScaler:
    def test_fit_produces_positive_temperature(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        probas = rng.dirichlet([1, 1], size=200)
        
        scaler = TemperatureScaler()
        scaler.fit(y_true, probas)
        assert scaler.fitted
        assert scaler.temperature > 0

    def test_transform_preserves_probability_sum(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        probas = rng.dirichlet([2, 1], size=100)
        
        scaler = TemperatureScaler()
        scaler.fit(y_true, probas)
        calibrated = scaler.transform(probas)
        
        assert calibrated.shape == probas.shape
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)

    def test_transform_produces_finite_values(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        probas = rng.dirichlet([1, 1], size=100)
        
        scaler = TemperatureScaler()
        scaler.fit(y_true, probas)
        calibrated = scaler.transform(probas)
        
        assert np.all(np.isfinite(calibrated))
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)

    def test_extreme_probabilities(self):
        """Test with near-zero and near-one probabilities."""
        y_true = np.array([0, 1, 0, 1] * 25)
        probas = np.array([[0.999, 0.001], [0.001, 0.999]] * 50)
        
        scaler = TemperatureScaler()
        scaler.fit(y_true, probas)
        calibrated = scaler.transform(probas)
        
        assert np.all(np.isfinite(calibrated))
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)

    def test_unfitted_raises(self):
        scaler = TemperatureScaler()
        probas = np.array([[0.5, 0.5]])
        with pytest.raises(RuntimeError):
            scaler.transform(probas)

    def test_fit_transform(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        probas = rng.dirichlet([1, 1], size=100)
        
        scaler = TemperatureScaler()
        calibrated = scaler.fit_transform(y_true, probas)
        
        assert scaler.fitted
        assert calibrated.shape == probas.shape
