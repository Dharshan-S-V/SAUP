"""Tests for metrics module."""

import numpy as np
import pytest
from saup.metrics import (
    classification_metrics,
    calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    conformal_metrics,
    selective_prediction_metrics,
    risk_coverage_curve,
    area_under_risk_coverage,
    ks_test,
    population_stability_index,
    categorical_psi,
    maximum_mean_discrepancy,
    effective_sample_size,
    reliability_diagram_data,
)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        m = classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0])
        m = classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_with_probabilities(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        m = classification_metrics(y_true, y_pred, y_prob)
        assert "roc_auc" in m
        assert 0 <= m["roc_auc"] <= 1

    def test_single_class(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        m = classification_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0


class TestCalibrationMetrics:
    def test_perfectly_calibrated(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        m = calibration_metrics(y_true, y_prob)
        assert m["brier_score"] < 0.01
        
    def test_ece_range(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        prob_pos = rng.uniform(0, 1, 100)
        ece = expected_calibration_error(y_true, prob_pos)
        assert 0 <= ece <= 1

    def test_mce_range(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        prob_pos = rng.uniform(0, 1, 100)
        mce = maximum_calibration_error(y_true, prob_pos)
        assert 0 <= mce <= 1

    def test_empty_input(self):
        ece = expected_calibration_error(np.array([]), np.array([]))
        assert ece == 0.0

    def test_reliability_diagram_data(self):
        y_true = np.array([0, 1, 0, 1])
        prob_pos = np.array([0.1, 0.9, 0.3, 0.7])
        centers, accs, counts = reliability_diagram_data(y_true, prob_pos)
        assert len(centers) == 15
        assert len(accs) == 15


class TestConformalMetrics:
    def test_full_coverage(self):
        y_true = np.array([0, 1, 0, 1])
        pred_sets = [[0, 1], [0, 1], [0, 1], [0, 1]]
        m = conformal_metrics(y_true, pred_sets)
        assert m["empirical_coverage"] == 1.0
        assert m["avg_set_size"] == 2.0

    def test_no_coverage(self):
        y_true = np.array([0, 1])
        pred_sets = [[1], [0]]
        m = conformal_metrics(y_true, pred_sets)
        assert m["empirical_coverage"] == 0.0

    def test_singleton_sets(self):
        y_true = np.array([0, 1, 0])
        pred_sets = [[0], [1], [0]]
        m = conformal_metrics(y_true, pred_sets)
        assert m["empirical_coverage"] == 1.0
        assert m["avg_set_size"] == 1.0

    def test_empty_sets(self):
        y_true = np.array([0, 1])
        pred_sets = [[], []]
        m = conformal_metrics(y_true, pred_sets)
        assert m["empty_set_rate"] == 1.0
        assert m["empirical_coverage"] == 0.0


class TestSelectivePrediction:
    def test_all_accepted(self):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1, 0])
        accepted = np.array([True, True, True])
        m = selective_prediction_metrics(y_true, y_pred, accepted)
        assert m["coverage"] == 1.0
        assert m["selective_accuracy"] == 1.0

    def test_none_accepted(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        accepted = np.array([False, False])
        m = selective_prediction_metrics(y_true, y_pred, accepted)
        assert m["abstention_rate"] == 1.0

    def test_risk_coverage(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 50)
        y_pred = rng.randint(0, 2, 50)
        conf = rng.uniform(0, 1, 50)
        coverages, risks = risk_coverage_curve(y_true, y_pred, conf)
        assert len(coverages) == 100
        aurc = area_under_risk_coverage(coverages, risks)
        assert 0 <= aurc <= 1


class TestShiftMetrics:
    def test_ks_identical(self):
        data = np.random.RandomState(42).normal(0, 1, 100)
        stat, pval = ks_test(data, data)
        assert stat == 0.0
        assert pval == 1.0

    def test_ks_different(self):
        rng = np.random.RandomState(42)
        source = rng.normal(0, 1, 500)
        target = rng.normal(3, 1, 500)
        stat, pval = ks_test(source, target)
        assert stat > 0.5
        assert pval < 0.01

    def test_psi_identical(self):
        data = np.random.RandomState(42).normal(0, 1, 500)
        psi = population_stability_index(data, data)
        assert psi < 0.05

    def test_psi_no_division_by_zero(self):
        source = np.array([1, 1, 1, 2, 2, 3])
        target = np.array([4, 4, 4, 5, 5, 6])
        psi = population_stability_index(source, target)
        assert np.isfinite(psi)

    def test_categorical_psi(self):
        source = np.array(["a", "a", "b", "c"])
        target = np.array(["a", "b", "b", "c"])
        psi = categorical_psi(source, target)
        assert np.isfinite(psi)

    def test_mmd_identical(self):
        rng = np.random.RandomState(42)
        data = rng.normal(0, 1, (100, 3))
        mmd = maximum_mean_discrepancy(data, data)
        assert mmd < 0.1

    def test_mmd_different(self):
        rng = np.random.RandomState(42)
        source = rng.normal(0, 1, (200, 3))
        target = rng.normal(5, 1, (200, 3))
        mmd = maximum_mean_discrepancy(source, target)
        assert mmd > 0.1

    def test_mmd_1d(self):
        source = np.random.RandomState(42).normal(0, 1, 100)
        target = np.random.RandomState(42).normal(2, 1, 100)
        mmd = maximum_mean_discrepancy(source, target)
        assert np.isfinite(mmd)

    def test_effective_sample_size(self):
        weights = np.ones(100)
        ess = effective_sample_size(weights)
        assert abs(ess - 100) < 0.01

    def test_ess_single_weight(self):
        weights = np.zeros(100)
        weights[0] = 1.0
        ess = effective_sample_size(weights)
        assert abs(ess - 1.0) < 0.01

    def test_ess_zero_weights(self):
        weights = np.zeros(10)
        ess = effective_sample_size(weights)
        assert ess == 0.0
