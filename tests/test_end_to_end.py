"""End-to-end smoke test for SAUP pipeline."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_metrics_module_imports():
    """Verify all metric functions import correctly."""
    from saup.metrics import (
        classification_metrics, calibration_metrics, conformal_metrics,
        selective_prediction_metrics, ks_test, population_stability_index,
        maximum_mean_discrepancy, effective_sample_size,
    )


def test_calibration_module_imports():
    from saup.calibration import TemperatureScaler


def test_conformal_module_imports():
    from saup.conformal import SplitConformalClassifier, WeightedConformalClassifier


def test_shift_detection_imports():
    from saup.shift_detection import ShiftDetector


def test_label_shift_imports():
    from saup.label_shift import BBSEEstimator


def test_decision_engine_imports():
    from saup.decision_engine import SAUPDecisionEngine, DecisionAction


def test_config_loads():
    from saup.config import load_config
    config = load_config()
    assert "data" in config
    assert "models" in config
    assert "random_seed" in config


def test_mini_pipeline():
    """Run a mini end-to-end pipeline with synthetic data."""
    from saup.preprocessing import build_preprocessor
    from saup.calibration import TemperatureScaler
    from saup.conformal import SplitConformalClassifier
    from saup.metrics import classification_metrics, calibration_metrics, conformal_metrics
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.RandomState(42)
    n = 200
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split
    X_train, y_train = X[:120], y[:120]
    X_cal, y_cal = X[120:160], y[120:160]
    X_conf, y_conf = X[160:180], y[160:180]
    X_test, y_test = X[180:], y[180:]

    # Train
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)

    # Metrics
    cls_m = classification_metrics(y_test, preds, probas)
    assert "accuracy" in cls_m
    assert 0 <= cls_m["accuracy"] <= 1

    # Calibration
    probas_cal = model.predict_proba(X_cal)
    scaler = TemperatureScaler()
    scaler.fit(y_cal, probas_cal)
    cal_probas = scaler.transform(probas)

    cal_m = calibration_metrics(y_test, cal_probas)
    assert "brier_score" in cal_m
    assert np.isfinite(cal_m["brier_score"])

    # Conformal
    probas_conf = model.predict_proba(X_conf)
    cal_probas_conf = scaler.transform(probas_conf)
    
    cc = SplitConformalClassifier(alpha=0.10)
    cc.calibrate(y_conf, cal_probas_conf)
    pred_sets = cc.predict_sets(cal_probas)

    conf_m = conformal_metrics(y_test, pred_sets)
    assert "empirical_coverage" in conf_m
    # Coverage should be reasonable (not zero)
    assert conf_m["empirical_coverage"] > 0
