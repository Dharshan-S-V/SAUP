"""Tests for shift detection module."""

import numpy as np
import pandas as pd
import pytest
from saup.shift_detection import ShiftDetector
from saup.data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def _make_source_df(n=500, seed=42):
    """Create a synthetic source DataFrame mimicking Adult features."""
    rng = np.random.RandomState(seed)
    data = {}
    for col in NUMERICAL_FEATURES:
        data[col] = rng.normal(50, 10, n)
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["A", "B", "C"], n)
    return pd.DataFrame(data)


class TestShiftDetector:
    def test_fit(self):
        df = _make_source_df()
        detector = ShiftDetector()
        detector.fit(df)
        assert detector.fitted
        assert detector.mmd_threshold > 0

    def test_no_shift_detected(self):
        df = _make_source_df(500)
        detector = ShiftDetector()
        detector.fit(df)
        
        # Same distribution
        target = _make_source_df(500, seed=43)
        report = detector.detect(target)
        
        assert "overall" in report
        # With same distribution, severity should be low
        assert report["overall"]["severity_score"] < 0.8

    def test_shift_detected(self):
        source = _make_source_df(500)
        detector = ShiftDetector()
        detector.fit(source)
        
        # Different distribution
        rng = np.random.RandomState(99)
        target_data = {}
        for col in NUMERICAL_FEATURES:
            target_data[col] = rng.normal(100, 20, 500)  # Very different
        for col in CATEGORICAL_FEATURES:
            target_data[col] = rng.choice(["A", "B", "C"], 500)
        target = pd.DataFrame(target_data)
        
        report = detector.detect(target)
        assert report["overall"]["shift_detected"]
        assert report["overall"]["severity_score"] > 0

    def test_constant_column(self):
        """Detector handles constant-valued columns."""
        df = _make_source_df()
        df["age"] = 30  # constant
        
        detector = ShiftDetector()
        detector.fit(df)
        
        target = _make_source_df(200, seed=99)
        target["age"] = 30
        report = detector.detect(target)
        assert "ks_results" in report

    def test_unfitted_raises(self):
        detector = ShiftDetector()
        with pytest.raises(RuntimeError):
            detector.detect(_make_source_df(100))
