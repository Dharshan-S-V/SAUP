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
        assert np.isfinite(report["mmd"]["value"])

    def test_unfitted_raises(self):
        detector = ShiftDetector()
        with pytest.raises(RuntimeError):
            detector.detect(_make_source_df(100))

    def test_feature_scale_disparity_detected(self):
        """MMD detects shift on small-scale feature even when large-scale feature is present."""
        rng = np.random.RandomState(42)
        n = 300
        # Feature 1 has scale 100,000; Feature 2 has scale 10
        source_df = pd.DataFrame({
            "large_feature": rng.normal(200000, 50000, n),
            "small_feature": rng.normal(30, 5, n),
        })
        
        detector = ShiftDetector(numerical_features=["large_feature", "small_feature"], categorical_features=[])
        detector.fit(source_df)
        
        # Target: large_feature is UNCHANGED, small_feature has severe shift (+25)
        target_df = pd.DataFrame({
            "large_feature": rng.normal(200000, 50000, n),
            "small_feature": rng.normal(55, 5, n),
        })
        
        report = detector.detect(target_df)
        assert np.isfinite(report["mmd"]["value"])
        assert report["mmd"]["value"] > report["mmd"]["threshold"]
        assert report["mmd"]["detected"]

    def test_duplicate_records_handled(self):
        """Detector handles duplicated records without zero-division or crash."""
        df = pd.DataFrame({
            "col1": [1.0, 1.0, 1.0, 2.0, 2.0] * 20,
            "col2": [5.0, 5.0, 5.0, 5.0, 5.0] * 20,
        })
        detector = ShiftDetector(numerical_features=["col1", "col2"], categorical_features=[])
        detector.fit(df)
        report = detector.detect(df)
        assert np.isfinite(report["mmd"]["value"])
        assert np.isfinite(report["mmd"]["threshold"])

    def test_small_sample_size(self):
        """Detector operates reliably with small sample sizes."""
        source_df = pd.DataFrame({"col1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        target_df = pd.DataFrame({"col1": [10.0, 20.0, 30.0]})
        detector = ShiftDetector(numerical_features=["col1"], categorical_features=[])
        detector.fit(source_df)
        report = detector.detect(target_df)
        assert np.isfinite(report["mmd"]["value"])

    def test_deterministic_and_reproducible(self):
        """Repeated detections on identical data produce identical unrounded MMD."""
        source_df = _make_source_df(200, seed=1)
        target_df = _make_source_df(200, seed=2)
        detector = ShiftDetector()
        detector.fit(source_df)
        
        report1 = detector.detect(target_df)
        report2 = detector.detect(target_df)
        assert report1["mmd"]["value"] == report2["mmd"]["value"]
        assert report1["mmd"]["threshold"] == report2["mmd"]["threshold"]
