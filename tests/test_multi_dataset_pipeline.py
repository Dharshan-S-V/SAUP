"""Unit and integration tests for multi-dataset preprocessing, shift generation, and shift detection."""

import numpy as np
import pandas as pd
import pytest

from saup.data_loader import (
    get_dataset_spec,
    load_and_clean_bank,
    load_and_clean_wine,
)
from saup.preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    prepare_features_target,
    transform_data,
)
from saup.shift_generation import generate_shift
from saup.shift_detection import ShiftDetector


def test_bank_preprocessing_pipeline():
    """Test preprocessing on Bank Marketing data without target leakage."""
    dev_df, test_df, _ = load_and_clean_bank("data/raw/bank", test_ratio=0.2, seed=42)
    spec = get_dataset_spec("bank")
    
    # Preprocessor
    preprocessor = build_preprocessor(
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
        scale=True,
    )
    
    X_dev, y_dev = prepare_features_target(
        dev_df,
        target_col=spec.target_col,
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
    )
    
    fit_preprocessor(preprocessor, X_dev[:500])
    X_trans = transform_data(preprocessor, X_dev[:100])
    
    assert X_trans.shape[0] == 100
    assert X_trans.shape[1] > len(spec.numerical_features)  # One-hot encoded
    assert np.all(np.isfinite(X_trans))


def test_wine_preprocessing_pipeline():
    """Test preprocessing on Wine Quality data."""
    dev_df, test_df, _ = load_and_clean_wine("data/raw/wine", variant="both", threshold=7, test_ratio=0.2, seed=42)
    spec = get_dataset_spec("wine")
    
    preprocessor = build_preprocessor(
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
        scale=True,
    )
    
    X_dev, y_dev = prepare_features_target(
        dev_df,
        target_col=spec.target_col,
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
    )
    
    fit_preprocessor(preprocessor, X_dev[:500])
    X_trans = transform_data(preprocessor, X_dev[:100])
    
    assert X_trans.shape[0] == 100
    assert np.all(np.isfinite(X_trans))


def test_shift_generation_on_bank():
    """Test covariate, label, mixed, and severe shift generation on Bank Marketing."""
    dev_df, _, _ = load_and_clean_bank("data/raw/bank", test_ratio=0.2, seed=42)
    spec = get_dataset_spec("bank")
    subset = dev_df.iloc[:1000].copy()
    
    for scenario in ["source", "covariate", "label", "mixed", "severe"]:
        shifted_df, meta = generate_shift(
            subset,
            scenario=scenario,
            severity="moderate",
            seed=42,
            target_col=spec.target_col,
            covariate_cols=spec.covariate_cols,
            perturb_cols=spec.perturb_cols,
        )
        assert len(shifted_df) > 0
        assert spec.target_col in shifted_df.columns
        assert set(shifted_df[spec.target_col].unique()).issubset({0, 1})
        assert meta["scenario"] in (scenario, f"{scenario}_shift")


def test_shift_generation_on_wine():
    """Test shift generation on Wine Quality."""
    dev_df, _, _ = load_and_clean_wine("data/raw/wine", variant="both", threshold=7, test_ratio=0.2, seed=42)
    spec = get_dataset_spec("wine")
    subset = dev_df.iloc[:1000].copy()
    
    for scenario in ["covariate", "label", "severe"]:
        shifted_df, meta = generate_shift(
            subset,
            scenario=scenario,
            severity="moderate",
            seed=42,
            target_col=spec.target_col,
            covariate_cols=spec.covariate_cols,
            perturb_cols=spec.perturb_cols,
        )
        assert len(shifted_df) > 0
        assert spec.target_col in shifted_df.columns
        assert set(shifted_df[spec.target_col].unique()).issubset({0, 1})


def test_shift_detection_on_bank():
    """Test ShiftDetector on Bank Marketing feature data."""
    dev_df, _, _ = load_and_clean_bank("data/raw/bank", test_ratio=0.2, seed=42)
    spec = get_dataset_spec("bank")
    
    features = spec.numerical_features + spec.categorical_features
    X_source = dev_df[features].iloc[:800]
    X_target = dev_df[features].iloc[800:1200]
    
    detector = ShiftDetector(
        numerical_features=spec.numerical_features,
        categorical_features=spec.categorical_features,
        mmd_subsample=500,
    )
    detector.fit(X_source, n_mmd_bootstrap=3)
    
    report = detector.detect(X_target)
    assert "overall" in report
    assert "shift_detected" in report["overall"]
    assert "mmd" in report
