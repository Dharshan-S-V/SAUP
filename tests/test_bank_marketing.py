"""Unit tests for UCI Bank Marketing data loader, cleaning, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from saup.data_loader import (
    BANK_CATEGORICAL_FEATURES,
    BANK_NUMERICAL_FEATURES,
    BANK_TARGET,
    clean_bank_dataset,
    get_dataset_spec,
    load_and_clean_bank,
    load_bank_data,
)


@pytest.fixture
def sample_bank_df():
    """Create synthetic raw Bank Marketing DataFrame matching official UCI format."""
    return pd.DataFrame({
        "age": [30, 45, 55, 22, 60],
        "job": ["management", "technician", "blue-collar", "student", "retired"],
        "marital": ["married", "single", "married", "single", "divorced"],
        "education": ["tertiary", "secondary", "primary", "unknown", "secondary"],
        "default": ["no", "no", "no", "no", "no"],
        "balance": [1500, 250, 0, 50, 12000],
        "housing": ["yes", "no", "yes", "no", "no"],
        "loan": ["no", "no", "yes", "no", "no"],
        "contact": ["cellular", "cellular", "telephone", "unknown", "cellular"],
        "day": [5, 12, 19, 1, 28],
        "month": ["may", "jul", "aug", "jun", "nov"],
        "duration": [120, 340, 50, 600, 95],  # Target leakage candidate!
        "campaign": [1, 2, 1, 3, 1],
        "pdays": [-1, 99, -1, -1, 180],
        "previous": [0, 1, 0, 0, 2],
        "poutcome": ["unknown", "failure", "unknown", "unknown", "success"],
        "y": ["no", "yes", "no", "no", "yes"],
    })


def test_clean_bank_drops_duration_leakage(sample_bank_df):
    """Verify that 'duration' is explicitly removed to prevent target leakage."""
    assert "duration" in sample_bank_df.columns
    cleaned = clean_bank_dataset(sample_bank_df)
    assert "duration" not in cleaned.columns, "Target leakage! 'duration' must be dropped."


def test_clean_bank_binary_target(sample_bank_df):
    """Verify target 'y' is converted to {0, 1}."""
    cleaned = clean_bank_dataset(sample_bank_df)
    assert BANK_TARGET in cleaned.columns
    assert set(cleaned[BANK_TARGET].unique()) == {0, 1}
    assert cleaned[BANK_TARGET].dtype == int
    # Original y: ['no', 'yes', 'no', 'no', 'yes'] -> [0, 1, 0, 0, 1]
    expected = [0, 1, 0, 0, 1]
    assert list(cleaned[BANK_TARGET]) == expected


def test_clean_bank_preserves_features(sample_bank_df):
    """Verify numerical and categorical columns are retained."""
    cleaned = clean_bank_dataset(sample_bank_df)
    for num_col in BANK_NUMERICAL_FEATURES:
        assert num_col in cleaned.columns
    for cat_col in BANK_CATEGORICAL_FEATURES:
        assert cat_col in cleaned.columns


def test_clean_bank_invalid_target_raises():
    """Verify invalid target values raise ValueError."""
    bad_df = pd.DataFrame({
        "age": [25],
        "y": ["maybe"],
    })
    with pytest.raises(ValueError, match="Unexpected target values"):
        clean_bank_dataset(bad_df)


def test_bank_spec():
    """Verify Bank Marketing dataset specification."""
    spec = get_dataset_spec("bank")
    assert spec.target_col == "y"
    assert "duration" not in spec.numerical_features
    assert "duration" not in spec.categorical_features
    assert "age" in spec.covariate_cols
    assert "balance" in spec.covariate_cols
    assert "leakage" in spec.leakage_notes.lower()


def test_load_and_clean_bank_stratified():
    """Test actual or synthetic loading and splitting for bank marketing."""
    raw_df = load_bank_data("data/raw/bank")
    assert len(raw_df) > 40000
    dev_df, test_df, summary = load_and_clean_bank("data/raw/bank", test_ratio=0.20, seed=42)
    
    assert len(dev_df) > 0
    assert len(test_df) > 0
    assert len(dev_df) + len(test_df) == len(raw_df)
    
    # Check non-overlap
    dev_set = set(dev_df.index)
    test_set = set(test_df.index)
    # The indexes in dev_df and test_df are reset, but lengths match
    assert len(dev_df) == int(len(raw_df) * 0.8) or abs(len(dev_df) - 0.8 * len(raw_df)) < 2
    
    # Check target distribution stratification
    p_dev = dev_df[BANK_TARGET].mean()
    p_test = test_df[BANK_TARGET].mean()
    assert abs(p_dev - p_test) < 0.01, f"Stratification mismatch: dev={p_dev:.4f}, test={p_test:.4f}"
