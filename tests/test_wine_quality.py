"""Unit tests for UCI Wine Quality data loader, thresholding, and splits."""

import numpy as np
import pandas as pd
import pytest

from saup.data_loader import (
    WINE_NUMERICAL_FEATURES,
    WINE_TARGET,
    clean_wine_dataset,
    get_dataset_spec,
    load_and_clean_wine,
    load_wine_data,
)


@pytest.fixture
def sample_wine_df():
    """Create synthetic raw Wine Quality DataFrame matching official format."""
    return pd.DataFrame({
        "fixed acidity": [7.4, 7.8, 7.8, 11.2, 7.4],
        "volatile acidity": [0.70, 0.88, 0.76, 0.28, 0.66],
        "citric acid": [0.00, 0.00, 0.04, 0.56, 0.00],
        "residual sugar": [1.9, 2.6, 2.3, 1.9, 1.8],
        "chlorides": [0.076, 0.098, 0.092, 0.075, 0.075],
        "free sulfur dioxide": [11.0, 25.0, 15.0, 17.0, 13.0],
        "total sulfur dioxide": [34.0, 67.0, 54.0, 60.0, 40.0],
        "density": [0.9978, 0.9968, 0.9970, 0.9980, 0.9978],
        "pH": [3.51, 3.20, 3.26, 3.16, 3.51],
        "sulphates": [0.56, 0.68, 0.65, 0.58, 0.56],
        "alcohol": [9.4, 9.8, 9.8, 10.5, 9.4],
        "quality": [5, 5, 7, 8, 6],
    })


def test_clean_wine_threshold(sample_wine_df):
    """Verify quality >= 7 mapping to binary target."""
    cleaned = clean_wine_dataset(sample_wine_df, threshold=7)
    assert WINE_TARGET in cleaned.columns
    # original quality: [5, 5, 7, 8, 6] -> quality >= 7: [0, 0, 1, 1, 0]
    expected = [0, 0, 1, 1, 0]
    assert list(cleaned[WINE_TARGET]) == expected
    assert set(cleaned[WINE_TARGET].unique()) == {0, 1}


def test_clean_wine_custom_threshold(sample_wine_df):
    """Verify custom threshold functionality."""
    cleaned_t6 = clean_wine_dataset(sample_wine_df, threshold=6)
    # quality >= 6: [0, 0, 1, 1, 1]
    assert list(cleaned_t6[WINE_TARGET]) == [0, 0, 1, 1, 1]


def test_wine_features_preserved(sample_wine_df):
    """Verify all 11 physicochemical features are present and numeric."""
    cleaned = clean_wine_dataset(sample_wine_df)
    for col in WINE_NUMERICAL_FEATURES:
        assert col in cleaned.columns
        assert np.issubdtype(cleaned[col].dtype, np.number)


def test_wine_spec():
    """Verify Wine Quality dataset specification."""
    spec = get_dataset_spec("wine")
    assert spec.target_col == "quality_binary"
    assert len(spec.numerical_features) == 11
    assert "alcohol" in spec.covariate_cols
    assert "volatile acidity" in spec.covariate_cols


def test_load_wine_data_variants():
    """Verify loading red alone, white alone, and combined."""
    df_red = load_wine_data("data/raw/wine", variant="red")
    assert len(df_red) == 1599
    
    df_white = load_wine_data("data/raw/wine", variant="white")
    assert len(df_white) == 4898
    
    df_both = load_wine_data("data/raw/wine", variant="both")
    assert len(df_both) == 1599 + 4898
    assert "wine_type" in df_both.columns
    assert set(df_both["wine_type"].unique()) == {"red", "white"}


def test_load_and_clean_wine_splits():
    """Verify wine train/test splitting and stratification."""
    dev_df, test_df, summary = load_and_clean_wine(
        "data/raw/wine", variant="both", threshold=7, test_ratio=0.20, seed=42
    )
    assert len(dev_df) > 0
    assert len(test_df) > 0
    assert len(dev_df) + len(test_df) == 6497
    
    p_dev = dev_df[WINE_TARGET].mean()
    p_test = test_df[WINE_TARGET].mean()
    assert abs(p_dev - p_test) < 0.01
    assert summary["threshold"] == 7
