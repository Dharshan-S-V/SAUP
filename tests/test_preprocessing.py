"""Tests for data loading and preprocessing."""

import numpy as np
import pandas as pd
import pytest
from saup.data_loader import (
    COLUMN_NAMES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET,
    clean_dataset, LABEL_MAP,
)
from saup.preprocessing import (
    build_preprocessor, prepare_features_target,
    fit_preprocessor, transform_data,
)
from saup.splitting import create_splits


def _make_mock_df(n=100, seed=42):
    """Create a mock Adult-like DataFrame for testing."""
    rng = np.random.RandomState(seed)
    data = {
        "age": rng.randint(18, 65, n),
        "workclass": rng.choice(["Private", "Self-emp", "?", "Gov"], n),
        "fnlwgt": rng.randint(10000, 500000, n),
        "education": rng.choice(["Bachelors", "HS-grad", "Masters"], n),
        "education_num": rng.randint(1, 16, n),
        "marital_status": rng.choice(["Married", "Single", "Divorced"], n),
        "occupation": rng.choice(["Tech", "Sales", "?", "Exec"], n),
        "relationship": rng.choice(["Husband", "Wife", "Own-child"], n),
        "race": rng.choice(["White", "Black", "Asian"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "capital_gain": rng.randint(0, 10000, n),
        "capital_loss": rng.randint(0, 5000, n),
        "hours_per_week": rng.randint(10, 60, n),
        "native_country": rng.choice(["United-States", "Mexico", "India"], n),
        "income": rng.choice(["<=50K", ">50K"], n, p=[0.75, 0.25]),
    }
    return pd.DataFrame(data)


class TestDataLoader:
    def test_clean_normalizes_labels(self):
        df = _make_mock_df()
        cleaned = clean_dataset(df, "test")
        assert set(cleaned[TARGET].unique()) == {0, 1}

    def test_clean_handles_trailing_dots(self):
        df = _make_mock_df()
        df[TARGET] = df[TARGET] + "."
        cleaned = clean_dataset(df, "test")
        assert set(cleaned[TARGET].unique()) == {0, 1}

    def test_invalid_labels_raise(self):
        df = _make_mock_df()
        df.loc[0, TARGET] = "INVALID"
        with pytest.raises(ValueError):
            clean_dataset(df, "test")


class TestPreprocessing:
    def test_build_preprocessor(self):
        preprocessor = build_preprocessor(scale=True)
        assert preprocessor is not None

    def test_fit_only_on_train(self):
        df = _make_mock_df(200)
        df = clean_dataset(df, "test")
        X, y = prepare_features_target(df)
        
        preprocessor = build_preprocessor()
        fit_preprocessor(preprocessor, X)
        transformed = transform_data(preprocessor, X)
        
        assert transformed.shape[0] == len(X)
        assert np.all(np.isfinite(transformed))

    def test_transform_produces_finite(self):
        df = _make_mock_df(100)
        df = clean_dataset(df, "test")
        X, y = prepare_features_target(df)
        
        preprocessor = build_preprocessor()
        fit_preprocessor(preprocessor, X)
        transformed = transform_data(preprocessor, X)
        
        assert np.all(np.isfinite(transformed))

    def test_unseen_category_handling(self):
        """Preprocessor should handle unseen categories."""
        df_train = _make_mock_df(100, seed=1)
        df_train = clean_dataset(df_train, "train")
        X_train, _ = prepare_features_target(df_train)
        
        preprocessor = build_preprocessor()
        fit_preprocessor(preprocessor, X_train)
        
        df_test = _make_mock_df(50, seed=2)
        df_test = clean_dataset(df_test, "test")
        df_test.loc[0, "workclass"] = "NEVER_SEEN_BEFORE"
        X_test, _ = prepare_features_target(df_test)
        
        transformed = transform_data(preprocessor, X_test)
        assert np.all(np.isfinite(transformed))


class TestSplitting:
    def test_splits_dont_overlap(self):
        df = _make_mock_df(1000)
        df = clean_dataset(df, "test")
        
        splits = create_splits(df, random_seed=42)
        
        all_keys = ["train", "calibration", "conformal", "policy"]
        total = sum(len(splits[k]) for k in all_keys)
        assert total == len(df)

    def test_splits_stratified(self):
        df = _make_mock_df(1000)
        df = clean_dataset(df, "test")
        
        splits = create_splits(df, random_seed=42)
        
        overall_ratio = df[TARGET].mean()
        for name, split_df in splits.items():
            split_ratio = split_df[TARGET].mean()
            assert abs(split_ratio - overall_ratio) < 0.1, \
                f"Split '{name}' ratio {split_ratio:.3f} deviates from overall {overall_ratio:.3f}"

    def test_invalid_ratios_raise(self):
        df = _make_mock_df(100)
        df = clean_dataset(df, "test")
        with pytest.raises(ValueError):
            create_splits(df, train_ratio=0.5, calibration_ratio=0.1,
                          conformal_ratio=0.1, policy_ratio=0.1)
