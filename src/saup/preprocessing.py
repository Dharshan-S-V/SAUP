"""Preprocessing pipelines for SAUP datasets (Adult, Bank Marketing, Wine Quality)."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from saup.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

logger = logging.getLogger(__name__)


def build_preprocessor(
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale: bool = True,
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for a dataset.
    
    Args:
        numerical_features: List of numerical column names. Defaults to Adult's.
        categorical_features: List of categorical column names. Defaults to Adult's.
        scale: Whether to apply StandardScaler to numerical features.
        
    Returns:
        Unfitted ColumnTransformer.
    """
    num_cols = numerical_features if numerical_features is not None else NUMERICAL_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    
    transformers = []
    
    # Numerical pipeline
    if len(num_cols) > 0:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scaler", StandardScaler()))
        numerical_pipeline = Pipeline(num_steps)
        transformers.append(("num", numerical_pipeline, num_cols))
    
    # Categorical pipeline
    if len(cat_cols) > 0:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop=None,
            )),
        ])
        transformers.append(("cat", categorical_pipeline, cat_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    
    return preprocessor


def prepare_features_target(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Split DataFrame into features and target.
    
    Args:
        df: DataFrame with all columns including target.
        target_col: Name of target column. Defaults to Adult's 'income'.
        numerical_features: List of numerical features. Defaults to Adult's.
        categorical_features: List of categorical features. Defaults to Adult's.
        
    Returns:
        Tuple of (features_df, target_array).
    """
    target = target_col if target_col is not None else TARGET
    num_cols = numerical_features if numerical_features is not None else NUMERICAL_FEATURES
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    
    feature_cols = [c for c in (num_cols + cat_cols) if c in df.columns]
    X = df[feature_cols].copy()
    y = df[target].values.astype(int)
    return X, y


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
) -> ColumnTransformer:
    """Fit the preprocessor on training data only.
    
    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training features DataFrame.
        
    Returns:
        Fitted ColumnTransformer.
    """
    preprocessor.fit(X_train)
    n_features = preprocessor.transform(X_train[:1]).shape[1]
    logger.info(
        "Preprocessor fitted. Input features: %d, Output features: %d",
        X_train.shape[1], n_features
    )
    return preprocessor


def transform_data(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Transform features using a fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        X: Features DataFrame.
        
    Returns:
        Transformed numpy array.
    """
    X_transformed = preprocessor.transform(X)
    
    # Validate output
    if not np.all(np.isfinite(X_transformed)):
        n_invalid = np.sum(~np.isfinite(X_transformed))
        logger.warning(
            "Transform produced %d non-finite values, replacing with 0",
            n_invalid
        )
        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_transformed


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Get feature names from a fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        
    Returns:
        List of feature name strings.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except (AttributeError, Exception):
        return [f"feature_{i}" for i in range(preprocessor.transform(
            pd.DataFrame(columns=NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        ).shape[1])]
