"""Data loading, cleaning, and metadata generation for SAUP datasets:
UCI Adult, UCI Bank Marketing, and UCI Wine Quality.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from saup.config import resolve_path

logger = logging.getLogger(__name__)

# =====================================================================
# UCI Adult Definitions (Preserved exactly)
# =====================================================================
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

NUMERICAL_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week"
]

CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

TARGET = "income"
LABEL_MAP = {"<=50K": 0, ">50K": 1}

# =====================================================================
# UCI Bank Marketing Definitions
# =====================================================================
# Official UCI Bank Marketing: binary classification of term deposit subscription
# Attribute 'duration' is explicitly excluded to prevent post-outcome target leakage!
BANK_RAW_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y"
]

BANK_NUMERICAL_FEATURES = [
    "age", "balance", "day", "campaign", "pdays", "previous"
]

BANK_CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome"
]

BANK_TARGET = "y"
BANK_LABEL_MAP = {"no": 0, "yes": 1}

# =====================================================================
# UCI Wine Quality Definitions
# =====================================================================
WINE_NUMERICAL_FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

# When combined, wine_type distinguishes red and white
WINE_COMBINED_CATEGORICAL_FEATURES = ["wine_type"]

WINE_RAW_TARGET = "quality"
WINE_TARGET = "quality_binary"
WINE_QUALITY_THRESHOLD = 7  # quality >= 7 is high quality (1), < 7 is standard/low (0)


# =====================================================================
# Dataset Specification Class & Registry
# =====================================================================
@dataclass
class DatasetSpec:
    """Specification describing dataset features, target, and shift attributes."""
    name: str
    target_col: str
    numerical_features: List[str]
    categorical_features: List[str]
    covariate_cols: List[str]
    perturb_cols: List[str]
    leakage_notes: str
    class_names: Dict[int, str] = field(default_factory=dict)


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        name="adult",
        target_col=TARGET,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        covariate_cols=["age", "hours_per_week"],
        perturb_cols=["age", "hours_per_week", "capital_gain", "capital_loss"],
        leakage_notes="Census survey data; demographic attributes recorded prior to income evaluation.",
        class_names={0: "<=50K", 1: ">50K"},
    ),
    "bank": DatasetSpec(
        name="bank_marketing",
        target_col=BANK_TARGET,
        numerical_features=BANK_NUMERICAL_FEATURES,
        categorical_features=BANK_CATEGORICAL_FEATURES,
        covariate_cols=["age", "balance"],
        perturb_cols=["age", "balance", "campaign", "previous"],
        leakage_notes="Call duration dropped to prevent target leakage (call duration is unknown prior to call and post-outcome).",
        class_names={0: "no", 1: "yes"},
    ),
    "wine": DatasetSpec(
        name="wine_quality",
        target_col=WINE_TARGET,
        numerical_features=WINE_NUMERICAL_FEATURES,
        categorical_features=WINE_COMBINED_CATEGORICAL_FEATURES,
        covariate_cols=["alcohol", "volatile acidity"],
        perturb_cols=["alcohol", "volatile acidity", "residual sugar", "density", "pH"],
        leakage_notes="Laboratory physicochemical tests available before sensory evaluation. Threshold quality>=7 marks premium wines.",
        class_names={0: "lower/standard (<7)", 1: "high (>=7)"},
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    """Retrieve DatasetSpec by name."""
    key = dataset_name.lower().replace("_marketing", "").replace("_quality", "")
    if key in ("bank", "bank_marketing"):
        return DATASET_SPECS["bank"]
    elif key in ("wine", "wine_quality"):
        return DATASET_SPECS["wine"]
    elif key in ("adult", "uci_adult"):
        return DATASET_SPECS["adult"]
    raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'adult', 'bank', or 'wine'.")


# =====================================================================
# UCI Adult Loader & Cleaner (Preserved)
# =====================================================================
def load_adult_data(raw_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load adult.data and adult.test files."""
    raw_path = resolve_path(raw_dir)
    train_file = raw_path / "adult.data"
    test_file = raw_path / "adult.test"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_file}")
    
    train_df = pd.read_csv(
        train_file,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values="?",
    )
    
    test_df = pd.read_csv(
        test_file,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )
    
    logger.info("Loaded Adult train: %d rows, test: %d rows", len(train_df), len(test_df))
    return train_df, test_df


def clean_dataset(df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    """Clean a single Adult dataset DataFrame."""
    df = df.copy()
    
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].str.rstrip(".")
    
    _validate_labels(df, dataset_name)
    df[TARGET] = df[TARGET].map(LABEL_MAP)
    
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_invalid = df[col].isna().sum()
            if n_invalid > 0:
                logger.warning(
                    "%s: %d invalid numeric values in '%s' coerced to NaN",
                    dataset_name, n_invalid, col
                )
    
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        logger.info("%s missing values:\n%s", dataset_name, missing_cols.to_string())
    else:
        logger.info("%s: No missing values", dataset_name)
    
    n_duplicates = df.duplicated().sum()
    logger.info("%s: %d duplicate rows found (kept)", dataset_name, n_duplicates)
    
    expected_cols = set(COLUMN_NAMES)
    actual_cols = set(df.columns)
    if expected_cols != actual_cols:
        logger.warning(
            "%s: Column mismatch. Expected: %s, Got: %s",
            dataset_name, expected_cols - actual_cols, actual_cols - expected_cols
        )
    
    return df


def _validate_labels(df: pd.DataFrame, dataset_name: str) -> None:
    """Validate Adult target labels."""
    unique_labels = set(df[TARGET].dropna().unique())
    expected_labels = set(LABEL_MAP.keys())
    unexpected = unique_labels - expected_labels
    if unexpected:
        raise ValueError(
            f"{dataset_name}: Unexpected target values: {unexpected}. Expected: {expected_labels}"
        )


def get_preprocessing_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = TARGET,
    num_features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None,
) -> Dict:
    """Generate preprocessing summary statistics."""
    num_cols = num_features if num_features is not None else NUMERICAL_FEATURES
    cat_cols = cat_features if cat_features is not None else CATEGORICAL_FEATURES
    
    summary = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_columns": list(train_df.columns),
        "train_missing": train_df.isnull().sum().to_dict(),
        "test_missing": test_df.isnull().sum().to_dict(),
        "train_duplicates": int(train_df.duplicated().sum()),
        "test_duplicates": int(test_df.duplicated().sum()),
        "train_label_dist": train_df[target_col].value_counts().to_dict() if target_col in train_df else {},
        "test_label_dist": test_df[target_col].value_counts().to_dict() if target_col in test_df else {},
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }
    return summary


def load_and_clean(
    raw_dir: str = "data/raw",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load, clean, and return summary for the Adult dataset."""
    train_raw, test_raw = load_adult_data(raw_dir)
    train_clean = clean_dataset(train_raw, "train")
    test_clean = clean_dataset(test_raw, "test")
    summary = get_preprocessing_summary(train_clean, test_clean)
    
    logger.info(
        "Adult Dataset loaded: train=%d, test=%d",
        summary["train_rows"], summary["test_rows"]
    )
    return train_clean, test_clean, summary


# =====================================================================
# UCI Bank Marketing Loader & Cleaner
# =====================================================================
def load_bank_data(raw_dir: str = "data/raw/bank") -> pd.DataFrame:
    """Load the Bank Marketing full dataset.
    
    Args:
        raw_dir: Path to directory containing bank-full.csv.
        
    Returns:
        Raw DataFrame.
    """
    raw_path = resolve_path(raw_dir)
    csv_file = raw_path / "bank-full.csv"
    
    if not csv_file.exists():
        # Fallback to general data/raw
        fallback = resolve_path("data/raw") / "bank-full.csv"
        if fallback.exists():
            csv_file = fallback
        else:
            raise FileNotFoundError(f"Bank Marketing data not found: {csv_file}")
    
    # Read semicolon-delimited CSV
    df = pd.read_csv(csv_file, sep=";", skipinitialspace=True)
    logger.info("Loaded Bank Marketing raw data: %d rows, %d columns", len(df), df.shape[1])
    return df


def clean_bank_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Bank Marketing dataset, preventing target leakage.
    
    Critical step:
    'duration' is dropped to prevent target leakage (unavailable at prediction time).
    Target 'y' is converted from ('yes', 'no') to (1, 0).
    Categorical 'unknown' values are preserved for the imputer/encoder.
    
    Args:
        df: Raw Bank Marketing DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Target validation and mapping
    if BANK_TARGET not in df.columns:
        raise ValueError(f"Target column '{BANK_TARGET}' missing from Bank Marketing data")
    
    unique_y = set(df[BANK_TARGET].unique())
    expected_y = set(BANK_LABEL_MAP.keys())
    if not unique_y.issubset(expected_y):
        raise ValueError(f"Unexpected target values in Bank data: {unique_y - expected_y}")
    
    df[BANK_TARGET] = df[BANK_TARGET].map(BANK_LABEL_MAP).astype(int)
    
    # TARGET LEAKAGE PREVENTION: Drop 'duration'
    if "duration" in df.columns:
        logger.info("Dropping 'duration' column to prevent post-outcome target leakage.")
        df = df.drop(columns=["duration"])
    
    # Validate numerical columns
    for col in BANK_NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                logger.warning("Bank: %d invalid numerical values in '%s' coerced to NaN", n_nan, col)
    
    logger.info(
        "Bank Marketing cleaned: %d rows, y dist: %s",
        len(df), df[BANK_TARGET].value_counts().to_dict()
    )
    return df


def load_and_clean_bank(
    raw_dir: str = "data/raw/bank",
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load, clean, split into dev/test, and return summary for Bank Marketing.
    
    Splits into:
    - development set (1 - test_ratio): used for train/cal/conf/policy splits
    - held-out test set (test_ratio): strictly held-out evaluation set
    
    Args:
        raw_dir: Directory containing bank-full.csv.
        test_ratio: Fraction reserved for final held-out test set.
        seed: Random seed for reproducible splitting.
        
    Returns:
        Tuple of (dev_df, test_df, summary).
    """
    raw_df = load_bank_data(raw_dir)
    cleaned_df = clean_bank_dataset(raw_df)
    
    # Stratified held-out test split
    y = cleaned_df[BANK_TARGET].values
    dev_df, test_df = train_test_split(
        cleaned_df,
        test_size=test_ratio,
        stratify=y,
        random_state=seed,
    )
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    summary = get_preprocessing_summary(
        dev_df,
        test_df,
        target_col=BANK_TARGET,
        num_features=BANK_NUMERICAL_FEATURES,
        cat_features=BANK_CATEGORICAL_FEATURES,
    )
    summary["dataset"] = "bank_marketing"
    summary["leakage_prevention"] = "Attribute 'duration' was dropped."
    
    logger.info("Bank Marketing loaded: dev=%d, test=%d", len(dev_df), len(test_df))
    return dev_df, test_df, summary


# =====================================================================
# UCI Wine Quality Loader & Cleaner
# =====================================================================
def load_wine_data(
    raw_dir: str = "data/raw/wine",
    variant: str = "both",
) -> pd.DataFrame:
    """Load red and/or white wine datasets.
    
    Args:
        raw_dir: Directory containing winequality-red.csv and winequality-white.csv.
        variant: 'both' (combined with 'wine_type'), 'red', or 'white'.
        
    Returns:
        Combined or single raw DataFrame.
    """
    raw_path = resolve_path(raw_dir)
    red_file = raw_path / "winequality-red.csv"
    white_file = raw_path / "winequality-white.csv"
    
    # Check fallback in data/raw
    if not red_file.exists():
        fallback_red = resolve_path("data/raw") / "winequality-red.csv"
        if fallback_red.exists():
            red_file = fallback_red
    if not white_file.exists():
        fallback_white = resolve_path("data/raw") / "winequality-white.csv"
        if fallback_white.exists():
            white_file = fallback_white
            
    if variant in ("both", "combined"):
        if not red_file.exists() or not white_file.exists():
            raise FileNotFoundError(f"Wine files not found: {red_file}, {white_file}")
        df_red = pd.read_csv(red_file, sep=";", skipinitialspace=True)
        df_red["wine_type"] = "red"
        df_white = pd.read_csv(white_file, sep=";", skipinitialspace=True)
        df_white["wine_type"] = "white"
        df = pd.concat([df_red, df_white], ignore_index=True)
        logger.info("Loaded combined Wine Quality: %d rows (red=%d, white=%d)", len(df), len(df_red), len(df_white))
    elif variant == "red":
        if not red_file.exists():
            raise FileNotFoundError(f"Red wine file not found: {red_file}")
        df = pd.read_csv(red_file, sep=";", skipinitialspace=True)
        logger.info("Loaded Red Wine Quality: %d rows", len(df))
    elif variant == "white":
        if not white_file.exists():
            raise FileNotFoundError(f"White wine file not found: {white_file}")
        df = pd.read_csv(white_file, sep=";", skipinitialspace=True)
        logger.info("Loaded White Wine Quality: %d rows", len(df))
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose 'both', 'red', or 'white'.")
        
    return df


def clean_wine_dataset(
    df: pd.DataFrame,
    threshold: int = WINE_QUALITY_THRESHOLD,
) -> pd.DataFrame:
    """Clean Wine Quality data and define binary target.
    
    Binary target definition:
    quality >= threshold -> 1 (high quality)
    quality < threshold -> 0 (standard/lower quality)
    
    Args:
        df: Raw DataFrame with 'quality' column.
        threshold: Cutoff for high quality (default: 7).
        
    Returns:
        Cleaned DataFrame with 'quality_binary' column.
    """
    df = df.copy()
    
    if WINE_RAW_TARGET not in df.columns:
        raise ValueError(f"Target '{WINE_RAW_TARGET}' column missing from wine data.")
    
    # Create binary target
    df[WINE_TARGET] = (df[WINE_RAW_TARGET] >= threshold).astype(int)
    
    # Ensure numerical columns are clean
    for col in WINE_NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                logger.warning("Wine: %d NaN values coerced in '%s'", n_nan, col)
    
    logger.info(
        "Wine Quality cleaned: %d rows. Threshold=%d -> dist: %s",
        len(df), threshold, df[WINE_TARGET].value_counts().to_dict()
    )
    return df


def load_and_clean_wine(
    raw_dir: str = "data/raw/wine",
    variant: str = "both",
    threshold: int = WINE_QUALITY_THRESHOLD,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load, clean, split into dev/test, and return summary for Wine Quality.
    
    Args:
        raw_dir: Directory containing wine files.
        variant: 'both', 'red', or 'white'.
        threshold: Score cutoff for binary classification (default: 7).
        test_ratio: Fraction reserved for held-out test set.
        seed: Random seed.
        
    Returns:
        Tuple of (dev_df, test_df, summary).
    """
    raw_df = load_wine_data(raw_dir, variant=variant)
    cleaned_df = clean_wine_dataset(raw_df, threshold=threshold)
    
    y = cleaned_df[WINE_TARGET].values
    dev_df, test_df = train_test_split(
        cleaned_df,
        test_size=test_ratio,
        stratify=y,
        random_state=seed,
    )
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    cat_features = WINE_COMBINED_CATEGORICAL_FEATURES if variant in ("both", "combined") else []
    
    summary = get_preprocessing_summary(
        dev_df,
        test_df,
        target_col=WINE_TARGET,
        num_features=WINE_NUMERICAL_FEATURES,
        cat_features=cat_features,
    )
    summary["dataset"] = f"wine_quality_{variant}"
    summary["threshold"] = threshold
    summary["design_decision"] = (
        f"Quality scores range 3-9. Quality >= {threshold} defines high quality (~20% positive class). "
        "Scores below this represent standard/table wines."
    )
    
    logger.info("Wine Quality loaded (%s): dev=%d, test=%d", variant, len(dev_df), len(test_df))
    return dev_df, test_df, summary


# =====================================================================
# Unified Dataset Loader Factory
# =====================================================================
def load_dataset_by_name(
    dataset_name: str,
    raw_dir: Optional[str] = None,
    test_ratio: float = 0.20,
    seed: int = 42,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, DatasetSpec]:
    """Unified loader factory returning (dev_df, test_df, summary, spec).
    
    Args:
        dataset_name: 'adult', 'bank_marketing' (or 'bank'), or 'wine_quality' (or 'wine').
        raw_dir: Optional path to raw data directory.
        test_ratio: Held-out test ratio for datasets without official test split.
        seed: Random seed.
        
    Returns:
        Tuple of (dev_df, test_df, summary, DatasetSpec).
    """
    name = dataset_name.lower().replace("_marketing", "").replace("_quality", "")
    
    if name in ("adult", "uci_adult"):
        dev_df, test_df, summary = load_and_clean(raw_dir=raw_dir or "data/raw")
        spec = get_dataset_spec("adult")
        return dev_df, test_df, summary, spec
        
    elif name in ("bank", "bank_marketing"):
        dev_df, test_df, summary = load_and_clean_bank(
            raw_dir=raw_dir or "data/raw/bank",
            test_ratio=test_ratio,
            seed=seed,
        )
        spec = get_dataset_spec("bank")
        return dev_df, test_df, summary, spec
        
    elif name in ("wine", "wine_quality"):
        variant = kwargs.get("variant", "both")
        threshold = kwargs.get("threshold", WINE_QUALITY_THRESHOLD)
        dev_df, test_df, summary = load_and_clean_wine(
            raw_dir=raw_dir or "data/raw/wine",
            variant=variant,
            threshold=threshold,
            test_ratio=test_ratio,
            seed=seed,
        )
        spec = get_dataset_spec("wine")
        if variant != "both":
            spec.categorical_features = []
        return dev_df, test_df, summary, spec
        
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected 'adult', 'bank', or 'wine'.")
