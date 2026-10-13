"""Reproducible data splitting with stratification and non-overlap guarantees."""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from saup.config import resolve_path

logger = logging.getLogger(__name__)


def create_splits(
    df: pd.DataFrame,
    target_col: str = "income",
    train_ratio: float = 0.70,
    calibration_ratio: float = 0.10,
    conformal_ratio: float = 0.10,
    policy_ratio: float = 0.10,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Create stratified, non-overlapping data splits from the development data.
    
    Split order: train (70%) → calibration (10%) → conformal (10%) → policy (10%)
    
    Args:
        df: Full development DataFrame (adult.data).
        target_col: Name of target column.
        train_ratio: Fraction for training.
        calibration_ratio: Fraction for probability calibration.
        conformal_ratio: Fraction for conformal calibration.
        policy_ratio: Fraction for policy validation.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Dictionary with keys 'train', 'calibration', 'conformal', 'policy'.
        
    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    total = train_ratio + calibration_ratio + conformal_ratio + policy_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    y = df[target_col].values
    indices = np.arange(len(df))
    
    # Split: train vs rest
    rest_ratio = 1.0 - train_ratio
    train_idx, rest_idx = train_test_split(
        indices, test_size=rest_ratio, stratify=y, random_state=random_seed
    )
    
    # Split rest: calibration vs (conformal + policy)
    y_rest = y[rest_idx]
    cal_frac_of_rest = calibration_ratio / rest_ratio
    cal_idx, remaining_idx = train_test_split(
        rest_idx, test_size=(1.0 - cal_frac_of_rest),
        stratify=y_rest, random_state=random_seed
    )
    
    # Split remaining: conformal vs policy
    y_remaining = y[remaining_idx]
    conformal_frac_of_remaining = conformal_ratio / (conformal_ratio + policy_ratio)
    conf_idx, policy_idx = train_test_split(
        remaining_idx, test_size=(1.0 - conformal_frac_of_remaining),
        stratify=y_remaining, random_state=random_seed
    )
    
    # Verify non-overlap
    all_indices = [set(train_idx), set(cal_idx), set(conf_idx), set(policy_idx)]
    names = ["train", "calibration", "conformal", "policy"]
    for i in range(len(all_indices)):
        for j in range(i + 1, len(all_indices)):
            overlap = all_indices[i] & all_indices[j]
            assert len(overlap) == 0, (
                f"Overlap between {names[i]} and {names[j]}: {len(overlap)} indices"
            )
    
    # Verify completeness
    total_idx = set(train_idx) | set(cal_idx) | set(conf_idx) | set(policy_idx)
    assert total_idx == set(indices), "Split indices don't cover all data"
    
    splits = {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "calibration": df.iloc[cal_idx].reset_index(drop=True),
        "conformal": df.iloc[conf_idx].reset_index(drop=True),
        "policy": df.iloc[policy_idx].reset_index(drop=True),
    }
    
    # Log split info
    for name, split_df in splits.items():
        dist = split_df[target_col].value_counts().to_dict()
        logger.info("Split '%s': %d rows, label dist: %s", name, len(split_df), dist)
    
    return splits


def save_split_metadata(
    splits: Dict[str, pd.DataFrame],
    random_seed: int,
    output_dir: str = "data/processed",
) -> Path:
    """Save metadata about the data splits.
    
    Args:
        splits: Dictionary of split DataFrames.
        random_seed: Random seed used.
        output_dir: Directory to save metadata.
        
    Returns:
        Path to metadata file.
    """
    out_path = resolve_path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "random_seed": random_seed,
        "splits": {},
    }
    
    for name, df in splits.items():
        metadata["splits"][name] = {
            "n_rows": len(df),
            "label_distribution": df["income"].value_counts().to_dict(),
        }
    
    meta_file = out_path / "split_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info("Split metadata saved to %s", meta_file)
    return meta_file
