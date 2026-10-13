"""Controlled distribution-shift generation for SAUP experiments (Adult, Bank, Wine)."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from saup.data_loader import TARGET

logger = logging.getLogger(__name__)

SEVERITY_MAP = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}


def _resolve_target_col(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """Resolve target column from explicit argument or DataFrame columns."""
    if target_col and target_col in df.columns:
        return target_col
    for candidate in [TARGET, "y", "quality_binary", "quality"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find target column in DataFrame with columns {list(df.columns)}")


def _resolve_covariate_cols(df: pd.DataFrame, covariate_cols: Optional[List[str]] = None) -> List[str]:
    """Resolve two numerical features for covariate shift."""
    if covariate_cols and all(c in df.columns for c in covariate_cols):
        return covariate_cols
    
    # Defaults by known dataset columns
    if "age" in df.columns and "hours_per_week" in df.columns:
        return ["age", "hours_per_week"]
    if "age" in df.columns and "balance" in df.columns:
        return ["age", "balance"]
    if "alcohol" in df.columns and "volatile acidity" in df.columns:
        return ["alcohol", "volatile acidity"]
    
    # Fallback to any two numeric columns
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [TARGET, "y", "quality_binary", "quality"]]
    if len(numeric_cols) >= 2:
        return numeric_cols[:2]
    elif len(numeric_cols) == 1:
        return numeric_cols
    raise ValueError("DataFrame does not contain enough numerical columns for covariate shift.")


def generate_covariate_shift(
    df: pd.DataFrame,
    severity: str = "moderate",
    seed: int = 42,
    target_size: Optional[int] = None,
    target_col: Optional[str] = None,
    covariate_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate covariate shift by importance-weighted sampling on features.
    
    Selection probability depends on standardized features only, NOT on the label directly.
    
    Args:
        df: Source DataFrame with all columns.
        severity: 'mild', 'moderate', or 'severe'.
        seed: Random seed.
        target_size: Desired sample size (default: half of source).
        target_col: Name of target column.
        covariate_cols: Two numerical features to tilt on.
        
    Returns:
        Tuple of (shifted_df, metadata_dict).
    """
    rng = np.random.RandomState(seed)
    strength = SEVERITY_MAP.get(severity, 0.6)
    target = _resolve_target_col(df, target_col)
    cols = _resolve_covariate_cols(df, covariate_cols)
    
    if target_size is None:
        target_size = len(df) // 2
    
    # Compute standardized z-scores
    z_scores = []
    for col in cols:
        vals = df[col].values.astype(float)
        std = vals.std()
        z = (vals - vals.mean()) / (std + 1e-8)
        z_scores.append(z)
    
    combined_z = np.mean(z_scores, axis=0)
    log_weights = strength * combined_z
    log_weights = np.clip(log_weights, -5, 5)
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    
    indices = rng.choice(len(df), size=target_size, replace=True, p=weights)
    shifted_df = df.iloc[indices].reset_index(drop=True)
    
    source_label_dist = df[target].value_counts(normalize=True).to_dict()
    target_label_dist = shifted_df[target].value_counts(normalize=True).to_dict()
    
    meta = {
        "scenario": "covariate_shift",
        "severity": severity,
        "strength": strength,
        "seed": seed,
        "source_size": len(df),
        "target_size": len(shifted_df),
        "source_label_dist": source_label_dist,
        "target_label_dist": target_label_dist,
        "covariate_cols": cols,
        "note": f"Selection depends on {cols} only, not labels directly",
    }
    
    logger.info(
        "Covariate shift (%s): %d → %d samples, label dist: %s → %s",
        severity, len(df), len(shifted_df), source_label_dist, target_label_dist
    )
    return shifted_df, meta


def generate_label_shift(
    df: pd.DataFrame,
    severity: str = "moderate",
    seed: int = 42,
    target_size: Optional[int] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate label shift by modifying the class prior distribution.
    
    Args:
        df: Source DataFrame.
        severity: 'mild', 'moderate', or 'severe'.
        seed: Random seed.
        target_size: Desired total sample size.
        target_col: Name of target column.
        
    Returns:
        Tuple of (shifted_df, metadata_dict).
    """
    rng = np.random.RandomState(seed)
    strength = SEVERITY_MAP.get(severity, 0.6)
    target = _resolve_target_col(df, target_col)
    
    if target_size is None:
        target_size = len(df) // 2
    
    source_p1 = float(df[target].mean())
    
    # Tilt target prior: increase positive class proportion
    target_p1 = min(source_p1 + strength * 0.3, 0.95)
    target_p0 = 1 - target_p1
    
    n_class1 = int(target_size * target_p1)
    n_class0 = target_size - n_class1
    
    df_0 = df[df[target] == 0]
    df_1 = df[df[target] == 1]
    
    if len(df_0) == 0 or len(df_1) == 0:
        raise ValueError(f"Cannot generate label shift with single class in {target}")
    
    sampled_0 = df_0.iloc[rng.choice(len(df_0), size=n_class0, replace=True)]
    sampled_1 = df_1.iloc[rng.choice(len(df_1), size=n_class1, replace=True)]
    
    shifted_df = pd.concat([sampled_0, sampled_1], ignore_index=True)
    shifted_df = shifted_df.iloc[rng.permutation(len(shifted_df))].reset_index(drop=True)
    
    meta = {
        "scenario": "label_shift",
        "severity": severity,
        "strength": strength,
        "seed": seed,
        "source_size": len(df),
        "target_size": len(shifted_df),
        "source_p1": source_p1,
        "target_p1": float(shifted_df[target].mean()),
        "note": "True labels used for population construction only; not given to detector/adapter",
    }
    
    logger.info(
        "Label shift (%s): P(Y=1) %.3f → %.3f, %d samples",
        severity, source_p1, shifted_df[target].mean(), len(shifted_df)
    )
    return shifted_df, meta


def generate_mixed_shift(
    df: pd.DataFrame,
    severity: str = "moderate",
    seed: int = 42,
    target_size: Optional[int] = None,
    target_col: Optional[str] = None,
    covariate_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate mixed shift combining covariate and label shift.
    
    Args:
        df: Source DataFrame.
        severity: Shift severity.
        seed: Random seed.
        target_size: Target sample size.
        target_col: Target column name.
        covariate_cols: Features for covariate component.
        
    Returns:
        Tuple of (shifted_df, metadata_dict).
    """
    target = _resolve_target_col(df, target_col)
    strength = SEVERITY_MAP.get(severity, 0.6)
    
    if target_size is None:
        target_size = len(df) // 2
    
    # 1. Label shift
    label_shifted, _ = generate_label_shift(
        df, severity=severity, seed=seed, target_size=target_size, target_col=target
    )
    
    # 2. Covariate shift
    shifted_df, _ = generate_covariate_shift(
        label_shifted, severity=severity, seed=seed + 1, target_size=target_size,
        target_col=target, covariate_cols=covariate_cols,
    )
    
    meta = {
        "scenario": "mixed_shift",
        "severity": severity,
        "strength": strength,
        "seed": seed,
        "source_size": len(df),
        "target_size": len(shifted_df),
        "source_label_dist": df[target].value_counts(normalize=True).to_dict(),
        "target_label_dist": shifted_df[target].value_counts(normalize=True).to_dict(),
        "note": "Stress test: neither covariate-shift nor label-shift assumptions hold perfectly",
    }
    
    logger.info("Mixed shift (%s): %d → %d samples", severity, len(df), len(shifted_df))
    return shifted_df, meta


def generate_severe_shift(
    df: pd.DataFrame,
    severity: str = "severe",
    seed: int = 42,
    target_size: Optional[int] = None,
    target_col: Optional[str] = None,
    perturb_cols: Optional[List[str]] = None,
    covariate_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate severe / OOD synthetic stress test with tail sampling and perturbation.
    
    Args:
        df: Source DataFrame.
        severity: Severity level.
        seed: Random seed.
        target_size: Target sample size.
        target_col: Name of target column.
        perturb_cols: Features to add controlled noise to.
        covariate_cols: Features to select tails from.
        
    Returns:
        Tuple of (shifted_df, metadata_dict).
    """
    rng = np.random.RandomState(seed)
    strength = SEVERITY_MAP.get(severity, 1.0)
    target = _resolve_target_col(df, target_col)
    tail_cols = _resolve_covariate_cols(df, covariate_cols)
    
    if target_size is None:
        target_size = len(df) // 3
    
    # Tail-focused sampling
    tail_scores = []
    for col in tail_cols:
        vals = df[col].values.astype(float)
        std = vals.std()
        tail_scores.append(np.abs((vals - vals.mean()) / (std + 1e-8)))
    
    combined_tail = np.sum(tail_scores, axis=0)
    log_weights = strength * 2.0 * combined_tail
    log_weights = np.clip(log_weights, -10, 10)
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    
    indices = rng.choice(len(df), size=target_size, replace=True, p=weights)
    shifted_df = df.iloc[indices].copy().reset_index(drop=True)
    
    # Numerical perturbation
    if perturb_cols is None:
        # Default perturbation columns based on available columns
        known_defaults = [
            "age", "hours_per_week", "capital_gain", "capital_loss",
            "balance", "campaign", "previous",
            "alcohol", "volatile acidity", "residual sugar", "density", "pH", "sulphates"
        ]
        perturb_cols = [c for c in known_defaults if c in shifted_df.columns]
    
    for col in perturb_cols:
        if col in shifted_df.columns:
            vals = shifted_df[col].values.astype(float)
            noise = rng.normal(0, strength * vals.std() * 0.3, size=len(vals))
            perturbed = vals + noise
            
            # Domain clipping
            if col == "age":
                perturbed = np.clip(perturbed, 17, 95)
            elif col == "hours_per_week":
                perturbed = np.clip(perturbed, 1, 99)
            elif col in ("capital_gain", "capital_loss", "campaign", "previous", "residual sugar", "free sulfur dioxide", "total sulfur dioxide"):
                perturbed = np.clip(perturbed, 0, None)
            elif col == "pH":
                perturbed = np.clip(perturbed, 2.5, 4.5)
            elif col == "alcohol":
                perturbed = np.clip(perturbed, 8.0, 16.0)
            
            shifted_df[col] = perturbed
    
    meta = {
        "scenario": "severe_shift",
        "severity": severity,
        "strength": strength,
        "seed": seed,
        "source_size": len(df),
        "target_size": len(shifted_df),
        "source_label_dist": df[target].value_counts(normalize=True).to_dict(),
        "target_label_dist": shifted_df[target].value_counts(normalize=True).to_dict(),
        "note": "SYNTHETIC stress test with tail sampling and numerical perturbation. "
                "Not evidence of performance on a genuine external population.",
    }
    
    logger.info("Severe shift (%s): %d → %d samples (synthetic)", severity, len(df), len(shifted_df))
    return shifted_df, meta


def generate_shift(
    df: pd.DataFrame,
    scenario: str,
    severity: str = "moderate",
    seed: int = 42,
    target_size: Optional[int] = None,
    target_col: Optional[str] = None,
    covariate_cols: Optional[List[str]] = None,
    perturb_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Dispatch to the appropriate shift generator.
    
    Args:
        df: Source DataFrame.
        scenario: One of 'source', 'covariate', 'label', 'mixed', 'severe'.
        severity: Shift severity level.
        seed: Random seed.
        target_size: Target sample size.
        target_col: Target column name.
        covariate_cols: Optional covariate tilt columns.
        perturb_cols: Optional perturbation columns for severe scenario.
        
    Returns:
        Tuple of (shifted_df, metadata_dict).
    """
    if scenario == "source":
        meta = {
            "scenario": "source",
            "severity": "none",
            "seed": seed,
            "source_size": len(df),
            "target_size": len(df),
            "note": "No shift applied; using source data as-is",
        }
        return df.copy(), meta
    elif scenario == "covariate":
        return generate_covariate_shift(
            df, severity=severity, seed=seed, target_size=target_size,
            target_col=target_col, covariate_cols=covariate_cols,
        )
    elif scenario == "label":
        return generate_label_shift(
            df, severity=severity, seed=seed, target_size=target_size,
            target_col=target_col,
        )
    elif scenario == "mixed":
        return generate_mixed_shift(
            df, severity=severity, seed=seed, target_size=target_size,
            target_col=target_col, covariate_cols=covariate_cols,
        )
    elif scenario == "severe":
        return generate_severe_shift(
            df, severity=severity, seed=seed, target_size=target_size,
            target_col=target_col, perturb_cols=perturb_cols, covariate_cols=covariate_cols,
        )
    else:
        raise ValueError(f"Unknown shift scenario: {scenario}")
