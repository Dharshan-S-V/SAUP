"""Baseline ML models for binary classification."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from saup.config import resolve_path

logger = logging.getLogger(__name__)

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
    logger.info("XGBoost is available")
except ImportError:
    HAS_XGBOOST = False
    logger.info("XGBoost not available; will skip XGBoost model")


def get_model(
    name: str,
    config: Dict[str, Any],
    random_seed: int = 42,
) -> Any:
    """Create a model instance by name.
    
    Args:
        name: Model name ('logistic_regression', 'random_forest', 'mlp', 'xgboost').
        config: Model configuration dictionary from YAML.
        random_seed: Random seed.
        
    Returns:
        Unfitted sklearn-compatible estimator.
        
    Raises:
        ValueError: If model name is unknown.
    """
    model_config = config.get("models", {}).get(name, {})
    
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=model_config.get("max_iter", 2000),
            C=model_config.get("C", 1.0),
            solver=model_config.get("solver", "lbfgs"),
            random_state=random_seed,
            n_jobs=-1,
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_config.get("n_estimators", 300),
            max_depth=model_config.get("max_depth", 20),
            min_samples_leaf=model_config.get("min_samples_leaf", 5),
            random_state=random_seed,
            n_jobs=model_config.get("n_jobs", -1),
        )
    elif name == "mlp":
        hidden = model_config.get("hidden_layer_sizes", [128, 64])
        if isinstance(hidden, list):
            hidden = tuple(hidden)
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=model_config.get("activation", "relu"),
            solver=model_config.get("solver", "adam"),
            max_iter=model_config.get("max_iter", 500),
            early_stopping=model_config.get("early_stopping", True),
            validation_fraction=model_config.get("validation_fraction", 0.15),
            alpha=model_config.get("alpha", 0.001),
            random_state=random_seed,
        )
    elif name == "xgboost":
        if not HAS_XGBOOST:
            raise ValueError("XGBoost is not installed")
        return XGBClassifier(
            n_estimators=model_config.get("n_estimators", 300),
            max_depth=model_config.get("max_depth", 6),
            learning_rate=model_config.get("learning_rate", 0.1),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def get_available_models(config: Dict[str, Any]) -> List[str]:
    """Get list of available model names.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        List of model name strings.
    """
    models = ["logistic_regression", "random_forest", "mlp"]
    if HAS_XGBOOST:
        models.append("xgboost")
    return models


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "model",
) -> Tuple[Any, float]:
    """Train a model and return it with training time.
    
    Args:
        model: Unfitted estimator.
        X_train: Training features.
        y_train: Training labels.
        model_name: Name for logging.
        
    Returns:
        Tuple of (fitted_model, training_time_seconds).
    """
    logger.info("Training %s on %d samples...", model_name, len(X_train))
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("%s trained in %.2f seconds", model_name, elapsed)
    return model, elapsed


def predict(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and probability estimates.
    
    Args:
        model: Fitted estimator.
        X: Feature array.
        
    Returns:
        Tuple of (predictions, probabilities) where probabilities has shape (n, 2).
    """
    preds = model.predict(X)
    probas = model.predict_proba(X)
    
    # Validate probabilities
    assert probas.shape[1] == 2, f"Expected 2-class probabilities, got shape {probas.shape}"
    assert np.all(np.isfinite(probas)), "Non-finite probabilities detected"
    
    # Check probabilities sum to 1
    row_sums = probas.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        logger.warning("Probability sums deviate from 1.0, renormalizing")
        probas = probas / row_sums[:, np.newaxis]
    
    return preds, probas


def save_model(model: Any, name: str, models_dir: str = "models") -> Path:
    """Save a fitted model to disk.
    
    Args:
        model: Fitted estimator.
        name: Model identifier.
        models_dir: Directory for model files.
        
    Returns:
        Path to saved model file.
    """
    out_dir = resolve_path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = out_dir / f"{name}.joblib"
    joblib.dump(model, filepath)
    logger.info("Model saved: %s", filepath)
    return filepath


def load_model(name: str, models_dir: str = "models") -> Any:
    """Load a fitted model from disk.
    
    Args:
        name: Model identifier.
        models_dir: Directory with model files.
        
    Returns:
        Fitted estimator.
    """
    filepath = resolve_path(models_dir) / f"{name}.joblib"
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    model = joblib.load(filepath)
    logger.info("Model loaded: %s", filepath)
    return model
