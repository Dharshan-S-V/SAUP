"""Configuration management for SAUP."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Project root: two levels up from src/saup/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to config file. Uses default.yaml if None.
        
    Returns:
        Configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logger.info("Loaded configuration from %s", path)
    return config


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def resolve_path(relative_path: str) -> Path:
    """Resolve a relative path against the project root.
    
    Args:
        relative_path: Path relative to project root.
        
    Returns:
        Absolute Path object.
    """
    return PROJECT_ROOT / relative_path


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create all required output directories.
    
    Args:
        config: Configuration dictionary.
    """
    dirs_to_create = [
        config["data"]["raw_dir"],
        config["data"]["processed_dir"],
        config["output"]["models_dir"],
        config["output"]["reports_dir"],
        config["output"]["figures_dir"],
        config["output"]["results_dir"],
    ]
    
    for d in dirs_to_create:
        path = resolve_path(d)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", path)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the project.
    
    Args:
        level: Logging level string.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
